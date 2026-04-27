"""Monte Carlo orchestrator: per-run perturbation + asyncio bridge.

The math lives in ``lib/montecarlo``. This module orchestrates: for each
run, snapshot the nominal editor, perturb selected components via
``MCSampler``, submit one sim through the runner with the runno-aware
callback wrapper, and record per-run actual values back into
``run_results[i]["params"]`` so downstream tools (batch_results,
measurement_stats) can correlate measurements with the perturbed values.
"""

import asyncio
import logging
from pathlib import Path

from spicelib import SpiceEditor

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import BatchJob
from ltspice_mcp.lib.montecarlo import (
    MCSampler,
    MismatchRule,
    ModelTolerance,
    ParamTolerance,
    expand_tolerances,
    extract_mosfet_instances,
    extract_model_card,
    find_mismatch_rule,
    format_value,
    inject_card_before_end,
    parse_model_params,
    parse_param_nominal,
    parse_value,
    perturb_model_in_text,
    perturb_param_in_text,
    render_variant_model_card,
    rewrite_instance_model,
    sample_instance_mismatch,
    sample_model_perturbation,
    variant_model_name,
)
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.runner_base import BatchRunnerBase, wrap_runner_for_runno_callbacks
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def _resolve_base_params(
    model_name: str,
    model_nominals: dict[str, dict[str, float]],
    run_perturbations: dict[str, dict[str, float]],
    stable_cache: dict[str, dict[str, float]],
    text: str,
) -> dict[str, float] | None:
    """Resolve a model's *currently effective* params for this run.

    For models also covered by Phase 1 process variation, layer the run's
    perturbations on top of the nominals. For models without a Phase 1
    rule, fall back to the precomputed ``stable_cache``; if that doesn't
    have an entry either, parse the card lazily from ``text``.
    """
    if model_name in model_nominals:
        params = dict(model_nominals[model_name])
        params.update(run_perturbations.get(model_name, {}))
        return params
    cached = stable_cache.get(model_name)
    if cached is not None:
        return cached
    card = extract_model_card(text, model_name)
    if card is None:
        return None
    parsed = parse_model_params(card)
    stable_cache[model_name] = parsed
    return parsed


def _build_mismatch_overrides(
    deltas: dict[str, float],
    rule: MismatchRule,
    base_params: dict[str, float],
) -> dict[str, float]:
    """Translate (ΔVTH, ΔK/K) into perturbed model-card param overrides."""
    overrides: dict[str, float] = {}
    if deltas["dvth"] != 0.0:
        nominal = base_params.get(rule.vth_param.upper())
        if nominal is not None:
            overrides[rule.vth_param] = nominal + deltas["dvth"]
    if deltas["dk_over_k"] != 0.0:
        nominal = base_params.get(rule.k_param.upper())
        if nominal is not None:
            overrides[rule.k_param] = nominal * (1.0 + deltas["dk_over_k"])
    return overrides


class MonteCarloRunner(BatchRunnerBase):
    """Monte Carlo analysis with per-run reproducibility and actual-value tracking.

    Replaces the previous spicelib.Montecarlo subclass. Each run perturbs
    the netlist via our own ``MCSampler``, submits via the runno-aware
    callback wrapper, and records the actual perturbed values keyed by
    component ref into ``run_results[runno-1]["params"]``.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        logger.debug(
            "MonteCarloRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    async def start_montecarlo(
        self, batch_job: BatchJob, state: SessionState
    ) -> None:
        """Submit the Monte Carlo analysis to a worker thread; return immediately."""
        cancel_event = self._register_cancel(batch_job.job_id)
        # Keyed by 1-based runno; populated at submission time and popped
        # by the callback so memory doesn't grow with run count.
        per_run_params: dict[int, dict[str, str]] = {}

        def run_completion_callback(raw_file, log_file, runno: int) -> None:
            if cancel_event.is_set():
                return
            self._bridge(
                self._handle_run_completion,
                batch_job.job_id,
                Path(raw_file),
                Path(log_file),
                state,
                runno,
                per_run_params.pop(runno, {}),
                context=f"MC run (job {batch_job.job_id})",
            )

        def execute_montecarlo() -> None:
            if batch_job.mc_config is None:
                raise BatchJobError(
                    f"Monte Carlo job {batch_job.job_id} has no Monte Carlo configuration"
                )

            mc_config = batch_job.mc_config
            runner = wrap_runner_for_runno_callbacks(self._build_sim_runner())
            self._register_runner(batch_job.job_id, runner)

            editor = SpiceEditor(str(batch_job.netlist))

            # ---- Phase 0: R/C/L tolerance resolution + nominal extraction ----
            all_refs = list(editor.get_components("*"))
            tol_map = expand_tolerances(
                all_refs,
                mc_config.type_tolerances,
                mc_config.component_overrides,
            )
            rcl_nominals: dict[str, float] = {}
            for ref in tol_map:
                try:
                    raw_val = editor.get_component_value(ref)
                except Exception:
                    continue
                parsed = parse_value(raw_val)
                if parsed is None:
                    logger.debug(
                        "MC job %s: skipping %s — value %r not parseable",
                        batch_job.job_id, ref, raw_val,
                    )
                    continue
                rcl_nominals[ref] = parsed

            # ---- Phase 1 setup: per-.MODEL nominals ----
            baseline_text = "".join(editor.netlist)
            model_tolerances: list[ModelTolerance] = list(
                mc_config.model_tolerances or []
            )
            model_nominals: dict[str, dict[str, float]] = {}
            for mt in model_tolerances:
                card = extract_model_card(baseline_text, mt.model_name)
                if card is None:
                    logger.warning(
                        "MC job %s: .MODEL %s not found in netlist; ignoring rule",
                        batch_job.job_id, mt.model_name,
                    )
                    continue
                model_nominals[mt.model_name] = parse_model_params(card)

            # ---- Phase 2 setup: MOSFET instance geometry + per-instance caches ----
            mismatch_rules = list(mc_config.mismatch_rules or [])
            mosfet_instances = (
                extract_mosfet_instances(baseline_text) if mismatch_rules else []
            )
            # Precompute per-instance state that's stable across runs:
            # - rule lookup (linear scan over rules → O(1) dict lookup per run)
            # - whether the instance's model is also being process-perturbed
            #   (those need fresh ``parse_model_params`` per run; others can
            #   reuse the baseline parse).
            instance_to_rule = {
                inst.ref: rule
                for inst in mosfet_instances
                if (rule := find_mismatch_rule(inst.ref, mismatch_rules)) is not None
            }
            perturbed_models = set(model_nominals.keys())
            stable_base_params: dict[str, dict[str, float]] = {}
            for inst in mosfet_instances:
                if inst.ref not in instance_to_rule:
                    continue
                if inst.model_name in perturbed_models:
                    continue  # base will shift each run; recompute then
                base_card = extract_model_card(baseline_text, inst.model_name)
                if base_card is not None:
                    stable_base_params[inst.model_name] = parse_model_params(base_card)

            # ---- Phase 3 setup: .PARAM nominals ----
            param_tolerances: list[ParamTolerance] = list(
                mc_config.param_tolerances or []
            )
            param_nominals: dict[str, float] = {}
            for pt in param_tolerances:
                nominal = parse_param_nominal(baseline_text, pt.name)
                if nominal is None:
                    logger.warning(
                        "MC job %s: .PARAM %s not found or non-numeric; ignoring rule",
                        batch_job.job_id, pt.name,
                    )
                    continue
                param_nominals[pt.name] = nominal

            # Empty-perturbation guard — give the user something actionable.
            if not (
                rcl_nominals
                or model_nominals
                or (mosfet_instances and mismatch_rules)
                or param_nominals
            ):
                raise BatchJobError(
                    "Monte Carlo: no perturbable parameters matched the rules. "
                    "Check that R/C/L prefixes, .MODEL names, M-instance W/L params, "
                    "and .PARAM names match the netlist."
                )

            sampler = MCSampler(seed=mc_config.seed)

            logger.info(
                "Starting Monte Carlo job %s: %d runs | R/C/L=%d, .MODEL=%d, "
                "mismatch instances=%d, .PARAM=%d, seed=%s",
                batch_job.job_id,
                batch_job.total_runs,
                len(rcl_nominals),
                len(model_nominals),
                len(mosfet_instances) if mismatch_rules else 0,
                len(param_nominals),
                mc_config.seed,
            )

            # Snapshot the baseline editor.netlist once, so per-iteration
            # reset is an in-memory list copy rather than a disk re-read.
            # ``editor.reset_netlist()`` works but reopens the file each
            # iteration; for a 1000-run job that's 1000 file reads we can
            # avoid. Snapshot is shallow-safe — the list contains strings
            # (immutable), and we replace the list reference per iteration.
            baseline_lines: list[str] = list(editor.netlist)
            # Sub-streams are also keyed by run index so two runs with the
            # same global seed produce independent samples. Run-index
            # isolation is what makes ``num_runs=N`` reproducible.

            for run_i in range(batch_job.total_runs):
                if cancel_event.is_set():
                    break
                runno = run_i + 1  # spicelib's runno is 1-based.
                run_params: dict[str, str] = {}

                # In-memory reset: replace the editor's line list with a
                # fresh copy of the baseline. Cheaper than reset_netlist()
                # which re-reads the file from disk per iteration.
                editor.netlist = list(baseline_lines)

                # Per-run sampler — sub-streams within this iteration use
                # short keys ("rcl:R1", "model:NMOS1.VTO", "mismatch:M1.dvth")
                # without colliding across runs, since the run namespace
                # is encoded once at derive() time.
                run_sampler = sampler.derive(f"run{runno}")

                # ---- R/C/L (per-ref stream within the run sampler) ----
                for ref, spec in tol_map.items():
                    if ref not in rcl_nominals:
                        continue
                    perturbed = run_sampler.sample(
                        rcl_nominals[ref],
                        spec,
                        stream=f"rcl:{ref}",
                    )
                    formatted = format_value(perturbed)
                    editor.set_component_value(ref, formatted)
                    run_params[ref] = formatted

                # The remaining phases work on text. Snapshot the current
                # editor state, apply text rewrites, then push back.
                text = "".join(editor.netlist)

                # ---- Phase 1: process variation (.MODEL perturbation) ----
                # ``run_perturbations[model]`` accumulates this run's
                # process-level deltas so Phase 2 mismatch can layer on top
                # of the perturbed (not nominal) base params.
                run_perturbations: dict[str, dict[str, float]] = {}
                for mt in model_tolerances:
                    nominals = model_nominals.get(mt.model_name)
                    if not nominals:
                        continue
                    perturbations = sample_model_perturbation(
                        run_sampler, mt.model_name, nominals, mt.parameters
                    )
                    if not perturbations:
                        continue
                    text = perturb_model_in_text(text, mt.model_name, perturbations)
                    run_perturbations[mt.model_name] = perturbations
                    for p, v in perturbations.items():
                        run_params[f"{mt.model_name}.{p}"] = format_value(v)

                # ---- Phase 2: mismatch (per-instance variant models) ----
                for instance in mosfet_instances:
                    rule = instance_to_rule.get(instance.ref)
                    if rule is None:
                        continue
                    deltas = sample_instance_mismatch(run_sampler, instance, rule)
                    if deltas["dvth"] == 0.0 and deltas["dk_over_k"] == 0.0:
                        continue
                    base_params = _resolve_base_params(
                        instance.model_name, model_nominals, run_perturbations,
                        stable_base_params, text,
                    )
                    if base_params is None:
                        continue
                    overrides = _build_mismatch_overrides(deltas, rule, base_params)
                    if not overrides:
                        continue
                    base_card = extract_model_card(text, instance.model_name)
                    if base_card is None:
                        continue
                    variant = variant_model_name(instance.model_name, instance.ref)
                    variant_card = render_variant_model_card(
                        base_card, variant, overrides
                    )
                    text = inject_card_before_end(text, variant_card)
                    text = rewrite_instance_model(text, instance.ref, variant)
                    run_params[f"{instance.ref}.dvth"] = format_value(deltas["dvth"])
                    run_params[f"{instance.ref}.dk_over_k"] = format_value(
                        deltas["dk_over_k"]
                    )

                # ---- Phase 3: .PARAM perturbation ----
                for pt in param_tolerances:
                    nominal = param_nominals.get(pt.name)
                    if nominal is None:
                        continue
                    delta = run_sampler.sample_offset(
                        nominal, pt.spec, stream=f"param:{pt.name}"
                    )
                    new_value = nominal + delta
                    text = perturb_param_in_text(text, pt.name, new_value)
                    run_params[f"PARAM.{pt.name}"] = format_value(new_value)

                # Push the rewritten text back into the editor's line list.
                # SpiceEditor expects each entry to be a single line ending
                # in "\n" (its save_netlist iterates the list verbatim).
                new_lines = text.splitlines(keepends=True)
                if new_lines and not new_lines[-1].endswith("\n"):
                    new_lines[-1] = new_lines[-1] + "\n"
                editor.netlist = new_lines

                per_run_params[runno] = run_params
                # Wrapped runner injects runno; spicelib's CallbackType is
                # the unwrapped (raw_file, log_file) shape.
                runner.run(editor, callback=run_completion_callback)  # type: ignore[arg-type]

            runner.wait_completion()

            if not cancel_event.is_set():
                self._bridge(
                    self._handle_mc_completion,
                    batch_job.job_id,
                    state,
                    context=f"MC completion (job {batch_job.job_id})",
                )

        try:
            emit_job_event("started", batch_job, total_runs=batch_job.total_runs)
            await asyncio.to_thread(execute_montecarlo)
        except Exception as e:
            self._mark_batch_failed(batch_job, state, e, kind="Monte Carlo")
        finally:
            self._cleanup(batch_job.job_id)

    def _handle_run_completion(
        self,
        job_id: str,
        raw_file: Path,
        log_file: Path,
        state: SessionState,
        runno: int | None = None,
        params: dict[str, str] | None = None,
    ) -> None:
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("Run completion for unknown MC batch job %s", job_id)
            return
        self._record_run_completion(
            batch_job, raw_file, log_file, state, kind="MC", runno=runno
        )
        # Stash actual perturbed values so batch_results / measurement_stats
        # can correlate measurements with each run's component values.
        if params and runno is not None:
            run_key = runno - 1
            if run_key in batch_job.run_results:
                batch_job.run_results[run_key]["params"] = dict(params)

    def _handle_mc_completion(self, job_id: str, state: SessionState) -> None:
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("MC completion for unknown batch job %s", job_id)
            return
        if batch_job.status == "cancelled":
            logger.debug(
                "MC job %s was cancelled — preserving %d partial results",
                job_id,
                len(batch_job.run_results),
            )
            return

        transition(
            batch_job,
            "completed",
            state=state,
            completed_runs=batch_job.completed_runs,
            total_runs=batch_job.total_runs,
        )

        logger.info(
            "Monte Carlo job %s completed: %d/%d runs finished",
            job_id,
            batch_job.completed_runs,
            batch_job.total_runs,
        )
