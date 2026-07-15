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
from dataclasses import dataclass
from pathlib import Path

from spicelib import SpiceEditor

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.format import format_spice_value
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import BatchJob, MonteCarloConfig
from ltspice_mcp.lib.montecarlo import (
    InstanceGeometry,
    MCSampler,
    MismatchRule,
    ModelTolerance,
    ParamTolerance,
    ToleranceSpec,
    expand_tolerances,
    extract_model_card,
    extract_mosfet_instances,
    find_mismatch_rule,
    parse_model_params,
    parse_param_nominal,
    parse_value,
    render_variant_model_card,
    sample_instance_mismatch,
    sample_model_perturbation,
    variant_model_name,
)
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.runner_base import (
    DEFAULT_MAX_PARALLEL,
    BatchRunnerBase,
    batch_run_filename,
    discard_generated_netlist,
)
from ltspice_mcp.lib.spice_lex import SpiceCard, emit, lex
from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end as _ops_inject_card
from ltspice_mcp.lib.spice_lex_views import (
    InstanceLine,
    ModelCard,
    ParamCard,
)
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def _resolve_base_params_from_cards(
    model_name: str,
    model_nominals: dict[str, dict[str, float]],
    run_perturbations: dict[str, dict[str, float]],
    stable_cache: dict[str, dict[str, float]],
    model_by_name: dict[str, SpiceCard],
) -> dict[str, float] | None:
    """Resolve a model's effective params for this run via the lookup dict.

    Order: Phase-1 nominals + run perturbations → stable cache → parse
    the model card body lazily. ``model_by_name`` is the per-run dict
    keyed by lowercased model name.
    """
    if model_name in model_nominals:
        params = dict(model_nominals[model_name])
        params.update(run_perturbations.get(model_name, {}))
        return params
    cached = stable_cache.get(model_name)
    if cached is not None:
        return cached
    card = model_by_name.get(model_name.lower())
    if card is None:
        return None
    parsed = parse_model_params(card.body)
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


@dataclass
class _MCPlan:
    """Resolved Monte Carlo perturbation plan, computed once from the baseline.

    Groups the per-rule nominal values and precomputed per-run lookups so the
    per-run perturbation pass takes explicit inputs instead of closing over the
    setup scope. ``stable_base_params`` is a per-job cache that the per-run pass
    may fill in lazily, so it is shared (not copied) across runs.
    """

    tol_map: dict[str, ToleranceSpec]
    rcl_nominals: dict[str, float]
    model_tolerances: list[ModelTolerance]
    model_nominals: dict[str, dict[str, float]]
    # Only populated when the config carries mismatch rules, so its truthiness
    # doubles as the "mismatch enabled" signal.
    mosfet_instances: list[InstanceGeometry]
    instance_to_rule: dict[str, MismatchRule]
    stable_base_params: dict[str, dict[str, float]]
    param_tolerances: list[ParamTolerance]
    param_nominals: dict[str, float]


def _resolve_mc_plan(
    job_id: str,
    baseline_text: str,
    baseline_cards: list[SpiceCard],
    mc_config: MonteCarloConfig,
) -> _MCPlan:
    """Resolve the perturbation plan once from the baseline netlist.

    Extracts nominal values and precomputes per-run lookups for each
    perturbation class: R/C/L component tolerances, per-.MODEL process
    variation, per-instance MOSFET mismatch, and .PARAM tolerances. Raises
    ``BatchJobError`` when no rule matches anything perturbable.
    """
    # R/C/L tolerance resolution + nominal extraction. Walk the lexed cards
    # instead of editor.get_components — works uniformly across flat and
    # hierarchical netlists.
    all_refs = [c.name for c in baseline_cards if c.kind == "instance" and c.name]
    tol_map = expand_tolerances(
        all_refs,
        mc_config.type_tolerances,
        mc_config.component_overrides,
    )
    baseline_inst_by_ref: dict[str, SpiceCard] = {
        c.name.lower(): c for c in baseline_cards if c.kind == "instance" and c.name
    }
    rcl_nominals: dict[str, float] = {}
    for ref in tol_map:
        card = baseline_inst_by_ref.get(ref.lower())
        if card is None:
            continue
        try:
            inst_view = InstanceLine.from_card(card)
        except Exception:
            continue
        raw_val = inst_view.value
        if raw_val is None:
            # M/Q/J/X have model name in the slot, no numeric value
            continue
        parsed = parse_value(raw_val)
        if parsed is None:
            # Parameter-driven values (``{RS}``) cannot be perturbed
            # at the component level — the user has to perturb the
            # .PARAM via ``param_tolerances`` instead. Surface this
            # rather than silently dropping.
            logger.warning(
                "MC job %s: %s value %r is not a numeric literal — "
                "skipping. To perturb parameter-driven values, add "
                "the underlying .PARAM name to ``param_tolerances``.",
                job_id,
                ref,
                raw_val,
            )
            continue
        rcl_nominals[ref] = parsed

    # Per-.MODEL nominals.
    model_tolerances: list[ModelTolerance] = list(mc_config.model_tolerances or [])
    model_nominals: dict[str, dict[str, float]] = {}
    for mt in model_tolerances:
        card = extract_model_card(baseline_text, mt.model_name)
        if card is None:
            logger.warning(
                "MC job %s: .MODEL %s not found in netlist; ignoring rule",
                job_id,
                mt.model_name,
            )
            continue
        model_nominals[mt.model_name] = parse_model_params(card)

    # MOSFET instance geometry + per-instance caches.
    mismatch_rules = list(mc_config.mismatch_rules or [])
    mosfet_instances = extract_mosfet_instances(baseline_text) if mismatch_rules else []
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

    # .PARAM nominals.
    param_tolerances: list[ParamTolerance] = list(mc_config.param_tolerances or [])
    param_nominals: dict[str, float] = {}
    for pt in param_tolerances:
        nominal = parse_param_nominal(baseline_text, pt.name)
        if nominal is None:
            logger.warning(
                "MC job %s: .PARAM %s not found or non-numeric; ignoring rule",
                job_id,
                pt.name,
            )
            continue
        param_nominals[pt.name] = nominal

    # Empty-perturbation guard — give the user something actionable.
    if not (rcl_nominals or model_nominals or mosfet_instances or param_nominals):
        raise BatchJobError(
            "Monte Carlo: no perturbable parameters matched the rules. "
            "Check that R/C/L prefixes, .MODEL names, M-instance W/L params, "
            "and .PARAM names match the netlist."
        )

    return _MCPlan(
        tol_map=tol_map,
        rcl_nominals=rcl_nominals,
        model_tolerances=model_tolerances,
        model_nominals=model_nominals,
        mosfet_instances=mosfet_instances,
        instance_to_rule=instance_to_rule,
        stable_base_params=stable_base_params,
        param_tolerances=param_tolerances,
        param_nominals=param_nominals,
    )


def _perturb_run(
    baseline_text: str,
    plan: _MCPlan,
    run_sampler: MCSampler,
) -> tuple[list[str], dict[str, float]]:
    """Apply one run's perturbations to a fresh copy of the baseline.

    Re-lexes ``baseline_text`` (cheaper than deep-copying the card tree),
    perturbs each rule class via ``run_sampler``, and returns the rewritten
    netlist lines plus the actual perturbed values (floats) keyed by
    ref/param. ``plan.stable_base_params`` may be filled in as a side effect.
    """
    run_params: dict[str, float] = {}

    # Re-lex baseline text per iteration. ~0.6 ms on a 200-card netlist —
    # measured ~2.4× faster than ``copy.deepcopy(baseline_cards)`` because
    # the dataclass tree (raw_lines + tokens) is expensive to clone.
    cards = lex(baseline_text).cards

    # Single pass over cards to build the three lookup tables.
    model_by_name: dict[str, SpiceCard] = {}
    instance_by_ref: dict[str, SpiceCard] = {}
    param_by_name: dict[str, SpiceCard] = {}
    for c in cards:
        if not c.name:
            continue
        key = c.name.lower()
        if c.kind == "model":
            model_by_name[key] = c
        elif c.kind == "instance":
            instance_by_ref[key] = c
        elif c.kind == "param":
            param_by_name[key] = c

    # R/C/L (per-ref stream within the run sampler).
    for ref, spec in plan.tol_map.items():
        if ref not in plan.rcl_nominals:
            continue
        inst_card = instance_by_ref.get(ref.lower())
        if inst_card is None:
            continue
        perturbed = run_sampler.sample(
            plan.rcl_nominals[ref],
            spec,
            stream=f"rcl:{ref}",
        )
        formatted = format_spice_value(perturbed)
        InstanceLine.from_card(inst_card).set_value(formatted)
        run_params[ref] = perturbed

    # Process variation (.MODEL perturbation). ``run_perturbations[model]``
    # accumulates this run's process-level deltas so the mismatch pass can
    # layer on top of the perturbed (not nominal) base params.
    run_perturbations: dict[str, dict[str, float]] = {}
    for mt in plan.model_tolerances:
        nominals = plan.model_nominals.get(mt.model_name)
        if not nominals:
            continue
        perturbations = sample_model_perturbation(
            run_sampler, mt.model_name, nominals, mt.parameters
        )
        if not perturbations:
            continue
        model_card = model_by_name.get(mt.model_name.lower())
        if model_card is None:
            continue
        model_view = ModelCard.from_card(model_card)
        for p, v in perturbations.items():
            model_view.set_param(p, v)
        run_perturbations[mt.model_name] = perturbations
        for p, v in perturbations.items():
            run_params[f"{mt.model_name}.{p}"] = v

    # Mismatch (per-instance variant models).
    for instance in plan.mosfet_instances:
        rule = plan.instance_to_rule.get(instance.ref)
        if rule is None:
            continue
        deltas = sample_instance_mismatch(run_sampler, instance, rule)
        if deltas["dvth"] == 0.0 and deltas["dk_over_k"] == 0.0:
            continue
        base_params = _resolve_base_params_from_cards(
            instance.model_name,
            plan.model_nominals,
            run_perturbations,
            plan.stable_base_params,
            model_by_name,
        )
        if base_params is None:
            continue
        overrides = _build_mismatch_overrides(deltas, rule, base_params)
        if not overrides:
            continue
        base_card = model_by_name.get(instance.model_name.lower())
        if base_card is None:
            continue
        base_card_text = "".join(base_card.raw_lines)
        variant = variant_model_name(instance.model_name, instance.ref)
        variant_card_text = render_variant_model_card(base_card_text, variant, overrides)
        new_card = _ops_inject_card(cards, variant_card_text)
        if new_card.name:
            model_by_name[new_card.name.lower()] = new_card
        inst_card = instance_by_ref.get(instance.ref.lower())
        if inst_card is not None:
            InstanceLine.from_card(inst_card).set_model(variant)
        run_params[f"{instance.ref}.dvth"] = deltas["dvth"]
        run_params[f"{instance.ref}.dk_over_k"] = deltas["dk_over_k"]

    # .PARAM perturbation.
    for pt in plan.param_tolerances:
        nominal = plan.param_nominals.get(pt.name)
        if nominal is None:
            continue
        delta = run_sampler.sample_offset(nominal, pt.spec, stream=f"param:{pt.name}")
        new_value = nominal + delta
        param_card = param_by_name.get(pt.name.lower())
        if param_card is not None:
            ParamCard.from_card(param_card).set_value(new_value)
        run_params[f"PARAM.{pt.name}"] = new_value

    # Emit once and return the rewritten lines. SpiceEditor expects each
    # entry to be one line ending in "\n".
    new_text = emit(cards)
    new_lines = new_text.splitlines(keepends=True)
    if new_lines and not new_lines[-1].endswith("\n"):
        new_lines[-1] = new_lines[-1] + "\n"
    return new_lines, run_params


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
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        logger.debug(
            "MonteCarloRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    async def start_montecarlo(self, batch_job: BatchJob, state: SessionState) -> None:
        """Submit the Monte Carlo analysis to a worker thread; return immediately."""
        cancel_event = self._register_cancel(batch_job.job_id)
        # Keyed by 1-based runno; populated at submission time and popped
        # by the callback so memory doesn't grow with run count. Values are
        # floats (the actual perturbed magnitudes), matching the sweep runner's
        # numeric per-run params so both job types expose the same type.
        per_run_params: dict[int, dict[str, float]] = {}

        def run_completion_callback(raw_file, log_file, runno: int) -> None:
            if cancel_event.is_set():
                return
            # raw_file is None when the sub-run aborted (callback_on_error);
            # pass it through as None so it is recorded as a failed run.
            self._bridge(
                self._handle_run_completion,
                batch_job.job_id,
                Path(raw_file) if raw_file else None,
                Path(log_file) if log_file else None,
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
            runner = self._gated_runner_for(batch_job.job_id, cancel_event)

            # Read baseline text directly via the encoding-aware pipeline.
            # The spicelib editor is kept around only as the runner-submission
            # vehicle; we never READ ``editor.netlist`` because spicelib's
            # ``SpiceEditor.netlist`` contains ``SpiceCircuit`` objects (not
            # strings) for ``.subckt`` blocks, which breaks any join over the
            # list on hierarchical netlists.
            # run_netlist (the '.options logopinfo' copy) is the source when set;
            # per-run files are still named after the original deck (run_filename).
            src_netlist = batch_job.run_netlist or batch_job.netlist
            baseline_text = read_spice_text(src_netlist)
            baseline_cards = lex(baseline_text).cards
            editor = SpiceEditor(str(src_netlist))

            plan = _resolve_mc_plan(batch_job.job_id, baseline_text, baseline_cards, mc_config)

            sampler = MCSampler(seed=mc_config.seed)

            logger.info(
                "Starting Monte Carlo job %s: %d runs | R/C/L=%d, .MODEL=%d, "
                "mismatch instances=%d, .PARAM=%d, seed=%s",
                batch_job.job_id,
                batch_job.total_runs,
                len(plan.rcl_nominals),
                len(plan.model_nominals),
                len(plan.mosfet_instances),
                len(plan.param_nominals),
                mc_config.seed,
            )

            # Sub-streams are also keyed by run index so two runs with the
            # same global seed produce independent samples. Run-index
            # isolation is what makes ``num_runs=N`` reproducible.

            for run_i in range(batch_job.total_runs):
                if cancel_event.is_set():
                    break
                runno = run_i + 1  # spicelib's runno is 1-based.
                run_sampler = sampler.derive(f"run{runno}")
                new_lines, run_params = _perturb_run(baseline_text, plan, run_sampler)
                editor.netlist = new_lines

                per_run_params[runno] = run_params
                # Wrapped runner injects runno; spicelib's CallbackType is the
                # unwrapped (raw_file, log_file) shape. ``run_filename`` carries
                # the job_id token (see batch_run_filename) so this sub-run's
                # Windows process is targetable by cancel()'s WSL taskkill;
                # ``exe_log`` captures ngspice's stdout diagnostics.
                runner.run(
                    editor,
                    callback=run_completion_callback,  # type: ignore[arg-type]
                    run_filename=batch_run_filename(batch_job.job_id, runno, batch_job.netlist),
                    exe_log=True,
                )

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
            await asyncio.to_thread(discard_generated_netlist, batch_job.run_netlist)

    def _handle_run_completion(
        self,
        job_id: str,
        raw_file: Path | None,
        log_file: Path | None,
        state: SessionState,
        runno: int | None = None,
        params: dict[str, float] | None = None,
    ) -> None:
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("Run completion for unknown MC batch job %s", job_id)
            return
        self._record_run_completion(batch_job, raw_file, log_file, state, kind="MC", runno=runno)
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

        # Account for any run that never reported (silent drop) before completing,
        # so completed_runs == total_runs and "completed" can't mask a shortfall.
        self._finalize_batch(batch_job, "MC")

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
