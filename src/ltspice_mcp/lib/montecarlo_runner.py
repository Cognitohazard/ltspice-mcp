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
    expand_tolerances,
    format_value,
    parse_value,
)
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.runner_base import BatchRunnerBase, wrap_runner_for_runno_callbacks
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


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

            # Resolve tolerances once: refs in the editor that match the
            # type/component rules. Components whose value can't be parsed
            # to a float (parametric expressions, behavioral sources)
            # silently drop out — perturbing a {param} value would corrupt
            # the netlist.
            all_refs = list(editor.get_components("*"))
            tol_map = expand_tolerances(
                all_refs,
                mc_config.type_tolerances,
                mc_config.component_overrides,
            )

            nominals: dict[str, float] = {}
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
                nominals[ref] = parsed

            if not nominals:
                raise BatchJobError(
                    "Monte Carlo: no perturbable components matched the tolerance rules. "
                    "Check the type prefixes (R/C/L) and ref names against the netlist."
                )

            sampler = MCSampler(seed=mc_config.seed)

            logger.info(
                "Starting Monte Carlo job %s: %d runs, %d perturbed components, "
                "type_tolerances=%s, component_overrides=%s, seed=%s",
                batch_job.job_id,
                batch_job.total_runs,
                len(nominals),
                list(mc_config.type_tolerances.keys()),
                list(mc_config.component_overrides.keys()),
                mc_config.seed,
            )

            for run_i in range(batch_job.total_runs):
                if cancel_event.is_set():
                    break
                runno = run_i + 1  # spicelib's runno is 1-based.
                # No reset_netlist() here — every perturbable ref is
                # overwritten each iteration, so prior values can't drift
                # in. This avoids re-reading the netlist file N times.
                run_params: dict[str, str] = {}
                for ref, spec in tol_map.items():
                    if ref not in nominals:
                        continue
                    perturbed = sampler.sample(nominals[ref], spec)
                    formatted = format_value(perturbed)
                    editor.set_component_value(ref, formatted)
                    run_params[ref] = formatted
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
