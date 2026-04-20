"""Monte Carlo wrapper: spicelib Montecarlo + asyncio."""

import asyncio
import logging
from pathlib import Path

from spicelib.sim.tookit.montecarlo import Montecarlo

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import BatchJob
from ltspice_mcp.lib.observability import emit_job_event
from ltspice_mcp.lib.runner_base import BatchRunnerBase
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


class MonteCarloRunner(BatchRunnerBase):
    """Monte Carlo analysis via spicelib Montecarlo, bridged to asyncio.

    Per-run callbacks fire in worker threads; they're bridged to the
    event loop via ``BatchRunnerBase._bridge``. Unlike sweeps, MC runs
    have no explicit per-run parameter values — deviations are
    statistical, not per-run trackable.
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

        def run_completion_callback(raw_file, log_file) -> None:
            if cancel_event.is_set():
                return
            self._bridge(
                self._handle_run_completion,
                batch_job.job_id,
                Path(raw_file),
                Path(log_file),
                state,
                context=f"MC run (job {batch_job.job_id})",
            )

        def execute_montecarlo() -> None:
            if batch_job.mc_config is None:
                raise BatchJobError(
                    f"Monte Carlo job {batch_job.job_id} has no Monte Carlo configuration"
                )

            runner = self._build_sim_runner()
            self._register_runner(batch_job.job_id, runner)
            mc_config = batch_job.mc_config
            mc = Montecarlo(str(batch_job.netlist), runner)

            for ref, (tol, dist) in mc_config.type_tolerances.items():
                mc.set_tolerance(ref, tol, distribution=dist)
                logger.debug(
                    "MC job %s: set type tolerance %s=%s (%s)",
                    batch_job.job_id,
                    ref,
                    tol,
                    dist,
                )

            for ref, (tol, dist) in mc_config.component_overrides.items():
                mc.set_tolerance(ref, tol, distribution=dist)
                logger.debug(
                    "MC job %s: set component override %s=%s (%s)",
                    batch_job.job_id,
                    ref,
                    tol,
                    dist,
                )

            logger.info(
                "Starting Monte Carlo job %s: %d runs, type_tolerances=%s, "
                "component_overrides=%s",
                batch_job.job_id,
                batch_job.total_runs,
                list(mc_config.type_tolerances.keys()),
                list(mc_config.component_overrides.keys()),
            )

            mc.run_analysis(
                callback=run_completion_callback,
                num_runs=batch_job.total_runs,
            )

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
    ) -> None:
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("Run completion for unknown MC batch job %s", job_id)
            return
        self._record_run_completion(batch_job, raw_file, log_file, state, kind="MC")

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
