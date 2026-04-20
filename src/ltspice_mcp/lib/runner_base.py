"""Shared scaffolding for simulation/sweep/Monte-Carlo runners.

The three runners all wrap spicelib's SimRunner with the same asyncio
integration pattern: a blocking submit runs in ``asyncio.to_thread``;
per-run callbacks fire in worker threads and bridge back to the event
loop via ``call_soon_threadsafe``; cancel sets an event + kills
spice processes. This module factors that shared machinery out so each
subclass only implements what's genuinely different — stepper setup
for sweeps, tolerance configuration for Monte Carlo, single-job
tracking for sim.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import BatchJob
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

_SIMRUNNER_TIMEOUT = 600
"""Generous spicelib-level fallback timeout; real timeout is enforced at
the tool layer via ``asyncio.wait_for``."""


class RunnerBase:
    """Shared constructor + thread-safe callback bridging."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel

    def _build_sim_runner(self) -> SimRunner:
        """Construct a spicelib SimRunner with this runner's settings."""
        return SimRunner(
            simulator=self.simulator_class,
            output_folder=str(self.output_folder),
            parallel_sims=self.max_parallel,
            timeout=_SIMRUNNER_TIMEOUT,
        )

    def _bridge(
        self, handler: Callable[..., Any], *args: Any, context: str = ""
    ) -> bool:
        """Schedule ``handler`` on the event loop from a worker thread.

        Returns True on success, False if the loop is closed (graceful
        shutdown in progress). The ``context`` string appears in the
        warning message when the bridge fails.
        """
        try:
            self.loop.call_soon_threadsafe(handler, *args)
        except RuntimeError as e:
            logger.warning(
                "Event loop closed, %s not recorded: %s",
                context or "callback",
                e,
            )
            return False
        return True


class BatchRunnerBase(RunnerBase):
    """Shared batch-job machinery (cancel, per-run completion, active-runner map).

    SweepRunner and MonteCarloRunner both execute N sub-simulations and
    record per-run results identically. Only the setup (SimStepper vs
    Montecarlo) and the per-batch completion handler differ.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_runners: dict[str, SimRunner] = {}

    def _cleanup(self, job_id: str) -> None:
        """Drop per-job runner + cancel-event references."""
        self._active_runners.pop(job_id, None)
        self._cancel_events.pop(job_id, None)

    def _register_cancel(self, job_id: str) -> threading.Event:
        """Register a cancel event for a batch job and return it."""
        ev = threading.Event()
        self._cancel_events[job_id] = ev
        return ev

    def _register_runner(self, job_id: str, runner: SimRunner) -> None:
        self._active_runners[job_id] = runner

    def _record_run_completion(
        self,
        batch_job: BatchJob,
        raw_file: Path,
        log_file: Path,
        state: SessionState,
        kind: str,
    ) -> None:
        """Append one successful sub-run to ``batch_job.run_results``.

        Params are always stored as an empty dict at this stage. Sweeps
        populate them from ``stepper.sim_info`` after run_all returns;
        Monte Carlo leaves them empty (deviations are statistical).
        """
        if batch_job.status in ("cancelled", "completed", "failed"):
            logger.debug(
                "%s job %s already in terminal state '%s', ignoring run completion",
                kind,
                batch_job.job_id,
                batch_job.status,
            )
            return

        run_index = batch_job.completed_runs
        batch_job.run_results[run_index] = {
            "raw_file": str(raw_file),
            "log_file": str(log_file),
            "params": {},
        }
        batch_job.completed_runs += 1
        state.persist_batch_progress(batch_job)

        logger.debug(
            "%s job %s: run %d complete (%d/%d), raw=%s",
            kind,
            batch_job.job_id,
            run_index,
            batch_job.completed_runs,
            batch_job.total_runs,
            raw_file.name,
        )

    async def cancel(
        self, batch_job: BatchJob, state: SessionState | None = None
    ) -> None:
        """Cancel a running batch job (sweep or Monte Carlo).

        Signals the cancel event so in-flight callbacks skip their
        loop-bridge, kills spice processes, and transitions the job to
        terminal ``cancelled``. Partial results are preserved.
        """
        kind = batch_job.job_type
        logger.info("Cancelling %s job %s", kind, batch_job.job_id)

        cancel_event = self._cancel_events.get(batch_job.job_id)
        if cancel_event is not None:
            cancel_event.set()

        runner = self._active_runners.get(batch_job.job_id)
        if runner is not None:
            try:
                await asyncio.to_thread(runner.kill_all_spice)
            except Exception as e:
                logger.warning("Error killing %s job %s: %s", kind, batch_job.job_id, e)

        if batch_job.status == "running":
            transition(
                batch_job,
                "cancelled",
                state=state,
                completed_runs=batch_job.completed_runs,
                total_runs=batch_job.total_runs,
            )

        logger.info(
            "%s job %s cancelled: %d partial results preserved",
            kind,
            batch_job.job_id,
            batch_job.completed_runs,
        )

    def _mark_batch_failed(
        self, batch_job: BatchJob, state: SessionState, exc: Exception, kind: str
    ) -> None:
        """Shared handling of a batch-level exception during execution."""
        if batch_job.status != "running":
            logger.info(
                "%s job %s already terminal (%s); ignoring exception %s",
                kind,
                batch_job.job_id,
                batch_job.status,
                exc,
            )
            return
        logger.error(
            "%s job %s failed: %s", kind, batch_job.job_id, exc, exc_info=True
        )
        batch_job.error = f"{kind} execution failed: {exc}"
        transition(
            batch_job,
            "failed",
            state=state,
            error=batch_job.error,
            completed_runs=batch_job.completed_runs,
            total_runs=batch_job.total_runs,
        )
