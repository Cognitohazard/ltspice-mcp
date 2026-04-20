"""Single-job simulation wrapper: spicelib SimRunner + asyncio."""

import asyncio
import logging
from pathlib import Path

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib import now
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES, SimulationJob
from ltspice_mcp.lib.log_parser import extract_error_context
from ltspice_mcp.lib.runner_base import RunnerBase
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate unique job ID for simulation tracking."""
    return generate_id("sim")


class SimulationRunner(RunnerBase):
    """Runs one spicelib simulation per job; bridges callbacks to asyncio.

    SimRunner's completion callback fires in a worker thread; we bridge
    it back to the event loop via ``RunnerBase._bridge`` so state mutation
    stays single-threaded.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        self._runners: dict[str, SimRunner] = {}
        logger.debug(
            "SimulationRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    async def start_simulation(
        self, netlist_path: Path, job: SimulationJob, state: SessionState
    ) -> None:
        """Submit simulation to a worker thread; return immediately.

        Completion is signaled via ``job.done_event.set()`` from the
        ``_handle_completion`` callback once the worker finishes.
        """
        job_id = job.job_id

        def completion_callback(raw_file: Path | None, log_file: Path | None) -> None:
            self._bridge(
                self._handle_completion,
                job_id,
                str(raw_file) if raw_file else "",
                str(log_file) if log_file else "",
                state,
                context=f"sim job {job_id}",
            )

        def submit_sim() -> SimRunner:
            runner = self._build_sim_runner()
            # LTspice rejects files without a .cir/.net/.sp extension.
            ext = netlist_path.suffix or ".net"
            runner.run(
                str(netlist_path),
                run_filename=f"{job_id}{ext}",
                callback=completion_callback,
                callback_on_error=True,
            )
            logger.info(
                "Submitted simulation job %s: netlist=%s, simulator=%s",
                job_id,
                netlist_path,
                self.simulator_class.__name__,
            )
            return runner

        try:
            runner = await asyncio.to_thread(submit_sim)
            self._runners[job_id] = runner
            job.task = runner
            transition(job, "running", state=state, simulator=job.simulator)
        except Exception as e:
            logger.error("Failed to submit simulation %s: %s", job_id, e, exc_info=True)
            job.error = f"Submission failed: {e}"
            transition(job, "failed", state=state, error=job.error, phase="submission")

    def _handle_completion(
        self, job_id: str, raw_file: str, log_file: str, state: SessionState
    ) -> None:
        """Finalize a simulation's state once spicelib reports it's done."""
        job = state.jobs.get(job_id)
        if not job:
            logger.warning("Completed job %s not found in state", job_id)
            return
        if job.status in ("cancelled", "completed", "failed"):
            logger.debug("Job %s already in terminal state: %s", job_id, job.status)
            return

        job.completed_at = now()
        job.raw_file = Path(raw_file)
        job.log_file = Path(log_file)

        try:
            raw_size = job.raw_file.stat().st_size if job.raw_file else 0
        except OSError as e:
            logger.debug("Could not stat raw file %s: %s", job.raw_file, e)
            raw_size = 0

        self._runners.pop(job_id, None)
        if raw_size == 0:
            try:
                if job.log_file and job.log_file.exists():
                    error_context = extract_error_context(job.log_file, max_lines=20)
                    job.error = (
                        f"Simulation failed (no output generated)\n\nLog excerpt:\n{error_context}"
                    )
                else:
                    job.error = "Simulation failed (no output generated, log file missing)"
            except OSError:
                job.error = "Simulation failed (no output generated, log file not accessible)"
            logger.warning("Simulation %s failed: %s", job_id, job.error)
            transition(job, "failed", state=state, error=job.error, phase="execution")
        else:
            logger.info(
                "Simulation %s completed successfully: raw=%s, log=%s",
                job_id,
                job.raw_file,
                job.log_file,
            )
            transition(job, "completed", state=state, raw_size_bytes=raw_size)

    async def _kill(self, job_id: str) -> None:
        """Kill the spice process for a job without touching job status.

        Used by both cancel() and the tool-layer timeout path — the
        latter wants to record status='timeout' rather than 'cancelled'.
        """
        runner = self._runners.get(job_id)
        if runner is None:
            logger.warning("Cannot kill job %s: runner not found", job_id)
            return
        try:
            await asyncio.to_thread(runner.kill_all_spice)
            logger.info("Killed spice process for %s", job_id)
        except Exception as e:
            logger.warning("Error killing spice for %s: %s", job_id, e)
        finally:
            self._runners.pop(job_id, None)

    async def cancel(
        self, job: SimulationJob, state: SessionState | None = None
    ) -> None:
        """Cancel a running simulation and record the cancelled state."""
        await self._kill(job.job_id)
        if job.status not in NON_TERMINAL_LIVE_STATUSES:
            # Already terminal (completion raced with cancel). Nothing to do.
            return
        job.error = "Cancelled by user"
        transition(job, "cancelled", state=state)
