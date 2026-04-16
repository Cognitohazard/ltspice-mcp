"""SimulationRunner wrapper for spicelib SimRunner with asyncio integration."""

import asyncio
import logging
from pathlib import Path

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib import now
from ltspice_mcp.lib.log_parser import extract_error_context
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.state import SessionState, SimulationJob

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate unique job ID for simulation tracking."""
    return generate_id("sim")


class SimulationRunner:
    """Wraps spicelib SimRunner with asyncio integration.

    SimRunner uses callbacks that fire in worker threads/processes.
    This class bridges those callbacks to the asyncio event loop
    using loop.call_soon_threadsafe() for thread safety.

    Attributes:
        loop: Asyncio event loop for thread-safe callback bridging
        simulator_class: Spicelib simulator class (LTspice, NGspice, etc.)
        output_folder: Directory where raw/log files are written
        max_parallel: Maximum number of parallel simulations
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        """Initialize SimulationRunner.

        Args:
            loop: Asyncio event loop reference for call_soon_threadsafe()
            simulator_class: Spicelib simulator class to use
            output_folder: Output directory for simulation files
            max_parallel: Maximum parallel simulations (default: 4)
        """
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel

        self._runners: dict[str, SimRunner] = {}

        logger.debug(
            f"SimulationRunner initialized: simulator={simulator_class.__name__}, "
            f"output={output_folder}, max_parallel={max_parallel}"
        )

    async def start_simulation(
        self, netlist_path: Path, job: SimulationJob, state: SessionState
    ) -> None:
        """Start simulation in background thread with asyncio integration.

        Creates a new SimRunner instance for this job, sets up callbacks,
        and submits the simulation. Callbacks fire in worker context and
        bridge back to event loop via call_soon_threadsafe().

        Args:
            netlist_path: Path to netlist file to simulate
            job: SimulationJob tracking this simulation
            state: SessionState for job updates

        Note:
            This method returns immediately after submitting the simulation.
            Completion is signaled via job.done_event.set() in the callback.
        """
        job_id = job.job_id

        def completion_callback(raw_file: Path | None, log_file: Path | None) -> None:
            """Called by SimRunner in worker thread when simulation completes."""
            try:
                self.loop.call_soon_threadsafe(
                    self._handle_completion,
                    job_id,
                    str(raw_file) if raw_file else "",
                    str(log_file) if log_file else "",
                    state,
                )
            except RuntimeError as e:
                # Event loop was closed - graceful shutdown in progress
                logger.warning(f"Event loop closed, job {job_id} completion not recorded: {e}")

        def submit_sim() -> SimRunner:
            """Submit simulation to SimRunner (runs in thread pool)."""
            netlist_str = str(netlist_path)

            runner = SimRunner(
                simulator=self.simulator_class,
                output_folder=str(self.output_folder),
                parallel_sims=self.max_parallel,
                timeout=600,  # Generous fallback; real timeout is at tool layer via asyncio.wait_for()
            )

            # run_filename must keep a netlist extension — LTspice ignores
            # files without .cir/.net/.sp extension (exits with code 1).
            ext = netlist_path.suffix or ".net"
            run_name = f"{job_id}{ext}"
            runner.run(
                netlist_str,
                run_filename=run_name,
                callback=completion_callback,
                callback_on_error=True,
            )

            logger.info(
                f"Submitted simulation job {job_id}: netlist={netlist_path}, "
                f"simulator={self.simulator_class.__name__}"
            )

            return runner

        try:
            runner = await asyncio.to_thread(submit_sim)
            self._runners[job_id] = runner

            job.status = "running"
            job.task = runner  # Store for cancellation

        except Exception as e:
            # Submission failed (before simulator even started)
            logger.error(f"Failed to submit simulation {job_id}: {e}", exc_info=True)
            job.status = "failed"
            job.error = f"Submission failed: {e}"
            job.completed_at = now()
            job.done_event.set()

    def _handle_completion(
        self, job_id: str, raw_file: str, log_file: str, state: SessionState
    ) -> None:
        """Handle simulation completion in event loop thread.

        Called via call_soon_threadsafe() from worker thread callback.

        Args:
            job_id: Job ID of completed simulation
            raw_file: Path to generated .raw file
            log_file: Path to generated .log file
            state: SessionState for job lookup and updates
        """
        job = state.jobs.get(job_id)
        if not job:
            logger.warning(f"Completed job {job_id} not found in state")
            return

        if job.status in ("cancelled", "completed", "failed"):
            logger.debug(f"Job {job_id} already in terminal state: {job.status}")
            return

        job.completed_at = now()
        job.raw_file = Path(raw_file)
        job.log_file = Path(log_file)

        try:
            raw_size = job.raw_file.stat().st_size if job.raw_file else 0
        except OSError as e:
            logger.debug("Could not stat raw file %s: %s", job.raw_file, e)
            raw_size = 0

        if raw_size == 0:
            job.status = "failed"
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

            logger.warning(f"Simulation {job_id} failed: {job.error}")
        else:
            job.status = "completed"
            logger.info(
                f"Simulation {job_id} completed successfully: "
                f"raw={job.raw_file}, log={job.log_file}"
            )

        self._runners.pop(job_id, None)
        job.done_event.set()

    async def cancel(self, job: SimulationJob) -> None:
        """Cancel a running simulation.

        Attempts to stop the SimRunner and kill the simulator process.
        Sets job status to cancelled and signals completion.

        Args:
            job: SimulationJob to cancel

        Note:
            spicelib SimRunner doesn't have a direct cancel method.
            We try to stop() the runner and clean up state.
        """
        job_id = job.job_id

        runner = self._runners.get(job_id)
        if runner is None:
            logger.warning(f"Cannot cancel job {job_id}: runner not found")
            return

        try:
            await asyncio.to_thread(runner.kill_all_spice)
            logger.info(f"Cancelled simulation {job_id}")
        except Exception as e:
            logger.warning(f"Error cancelling simulation {job_id}: {e}")

        job.status = "cancelled"
        job.completed_at = now()
        job.error = "Cancelled by user"

        self._runners.pop(job_id, None)
        job.done_event.set()
