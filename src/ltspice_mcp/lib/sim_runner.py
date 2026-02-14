"""SimulationRunner wrapper for spicelib SimRunner with asyncio integration."""

import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Type

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib.log_parser import extract_error_context, parse_success_summary
from ltspice_mcp.lib.wsl import to_windows_path
from ltspice_mcp.state import SessionState, SimulationJob

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate unique job ID for simulation tracking.

    Format: sim_{timestamp}_{uuid_short}
    - Timestamp for sortability (chronological ordering)
    - UUID for uniqueness (collision resistance)

    Returns:
        Job ID string (e.g., "sim_1707916800_a3f7b2c4")
    """
    return f"sim_{int(time.time())}_{uuid.uuid4().hex[:8]}"


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
        simulator_class: Type,
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

        # Track runner instances per job (created lazily)
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

        def completion_callback(raw_file: str, log_file: str) -> None:
            """Called by SimRunner in worker thread when simulation completes."""
            try:
                # Bridge to event loop thread-safely
                self.loop.call_soon_threadsafe(
                    self._handle_completion, job_id, raw_file, log_file, state
                )
            except RuntimeError as e:
                # Event loop was closed - graceful shutdown in progress
                logger.warning(
                    f"Event loop closed, job {job_id} completion not recorded: {e}"
                )

        def submit_sim() -> SimRunner:
            """Submit simulation to SimRunner (runs in thread pool)."""
            # Convert netlist path to Windows format if using LTSpice in WSL
            netlist_str = str(netlist_path)
            if self.simulator_class.__name__ == "LTspice":
                netlist_str = to_windows_path(netlist_path)
                logger.debug(f"Converted netlist path for LTSpice: {netlist_str}")

            # Create new SimRunner instance for this job
            runner = SimRunner(
                simulator=self.simulator_class,
                output_folder=str(self.output_folder),
                parallel_sims=self.max_parallel,
                timeout=None,  # We handle timeout at tool layer with asyncio.wait_for()
            )

            # Submit simulation (returns RunTask immediately)
            # run_filename parameter controls output naming: {job_id}.raw, {job_id}.log
            task = runner.run(
                netlist_str,
                run_filename=job_id,
                callback=completion_callback,
            )

            logger.info(
                f"Submitted simulation job {job_id}: netlist={netlist_path}, "
                f"simulator={self.simulator_class.__name__}"
            )

            # Store runner for potential cancellation
            return runner

        # Submit to thread pool using asyncio.to_thread
        try:
            runner = await asyncio.to_thread(submit_sim)
            self._runners[job_id] = runner

            # Update job status
            job.status = "running"
            job.task = runner  # Store for cancellation

        except Exception as e:
            # Submission failed (before simulator even started)
            logger.error(f"Failed to submit simulation {job_id}: {e}", exc_info=True)
            job.status = "failed"
            job.error = f"Submission failed: {e}"
            job.done_event.set()

            # Bridge error to event loop
            self.loop.call_soon_threadsafe(
                self._handle_error, job_id, f"Submission failed: {e}", state
            )

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
        # Look up job
        job = state.jobs.get(job_id)
        if not job:
            logger.warning(f"Completed job {job_id} not found in state")
            return

        # Check if job was already cancelled or completed
        if job.status in ("cancelled", "completed", "failed"):
            logger.debug(f"Job {job_id} already in terminal state: {job.status}")
            return

        # Store file paths
        from datetime import datetime

        job.completed_at = datetime.now()
        job.raw_file = Path(raw_file)
        job.log_file = Path(log_file)

        # Check if simulation actually succeeded (raw file exists and has content)
        if not job.raw_file.exists() or job.raw_file.stat().st_size == 0:
            # Simulation failed - extract error from log
            job.status = "failed"
            if job.log_file.exists():
                error_context = extract_error_context(job.log_file, max_lines=20)
                job.error = f"Simulation failed (no output generated)\n\nLog excerpt:\n{error_context}"
            else:
                job.error = "Simulation failed (no output generated, log file missing)"

            logger.warning(f"Simulation {job_id} failed: {job.error}")
        else:
            # Simulation succeeded
            job.status = "completed"
            logger.info(
                f"Simulation {job_id} completed successfully: "
                f"raw={job.raw_file}, log={job.log_file}"
            )

        # Clean up runner reference
        if job_id in self._runners:
            del self._runners[job_id]

        # Signal completion
        job.done_event.set()

    def _handle_error(self, job_id: str, error: str, state: SessionState) -> None:
        """Handle simulation error in event loop thread.

        Called via call_soon_threadsafe() when submission or execution fails.

        Args:
            job_id: Job ID of failed simulation
            error: Error message
            state: SessionState for job lookup and updates
        """
        job = state.jobs.get(job_id)
        if not job:
            logger.warning(f"Error for unknown job {job_id}: {error}")
            return

        from datetime import datetime

        job.status = "failed"
        job.error = error
        job.completed_at = datetime.now()

        # Clean up runner reference
        if job_id in self._runners:
            del self._runners[job_id]

        # Signal completion
        job.done_event.set()

        logger.error(f"Simulation {job_id} error: {error}")

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

        # Get runner reference
        runner = self._runners.get(job_id)
        if runner is None:
            logger.warning(f"Cannot cancel job {job_id}: runner not found")
            return

        # Try to stop the runner
        try:
            # SimRunner.stop() exists but may not immediately kill processes
            await asyncio.to_thread(runner.stop)
            logger.info(f"Cancelled simulation {job_id}")
        except Exception as e:
            logger.warning(f"Error stopping simulation {job_id}: {e}")

        # Update job state
        from datetime import datetime

        job.status = "cancelled"
        job.completed_at = datetime.now()
        job.error = "Cancelled by user"

        # Clean up runner reference
        if job_id in self._runners:
            del self._runners[job_id]

        # Signal completion
        job.done_event.set()
