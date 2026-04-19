"""SweepRunner wrapper for spicelib SimStepper with asyncio integration."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from spicelib import SpiceEditor
from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib import now
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.sweep_utils import generate_sweep_range
from ltspice_mcp.state import BatchJob, SessionState

if TYPE_CHECKING:
    from spicelib.sim.sim_stepping import SimStepper


def _create_stepper(editor: object, runner: SimRunner) -> SimStepper:
    """Create a SimStepper wrapping an editor and runner.

    spicelib types SimStepper's first arg as abstract BaseEditor —
    the type: ignore is isolated here.
    """
    from spicelib.sim.sim_stepping import SimStepper

    return SimStepper(editor, runner)  # type: ignore[reportAbstractUsage, reportArgumentType]

logger = logging.getLogger(__name__)


class SweepRunner:
    """Wraps spicelib SimStepper with asyncio integration for parameter sweeps.

    SimStepper uses a blocking run_all() that blocks the calling thread.
    This class bridges SimStepper's synchronous execution to the asyncio event
    loop using asyncio.to_thread() for blocking work and loop.call_soon_threadsafe()
    for callback bridging — mirroring the SimulationRunner pattern from Phase 3.

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
        """Initialize SweepRunner.

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
        self._cancel_events: dict[str, threading.Event] = {}
        self._active_runners: dict[str, SimRunner] = {}

        logger.debug(
            f"SweepRunner initialized: simulator={simulator_class.__name__}, "
            f"output={output_folder}, max_parallel={max_parallel}"
        )

    async def start_sweep(self, batch_job: BatchJob, state: SessionState) -> None:
        """Start parameter sweep in background thread with asyncio integration.

        Creates a SimStepper instance for the sweep dimensions, executes all
        runs in a thread pool, and bridges per-run callbacks to the event loop
        via call_soon_threadsafe().

        Args:
            batch_job: BatchJob tracking this sweep execution
            state: SessionState for job updates

        Note:
            This method returns after submitting the sweep to the thread pool.
            Completion is signaled via batch_job.done_event.set() in the completion handler.
        """
        cancel_event = threading.Event()
        self._cancel_events[batch_job.job_id] = cancel_event

        def run_completion_callback(raw_file, log_file) -> None:
            """Called by SimRunner for each completed run (in worker thread).

            Bridges per-run completion to the event loop thread-safely.
            raw_file and log_file may be Path or str depending on spicelib version.
            """
            try:
                if cancel_event.is_set():
                    return
                raw_file = Path(raw_file)
                log_file = Path(log_file)
                self.loop.call_soon_threadsafe(
                    self._handle_run_completion,
                    batch_job.job_id,
                    raw_file,
                    log_file,
                    state,
                )
            except RuntimeError as e:
                # Event loop was closed - graceful shutdown in progress
                logger.warning(
                    f"Event loop closed, sweep run completion not recorded "
                    f"for job {batch_job.job_id}: {e}"
                )

        def execute_sweep() -> None:
            """Execute SimStepper in thread pool (blocking call - safe in worker thread)."""
            # Pass Linux path — WSL path conversion is handled by the simulator
            netlist_path = batch_job.netlist
            netlist_str = str(netlist_path)

            # Create fresh SpiceEditor per batch to avoid race conditions
            # (anti-pattern: sharing SpiceEditor across concurrent batch jobs)
            editor = SpiceEditor(netlist_str)

            # Create SimRunner for this sweep execution
            runner = SimRunner(
                simulator=self.simulator_class,
                output_folder=str(self.output_folder),
                parallel_sims=self.max_parallel,
                timeout=600,  # Tool layer handles timeout via asyncio.wait_for()
            )
            self._active_runners[batch_job.job_id] = runner

            # Create SimStepper wrapping editor + runner
            stepper = _create_stepper(editor, runner)

            # Add each sweep dimension
            if batch_job.sweep_config is None:
                raise BatchJobError(f"Sweep job {batch_job.job_id} has no sweep configuration")
            for dim in batch_job.sweep_config.dimensions:
                values = generate_sweep_range(dim.start, dim.stop, dim.step, dim.points, dim.scale)
                if dim.type == "component":
                    stepper.add_value_sweep(dim.name, values)
                elif dim.type == "parameter":
                    stepper.add_param_sweep(dim.name, values)
                else:
                    raise ValueError(
                        f"Unknown sweep dimension type '{dim.type}'. "
                        f"Expected 'component' or 'parameter'."
                    )

            logger.info(
                f"Starting sweep job {batch_job.job_id}: "
                f"{stepper.total_number_of_simulations()} total runs, "
                f"{len(batch_job.sweep_config.dimensions)} dimensions"
            )

            # Execute all sweep runs (blocks until complete - safe in thread pool)
            # wait_completion=True ensures stepper.sim_info is fully populated before return
            stepper.run_all(callback=run_completion_callback, wait_completion=True)

            # Bridge sweep completion to event loop for final state update
            if not cancel_event.is_set():
                self.loop.call_soon_threadsafe(
                    self._handle_sweep_completion,
                    batch_job.job_id,
                    stepper,
                    state,
                )

        # Submit to thread pool using asyncio.to_thread (non-blocking)
        try:
            batch_job.status = "running"
            state.persist_job(batch_job)
            await asyncio.to_thread(execute_sweep)
        except Exception as e:
            if batch_job.status == "cancelled":
                logger.info(f"Sweep job {batch_job.job_id} stopped after cancellation")
                return
            # Submission or execution failed - update batch job status
            logger.error(f"Sweep job {batch_job.job_id} failed: {e}", exc_info=True)
            batch_job.status = "failed"
            batch_job.error = f"Sweep execution failed: {e}"
            batch_job.completed_at = now()
            batch_job.done_event.set()
            state.persist_job(batch_job)
        finally:
            self._active_runners.pop(batch_job.job_id, None)
            self._cancel_events.pop(batch_job.job_id, None)

    def _handle_run_completion(
        self,
        job_id: str,
        raw_file: Path,
        log_file: Path,
        state: SessionState,
    ) -> None:
        """Handle per-run completion in event loop thread.

        Called via call_soon_threadsafe() from worker thread callback.
        Increments completed_runs and stores file paths for each run.

        Args:
            job_id: Batch job ID
            raw_file: Path to generated .raw file for this run
            log_file: Path to generated .log file for this run
            state: SessionState for job lookup
        """
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning(f"Run completion for unknown batch job {job_id}")
            return

        if batch_job.status in ("cancelled", "completed", "failed"):
            logger.debug(
                f"Sweep job {job_id} already in terminal state '{batch_job.status}', "
                f"ignoring run completion"
            )
            return

        run_index = batch_job.completed_runs

        batch_job.run_results[run_index] = {
            "raw_file": str(raw_file),
            "log_file": str(log_file),
            "params": {},  # Populated in _handle_sweep_completion from sim_info
        }

        batch_job.completed_runs += 1
        state.persist_batch_progress(batch_job)

        logger.debug(
            f"Sweep job {job_id}: run {run_index} complete "
            f"({batch_job.completed_runs}/{batch_job.total_runs}), "
            f"raw={raw_file.name}"
        )

    def _handle_sweep_completion(
        self,
        job_id: str,
        stepper: SimStepper,
        state: SessionState,
    ) -> None:
        """Handle overall sweep completion in event loop thread.

        Called via call_soon_threadsafe() after stepper.run_all() returns.
        Populates run_results params from stepper.sim_info and marks job complete.

        Args:
            job_id: Batch job ID
            stepper: Completed SimStepper instance with sim_info populated
            state: SessionState for job lookup
        """
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning(f"Sweep completion for unknown batch job {job_id}")
            return

        # Guard: skip if already cancelled (partial results preserved)
        if batch_job.status == "cancelled":
            logger.debug(
                f"Sweep job {job_id} was cancelled — preserving "
                f"{len(batch_job.run_results)} partial results"
            )
            return

        # Populate params in run_results from stepper.sim_info
        # sim_info is keyed by runno (int), values are {param: value, 'netlist': filename}
        # run_results is keyed by run_index (0-based completed_runs counter)
        # We match by index order since both are populated in the same order
        sim_info_items = sorted(stepper.sim_info.items())  # Sort by runno for stable ordering

        for order_idx, (_runno, info) in enumerate(sim_info_items):
            if order_idx in batch_job.run_results:
                # Normalize param values to float for consistent filtering
                # (research pitfall 4: values may be stored as engineering notation strings)
                params = {}
                for key, val in info.items():
                    if key == "netlist":
                        continue  # Skip the netlist filename entry
                    try:
                        # Try to parse as SPICE value (handles "1k", "10n", etc.)
                        params[key] = parse_spice_value(str(val))
                    except (ValueError, TypeError):
                        # Keep as string if not parseable as float
                        params[key] = val
                batch_job.run_results[order_idx]["params"] = params

        # Mark job as completed
        batch_job.status = "completed"
        batch_job.completed_at = now()
        batch_job.done_event.set()
        state.persist_job(batch_job)

        logger.info(
            f"Sweep job {job_id} completed: "
            f"{batch_job.completed_runs}/{batch_job.total_runs} runs finished"
        )

    async def cancel(
        self, batch_job: BatchJob, state: SessionState | None = None
    ) -> None:
        """Cancel a running sweep batch job.

        Sets job status to cancelled and signals completion. Partial results
        from completed runs are preserved in run_results per user decision.

        Args:
            batch_job: BatchJob to cancel
            state: Optional SessionState for persistence of the cancelled record.
        """
        logger.info(f"Cancelling sweep job {batch_job.job_id}")

        cancel_event = self._cancel_events.get(batch_job.job_id)
        if cancel_event is not None:
            cancel_event.set()

        runner = self._active_runners.get(batch_job.job_id)
        if runner is not None:
            try:
                await asyncio.to_thread(runner.kill_all_spice)
            except Exception as e:
                logger.warning(f"Error killing sweep job {batch_job.job_id}: {e}")

        # Update job state - partial results preserved (run_results keeps completed entries)
        batch_job.status = "cancelled"
        batch_job.completed_at = now()
        batch_job.done_event.set()
        if state is not None:
            state.persist_job(batch_job)

        logger.info(
            f"Sweep job {batch_job.job_id} cancelled: "
            f"{batch_job.completed_runs} partial results preserved"
        )
