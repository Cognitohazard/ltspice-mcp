"""MonteCarloRunner wrapper for spicelib Montecarlo with asyncio integration."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Type

from spicelib.sim.sim_runner import SimRunner
from spicelib.sim.tookit.montecarlo import Montecarlo

from ltspice_mcp.state import BatchJob, SessionState

logger = logging.getLogger(__name__)


class MonteCarloRunner:
    """Wraps spicelib Montecarlo with asyncio integration for Monte Carlo analysis.

    Montecarlo uses a blocking run_analysis() that blocks the calling thread.
    This class bridges Montecarlo's synchronous execution to the asyncio event
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
        simulator_class: Type,
        output_folder: Path,
        max_parallel: int = 4,
    ):
        """Initialize MonteCarloRunner.

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

        logger.debug(
            f"MonteCarloRunner initialized: simulator={simulator_class.__name__}, "
            f"output={output_folder}, max_parallel={max_parallel}"
        )

    async def start_montecarlo(self, batch_job: BatchJob, state: SessionState) -> None:
        """Start Monte Carlo analysis in background thread with asyncio integration.

        Creates a Montecarlo instance, applies tolerances from mc_config,
        and executes all runs in a thread pool. Per-run callbacks bridge to the
        event loop via call_soon_threadsafe().

        Args:
            batch_job: BatchJob tracking this Monte Carlo execution
            state: SessionState for job updates

        Note:
            This method returns after submitting the analysis to the thread pool.
            Completion is signaled via batch_job.done_event.set() in the completion handler.
        """

        def run_completion_callback(raw_file, log_file) -> None:
            """Called by SimRunner for each completed run (in worker thread).

            Bridges per-run completion to the event loop thread-safely.
            raw_file and log_file may be Path or str depending on spicelib version.
            """
            try:
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
                    f"Event loop closed, MC run completion not recorded "
                    f"for job {batch_job.job_id}: {e}"
                )

        def execute_montecarlo() -> None:
            """Execute Montecarlo in thread pool (blocking call - safe in worker thread)."""
            # Pass Linux path — WSL path conversion is handled by the simulator
            netlist_path = batch_job.netlist
            netlist_str = str(netlist_path)

            # Create SimRunner for this Monte Carlo execution
            runner = SimRunner(
                simulator=self.simulator_class,
                output_folder=str(self.output_folder),
                parallel_sims=self.max_parallel,
                timeout=600,  # Tool layer handles timeout via asyncio.wait_for()
            )

            # Create Montecarlo - takes circuit_file str (not SpiceEditor)
            # It manages its own editor internally
            assert batch_job.mc_config is not None
            mc_config = batch_job.mc_config
            mc = Montecarlo(netlist_str, runner)

            # Apply type-level tolerances first (prefix like "R", "C", "L")
            # These set defaults for all components of that type
            for ref, (tol, dist) in mc_config.type_tolerances.items():
                mc.set_tolerance(ref, tol, distribution=dist)
                logger.debug(
                    f"MC job {batch_job.job_id}: set type tolerance {ref}={tol} ({dist})"
                )

            # Apply component-level overrides (specific refs like "R1", "C3")
            # These override the type-level defaults for individual components
            for ref, (tol, dist) in mc_config.component_overrides.items():
                mc.set_tolerance(ref, tol, distribution=dist)
                logger.debug(
                    f"MC job {batch_job.job_id}: set component override {ref}={tol} ({dist})"
                )

            logger.info(
                f"Starting Monte Carlo job {batch_job.job_id}: "
                f"{batch_job.total_runs} runs, "
                f"type_tolerances={list(mc_config.type_tolerances.keys())}, "
                f"component_overrides={list(mc_config.component_overrides.keys())}"
            )

            # Execute all Monte Carlo runs (blocks until complete - safe in thread pool)
            mc.run_analysis(
                callback=run_completion_callback,
                num_runs=batch_job.total_runs,
            )

            # Bridge Monte Carlo completion to event loop for final state update
            self.loop.call_soon_threadsafe(
                self._handle_mc_completion,
                batch_job.job_id,
                state,
            )

        # Submit to thread pool using asyncio.to_thread (non-blocking)
        try:
            batch_job.status = "running"
            await asyncio.to_thread(execute_montecarlo)
        except Exception as e:
            # Submission or execution failed - update batch job status
            logger.error(
                f"Monte Carlo job {batch_job.job_id} failed: {e}", exc_info=True
            )
            batch_job.status = "failed"
            batch_job.error = f"Monte Carlo execution failed: {e}"
            batch_job.completed_at = datetime.now()
            batch_job.done_event.set()

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
        Monte Carlo runs don't have explicit per-run parameter values like
        sweeps — params dict is left empty (deviation info not available per-run).

        Args:
            job_id: Batch job ID
            raw_file: Path to generated .raw file for this run
            log_file: Path to generated .log file for this run
            state: SessionState for job lookup
        """
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning(f"Run completion for unknown MC batch job {job_id}")
            return

        # Guard: skip if job already in terminal state (cancelled, completed, failed)
        if batch_job.status in ("cancelled", "completed", "failed"):
            logger.debug(
                f"MC job {job_id} already in terminal state '{batch_job.status}', "
                f"ignoring run completion"
            )
            return

        # Determine run index from completed_runs count
        run_index = batch_job.completed_runs

        # Store run result - MC runs don't have explicit per-run params like sweeps
        # params remains empty; deviation info is statistical, not per-run trackable
        batch_job.run_results[run_index] = {
            "raw_file": str(raw_file),
            "log_file": str(log_file),
            "params": {},  # MC runs: no explicit per-run parameter values
        }

        batch_job.completed_runs += 1

        logger.debug(
            f"MC job {job_id}: run {run_index} complete "
            f"({batch_job.completed_runs}/{batch_job.total_runs}), "
            f"raw={raw_file.name}"
        )

    def _handle_mc_completion(self, job_id: str, state: SessionState) -> None:
        """Handle overall Monte Carlo completion in event loop thread.

        Called via call_soon_threadsafe() after mc.run_analysis() returns.
        Marks the batch job as completed.

        Args:
            job_id: Batch job ID
            state: SessionState for job lookup
        """
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning(f"MC completion for unknown batch job {job_id}")
            return

        # Guard: skip if already cancelled (partial results preserved)
        if batch_job.status == "cancelled":
            logger.debug(
                f"MC job {job_id} was cancelled — preserving "
                f"{len(batch_job.run_results)} partial results"
            )
            return

        # Mark job as completed
        batch_job.status = "completed"
        batch_job.completed_at = datetime.now()
        batch_job.done_event.set()

        logger.info(
            f"Monte Carlo job {job_id} completed: "
            f"{batch_job.completed_runs}/{batch_job.total_runs} runs finished"
        )

    async def cancel(self, batch_job: BatchJob) -> None:
        """Cancel a running Monte Carlo batch job.

        Sets job status to cancelled and signals completion. Partial results
        from completed runs are preserved in run_results per user decision.

        Args:
            batch_job: BatchJob to cancel
        """
        logger.info(f"Cancelling Monte Carlo job {batch_job.job_id}")

        # Update job state - partial results preserved (run_results keeps completed entries)
        batch_job.status = "cancelled"
        batch_job.completed_at = datetime.now()
        batch_job.done_event.set()

        logger.info(
            f"Monte Carlo job {batch_job.job_id} cancelled: "
            f"{batch_job.completed_runs} partial results preserved"
        )
