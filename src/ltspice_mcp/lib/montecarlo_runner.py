"""MonteCarloRunner wrapper for spicelib Montecarlo with asyncio integration."""

import asyncio
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Type

from spicelib.sim.sim_runner import SimRunner
from spicelib.sim.tookit.montecarlo import Montecarlo

from ltspice_mcp.lib.wsl import to_windows_path
from ltspice_mcp.state import BatchJob, SessionState

logger = logging.getLogger(__name__)


class SeededMontecarlo(Montecarlo):
    """Montecarlo subclass that supports best-effort reproducible seeding.

    spicelib's Montecarlo._get_sim_value() creates new random.Random() instances
    on each call without arguments, so they seed themselves from os.urandom(),
    ignoring module-level random.seed() in most Python implementations.

    This subclass calls random.seed(seed) before run_analysis() as a best-effort
    reproducibility measure. Note that this does NOT guarantee reproducibility
    across different Python versions or spicelib versions since the internal
    random.Random() instances use os.urandom() seeding.

    Limitation: True reproducibility requires patching spicelib's _get_sim_value
    to use a seeded Random instance. This is left for a future enhancement.
    """

    def __init__(self, circuit_file, runner=None, seed: int | None = None):
        """Initialize SeededMontecarlo.

        Args:
            circuit_file: Path to the netlist file (str)
            runner: SimRunner instance for executing simulations
            seed: Optional RNG seed for best-effort reproducibility
        """
        super().__init__(circuit_file, runner)
        self._seed = seed

    def run_analysis(self, **kwargs):
        """Run Monte Carlo analysis with optional best-effort seeding.

        If a seed was provided, calls random.seed(seed) before running to seed
        the module-level random state. This provides best-effort reproducibility
        in Python implementations where random.Random() without args uses module-
        level state (not guaranteed across all Python versions).

        Args:
            **kwargs: Passed directly to Montecarlo.run_analysis()
        """
        if self._seed is not None:
            # Best-effort: seed module-level random state before run_analysis.
            # NOTE: spicelib internally creates random.Random() instances without
            # a seed argument, which use os.urandom() in CPython. This seed call
            # may not achieve full reproducibility across all Python implementations.
            random.seed(self._seed)
            logger.debug(
                f"SeededMontecarlo: applied seed={self._seed} (best-effort reproducibility)"
            )
        super().run_analysis(**kwargs)


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

        Creates a SeededMontecarlo instance, applies tolerances from mc_config,
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
            # Convert netlist path to Windows format if using LTSpice in WSL
            netlist_path = batch_job.netlist
            if self.simulator_class.__name__ == "LTspice":
                netlist_str = to_windows_path(netlist_path)
                logger.debug(
                    f"Converted netlist path for LTSpice: {netlist_path} -> {netlist_str}"
                )
            else:
                netlist_str = str(netlist_path)

            # Create SimRunner for this Monte Carlo execution
            runner = SimRunner(
                simulator=self.simulator_class,
                output_folder=str(self.output_folder),
                parallel_sims=self.max_parallel,
                timeout=None,  # Tool layer handles timeout via asyncio.wait_for()
            )

            # Create SeededMontecarlo - takes circuit_file str (not SpiceEditor)
            # It manages its own editor internally (research pattern 3)
            mc_config = batch_job.mc_config
            mc = SeededMontecarlo(
                netlist_str,
                runner,
                seed=mc_config.seed if mc_config.seed is not None else None,
            )

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
                f"seed={mc_config.seed}, "
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
