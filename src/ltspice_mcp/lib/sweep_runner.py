"""Parameter-sweep wrapper: spicelib SimStepper + asyncio."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from spicelib import SpiceEditor

from ltspice_mcp.errors import BatchJobError
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import BatchJob
from ltspice_mcp.lib.runner_base import BatchRunnerBase, wrap_runner_for_runno_callbacks
from ltspice_mcp.state import SessionState

if TYPE_CHECKING:
    from spicelib.sim.sim_runner import SimRunner
    from spicelib.sim.sim_stepping import SimStepper


def _create_stepper(editor: object, runner: SimRunner) -> SimStepper:
    """Create a SimStepper wrapping an editor and runner.

    spicelib types SimStepper's first arg as abstract BaseEditor —
    the type: ignore is isolated here.
    """
    from spicelib.sim.sim_stepping import SimStepper

    return SimStepper(editor, runner)  # type: ignore[reportAbstractUsage, reportArgumentType]


logger = logging.getLogger(__name__)


class SweepRunner(BatchRunnerBase):
    """Parameter-sweep execution via spicelib SimStepper, bridged to asyncio.

    Each sweep runs N sub-simulations, one per stepper iteration. Per-run
    completion callbacks fire in worker threads and are bridged back to
    the event loop by ``BatchRunnerBase._bridge``. ``stepper.sim_info`` is
    consulted once at the end to populate per-run parameter values.
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
            "SweepRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    async def start_sweep(self, batch_job: BatchJob, state: SessionState) -> None:
        """Submit the sweep to a worker thread; return immediately."""
        cancel_event = self._register_cancel(batch_job.job_id)

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
                context=f"sweep run (job {batch_job.job_id})",
            )

        def execute_sweep() -> None:
            if batch_job.sweep_config is None:
                raise BatchJobError(f"Sweep job {batch_job.job_id} has no sweep configuration")

            editor = SpiceEditor(str(batch_job.netlist))
            runner = wrap_runner_for_runno_callbacks(self._build_sim_runner())
            self._register_runner(batch_job.job_id, runner)
            stepper = _create_stepper(editor, runner)

            for dim in batch_job.sweep_config.dimensions:
                values = dim.resolved_values()
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
                "Starting sweep job %s: %d total runs, %d dimensions",
                batch_job.job_id,
                stepper.total_number_of_simulations(),
                len(batch_job.sweep_config.dimensions),
            )

            # The wrapped runner injects runno as a third arg; spicelib's
            # CallbackType is the unwrapped (raw_file, log_file) shape.
            stepper.run_all(callback=run_completion_callback, wait_completion=True)  # type: ignore[arg-type]

            if not cancel_event.is_set():
                self._bridge(
                    self._handle_sweep_completion,
                    batch_job.job_id,
                    stepper,
                    state,
                    context=f"sweep completion (job {batch_job.job_id})",
                )

        try:
            from ltspice_mcp.lib.observability import emit_job_event

            emit_job_event("started", batch_job, total_runs=batch_job.total_runs)
            await asyncio.to_thread(execute_sweep)
        except Exception as e:
            self._mark_batch_failed(batch_job, state, e, kind="sweep")
        finally:
            self._cleanup(batch_job.job_id)

    def _handle_run_completion(
        self,
        job_id: str,
        raw_file: Path,
        log_file: Path,
        state: SessionState,
        runno: int | None = None,
    ) -> None:
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("Run completion for unknown sweep job %s", job_id)
            return
        self._record_run_completion(
            batch_job, raw_file, log_file, state, kind="Sweep", runno=runno
        )

    def _handle_sweep_completion(
        self,
        job_id: str,
        stepper: SimStepper,
        state: SessionState,
    ) -> None:
        """Populate per-run params from ``stepper.sim_info`` and finalize.

        ``sim_info`` is keyed by runno (1-based int); ``run_results`` is
        keyed by 0-based runno parsed from the raw_file basename
        (see ``runner_base._record_run_completion``). The (runno - 1)
        lookup pairs them deterministically regardless of completion
        order, which is what makes parallel sweeps correctly labeled.
        """
        batch_job = state.batch_jobs.get(job_id)
        if not batch_job:
            logger.warning("Sweep completion for unknown batch job %s", job_id)
            return
        if batch_job.status == "cancelled":
            logger.debug(
                "Sweep job %s was cancelled — preserving %d partial results",
                job_id,
                len(batch_job.run_results),
            )
            return

        for runno, info in stepper.sim_info.items():
            run_key = runno - 1
            if run_key not in batch_job.run_results:
                continue
            params = {}
            for key, val in info.items():
                if key == "netlist":
                    continue
                try:
                    params[key] = parse_spice_value(str(val))
                except (ValueError, TypeError):
                    params[key] = val
            batch_job.run_results[run_key]["params"] = params

        transition(
            batch_job,
            "completed",
            state=state,
            completed_runs=batch_job.completed_runs,
            total_runs=batch_job.total_runs,
        )

        logger.info(
            "Sweep job %s completed: %d/%d runs finished",
            job_id,
            batch_job.completed_runs,
            batch_job.total_runs,
        )
