"""Single-job simulation wrapper: spicelib SimRunner + asyncio."""

import asyncio
import logging
from pathlib import Path

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib import now
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    SimulationJob,
)
from ltspice_mcp.lib.log_parser import extract_error_context
from ltspice_mcp.lib.runner_base import RunnerBase
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token
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
        # Session-level concurrency gate: each run_simulation job builds its own
        # spicelib SimRunner (one task each), so spicelib's per-runner
        # ``parallel_sims`` never bounds the number of *independent* jobs. This
        # semaphore caps concurrent jobs at ``max_parallel`` (a job holds a slot
        # while queued→running until the process is confirmed gone). Slot release
        # is idempotent via ``_slots_held``. Scope/limitations (per-instance gate;
        # not cross-runner) are tracked in open_followups item 6.
        self._sema = asyncio.Semaphore(max_parallel)
        self._slots_held: set[str] = set()
        logger.debug(
            "SimulationRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    def _release_slot(self, job_id: str) -> None:
        """Release the concurrency slot held by ``job_id`` (idempotent).

        Safe to call from any completion / cancel / timeout / failure path and
        from either the loop thread or a bridged callback — the ``_slots_held``
        guard ensures exactly one ``Semaphore.release()`` per acquired slot.
        Must run on the event-loop thread (asyncio.Semaphore is not
        thread-safe); all callers do.
        """
        if job_id in self._slots_held:
            self._slots_held.discard(job_id)
            self._sema.release()

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

        # Acquire a concurrency slot before launching. If ``max_parallel`` sims
        # are already running, this awaits and the job stays "queued" until a
        # slot frees — the missing global gate that let N>max_parallel run.
        await self._sema.acquire()
        self._slots_held.add(job_id)
        # The job may have been cancelled / timed out while waiting here for a
        # slot. Don't launch it: release the slot and bail. Without this the
        # woken task would attempt an illegal <terminal>→running transition
        # (logged as a spurious error) or — for a timed-out job — start an
        # orphan sim the user was already told had ended.
        if job.status in TERMINAL_STATUSES:
            self._release_slot(job_id)
            return
        try:
            # Transition BEFORE submitting: ngspice can complete in <100ms,
            # racing the callback against asyncio.to_thread's resumption.
            # If the callback fires first and finds the job in "queued" state,
            # the queued→completed transition is illegal.
            transition(job, "running", state=state, simulator=job.simulator)
            runner = await asyncio.to_thread(submit_sim)
            if job.status not in TERMINAL_STATUSES:
                self._runners[job_id] = runner
                job.task = runner
            # If terminal already (cancel raced the submit), the submitted sim's
            # completion callback still fires _handle_completion, which releases
            # the slot — no release here to avoid a double-free.
        except Exception as e:
            # Submission failed: no completion callback will fire, so release the
            # slot here (idempotent).
            self._release_slot(job_id)
            logger.error("Failed to submit simulation %s: %s", job_id, e, exc_info=True)
            if job.status not in TERMINAL_STATUSES:
                job.error = f"Submission failed: {e}"
                transition(job, "failed", state=state, error=job.error, phase="submission")

    def _handle_completion(
        self, job_id: str, raw_file: str, log_file: str, state: SessionState
    ) -> None:
        """Finalize a simulation's state once spicelib reports it's done."""
        # Free the concurrency slot first, regardless of outcome — covers normal
        # completion AND the case where the sim's callback fires after a cancel /
        # timeout already marked the job terminal (idempotent via _slots_held).
        self._release_slot(job_id)
        job = state.jobs.get(job_id)
        if not job:
            logger.warning("Completed job %s not found in state", job_id)
            return
        if job.status in TERMINAL_STATUSES:
            logger.debug("Job %s already in terminal state: %s", job_id, job.status)
            return

        job.completed_at = now()
        # Bug I guard: spicelib signals failure by passing ``raw_file="."``
        # (a directory placeholder) and a ``.fail`` log file. Treat that as
        # "no raw produced" rather than storing ``Path(".")`` and trying to
        # stat the working directory below. The ``"."`` string and ``.fail``
        # suffix together cover spicelib's signalling without an extra stat.
        raw_path = Path(raw_file)
        log_path = Path(log_file)
        raw_is_placeholder = raw_file in ("", ".") or log_path.suffix == ".fail"
        if raw_is_placeholder:
            job.raw_file = None
        else:
            job.raw_file = raw_path
        job.log_file = log_path

        if raw_is_placeholder:
            raw_size = 0
        else:
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

    async def kill(self, job_id: str) -> None:
        """Kill the spice process for a job without touching job status.

        Used by both cancel() and the tool-layer timeout path — the
        latter wants to record status='timeout' rather than 'cancelled',
        so it manages job state itself and only delegates the SIGKILL.

        On WSL the actual simulator is a Windows process invisible to spicelib's
        ``kill_all_spice`` (which name-matches the Linux psutil table), so this
        also taskkills the specific Windows process by job_id — see
        ``kill_windows_ltspice_by_token``.

        It deliberately does NOT release the concurrency slot. Termination here
        is best-effort (the WSL taskkill can fail/return 0; ``kill_all_spice``
        exceptions are swallowed), so freeing the slot now would let a queued job
        launch alongside a still-running orphan, violating ``max_parallel`` in the
        exact failure mode the cap exists for. The slot is released only when the
        process is confirmed gone — i.e. when the completion callback fires
        ``_handle_completion`` (spicelib invokes it with ``callback_on_error=True``
        whenever the worker's subprocess returns, including after a kill or the
        spicelib-level timeout). A successful kill makes that fire almost
        immediately; a failed kill keeps the slot reserved until the orphan
        actually ends.
        """
        runner = self._runners.get(job_id)
        try:
            await asyncio.to_thread(self._terminate_processes, job_id, runner)
        finally:
            self._runners.pop(job_id, None)

    def _terminate_processes(self, job_id: str, runner: SimRunner | None) -> None:
        """Blocking process termination (runs in a worker thread).

        WSL: taskkill the Windows LTspice process matching this job_id.
        Native/Wine: spicelib's ``kill_all_spice`` (a harmless no-op on WSL,
        where no Linux process carries the simulator's name).
        """
        try:
            killed = kill_windows_ltspice_by_token(job_id)
            if killed:
                logger.info("Killed %d Windows sim process(es) for %s", killed, job_id)
        except Exception as e:
            logger.warning("WSL process kill for %s failed: %s", job_id, e)
        if runner is None:
            logger.debug("No spicelib runner tracked for %s (already finalized?)", job_id)
            return
        try:
            runner.kill_all_spice()
        except Exception as e:
            logger.warning("kill_all_spice for %s failed: %s", job_id, e)

    async def cancel(self, job: SimulationJob, state: SessionState | None = None) -> None:
        """Cancel a running simulation and record the cancelled state.

        Marks the job ``cancelled`` BEFORE killing the process: when the killed
        sim's completion callback later fires, ``_handle_completion`` sees a
        terminal status and discards the (now partial/truncated) raw instead of
        storing it as a success.
        """
        if job.status in NON_TERMINAL_LIVE_STATUSES:
            job.error = "Cancelled by user"
            transition(job, "cancelled", state=state)
        await self.kill(job.job_id)
