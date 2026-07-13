"""Single-job simulation wrapper: spicelib SimRunner + asyncio."""

import asyncio
import logging
from pathlib import Path
from typing import NamedTuple

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib import now
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    SimulationJob,
)
from ltspice_mcp.lib.log_parser import (
    extract_error_context,
    extract_log_diagnostics,
    is_op_stepping_failure,
    op_ladder_exhausted,
)
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.runner_base import (
    DEFAULT_MAX_PARALLEL,
    RunnerBase,
    discard_logopinfo_netlist,
)
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def generate_job_id() -> str:
    """Generate unique job ID for simulation tracking."""
    return generate_id("sim")


# Terminal statuses reached by killing a live simulator (vs. a clean finish).
# A run ended this way leaves a partial output behind — a timed-out LTspice
# .raw can reach several GB — which must be reclaimed, unlike a completed run's
# artifacts which the user still reads.
_KILLED_STATUSES = ("cancelled", "timeout")


class RunOutcome(NamedTuple):
    """Filesystem-derived facts about a finished run, collected off the loop."""

    raw_file: str
    """Path of the produced raw, or "" when no raw data exists."""
    log_file: str
    """Path of the produced log, or "" when spicelib reported none."""
    raw_size: int
    error: str | None
    """Failure message (with log excerpt) when the run failed; None otherwise.
    ``error is None and raw_size == 0`` is the log-only completion: a clean
    simulator exit whose results (if any) live in the log, not a raw file."""


def collect_run_outcome(raw_file: str, log_file: str) -> RunOutcome:
    """Stat/read a finished run's artifacts and classify the outcome.

    Must run on a worker thread, never the event loop: the log read below is
    unbounded file I/O that can stall on a pathological abort log or a hung
    network/DrvFs mount, and a stalled event loop freezes every in-flight
    request in the server process, not just this job.
    """
    log_path = Path(log_file)
    # spicelib signals a simulator failure (nonzero exit) by renaming the log
    # to ``.fail`` and passing no real raw path ("" or "."). Relay that
    # verdict — it is the simulator's own exit status.
    sim_failed = raw_file in ("", ".") or log_path.suffix == ".fail"
    raw_size = 0
    if not sim_failed:
        try:
            raw_size = Path(raw_file).stat().st_size
        except FileNotFoundError:
            raw_size = 0
        except OSError as e:
            # The raw exists (or at least isn't provably absent) but can't be
            # statted — permissions, a flaky mount. That is an artifact-access
            # failure, not a log-only run; keep the path so the caller can
            # diagnose it instead of reporting a false success.
            return RunOutcome(
                raw_file, log_file, 0, f"Simulation finished but its raw file is unreadable: {e}"
            )
    if raw_size > 0:
        return RunOutcome(raw_file, log_file, raw_size, None)

    try:
        log_exists = bool(log_file) and log_path.exists()
    except OSError:
        log_exists = False

    # Clean exit but no raw data: a deck driven by a .control script (ngspice)
    # legitimately prints its results to the log and writes no raw at all.
    # When the log parses free of errors, that's a completed log-only run,
    # not a failure. An OP "gmin stepping failed" rung on its own is a
    # recoverable mid-ladder step (ngspice tries the next method and may solve),
    # not a terminal error — with no raw to gate on (unlike build_simulation_
    # summary's raw-validity check), keep the run failed only if a genuine
    # terminal error is present OR the whole stepping ladder was exhausted.
    if not sim_failed and log_exists:
        errors = extract_log_diagnostics(log_path)["errors"]
        non_rung = [e for e in errors if not is_op_stepping_failure(e)]
        if not non_rung and not op_ladder_exhausted(errors):
            return RunOutcome("", log_file, 0, None)

    if log_exists:
        context = extract_error_context(log_path, max_lines=20)
        error = f"Simulation failed (no output generated)\n\nLog excerpt:\n{context}"
    else:
        error = "Simulation failed (no output generated, log file missing)"
    return RunOutcome("" if sim_failed else raw_file, log_file, 0, error)


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
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        # Session-level concurrency gate: each run_simulation job builds its own
        # spicelib SimRunner (one task each), so spicelib's per-runner
        # ``parallel_sims`` never bounds the number of *independent* jobs. This
        # semaphore caps concurrent jobs at ``max_parallel`` (a job holds a slot
        # while queued→running until the process is confirmed gone). Slot release
        # is idempotent via ``_slots_held``. Scope/limitations (per-instance gate;
        # not cross-runner) are known and deferred.
        self._sema = asyncio.Semaphore(max_parallel)
        self._slots_held: set[str] = set()
        logger.debug(
            "SimulationRunner initialized: simulator=%s, output=%s, max_parallel=%d",
            simulator_class.__name__,
            output_folder,
            max_parallel,
        )

    def has_active_work(self) -> bool:
        """Whether any job launched by this instance still holds a slot."""
        return bool(self._slots_held)

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
            # Collect all filesystem facts HERE, on spicelib's worker thread.
            # The bridged handler runs on the event loop, where a stalled
            # read would freeze every in-flight request in the process.
            try:
                outcome = collect_run_outcome(
                    str(raw_file) if raw_file else "",
                    str(log_file) if log_file else "",
                )
            except Exception as e:  # spicelib swallows callback exceptions;
                # a raise here would leave the job dangling forever.
                outcome = RunOutcome("", "", 0, f"Simulation failed (outcome collection: {e})")
            self._bridge(
                self._handle_completion,
                job_id,
                outcome,
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
                # Capture the simulator's stdout/stderr into a sibling
                # ``.exe.log`` so ngspice's stdout-only diagnostics (which
                # bypass the ``-o`` log) are visible to extract_log_diagnostics.
                exe_log=True,
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
        try:
            # The job may have been cancelled / timed out while waiting here for
            # a slot. Don't launch it: release the slot and bail. Without this the
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
                    job.task = runner
                # If terminal already (cancel raced the submit), the submitted
                # sim's completion callback still fires _handle_completion, which
                # releases the slot — no release here to avoid a double-free.
            except Exception as e:
                # Submission failed: no completion callback will fire, so release
                # the slot here (idempotent).
                self._release_slot(job_id)
                logger.error("Failed to submit simulation %s: %s", job_id, e, exc_info=True)
                if job.status not in TERMINAL_STATUSES:
                    job.error = f"Submission failed: {e}"
                    transition(job, "failed", state=state, error=job.error, phase="submission")
        finally:
            # A generated runnable (a logopinfo-augmented copy) was passed instead
            # of the user's own netlist; spicelib has already staged it into the
            # run folder by now (_prepare_sim runs synchronously inside run()), so
            # the per-job source copy is no longer needed. The marker guard inside
            # the helper makes this incapable of touching the user's file.
            await asyncio.to_thread(discard_logopinfo_netlist, netlist_path)

    def _handle_completion(self, job_id: str, outcome: RunOutcome, state: SessionState) -> None:
        """Finalize a simulation's state once spicelib reports it's done.

        Runs on the event loop (bridged from the worker thread); every
        filesystem fact arrives pre-collected in ``outcome`` so nothing here
        can block the loop — see ``collect_run_outcome``.
        """
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
            # A killed run exits nonzero, so spicelib renamed its log to
            # ``.fail`` — the timeout path derived ``{job_id}.log`` before that
            # rename. Point the job at the file that actually exists so the
            # post-mortem excerpt stays readable from check_job.
            if outcome.log_file and job.log_file != Path(outcome.log_file):
                job.log_file = Path(outcome.log_file)
                state.persist_job(job)
            # This callback fires when the simulator process finally exits, so
            # a killed run's file handle is now released and its partial output
            # is safe to delete. Without this, a cancelled/timed-out run strands
            # its (possibly multi-GB) partial .raw on disk forever. The unlinks
            # go to a worker thread — they too can stall on a hung mount.
            if job.status in _KILLED_STATUSES:
                self.loop.run_in_executor(None, self._remove_run_artifacts, job_id)
            return

        job.completed_at = now()
        job.raw_file = Path(outcome.raw_file) if outcome.raw_file else None
        job.log_file = Path(outcome.log_file) if outcome.log_file else None

        if outcome.error is not None:
            job.error = outcome.error
            logger.warning("Simulation %s failed: %s", job_id, job.error)
            transition(job, "failed", state=state, error=job.error, phase="execution")
        else:
            logger.info(
                "Simulation %s completed: raw=%s, log=%s",
                job_id,
                job.raw_file or "(log-only)",
                job.log_file,
            )
            transition(job, "completed", state=state, raw_size_bytes=outcome.raw_size)

    async def kill(self, job_id: str) -> None:
        """Kill the spice process for a job without touching job status.

        Used by both cancel() and the tool-layer timeout path — the
        latter wants to record status='timeout' rather than 'cancelled',
        so it manages job state itself and only delegates the SIGKILL.

        Both kill mechanisms are scoped by the job_id token in the simulator's
        command line (see ``_terminate_processes``), so a parallel server
        session's simulators are never touched.

        It deliberately does NOT release the concurrency slot. Termination here
        is best-effort (either kill can fail/return 0; exceptions are
        swallowed), so freeing the slot now would let a queued job
        launch alongside a still-running orphan, violating ``max_parallel`` in the
        exact failure mode the cap exists for. The slot is released only when the
        process is confirmed gone — i.e. when the completion callback fires
        ``_handle_completion`` (spicelib invokes it with ``callback_on_error=True``
        whenever the worker's subprocess returns, including after a kill or the
        spicelib-level timeout). A successful kill makes that fire almost
        immediately; a failed kill keeps the slot reserved until the orphan
        actually ends.
        """
        await asyncio.to_thread(self._terminate_processes, job_id)

    def _remove_run_artifacts(self, job_id: str) -> None:
        """Best-effort removal of a killed job's heavy on-disk artifacts.

        The run netlist, .raw, .log and .exe.log all share the ``{job_id}``
        stem in the output folder (run_filename is ``{job_id}{ext}``), and the
        job_id is unique, so a glob on that stem reaches exactly this run's
        files and nothing else. The logs are kept: they are small and they are
        the post-mortem for a timed-out/cancelled run — the timeout response
        points ``job.log_file`` at one. Errors are swallowed — a still-locked
        or already-gone file must not break completion handling.
        """
        try:
            stale = list(self.output_folder.glob(f"{job_id}.*"))
        except OSError as e:
            logger.debug("Could not list artifacts for %s: %s", job_id, e)
            return
        for path in stale:
            # Keep the post-mortem logs: {id}.log, {id}.exe.log, and the
            # {id}.fail spicelib renames the log to on a nonzero (killed) exit.
            if path.suffix in (".log", ".fail"):
                continue
            try:
                path.unlink()
            except OSError as e:
                logger.debug("Could not remove stale artifact %s: %s", path, e)

    def _terminate_processes(self, job_id: str) -> None:
        """Blocking process termination (runs in a worker thread).

        WSL + LTspice: the simulator is a Windows process invisible to the
        Linux psutil table — taskkill it by the job_id token in its command
        line. Everything else (native LTspice/Wine, ngspice/qspice/xyce on
        any OS, including ngspice on WSL where it IS a Linux process): psutil
        kill scoped by the same token. Both matches require the job_id, so a
        parallel server session's simulators can never be collateral.
        (spicelib's name-global ``kill_all_spice`` is deliberately not used.)
        """
        try:
            killed = kill_windows_ltspice_by_token(job_id)
            if killed:
                logger.info("Killed %d Windows sim process(es) for %s", killed, job_id)
        except Exception as e:
            logger.warning("WSL process kill for %s failed: %s", job_id, e)
        try:
            killed = kill_simulator_by_token(
                job_id, simulator_executable_names(self.simulator_class)
            )
            if killed:
                logger.info("Killed %d local sim process(es) for %s", killed, job_id)
        except Exception as e:
            logger.warning("Scoped process kill for %s failed: %s", job_id, e)

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
