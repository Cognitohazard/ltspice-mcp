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
import contextlib
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES, BatchJob
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token
from ltspice_mcp.state import SessionState

# Trailing `_<digits>` in spicelib-generated raw/log filenames. spicelib's
# SimRunner._run_file_name produces "<stem>_<runno><suffix>" (1-based runno).
_RUNNO_RE = re.compile(r"_(\d+)$")

# Marker embedded in the name of a generated '.options logopinfo' netlist copy
# (built by tools._base.inject_logopinfo). All cleanup keys on it, so the
# producer and every consumer share one constant — change the name scheme here
# and both sides move together.
LOGOPINFO_MARKER = ".logopinfo"


def discard_logopinfo_netlist(path: Path | None) -> None:
    """Delete a generated '.options logopinfo' netlist copy. No-op when ``path``
    is None or lacks the marker, so this can only ever remove the generated copy,
    never the user's own deck."""
    if path is not None and LOGOPINFO_MARKER in path.name:
        with contextlib.suppress(OSError):
            path.unlink()


def _parse_runno(raw_file: Path) -> int | None:
    """Extract spicelib's 1-based runno from a raw/log filename, or None.

    Returns None for files whose basename doesn't match the spicelib pattern
    (e.g., one-shot sims via `run_simulation`, which use job_id stems).
    Used as a fallback when the runno can't be captured at submission via
    ``wrap_runner_for_runno_callbacks``.
    """
    match = _RUNNO_RE.search(raw_file.stem)
    if not match:
        return None
    return int(match.group(1))


def batch_run_filename(job_id: str, runno: int, netlist: Path) -> str:
    """Per-run filename for a batch sub-run: ``"{job_id}_{runno}{ext}"``.

    The ``job_id`` prefix is the token ``BatchRunnerBase.cancel`` taskkills by on
    WSL (substring match), and the trailing ``_{runno}`` is what ``_parse_runno``
    reads back — this is the single producer of that naming contract, so a new
    batch runner can't silently break cancel/runno-parsing. ``ext`` falls back to
    ``.net`` (LTspice/ngspice reject extensionless netlists).
    """
    return f"{job_id}_{runno}{netlist.suffix or '.net'}"


class BatchCancelledError(Exception):
    """Raised inside a batch worker thread when its job has been cancelled.

    Aborts the submission loop (spicelib's ``run_all`` for sweeps, the
    per-run loop for Monte Carlo) so a cancelled batch stops launching its
    remaining queued runs. ``_mark_batch_failed`` recognizes it and leaves
    the job's status to the cancel path rather than marking it failed.
    """


def gate_runner_on_cancel(
    runner: SimRunner, cancel_event: threading.Event, job_id: str
) -> SimRunner:
    """Make ``runner.run`` refuse new submissions once ``cancel_event`` is set.

    Killing a batch's processes frees simulator slots, which lets the
    submission loop blocked inside ``runner.run`` resume and launch the
    *next* queued run of a job the user just cancelled. This gate aborts
    those later submissions at the entry point. (The one submission already
    inside ``runner.run`` when the event is set can still spawn — the
    re-scan loop in ``BatchRunnerBase.cancel`` catches that process.)
    """
    original_run = runner.run

    def cancel_gated_run(*args: Any, **kwargs: Any) -> Any:
        if cancel_event.is_set():
            raise BatchCancelledError(f"batch job {job_id} cancelled; not launching further runs")
        return original_run(*args, **kwargs)

    runner.run = cancel_gated_run  # type: ignore[method-assign]
    return runner


def wrap_runner_for_runno_callbacks(runner: SimRunner) -> SimRunner:
    """Make ``runner.run`` inject ``task.runno`` into the user's callback.

    spicelib's per-run callback is ``(raw_file, log_file)`` — no task
    ref, no runno.  We wrap the user's callback in a closure that reads
    ``runno`` from a mutable ref, pass the wrapper to ``original_run``
    so the callback is set BEFORE the thread starts, then fill in the
    ref from ``task.runno`` after ``run()`` returns.

    The previous approach (post-patching ``task.callback`` after
    ``task.start()``) races with fast simulators like ngspice (~50 ms)
    — the thread can finish and check ``self.callback`` (still ``None``)
    before the patch is applied.

    Idempotent via a sentinel attribute on the wrapper.
    """
    if getattr(runner.run, "_runno_aware", False):
        return runner

    original_run = runner.run

    def runno_aware_run(*args, **kwargs):
        user_callback = kwargs.pop("callback", None)
        user_callback_args = kwargs.pop("callback_args", None)
        if user_callback is None:
            return original_run(*args, callback=None, callback_args=None, **kwargs)

        # Predict the runno synchronously: spicelib increments _runno in
        # _prepare_sim (called inside run() before the thread starts), then
        # assigns RunTask(runno=self._runno). Reading _runno+1 here is safe
        # because run() hasn't been called yet. This avoids the race where
        # a fast simulator completes before original_run returns the task.
        predicted_runno = runner._runno + 1

        def runno_bound(raw_file: object, log_file: object) -> object:
            return user_callback(raw_file, log_file, runno=predicted_runno)

        task = original_run(
            *args, callback=runno_bound, callback_args=user_callback_args, **kwargs
        )
        return task

    runno_aware_run._runno_aware = True  # type: ignore[attr-defined]
    runner.run = runno_aware_run  # type: ignore[method-assign]
    return runner


logger = logging.getLogger(__name__)

_SIMRUNNER_TIMEOUT = 600
"""Generous spicelib-level fallback timeout; real timeout is enforced at
the tool layer via ``asyncio.wait_for``."""

_CANCEL_KILL_MAX_PASSES = 5
"""Upper bound on cancel's kill/re-scan passes (see ``BatchRunnerBase.cancel``)."""

_CANCEL_KILL_RESCAN_DELAY = 0.5
"""Seconds between cancel kill passes — long enough for a resumed submission's
process to become visible to the next scan."""


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

    def _bridge(self, handler: Callable[..., Any], *args: Any, context: str = "") -> bool:
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

    def _gated_runner_for(self, job_id: str, cancel_event: threading.Event) -> SimRunner:
        """Build, wrap, cancel-gate, and register this batch's spicelib runner.

        The gate stops the submission loop (spicelib's ``run_all`` for
        sweeps, the per-run loop for Monte Carlo) from launching the
        remaining queued runs after ``cancel()`` kills the in-flight ones —
        the kill frees simulator slots, which would otherwise resume
        submission of a job the user just cancelled.
        """
        runner = gate_runner_on_cancel(
            wrap_runner_for_runno_callbacks(self._build_sim_runner()), cancel_event, job_id
        )
        self._register_runner(job_id, runner)
        return runner

    def _record_run_completion(
        self,
        batch_job: BatchJob,
        raw_file: Path,
        log_file: Path,
        state: SessionState,
        kind: str,
        runno: int | None = None,
    ) -> None:
        """Append one successful sub-run to ``batch_job.run_results``.

        ``run_results`` is keyed by 0-based runno. The runno is passed
        explicitly when available (callers using
        ``wrap_runner_for_runno_callbacks``) — that's the canonical path
        and what makes parallel execution correctly labeled. As a
        fallback for callbacks that don't have it, the runno is parsed
        from the raw_file basename (spicelib names files
        ``<stem>_<runno><suffix>``); failing that, completion order is
        used (a third-best signal for environments where neither
        mechanism applies).

        Params are stored empty at this stage. Sweeps populate them from
        ``stepper.sim_info`` after run_all returns; Monte Carlo leaves
        them empty (deviations are statistical).
        """
        if batch_job.status in TERMINAL_STATUSES:
            logger.debug(
                "%s job %s already in terminal state '%s', ignoring run completion",
                kind,
                batch_job.job_id,
                batch_job.status,
            )
            return

        if runno is None:
            runno = _parse_runno(raw_file)
        # 0-based key preserves the existing "first run = key 0" convention.
        run_index = (runno - 1) if runno is not None else batch_job.completed_runs
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

    async def cancel(self, batch_job: BatchJob, state: SessionState | None = None) -> None:
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

        # WSL: taskkill the Windows simulator processes whose per-run filenames
        # carry this job's token. Batch sub-runs are named with the job id plus a
        # per-run index (see batch_run_filename), so the substring token match in
        # kill_windows_ltspice_by_token hits every run of this batch, the same way
        # the single-run cancel path terminates its process.
        #
        # One pass is not enough: killing the in-flight runs frees simulator
        # slots, which can resume a submission already blocked inside
        # ``runner.run`` — its process appears a beat *after* the first pass
        # (observed live: a Monte-Carlo child created the same second as the
        # cancel survived it and kept simulating). Break on a clean pass,
        # EXCEPT at attempt 1: a clean scan only ~0.5s after a kill can still
        # miss the resumed submission's process (it showed up ~1s in live),
        # so the window stays open through the attempt-2 scan. A clean FIRST
        # pass means no slot was freed by us — no resume race — and breaks
        # immediately; off WSL every pass is a cheap no-op returning 0, so
        # the loop exits there with no sleep.
        try:
            killed_total = 0
            for attempt in range(_CANCEL_KILL_MAX_PASSES):
                killed = await asyncio.to_thread(kill_windows_ltspice_by_token, batch_job.job_id)
                killed_total += killed
                if killed == 0 and attempt != 1:
                    break
                await asyncio.sleep(_CANCEL_KILL_RESCAN_DELAY)
            if killed_total:
                logger.info(
                    "Killed %d Windows %s process(es) for %s", killed_total, kind, batch_job.job_id
                )
        except Exception as e:
            logger.warning("WSL process kill for %s job %s failed: %s", kind, batch_job.job_id, e)

        # Native/Wine: spicelib's kill_all_spice (a harmless no-op on WSL).
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
        if isinstance(exc, BatchCancelledError):
            # The run-gate aborted the submission loop of a cancelled job.
            # cancel() owns the terminal transition (it may not have made it
            # yet — the gate can fire while cancel() is still mid-kill), so
            # don't mark the job failed over it.
            logger.debug("%s job %s submission loop aborted by cancel", kind, batch_job.job_id)
            return
        if batch_job.status != "running":
            logger.info(
                "%s job %s already terminal (%s); ignoring exception %s",
                kind,
                batch_job.job_id,
                batch_job.status,
                exc,
            )
            return
        logger.error("%s job %s failed: %s", kind, batch_job.job_id, exc, exc_info=True)
        batch_job.error = f"{kind} execution failed: {exc}"
        transition(
            batch_job,
            "failed",
            state=state,
            error=batch_job.error,
            completed_runs=batch_job.completed_runs,
            total_runs=batch_job.total_runs,
        )
