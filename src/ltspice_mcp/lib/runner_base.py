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
from typing import TYPE_CHECKING, Any, NamedTuple

from spicelib.sim.sim_runner import SimRunner

from ltspice_mcp.lib.encoding import read_spice_text
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES, BatchJob
from ltspice_mcp.lib.log_parser import (
    classify_failure_code,
    extract_error_context,
    extract_log_diagnostics,
    is_op_stepping_failure,
    op_ladder_exhausted,
)
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.spice_validator import ANALYSIS_KINDS
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token

if TYPE_CHECKING:
    from ltspice_mcp.state import SessionState

# Trailing `_<digits>` in spicelib-generated raw/log filenames. spicelib's
# SimRunner._run_file_name produces "<stem>_<runno><suffix>" (1-based runno).
_RUNNO_RE = re.compile(r"_(\d+)$")

# Marker embedded in the name of a generated '.options logopinfo' netlist copy
# (built by inject_logopinfo below). All cleanup keys on it, so the
# producer and every consumer share one constant — change the name scheme here
# and both sides move together.
LOGOPINFO_MARKER = ".logopinfo"

# Same idea for a generated ngspice ".control" write-injection copy (built by
# inject_ngspice_control_write below).
NGSPICE_CONTROL_WRITE_MARKER = ".ctrlwrite"

_GENERATED_NETLIST_MARKERS = (LOGOPINFO_MARKER, NGSPICE_CONTROL_WRITE_MARKER)

# Fallback concurrency cap used when a caller doesn't pass ``max_parallel``.
# The real cap comes from ``config.max_parallel_sims``; this default only
# applies to direct runner construction (mostly tests). Every runner
# constructor and the RunnerManager factory methods share this one value.
DEFAULT_MAX_PARALLEL = 4


class RunOutcome(NamedTuple):
    """Filesystem-derived facts about a finished run, collected off the loop."""

    raw_file: str
    log_file: str
    raw_size: int
    error: str | None
    observations: tuple[dict, ...] = ()
    failure_code: str | None = None
    failure_evidence: dict[str, Any] | None = None


_RAW_PRODUCING_ANALYSES: frozenset[str] = frozenset(f".{kind}" for kind in ANALYSIS_KINDS)
_INCLUDE_DIRECTIVES: frozenset[str] = frozenset({".include", ".inc", ".lib"})
_MAX_INCLUDE_DEPTH = 3


def _include_target(rest: str) -> str | None:
    """Return the file token from an include or library directive."""
    rest = rest.strip()
    if not rest:
        return None
    if rest[0] in "\"'":
        end = rest.find(rest[0], 1)
        return rest[1:end] if end != -1 else None
    return rest.split(None, 1)[0]


def deck_requests_raw(netlist: Path | None) -> tuple[list[str], bool]:
    """Snapshot a deck's raw-producing analyses and ``.save`` presence."""
    if netlist is None:
        return [], False
    analyses: list[str] = []
    has_save = False
    has_control = False
    seen: set[Path] = set()

    def scan(path: Path, depth: int) -> None:
        nonlocal has_save, has_control
        if depth > _MAX_INCLUDE_DEPTH:
            return
        try:
            key = path.resolve()
        except OSError:
            return
        if key in seen:
            return
        seen.add(key)
        try:
            text = read_spice_text(path)
        except OSError:
            return
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("."):
                continue
            parts = stripped.split(None, 1)
            head = parts[0].lower()
            if head == ".end":
                break
            if head == ".control":
                has_control = True
            elif head in _RAW_PRODUCING_ANALYSES:
                if head not in analyses:
                    analyses.append(head)
            elif head == ".save":
                has_save = True
            elif head in _INCLUDE_DIRECTIVES and len(parts) > 1:
                target = _include_target(parts[1])
                if target is not None:
                    scan(path.parent / target, depth + 1)

    scan(netlist, 0)
    if has_control:
        return [], has_save
    return analyses, has_save


def _missing_required_raw_outcome(
    log_file: str,
    log_path: Path,
    analyses: list[str],
    has_save: bool,
) -> RunOutcome:
    """Build the failure facts for an expected but absent raw artifact."""
    analysis_str = "/".join(analyses)
    excerpt = extract_error_context(log_path, max_lines=20)
    if has_save:
        workaround = (
            " The deck sets a '.save' list; if it omits nodes the analysis "
            "probes, LTspice 26.0.2 has been observed to exit 0 without writing "
            "a .raw. List every probed node in the .save (or remove the .save "
            "directive) and re-run — a full .save list is the known workaround."
        )
    else:
        workaround = (
            " The simulator reported no error, so a re-run may succeed; if it "
            "recurs, check the analysis directive and any .save list."
        )
    excerpt_block = f"\n\nLog excerpt:\n{excerpt}" if excerpt else ""
    error = (
        "Simulation exited cleanly but produced no .raw waveform file, which the "
        f"deck's {analysis_str} analysis requires — the waveform results are "
        f"absent.{workaround}{excerpt_block}"
    )
    observation = {
        "code": "missing_required_raw",
        "kind": "reconciliation",
        "detail": (
            f"The deck requested a {analysis_str} analysis but the simulator "
            "exited without writing a .raw file; waveform results are absent."
        ),
        "evidence": {
            "expected_artifact": "raw",
            "analyses": analyses,
            "has_save_list": has_save,
        },
    }
    return RunOutcome("", log_file, 0, error, observations=(observation,))


def collect_run_outcome(
    raw_file: str,
    log_file: str,
    requirements: tuple[list[str], bool] | None = None,
    exit_code: int | None = None,
) -> RunOutcome:
    """Collect and classify completion artifacts on a worker thread.

    ``exit_code`` is the simulator process's own exit status, relayed as a
    fact when the run failed — the one signal that separates a process killed
    from outside from a deck the simulator declined.
    """
    log_path = Path(log_file)
    sim_failed = raw_file in ("", ".") or log_path.suffix == ".fail"
    raw_size = 0
    if not sim_failed:
        try:
            raw_size = Path(raw_file).stat().st_size
        except FileNotFoundError:
            raw_size = 0
        except OSError as exc:
            return RunOutcome(
                raw_file,
                log_file,
                0,
                f"Simulation finished but its raw file is unreadable: {exc}",
            )
    if raw_size > 0:
        return RunOutcome(raw_file, log_file, raw_size, None)

    try:
        log_exists = bool(log_file) and log_path.exists()
    except OSError:
        log_exists = False
    errors = extract_log_diagnostics(log_path)["errors"] if log_exists else []
    # The clean-exit branch below asserts "exited cleanly" from log-error
    # ABSENCE, so a killed process with a quiet log would be reported as a
    # clean exit — gate it on the one direct fact about exit status.
    if exit_code in (None, 0) and not sim_failed and log_exists:
        non_rung = [error for error in errors if not is_op_stepping_failure(error)]
        if not non_rung and not op_ladder_exhausted(errors):
            analyses, has_save = requirements if requirements is not None else ([], False)
            if not analyses:
                return RunOutcome("", log_file, 0, None)
            return _missing_required_raw_outcome(log_file, log_path, analyses, has_save)

    if log_exists:
        context = extract_error_context(log_path, max_lines=20)
        error = f"Simulation failed (no output generated)\n\nLog excerpt:\n{context}"
    else:
        error = "Simulation failed (no output generated, log file missing)"
    # The diagnostics above already name the cause; classifying here is what
    # turns it into a code a caller can branch on instead of prose it must read.
    code, evidence = classify_failure_code(errors)
    if exit_code not in (None, 0):
        error += f"\nSimulator exit code: {exit_code}"
        # Copy rather than mutate: classify_failure_code's return is typed
        # narrower than the relayed shape, and this is the cold path.
        evidence = {**(evidence or {}), "exit_code": exit_code}
    return RunOutcome(
        "" if sim_failed else raw_file,
        log_file,
        0,
        error,
        failure_code=code,
        failure_evidence=evidence,
    )


def inject_logopinfo(netlist_path: Path, simulator: type, job_id: str) -> Path:
    """Return a runnable netlist with ``.options logopinfo`` added, for LTspice ``.op`` runs.

    LTspice writes each semiconductor's small-signal operating point (gm, gds,
    vth, vdsat, junction caps) to the ``.log`` only under ``.options logopinfo``,
    and only for ``.op`` analyses — so adding it lets ``operating_point`` read
    those params back by name. ngspice uses ``@dev[param]`` raw traces instead
    and needs nothing here.

    Append-only into a per-job sibling file (a leading-dot, ``job_id``-stamped
    name) so the simulator sees the caller's deck byte-for-byte plus the one
    directive; the original is never touched and relative ``.include``/``.lib``
    paths still resolve from the same directory. That untouched original is what
    lets the experiments path inject at submit time without moving the staged
    deck's recorded digest. The ``job_id`` stamp keeps two concurrent or queued
    runs of the same netlist from clobbering each other's augmented copy; the
    submitting caller deletes it once spicelib has staged the run. Returns the
    original path unchanged when injection doesn't apply (non-LTspice, non-text
    netlist, no ``.op``, or ``logopinfo`` already present) or the sibling can't
    be written.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    if not (isinstance(simulator, type) and issubclass(simulator, LTspice)):
        return netlist_path
    if netlist_path.suffix.lower() not in (".cir", ".net", ".sp"):
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    # Detect on the raw bytes (the directives are ASCII) — same plane the .end
    # splice below works on, so no decode round-trip is needed.
    if b"logopinfo" in data.lower():
        return netlist_path
    # ``.op\b`` excludes ``.options`` (the 't' blocks the word boundary); only a
    # real .op analysis emits the operating-point block. ``.dc`` does not.
    if not re.search(rb"(?im)^[ \t]*\.op\b", data):
        return netlist_path

    # Byte-level insertion before the final ``.end`` keeps the original encoding
    # intact (the added line is pure ASCII). ``.end\b`` skips ``.ends``.
    line = b".options logopinfo\n"
    ends = list(re.finditer(rb"(?im)^[ \t]*\.end\b.*$", data))
    if ends:
        at = ends[-1].start()
        augmented = data[:at] + line + data[at:]
    else:
        augmented = data + (b"" if not data or data.endswith(b"\n") else b"\n") + line

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{LOGOPINFO_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


# A ``.control``...``.endc`` block, case-insensitive. Group 1 is the body —
# everything between the ``.control`` line and the ``.endc`` line — so
# ``match.end(1)`` is exactly where the ``.endc`` line begins (the fallback
# insertion point when the block has no ``quit``/``exit``).
_RE_CONTROL_BLOCK = re.compile(rb"(?ims)^[ \t]*\.control\b[^\n]*\n(.*?)^[ \t]*\.endc\b[^\n]*$")
# A ``write``/``wrdata`` command starting a line, anywhere in the deck — not
# just inside the block, since a script could call either from a subckt or a
# second block this pass doesn't otherwise recognize.
_RE_EXISTING_WRITE = re.compile(rb"(?im)^[ \t]*(?:write|wrdata)\b")
# ``quit``/``exit`` end control-script execution; a command placed after one
# would never run, so the injected ``write`` must land before the LAST one.
_RE_QUIT_EXIT = re.compile(rb"(?im)^[ \t]*(?:quit|exit)\b.*$")
# A tail (from just after a quit/exit line to the block's .endc) that is only
# blank lines and ``*`` comments — i.e. the quit/exit was the block's LAST
# statement. Used to tell a script-ending trailing quit from one nested in an
# if/while (which must NOT anchor the injected write, or it lands inside that
# conditional and never runs on the success path).
_RE_TRIVIAL_TAIL = re.compile(rb"(?m)\A(?:[ \t]*(?:\*.*)?(?:\n|\Z))*\Z")


def inject_ngspice_control_write(
    netlist_path: Path, simulator: type, job_id: str, output_folder: Path
) -> Path:
    """Return a runnable netlist with a ``write`` injected into its
    ``.control`` block, for ngspice decks that drive their own analyses via
    scripting.

    This is ngspice runtime behavior, not a spicelib bug: a ``.control``
    block replaces the raw ngspice would otherwise write from the ``-r
    <rawfile>`` switch spicelib always passes — the script runs instead, and
    unless it calls ``write``/``wrdata`` itself, no raw file is ever
    produced. ``collect_run_outcome`` already classifies that as a clean
    log-only completion (not a failure), but nothing then exists for
    get_waveform/signal_stats/etc. to read. Injecting a canonical ``write
    <rawpath>`` gives the deck a raw at the exact path the runner expects for
    this job, so the existing raw>0 code path (raw_parser + every analysis
    tool) picks it up unchanged — no new parser, no new tool.

    Limitation: a bare ``write`` captures ngspice's current/last plot only. A
    script that runs multiple analyses, or writes per Monte-Carlo iteration
    inside a loop, needs its own explicit writes to capture each one — guard
    (c) below leaves any deck that already writes its own output alone
    rather than duplicating or fighting it. A second, unrelated limitation:
    ngspice's ``write`` parser cannot handle a target containing whitespace
    at all — neither quoting nor backslash-escaping works, both fail with
    "No such file or directory" (verified empirically). So a run whose
    output folder path contains a space can't get an auto-injected write
    either (guard (d)) — that run just stays log-only, same as today.

    Guards (all required, or the original path is returned unchanged):
    (a) ngspice only (LTspice has no ``.control``; ``inject_logopinfo``
        covers its own op-point injection separately).
    (b) exactly one ``.control``...``.endc`` block (ambiguous otherwise —
        e.g. which block's last analysis is "the" result).
    (c) no existing ``write``/``wrdata`` anywhere in the deck — never
        override a user who already captures their own output.
    (d) the write target has no whitespace (see the limitation above).

    The ``write`` target is the ABSOLUTE path ``{output_folder}/{job_id}.raw``
    — the same path spicelib's own (suppressed) ``-r`` would use, since it
    derives the rawfile from the staged netlist's own path via
    ``.with_suffix('.raw')``. It must be absolute: the runner's SimRunner
    passes no ``cwd``, so ngspice inherits the MCP server's own working
    directory, not the output folder — a relative ``write`` target would land
    there instead. Written UNQUOTED — see the whitespace limitation above.
    Inserted before the block's LAST ``quit``/``exit`` (if any) so it
    actually runs — those commands end script execution, so a ``write``
    placed after one would never fire; otherwise inserted just before
    ``.endc``.

    Same per-job sibling-file technique as ``inject_logopinfo`` (see its
    docstring): append-only into a leading-dot, ``job_id``-stamped copy so
    the user's deck is never touched and relative ``.include``/``.lib``
    paths still resolve. Returns the original path when injection doesn't
    apply or the sibling can't be written.

    Scope: one run at a time — a single job, or one experiment case, each of
    which has a static raw path derived from its own token. A sweep/Monte-Carlo
    batch's per-sub-run raw naming isn't static that way, so this is not wired
    into those batch paths.
    """
    from spicelib.simulators.ngspice_simulator import NGspiceSimulator

    if not (isinstance(simulator, type) and issubclass(simulator, NGspiceSimulator)):
        return netlist_path
    if netlist_path.suffix.lower() not in (".cir", ".net", ".sp"):
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    if _RE_EXISTING_WRITE.search(data):
        return netlist_path
    blocks = list(_RE_CONTROL_BLOCK.finditer(data))
    if len(blocks) != 1:
        return netlist_path
    block = blocks[0]
    body_start, body_end = block.start(1), block.end(1)

    raw_path = (output_folder / f"{job_id}.raw").as_posix()
    # ngspice's `write` parser cannot handle a spaced target at all — not
    # quoted, not escaped (verified empirically) — so a spaced output folder
    # can't get an auto-injected write; that run just stays log-only.
    if any(c.isspace() for c in raw_path):
        return netlist_path
    write_line = f"write {raw_path}\n".encode()

    # Insert before .endc, UNLESS the block's last statement is an
    # unconditional trailing quit/exit — a write after that would never run. A
    # quit/exit nested in an if/while is not the last statement (an ``end`` and
    # possibly more follow it), so anchoring on it is skipped: the write goes
    # before .endc and runs on the normal path.
    insert_at = body_end
    quit_matches = list(_RE_QUIT_EXIT.finditer(data, body_start, body_end))
    if quit_matches and _RE_TRIVIAL_TAIL.match(data[quit_matches[-1].end() : body_end]):
        insert_at = quit_matches[-1].start()
    augmented = data[:insert_at] + write_line + data[insert_at:]

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{NGSPICE_CONTROL_WRITE_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


def discard_generated_netlist(path: Path | None) -> None:
    """Delete a generated per-job netlist copy (an ``.options logopinfo``
    injection or an ngspice ``.control`` write injection). No-op when ``path``
    is None or carries neither marker, so this can only ever remove a
    generated copy, never the user's own deck."""
    if path is not None and any(marker in path.name for marker in _GENERATED_NETLIST_MARKERS):
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


def _failed_run_entry(log_file: str = "") -> dict:
    """A ``run_results`` entry marking a batch sub-run failed (no raw produced)."""
    return {"raw_file": "", "log_file": log_file, "params": {}, "failed": True}


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
        # Fire the completion callback even for a failed sub-run, so a run that
        # aborts (or produces no raw) is recorded as failed instead of vanishing
        # from the batch. Without this a dropped run leaves completed_runs+
        # failed_runs < total_runs while the batch still reports "completed".
        kwargs.setdefault("callback_on_error", True)
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
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        self.loop = loop
        self.simulator_class = simulator_class
        self.output_folder = output_folder
        self.max_parallel = max_parallel
        # Submitted SimRunners, held until their simulation thread is done.
        # See _retire_finished_runners for why letting one go early is a trap.
        self._inflight_runners: dict[str, SimRunner] = {}

    def _build_sim_runner(self) -> SimRunner:
        """Construct a spicelib SimRunner with this runner's settings."""
        return SimRunner(
            simulator=self.simulator_class,
            output_folder=str(self.output_folder),
            parallel_sims=self.max_parallel,
            timeout=_SIMRUNNER_TIMEOUT,
        )

    def _kill_by_token(self, token: str, context_label: str = "") -> None:
        """Best-effort blocking termination scoped to a command-line token."""
        subject = f"{context_label} {token}".strip()
        try:
            killed = kill_windows_ltspice_by_token(token)
            if killed:
                logger.info("Killed %d Windows sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("WSL process kill for %s failed: %s", subject, exc)
        try:
            killed = kill_simulator_by_token(
                token,
                simulator_executable_names(self.simulator_class),
            )
            if killed:
                logger.info("Killed %d local sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("Scoped process kill for %s failed: %s", subject, exc)

    def submit_netlist(
        self,
        netlist: Path,
        run_filename: str,
        callback: Callable[[Any], Any],
    ) -> SimRunner:
        """Submit one deck and bridge its filesystem-derived outcome to the loop.

        This is the job-agnostic single-run primitive shared by legacy
        ``SimulationRunner`` jobs and experiment cases. It knows only the deck,
        the simulator-facing filename, and an event-loop callback; registration,
        lifecycle, persistence, and concurrency remain with its callers.

        Call from a worker thread. The requirements snapshot and completion
        artifact reads intentionally happen on spicelib's worker threads.

        The SimRunner is returned AND retained here. spicelib's
        ``SimRunner.__del__`` calls ``wait_completion()``, so a caller that
        discards the return value has its thread pinned inside the destructor
        for the entire simulation — the calling coroutine never resumes, so it
        never reaches the code that watches for a cancellation, and the run
        becomes unstoppable. Retaining at this choke point means no caller can
        re-arm that by forgetting to keep it.
        """
        requirements = deck_requests_raw(netlist)

        def completion_callback(raw_file: Path | None, log_file: Path | None) -> None:
            # This runner is fresh per submission, so active_tasks holds
            # exactly this run's task — appended before its thread starts,
            # so it is present whenever the callback can fire.
            try:
                outcome = collect_run_outcome(
                    str(raw_file) if raw_file else "",
                    str(log_file) if log_file else "",
                    requirements,
                    # spicelib invokes the callback from the RunTask's own
                    # thread, and the task IS a Thread subclass carrying its
                    # retcode — so the current thread is the exact task,
                    # race-free. Any other calling thread reads None.
                    exit_code=getattr(threading.current_thread(), "retcode", None),
                )
            except Exception as exc:
                outcome = RunOutcome(
                    "",
                    "",
                    0,
                    f"Simulation failed (outcome collection: {exc})",
                )
            self._bridge(callback, outcome, context=f"run {run_filename}")

        self._retire_finished_runners()
        runner = self._build_sim_runner()
        runner.run(
            str(netlist),
            run_filename=run_filename,
            callback=completion_callback,
            callback_on_error=True,
            exe_log=True,
        )
        self._inflight_runners[run_filename] = runner
        return runner

    def _retire_finished_runners(self) -> None:
        """Release SimRunners whose simulation threads have all exited.

        Pruned on the way into the next submission rather than from a
        completion callback: dropping the last reference runs
        ``SimRunner.__del__`` -> ``wait_completion()``, which waits on
        ``active_tasks`` — from inside a task's own callback that would be the
        task waiting for itself. Liveness is read off the RunTask threads
        instead of spicelib's bookkeeping, which only updates when something
        calls into it.
        """
        for key, runner in list(self._inflight_runners.items()):
            if not any(task.is_alive() for task in runner.active_tasks):
                self._inflight_runners.pop(key, None)

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
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        self._cancel_events: dict[str, threading.Event] = {}

    def _cleanup(self, job_id: str) -> None:
        """Drop the per-job cancel-event reference."""
        self._cancel_events.pop(job_id, None)

    def owns_batch_job(self, job_id: str) -> bool:
        """Whether this instance launched (and can therefore cancel) ``job_id``."""
        return job_id in self._cancel_events

    def has_active_work(self) -> bool:
        """Whether any batch launched by this instance is still in flight."""
        return bool(self._cancel_events)

    def _register_cancel(self, job_id: str) -> threading.Event:
        """Register a cancel event for a batch job and return it."""
        ev = threading.Event()
        self._cancel_events[job_id] = ev
        return ev

    def _gated_runner_for(self, job_id: str, cancel_event: threading.Event) -> SimRunner:
        """Build, wrap, and cancel-gate this batch's spicelib runner.

        The gate stops the submission loop (spicelib's ``run_all`` for
        sweeps, the per-run loop for Monte Carlo) from launching the
        remaining queued runs after ``cancel()`` kills the in-flight ones —
        the kill frees simulator slots, which would otherwise resume
        submission of a job the user just cancelled.
        """
        return gate_runner_on_cancel(
            wrap_runner_for_runno_callbacks(self._build_sim_runner()), cancel_event, job_id
        )

    def _record_run_completion(
        self,
        batch_job: BatchJob,
        raw_file: Path | None,
        log_file: Path | None,
        state: SessionState,
        kind: str,
        runno: int | None = None,
    ) -> None:
        """Record one finished sub-run (successful or failed) in ``run_results``.

        ``run_results`` is keyed by 0-based runno. The runno is passed
        explicitly when available (callers using
        ``wrap_runner_for_runno_callbacks``) — that's the canonical path
        and what makes parallel execution correctly labeled. As a
        fallback for callbacks that don't have it, the runno is parsed
        from the raw_file basename (spicelib names files
        ``<stem>_<runno><suffix>``); failing that, completion order is
        used (a third-best signal for environments where neither
        mechanism applies).

        A run whose simulation aborted arrives here with ``raw_file`` None
        (or pointing at a raw that was never written); it is recorded as a
        failed entry so the batch counts stay honest — ``completed_runs``
        advances for every finished run, ``failed_runs`` for the failed
        subset, and ``successful`` (``completed_runs - failed_runs``) is the
        real success count. A failed entry has an empty ``raw_file`` so the
        aggregation readers skip it cleanly.

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

        failed = raw_file is None or not raw_file.exists()
        if runno is None and raw_file is not None:
            runno = _parse_runno(raw_file)
        # 0-based key preserves the existing "first run = key 0" convention.
        run_index = (runno - 1) if runno is not None else batch_job.completed_runs
        if failed:
            batch_job.run_results[run_index] = _failed_run_entry(
                str(log_file) if log_file is not None else ""
            )
            batch_job.failed_runs += 1
        else:
            batch_job.run_results[run_index] = {
                "raw_file": str(raw_file),
                "log_file": str(log_file) if log_file is not None else "",
                "params": {},
            }
        batch_job.completed_runs += 1
        state.persist_batch_progress(batch_job)

        logger.debug(
            "%s job %s: run %d %s (%d/%d, %d failed)",
            kind,
            batch_job.job_id,
            run_index,
            "failed" if failed else "complete",
            batch_job.completed_runs,
            batch_job.total_runs,
            batch_job.failed_runs,
        )

    def _finalize_batch(self, batch_job: BatchJob, kind: str) -> None:
        """Reconcile any sub-run that never reported a completion at all.

        ``callback_on_error`` makes a failed run report itself, but spicelib
        can still omit a run entirely (a submission the stepper never launched,
        a run dropped before its task existed). Any 0-based index missing from
        ``run_results`` after the submission loop returned produced no result:
        record it as failed so ``completed_runs == total_runs`` holds and a
        terminal ``completed`` can never mask a silent shortfall. Call this on
        the non-cancelled completion path, before transitioning to completed.
        """
        missing = [i for i in range(batch_job.total_runs) if i not in batch_job.run_results]
        for i in missing:
            batch_job.run_results[i] = _failed_run_entry()
        if missing:
            batch_job.failed_runs += len(missing)
            batch_job.completed_runs = len(batch_job.run_results)
            logger.warning(
                "%s job %s: %d run(s) produced no result and were recorded as failed (indices %s)",
                kind,
                batch_job.job_id,
                len(missing),
                missing,
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

        # Kill the batch's simulator processes by job-id token. Batch sub-runs
        # are named with the job id plus a per-run index (see
        # batch_run_filename), so the substring token match hits every run of
        # this batch and nothing else — in particular never a parallel server
        # session's simulators. Two mechanisms, at most one matches per
        # platform: the WSL taskkill for Windows-side LTspice (invisible to
        # Linux psutil) and the psutil kill for everything else (native
        # LTspice/Wine, ngspice/qspice/xyce — including ngspice on WSL).
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
        # immediately.
        try:
            exe_names = simulator_executable_names(self.simulator_class)
            killed_total = 0
            for attempt in range(_CANCEL_KILL_MAX_PASSES):
                killed = await asyncio.to_thread(kill_windows_ltspice_by_token, batch_job.job_id)
                killed += await asyncio.to_thread(
                    kill_simulator_by_token, batch_job.job_id, exe_names
                )
                killed_total += killed
                if killed == 0 and attempt != 1:
                    break
                await asyncio.sleep(_CANCEL_KILL_RESCAN_DELAY)
            if killed_total:
                logger.info(
                    "Killed %d Windows %s process(es) for %s", killed_total, kind, batch_job.job_id
                )
        except Exception as e:
            logger.warning("Process kill for %s job %s failed: %s", kind, batch_job.job_id, e)

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
