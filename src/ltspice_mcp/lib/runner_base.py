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
from ltspice_mcp.lib.log_parser import (
    classify_failure_code,
    extract_error_context,
    extract_log_diagnostics,
    is_op_stepping_failure,
    op_ladder_exhausted,
)
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.simulator import current_ngbehavior, is_ngspice
from ltspice_mcp.lib.spice_lex import SpiceLexError, cards_from_path, tokenize_body
from ltspice_mcp.lib.spice_validator import ANALYSIS_KINDS
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token

if TYPE_CHECKING:
    pass

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


def _deck_has_sectioned_lib(netlist: Path) -> bool:
    """True when the deck carries a ``.lib <file> <section>`` directive.

    The section-selecting form — the standard PDK corner idiom — as opposed to
    LTspice's section-less ``.lib <file>``. Lexed rather than string-matched so
    an inline comment or a quoted path with spaces cannot inflate the token
    count into a section that is not there.
    """
    try:
        cards = cards_from_path(netlist).cards
    except (OSError, ValueError, SpiceLexError):
        return False
    for card in cards:
        if card.kind != "directive":
            continue
        try:
            tokens = tokenize_body(card.body)
        except SpiceLexError:
            continue
        if len(tokens) >= 3 and tokens[0].text.lower() == ".lib":
            return True
    return False


def _is_ngspice_lib_section_failure(netlist: Path | None, simulator: type | None) -> str | None:
    """The active ``ngbehavior`` when it is what broke a sectioned ``.lib``.

    ngspice's LTspice/PSPICE compatibility modes read ``.lib <file> <section>``
    as two plain includes and drop the section, so the corner select comes back
    as a missing include file. Fires only when the run used ngspice, the mode
    still contains ``lt`` or ``ps`` (a mode with neither parses the section
    correctly), and the deck really does select a section. The deck read is
    last, so the two cheap checks gate it.
    """
    if netlist is None or not is_ngspice(simulator):
        return None
    mode = (current_ngbehavior() or "").lower()
    if "lt" not in mode and "ps" not in mode:
        return None
    if not _deck_has_sectioned_lib(netlist):
        return None
    return mode


def collect_run_outcome(
    raw_file: str,
    log_file: str,
    requirements: tuple[list[str], bool] | None = None,
    exit_code: int | None = None,
    *,
    netlist: Path | None = None,
    simulator: type | None = None,
) -> RunOutcome:
    """Collect and classify completion artifacts on a worker thread.

    ``exit_code`` is the simulator process's own exit status, relayed as a
    fact when the run failed — the one signal that separates a process killed
    from outside from a deck the simulator declined.

    ``netlist`` and ``simulator`` are what the deck ran as, and are read only
    to tell one missing include from another: ngspice in a compatibility mode
    reports a sectioned ``.lib`` as a missing file, and that failure has a
    configuration fix the generic one does not.
    """
    log_path = Path(log_file)
    sim_failed = raw_file in ("", ".") or log_path.suffix == ".fail"
    raw_size = 0
    if not sim_failed:
        try:
            raw_size = Path(raw_file).stat().st_size
        except FileNotFoundError:
            # Windows maps "a path component is a file" to ERROR_PATH_NOT_FOUND
            # and raises FileNotFoundError, where POSIX raises NotADirectoryError
            # and lands in the branch below. Left undistinguished, the same
            # broken location reads as an unreadable raw on Linux and as a clean
            # run that produced nothing on Windows. Only the immediate parent is
            # checked; a file further up the chain still reads as a plain
            # absence, which is the same answer both platforms already give.
            parent = Path(raw_file).parent
            try:
                location_is_not_a_directory = parent.exists() and not parent.is_dir()
            except OSError:
                location_is_not_a_directory = False
            if location_is_not_a_directory:
                return RunOutcome(
                    raw_file,
                    log_file,
                    0,
                    "Simulation finished but its raw file is unreadable: "
                    f"{parent} is not a directory",
                )
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
    if code == "missing_include":
        mode = _is_ngspice_lib_section_failure(netlist, simulator)
        if mode is not None:
            code = "ngspice_lib_section"
            evidence = {**(evidence or {}), "ngbehavior": mode}
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


logger = logging.getLogger(__name__)

_SIMRUNNER_TIMEOUT = 600
"""Generous spicelib-level fallback timeout; real timeout is enforced at
the tool layer via ``asyncio.wait_for``."""

_CANCEL_KILL_MAX_PASSES = 5
"""Upper bound on cancel's kill/re-scan passes (see ``BatchRunnerBase.cancel``)."""

_CANCEL_KILL_RESCAN_DELAY = 0.5
"""Seconds between cancel kill passes — long enough for a resumed submission's
process to become visible to the next scan."""


class _NonBlockingSimRunner(SimRunner):
    """A SimRunner whose destructor cannot pin the thread that drops it.

    spicelib's ``__del__`` calls ``wait_completion(timeout=None)``, which loops
    ``while active_tasks: sleep(1)`` and takes its deadline only from tasks
    that have already started — a task that is not alive and never started
    yields no deadline and is never retired, so the loop has no exit at all.
    The last reference is dropped by ordinary garbage collection, so that wait
    lands on whatever thread happened to allocate: the event loop, or the
    thread running the test suite (a 12-minute silent CI hang, 2026-09-07).

    Nothing here needs the destructor. Completion reaches this module through
    the run callback, liveness is read off the task threads, and the only other
    thing it does on timeout is ``kill_all_spice()`` — the name-global kill
    this project deliberately never uses, because it would reach another
    session's simulator. See ``docs/spicelib_bugs.md`` Bug 10.
    """

    def __del__(self) -> None:
        return


class RunnerBase:
    """Shared constructor, launch capacity, and thread-safe callback bridging."""

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
        self._max_parallel = max_parallel
        # The launch permits every simulator process this runner starts must
        # hold for its whole life. It lives on the RUNNER, so concurrent jobs
        # share one cap instead of each getting a private one; a job may divide
        # its own share further, but nothing gets past this.
        self._launch_slots = asyncio.Semaphore(max_parallel)
        self._slots_out = 0
        # Submitted SimRunners, held until their simulation thread is done.
        # See _retire_finished_runners for why letting one go early is a trap.
        self._inflight_runners: dict[str, SimRunner] = {}

    @property
    def max_parallel(self) -> int:
        """Simulator processes this runner will have in flight at once."""
        return self._max_parallel

    @max_parallel.setter
    def max_parallel(self, value: int) -> None:
        self._max_parallel = value
        # A semaphore's permit count cannot be changed while permits are out
        # without losing track of them, so a new cap takes effect the next time
        # the runner is idle; runs already admitted keep the cap they started
        # under. With no permits out there can be no waiter either (a waiter
        # only exists once the permits are gone), so nothing is stranded here.
        if self._slots_out == 0:
            self._launch_slots = asyncio.Semaphore(value)

    async def acquire_launch_slot(self) -> None:
        """Take one launch permit. Call on the event loop, release when done."""
        await self._launch_slots.acquire()
        self._slots_out += 1

    def release_launch_slot(self) -> None:
        """Return a permit taken by ``acquire_launch_slot``."""
        self._launch_slots.release()
        self._slots_out -= 1

    def _build_sim_runner(self) -> SimRunner:
        """Construct a spicelib SimRunner with this runner's settings."""
        return _NonBlockingSimRunner(
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

        This is the job-agnostic single-run primitive every experiment case
        runs on. It knows only the deck, the simulator-facing filename, and an
        event-loop callback; registration, lifecycle, persistence, and
        concurrency remain with its callers.

        Call from a worker thread. The requirements snapshot and completion
        artifact reads intentionally happen on spicelib's worker threads.

        The SimRunner is returned AND retained here, because the kill and
        liveness paths need the handle for as long as the simulation runs.
        Retaining at this choke point means no caller can lose it by
        forgetting to keep the return value. (Dropping one is no longer
        dangerous in itself: ``_NonBlockingSimRunner`` removes the destructor
        that used to pin the dropping thread for the whole simulation.)
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
                    netlist=netlist,
                    simulator=self.simulator_class,
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

        Liveness is read off the RunTask threads rather than spicelib's own
        bookkeeping, which only updates when something calls into it — and
        which never retires a task that was appended but never started.
        Pruned on the way into the next submission rather than from a
        completion callback, so a task is never released from inside its own
        callback.
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
