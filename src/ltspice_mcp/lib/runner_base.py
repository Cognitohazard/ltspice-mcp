"""Simulator submission, outcome classification and process ownership.

The experiment coordinator submits through this job-independent primitive.
Blocking launch and completion work runs on worker threads; callbacks return
to the event loop. Retained simulator handles support scoped cancellation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

from spicelib.sim.sim_runner import SimRunner  # noqa: TID251 - this module owns every launch

from ltspice_mcp.lib import RUN_DECK_SUFFIXES
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.deck_staging import (
    closure_depth,
    resolve_reference,
    scan_include_references,
)
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.encoding import (
    bom_of,
    decode_spice_bytes_with_encoding,
    encode_spice_text_strictly,
    read_spice_text,
)
from ltspice_mcp.lib.log_parser import (
    classify_failure_code,
    is_op_stepping_failure,
    op_ladder_exhausted,
)
from ltspice_mcp.lib.ngspice_driver import control_path
from ltspice_mcp.lib.ngspice_mode import ModeCheck, check_run
from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY, NativeLaunchPolicy
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.raw_parser import other_raws
from ltspice_mcp.lib.simulator import NGSPICE_NO_USER_INIT, current_ngbehavior, is_ngspice
from ltspice_mcp.lib.simulator_build import executable_path, ltspice_before_24, reported_build
from ltspice_mcp.lib.spice_lex import (
    SpiceCard,
    SpiceLexError,
    control_blocks,
    emit,
    lex_deck,
    lex_for_edit,
    line_break,
    tokenize_body,
)
from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end
from ltspice_mcp.lib.spice_validator import ANALYSIS_KINDS
from ltspice_mcp.lib.wsl import kill_windows_ltspice_by_token

if TYPE_CHECKING:
    from spicelib.editor.base_editor import BaseEditor

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


@dataclass(frozen=True)
class NativeLaunchContext:
    """A prepared ngspice setup, with its separate electrical input."""

    input_deck: Path
    cwd: Path
    verify_execution: Callable[[], None] | None = None
    policy: NativeLaunchPolicy = LAUNCH_POLICY


def _launch_switches(simulator: type, native: NativeLaunchContext | None) -> list[str] | None:
    """The command-line switches a run starts with.

    A native setup's come from its recorded policy, which holds
    ``NGSPICE_NO_USER_INIT``. Every other ngspice run gets that switch here,
    so no user startup file replaces the ``-D ngbehavior=`` the server models
    the deck in. The recoverable adapter accepts only this switch and puts
    it on its own command line.
    """
    if native is not None:
        return list(native.policy.switches)
    return [NGSPICE_NO_USER_INIT] if is_ngspice(simulator) else None


class RunOutcome(NamedTuple):
    """Filesystem-derived facts about a finished run, collected off the loop."""

    raw_file: str
    log_file: str
    raw_size: int
    error: str | None
    observations: tuple[dict, ...] = ()
    failure_code: str | None = None
    failure_evidence: dict[str, Any] | None = None
    # The end of the log (or the lines around its errors) as a failed run left
    # it, apart from the prose ``error`` wraps it in. A run the coordinator
    # stopped keeps this, not the error: "no output generated" describes a
    # run that ended on its own.
    log_excerpt: str | None = None
    # The build the run named in its own output (``simulator_build``). Read
    # for every run, stopped ones included: which program ran is a fact about
    # the run whatever became of its results.
    simulator_version: str | None = None
    # The kinds of other raw the run wrote beside its raw (``other_raws``).
    other_raws: tuple[str, ...] = ()


_REQUESTED_ANALYSES: frozenset[str] = frozenset(f".{kind}" for kind in ANALYSIS_KINDS) | {
    ".tf",
    ".pz",
    ".sens",
    ".disto",
}
_MAX_INCLUDE_DEPTH = 3


@dataclass(frozen=True)
class DeckRequirements:
    """Requested analysis directives and output controls from one deck scan."""

    analyses: tuple[str, ...] = ()
    has_save: bool = False
    has_control: bool = False

    @property
    def raw_analyses(self) -> tuple[str, ...]:
        return () if self.has_control else self.analyses


#: What one file of a deck asks for, card by card: an analysis directive's head,
#: ``.save``, a ``.control`` block, or another file it pulls in (resolved).
_DeckItem = tuple[str, str | Path]


def _deck_items(path: Path, *, root: bool) -> tuple[_DeckItem, ...]:
    """What the file at ``path`` asks for, in card order: the deck itself when
    ``root`` (its first line its title), else a file it pulls in. Raises
    ``OSError`` when the file cannot be read."""
    cards = lex_deck(read_spice_text(path), title=root).cards
    try:
        references = {
            id(reference.card): reference
            for reference in scan_include_references(
                cards, path, depth=closure_depth(0 if root else 1)
            )
        }
    except SpiceLexError:
        references = {}  # a malformed directive: the simulator's to report
    items: list[_DeckItem] = []
    for card in cards:
        if card.kind == "control":
            items.append(("control", ""))
            continue
        parts = card.body.split(None, 1)
        if card.kind != "directive" or not parts:
            continue
        head = parts[0].lower()
        if head in _REQUESTED_ANALYSES:
            items.append(("analysis", head))
        elif head == ".save":
            items.append(("save", ""))
        elif (reference := references.get(id(card))) is not None:
            items.append(("include", resolve_reference(path.parent, reference.raw_path)))
    return tuple(items)


#: An included file's items, kept while the file is unchanged: every case of a
#: job pulls in the same libraries, which can run to tens of thousands of lines.
_INCLUDED_ITEMS: FileCache[tuple[_DeckItem, ...]] = FileCache(maxsize=256)


def deck_requirements(netlist: Path | None) -> DeckRequirements:
    """Snapshot requested analyses, ``.save`` and ``.control`` in one walk.

    It reads the cards the simulator reads (``spice_lex.lex_deck``): a
    directive continued on a ``+`` line is one card, and neither the deck's
    title nor a card after an ``.END`` counts. The files it follows are the
    ones staging reads the deck to pull in (``scan_include_references``), so
    a library's own ``.lib section`` line is a section, not a file.
    """
    if netlist is None:
        return DeckRequirements()
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
            items = (
                _deck_items(path, root=True)
                if depth == 0
                else _INCLUDED_ITEMS.get(key, lambda _key: _deck_items(path, root=False))
            )
        except OSError:
            return
        for kind, value in items:
            if kind == "control":
                has_control = True
            elif kind == "save":
                has_save = True
            elif kind == "analysis":
                if value not in analyses:
                    analyses.append(str(value))
            else:
                scan(Path(value), depth + 1)

    scan(netlist, 0)
    return DeckRequirements(tuple(analyses), has_save, has_control)


def _missing_required_raw_outcome(
    log_file: str,
    excerpt: str | None,
    analyses: tuple[str, ...],
    has_save: bool,
) -> RunOutcome:
    """Build the failure facts for an expected but absent raw artifact."""
    analysis_str = "/".join(analyses)
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
            "analyses": list(analyses),
            "has_save_list": has_save,
        },
    }
    return RunOutcome("", log_file, 0, error, observations=(observation,))


def _deck_has_sectioned_lib(netlist: Path) -> bool:
    """True when the deck carries a ``.lib <file> <section>`` directive.

    The section-selecting form — the standard PDK corner idiom — as opposed to
    LTspice's section-less ``.lib <file>``. Lexed rather than string-matched so
    an inline comment or a quoted path with spaces cannot inflate the token
    count into a section that is not there, and read as the simulator reads
    the deck (``spice_lex.lex_deck``), as the ``lib-section-ngspice`` lint
    rule reads it: neither its title nor a card after ``.END`` selects one.
    """
    try:
        cards = lex_deck(read_spice_text(netlist)).cards
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


def _is_ngspice_lib_section_failure(
    netlist: Path | None, simulator: type | None, ngbehavior: str | None = None
) -> str | None:
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
    mode = (ngbehavior if ngbehavior is not None else current_ngbehavior() or "").lower()
    if "lt" not in mode and "ps" not in mode:
        return None
    if not _deck_has_sectioned_lib(netlist):
        return None
    return mode


def _with_mode_check(outcome: RunOutcome, check: ModeCheck) -> RunOutcome:
    """Fail a run ngspice read in another compatibility mode than the server
    passed it, and say so on a run with results whose mode went unchecked.

    The server read the deck in the mode it passed (lint, hierarchy,
    variations, failure classification), so results read in another mode
    answer for a different circuit. A failed run is checked too, since its
    failure is classified in the passed mode; only one that left no log says
    nothing, as its failure already says the log is missing.
    """
    if check.status == "agrees":
        return outcome
    if check.status == "unchecked":
        if outcome.error is not None and check.reason == "no_log":
            return outcome
        observation = {
            "code": "ngspice_mode_unverified",
            "kind": "execution",
            "detail": (
                "The server could not check that ngspice read the deck in the compatibility "
                f"mode it passed (ngbehavior={check.mode!r}): {check.explanation()}."
            ),
            "evidence": check.evidence(),
        }
        return outcome._replace(observations=(*outcome.observations, observation))
    logged = " ".join(check.logged or ()) or "none"
    expected = " ".join(check.expected or ()) or "none"
    message = (
        f"ngspice read the deck in compatibility modes '{logged}', not in '{expected}', "
        f"those of ngbehavior={check.mode!r}, which the server passed and read the deck in. "
        "Something ngspice ran before the deck set another mode; every run skips a user's "
        ".spiceinit, so on an ordinary run that is most likely the installation's spinit."
    )
    code = "ngspice_mode_mismatch"
    return outcome._replace(
        error=message if outcome.error is None else f"{message}\n\n{outcome.error}",
        failure_code=code,
        failure_evidence={**(outcome.failure_evidence or {}), **check.evidence()},
    )


def _raw_completion_outcome(raw_file: str, log_file: str) -> RunOutcome | None:
    """Return a positive RAW or unreadable-location outcome before log capture."""
    sim_failed = raw_file in ("", ".") or Path(log_file).suffix == ".fail"
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
        written = tuple(other_raws(Path(raw_file)))
        return RunOutcome(raw_file, log_file, raw_size, None, other_raws=written)
    return None


def _native_log_output(logs: DecodedLog, requirements: DeckRequirements) -> dict[str, Any] | None:
    """Identify produced native prints without asserting full analysis coverage."""
    section = logs.section("native_tables")
    if section["status"] != "parsed" or not requirements.analyses:
        return None
    families: set[str] = set()
    for block in section["value"]:
        label = block["printed_analysis_label"]
        if label is None and block["listing_current"] is not None:
            label = block["listing_current"]["analysis_label"]
        if label is None:
            continue
        analysis = {
            "Transfer Function": ".tf",
            "Pole-Zero Analysis": ".pz",
            "Sensitivity Analysis": ".sens",
        }.get(label)
        if label.startswith("DISTORTION -"):
            analysis = ".disto"
        if analysis is not None:
            families.add(analysis)
    if not set(requirements.analyses).issubset(families):
        return None
    return {
        "code": "native_log_output",
        "kind": "reconciliation",
        "detail": (
            "Native printed results were captured without a RAW file. "
            "Printed blocks do not establish full analysis coverage."
        ),
        "evidence": {
            "analyses": list(requirements.analyses),
            "printed_blocks": len(section["value"]),
            "analysis_extent": "unknown",
        },
    }


def collect_run_outcome(
    raw_file: str,
    log_file: str,
    requirements: DeckRequirements | None = None,
    exit_code: int | None = None,
    *,
    logs: DecodedLog | None,
    logs_error: str | None = None,
    netlist: Path | None = None,
    simulator: type | None = None,
    ngbehavior: str | None = None,
    simulator_exception: str | None = None,
) -> RunOutcome:
    """Classify a cold completion using already captured log facts.

    The RAW metadata fast path runs before capture, in the submission callback.
    No log source is reopened here. Exit status and spicelib's caught exception
    remain separate process facts even when capture or section decoding fails.
    The deck may still be read to classify ngspice's sectioned-library failure.
    """
    sim_failed = raw_file in ("", ".") or Path(log_file).suffix == ".fail"
    errors: list[str] = []
    context: str | None = None
    log_present = False
    failures = [logs_error] if logs_error else []
    if logs is None:
        if not failures:
            failures.append("Completion log facts are unavailable")
    else:
        log_present = any(item.role == "log" for item in logs.captured.files)
        diagnostics = logs.section("diagnostics")
        excerpt = logs.section("error_context")
        for name, section in (("diagnostics", diagnostics), ("error_context", excerpt)):
            if section["status"] == "error":
                failure = section["error"]
                failures.append(f"{name}: {failure['type']}: {failure['message']}")
            elif log_present and section["status"] != "parsed":
                failures.append(f"Completion log section {name!r} is unavailable")
        if diagnostics["status"] == "parsed":
            errors = diagnostics["value"]["errors"]
        if excerpt["status"] == "parsed":
            context = excerpt["value"]
    # Quiet diagnostics alone prove neither process success nor produced data.
    if (
        exit_code == 0
        and not simulator_exception
        and not sim_failed
        and log_present
        and not failures
    ):
        non_rung = [error for error in errors if not is_op_stepping_failure(error)]
        if not non_rung and not op_ladder_exhausted(errors):
            requirements = requirements or DeckRequirements()
            if (
                logs is not None
                and not errors
                and is_ngspice(simulator)
                and (observation := _native_log_output(logs, requirements))
            ):
                return RunOutcome("", log_file, 0, None, observations=(observation,))
            analyses, has_save = requirements.raw_analyses, requirements.has_save
            if analyses:
                return _missing_required_raw_outcome(log_file, context, analyses, has_save)

    if context is not None:
        error = f"Simulation failed (no output generated)\n\nLog excerpt:\n{context}"
    else:
        error = "Simulation failed (no output generated, log file missing)"
    if failures:
        error += "\nCompletion log facts unavailable: " + "; ".join(failures)
    # The diagnostics above already name the cause; classifying here is what
    # turns it into a code a caller can branch on instead of prose it must read.
    code, evidence = classify_failure_code(errors)
    if code == "missing_include":
        mode = _is_ngspice_lib_section_failure(netlist, simulator, ngbehavior)
        if mode is not None:
            code = "ngspice_lib_section"
            evidence = {**(evidence or {}), "ngbehavior": mode}
    if exit_code not in (None, 0):
        error += f"\nSimulator exit code: {exit_code}"
        # Copy rather than mutate: classify_failure_code's return is typed
        # narrower than the relayed shape, and this is the cold path.
        evidence = {**(evidence or {}), "exit_code": exit_code}
    if simulator_exception:
        error += f"\nThe simulator call raised: {simulator_exception}"
        evidence = {**(evidence or {}), "simulator_exception": simulator_exception}
    return RunOutcome(
        "" if sim_failed else raw_file,
        log_file,
        0,
        error,
        failure_code=code,
        failure_evidence=evidence,
        log_excerpt=context,
    )


def inject_logopinfo(
    netlist_path: Path, simulator: type, job_id: str, *, reported: str | None = None
) -> Path:
    """Return a runnable netlist with ``.options logopinfo`` added, for LTspice ``.op`` runs.

    LTspice 24 and later write each semiconductor's small-signal operating
    point (gm, gds, vth, vdsat, junction caps) to the ``.log`` only under
    ``.options logopinfo``, and only for ``.op`` analyses — so adding it lets
    ``operating_point`` read those params back by name. ngspice uses
    ``@dev[param]`` raw traces instead and needs nothing here. Neither does
    LTspice XVII or IV: they print the block for every ``.op``, and stop with
    "unrecognized option" on a deck that carries the option.

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

    The deck is read as LTspice reads it (``spice_lex.lex_for_edit``): an
    ``.op`` or an option in its title or after its ``.END`` is not one LTspice
    sees, and the option goes in before the ``.END`` that ends the deck. The
    deck is rewritten in its own encoding, and the option's line ends with
    the deck's own line break, so every byte the option does not add is the
    caller's.
    """
    from spicelib.simulators.ltspice_simulator import LTspice

    if not (isinstance(simulator, type) and issubclass(simulator, LTspice)):
        return netlist_path
    # ``reported`` is the build an earlier run on this executable named
    # (``services.reported_build_of``): an LTspice before 24 under another name.
    if ltspice_before_24(executable_path(simulator), reported):
        return netlist_path
    if netlist_path.suffix.lower() not in RUN_DECK_SUFFIXES:
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    text, codec = decode_spice_bytes_with_encoding(data)
    bom = bom_of(data)
    try:
        deck = lex_for_edit(text, deck=True)
        # A deck whose bytes did not decode exactly cannot be written back
        # with only the option added.
        if bom + encode_spice_text_strictly(text, codec) != data:
            return netlist_path
    except (SpiceLexError, UnicodeError):
        return netlist_path
    if any("logopinfo" in card.body.casefold() for card in deck.read):
        return netlist_path
    # Only a real .op analysis emits the operating-point block; .dc does not.
    if ".op" not in deck_requirements(netlist_path).analyses:
        return netlist_path

    inject_card_before_end(deck.cards, ".options logopinfo\n")
    try:
        augmented = bom + encode_spice_text_strictly(emit(deck.cards), codec)
    except UnicodeError:
        return netlist_path

    run_path = netlist_path.with_name(
        f".{netlist_path.stem}.{job_id}{LOGOPINFO_MARKER}{netlist_path.suffix}"
    )
    try:
        run_path.write_bytes(augmented)
    except OSError:
        return netlist_path
    return run_path


# A ``write``/``wrdata`` command starting a line ngspice reads, anywhere in the
# deck — not just inside the block, since a script could call either from a
# subckt or a second block this pass doesn't otherwise recognize.
_RE_EXISTING_WRITE = re.compile(r"[ \t]*(?:write|wrdata)\b", re.IGNORECASE)
# ``quit``/``exit`` end control-script execution; a command placed after one
# would never run, so the injected ``write`` must land before the LAST one.
_RE_QUIT_EXIT = re.compile(r"[ \t]*(?:quit|exit)\b", re.IGNORECASE)


def _blank_or_comment(card: SpiceCard) -> bool:
    """Whether a ``.control`` line is blank or a ``*`` comment. A block whose
    lines after a quit/exit are all such ends with it: that quit/exit is the
    script's last statement, not one nested in an if/while (which must NOT
    anchor the injected write, or it lands inside that conditional and never
    runs on the success path)."""
    return all(not raw.strip() or raw.lstrip().startswith("*") for raw in card.raw_lines)


def inject_ngspice_control_write(
    netlist_path: Path, simulator: type, job_id: str, output_folder: Path
) -> Path:
    """Return a runnable netlist with a ``write`` injected into its
    ``.control`` block, for ngspice decks that drive their own analyses via
    scripting.

    This is ngspice runtime behavior, not a spicelib bug: a ``.control``
    block replaces the raw ngspice would otherwise write from the ``-r
    <rawfile>`` switch spicelib always passes — the script runs instead, and
    unless it calls ``write``/``wrdata`` itself, ngspice may write no raw
    file. ngspice-42 still writes the ``-r`` raw after a script that only
    ``run``s the deck's analyses, but not after one that ends with ``quit``
    or issues the analysis as a command (``tran``, ``op``).
    ``collect_run_outcome`` already classifies that as a clean log-only
    completion (not a failure), but nothing then exists for
    get_waveform/signal_stats/etc. to read. Injecting a canonical ``write
    <rawpath>`` gives the deck a raw at the exact path the runner expects for
    this job, so the existing raw>0 code path (raw_parser + every analysis
    tool) picks it up unchanged — no new parser, no new tool.

    Limitation: a bare ``write`` captures ngspice's current/last plot only. A
    script that runs multiple analyses, or writes per Monte-Carlo iteration
    inside a loop, needs its own explicit writes to capture each one — guard
    (c) below leaves any deck that already writes its own output alone
    rather than duplicating or fighting it. A second, unrelated limitation:
    the target is written by ``ngspice_driver.control_path``, which
    single-quotes a name with a space (ngspice keeps double quotes as part
    of the name) and refuses one the control language would change, such as
    a name with braces (guard (d)); such a run stays log-only.

    The deck is read as the simulator reads it (``spice_lex.lex_for_edit``):
    its title and a line after its ``.END`` are neither a block nor a
    ``write``, so a deck titled "Write-up of an RC filter" still gets one.

    Guards (all required, or the original path is returned unchanged):
    (a) ngspice only (LTspice has no ``.control``; ``inject_logopinfo``
        covers its own op-point injection separately).
    (b) exactly one ``.control``...``.endc`` block (ambiguous otherwise —
        e.g. which block's last analysis is "the" result).
    (c) no existing ``write``/``wrdata`` anywhere in the deck — never
        override a user who already captures their own output.
    (d) the write target has a spelling that reaches ngspice intact
        (``control_path``; see the limitation above).
    (e) the deck's bytes decode exactly, so it can be written back in its
        own encoding with only the ``write`` added, which ends with the
        deck's own line break.

    The ``write`` target is the ABSOLUTE path ``{output_folder}/{job_id}.raw``
    — the same path spicelib's own (suppressed) ``-r`` would use, since it
    derives the rawfile from the staged netlist's own path via
    ``.with_suffix('.raw')``. It must be absolute: the runner's SimRunner
    passes no ``cwd``, so ngspice inherits the MCP server's own working
    directory, not the output folder — a relative ``write`` target would land
    there instead. Quoted only when it needs to be (``control_path``).
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
    if netlist_path.suffix.lower() not in RUN_DECK_SUFFIXES:
        return netlist_path
    try:
        data = netlist_path.read_bytes()
    except OSError:
        return netlist_path

    text, codec = decode_spice_bytes_with_encoding(data)
    bom = bom_of(data)
    try:
        deck = lex_for_edit(text, deck=True)
        # A deck whose bytes did not decode exactly cannot be written back
        # with only the write added.
        if bom + encode_spice_text_strictly(text, codec) != data:
            return netlist_path
    except (SpiceLexError, UnicodeError):
        return netlist_path
    if any(_RE_EXISTING_WRITE.match(raw) for card in deck.read for raw in card.raw_lines):
        return netlist_path
    blocks = control_blocks(deck.read)
    if len(blocks) != 1:
        return netlist_path
    body = blocks[0][1:-1]

    try:
        raw_path = control_path((output_folder / f"{job_id}.raw").as_posix(), codec)
    except ValueError:
        # A folder name the control language would change (a brace, a
        # quote), or a non-ASCII one the deck's codec would make invalid
        # UTF-8, which stops ngspice: the run stays log-only instead.
        return netlist_path

    # Insert before .endc, UNLESS the block's last statement is an
    # unconditional trailing quit/exit — a write after that would never run. A
    # quit/exit nested in an if/while is not the last statement (an ``end`` and
    # possibly more follow it), so anchoring on it is skipped: the write goes
    # before .endc and runs on the normal path.
    anchor = blocks[0][-1]
    quits = [i for i, card in enumerate(body) if _RE_QUIT_EXIT.match(card.raw_lines[0])]
    if quits and all(_blank_or_comment(card) for card in body[quits[-1] + 1 :]):
        anchor = body[quits[-1]]
    at = next(i for i, card in enumerate(deck.cards) if card is anchor)
    deck.cards.insert(
        at,
        SpiceCard(
            kind="control",
            raw_lines=[f"write {raw_path}{line_break(deck.cards)}"],
            body="",
            line_start=anchor.line_start,
            scope=anchor.scope,
        ),
    )
    try:
        augmented = bom + encode_spice_text_strictly(emit(deck.cards), codec)
    except UnicodeError:
        return netlist_path

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

SUBPROCESS_TIMEOUT_CEILING_S = 4_000_000.0
"""The largest bound handed to spicelib's ``subprocess.run``, and what "no
bound" becomes. On Windows ``subprocess`` waits by passing the timeout to
``WaitForSingleObject`` as whole milliseconds in a 32-bit DWORD, which cannot
express more than about 49.7 days. spicelib cannot run without a number
either: its ``SimRunner.run`` computes ``timeout + 1`` before launching, so
``None`` raises ``TypeError`` (``docs/spicelib_bugs.md`` Bug 14)."""


class NativePrelaunchRefused(Exception):
    """The copied native setup was refused before a simulator task existed."""


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

    prelaunch_check: Callable[[Path], None] | None = None

    def _prepare_sim(self, netlist: str | Path | BaseEditor, run_filename: str | None) -> Path:
        copied = super()._prepare_sim(netlist, run_filename)
        # SimRunner calls this before constructing or starting the RunTask.
        # Check the copied deck while no simulator can have consumed it yet.
        if self.prelaunch_check is not None:
            try:
                self.prelaunch_check(copied)
            except Exception as exc:
                raise NativePrelaunchRefused(str(exc)) from exc
        return copied

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
        # Permits held plus callers waiting for one, including a caller already
        # handed a released permit that has not resumed yet.
        self._slots_claimed = 0
        # Submitted SimRunners, held until their simulation thread is done.
        # See _retire_finished_runners for why letting one go early is a trap.
        self._inflight_runners: dict[str, SimRunner] = {}

    @property
    def max_parallel(self) -> int:
        """Simulator processes this runner will have in flight at once.

        The requested cap. A change made while runs hold permits is enforced
        once the runner is idle (see ``_rebuild_slots_if_idle``).
        """
        return self._max_parallel

    @max_parallel.setter
    def max_parallel(self, value: int) -> None:
        self._max_parallel = value
        self._rebuild_slots_if_idle()

    def _rebuild_slots_if_idle(self) -> None:
        """Give the launch permits the current cap, but only when idle.

        A semaphore's permit count cannot be changed while permits are out
        without losing track of them, so a new cap takes effect the next time
        the runner is idle: when the setter finds it idle, or when the last
        claim on a permit ends. Runs already admitted keep the cap they started
        under. Idle counts waiters too: releasing a permit hands it to the next
        waiter before that waiter resumes, and a semaphore replaced in that gap
        would let the woken run and a newcomer both launch past the cap.
        """
        if self._slots_claimed == 0:
            self._launch_slots = asyncio.Semaphore(self._max_parallel)

    async def acquire_launch_slot(self) -> None:
        """Take one launch permit. Call on the event loop, release when done."""
        self._slots_claimed += 1
        try:
            await self._launch_slots.acquire()
        except BaseException:
            self._drop_slot_claim()
            raise

    def release_launch_slot(self) -> None:
        """Return a permit taken by ``acquire_launch_slot``."""
        self._launch_slots.release()
        self._drop_slot_claim()

    def _drop_slot_claim(self) -> None:
        self._slots_claimed -= 1
        self._rebuild_slots_if_idle()

    def _build_sim_runner(
        self,
        *,
        timeout_s: float | None = None,
        cwd: Path | None = None,
        prelaunch_check: Callable[[Path], None] | None = None,
        simulator_class: type | None = None,
    ) -> SimRunner:
        """Construct a spicelib SimRunner with this runner's settings.

        ``timeout_s`` reaches ``subprocess.run`` inside spicelib, which kills
        the process it started when the time runs out. None, and anything past
        ``SUBPROCESS_TIMEOUT_CEILING_S``, becomes that ceiling.
        """
        runner = _NonBlockingSimRunner(
            simulator=simulator_class or self.simulator_class,
            output_folder=str(self.output_folder),
            parallel_sims=self.max_parallel,
            timeout=(
                SUBPROCESS_TIMEOUT_CEILING_S
                if timeout_s is None
                else min(timeout_s, SUBPROCESS_TIMEOUT_CEILING_S)
            ),
            cwd=cwd,
        )
        runner.prelaunch_check = prelaunch_check
        return runner

    def _kill_by_token(self, token: str, context_label: str = "") -> None:
        """Best-effort blocking termination scoped to a command-line token."""
        subject = f"{context_label} {token}".strip()
        # The program this runner's class launches, which for a named
        # executable is that build's own, never the family default's.
        names = simulator_executable_names(self.simulator_class)
        try:
            killed = kill_windows_ltspice_by_token(token, names)
            if killed:
                logger.info("Killed %d Windows sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("WSL process kill for %s failed: %s", subject, exc)
        try:
            killed = kill_simulator_by_token(token, names)
            if killed:
                logger.info("Killed %d local sim process(es) for %s", killed, subject)
        except Exception as exc:
            logger.warning("Scoped process kill for %s failed: %s", subject, exc)

    def submit_netlist(
        self,
        netlist: Path,
        run_filename: str,
        callback: Callable[[Any], Any],
        *,
        native: NativeLaunchContext | None = None,
        timeout_s: float | None = None,
        simulator_class: type | None = None,
        prelaunch_check: Callable[[Path], None] | None = None,
        completion_logs: Callable[[Path | None, Path | None], DecodedLog] | None = None,
    ) -> SimRunner:
        """Submit one deck and bridge its filesystem-derived outcome to the loop.

        This is the job-agnostic single-run primitive every experiment case
        runs on. It knows only the deck, the simulator-facing filename, and an
        event-loop callback; registration, lifecycle, persistence, and
        concurrency remain with its callers.

        ``timeout_s`` bounds the simulator process inside spicelib, and is the
        caller's to choose: this layer adds no bound of its own. A run it ends
        completes through ``callback`` like any other failure, with exit code
        -2 and the timeout named in the outcome.

        Call from a worker thread. The requirements snapshot and completion
        artifact reads intentionally happen on spicelib's worker threads.

        The SimRunner is returned AND retained here, because the kill and
        liveness paths need the handle for as long as the simulation runs.
        Retaining at this choke point means no caller can lose it by
        forgetting to keep the return value. (Dropping one is no longer
        dangerous in itself: ``_NonBlockingSimRunner`` removes the destructor
        that used to pin the dropping thread for the whole simulation.)
        """
        if native is not None and not is_ngspice(self.simulator_class):
            raise ValueError("native statistical setup requires ngspice")
        electrical_deck = native.input_deck if native else netlist
        requirements = deck_requirements(electrical_deck)
        simulator = simulator_class or self.simulator_class
        # The mode ngspice is handed and the server reads the deck in: a native
        # setup's driver sets its own, every other run gets the class's -D.
        # The failure classifier and the mode check both read this one.
        passed_mode = (
            native.policy.ngbehavior
            if native
            else getattr(simulator, "_compatibility_mode", None) or ""
        )

        def verify_copied(copied: Path) -> None:
            if native is not None and native.verify_execution is not None:
                native.verify_execution()
            if prelaunch_check is not None:
                prelaunch_check(copied)

        def completion_callback(raw_file: Path | None, log_file: Path | None) -> None:
            # This runner is fresh per submission, so active_tasks holds
            # exactly this run's task — appended before its thread starts,
            # so it is present whenever the callback can fire.
            version: str | None = None
            raw_name = str(raw_file) if raw_file else ""
            log_name = str(log_file) if log_file else ""
            task = threading.current_thread()
            exit_code = getattr(task, "retcode", None)
            simulator_exception = getattr(task, "exception_text", None)
            try:
                # Read here, before the callback: a stopped case's artifacts
                # are removed once it lands. Both outcomes below carry it.
                version = reported_build(log_file, raw_file)
                outcome = _raw_completion_outcome(raw_name, log_name)
                if outcome is None:
                    logs = None
                    logs_error = None
                    if completion_logs is None:
                        logs_error = "No completion log loader was supplied"
                    else:
                        try:
                            logs = completion_logs(
                                Path(raw_name) if raw_name not in ("", ".") else None,
                                Path(log_name) if log_name not in ("", ".") else None,
                            )
                        except Exception as exc:
                            logs_error = f"{type(exc).__name__}: {exc}"
                    outcome = collect_run_outcome(
                        raw_name,
                        log_name,
                        requirements,
                        netlist=electrical_deck,
                        simulator=simulator,
                        ngbehavior=passed_mode,
                        exit_code=exit_code,
                        simulator_exception=simulator_exception,
                        logs=logs,
                        logs_error=logs_error,
                    )
                outcome = outcome._replace(simulator_version=version)
                if is_ngspice(simulator):
                    log_path = Path(log_name) if log_name not in ("", ".") else None
                    outcome = _with_mode_check(
                        outcome,
                        check_run(log_path, simulator, passed_mode, driver=native is not None),
                    )
            except Exception as exc:
                outcome = collect_run_outcome(
                    raw_name,
                    log_name,
                    requirements,
                    logs=None,
                    logs_error=f"Outcome collection: {type(exc).__name__}: {exc}",
                    exit_code=exit_code,
                    simulator_exception=simulator_exception,
                )._replace(simulator_version=version)
            self._bridge(callback, outcome, context=f"run {run_filename}")

        self._retire_finished_runners()
        runner = self._build_sim_runner(
            timeout_s=timeout_s,
            cwd=native.cwd if native else None,
            prelaunch_check=verify_copied if native is not None or prelaunch_check else None,
            simulator_class=simulator,
        )
        runner.run(
            str(netlist),
            run_filename=run_filename,
            callback=completion_callback,
            callback_on_error=True,
            exe_log=True,
            switches=_launch_switches(simulator, native),
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
