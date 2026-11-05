"""What every ``edit_schematic`` call the suite makes answers, held to a record.

Each call is keyed by its content and compared with
``fixtures/sheet_edit_record.json``, so a change to how an edit is applied
shows as a call answering differently. A sheet is compared as its records,
not its bytes, so two ways of writing the same records compare equal.

- **The key** is the sheet before the call, read as records; the arguments
  the call gives, leaving out each one at its default, with paths, and a
  digest of the sheet it was handed, replaced; the symbol files the sheet and
  its new parts resolve to; and the files beside the sheet an edit of it can
  read, each digested with LF line breaks: symbols, libraries and netlists,
  which a part, an include or a reference can name; the sheet's own plot
  settings; and another sheet only where it is a block's or the call names it.
  So a call is matched wherever, on whichever platform and in whatever order
  a test makes it, and an input field added later, or a sheet left beside
  the one edited, changes no key. A machine whose own LTspice library
  resolves a symbol makes a different key, which is not compared.
- **The answer** is a digest of the committed sheet's records, grouped by kind
  in their order, each formatted, with LF line endings and the sheet's codec;
  and a digest of the complete reply as it is sent (without the fields that
  say nothing), with paths, digests and the build id replaced. A call that
  raised keeps the error's type and message, with the same replacements.
- **A call answering differently** fails the test that made it. A call that
  answered two ways while the record was made is kept as varying, and is not
  compared: its answer turns on something no file holds, such as a stand-in
  LTspice window, a held lock or an injected failure. A call whose sheet
  another writer changed before it could be read back is neither recorded nor
  compared. The run's summary counts each kind of call.
- **A test marked** ``edit_reply_per_platform`` makes calls whose replies
  differ on Windows and Linux, such as one reading a path only Windows takes
  as absolute. Their keys and their entries hold the platform, so each
  platform compares them only with a recording made on it. A recording keeps
  only its own platform's entries for them, so one made on Windows replaces
  the Linux entries with Windows ones.
- **A stale record fails the run.** Each entry names the tests that made its
  call. In a run of the whole suite, an entry no call matched is stale when a
  test that made it passed, or when none of them ran: a test that no longer
  exists makes no call. It is not stale when its tests were skipped or
  failed, or resolved a symbol outside the suite, since their calls were then
  not made, or not keyed as CI keys them. A stale entry fails the run. A run
  of part of the suite (``-k`` or ``-m``, named files or tests, a stop after
  a failure, a worker that went down) cannot tell an entry no test makes from
  one whose test did not run, so it fails on none and reports the entries of
  tests that passed without making their call.

Record again with ``LTSPICE_MCP_RECORD_SHEET_EDITS=1`` over the whole suite,
on a machine without an LTspice library, so the keys are the ones CI makes. A
recording of part of the suite, or one that resolves a symbol from anywhere
but the suite or the sheet's folder, writes nothing. Registered as a plugin
from ``tests/conftest.py``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path, PurePath
from typing import Any

import pytest

from ltspice_mcp.errors import LTSpiceMCPError
from ltspice_mcp.lib import asc_document
from ltspice_mcp.lib.plot_settings import plot_settings_path
from ltspice_mcp.lib.symbol_library import SymbolResolver
from ltspice_mcp.tools import schematic_edit
from ltspice_mcp.tools._base import as_sent, safe_path
from tests._proposal_record import write_record
from tests._schematic_fixtures import TESTS
from tests.ltspice_recorder import path_pattern

RECORD = TESTS / "fixtures" / "sheet_edit_record.json"
RECORD_ENV = "LTSPICE_MCP_RECORD_SHEET_EDITS"
PER_PLATFORM = "edit_reply_per_platform"
_OUTPUT = "sheet_edit_record"
_PLATFORM = "windows" if sys.platform == "win32" else "posix"

#: What an edit can read beside its sheet whichever file a part, a directive's
#: include or a compare reference names: symbols, libraries and netlists. Which
#: of them it reads is decided where each is read, so the key holds them all.
_NAMED = frozenset({".asy", ".lib", ".inc", ".mod", ".sub", ".cir", ".net", ".sp"})

#: A path under the work folder, once the folder is ``WORK``.
_UNDER_WORK = re.compile(r"WORK([\\/][^\s\"',;]*)")

#: A SHA-256 digest that is not one of a call's sheets: the digest of bytes a test
#: wrote, which differ with the platform's line breaks.
_SHA256 = re.compile(r"\b[0-9a-f]{64}\b")

_recording = os.environ.get(RECORD_ENV) == "1"
_record: dict[str, Any] = {}
#: While recording: each key, each answer it gave, and the tests that made it.
_answers: dict[str, dict[str, set[str]]] = {}
#: While recording: the keys that hold the platform.
_platform_keys: set[str] = set()
#: Symbols resolved from outside the suite and the sheet's folder.
_foreign: set[str] = set()
#: The tests that resolved such a symbol: their keys are this machine's own.
_foreign_tests: set[str] = set()
#: Every key a call made.
_made: set[str] = set()
_counts: Counter[str] = Counter()
#: The tests whose calls made a key the record does not hold.
_unrecorded: set[str] = set()
_mismatches: list[str] = []
_test = ""
_per_platform = False
#: Why the run is not one of the whole suite, as each process saw it.
_partial: set[str] = set()
#: Each test's outcome as the run reports it: passed, failed or skipped.
_outcomes: dict[str, str] = {}
#: The tests the run finished, teardown included.
_finished: set[str] = set()
#: The collectors reported skipped: no test under one ran.
_skipped_collectors: set[str] = set()
#: Found when the run finished: why it is not one of the whole suite, and the
#: entries of the record no call matched.
_not_whole: list[str] = []
_stale: list[str] = []


def _digest(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:16]


#: A line break in UTF-8 or UTF-16 LE. A test writes a file with the platform's
#: line breaks, so a file's digest is taken with LF ones.
_LINE_BREAK = re.compile(rb"\r(?=\x00?\n)\x00?")


@functools.cache
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    return _digest(_LINE_BREAK.sub(b"", Path(path).read_bytes()))


def _digest_of(path: Path) -> str:
    stat = path.stat()
    return _file_digest(str(path), stat.st_mtime_ns, stat.st_size)


def sheet_form(data: bytes) -> str:
    """The digest of a sheet's records: grouped by kind in their order, each in its
    formatted spelling (a directive's text stripped, as spicelib reads it), LF line
    endings, and the codec and byte order mark the sheet is in."""
    doc = asc_document.parse_asc(data)
    lines = [f"{doc.encoding} {doc.bom.hex()}"]
    for record in sorted(doc.records, key=lambda r: 99 if r.rank is None else r.rank):
        if isinstance(record, asc_document.Text):
            record = record.changed(body=record.body.strip())
        lines.extend(record.format())
    return _digest("\n".join(lines))


def scrubbed(
    value: Any,
    folders: tuple[PurePath, ...],
    tokens: dict[str, str],
    given: tuple[str, ...] = (),
) -> Any:
    """``value`` with each string's folders as ``WORK`` (a path under one written
    with ``/``), each absolute POSIX path in ``given`` as given where Windows put
    a drive before it or the host resolved a link in it (macOS's ``/etc`` is
    ``/private/etc``), each token's text replaced by its name, and any other
    SHA-256 digest as ``SHA256``. So a string naming a path or a file's digest
    reads the same on Linux, Windows and macOS."""
    if isinstance(value, dict):
        return {k: scrubbed(v, folders, tokens, given) for k, v in value.items()}
    if isinstance(value, list):
        return [scrubbed(v, folders, tokens, given) for v in value]
    if not isinstance(value, str):
        return value
    for folder in folders:
        # path_pattern leaves out the separator a POSIX path starts with.
        pattern = re.compile(r"[\\/]?" + path_pattern(str(folder)).pattern, re.IGNORECASE)
        value = pattern.sub("WORK", value)
    value = _UNDER_WORK.sub(lambda m: "WORK" + m.group(1).replace("\\", "/"), value)
    for path in given:
        on_a_drive = re.compile(r"[A-Za-z]:[\\/]+" + path_pattern(path).pattern, re.IGNORECASE)
        value = on_a_drive.sub(lambda _, path=path: path, value)
        resolved = os.path.realpath(path)
        if resolved != path:
            value = value.replace(resolved, path)
    for text, name in tokens.items():
        value = value.replace(text, name)
    return _SHA256.sub("SHA256", value)


def _strings(value: Any) -> Iterator[str]:
    """Every string in ``value``, a call's arguments as JSON."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def _given_paths(value: Any) -> tuple[str, ...]:
    """The absolute POSIX paths among ``value``'s strings."""
    return tuple(text for text in _strings(value) if len(text) > 1 and text.startswith("/"))


def _file_name(text: str) -> str:
    """The last part of a path written with either separator, folded."""
    return PurePath(text.replace("\\", "/")).name.casefold()


def _asked(args: Any) -> dict[str, Any]:
    """The arguments ``args`` gives: each field, nested ops' among them, that is
    not at its default.

    Not the rule a reply is sent by (``as_sent``): a field left out of a reply
    reads as empty, but an argument left out reads as its default, so
    ``return_views=[]`` asks for no views where an omitted one asks for the
    default's. A field added to the input later is at its default in every call
    made before it, so it changes no key.
    """
    return args.model_dump(mode="json", exclude_defaults=True)


def _beside(target: Path, names: set[str], arguments: dict[str, Any]) -> list[list[str]]:
    """Each file beside ``target`` an edit of it can read, with its digest.

    That is a file of a kind in ``_NAMED``; the sheet's own plot settings
    (``plot_settings_path``); and a sheet that is a block's or that
    ``arguments`` name. A block's sheet has a symbol's name: a symbol beside it,
    as ``deck_prep.stage_sheet`` stages a block's sheet with its symbol, or a
    part in ``names``, as spicelib opens a block's sheet. No other sheet or plot
    settings file is read, so the sheet a fixture copies into every work folder
    changes no key.
    """
    files = sorted(
        (path for path in target.parent.iterdir() if path != target and path.is_file()),
        key=lambda path: path.name,
    )
    blocks = {path.stem.casefold() for path in files if path.suffix.lower() == ".asy"}
    blocks |= {_file_name(name) for name in names}
    named = {_file_name(text) for text in _strings(arguments) if "\n" not in text}
    plot = plot_settings_path(target)

    def read(path: Path) -> bool:
        suffix = path.suffix.lower()
        if suffix == ".asc":
            return path.stem.casefold() in blocks or path.name.casefold() in named
        if suffix == ".plt":
            return path == plot
        return suffix in _NAMED

    return [[path.name, _digest_of(path)] for path in files if read(path)]


def _symbols(target: Path, names: set[str], resolver: SymbolResolver) -> list[Any]:
    """Each symbol name with the digest of the file ``resolver`` finds for it."""
    found = []
    for name in sorted(names):
        path = resolver.resolve(name)
        inside = path is None or path.is_relative_to(TESTS) or path.is_relative_to(target.parent)
        if not inside:
            _foreign.add(f"{name} ({path})")
            _foreign_tests.add(_test)
        found.append([name, None if path is None else _digest_of(path)])
    return found


def key_of(
    target: Path,
    before: bytes | None,
    args: Any,
    folders: tuple[Path, ...],
    *,
    resolver: SymbolResolver,
    per_platform: bool = False,
) -> str:
    """The key of an ``edit_schematic`` call with ``args`` on ``target``, which
    held ``before``, its symbols found with ``resolver``; with the platform in
    it when ``per_platform``."""
    names = {op.symbol for op in args.ops if getattr(op, "op", "") == "add_component"}
    tokens: dict[str, str] = {}
    form = None
    if before is not None:
        tokens[hashlib.sha256(before).hexdigest()] = "BEFORE"
        with contextlib.suppress(Exception):
            names |= {part.symbol for part in asc_document.parse_asc(before).symbols}
            form = sheet_form(before)
    arguments = {**_asked(args), "path": target.name}
    return _digest(
        json.dumps(
            {
                "before": form if before is None or form else _digest(before),
                "args": scrubbed(arguments, folders, tokens),
                "symbols": _symbols(target, names, resolver),
                "beside": _beside(target, names, arguments),
                **({"platform": _PLATFORM} if per_platform else {}),
            },
            sort_keys=True,
        )
    )


def _observe(key: str, answer: dict[str, Any]) -> None:
    if _recording:
        _counts["recorded"] += 1
        _answers.setdefault(key, {}).setdefault(json.dumps(answer, sort_keys=True), set()).add(
            _test
        )
        if _per_platform:
            _platform_keys.add(key)
        return
    recorded = _record.get(key)
    if recorded is None and _per_platform:
        _counts["not recorded on this platform"] += 1
        return
    if recorded is None:
        _counts["not recorded"] += 1
        _unrecorded.add(_test)
        return
    _counts["compared"] += 1
    if recorded.get("answer") != answer:
        _mismatches.append(f"call {key}: recorded {recorded.get('answer')}, answered {answer}")


_evaluate = schematic_edit._evaluate_edit_schematic  # pyright: ignore[reportPrivateUsage]


async def _recorded_evaluate(args: Any, state: Any, *, present_mcp_views: bool = False) -> Any:
    try:
        target = safe_path(args.path, state)
    except (LTSpiceMCPError, OSError, ValueError):
        # A path the call is refused for: the edit answers that itself.
        return await _evaluate(args, state, present_mcp_views=present_mcp_views)
    folders = (target.parent, Path(state.working_dir))
    before = target.read_bytes() if target.is_file() else None
    key = key_of(
        target,
        before,
        args,
        folders,
        resolver=state.symbols.for_sheet(target),
        per_platform=_per_platform,
    )
    _made.add(key)
    if not _recording and "varies" in _record.get(key, {}):
        _counts["varying"] += 1
        return await _evaluate(args, state, present_mcp_views=present_mcp_views)
    given = _given_paths(args.model_dump(mode="json"))
    tokens = {} if before is None else {hashlib.sha256(before).hexdigest(): "BEFORE"}
    try:
        evaluation = await _evaluate(args, state, present_mcp_views=present_mcp_views)
    except Exception as exc:
        message = scrubbed(str(exc), folders, tokens, given)
        _observe(key, {"raised": type(exc).__name__, "message": message})
        raise
    # The complete reply as it is sent: without the fields that say nothing.
    data = as_sent(schematic_edit.complete_edit_schematic_data(evaluation, args), "edit_schematic")
    sheet = None
    if data.get("commit_state") == "committed":
        after = target.read_bytes()
        digest = hashlib.sha256(after).hexdigest()
        if digest != data.get("sha256"):
            _counts["overwritten before read"] += 1
            return evaluation
        tokens[digest] = "AFTER"
        sheet = sheet_form(after)
    if data.get("build_id"):
        tokens[data["build_id"]] = "BUILD"
    reply = json.dumps(scrubbed(data, folders, tokens, given), sort_keys=True)
    _observe(key, {"sheet": sheet, "reply": _digest(reply)})
    return evaluation


def take_mismatches() -> list[str]:
    """The calls found answering differently from the record since the last take."""
    found = list(_mismatches)
    _mismatches.clear()
    return found


def pytest_configure(config: pytest.Config) -> None:
    """Wrap the evaluator in this process, and read the record unless making it."""
    global _record
    config.addinivalue_line(
        "markers",
        f"{PER_PLATFORM}(reason): the test's edit_schematic replies differ on Windows "
        "and Linux, so its calls are compared only with a recording made on the same one",
    )
    schematic_edit._evaluate_edit_schematic = _recorded_evaluate  # pyright: ignore[reportPrivateUsage]
    if not _recording and RECORD.is_file():
        _record = json.loads(RECORD.read_text(encoding="utf-8"))


def _merge(seen: dict[str, Any]) -> None:
    for key, answers in seen["answers"].items():
        for answer, tests in answers.items():
            _answers.setdefault(key, {}).setdefault(answer, set()).update(tests)
    _platform_keys.update(seen["platform_keys"])
    _foreign.update(seen["foreign"])
    _foreign_tests.update(seen["foreign_tests"])
    _made.update(seen["made"])
    _partial.update(seen["partial"])
    _unrecorded.update(seen["unrecorded"])
    _counts.update(seen["counts"])


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """Take what an xdist worker saw. A worker that went down without handing
    it over took its calls with it, so the run cannot say which were made."""
    seen = getattr(node, "workeroutput", {}).get(_OUTPUT)
    if error is None and seen:
        _merge(seen)
    else:
        worker = getattr(getattr(node, "gateway", None), "id", "a worker")
        _partial.add(f"worker {worker} went down before handing over its calls")


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Name the test whose calls follow, before its fixtures make any."""
    global _test, _per_platform
    _test = item.nodeid
    _per_platform = item.get_closest_marker(PER_PLATFORM) is not None


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    """Note each test's outcome, and that the run finished it. Under xdist the
    controller is handed every worker's reports, a crashed test's among them."""
    if report.when == "teardown":
        _finished.add(report.nodeid)
    elif report.when == "call" or not report.passed:
        _outcomes[report.nodeid] = report.outcome


def pytest_collectreport(report: pytest.CollectReport) -> None:
    """Note a collector that failed, so the run is not one of the whole suite,
    and one that was skipped, so the tests under it did not run."""
    if report.failed:
        _partial.add(f"{report.nodeid or 'the session'} could not be collected")
    elif report.skipped:
        _skipped_collectors.add(report.nodeid)


def pytest_deselected(items: list[pytest.Item]) -> None:
    """Called for ``-k``, ``-m``, ``--deselect`` and ``--sw``."""
    if items:
        _partial.add(f"{len(items)} tests deselected")


def _why_not_whole(session: pytest.Session) -> list[str]:
    """Why ``session`` ran less than the whole suite, which is what ``pytest``
    collects with no arguments (``testpaths``, else the root folder), and ran
    all of it: nothing when it did."""
    config = session.config
    why = set(_partial)
    given = [
        Path(config.invocation_params.dir, arg).resolve() for arg in config.args if "::" not in arg
    ]
    roots = [config.rootpath / path for path in config.getini("testpaths")] or [config.rootpath]
    for root in roots:
        if not any(root.resolve().is_relative_to(path) for path in given):
            why.add(f"only part of {root.name} collected")
    if config.getoption("ignore", None) or config.getoption("ignore_glob", None):
        why.add("paths ignored")
    if config.getoption("lf", False):
        why.add("only the last failures run")
    if not session.testscollected:
        why.add("no tests collected")
    unfinished = session.testscollected - len(_finished)
    if unfinished > 0:
        why.add(f"{unfinished} collected tests not finished")
    return sorted(why)


def _status(test: str) -> str | None:
    """How ``test`` ran: passed, failed or skipped, or foreign when a call it
    made resolved a symbol outside the suite; None when it did not run."""
    if test in _foreign_tests:
        return "foreign"
    if test in _outcomes:
        return _outcomes[test]
    if any(
        test.startswith((collector + "::", collector + "/")) for collector in _skipped_collectors
    ):
        return "skipped"
    return None


def _find_stale(whole: bool) -> list[str]:
    """The record's entries for this platform that no call matched, though a
    test that made one passed, or, in a run of the whole suite, though no test
    that made one ran at all: it no longer exists. Each with its tests."""
    stale = []
    for key, entry in sorted(_record.items()):
        if key in _made or entry.get("platform", _PLATFORM) != _PLATFORM:
            continue
        tests = entry.get("tests", [])
        ran = [_status(test) for test in tests]
        if "passed" in ran or (whole and all(status is None for status in ran)):
            stale.append(f"{key}: {', '.join(tests) or 'no test'}")
    return stale


def _new_record() -> dict[str, Any]:
    record: dict[str, Any] = {}
    for key, answers in sorted(_answers.items()):
        entry: dict[str, Any] = (
            {"answer": json.loads(next(iter(answers)))} if len(answers) == 1 else {"varies": True}
        )
        if key in _platform_keys:
            entry["platform"] = _PLATFORM
        entry["tests"] = sorted(set().union(*answers.values()))
        record[key] = entry
    return record


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Hand a worker's calls to the controller. At the controller, write the
    record from a recording of the whole suite, or fail a run of the whole
    suite on the entries of the record no call matched."""
    if hasattr(session.config, "workerinput"):
        session.config.workeroutput[_OUTPUT] = {  # pyright: ignore[reportAttributeAccessIssue]
            "answers": {k: {a: sorted(t) for a, t in v.items()} for k, v in _answers.items()},
            "platform_keys": sorted(_platform_keys),
            "foreign": sorted(_foreign),
            "foreign_tests": sorted(_foreign_tests),
            "made": sorted(_made),
            "partial": sorted(_partial),
            "unrecorded": sorted(_unrecorded),
            "counts": dict(_counts),
        }
        return
    _not_whole[:] = _why_not_whole(session)
    if _recording:
        if _answers and not _foreign and not _not_whole:
            write_record(RECORD, _new_record())
        return
    _stale[:] = _find_stale(whole=not _not_whole)
    if _stale and not _not_whole and session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_terminal_summary(terminalreporter: Any) -> None:
    write = terminalreporter.write_line
    partial = "; ".join(_not_whole)
    again = (
        f"Record it again with {RECORD_ENV}=1 over the whole suite (tests/sheet_edit_record.py)."
    )
    if _recording and _foreign:
        write(
            f"{RECORD.name} not written: symbols resolved outside the suite: "
            + ", ".join(sorted(_foreign)[:5])
        )
    elif _recording and _answers and _not_whole:
        write(f"{RECORD.name} not written: not a run of the whole suite ({partial})")
    if _counts:
        write(
            f"edit_schematic calls against {RECORD.name}: "
            + ", ".join(f"{count} {what}" for what, count in sorted(_counts.items()))
        )
    for test in sorted(_unrecorded)[:10]:
        write(f"  not recorded: {test}")
    if _stale and not _not_whole:
        write(
            f"{RECORD.name} is stale: {len(_stale)} of its entries matched no call. {again}",
            red=True,
        )
    elif _stale:
        write(
            f"{len(_stale)} entries of {RECORD.name} matched no call of a test that passed. "
            f"This is not a run of the whole suite ({partial}), so it does not fail on them; "
            f"a run of the whole suite does. {again}"
        )
    elif _counts and _not_whole and not _recording:
        write(
            f"{RECORD.name} checked for stale entries only where a test ran: not a run of "
            f"the whole suite ({partial})"
        )
    for line in _stale[:10]:
        write(f"  {line}")


@pytest.fixture(autouse=True)
def _sheet_edit_record() -> Iterator[None]:
    """Fail the running test if a call it made answered differently from the record."""
    yield
    found = take_mismatches()
    if found:
        pytest.fail(
            f"edit_schematic answered differently from {RECORD.name}:\n" + "\n".join(found)
        )
