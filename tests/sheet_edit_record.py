"""What every ``edit_schematic`` call the suite makes answers, held to a record.

Each call is keyed by its content and compared with
``fixtures/sheet_edit_record.json``, so a change to how an edit is applied
shows as a call answering differently. A sheet is compared as its records,
not its bytes, so two ways of writing the same records compare equal.

- **The key** is the sheet before the call, read as records; the call's
  arguments, with paths, and a digest of the sheet it was handed, replaced;
  the symbol files the sheet and its new parts resolve to; and the files
  beside the sheet an edit can read (sheets, symbols, plot settings,
  netlists). So a call is matched wherever and in whatever order a
  test makes it. A machine whose own LTspice library resolves a symbol makes a
  different key, which is not compared.
- **The answer** is a digest of the committed sheet's records, grouped by kind
  in their order, each formatted, with LF line endings and the sheet's codec;
  and a digest of the complete reply, with paths, the sheets' digests and the
  build id replaced. A call that raised keeps the error's type and message.
- **A call answering differently** fails the test that made it. A call that
  answered two ways while the record was made is kept as varying, with the
  tests that made it, and is not compared: its answer turns on something no
  file holds, such as a stand-in LTspice window, a held lock or an injected
  failure. A call whose sheet another writer changed before it could be read
  back is neither recorded nor compared. The run's summary counts each kind
  of call.

Record again with ``LTSPICE_MCP_RECORD_SHEET_EDITS=1`` over the whole suite,
on a machine without an LTspice library, so the keys are the ones CI makes. A
recording that resolves a symbol from anywhere but the suite or the sheet's
folder writes nothing. Registered as a plugin from ``tests/conftest.py``.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import asc_document
from ltspice_mcp.lib.sheet import editor_symbol_resolver
from ltspice_mcp.tools import schematic_edit
from ltspice_mcp.tools._base import safe_path
from tests._proposal_record import write_record
from tests._schematic_fixtures import TESTS
from tests.ltspice_recorder import path_pattern

RECORD = TESTS / "fixtures" / "sheet_edit_record.json"
RECORD_ENV = "LTSPICE_MCP_RECORD_SHEET_EDITS"
_OUTPUT = "sheet_edit_record"

#: What an edit can read beside its sheet: other sheets, symbols, plot settings,
#: and a reference netlist or a file one includes.
_READABLE = frozenset({".asc", ".asy", ".plt", ".cir", ".net", ".sp", ".lib", ".inc", ".mod"})

#: A path under the work folder, once the folder is ``WORK``.
_UNDER_WORK = re.compile(r"WORK([\\/][^\s\"',;]*)")

_recording = os.environ.get(RECORD_ENV) == "1"
_record: dict[str, Any] = {}
#: While recording: each key, each answer it gave, and the tests that made it.
_answers: dict[str, dict[str, set[str]]] = {}
#: Symbols resolved from outside the suite and the sheet's folder while recording.
_foreign: set[str] = set()
_counts: Counter[str] = Counter()
#: The tests whose calls made a key the record does not hold.
_unrecorded: set[str] = set()
_mismatches: list[str] = []
_test = ""


def _digest(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()[:16]


@functools.cache
def _file_digest(path: str, mtime_ns: int, size: int) -> str:
    return _digest(Path(path).read_bytes())


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


def _scrubbed(value: Any, folders: tuple[Path, ...], tokens: dict[str, str]) -> Any:
    """``value`` with each string's folders as ``WORK`` (a path under one written
    with ``/``) and each token's text replaced by its name."""
    if isinstance(value, dict):
        return {k: _scrubbed(v, folders, tokens) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrubbed(v, folders, tokens) for v in value]
    if not isinstance(value, str):
        return value
    for folder in folders:
        value = path_pattern(str(folder)).sub("WORK", value)
    value = _UNDER_WORK.sub(lambda m: "WORK" + m.group(1).replace("\\", "/"), value)
    for text, name in tokens.items():
        value = value.replace(text, name)
    return value


def _given_paths(value: Any) -> dict[str, str]:
    """Each absolute POSIX path among ``value``'s strings, as this platform
    resolves it, mapped to the path as given. Windows puts a drive before such a
    path, so an answer naming one would otherwise differ from Linux's."""
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return {k: v for item in value for k, v in _given_paths(item).items()}
    if not isinstance(value, str) or not value.startswith("/") or len(value) < 2:
        return {}
    with contextlib.suppress(OSError, ValueError):
        resolved = str(Path(value).resolve())
        if resolved != value:
            return {resolved: value}
    return {}


def _symbols(target: Path, names: set[str]) -> list[Any]:
    """Each symbol name with the digest of the file it resolves to."""
    resolver = editor_symbol_resolver(target)
    found = []
    for name in sorted(names):
        path = resolver.resolve(name)
        inside = path is None or path.is_relative_to(TESTS) or path.is_relative_to(target.parent)
        if _recording and not inside:
            _foreign.add(f"{name} ({path})")
        found.append([name, None if path is None else _digest_of(path)])
    return found


def key_of(target: Path, before: bytes | None, args: Any, folders: tuple[Path, ...]) -> str:
    """The key of an ``edit_schematic`` call with ``args`` on ``target``, which
    held ``before``."""
    names = {op.symbol for op in args.ops if getattr(op, "op", "") == "add_component"}
    tokens: dict[str, str] = {}
    form = None
    if before is not None:
        tokens[hashlib.sha256(before).hexdigest()] = "BEFORE"
        with contextlib.suppress(Exception):
            names |= {part.symbol for part in asc_document.parse_asc(before).symbols}
            form = sheet_form(before)
    arguments = {**args.model_dump(mode="json"), "target": target.name}
    beside = [
        [path.name, _digest_of(path)]
        for path in sorted(target.parent.iterdir())
        if path.suffix.lower() in _READABLE and path != target and path.is_file()
    ]
    return _digest(
        json.dumps(
            {
                "before": form if before is None or form else _digest(before),
                "args": _scrubbed(arguments, folders, tokens),
                "symbols": _symbols(target, names),
                "beside": beside,
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
        return
    recorded = _record.get(key)
    if recorded is None:
        _counts["not recorded"] += 1
        _unrecorded.add(_test)
        return
    _counts["compared"] += 1
    if recorded != answer:
        _mismatches.append(f"call {key}: recorded {recorded}, answered {answer}")


_evaluate = schematic_edit._evaluate_edit_schematic  # pyright: ignore[reportPrivateUsage]


async def _recorded_evaluate(args: Any, state: Any, *, present_mcp_views: bool = False) -> Any:
    try:
        target = safe_path(args.target, state)
    except Exception:
        return await _evaluate(args, state, present_mcp_views=present_mcp_views)
    folders = (target.parent, Path(state.working_dir))
    before = target.read_bytes() if target.is_file() else None
    key = key_of(target, before, args, folders)
    if not _recording and "varies" in _record.get(key, {}):
        _counts["varying"] += 1
        return await _evaluate(args, state, present_mcp_views=present_mcp_views)
    tokens = _given_paths(args.model_dump(mode="json"))
    if before is not None:
        tokens[hashlib.sha256(before).hexdigest()] = "BEFORE"
    try:
        evaluation = await _evaluate(args, state, present_mcp_views=present_mcp_views)
    except Exception as exc:
        message = _scrubbed(str(exc), folders, tokens)
        _observe(key, {"raised": type(exc).__name__, "message": message})
        raise
    data = schematic_edit.complete_edit_schematic_data(evaluation, args)
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
    reply = json.dumps(_scrubbed(data, folders, tokens), sort_keys=True)
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
    schematic_edit._evaluate_edit_schematic = _recorded_evaluate  # pyright: ignore[reportPrivateUsage]
    if not _recording and RECORD.is_file():
        _record = json.loads(RECORD.read_text(encoding="utf-8"))


def _merge(seen: dict[str, Any]) -> None:
    for key, answers in seen["answers"].items():
        for answer, tests in answers.items():
            _answers.setdefault(key, {}).setdefault(answer, set()).update(tests)
    _foreign.update(seen["foreign"])
    _unrecorded.update(seen["unrecorded"])
    _counts.update(seen["counts"])


@pytest.hookimpl(optionalhook=True)
def pytest_testnodedown(node: Any, error: Any) -> None:
    """Take what an xdist worker saw."""
    seen = getattr(node, "workeroutput", {}).get(_OUTPUT)
    if seen:
        _merge(seen)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Hand a worker's calls to the controller; at the controller, write the record."""
    if hasattr(session.config, "workerinput"):
        session.config.workeroutput[_OUTPUT] = {  # pyright: ignore[reportAttributeAccessIssue]
            "answers": {k: {a: sorted(t) for a, t in v.items()} for k, v in _answers.items()},
            "foreign": sorted(_foreign),
            "unrecorded": sorted(_unrecorded),
            "counts": dict(_counts),
        }
        return
    if not _recording or _foreign or not _answers:
        return
    record: dict[str, Any] = {}
    for key, answers in sorted(_answers.items()):
        if len(answers) == 1:
            record[key] = json.loads(next(iter(answers)))
        else:
            record[key] = {"varies": sorted(set().union(*answers.values()))}
    write_record(RECORD, record)


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if _foreign:
        terminalreporter.write_line(
            f"{RECORD.name} not written: symbols resolved outside the suite: "
            + ", ".join(sorted(_foreign)[:5])
        )
    if _counts:
        terminalreporter.write_line(
            f"edit_schematic calls against {RECORD.name}: "
            + ", ".join(f"{count} {what}" for what, count in sorted(_counts.items()))
        )
    for test in sorted(_unrecorded)[:10]:
        terminalreporter.write_line(f"  not recorded: {test}")


@pytest.fixture(autouse=True)
def _sheet_edit_record(request: pytest.FixtureRequest) -> Iterator[None]:
    """Name the running test for a recording, and fail it if a call answered
    differently from the record."""
    global _test
    _test = request.node.nodeid
    yield
    found = take_mismatches()
    if found:
        pytest.fail(
            f"edit_schematic answered differently from {RECORD.name}:\n" + "\n".join(found)
        )
