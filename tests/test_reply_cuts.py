"""Every list in a reply declares how it may be cut, and only the reply layer
knows about reply size.

Design principle "A short reply, the data behind an id" (``docs/DESIGN.md``):
a reply too long to send whole is cut, stored, and read on, the same way in
every tool. That needs every list to say how it may be cut
(``lib/reply_cuts.py``), one writer of the sentence that says how to read the
rest (``lib/continuation.py``), and operations that return their whole answer
and leave the cut to the reply layer. This file holds the three to it:

- the completeness test: every list in every declared output schema carries a
  declaration, and a list of rows names key fields its rows have;
- the table of lists sent whole, each with what bounds it where it is made;
- the agreement with the heuristic in
  ``tests/test_capped_lists_say_how_to_continue.py``: a list that is cut today
  is declared cuttable, and a list declared never cut is not cut today;
- the scan for code outside the reply layer that knows about cursors or reply
  budgets, with today's sites allowed by function until their tool moves;
- the scan for continuation sentences written by hand.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from types import ModuleType
from typing import Any

import pytest

from ltspice_mcp.lib.continuation import Unfilled, call_to_continue, mcp_call, python_call
from ltspice_mcp.lib.reply_cuts import (
    CUT,
    ITEMS,
    NEVER,
    ReplyList,
    read_cut,
    rows,
    schema_lists,
    series,
)
from tests._source_scan import PACKAGE, parsed, scopes
from tests.conftest import declared_output_schemas
from tests.test_capped_lists_say_how_to_continue import _cut_lists

# ---------------------------------------------------------------------------
# The schemas
# ---------------------------------------------------------------------------


def _handler_name(module: ModuleType, handler: Any) -> str:
    return f"{module.__name__.rsplit('.', 1)[-1]}.{handler.__name__}"


def _schemas() -> dict[str, dict[str, Any]]:
    """Every declared output schema, registered or delegated, by
    ``module.handler``."""
    return {
        _handler_name(module, handler): schema
        for module, handler, schema in declared_output_schemas()
    }


def _lists() -> Iterator[tuple[str, ReplyList]]:
    for name, schema in sorted(_schemas().items()):
        for found in schema_lists(schema):
            yield name, found


def _unbranched(path: str) -> str:
    """A path without its union branches: the jobs actions share their lists."""
    return re.sub(r"<(?:anyOf|oneOf|allOf)\d+>\.?", "", path)


def _reachable(value: Any, into: set[int]) -> None:
    """Add to ``into`` the id of every dict reachable from ``value``."""
    if isinstance(value, dict):
        if id(value) in into:
            return
        into.add(id(value))
        for item in value.values():
            _reachable(item, into)
    elif isinstance(value, list):
        for item in value:
            _reachable(item, into)


#: The one tool whose lists are not declared yet: the schematic editor moves
#: onto the shared sheet model first, and its replies are pinned byte for byte
#: by ``tests/fixtures/sheet_edit_record.json``. A list made in
#: ``tools/schematic_edit.py`` is exempt; one it shares with another module is
#: not.
_EXEMPT_HANDLER = "schematic_edit.handle_edit_schematic"


def _made_in_schematic_edit() -> set[int]:
    """The ids of the schema dicts made in ``tools/schematic_edit.py``: those
    its own constants reach and no other module's constants or schemas do."""
    import sys

    import ltspice_mcp.tools.schematic_edit as schematic_edit

    own: set[int] = set()
    for name, value in vars(schematic_edit).items():
        if name.lstrip("_").isupper():
            _reachable(value, own)
    elsewhere: set[int] = set()
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("ltspice_mcp.") or module is schematic_edit:
            continue
        for name, value in vars(module).items():
            if isinstance(value, (dict, list)) and name.lstrip("_").isupper():
                _reachable(value, elsewhere)
    for name, schema in _schemas().items():
        if name != _EXEMPT_HANDLER:
            _reachable(schema, elsewhere)
    return own - elsewhere


# ---------------------------------------------------------------------------
# The vocabulary
# ---------------------------------------------------------------------------


def test_the_four_declarations_read_back():
    assert read_cut("a", {"type": "array", CUT: NEVER}).kind == "never"
    assert read_cut("a", {"type": "array", CUT: ITEMS}).kind == "items"
    row = read_cut("a", {"type": "array", CUT: rows("case_id", "run_index")})
    assert (row.kind, row.keys, row.projected_by) == ("rows", ("case_id", "run_index"), None)
    projected = read_cut("a", {"type": "array", CUT: rows("k", projected_by="run_fields")})
    assert projected.projected_by == "run_fields"
    sampled = read_cut("a", {"type": "array", CUT: series("waveform")})
    assert (sampled.kind, sampled.series) == ("series", "waveform")
    assert read_cut("a", {"type": "array"}).kind is None


@pytest.mark.parametrize(
    "declared",
    ["sometimes", {"rows": []}, {"rows": ["k"], "extra": 1}, {"series": ""}, {"items": True}],
)
def test_a_malformed_declaration_is_refused(declared):
    with pytest.raises(ValueError, match="x-cut"):
        read_cut("a.b", {"type": "array", CUT: declared})


def test_the_walk_finds_every_list_where_it_sits():
    schema = {
        "type": "object",
        "$defs": {"Row": {"type": "object", "properties": {"tags": {"type": "array"}}}},
        "properties": {
            "rows": {"type": "array", CUT: ITEMS, "items": {"$ref": "#/$defs/Row"}},
            "by_name": {"type": "object", "additionalProperties": {"type": ["array", "null"]}},
        },
        "oneOf": [{"properties": {"other": {"type": "array", CUT: NEVER}}}],
    }
    found = {(item.path, item.kind) for item in schema_lists(schema)}
    assert found == {
        ("rows", "items"),
        ("rows[].tags", None),
        ("by_name.*", None),
        ("<oneOf0>.other", "never"),
    }


def test_the_walk_finds_the_lists_of_every_tool():
    # A walk that found nothing would pass every test below.
    found = {(name, _unbranched(item.path)) for name, item in _lists()}
    for expected in (
        ("jobs.handle_jobs", "runs.items[].diagnostics"),
        ("experiments.handle_run_experiments", "runs.items"),
        ("verify.handle_verify_circuit", "findings"),
        ("analyze.handle_analyze_results", "results.*.per_run.items[].value.failed_measurements"),
        ("inspect_tools.handle_inspect", "results[].data.measurements"),
        ("analysis.handle_plot_waveform", "traces"),
        ("inspect_tools.handle_trace_net", "pins"),
    ):
        assert expected in found, sorted(found)


# ---------------------------------------------------------------------------
# 1. Every list is declared
# ---------------------------------------------------------------------------


def test_every_list_in_every_reply_declares_how_it_may_be_cut():
    exempt = _made_in_schematic_edit()
    undeclared = [
        f"{name}: {item.path}"
        for name, item in _lists()
        if item.kind is None and not (name == _EXEMPT_HANDLER and id(item.schema) in exempt)
    ]
    assert not undeclared, (
        f"Declare how each of these lists may be cut ({CUT}: never, items, rows(...) or "
        f"series(...), lib/reply_cuts.py), on the shared schema where it has one: {undeclared}"
    )


def test_the_exemption_covers_only_the_schematic_editor_lists():
    exempt = _made_in_schematic_edit()
    shared = [
        f"{name}: {item.path}"
        for name, item in _lists()
        if name != _EXEMPT_HANDLER and id(item.schema) in exempt
    ]
    assert not shared
    # The exemption is in use: when it is not, delete it.
    assert any(
        name == _EXEMPT_HANDLER and item.kind is None and id(item.schema) in exempt
        for name, item in _lists()
    )


def test_a_list_of_rows_names_key_fields_its_rows_carry():
    wrong = []
    for name, item in _lists():
        if item.kind != "rows":
            continue
        row = item.row
        properties = row.get("properties", {})
        required = set(row.get("required", ()))
        for key in item.keys:
            if key not in properties:
                wrong.append(f"{name}: {item.path} names {key!r}, which its rows do not declare")
            elif key not in required and item.projected_by is None:
                wrong.append(
                    f"{name}: {item.path} names {key!r}, which its rows do not require; "
                    "require it, or name the argument that projects the row (projected_by)"
                )
    assert not wrong, wrong


def test_lists_of_one_series_sit_in_one_object():
    # A series is thinned together, so its lists are siblings of one reply object.
    parents: dict[tuple[str, str], set[str]] = {}
    for name, item in _lists():
        if item.kind == "series":
            assert item.series is not None
            parents.setdefault((name, item.series), set()).add(item.path.rpartition(".")[0])
    assert all(len(found) == 1 for found in parents.values()), parents


# ---------------------------------------------------------------------------
# 2. What bounds a list sent whole
# ---------------------------------------------------------------------------

_UNBOUNDED = "unbounded: no cap where it is made"

#: Each list declared ``never``, by handler and path (union branches left out),
#: with what bounds it where it is made. A list with no bound says so; it is
#: listed to be bounded, not cut.
NEVER_BOUNDS: dict[tuple[str, str], str] = {}


def test_every_list_sent_whole_states_its_bound():
    declared = {(name, _unbranched(item.path)) for name, item in _lists() if item.kind == "never"}
    missing = sorted(declared - NEVER_BOUNDS.keys())
    stale = sorted(NEVER_BOUNDS.keys() - declared)
    assert not missing, f"State what bounds each list sent whole: {missing}"
    assert not stale, f"These are no longer lists sent whole: {stale}"


# ---------------------------------------------------------------------------
# 3. Agreement with the heuristic
# ---------------------------------------------------------------------------

#: What the inspect page counters count: the data lists an item pages.
_INSPECT_PAGED = tuple(
    f"results[].data.{key}"
    for key in (
        "docs",
        "sections",
        "designs",
        "only_in_window",
        "only_in_file",
        "plots",
        "signals",
        "table",
        "native_tables",
        "measurements",
    )
)

#: Each place the heuristic finds a cut (an object with a ``truncated`` flag, or
#: a count of what a list left out), with the lists it cuts.
CUT_TODAY: dict[tuple[str, str], tuple[str, ...]] = {
    ("inspect_tools.handle_inspect", "results[].data.designs[]:difference_omitted"): (
        "results[].data.designs[].only_in_window",
        "results[].data.designs[].only_in_file",
    ),
    ("inspect_tools.handle_inspect", "results[].page"): _INSPECT_PAGED,
    ("inspect_tools.handle_inspect", "results[].page.collections.*"): _INSPECT_PAGED,
    ("verify.handle_verify_circuit", "findings_omitted.*:omitted"): ("findings",),
    ("experiments.handle_run_experiments", "runs"): ("runs.items",),
    ("schematic_edit.handle_edit_schematic", "wiring.label_only_pins"): (
        "wiring.label_only_pins.items",
    ),
    ("schematic_edit.handle_edit_schematic", "views.touched"): ("views.touched.items",),
    ("schematic_edit.handle_edit_schematic", "views.pin_legend"): ("views.pin_legend.items",),
    ("schematic_edit.handle_edit_schematic", "views.preexisting"): ("views.preexisting.items",),
    ("analysis.handle_plot_waveform", "<root>:traces_truncated"): ("traces",),
    # What run_code cuts is its captured output, a string, not a list.
    ("run_code.handle_run_code", "<root>"): (),
    ("analyze.handle_analyze_results", "coverage.missing_cases"): (
        "coverage.missing_cases.items",
    ),
    ("analyze.handle_analyze_results", "results.*.spec.fail_cases"): (
        "results.*.spec.fail_cases.items",
    ),
    ("analyze.handle_analyze_results", "results.*.per_run"): ("results.*.per_run.items",),
    **{
        ("jobs.handle_jobs", f"<oneOf{branch}>.runs"): (f"<oneOf{branch}>.runs.items",)
        for branch in (0, 1, 5)
    },
    **{
        ("jobs.handle_jobs", f"<oneOf{branch}>"): (f"<oneOf{branch}>.items",)
        for branch in (2, 3, 4)
    },
}


def test_the_lists_cut_today_are_named():
    found = {(name, path) for name, schema in _schemas().items() for path, _ in _cut_lists(schema)}
    assert found == CUT_TODAY.keys(), (
        f"Name the lists each cut cuts: new {sorted(found - CUT_TODAY.keys())}, "
        f"gone {sorted(CUT_TODAY.keys() - found)}"
    )


def test_a_list_cut_today_is_declared_cuttable():
    exempt = _made_in_schematic_edit()
    lists = {(name, item.path): item for name, item in _lists()}
    wrong = []
    for (name, _), paths in CUT_TODAY.items():
        for path in paths:
            item = lists.get((name, path))
            if item is None:
                wrong.append(f"{name}: {path} is not a list")
            elif item.kind is None and id(item.schema) in exempt:
                continue
            elif item.kind not in ("items", "rows", "series"):
                wrong.append(f"{name}: {path} is cut today and declared {item.kind}")
    assert not wrong, wrong


def test_a_list_sent_whole_carries_no_sign_of_a_cut():
    # A count of what a list left out is named for it (rows_omitted,
    # rows_truncated), or sits in its entries.
    wrong = []
    for name, schema in _schemas().items():
        for item in schema_lists(schema):
            if item.kind != "never":
                continue
            field = item.path.rpartition(".")[2].removesuffix("[]")
            siblings = (item.parent or {}).get("properties", {})
            for sign in (f"{field}_omitted", f"{field}_truncated"):
                if sign in siblings:
                    wrong.append(f"{name}: {item.path} has {sign}")
            entry = item.row.get("properties", {})
            if "truncated" in entry or "omitted" in entry:
                wrong.append(f"{name}: {item.path} entries say they were cut")
    assert not wrong, wrong


# ---------------------------------------------------------------------------
# 4. Only the reply layer knows about reply size
# ---------------------------------------------------------------------------

#: The modules whose code is the reply-size machinery. ``cursor_codec`` also
#: holds the canonical hash, so only its cursor names count.
_MACHINERY_MODULES = frozenset({"response_budget", "pagination", "cursor_codec"})
_WHOLLY_MACHINERY = frozenset({"response_budget", "pagination"})

#: Names of the machinery: the response budget, the shared page schema, the
#: paginator in ``lib/pin_legend.py``, and the cursor functions of
#: ``lib/result_store.py`` and ``lib/cursor_codec.py``.
_MACHINERY_NAMES = frozenset(
    {
        "ResponseBudget",
        "resolve_response_budget",
        "page_schema",
        "PageCursorError",
        "encode_page_cursor",
        "decode_page_cursor",
        "encode_pair_cursor",
        "decode_pair_cursor",
        "paginate_view",
        "paginate_pair",
        "encode_cursor",
        "decode_cursor",
        "reencode_cursor",
        "cursor_result_set_id",
        "cursor_view",
        "CursorError",
    }
)

#: Names a cursor or a reply budget goes by, as a parameter, an argument, a
#: field read or a variable. Matched whole: ``analysis_budget_s`` and the
#: parser's memory budgets are other things.
_CURSOR_NAMES = frozenset({"cursor", "next_cursor", "view_cursors", "findings_cursor", "budget"})

#: The tool modules of the six operations; ``lib/`` is scanned whole.
_OPERATION_MODULES = (
    "tools/experiments.py",
    "tools/receipts.py",
    "tools/jobs.py",
    "tools/analyze.py",
    "tools/inspect_tools.py",
    "tools/schematic_edit.py",
    "tools/verify.py",
)

#: The machinery itself, which is the reply layer's until it moves there.
_MACHINERY_SCOPES = frozenset(
    {
        *(f"lib/pin_legend.py:{name}" for name in _MACHINERY_NAMES),
        "lib/pin_legend.py:_window",
        *(f"lib/result_store.py:{name}" for name in _MACHINERY_NAMES),
        "lib/result_store.py:_decode_cursor_body",
    }
)


def _mentions(tree: ast.AST) -> dict[str, list[str]]:
    """Each scope of ``tree`` that mentions the reply-size machinery, with
    what it mentions."""
    found: dict[str, list[str]] = {}
    for scope, nodes in scopes(tree):
        said: list[str] = []
        for node in nodes:
            if isinstance(node, ast.ImportFrom):
                module = (node.module or "").rsplit(".", 1)[-1]
                if module in _WHOLLY_MACHINERY:
                    said.append(f"import from {module}")
                said.extend(
                    f"import {alias.name}"
                    for alias in node.names
                    if alias.name in _MACHINERY_NAMES | _MACHINERY_MODULES
                )
            elif isinstance(node, ast.Import):
                said.extend(
                    f"import {alias.name}"
                    for alias in node.names
                    if alias.name.rsplit(".", 1)[-1] in _MACHINERY_MODULES
                )
            elif isinstance(node, ast.Name) and node.id in (
                _MACHINERY_NAMES | _WHOLLY_MACHINERY | _CURSOR_NAMES
            ):
                said.append(node.id)
            elif isinstance(node, ast.Attribute) and node.attr in (
                _MACHINERY_NAMES | _CURSOR_NAMES
            ):
                said.append(f".{node.attr}")
            elif isinstance(node, ast.arg) and node.arg in _CURSOR_NAMES:
                said.append(f"parameter {node.arg}")
            elif isinstance(node, ast.keyword) and node.arg in _CURSOR_NAMES:
                said.append(f"{node.arg}=")
        if said:
            found[scope] = sorted(set(said))
    return found


def _scanned() -> dict[str, ast.Module]:
    paths = [
        *sorted((PACKAGE / "lib").rglob("*.py")),
        *(PACKAGE / name for name in _OPERATION_MODULES),
    ]
    return {
        path.relative_to(PACKAGE).as_posix(): parsed(path)
        for path in paths
        if path.stem not in _MACHINERY_MODULES
    }


#: Functions where a name the scan matches means something else.
_OTHER_MEANINGS: dict[str, str] = {
    "lib/parser_capture.py:_copy_file": "budget: the bytes the parser may copy",
    "lib/parser_capture.py:capture_inputs": "budget: the bytes the parser may copy",
    "lib/raw_header.py:_Scanner.line": "budget: the bytes a header line may take",
    "lib/spice_lex_views.py:_runs_without": "cursor: a position in a netlist line",
}


def _sites() -> dict[str, list[str]]:
    sites: dict[str, list[str]] = {}
    for module, tree in _scanned().items():
        for scope, said in _mentions(tree).items():
            site = f"{module}:{scope}"
            if site not in _MACHINERY_SCOPES and site not in _OTHER_MEANINGS:
                sites[site] = said
    return sites


#: Each function that knows about cursors or reply budgets today, with the tool
#: whose move onto the reply layer takes it off this list. The list only
#: shrinks.
REPLY_SIZE_SITES: dict[str, str] = {}


def test_only_the_reply_layer_knows_about_reply_size():
    sites = _sites()
    unlisted = {site: said for site, said in sites.items() if site not in REPLY_SIZE_SITES}
    assert not unlisted, (
        "An operation returns its whole answer; cutting it, cursors and reply budgets "
        f"are the reply layer's (tools/_base.py): {unlisted}"
    )


def test_the_allowed_sites_still_need_their_place():
    stale = sorted(set(REPLY_SIZE_SITES) - set(_sites()))
    assert not stale, f"These no longer mention reply size; take them off the list: {stale}"
    found = {
        f"{module}:{scope}" for module, tree in _scanned().items() for scope in _mentions(tree)
    }
    assert not set(_OTHER_MEANINGS) - found


def test_the_scan_catches_a_planted_mention():
    planted = ast.parse(
        "from ltspice_mcp.lib.pagination import page\n"
        "from ltspice_mcp.lib import response_budget\n"
        "from ltspice_mcp.lib.cursor_codec import canonical_hash\n"
        "def measure(rows, cursor=None):\n"
        "    return rows\n"
        "def trim(data):\n"
        "    return response_budget.apply_trim(data, ['a'])\n"
        "def shape():\n"
        "    return page_schema()\n"
        "def read(query):\n"
        "    return query.findings_cursor\n"
        "def call(run):\n"
        "    return run(budget=4000)\n"
        "def other(analysis_budget_s, memory_budget_bytes):\n"
        "    return analysis_budget_s\n"
    )
    found = _mentions(planted)
    assert found == {
        "<module>": ["import from pagination", "import response_budget"],
        "measure": ["parameter cursor"],
        "trim": ["response_budget"],
        "shape": ["page_schema"],
        "read": [".findings_cursor"],
        "call": ["budget="],
    }


# ---------------------------------------------------------------------------
# 5. One writer of the sentence that says how to read the rest
# ---------------------------------------------------------------------------


def test_the_call_is_written_one_way_with_its_python_form():
    assert call_to_continue("jobs", {"action": "runs", "job_id": "j1", "cursor": "c2"}) == (
        'jobs(action="runs", job_id="j1", cursor="c2") '
        '(Python: api.jobs(action="runs", job_id="j1", cursor="c2", raw_page=True))'
    )
    resume = {"continue": {"result_set_id": "rs1", "cursor": Unfilled("next.cursor")}}
    assert mcp_call("analyze_results", resume) == (
        'analyze_results(continue={"result_set_id": "rs1", "cursor": <next.cursor>})'
    )
    assert python_call("analyze_results", resume) == (
        'api.analyze_results(raw_page=True, **{"continue": '
        '{"result_set_id": "rs1", "cursor": <next.cursor>}})'
    )
    queries = {"queries": [{"kind": "net", "cursor": "n1", "strict": True, "limit": None}]}
    assert python_call("inspect", queries) == (
        'api.inspect(queries=[{"kind": "net", "cursor": "n1", "strict": True, '
        '"limit": None}], raw_page=True)'
    )


#: An argument that continues a cut reply, written into a call.
_CONTINUATION_ARGUMENT = re.compile(
    r"(?<![\w.])(?:cursor|continue|findings_cursor|result_set_id)=(?!=)"
)

#: Where a continuation sentence may be written by hand: the writer itself, and
#: the schematic editor, whose reply text is pinned until it moves.
_WRITERS = frozenset({"lib/continuation.py", "tools/schematic_edit.py"})


def _docstrings(tree: ast.AST) -> set[int]:
    ids: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and node.body
        ):
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def _hand_written(tree: ast.AST) -> list[str]:
    """Each string in ``tree``, not a docstring, that writes a continuation
    argument into a call."""
    skip = _docstrings(tree)
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if isinstance(part, ast.Constant) else "{}" for part in node.values
            )
            for part in node.values:
                skip.add(id(part))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in skip:
                continue
            text = node.value
        else:
            continue
        if _CONTINUATION_ARGUMENT.search(text):
            found.append(f"line {node.lineno}: {text[:80]!r}")
    return found


def test_no_continuation_sentence_is_written_by_hand():
    written = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        name = path.relative_to(PACKAGE).as_posix()
        if name not in _WRITERS and (found := _hand_written(parsed(path))):
            written[name] = found
    assert not written, (
        f"Write the call that reads the rest with lib/continuation.call_to_continue: {written}"
    )


def test_the_sentence_scan_catches_a_planted_sentence():
    planted = ast.parse(
        "def hint(job_id, cursor):\n"
        '    """Not this: cursor=c in a docstring."""\n'
        "    a = f\"continue with jobs(action='runs', job_id={job_id!r}, cursor={cursor!r}).\"\n"
        '    b = "call analyze_results with continue={result_set_id, cursor} from next"\n'
        '    c = "the cursor is stale"\n'
        "    return a, b, c\n"
    )
    found = _hand_written(planted)
    assert [line.split(":")[0] for line in found] == ["line 3", "line 4"]
