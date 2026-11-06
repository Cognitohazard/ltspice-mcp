"""A reply list cut at a cap says how many there were and where the rest are.

A count with no route leaves the caller knowing there is more and no way to
read it. Every function that slices or compares by a module-level cap either
goes through :func:`ltspice_mcp.lib.format.cap_note` or ``cap_list``, or is
listed here with the reason it needs neither.

A cap is a module-level name bound to a whole number (a literal, arithmetic on
literals, or another such name) that either bounds a slice somewhere in the
package or is named as a bound (a ``CAP``, ``LIMIT`` or ``MAX`` part, as in
``_FAILURE_CAP``, ``JOBS_PAGE_LIMIT`` or ``MAX_PAGE_SIZE``), read by name or as
a module's attribute. A cap passed to a function that does the slicing is not
followed.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import python_route_text
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import experiments as experiments_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests._source_scan import modules
from tests.conftest import recorded_fixture_simulator, stage_recorded_fixture
from tests.test_run_experiments import (
    _args,
    _assert_schema,
    _completed_analysis_receipt,
    _status_payload,
)

_NOTE_HELPERS = frozenset({"cap_note", "cap_list"})

#: The name parts that say a whole number bounds something.
_BOUND_WORDS = frozenset({"CAP", "LIMIT", "MAX"})

#: Caps that bound no list a reply gives, and what each bounds instead.
_NOT_REPLY_LISTS: dict[str, str] = {
    "MAX_BYTES": "refuses a deck hierarchy over it",
    "MAX_CARDS": "refuses a deck hierarchy over it",
    "MAX_INSTANCES": "refuses a deck hierarchy over it",
    "MAX_INSTANCE_DEPTH": "refuses a deck hierarchy over it",
    "MAX_EXPRESSION_LENGTH": "refuses a parameter expression over it",
    "MAX_EXPRESSION_NODES": "refuses a parameter expression over it",
    "MAX_PARAMETER_DEPTH": "refuses a parameter expression over it",
    "_MAX_INCLUDE_DEPTH": "how deep an include chain is followed",
    "_MAX_DEPTH": "how deep the Python reference renders a model",
    "_MAX_FIELD_DEPTH": "how deep the reference table flattens nested arguments",
    "_MAX_NESTED_FIELDS": "where the reference table's walk stops descending",
    "SEED_MAX": "the largest seed a simulator takes",
    "KILL_MAX_PASSES": "how often a kill is retried",
    "_MAX_PROFILE_BYTES": "how much of a settings file is read",
    "_DIGEST_CAP_BYTES": "how much of an executable is hashed",
    "_LOG_READ_CAP_BYTES": "how much of a log is read",
    "_NETLIST_RESOURCE_CAP_BYTES": "how much of a netlist a resource serves; over it is refused",
    "_CHUNK_BYTES": "the size of one read from a raw file",
    "_EXPORT_MAX_ROWS": "refuses an export over it, naming it",
    "_PLOT_MAX_CELLS": "refuses a plot over it, naming it",
    "_WIDGET_MAX_BYTES": "refuses a widget over it, naming it",
    "_ANNOTATION_MAX_POINTS": "thins a curve's annotation points evenly, not items listed",
    "_RESID_UNIFORM_N": "the points a curve is resampled on",
    "RESULT_CHARS": "cuts one text, not a list",
    "_TAG_QUOTE_CHARS": "cuts one text, not a list",
    "_REPORTED_CHARS": "cuts one text, not a list",
    "_STEM_MAX_LEN": "cuts one file name, not a list",
    "_REFERENCE_DESCRIPTION_CHARS": "cuts one text, not a list",
}

#: Functions that slice or bound by a cap without the shared note, and why
#: that is right for them.
_ALLOWED: dict[str, str] = {
    "api/_detach.py:_prune_handoff_logs": "prunes old handoff logs on disk, not a reply",
    "errors.py:compact_validation_error": (
        "a refusal's sample of validation errors, ending in 'and N more'"
    ),
    "lib/recipes.py:_listed": "a refusal's sample of step values, ending in '(N in all)'",
    "tools/analyze.py:_snapshot_from_assembly": (
        "an internal snapshot; answer_total keeps the total and the cursor bases the route"
    ),
    "tools/reference_index.py:_field_entries": (
        "known exception: cuts a reference entry's fields at _MAX_FIELDS with no "
        "total or route; no branch is near the cap today"
    ),
    "lib/deck_prep.py:stage_sheet": "bounds the files staged for an export, not a reply list",
    "lib/log_parser.py:_render_file_line_family": (
        "line numbers inside one diagnostic, ending in a '(+N more)' tail"
    ),
    "lib/log_parser.py:extract_error_context": "bounds how much of a log is read",
    "lib/raw_parser.py:_read_plot_header": "bounds how much of a raw header is read",
    "lib/raw_parser.py:_ascii_plot_end": "bounds how much of an ASCII raw is scanned",
    "lib/raw_parser.py:_ascii_last_point": "bounds how much of an ASCII raw is scanned",
    "lib/runner_manager.py:_get_or_create": "bounds a cache, not a reply",
    "tools/analyze.py:_collapse_failures": "a row's sample of places; its count is the total",
    "tools/analyze.py:_no_rows_warning": "a sample of places, ending in 'and N more'",
    "tools/analyze.py:_relay_solve_failures": "a sample of runs; run_count is the total",
    "tools/analyze.py:_log_precision_observations": "a sample of runs; run_count is the total",
    "tools/inspect_tools.py:_do_reference": "names the cap in the hint's lever; slices nothing",
    "tools/jobs.py:_collect_circuit_groups": "a sample of recent jobs; recent_jobs_total is the total",
    "tools/receipts.py:_render_failures": "a row's sample of cases; its count is the total",
    "tools/verify.py:render_verify_circuit": (
        "each capped rule's findings_omitted entry states its total and carries the "
        "cursor that lists the rest, and the hint names both"
    ),
}


def _is_whole_number(node: ast.AST, known: set[str]) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int) and not isinstance(node.value, bool)
    if isinstance(node, ast.BinOp):
        return not isinstance(node.op, ast.Div) and all(
            _is_whole_number(side, known) for side in (node.left, node.right)
        )
    if isinstance(node, ast.UnaryOp):
        return _is_whole_number(node.operand, known)
    return isinstance(node, ast.Name) and node.id in known


def _names_in(node: ast.AST) -> set[str]:
    """The names ``node`` reads, bare or as a module's attribute."""
    return {
        inner.id if isinstance(inner, ast.Name) else inner.attr
        for inner in ast.walk(node)
        if isinstance(inner, ast.Name | ast.Attribute)
    }


def _caps(trees: Mapping[str, ast.Module]) -> set[str]:
    """The caps of ``trees``, as the module docstring defines them."""
    whole: set[str] = set()
    while True:
        grown = whole | {
            target.id
            for tree in trees.values()
            for node in tree.body
            if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name) and _is_whole_number(node.value, whole)
        }
        if grown == whole:
            break
        whole = grown
    slicing = {
        name
        for tree in trees.values()
        for node in ast.walk(tree)
        if isinstance(node, ast.Slice)
        for name in _names_in(node)
    }
    return {
        name for name in whole if name in slicing or _BOUND_WORDS & set(name.strip("_").split("_"))
    }


def _calls(function: ast.AST) -> set[str]:
    return {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
    }


def _capping_functions(trees: Mapping[str, ast.Module]) -> dict[str, set[str]]:
    """Every function of ``trees`` that slices or compares by a cap, with the
    names it calls. A call to a function of the same module that calls a note
    helper counts as a call to that helper: the note is then built in one place
    that another route (a stored analysis's render) can build it from too."""
    caps = _caps(trees) - set(_NOT_REPLY_LISTS)
    found: dict[str, set[str]] = {}
    for where, tree in trees.items():
        functions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        noting = {function.name for function in functions if _calls(function) & _NOTE_HELPERS}
        for function in functions:
            bounded = any(
                isinstance(node, (ast.Compare, ast.Slice)) and _names_in(node) & caps
                for node in ast.walk(function)
            )
            if not bounded:
                continue
            calls = _calls(function)
            if calls & noting:
                calls |= _NOTE_HELPERS
            found[f"{where}:{function.name}"] = calls
    return found


def _unnoted(found: Mapping[str, set[str]]) -> set[str]:
    return {name for name, calls in found.items() if not calls & _NOTE_HELPERS}


def test_every_capped_reply_list_states_its_total_and_route() -> None:
    found = _capping_functions(modules())
    assert "tools/verify.py:render_verify_circuit" in found
    assert "tools/analyze.py:_assemble" in found
    missing = sorted(_unnoted(found) - set(_ALLOWED))
    assert not missing, (
        "these cut a list at a cap without saying the total and the route to the rest "
        "(lib/format.py cap_note or cap_list), and are not listed with a reason: "
        + ", ".join(missing)
    )
    stale = sorted(set(_ALLOWED) - set(found))
    assert not stale, "allowed entries that no longer cap anything: " + ", ".join(stale)
    assert set(_NOT_REPLY_LISTS) <= _caps(modules()), "listed caps that are no longer caps"


def test_the_cap_rule_catches_what_it_names() -> None:
    planted = ast.parse(
        "_OPEN_DESIGNS_LIMIT = 50\n"
        "_SHOWN = 2 * 10\n"
        "_SHOWN_AGAIN = _SHOWN\n"
        "_ROWS_MAX = 1 << 6\n"
        "_PAGE_CAP: int = 25\n"
        "_TIMEOUT = 3.5\n"
        "_RETRIES = 3\n"
        "def limited(designs):\n"
        "    return designs[:_OPEN_DESIGNS_LIMIT]\n"
        "async def sliced(rows):\n"
        "    return rows[_SHOWN_AGAIN:]\n"
        "def compared(rows):\n"
        "    return len(rows) > _ROWS_MAX\n"
        "class Page:\n"
        "    def window(self, index, first):\n"
        "        return first <= index < first + other._PAGE_CAP\n"
        "def noted(rows):\n"
        "    return cap_note(rows[:_PAGE_CAP])\n"
        "def _note_text(rows):\n"
        "    return cap_note(len(rows))\n"
        "def delegated(rows):\n"
        "    return rows[:_PAGE_CAP], _note_text(rows)\n"
        "def _plain_text(rows):\n"
        "    return str(len(rows))\n"
        "def undelegated(rows):\n"
        "    return rows[:_PAGE_CAP], _plain_text(rows)\n"
        "def unbounded(rows, n):\n"
        "    return rows[:n], rows[:3], _TIMEOUT > 1, _RETRIES == 3\n"
    )
    found = _capping_functions({"planted.py": planted})
    assert set(found) == {
        "planted.py:" + name
        for name in (
            "limited",
            "sliced",
            "compared",
            "window",
            "noted",
            "delegated",
            "undelegated",
        )
    }
    assert _unnoted(found) == {
        "planted.py:" + name for name in ("limited", "sliced", "compared", "window", "undelegated")
    }


async def test_capped_failures_give_the_true_total_and_the_route(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    count = analyze_mod._FAILURE_CAP + 5
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw), "label": "rc"}],
            # A signal of its own per recipe, so each fails for its own reason.
            "recipes": [
                {"key": f"k{index}", "metric": "signal_stats", "signal": f"V(nx{index})"}
                for index in range(count)
            ],
        }
    )
    data = (await handle_analyze_results(args, state_no_sim)).structured_content
    assert data is not None
    assert len(data["failures"]) == analyze_mod._FAILURE_CAP
    (note,) = (o for o in data["observations"] if o["code"] == "failures_truncated")
    route = python_route_text(state_no_sim.tool_dispatch)
    assert note["detail"] == (
        f"failures: showing {analyze_mod._FAILURE_CAP} of {count} failure records; "
        f"api.analyze_results returns every record. {route} Coverage and recipe "
        "result presence still reflect the full call."
    )


async def _capped_failures_on_every_route(
    state: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[int, dict[str, dict[str, Any]]]:
    """One job's run, analysed with more failing recipes than a page lists:
    the analysis attached to run_experiments, as its receipt and jobs(status)
    show it, and analyze_results over the same job."""
    recorded_fixture_simulator(monkeypatch)
    count = analyze_mod._FAILURE_CAP + 5
    recipes = [
        {"key": f"k{index}", "metric": "signal_stats", "signal": f"V(nx{index})"}
        for index in range(count)
    ]
    deck = work_dir / "capped.cir"
    deck.write_text("* deck\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n", encoding="utf-8")
    receipt = _assert_schema(
        await _completed_analysis_receipt(
            _args(deck, "capped-failures", analyze={"recipes": recipes}), state
        )
    )
    status = await _status_payload(receipt["job_id"], state)
    # Under the label the attached analysis gives its own job, so each failure
    # names the same place.
    source = {"job_id": receipt["job_id"], "label": experiments_mod._ATTACHED_ANALYSIS_LABEL}
    direct = await handle_analyze_results(
        AnalyzeResultsInput.model_validate({"sources": [source], "recipes": recipes}), state
    )
    assert direct.structured_content is not None
    return count, {
        "analyze_results": direct.structured_content,
        "run_experiments": receipt["analysis"]["result"],
        "jobs": status["analysis"]["result"],
    }


def _truncation(data: dict[str, Any]) -> str:
    (note,) = (o for o in data["observations"] if o["code"] == "failures_truncated")
    return note["detail"]


async def test_capped_failures_give_the_same_total_on_every_route(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    count, replies = await _capped_failures_on_every_route(state_with_sim, work_dir, monkeypatch)
    for route, data in replies.items():
        assert len(data["failures"]) == analyze_mod._FAILURE_CAP, route
        assert _truncation(data).startswith(
            f"failures: showing {analyze_mod._FAILURE_CAP} of {count} failure records; "
            "api.analyze_results returns every record."
        ), route
    assert {route: data["failures"] for route, data in replies.items()} == {
        route: replies["analyze_results"]["failures"] for route in replies
    }


async def test_capped_failures_name_the_same_route_on_every_route(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _count, replies = await _capped_failures_on_every_route(state_with_sim, work_dir, monkeypatch)
    route = python_route_text(state_with_sim.tool_dispatch)
    assert {name: _truncation(data) for name, data in replies.items()} == {
        name: _truncation(replies["analyze_results"]) for name in replies
    }
    assert route in _truncation(replies["analyze_results"])
