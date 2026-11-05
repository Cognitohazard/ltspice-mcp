"""A reply list cut at a cap says how many there were and where the rest are.

A count with no route leaves the caller knowing there is more and no way to
read it. Every list a reply slices at a module-level ``*_CAP`` either goes
through :func:`ltspice_mcp.lib.format.cap_note` or ``cap_list``, or is listed
here with the reason it needs neither.
"""

from __future__ import annotations

import ast
from pathlib import Path

from ltspice_mcp.errors import python_route_text
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests._source_scan import PACKAGE, parsed
from tests.conftest import stage_recorded_fixture

_NOTE_HELPERS = frozenset({"cap_note", "cap_list"})

#: Functions that slice or bound by a ``*_CAP`` without the shared note, and why
#: that is right for them.
_ALLOWED: dict[str, str] = {
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


def _capping_functions() -> dict[str, set[str]]:
    """Every function that slices or bounds by a ``*_CAP``, with the helpers it calls."""
    found: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        tree = parsed(path)
        caps = {
            target.id
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name) and target.id.endswith("_CAP")
        } | {
            alias.asname or alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
            if (alias.asname or alias.name).endswith("_CAP")
        }
        if not caps:
            continue
        where = path.relative_to(PACKAGE).as_posix()
        for function in ast.walk(tree):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            bounded = any(
                isinstance(node, (ast.Compare, ast.Slice))
                and any(isinstance(n, ast.Name) and n.id in caps for n in ast.walk(node))
                for node in ast.walk(function)
            )
            if not bounded:
                continue
            calls = {
                node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                for node in ast.walk(function)
                if isinstance(node, ast.Call)
            }
            found[f"{where}:{function.name}"] = calls
    return found


def test_every_capped_reply_list_states_its_total_and_route() -> None:
    found = _capping_functions()
    assert "tools/verify.py:render_verify_circuit" in found
    assert "tools/analyze.py:_assemble" in found
    missing = sorted(
        name for name, calls in found.items() if not calls & _NOTE_HELPERS and name not in _ALLOWED
    )
    assert not missing, (
        "these cut a list at a cap without saying the total and the route to the rest "
        "(lib/format.py cap_note or cap_list), and are not listed with a reason: "
        + ", ".join(missing)
    )
    stale = sorted(set(_ALLOWED) - set(found))
    assert not stale, "allowed entries that no longer cap anything: " + ", ".join(stale)


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
