"""Every recipe asked for has an entry in ``results``, rows or none.

Whether ``results[key]`` existed used to depend on why the recipe had no
rows: a recipe whose only source named an unknown job kept its key, one that
failed on every run lost it, and a Python caller indexing the result as the
documented example does got a ``KeyError`` for some causes only. Now each
cause gives the same entry on the MCP page, the Python API and the attached
analysis: no rows, and, where a failure explains it, a warning pointing at it.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analyze import (
    AnalyzeResultsInput,
    capture_attached_analysis,
    handle_analyze_results,
    render_attached_analysis,
)
from tests.conftest import SyncApi, stage_recorded_fixture

_GOOD = {"key": "ok", "metric": "signal_stats", "signal": "V(out)"}


def _cases(raw: Path) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], bool]]:
    """cause -> (sources, recipes, whether a failure explains the missing rows)."""
    path_source = [{"path": str(raw), "label": "rc"}]
    return {
        "fails on every run": (
            path_source,
            [{"key": "k", "metric": "signal_stats", "signal": "V(nonexistent)"}],
            True,
        ),
        "recipe invalid": (path_source, [{"key": "k", "metric": "signal_stats"}], True),
        "only source an unknown job": (
            [{"job_id": "exp_nope_123"}],
            [{"key": "k", "metric": "signal_stats", "signal": "V(out)"}],
            False,
        ),
        # A source that names nothing is a coverage gap (missing_cases), not a
        # recipe failure, whether it names a job or a file: no failure to point at.
        "only source a missing file": (
            [{"path": str(raw.with_name("nope.raw")), "label": "gone"}],
            [{"key": "k", "metric": "signal_stats", "signal": "V(out)"}],
            False,
        ),
        "partial failure": (
            path_source,
            [_GOOD, {"key": "k", "metric": "signal_stats", "signal": "V(nonexistent)"}],
            True,
        ),
    }


_CAUSES = [
    "fails on every run",
    "recipe invalid",
    "only source an unknown job",
    "only source a missing file",
    "partial failure",
]


def _no_rows(entry: dict[str, Any], explained: bool) -> None:
    rows = entry.get("values", []) + entry.get("per_run", {}).get("items", [])
    assert rows == []
    pointers = [warning for warning in entry.get("warnings", []) if warning.startswith("No rows:")]
    assert bool(pointers) is explained, entry.get("warnings", [])


@pytest.mark.parametrize("cause", _CAUSES)
def test_a_recipe_with_no_rows_keeps_its_entry_on_every_door(
    state_no_sim: SessionState, work_dir: Path, cause: str
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    sources, recipes, explained = _cases(raw)[cause]
    request: dict[str, Any] = {"sources": sources, "recipes": recipes}
    keys = {recipe["key"] for recipe in recipes}

    # Each door on its own loop, as the Python API runs its own.
    page = asyncio.run(
        handle_analyze_results(AnalyzeResultsInput.model_validate(request), state_no_sim)
    ).structured_content
    assert page is not None
    complete = SyncApi(state_no_sim).analyze_results(**request)
    stored = asyncio.run(
        capture_attached_analysis(AnalyzeResultsInput.model_validate(request), state_no_sim)
    )
    attached = render_attached_analysis(stored, fields=None)

    for door in (page, complete, attached):
        assert set(door["results"]) == keys
        _no_rows(door["results"]["k"], explained)
    if "ok" in keys:
        assert page["results"]["ok"]["values"]
    assert page["results"]["k"].get("warnings", []) == complete["results"]["k"].get("warnings", [])
    assert page["results"]["k"].get("warnings", []) == attached["results"]["k"].get("warnings", [])


async def test_the_entries_keep_the_order_the_recipes_were_asked_in(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    recipes = [
        {"key": "first", "metric": "signal_stats"},
        _GOOD,
        {"key": "last", "metric": "signal_stats", "signal": "V(nonexistent)"},
    ]
    page = (
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {"sources": [{"path": str(raw), "label": "rc"}], "recipes": recipes}
            ),
            state_no_sim,
        )
    ).structured_content
    assert page is not None
    assert list(page["results"]) == ["first", "ok", "last"]
    assert page["outcome"] == "partial"
