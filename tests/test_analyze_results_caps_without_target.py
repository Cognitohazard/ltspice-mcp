"""With no target, every list of an MCP reply has a stated cap.

``[analysis] default_budget = 0`` and no ``budget`` is no target: the reply
is not cut by size, but no list of it is unbounded either. Each list the
presenter may cut is held at its cap (``analyze.answer_caps``), said through
``cut`` and read on with its cursor; every other list is declared never cut,
with its bound (``tests/test_reply_cuts.py``). The Python API returns all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.presenter import cuttable_lists, resolve_pointer
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import SyncApi
from tests.test_analyze_results_through_the_presenter import _analyze, _job

_RUNS = 400


def _request(job_id: str) -> dict:
    return {
        "sources": [{"job_id": job_id, "label": "dut"}],
        "recipes": [
            {
                "key": "v",
                "metric": "value",
                "expr": "V(out)",
                "at": "100u",
                "spec": {"max": 0.1},
            }
        ],
        "group_by": ["R"],
        "include": {"outliers": True, "signals_available": True, "provenance": True},
    }


@pytest.mark.asyncio
async def test_no_list_of_a_reply_without_a_target_is_unbounded(
    state_no_sim: SessionState, work_dir: Path
):
    state_no_sim.config.default_budget = 0
    request = _request(_job(state_no_sim, work_dir, _RUNS))
    data = await _analyze(state_no_sim, request)
    lists = cuttable_lists(data, analyze_mod.OUTPUT_SCHEMA)
    assert {pointer for pointer, _ in lists} >= {
        "/source_hashes",
        "/signals_available",
        "/results/v/groups",
        "/results/v/spec/outliers",
    }
    for pointer, _kind in lists:
        shown = resolve_pointer(data, pointer)
        assert len(shown) <= analyze_mod.MAX_PAGE_SIZE, pointer
        entry = data.get("cut", {}).get(pointer)
        if entry is not None and entry["returned"] < entry["total"]:
            assert entry["next"], pointer
    assert data["cut"]["/source_hashes"]["total"] == _RUNS


def test_the_python_api_returns_every_entry(state_no_sim: SessionState, work_dir: Path):
    state_no_sim.config.default_budget = 0
    data = SyncApi(state_no_sim).analyze_results(
        **_request(_job(state_no_sim, work_dir, _RUNS, job_id="exp_api_all"))
    )
    assert len(data["source_hashes"]) == _RUNS


def test_every_list_the_schema_lets_the_presenter_cut_has_a_cap_without_a_target():
    """Over the output schema: an answer holding every list it declares (made
    from the schema) gets a cap for each one the presenter may cut; every
    other list is declared never cut, with its bound in
    ``tests/test_reply_cuts.py`` (``NEVER_BOUNDS``)."""
    from tests.test_presenter import _answer

    answer = _answer(analyze_mod.OUTPUT_SCHEMA)
    lists = {pointer for pointer, _ in cuttable_lists(answer, analyze_mod.OUTPUT_SCHEMA)}
    assert lists, "the made-up answer holds the schema's lists"
    caps = analyze_mod.answer_caps(answer, {}, target=None)
    assert lists <= set(caps), sorted(lists - set(caps))
