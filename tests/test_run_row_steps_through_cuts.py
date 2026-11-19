"""A run row's ``steps`` fact is the same in a cut reply, on its pages and in
a replayed request.

The run rows of a receipt are cut through the presenter and read on from the
stored answer; a row on a later page, and a row of the receipt a repeated
``run_experiments`` request replays, carry ``steps`` exactly as a row of the
first reply does. The runs are a recording of each LTspice build, handed back
by the suite's stand-in simulator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import replaying_simulator
from tests.test_run_row_steps import BUILDS, _jobs, _recorded

_CASES = 12
_BUDGET = 600
_STEPS = {"count": 3, "parameters": ["r"]}


async def _run(state: SessionState, payload: dict[str, Any]) -> dict[str, Any]:
    result = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    return result.structured_content


async def _every_page(
    state: SessionState, job_id: str, first: dict[str, Any]
) -> list[dict[str, Any]]:
    """The rows of ``first`` and of every page its run rows' cursor reads."""
    rows = list(first["runs"]["items"])
    cursor = first["cut"]["/runs/items"].get("next")
    while cursor is not None:
        page = await _jobs(state, action="runs", job_id=job_id, cursor=cursor, budget=_BUDGET)
        assert page["items"], "every page holds at least one row"
        rows.extend(page["items"])
        cursor = page["cut"]["/items"].get("next")
        assert len(rows) <= _CASES
    return rows


@pytest.mark.asyncio
@pytest.mark.parametrize("build", BUILDS)
async def test_a_rows_steps_are_the_same_on_every_page_and_in_a_replay(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch, build: str
):
    raw, log = _recorded(build, "step_tran")
    replaying_simulator(monkeypatch, raw, log)
    deck = work_dir / "rc.cir"
    deck.write_text("* rc\nV1 in 0 1\nR1 in out {r}\nC1 out 0 1n\n.param r=1k\n.tran 1m\n.end\n")
    payload: dict[str, Any] = {
        "request_id": f"steps-cut-{build}",
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": 60},
        "lint": "off",
        "variations": [{"kind": "assign", "assign": {"r": [f"{i + 1}k" for i in range(_CASES)]}}],
        "budget": _BUDGET,
    }
    first = await _run(state_with_sim, payload)
    job_id = first["job_id"]
    assert "/runs/items" in first["cut"], "the first reply cuts its run rows"
    assert first["runs"]["items"][0]["steps"] == _STEPS

    whole = await _jobs(state_with_sim, action="runs", job_id=job_id, budget=10**6)
    assert "cut" not in whole and len(whole["items"]) == _CASES
    assert all(row["steps"] == _STEPS for row in whole["items"]), whole["items"]

    paged = await _every_page(state_with_sim, job_id, first)
    assert paged == whole["items"], "a row on a later page is the row of the whole table"

    replay = await _run(state_with_sim, payload)
    assert replay["replayed"] is True
    assert replay["runs"]["items"] == first["runs"]["items"]
    assert await _every_page(state_with_sim, job_id, replay) == whole["items"]

    status = await _jobs(state_with_sim, action="status", job_id=job_id, budget=_BUDGET)
    assert status["runs"]["items"] == first["runs"]["items"]
