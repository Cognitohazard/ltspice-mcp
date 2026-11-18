"""Run receipts and job replies go through the presenter.

Design principle "A short reply, the data behind an id" (``docs/DESIGN.md``):
a receipt's run rows are cut as rows to fit the reply's target, at least one
always stays, and the rest are read back from the job's own record by the
cursor the reply's ``cut`` names. A reply that cannot meet its target says so,
and no note claims that nothing was dropped.

A standing note about a job is said in full once per session and by its code
after that; ``jobs(action="status")`` sends every note in full, and a new
session starts with nothing remembered.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from tests.conftest import (
    LIVENESS_S,
    make_experiment_job,
    recorded_fixture_simulator,
    submit_experiment,
)

_GRID_DECK = "* grid\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"

_GRID_RECIPES: list[dict[str, Any]] = [
    {"key": f"v_{at}", "metric": "value", "expr": "V(out)", "at": at}
    for at in ("100u", "500u", "900u")
]

#: What a reply may never say while a run row is left out of it.
_CLAIMS_OF_COMPLETENESS = (
    "no run, failure, or analysis fact was dropped",
    "no run or receipt was dropped",
    "nothing was dropped",
    "returned all recorded runs",
)

_RUN_ROWS = "/runs/items"


@pytest.fixture
def grid_deck(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The grid's deck, run by a stand-in simulator that hands back a recorded raw+log."""
    recorded_fixture_simulator(monkeypatch)
    deck = work_dir / "grid.cir"
    deck.write_text(_GRID_DECK)
    return deck


async def _grid_receipt(
    state: SessionState, deck: Path, cases: int, budget: int | None
) -> dict[str, Any]:
    """A completed ``cases``-case grid, with an attached analysis."""
    values = [f"{index}k" for index in range(1, cases // 4 + 1)]
    payload: dict[str, Any] = {
        "request_id": f"grid-{cases}-{budget}",
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
        "variations": [
            {"kind": "assign", "assign": {"R1": values, "C1": ["1u", "2u"], "V1": ["1", "2"]}}
        ],
        "analyze": {"recipes": _GRID_RECIPES},
    }
    if budget is not None:
        payload["budget"] = budget
    _is_error, data = await submit_experiment(state, payload)
    assert data["status"] == "completed", data["hint"]
    assert data["completeness"]["produced"] == cases
    return data


async def _jobs(state: SessionState, **values: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(values), state)
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


def _observations(data: dict[str, Any], code: str) -> list[dict[str, Any]]:
    return [item for item in data.get("observations", []) if item.get("code") == code]


def _said(data: dict[str, Any]) -> str:
    """Everything a reply says in words: its hint and every observation."""
    notes = [str(item.get("detail", "")) for item in data.get("observations", [])]
    return " ".join([str(data.get("hint", "")), *notes]).lower()


# ---------------------------------------------------------------------------
# A small budget keeps a run row, or says it could not
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["run", "status", "wait"])
async def test_a_receipt_under_600_keeps_a_run_row_or_says_the_target_was_not_met(
    state_with_sim: SessionState, grid_deck: Path, delivery: str
):
    receipt = await _grid_receipt(state_with_sim, grid_deck, 64, 600)
    data = (
        receipt
        if delivery == "run"
        else await _jobs(state_with_sim, action=delivery, job_id=receipt["job_id"], budget=600)
    )

    rows = data["runs"].get("items", [])
    not_met = _observations(data, "budget_not_met")
    assert rows or not_met, "a receipt with no run row must say its target was not met"
    assert len(not_met) <= 1, "the target is said to be missed once"
    assert len(rows) >= 1, "at least one run row always stays"
    said = _said(data)
    for claim in _CLAIMS_OF_COMPLETENESS:
        assert claim not in said, claim
    # The cut says what is left out and how to read it.
    cut = data["cut"][_RUN_ROWS]
    assert cut["total"] == 64
    assert cut["returned"] == len(rows) < 64
    assert cut["next"]
    assert data["job_id"] in data["hint"]


# ---------------------------------------------------------------------------
# Reading every page rebuilds the job's table
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_page_of_run_rows_rebuilds_the_jobs_table(
    state_with_sim: SessionState, grid_deck: Path
):
    receipt = await _grid_receipt(state_with_sim, grid_deck, 64, 600)
    job_id = receipt["job_id"]
    state_with_sim.config.default_budget = 0
    whole = await _jobs(state_with_sim, action="runs", job_id=job_id)
    assert "cut" not in whole, "with no target the table is sent whole"
    assert len(whole["items"]) == 64

    rows = list(receipt["runs"]["items"])
    cursor = receipt["cut"][_RUN_ROWS].get("next")
    pages = 0
    while cursor is not None:
        page = await _jobs(state_with_sim, action="runs", job_id=job_id, cursor=cursor, budget=600)
        entry = page["cut"]["/items"]
        assert entry["offset"] == len(rows)
        assert page["items"], "every page holds at least one row"
        rows.extend(page["items"])
        cursor = entry.get("next")
        pages += 1
        assert pages <= 64
    assert rows == whole["items"]


# ---------------------------------------------------------------------------
# A standing note is said once per session
# ---------------------------------------------------------------------------

_STANDING = [
    {
        "code": "micro_sign_folded",
        "kind": "staging",
        "detail": "The deck spelled a micro sign as 'u' in 3 values; it was read as micro.",
    },
    {
        "code": "ngspice_mode_unverified",
        "kind": "provenance",
        "detail": "The run's ngspice compatibility mode could not be read from its log.",
    },
    {
        "code": "results_not_beside_sheet",
        "kind": "provenance",
        "detail": "The results of amp.asc are in the job folder; nothing was written beside it.",
    },
]


def _running_job(state: SessionState) -> str:
    """A running job this session does not run, with the standing notes of
    a job and a request whose source is listed. With job records not kept, a
    wait on it answers at once."""
    job = make_experiment_job(state, job_id="exp_standing", count=3, status="running")
    job.observations = [dict(note) for note in _STANDING]
    return job.job_id


@pytest.mark.asyncio
async def test_a_wait_says_a_standing_note_in_full_once_and_by_its_code_after(
    state_no_sim: SessionState, config
):
    job_id = _running_job(state_no_sim)

    first = await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)
    for note in _STANDING:
        (sent,) = _observations(first, note["code"])
        assert sent["detail"] == note["detail"]
    assert "said_before" not in first["source"][0]
    assert "path" in first["source"][0]

    second = await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)
    for note in _STANDING:
        (sent,) = _observations(second, note["code"])
        assert "detail" not in sent
        assert sent["said_before"] == 1
    assert second["source"][0] == {"circuit": "dut", "said_before": 1}
    assert 'jobs(action="status", job_id="exp_standing")' in second["hint"]
    # What changes between replies is never shortened.
    assert second["progress"] == first["progress"]
    assert second["runs"]["items"] == first["runs"]["items"]

    status = await _jobs(state_no_sim, action="status", job_id=job_id)
    for note in _STANDING:
        (sent,) = _observations(status, note["code"])
        assert sent["detail"] == note["detail"]
    assert "path" in status["source"][0]

    fresh = SessionState.create(config, available={})
    make_experiment_job(fresh, job_id=job_id, count=3, status="running").observations = [
        dict(note) for note in _STANDING
    ]
    anew = await _jobs(fresh, action="wait", job_id=job_id, timeout_s=0)
    for note in _STANDING:
        (sent,) = _observations(anew, note["code"])
        assert sent["detail"] == note["detail"]


@pytest.mark.asyncio
async def test_a_note_whose_text_changed_is_new(state_no_sim: SessionState):
    job_id = _running_job(state_no_sim)
    await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)
    job = state_no_sim.all_jobs[job_id]
    changed = dict(_STANDING[2], detail="The results of amp.asc and buf.asc are in the job folder.")
    job.observations = [dict(_STANDING[0]), changed]

    again = await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)

    (placement,) = _observations(again, "results_not_beside_sheet")
    assert placement["detail"] == changed["detail"]
    (micro,) = _observations(again, "micro_sign_folded")
    assert micro["said_before"] == 1
