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

import asyncio
import threading
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import now, recent
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES
from ltspice_mcp.lib.said_once import SHORTENS, NotesSent
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import jobs as jobs_module
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from ltspice_mcp.tools.receipts import DELIVERY, RUN_EXPERIMENTS_OUTPUT_SCHEMA
from tests.conftest import (
    LIVENESS_S,
    SyncApi,
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


#: What the simulator said about the first run, and the .meas it did not give.
_DIAGNOSED = {
    "diagnostics": [
        {
            "code": "timestep_too_small",
            "count": 1,
            "first_text": "Timestep too small",
            "channel": "log",
        }
    ]
}
_MEASURED = {"failed": ["vmax"]}


def _running_job(state: SessionState) -> str:
    """A running job this session does not run, with the standing notes of
    a job and a request whose source is listed, and one run that ended with
    a diagnostic and a failed measurement. With job records not kept, a wait
    on it answers at once."""
    job = make_experiment_job(state, job_id="exp_standing", count=3, status="running")
    job.observations = [dict(note) for note in _STANDING]
    ended = job.cases[0]
    ended.status = "produced"
    ended.submitted_at = now()
    ended.log_diagnostics = dict(_DIAGNOSED)
    ended.measurement_facts = dict(_MEASURED)
    job.completeness.recount(job.cases)
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
    # What is about this reply's outcome is never shortened: progress, and the
    # run rows with their diagnostics and measurement fields.
    assert second["progress"] == first["progress"]
    assert second["runs"]["items"] == first["runs"]["items"]
    row = second["runs"]["items"][0]
    assert row["diagnostics"] == _DIAGNOSED["diagnostics"]
    assert row["failed_measurements"] == ["vmax"]

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
async def test_the_route_to_the_measurements_recipe_is_said_once(state_no_sim: SessionState):
    job = make_experiment_job(state_no_sim, job_id="exp_route", count=2)
    for case in job.cases:
        case.submitted_at = now()
        case.measurement_facts = dict(_MEASURED)
    route = '"metric": "measurements"'

    first = await _jobs(state_no_sim, action="wait", job_id=job.job_id, timeout_s=0)
    second = await _jobs(state_no_sim, action="wait", job_id=job.job_id, timeout_s=0)
    status = await _jobs(state_no_sim, action="status", job_id=job.job_id)

    assert route in first["hint"]
    assert route not in second["hint"]
    assert "measurements recipe" in second["hint"]
    assert 'jobs(action="status", job_id="exp_route")' in second["hint"]
    assert route in status["hint"]
    # The rows that name the failed .meas are never shortened.
    assert [row["failed_measurements"] for row in second["runs"]["items"]] == [["vmax"]] * 2


def test_status_sends_in_full_and_a_wait_and_a_run_shorten():
    """The one table of which deliveries shorten a note sent before."""
    assert dict(SHORTENS) == {"status": False, "wait": True, "run": True, "resume": True}


@pytest.mark.asyncio
async def test_the_python_api_returns_every_note_in_full(state_no_sim: SessionState):
    job_id = _running_job(state_no_sim)
    api = SyncApi(state_no_sim)
    for _ in range(2):
        data = await asyncio.to_thread(lambda: api.jobs(action="wait", job_id=job_id, timeout_s=0))
        for note in _STANDING:
            (sent,) = _observations(data, note["code"])
            assert sent["detail"] == note["detail"]
        assert "path" in data["source"][0]
    assert len(state_no_sim.notes_sent) == 0, "the API remembers nothing it sent"


@pytest.mark.asyncio
async def test_a_note_whose_text_changed_is_new(state_no_sim: SessionState):
    job_id = _running_job(state_no_sim)
    await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)
    job = state_no_sim.all_jobs[job_id]
    changed = dict(
        _STANDING[2], detail="The results of amp.asc and buf.asc are in the job folder."
    )
    job.observations = [dict(_STANDING[0]), changed]

    again = await _jobs(state_no_sim, action="wait", job_id=job_id, timeout_s=0)

    (placement,) = _observations(again, "results_not_beside_sheet")
    assert placement["detail"] == changed["detail"]
    (micro,) = _observations(again, "micro_sign_folded")
    assert micro["said_before"] == 1


# ---------------------------------------------------------------------------
# A job's answer and its deliveries
# ---------------------------------------------------------------------------


def _delivery_fields() -> set[str]:
    """The reply fields each schema marks as the delivery's, not the answer's."""
    fields: set[str] = set()
    for schema in (RUN_EXPERIMENTS_OUTPUT_SCHEMA, *JOBS_OUTPUT_SCHEMA["oneOf"]):
        fields |= {
            name for name, node in schema["properties"].items() if node.get(DELIVERY) is True
        }
    return fields


@pytest.mark.asyncio
async def test_a_finished_jobs_deliveries_carry_one_answer(
    state_with_sim: SessionState, grid_deck: Path
):
    """The run receipt, the last wait and a status of a finished job are
    deliveries of one answer: set aside the fields the schemas mark as the
    delivery's, and they are the same. Each is read as a session's first
    reply, so every standing note is in full."""
    assert _delivery_fields() == {"action", "timed_out", "replayed"}
    receipt = await _grid_receipt(state_with_sim, grid_deck, 4, None)
    state_with_sim.notes_sent = NotesSent()
    waited = await _jobs(
        state_with_sim, action="wait", job_id=receipt["job_id"], timeout_s=LIVENESS_S
    )
    state_with_sim.notes_sent = NotesSent()
    status = await _jobs(state_with_sim, action="status", job_id=receipt["job_id"])

    answers = [
        {key: value for key, value in data.items() if key not in _delivery_fields()}
        for data in (receipt, waited, status)
    ]
    assert answers[0] == answers[1] == answers[2]


class _HeldRuns:
    """Simulator runs that end only when released: each launch is kept and
    finished by :meth:`release`, on the loop, as the stand-in finishes them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        recorded_fixture_simulator(monkeypatch)
        finish = ExperimentRunner.submit_netlist
        self.held: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        self.finish = finish
        self.launched = threading.Event()

        def held(runner: Any, *args: Any, **kwargs: Any) -> object:
            self.held.append((runner, args, kwargs))
            self.launched.set()
            return object()

        monkeypatch.setattr(ExperimentRunner, "submit_netlist", held)

    async def release(self) -> None:
        """Finish every run launched so far, once at least one was launched."""
        assert await asyncio.to_thread(self.launched.wait, LIVENESS_S)
        for runner, args, kwargs in self.held:
            self.finish(runner, *args, **kwargs)
        self.held.clear()


@pytest.mark.asyncio
async def test_a_cancelled_wait_leaves_its_job_running(
    state_with_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """A wait holds nothing that can stop the job: cancelling the call that
    waits leaves the job running, and it finishes when its runs do. A wait
    that timed out names the job and the call that waits again."""
    runs = _HeldRuns(monkeypatch)
    deck = work_dir / "held.cir"
    deck.write_text(_GRID_DECK)
    _is_error, receipt = await submit_experiment(
        state_with_sim,
        {
            "request_id": "held-wait",
            "circuits": [{"path": str(deck), "id": "dut"}],
            "execution": {"wait_s": 0},
        },
    )
    job_id = receipt["job_id"]
    assert receipt["outcome"] == "in_progress"

    timed_out = await _jobs(state_with_sim, action="wait", job_id=job_id, timeout_s=0)
    assert timed_out["timed_out"] is True
    assert timed_out["job_id"] == job_id
    assert f"jobs(action='wait', job_id='{job_id}')" in timed_out["hint"]

    registry = state_with_sim.job_registry
    entered = asyncio.Event()
    waiting = registry.wait

    async def wait_entered(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        return await waiting(*args, **kwargs)

    monkeypatch.setattr(registry, "wait", wait_entered)
    call = asyncio.create_task(
        handle_jobs(
            JobsInput.model_validate(
                {"action": "wait", "job_id": job_id, "timeout_s": LIVENESS_S}
            ),
            state_with_sim,
        )
    )
    await asyncio.wait_for(entered.wait(), LIVENESS_S)
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    monkeypatch.setattr(registry, "wait", waiting)

    assert state_with_sim.all_jobs[job_id].status not in TERMINAL_STATUSES
    await runs.release()
    done = await _jobs(state_with_sim, action="wait", job_id=job_id, timeout_s=LIVENESS_S)
    assert done["status"] == "completed"


@pytest.mark.asyncio
async def test_a_cut_receipt_and_its_pages_write_no_file(
    state_with_sim: SessionState, grid_deck: Path
):
    """Run rows are read back from the job's record, so neither a cut receipt
    nor a page of its rows stores anything, and a reply that fits writes
    nothing either."""
    receipt = await _grid_receipt(state_with_sim, grid_deck, 64, 600)
    page = await _jobs(
        state_with_sim,
        action="runs",
        job_id=receipt["job_id"],
        cursor=receipt["cut"][_RUN_ROWS]["next"],
        budget=600,
    )
    fits = await _jobs(state_with_sim, action="status", job_id=receipt["job_id"], budget=10**6)

    assert "cut" in page and "cut" not in fits
    for data in (receipt, page, fits):
        assert "stored_answer" not in data
    assert not list(state_with_sim.store.results_dir.glob("sa_*.json"))


@pytest.mark.asyncio
async def test_no_note_claims_completeness_while_a_cut_is_present(
    state_with_sim: SessionState, grid_deck: Path
):
    """Every reply about a job that holds less than all its rows names the
    job, says what is left out in ``cut`` and names the way back, and says
    nowhere that nothing was left out."""
    receipt = await _grid_receipt(state_with_sim, grid_deck, 64, 600)
    job_id = receipt["job_id"]
    replies = {
        "run": receipt,
        "status": await _jobs(state_with_sim, action="status", job_id=job_id, budget=600),
        "wait": await _jobs(state_with_sim, action="wait", job_id=job_id, budget=600),
        "runs": await _jobs(state_with_sim, action="runs", job_id=job_id, budget=600),
    }
    replies["runs page"] = await _jobs(
        state_with_sim,
        action="runs",
        job_id=job_id,
        cursor=replies["runs"]["cut"]["/items"]["next"],
        budget=600,
    )
    for name, data in replies.items():
        assert data["job_id"] == job_id, name
        assert data.get("cut"), name
        said = _said(data)
        for claim in _CLAIMS_OF_COMPLETENESS:
            assert claim not in said, (name, claim)
        for entry in data["cut"].values():
            if "next" in entry:
                assert entry["next"] in data["hint"], name


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "wait"])
async def test_a_receipt_with_no_attached_analysis_says_nothing_of_one(
    state_no_sim: SessionState, action: str
):
    """A receipt over an explicit budget whose job attached no analysis is
    cut by its rows alone: no note says an attached analysis was trimmed."""
    job = make_experiment_job(state_no_sim, job_id="exp_plain", count=3)
    job.observations = [
        {"code": f"note_{index}", "kind": "provenance", "detail": "x" * 200} for index in range(40)
    ]
    extra = {"timeout_s": 0} if action == "wait" else {}
    data = await _jobs(state_no_sim, action=action, job_id=job.job_id, budget=500, **extra)

    assert "analysis" not in data
    assert _observations(data, "budget_truncated") == []
    assert len(_observations(data, "budget_not_met")) == 1
    assert data["runs"]["items"], "at least one run row stays"


@pytest.mark.asyncio
async def test_a_list_page_is_cut_to_the_calls_budget(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
    for index in range(3):
        circuit = work_dir / f"c{index}.cir"
        circuit.write_text(_GRID_DECK)
        await asyncio.to_thread(recent.touch, circuit)
    first = await _jobs(state_no_sim, action="list", limit=1, budget=20_000)
    targets: list[Any] = []
    real = jobs_module.stored_page

    async def spy(*args: Any, **kwargs: Any) -> Any:
        targets.append(kwargs.get("target"))
        return await real(*args, **kwargs)

    monkeypatch.setattr(jobs_module, "stored_page", spy)
    second = await _jobs(
        state_no_sim, action="list", limit=1, budget=20_000, cursor=first["next_cursor"]
    )

    assert targets == [20_000]
    assert second["cut"]["/items"]["offset"] == 1
    assert second["items"][0] != first["items"][0]
