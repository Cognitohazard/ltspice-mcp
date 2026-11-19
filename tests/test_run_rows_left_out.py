"""What a reply says of the run rows it left out.

A fact about a run's outcome is never cut: when a reply about a job holds
fewer run rows than the job has, ``runs_not_shown`` names, per kind of fact,
the runs after the ones it holds that carry a diagnostic, a measurement that
gave no value, a measurement or diagnostic not read, or a status other than
produced, with a cursor that reads each one's row in one call. A reply with no
target keeps a cap of 50 run rows; the Python API returns them all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib import experiment_runner, now
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import experiments as experiments_module
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, handle_jobs
from ltspice_mcp.tools.receipts import RUN_EXPERIMENTS_OUTPUT_SCHEMA
from tests.conftest import (
    LIVENESS_S,
    SyncApi,
    make_experiment_job,
    recorded_fixture_simulator,
    submit_experiment,
)

_TIMESTEP = {
    "diagnostics": [
        {
            "code": "timestep_too_small",
            "count": 1,
            "first_text": "Timestep too small",
            "channel": "log",
        }
    ]
}

_GRID_DECK = "* grid\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"


async def _jobs(state: SessionState, **values: Any) -> dict[str, Any]:
    result = await handle_jobs(JobsInput.model_validate(values), state)
    assert result.structured_content is not None
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(result.structured_content)
    return result.structured_content


def _ended(job: Any) -> None:
    """Every case of ``job`` ran, said nothing and gave every measurement."""
    for case in job.cases:
        case.submitted_at = now()
        case.log_diagnostics = {}
        case.measurement_facts = {}


def _job_with_one_diagnosed_run(state: SessionState, *, count: int, at: int) -> str:
    """A finished job whose run ``at`` alone has a diagnostic, and whose
    other notes take about 3000 tokens, as an attached analysis would."""
    job = make_experiment_job(state, job_id=f"exp_left_{count}", count=count)
    _ended(job)
    job.cases[at].log_diagnostics = dict(_TIMESTEP)
    job.observations = [
        {"code": f"note_{index}", "kind": "provenance", "detail": "x" * 400} for index in range(30)
    ]
    return job.job_id


def _named(data: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    """The runs ``runs_not_shown`` names for ``kind``."""
    runs = data.get("runs_not_shown", {}).get("runs", [])
    return [run for run in runs if kind in run["kinds"]]


async def _row_by_cursor(state: SessionState, job_id: str, cursor: str) -> dict[str, Any]:
    page = await _jobs(state, action="runs", job_id=job_id, cursor=cursor)
    return page["items"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["status", "wait", "runs"])
async def test_a_diagnostic_on_a_run_left_out_is_named_under_the_default(
    state_no_sim: SessionState, delivery: str
):
    # A runs page carries no observations, so its rows alone must be over.
    count, at = (150, 141) if delivery == "runs" else (45, 41)
    job_id = _job_with_one_diagnosed_run(state_no_sim, count=count, at=at)
    extra = {"timeout_s": 0} if delivery == "wait" else {}
    data = await _jobs(state_no_sim, action=delivery, job_id=job_id, **extra)

    rows = data["runs"]["items"] if delivery != "runs" else data["items"]
    assert len(rows) < at, "the premise: the default target cuts the rows"
    name = f"case-{at:04d}"
    assert all(row["case_id"] != name for row in rows)
    assert data["runs_not_shown"]["counts"] == {"diagnostics.timestep_too_small": 1}
    (named,) = _named(data, "diagnostics.timestep_too_small")
    assert (named["case_id"], named["run_index"]) == (name, at)
    row = await _row_by_cursor(state_no_sim, job_id, named["cursor"])
    assert row["case_id"] == name
    assert row["diagnostics"] == _TIMESTEP["diagnostics"]


@pytest.mark.asyncio
async def test_a_small_budget_names_the_one_run_with_a_failed_measurement(
    state_no_sim: SessionState,
):
    job = make_experiment_job(state_no_sim, job_id="exp_left_20", count=20)
    _ended(job)
    job.cases[17].log_diagnostics = dict(_TIMESTEP)
    job.cases[17].measurement_facts = {"failed": ["vmax"]}

    for delivery in ("status", "wait"):
        extra = {"timeout_s": 0} if delivery == "wait" else {}
        data = await _jobs(state_no_sim, action=delivery, job_id=job.job_id, budget=800, **extra)
        assert len(data["runs"]["items"]) < 18, delivery
        for kind in ("failed_measurements", "diagnostics.timestep_too_small"):
            (named,) = _named(data, kind)
            assert named["case_id"] == "case-0017", (delivery, kind)
        assert set(data["runs_not_shown"]["counts"]) == {
            "failed_measurements",
            "diagnostics.timestep_too_small",
        }


@pytest.mark.asyncio
async def test_a_cut_of_clean_rows_names_nothing(state_no_sim: SessionState):
    job = make_experiment_job(state_no_sim, job_id="exp_clean", count=60)
    _ended(job)
    data = await _jobs(state_no_sim, action="status", job_id=job.job_id, budget=600)
    assert "cut" in data
    assert "runs_not_shown" not in data


@pytest.mark.asyncio
async def test_the_python_api_returns_every_row_and_its_facts(state_no_sim: SessionState):
    job_id = _job_with_one_diagnosed_run(state_no_sim, count=45, at=41)
    data = await asyncio.to_thread(
        lambda: SyncApi(state_no_sim).jobs(action="status", job_id=job_id)
    )
    assert len(data["runs"]["items"]) == 45
    assert data["runs"]["items"][41]["diagnostics"] == _TIMESTEP["diagnostics"]
    assert "runs_not_shown" not in data and "cut" not in data


@pytest.fixture
def grid_deck(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    recorded_fixture_simulator(monkeypatch)
    deck = work_dir / "grid.cir"
    deck.write_text(_GRID_DECK)
    return deck


def _grid(deck: Path, cases: int, request_id: str, **extra: Any) -> dict[str, Any]:
    values = [f"{index}k" for index in range(1, cases // 4 + 1)]
    return {
        "request_id": request_id,
        "circuits": [{"path": str(deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
        "variations": [
            {"kind": "assign", "assign": {"R1": values, "C1": ["1u", "2u"], "V1": ["1", "2"]}}
        ],
        **extra,
    }


@pytest.mark.asyncio
async def test_a_run_receipt_names_the_runs_it_left_out(
    state_with_sim: SessionState, grid_deck: Path
):
    """Through run_experiments: a replay of a finished 20-run job, one run of
    which ended with a diagnostic and a failed measurement."""
    payload = _grid(grid_deck, 20, "left-out-run", budget=800)
    _is_error, first = await submit_experiment(state_with_sim, payload)
    job = state_with_sim.all_jobs[first["job_id"]]
    job.cases[17].log_diagnostics = dict(_TIMESTEP)
    job.cases[17].measurement_facts = {"failed": ["vmax"]}

    _is_error, replay = await submit_experiment(state_with_sim, payload)

    assert replay["replayed"] is True
    assert len(replay["runs"]["items"]) < 18
    (named,) = _named(replay, "failed_measurements")
    assert named["case_id"] == job.cases[17].case_id
    row = await _row_by_cursor(state_with_sim, job.job_id, named["cursor"])
    assert row["failed_measurements"] == ["vmax"]


# ---------------------------------------------------------------------------
# No target: a cap of 50 run rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("delivery", ["status", "wait", "runs"])
async def test_no_target_keeps_a_cap_of_50_rows(state_no_sim: SessionState, delivery: str):
    state_no_sim.config.default_budget = 0
    job = make_experiment_job(state_no_sim, job_id="exp_uncapped", count=120)
    extra = {"timeout_s": 0} if delivery == "wait" else {}
    data = await _jobs(state_no_sim, action=delivery, job_id=job.job_id, **extra)

    pointer = "/items" if delivery == "runs" else "/runs/items"
    cut = data["cut"][pointer]
    assert (cut["total"], cut["returned"]) == (120, 50)
    page = await _jobs(state_no_sim, action="runs", job_id=job.job_id, cursor=cut["next"])
    assert page["cut"]["/items"]["offset"] == 50
    assert page["cut"]["/items"]["returned"] == 50

    whole = await asyncio.to_thread(
        lambda: SyncApi(state_no_sim).jobs(action="status", job_id=job.job_id)
    )
    assert len(whole["runs"]["items"]) == 120


@pytest.mark.asyncio
async def test_a_run_receipt_with_no_target_keeps_a_cap_of_50_rows(
    state_with_sim: SessionState, grid_deck: Path
):
    state_with_sim.config.default_budget = 0
    _is_error, data = await submit_experiment(state_with_sim, _grid(grid_deck, 60, "capped-run"))
    assert data["completeness"]["produced"] == 60
    assert data["cut"]["/runs/items"]["returned"] == 50


# ---------------------------------------------------------------------------
# A replay is sent in full
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_replay_sends_every_note_in_full(state_with_sim: SessionState, grid_deck: Path):
    """A replay by request_id is how a caller recovers a reply it lost, so it
    sends every standing note in full, whatever the session sent before."""
    payload = _grid(grid_deck, 4, "replayed-notes")
    _is_error, first = await submit_experiment(state_with_sim, payload)
    assert "path" in first["source"][0]

    _is_error, replay = await submit_experiment(state_with_sim, payload)
    assert replay["replayed"] is True
    assert "path" in replay["source"][0], "a replay recovers a lost reply: in full"

    # What it sent in full counts as sent: the next wait sends it by code.
    waited = await _jobs(state_with_sim, action="wait", job_id=first["job_id"], timeout_s=0)
    assert waited["source"][0]["said_before"] == 2


# ---------------------------------------------------------------------------
# The routing-failure receipt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_routing_failure_receipt_is_presented_like_every_receipt(
    state_with_sim: SessionState, grid_deck: Path, monkeypatch: pytest.MonkeyPatch
):
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise experiments_module.DeckStagingError(
            "artifact_route_unavailable", "No Windows-native temp directory is available."
        )

    monkeypatch.setattr(experiments_module, "resolve_experiment_paths", refuse)
    state_with_sim.config.default_budget = 0
    result = await experiments_module.handle_run_experiments(
        experiments_module.RunExperimentsInput.model_validate(_grid(grid_deck, 120, "routed")),
        state_with_sim,
    )
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(data)
    assert data["completeness"]["expanded"] == 120
    cut = data["cut"]["/runs/items"]
    assert (cut["total"], cut["returned"]) == (120, 50)
    assert data["stored_answer"]["path"] in data["hint"]
    # No tool reads a stored receipt by cursor: the reply offers none.
    assert "next" not in cut
    assert "next_cursor" not in data["runs"]


@pytest.mark.asyncio
async def test_a_routing_failure_leaves_the_failed_runs_to_its_failures(
    state_with_sim: SessionState, grid_deck: Path, monkeypatch: pytest.MonkeyPatch
):
    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise experiments_module.DeckStagingError(
            "artifact_route_unavailable", "No Windows-native temp directory is available."
        )

    monkeypatch.setattr(experiments_module, "resolve_experiment_paths", refuse)
    state_with_sim.config.default_budget = 0
    result = await experiments_module.handle_run_experiments(
        experiments_module.RunExperimentsInput.model_validate(
            _grid(grid_deck, 120, "routed-projected", run_fields=["case_id"])
        ),
        state_with_sim,
    )
    data = result.structured_content
    assert data is not None
    assert all(set(row) == {"case_id"} for row in data["runs"]["items"])
    # Every case failed, and the never-cut failures say so: no row left out
    # carries a fact they do not already name.
    assert len(data["failures"]) == 120
    assert "runs_not_shown" not in data
    assert "cursor" not in data["hint"].split("complete receipt")[0]
    assert experiment_runner is not None


@pytest.mark.asyncio
async def test_an_offset_cursor_of_an_earlier_release_says_how_to_start_again(
    state_no_sim: SessionState,
):
    job = make_experiment_job(state_no_sim, job_id="exp_old_cursor", count=3)
    for action in ("runs", "list"):
        extra = {"job_id": job.job_id} if action == "runs" else {}
        data = await _jobs(state_no_sim, action=action, cursor="o:50", **extra)
        assert data["error"]["code"] == "invalid_cursor"
        assert "no cursor starts again" in data["error"]["message"], action


@pytest.mark.asyncio
async def test_the_python_apis_own_wait_remembers_no_note(
    state_no_sim: SessionState,
):
    """The Python API polls a wait for its own use: that page reaches no
    caller, so the session's note memory keeps nothing from it."""
    job = make_experiment_job(state_no_sim, job_id="exp_api_poll", count=2)
    await asyncio.to_thread(lambda: SyncApi(state_no_sim).wait(job.job_id))
    assert len(state_no_sim.notes_sent) == 0
    data = await _jobs(state_no_sim, action="wait", job_id=job.job_id, timeout_s=0)
    assert "path" in data["source"][0]
