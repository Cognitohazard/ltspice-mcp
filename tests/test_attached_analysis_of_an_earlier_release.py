"""A job whose attached analysis the release before this one stored.

Its record holds the analysis in that release's shape, under the same store
version. This release tells it from its own by the analysis' own version
(``analysis_snapshot.ANSWER_VERSION``, read by ``classify``), never by its shape failing, and on every route
that renders it (``run_experiments`` replaying the job, ``jobs`` status and
wait, the Python API) answers with the receipt, its status and its run rows:
only the analysis part is replaced, by a note with a code that says to run
``analyze_results`` over the job. A recovery journal written by that release,
with an attached analysis in a job it holds, loads and round-trips.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_types import AnalysisStage
from ltspice_mcp.lib.recovery_journal import (
    candidate_job,
    load_journal,
    new_root_journal,
    save_journal,
)
from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT, Store
from ltspice_mcp.lib.store import envelope as store_envelope
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import LIVENESS_S, SyncApi, recorded_fixture_simulator, submit_experiment
from tests.test_analyze_results_cut_replies import _EARLIER_RELEASE_ANALYSIS
from tests.test_recovery_records import recovery_job
from tests.test_response_budget import _GRID_DECK, _grid_receipt, _jobs

_CODE = "attached_analysis_not_rendered"


@pytest.fixture
def grid_deck(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    recorded_fixture_simulator(monkeypatch)
    deck = work_dir / "grid.cir"
    deck.write_text(_GRID_DECK)
    return deck


_RECIPES = [{"key": "v_100u", "metric": "value", "expr": "V(out)", "at": "100u"}]


def _replaced_by_the_note(data: dict[str, Any], cases: int) -> None:
    assert data["status"] == "completed"
    assert data["runs"]["total"] == cases and data["runs"]["items"]
    analysis = data["analysis"]
    assert not analysis.get("result")
    assert "analyze_results" in analysis["error"]
    (note,) = [o for o in analysis["observations"] if o["code"] == _CODE]
    assert "analyze_results" in note["detail"]


@pytest.mark.asyncio
async def test_every_route_answers_a_job_whose_analysis_an_earlier_release_stored(
    state_with_sim: SessionState, grid_deck: Path
):
    receipt = await _grid_receipt(state_with_sim, grid_deck, 4, None, recipes=_RECIPES)
    job = state_with_sim.job_registry.get_or_load(receipt["job_id"])
    assert job is not None and job.analysis is not None
    job.analysis.result = _EARLIER_RELEASE_ANALYSIS

    for action in ("status", "wait"):
        _replaced_by_the_note(
            await _jobs(state_with_sim, action=action, job_id=job.job_id), cases=4
        )
    replay = {
        "request_id": receipt["request_id"],
        "circuits": [{"path": str(grid_deck), "id": "dut"}],
        "execution": {"wait_s": LIVENESS_S},
        "variations": [
            {"kind": "assign", "assign": {"R1": ["1k"], "C1": ["1u", "2u"], "V1": ["1", "2"]}}
        ],
        "analyze": {"recipes": _RECIPES},
    }
    is_error, replayed = await submit_experiment(state_with_sim, replay)
    assert not is_error and replayed["replayed"]
    _replaced_by_the_note(replayed, cases=4)


def test_the_python_api_answers_such_a_job(state_no_sim: SessionState):
    from tests.conftest import make_experiment_job

    job = make_experiment_job(state_no_sim, job_id="exp_earlier_note", count=2)
    job.analysis = AnalysisStage(
        status="completed", request={"recipes": []}, result=_EARLIER_RELEASE_ANALYSIS
    )
    data = SyncApi(state_no_sim).jobs(action="status", job_id=job.job_id)
    _replaced_by_the_note(data, 2)


def test_an_analysis_is_told_by_its_version_not_by_its_shape():
    """A record of this shape but no version (an earlier release that wrote
    an ``answer`` key, say) is not rendered; one of this version is."""
    answer = {"outcome": "complete", "results": {}, "observations": [], "failures": []}
    unversioned = store_envelope(KIND_ANALYSIS_SNAPSHOT, answer=answer, snapshot={}, request={})
    with pytest.raises(analyze_mod.AttachedAnalysisNotRendered):
        analyze_mod.stored_analysis(unversioned)
    current = analyze_mod.attached_analysis_record(answer, {}, {})
    assert analyze_mod.stored_analysis(current).answer == answer


def test_a_journal_of_an_earlier_release_with_an_attached_analysis_round_trips(
    tmp_path: Path,
):
    job = recovery_job(tmp_path)
    assert job.recovery is not None
    job.owner_pid = job.recovery.owner.pid
    job.status = "queued"
    job.simulator_executable = job.recovery.execution.executable
    job.analysis = AnalysisStage(
        status="completed", request={"recipes": _RECIPES}, result=_EARLIER_RELEASE_ANALYSIS
    )
    store = Store(tmp_path)
    journal = new_root_journal(job)
    save_journal(store, journal)
    restored = load_journal(store, job.request_id)
    assert restored is not None and restored == journal
    candidate = candidate_job(restored.root, store)
    assert candidate.analysis is not None
    assert candidate.analysis.result == _EARLIER_RELEASE_ANALYSIS
    assert json.loads(json.dumps(experiment_store.serialize_job(candidate), default=str)) == (
        restored.root.candidate
    )
