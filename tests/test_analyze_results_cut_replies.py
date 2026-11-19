"""What a cut analysis reply says, held against the ways it went wrong.

Following a reply's ``next`` from page to page ends; a thinned waveform is
thinned for the rows a reply shows, and its ``cut`` names only those; the
Python API gets a receipt's attached analysis with every failure; a row an
observation names after a source drifted is the row it was made of; and the
part on rows left out names at most three kinds, whatever a recipe is keyed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import analysis_snapshot, not_shown
from ltspice_mcp.lib.experiment_types import AnalysisStage
from ltspice_mcp.lib.store import KIND_ANALYSIS_SNAPSHOT
from ltspice_mcp.lib.store import envelope as store_envelope
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import SyncApi, make_experiment_job
from tests.test_analyze_results_through_the_presenter import _analyze, _job


@pytest.mark.asyncio
async def test_following_next_from_page_to_page_ends_and_reads_every_list(
    state_no_sim: SessionState, work_dir: Path
):
    job = _job(state_no_sim, work_dir, 40)
    request = {
        "sources": [{"job_id": job, "label": "dut"}],
        "recipes": [
            {"key": "a", "metric": "value", "expr": "V(out)", "at": "100u"},
            {"key": "b", "metric": "value", "expr": "V(out)", "at": "900u"},
        ],
        "include": {"per_run": {"limit": 100}},
        "budget": 1500,
    }
    first = await _analyze(state_no_sim, request)
    seen: dict[str, list[Any]] = {
        key: list(first["results"][key]["per_run"].get("items", [])) for key in ("a", "b")
    }
    # The reply's next reads on in one list and ends with it; each other cut
    # list reads on from its own cut entry.
    starts = {
        key: first["cut"].get(f"/results/{key}/per_run/items", {}).get("next") for key in seen
    }
    for key, cursor in starts.items():
        reply = (
            {"next": {"result_set_id": first["result_set_id"], "cursor": cursor}} if cursor else {}
        )
        for _ in range(200):
            if "next" not in reply:
                break
            reply = await _analyze(state_no_sim, {"continue": reply["next"]})
            seen[key].extend(reply["results"][key]["per_run"].get("items", []))
        else:
            pytest.fail("following next never ended")
    assert first["next"]["cursor"] == starts["a"]
    whole = await asyncio.to_thread(
        lambda: (
            __import__("tests.conftest", fromlist=["SyncApi"])
            .SyncApi(state_no_sim)
            .analyze_results(**{key: value for key, value in request.items() if key != "budget"})
        )
    )
    for key in ("a", "b"):
        assert seen[key] == whole["results"][key]["per_run"]["items"], key


@pytest.mark.asyncio
async def test_a_series_is_thinned_for_the_rows_the_reply_shows(
    state_no_sim: SessionState, work_dir: Path
):
    request = {
        "recipes": [{"key": "w", "metric": "waveform", "signals": ["V(out)"], "max_points": 400}],
        "include": {"per_run": {"limit": 100}},
        "budget": 3000,
    }
    few = await _analyze(
        state_no_sim,
        {"sources": [{"job_id": _job(state_no_sim, work_dir, 3, "few"), "label": "d"}], **request},
    )
    many = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": _job(state_no_sim, work_dir, 30, "many"), "label": "d"}],
            **request,
        },
    )
    shown = many["results"]["w"]["per_run"]["items"]
    # Every cut entry names a list the reply holds.
    for pointer in many["cut"]:
        if pointer.endswith("/value/x"):
            index = int(pointer.split("/")[-3])
            assert index < len(shown), pointer
    # Rows the reply leaves out do not hold the rows it shows at the floor:
    # drawn for 30 rows, every row was at 32 points.
    assert len(shown[0]["value"]["x"]) > analyze_mod.THIN_FLOOR_POINTS
    assert len(few["results"]["w"]["per_run"]["items"][0]["value"]["x"]) >= len(
        shown[0]["value"]["x"]
    )


def test_the_python_api_gets_an_attached_analysis_with_every_failure(
    state_no_sim: SessionState,
):
    """Through the Python API's own door: the attached analysis is the
    complete answer, every failure one per place, as analyze_results gives it."""
    job = make_experiment_job(state_no_sim, job_id="exp_attached_failures", count=1)
    job.analysis = AnalysisStage(
        status="completed",
        request={"recipes": []},
        result=analysis_snapshot.envelope(
            {
                "answer": {
                    "outcome": "partial",
                    "results": {},
                    "observations": [],
                    "failures": [
                        {"code": "x", "stage": "s", "where": f"run-{index}", "message": "same"}
                        for index in range(3)
                    ],
                },
                "snapshot": {},
                "request": {},
            }
        ),
    )
    status = SyncApi(state_no_sim).jobs(action="status", job_id=job.job_id)
    assert len(status["analysis"]["result"]["failures"]) == 3


def test_the_part_on_rows_left_out_names_at_most_three_kinds():
    facts = not_shown.Facts(
        (
            (0, {"case_id": "a"}, ("k1", "k2")),
            (1, {"case_id": "b"}, ("k3", "k4")),
            (2, {"case_id": "c"}, ("k5",)),
        ),
        frozenset(),
    )
    part = not_shown.not_shown(
        {"g": not_shown.LeftOut(facts, 0, lambda _index: None)},
        bound=10_000,
        field="f",
        entries="rows",
    )
    named = [kind for row in part["g"]["rows"] for kind in row["kinds"]]
    assert named == ["k1", "k2", "k3"]


def test_a_recipe_keyed_like_the_folded_group_is_not_folded_into_it():
    facts = not_shown.Facts(
        tuple((index, {"case_id": f"c{index}"}, (f"k{index}",)) for index in range(40)),
        frozenset(),
    )
    groups = {
        key: not_shown.LeftOut(facts, 0, lambda _index: None) for key in ("*", "a", "b", "c")
    }
    part = not_shown.not_shown(groups, bound=300, field="f", entries="rows")
    assert part["*"]["counts"] == {f"k{index}": 1 for index in range(40)}
    assert "groups" not in part["*"]


def test_an_observations_rows_follow_the_rows_kept_after_a_source_drifted():
    def record(manifest: str) -> analyze_mod.Record:
        return analyze_mod.Record(
            manifest_id=manifest, source=manifest, identity=analyze_mod.RowIdentity(), value={}
        )

    observation = analyze_mod.Observation(
        code="c",
        kind="value",
        detail="d",
        evidence={
            "recipe": "k",
            "rows": [{"source": "A", "index": 0}, {"source": "A", "index": 2}],
        },
    )
    unit = analyze_mod.WorkUnit(
        key="k",
        recipe=analyze_mod.validate_recipe({"key": "k", "metric": "summary"}),
        records=[record("A"), record("B"), record("A")],
        failures=[],
        pending=[],
        eligible_ids={"A", "B"},
        position=0,
        observations=[observation],
    )
    analyze_mod._discard_drifted(  # pyright: ignore[reportPrivateUsage]
        [unit], [], {"B": analyze_mod.SourceFault("source_drift")}
    )
    assert [r.manifest_id for r in unit.records] == ["A", "A"]
    assert observation.evidence is not None
    assert [row["index"] for row in observation.evidence["rows"]] == [0, 1]


#: The attached analysis the release before this one stored: a snapshot of
#: its own assembly, with no complete answer in it, in the store's envelope
#: of the same store version, with no analysis version.
_EARLIER_RELEASE_ANALYSIS = store_envelope(
    KIND_ANALYSIS_SNAPSHOT,
    **{
        "top": {"outcome": "complete"},
        "results": {"v": {"values": []}},
        "missing_offset": 0,
        "coverage_cursor_base": None,
        "natural_cursor_base": None,
        "natural_has_next": False,
        "natural_deferred": False,
    },
)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["status", "wait", "runs"])
async def test_a_job_from_an_earlier_release_still_answers_with_its_runs(
    state_no_sim: SessionState, action: str
):
    """Only the attached analysis is replaced, by the note that says to run
    analyze_results over the job; the receipt and its run rows answer."""
    from ltspice_mcp.tools.jobs import JobsInput, handle_jobs

    job = make_experiment_job(state_no_sim, job_id=f"exp_earlier_{action}", count=3)
    job.analysis = AnalysisStage(
        status="completed", request={"recipes": []}, result=_EARLIER_RELEASE_ANALYSIS
    )
    result = await handle_jobs(
        JobsInput.model_validate({"action": action, "job_id": job.job_id}), state_no_sim
    )
    assert not result.is_error, result.content
    data = result.structured_content
    assert data is not None
    rows = data["items"] if action == "runs" else data["runs"]
    assert len(rows) == 3
    if action != "runs":
        analysis = data["analysis"]
        assert analysis["status"] == "completed"
        assert "result" not in analysis
        assert "analyze_results" in analysis["error"]


@pytest.mark.parametrize("action", ["status", "wait", "runs"])
def test_the_python_api_answers_a_job_from_an_earlier_release(
    state_no_sim: SessionState, action: str
):
    job = make_experiment_job(state_no_sim, job_id=f"exp_earlier_api_{action}", count=3)
    job.analysis = AnalysisStage(
        status="completed", request={"recipes": []}, result=_EARLIER_RELEASE_ANALYSIS
    )
    data = SyncApi(state_no_sim).jobs(action=action, job_id=job.job_id)
    assert len(data["items"] if action == "runs" else data["runs"]) == 3
    if action != "runs":
        assert "analyze_results" in data["analysis"]["error"]
