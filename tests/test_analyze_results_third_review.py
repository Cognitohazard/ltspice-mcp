"""What a reply says when the call's bound, a full disk or a slow file ends
a step, and the one place that says whether a stored analysis is rendered.

- A step the call's bound ends is named as not done (the signal listing
  names the sources it did not read), never left out as if it found nothing.
- A stored answer never sits at its path with a time that reads as expired.
- Publishing an artifact of an answered recipe is not ended by the call's
  bound, a comparison with the file already there that cannot finish
  replaces it, and a temp left behind is removed by the next sweep.
- A reply whose answer was not kept still says its lists were cut.
- ``analysis_snapshot.classify`` alone says whether this build renders a
  stored analysis.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded
from ltspice_mcp.lib import analysis_snapshot, result_store, services
from ltspice_mcp.lib import presenter as presenter_module
from ltspice_mcp.lib.presenter import cleanup, present
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from tests.conftest import stage_recorded_fixture
from tests.test_analyze_results_call_bound import _CSV, _slow_csv
from tests.test_analyze_results_review_round import _EARLIER_RELEASE_ANALYSIS
from tests.test_analyze_results_through_the_presenter import _analyze, _job
from tests.test_presenter_unkept_answer import full_disk  # noqa: F401

# ---------------------------------------------------------------------------
# A. A step the bound ends is named
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_signal_listing_the_bound_ended_names_the_sources_it_did_not_read(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    state_no_sim.config.analysis_budget_s = 1.0
    _slow_csv(monkeypatch)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [_CSV],
            "include": {"signals_available": True},
        },
    )
    (failure,) = [f for f in data["failures"] if f.get("stage") == "signals"]
    assert failure["code"] == "analysis_deadline"
    assert failure["where"] == "signals_available"
    assert "dut" in failure["message"] and "call of its own" in failure["message"]


# ---------------------------------------------------------------------------
# C. A stored answer is never at its path with an expired time
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"rows": {"type": "array", "items": {"type": "string"}, "x-cut": "items"}},
}


def test_a_sweep_right_after_the_rename_keeps_the_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = Store(tmp_path)
    real_replace = os.replace
    swept: list[int] = []

    def replace_then_sweep(src: Any, dst: Any) -> None:
        real_replace(src, dst)
        if Path(dst).name.startswith(presenter_module.STORED_PREFIX):
            # Another request's sweep, at the worst moment.
            swept.append(cleanup(store))

    monkeypatch.setattr(os, "replace", replace_then_sweep)
    shown = present(
        {"rows": [f"row {index} " + "x" * 40 for index in range(100)]},
        _SCHEMA,
        target=300,
        tool="t",
        snapshot={},
        store=store,
        ttl_hours=1,
    )
    assert swept == [0]
    assert shown.stored is not None and shown.stored.path.exists()


# ---------------------------------------------------------------------------
# D. Publishing an answered recipe's artifact
# ---------------------------------------------------------------------------


def _csv_request(raw: Path) -> dict[str, Any]:
    return {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [_CSV]}


@pytest.mark.asyncio
async def test_a_comparison_that_cannot_finish_replaces_the_published_file(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    first = await _analyze(state_no_sim, _csv_request(raw))
    path = Path(first["results"]["csv"]["values"][0]["value"]["artifact"]["path"])
    real = services.bounded_parse

    async def slow_on_the_published_file(file: Path, *args: Any, **kwargs: Any) -> Any:
        if file == path:
            raise AnalysisDeadlineExceeded(f"Analysis work for {file.name} exceeded its deadline")
        return await real(file, *args, **kwargs)

    monkeypatch.setattr(services, "bounded_parse", slow_on_the_published_file)
    again = await _analyze(state_no_sim, _csv_request(raw))
    assert "failures" not in again, again["failures"]
    assert again["results"]["csv"]["values"][0]["value"]["artifact"]["path"] == str(path)


@pytest.mark.asyncio
async def test_an_answered_recipes_artifact_is_not_lost_to_the_calls_bound(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The recipes use the whole bound; publishing the CSV the first one
    wrote gets the publish's own deadline, not what is left of the call's."""
    state_no_sim.config.analysis_budget_s = 1.0
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    real = services.bounded_parse
    publish_deadlines: list[float] = []

    async def recorded(file: Path, *args: Any, timeout_s: float, **kwargs: Any) -> Any:
        if file.name.endswith(".pending"):
            publish_deadlines.append(timeout_s)
        return await real(file, *args, timeout_s=timeout_s, **kwargs)

    monkeypatch.setattr(services, "bounded_parse", recorded)
    first = {**_CSV, "key": "first"}
    _slow_csv(monkeypatch, after_calls=1)
    data = await _analyze(
        state_no_sim,
        {"sources": [{"path": str(raw), "label": "dut"}], "recipes": [first, _CSV]},
    )
    assert data["results"]["first"]["values"][0]["value"]["artifact"]["path"]
    assert publish_deadlines and min(publish_deadlines) >= analyze_mod.PUBLISH_DEADLINE_S


def test_a_temp_left_behind_by_a_publish_is_removed_by_the_next_sweep(work_dir: Path):
    store = Store(work_dir)
    folder = store.result_artifacts("rs_" + "0" * 32)
    folder.mkdir(parents=True)
    stale = folder / ".abc.123.def.pending"
    fresh = folder / ".abc.456.fed.pending"
    stale.write_bytes(b"partial")
    fresh.write_bytes(b"partial")
    old = time.time() - result_store.STALE_PENDING_S - 60
    os.utime(stale, (old, old))
    result_store.cleanup(work_dir)
    assert not stale.exists()
    assert fresh.exists(), "a temp a publish may still be writing is kept"


# ---------------------------------------------------------------------------
# E. A reply whose answer was not kept still says its lists were cut
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_page_object_of_an_answer_not_kept_says_rows_were_left_out(
    state_no_sim: SessionState,
    work_dir: Path,
    full_disk: None,  # noqa: F811
):
    job = _job(state_no_sim, work_dir, 40)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job, "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "100u"}],
            "include": {"per_run": {"limit": 100}},
            "budget": 800,
        },
    )
    assert data["stored_answer"]["not_kept"]
    page = data["results"]["v"]["per_run"]
    assert page["returned"] < page["total"] == 40
    assert page["truncated"] is True
    assert "next_cursor" not in page


# ---------------------------------------------------------------------------
# F. One place says whether this build renders a stored analysis
# ---------------------------------------------------------------------------


def test_classify_is_the_one_answer_to_whether_a_stored_analysis_is_rendered():
    answer = {"outcome": "complete", "results": {}, "observations": [], "failures": []}
    current = analysis_snapshot.envelope({"answer": answer, "snapshot": {}, "request": {}})
    assert analysis_snapshot.classify(current) == "snapshot"
    assert analyze_mod.stored_analysis(current).answer == answer
    assert analysis_snapshot.classify(_EARLIER_RELEASE_ANALYSIS) == "legacy"
    with pytest.raises(analyze_mod.AttachedAnalysisNotRendered):
        analyze_mod.stored_analysis(_EARLIER_RELEASE_ANALYSIS)
    # No second place reads the version.
    sources = Path(analyze_mod.__file__).read_text(encoding="utf-8")
    assert "answer_version" not in sources


@pytest.mark.asyncio
async def test_a_check_after_the_reads_the_bound_ends_names_the_rows_it_withdrew(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The check of every source after the reads runs out of time: the rows
    it could not confirm are withdrawn, and the failure says why."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    real = services.load_artifacts
    reads: list[int] = []

    async def the_check_runs_out(source: Any, state: Any, **kwargs: Any) -> Any:
        reads.append(1)
        # The read for the result set, the check before and the recipe's own
        # read answer; the check after the reads is the one that runs out.
        if len(reads) > 3:
            raise AnalysisDeadlineExceeded("the check after the reads ran out of time")
        return await real(source, state, **kwargs)

    monkeypatch.setattr(services, "load_artifacts", the_check_runs_out)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "500u"}],
        },
    )
    assert not data["results"]["v"].get("values")
    assert {f["code"] for f in data["failures"]} == {"analysis_deadline"}


# ---------------------------------------------------------------------------
# A cut page of jobs(action="list") in its new shape
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("circuits", [1, 4])
async def test_every_page_of_a_cut_list_rebuilds_it_and_matches_the_schema(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch, circuits: int
):
    import asyncio

    from ltspice_mcp.lib import recent
    from ltspice_mcp.tools.jobs import handle_jobs
    from tests.test_jobs_tool import _args, _assert_jobs_schema, _circuit

    monkeypatch.setenv("LTSPICE_MCP_HOME", str(work_dir / "recent-state"))
    for index in range(circuits):
        await asyncio.to_thread(recent.touch, _circuit(work_dir, f"c{index}.cir"))
    whole = _assert_jobs_schema(await handle_jobs(_args("list", limit=100), state_no_sim))
    page = _assert_jobs_schema(await handle_jobs(_args("list", limit=1), state_no_sim))
    read = list(page["items"])
    if circuits == 1:
        # A list of one entry is never cut: no page, no cursor.
        assert "cut" not in page and "page_of" not in page
    while page.get("next_cursor"):
        page = _assert_jobs_schema(
            await handle_jobs(_args("list", limit=1, cursor=page["next_cursor"]), state_no_sim)
        )
        assert page["page_of"] == "/items"
        assert set(page["cut"]) == {"/items"}
        read.extend(page["items"])
    assert read == whole["items"]
    # The last page says it holds the end of the list.
    assert page.get("truncated") is not True


# ---------------------------------------------------------------------------
# A measurement's stats.no_number through a cut reply, a page and a lean row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_measurements_no_number_counts_pass_through_cuts_and_pages(
    state_no_sim: SessionState, work_dir: Path
):
    """``stats.no_number`` is a map of counts, not a list: a cut reply, each
    page and the lean row of a stepped run carry it as the complete answer
    does (LTspice XVII's ``log/meas_step``, whose 0s are not established)."""
    import asyncio
    import shutil

    from tests import _ltspice_recorded as rec
    from tests.conftest import SyncApi, make_experiment_job

    log = work_dir / "case.log"
    shutil.copyfile(rec.recorded("ltspice17", "log/meas_step.log"), log)
    job = make_experiment_job(
        state_no_sim, job_id="exp_no_number", count=30, raw=work_dir / "case.raw"
    )
    request: dict[str, Any] = {
        "sources": [{"job_id": job.job_id, "label": "dut"}],
        "recipes": [{"key": "m", "metric": "measurements"}],
    }
    api = SyncApi(state_no_sim)
    whole = await asyncio.to_thread(lambda: api.analyze_results(**request))
    expected = [row["value"] for row in whole["results"]["m"]["values"]]
    assert any(
        entry.get("no_number") for value in expected for entry in value["stats"].values()
    ), "the premise: a stepped row counts steps with no number"

    first = await _analyze(state_no_sim, {**request, "budget": 1500})
    entry = first["cut"]["/results/m/values"]
    read = [row["value"] for row in first["results"]["m"]["values"]]
    cursor = entry.get("next")
    while cursor:
        page = await _analyze(
            state_no_sim, {"continue": {"result_set_id": first["result_set_id"], "cursor": cursor}}
        )
        read.extend(row["value"] for row in page["results"]["m"]["values"])
        cursor = page["cut"]["/results/m/values"].get("next")

    def counts(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # A reply leaves an empty map out; the Python API keeps it.
        return [
            {
                name: {k: v for k, v in entry.items() if v != {}}
                for name, entry in value["stats"].items()
            }
            for value in values
        ]

    assert "/results/m/values" in first["cut"], "the premise: the reply is cut"
    assert counts(read) == counts(expected)
