"""``analyze_results`` replies go through the presenter.

Design principle "A short reply, the data behind an id" (``docs/DESIGN.md``):
one call evaluates every recipe it is asked for, and the answer is sent whole
when it fits its target; a longer one is stored whole and its lists cut, each
cut named in ``cut`` with the cursor that reads the rest from the stored
answer. A page evaluates nothing, so it is never refused or emptied because a
source moved: it carries the snapshot of the sources its answer was computed
from. "A check says what it did not look at": a row left out that carries a
fact is named, with a cursor that starts at it.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.lib.presenter import resolve_pointer
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import OUTPUT_SCHEMA, AnalyzeResultsInput, handle_analyze_results
from tests import _ltspice_recorded as recorded_ltspice
from tests.conftest import (
    FIXTURES_DIR,
    LIVENESS_S,
    SyncApi,
    make_experiment_job,
    recorded_fixture_simulator,
    stage_recorded_fixture,
    submit_experiment,
)

_VALUE = {"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}
_ROWS = "/results/v/per_run/items"

_GRID_DECK = "* grid\nV1 in 0 1\nR1 in out 1k\nC1 out 0 1u\n.tran 1m\n.end\n"
_GRID_RECIPES: list[dict[str, Any]] = [
    {"key": f"v_{at}", "metric": "value", "expr": "V(out)", "at": at}
    for at in ("100u", "500u", "900u")
]

_SWEEP_LOGS = [FIXTURES_DIR / f"ltspice_sweep_meas_run{index}.log" for index in range(3)]
_SWEEP_DECK = (
    ".tran 1m\n.meas tran vfinal FIND V(out) AT=0.9m\n.meas tran tcross WHEN V(out)=0.5\n.end\n"
)
#: LTspice XVII's log of the sweep's FIND with V(out) missing from .save,
#: which it reports as FAIL'ed.
_FAILED_VFINAL = recorded_ltspice.recorded("ltspice17", "deck/save_omits_meas.log")


async def _analyze(state: SessionState, request: dict[str, Any]) -> dict[str, Any]:
    result = await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state)
    data = result.structured_content
    assert data is not None, result.content
    assert not result.is_error, data
    jsonschema.Draft202012Validator(OUTPUT_SCHEMA).validate(data)
    return data


def _job(state: SessionState, work_dir: Path, count: int, job_id: str = "exp_many") -> str:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    return make_experiment_job(state, job_id=job_id, count=count, raw=raw).job_id


def _request(job_id: str, **include: Any) -> dict[str, Any]:
    return {
        "sources": [{"job_id": job_id, "label": "dut"}],
        "recipes": [_VALUE],
        "include": {"per_run": True, **include},
    }


def _meas_job(state: SessionState, count: int, failed_at: int) -> str:
    """A finished sweep of ``count`` cases; the one at ``failed_at`` FAIL'ed vfinal."""
    job = make_experiment_job(state, job_id="exp_meas_many", count=count)
    for index, case in enumerate(job.cases):
        assert case.log_file is not None
        log = _FAILED_VFINAL if index == failed_at else _SWEEP_LOGS[index % len(_SWEEP_LOGS)]
        shutil.copy(log, case.log_file)
    job.sources[0].path.write_text(_SWEEP_DECK, encoding="utf-8")
    return job.job_id


def _observed(data: Any, code: str) -> list[dict[str, Any]]:
    """Every observation with ``code`` anywhere in ``data``."""
    found: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("code") == code and "detail" in node:
                found.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(data)
    return found


async def _every_row(state: SessionState, first: dict[str, Any], pointer: str) -> list[Any]:
    """The rows at ``pointer``: the first reply's, then every page's."""
    rows = list(resolve_pointer(first, pointer))
    entry = first["cut"][pointer]
    while "next" in entry:
        page = await _analyze(
            state,
            {"continue": {"result_set_id": first["result_set_id"], "cursor": entry["next"]}},
        )
        rows.extend(resolve_pointer(page, pointer))
        entry = page["cut"][pointer]
    return rows


# ---------------------------------------------------------------------------
# A sweep with an attached analysis
# ---------------------------------------------------------------------------


@pytest.fixture
def grid_deck(work_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    recorded_fixture_simulator(monkeypatch)
    deck = work_dir / "grid.cir"
    deck.write_text(_GRID_DECK)
    return deck


@pytest.mark.asyncio
async def test_a_sweep_of_16_cases_with_three_value_recipes_attached_meets_the_default(
    state_with_sim: SessionState, grid_deck: Path
):
    """Under the default target (4000), a receipt whose attached analysis is
    longer than the target cuts the analysis' rows with a cursor, as it cuts
    its own run rows, and does not say the target was not met."""
    values = [f"{index}k" for index in range(1, 5)]
    _is_error, data = await submit_experiment(
        state_with_sim,
        {
            "request_id": "grid-16-attached",
            "circuits": [{"path": str(grid_deck), "id": "dut"}],
            "execution": {"wait_s": LIVENESS_S},
            "variations": [
                {"kind": "assign", "assign": {"R1": values, "C1": ["1u", "2u"], "V1": ["1", "2"]}}
            ],
            "analyze": {"recipes": _GRID_RECIPES},
        },
    )
    assert data["completeness"]["produced"] == 16
    assert not _observed(data, "budget_not_met"), _observed(data, "budget_not_met")
    assert not _observed(data, "budget_truncated")
    result = data["analysis"]["result"]
    for key in ("v_100u", "v_500u", "v_900u"):
        assert result["results"][key]["metric"] == "value"
    # What the analysis cut is read on through analyze_results.
    for pointer, entry in result.get("cut", {}).items():
        if "next" in entry:
            page = await _analyze(
                state_with_sim,
                {"continue": {"result_set_id": result["result_set_id"], "cursor": entry["next"]}},
            )
            assert resolve_pointer(page, pointer)


# ---------------------------------------------------------------------------
# Reading on slices the stored answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_every_page_of_a_recipes_rows_rebuilds_them_with_no_evaluation_after_the_first(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    state_no_sim.config.default_budget = 0
    job_id = _job(state_no_sim, work_dir, 120)
    evaluated: list[str] = []
    evaluate = analyze_mod._evaluate_item

    async def counted(recipe: Any, *args: Any, **kwargs: Any) -> Any:
        evaluated.append(recipe.key)
        return await evaluate(recipe, *args, **kwargs)

    monkeypatch.setattr(analyze_mod, "_evaluate_item", counted)
    first = await _analyze(state_no_sim, _request(job_id))
    assert evaluated == ["v"]
    rows = await _every_row(state_no_sim, first, _ROWS)
    assert evaluated == ["v"], "a page evaluated a recipe again"

    whole = await asyncio.to_thread(
        lambda: SyncApi(state_no_sim).analyze_results(**_request(job_id))
    )
    assert rows == whole["results"]["v"]["per_run"]["items"]
    assert [row["case_id"] for row in rows] == [f"case-{index:04d}" for index in range(120)]


@pytest.mark.asyncio
async def test_the_per_run_cursor_reads_the_same_page_as_continue(
    state_no_sim: SessionState, work_dir: Path
):
    state_no_sim.config.default_budget = 0
    job_id = _job(state_no_sim, work_dir, 80)
    first = await _analyze(state_no_sim, _request(job_id))
    per_run = first["results"]["v"]["per_run"]
    assert per_run["truncated"] is True
    assert per_run["next_cursor"] == first["cut"][_ROWS]["next"]
    by_page_cursor = await _analyze(
        state_no_sim, _request(job_id, per_run={"cursor": per_run["next_cursor"]})
    )
    by_continue = await _analyze(
        state_no_sim,
        {"continue": {"result_set_id": first["result_set_id"], "cursor": per_run["next_cursor"]}},
    )
    assert by_page_cursor["results"]["v"]["per_run"]["items"][0]["case_id"] == "case-0050"
    assert by_page_cursor == by_continue


@pytest.mark.asyncio
async def test_with_no_target_per_run_holds_50_rows_and_says_so_through_cut(
    state_no_sim: SessionState, work_dir: Path
):
    state_no_sim.config.default_budget = 0
    data = await _analyze(state_no_sim, _request(_job(state_no_sim, work_dir, 120)))
    cut = data["cut"][_ROWS]
    assert (cut["total"], cut["returned"]) == (120, 50)
    assert len(data["results"]["v"]["per_run"]["items"]) == 50
    assert data["results"]["v"]["per_run"]["total"] == 120


@pytest.mark.asyncio
async def test_a_page_read_after_its_source_changed_returns_the_stored_rows_and_its_snapshot(
    state_no_sim: SessionState, work_dir: Path
):
    state_no_sim.config.default_budget = 0
    job_id = _job(state_no_sim, work_dir, 80)
    first = await _analyze(state_no_sim, _request(job_id))
    snapshot = first["stored_answer"]["snapshot"]
    whole = await asyncio.to_thread(
        lambda: SyncApi(state_no_sim).analyze_results(**_request(job_id))
    )

    raw = work_dir / "ltspice_tran_rc.raw"
    raw.write_bytes(raw.read_bytes()[:-64])
    page = await _analyze(
        state_no_sim,
        {
            "continue": {
                "result_set_id": first["result_set_id"],
                "cursor": first["cut"][_ROWS]["next"],
            }
        },
    )
    assert (
        page["results"]["v"]["per_run"]["items"] == whole["results"]["v"]["per_run"]["items"][50:]
    )
    assert page["stored_answer"]["snapshot"] == snapshot
    assert not [failure for failure in page["failures"] if failure["code"] == "source_drift"]


# ---------------------------------------------------------------------------
# What a reply leaves out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_row_left_out_with_a_failed_measurement_is_named_with_a_cursor_at_it(
    state_no_sim: SessionState,
):
    state_no_sim.config.default_budget = 0
    job_id = _meas_job(state_no_sim, 120, failed_at=100)
    request = {
        "sources": [{"job_id": job_id, "label": "sweep"}],
        "recipes": [{"key": "m", "metric": "measurements"}],
        "include": {"per_run": True},
    }
    data = await _analyze(state_no_sim, request)
    assert data["cut"]["/results/m/per_run/items"]["returned"] == 50
    note = data["rows_not_shown"]["m"]
    assert note["counts"]["failed_measurements"] == 1
    (named,) = [row for row in note["rows"] if "failed_measurements" in row["kinds"]]
    assert named["case_id"] == "case-0100"
    page = await _analyze(
        state_no_sim,
        {"continue": {"result_set_id": data["result_set_id"], "cursor": named["cursor"]}},
    )
    row = page["results"]["m"]["per_run"]["items"][0]
    assert row["case_id"] == "case-0100"
    assert row["value"]["failed_measurements"] == ["vfinal"]


@pytest.mark.asyncio
async def test_a_reply_under_a_budget_keeps_the_outliers_asked_for(
    state_no_sim: SessionState, work_dir: Path
):
    job_id = _job(state_no_sim, work_dir, 40)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job_id, "label": "dut"}],
            "recipes": [{**_VALUE, "spec": {"max": 0.1}}],
            "include": {"outliers": True, "signals_available": True},
            "budget": 1500,
        },
    )
    spec = data["results"]["v"]["spec"]
    assert spec["fail_count"] == 40
    assert spec["outliers"], "the budget dropped the outliers the caller asked for"
    assert data["signals_available"], "the budget dropped the signals the caller asked for"


@pytest.mark.asyncio
async def test_a_summarys_default_row_keeps_the_measurements_its_log_printed_failed(
    state_no_sim: SessionState, work_dir: Path
):
    """A field of a row's value that states a fact survives the lean row."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    shutil.copy(
        recorded_ltspice.recorded("ltspice26", "log/meas_failed.log"), raw.with_suffix(".log")
    )
    request: dict[str, Any] = {
        "sources": [{"path": str(raw), "label": "dut"}],
        "recipes": [{"key": "s", "metric": "summary"}],
    }
    (whole,) = (await _analyze(state_no_sim, {**request, "include": {"fields": ["value"]}}))[
        "results"
    ]["s"]["values"]
    assert whole["value"]["failed_measurements"] == ["never", "depends"]
    # The default row and a projection that names another field keep it.
    for include in ({}, {"fields": ["value.sim_type"]}):
        (row,) = (await _analyze(state_no_sim, {**request, "include": include}))["results"]["s"][
            "values"
        ]
        assert row["value"].get("failed_measurements") == ["never", "depends"], include
