"""What ``include.signals_available`` says of each run a call covers.

Design principle "Every check reports what's not covered" (``docs/DESIGN.md``):
every run the call covers is in ``signals_available`` or is named, with the
reason, in a failure at stage ``signals``. A run is named when its results
file is missing, when that file cannot be read, and when the call's bound
ended the listing before it was read, whether the call's recipes read the
results file or only the log.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.state import SessionState
from tests.conftest import FIXTURES_DIR, make_experiment_job, stage_recorded_fixture
from tests.test_analyze_results_call_bound import _CSV, _slow_csv, _slow_read
from tests.test_analyze_results_through_the_presenter import _SWEEP_DECK, _SWEEP_LOGS, _analyze

_VALUE = {"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}
_MEASUREMENTS = {"key": "m", "metric": "measurements"}


def _job_with_run_2(state: SessionState, damage: str) -> str:
    """A finished job of four runs, each with its own results file and log;
    run 2's results file is deleted (``missing``) or not a results file
    (``unreadable``)."""
    job = make_experiment_job(state, job_id=f"exp_signals_{damage}", count=4)
    for index, case in enumerate(job.cases):
        assert case.raw_file is not None and case.log_file is not None
        shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.raw", case.raw_file)
        shutil.copy(_SWEEP_LOGS[index % len(_SWEEP_LOGS)], case.log_file)
    job.sources[0].path.write_text(_SWEEP_DECK, encoding="utf-8")
    damaged = job.cases[2].raw_file
    assert damaged is not None
    if damage == "missing":
        damaged.unlink()
    else:
        damaged.write_bytes(b"this is not a results file\n")
    return job.job_id


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ["missing", "unreadable"])
@pytest.mark.parametrize("recipe", [_VALUE, _MEASUREMENTS], ids=["reads_raw", "reads_log"])
async def test_every_run_is_listed_or_named_with_the_reason(
    state_no_sim: SessionState, damage: str, recipe: dict[str, Any]
):
    job_id = _job_with_run_2(state_no_sim, damage)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"job_id": job_id, "label": "dut"}],
            "recipes": [recipe],
            "include": {"signals_available": True, "provenance": True},
        },
    )
    runs = {run["manifest_id"]: run for run in data["source_hashes"]}
    assert len(runs) == 4, data["source_hashes"]
    (damaged,) = [m for m, run in runs.items() if run["raw_path"].endswith("case-2.raw")]
    listed = {entry["manifest_id"] for entry in data.get("signals_available", [])}
    assert listed == set(runs) - {damaged}, data.get("signals_available")
    named = [f for f in data.get("failures", []) if f.get("stage") == "signals"]
    assert len(named) == 1, data.get("failures")
    (failure,) = named
    assert failure["where"] == "signals_available"
    assert runs[damaged]["label"] in failure["message"], failure
    if damage == "missing":
        assert failure["code"] == "results_missing", failure
        assert "results file" in failure["message"], failure


@pytest.mark.asyncio
async def test_a_source_whose_signals_could_not_be_read_is_named_with_the_reason(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    real = services.load_raw
    calls: list[int] = []

    async def unreadable_for_the_listing(source: Any, state: Any) -> Any:
        calls.append(1)
        if len(calls) > 1:
            raise ResultError("the raw could not be read for its signals")
        return await real(source, state)

    monkeypatch.setattr(services, "load_raw", unreadable_for_the_listing)
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "500u"}],
            "include": {"signals_available": True},
        },
    )
    (failure,) = [f for f in data.get("failures", []) if f.get("stage") == "signals"]
    assert failure["where"] == "signals_available"
    assert "dut" in failure["message"]
    assert "could not be read for its signals" in failure["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("slow", [False, True], ids=["read", "slow_read"])
async def test_a_signal_listing_the_bound_ended_names_the_sources_it_did_not_read(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch, slow: bool
):
    _slow_csv(monkeypatch)
    _slow_read(monkeypatch, slow)
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        {
            "sources": [{"path": str(raw), "label": "dut"}],
            "recipes": [_CSV],
            "include": {"signals_available": True},
        },
    )
    (failure,) = [f for f in data.get("failures", []) if f.get("stage") == "signals"]
    assert failure["code"] == "analysis_deadline", data.get("failures")
    assert failure["where"] == "signals_available"
    assert "dut" in failure["message"] and "call of its own" in failure["message"]
