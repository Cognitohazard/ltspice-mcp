"""Public log consumers read captured facts without opening source bodies."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from tests.conftest import (
    LTSPICE_TRAN_RC_VFINAL,
    SyncApi,
    make_experiment_job,
    stage_recorded_fixture,
)


def forbid_parent_reads(monkeypatch: pytest.MonkeyPatch, raw: Path) -> None:
    sources = {raw, raw.with_suffix(".log"), raw.with_suffix(".exe.log")}
    original_open = Path.open

    def guarded_open(path: Path, *args: Any, **kwargs: Any):
        if path in sources:
            raise AssertionError("A parent consumer opened a source body")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)


def test_api_measurements_use_captured_facts_without_parent_reads(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = make_experiment_job(
        state_no_sim, job_id="captured-measurements", raw=raw, case_id="selected", run_index=4
    )
    forbid_parent_reads(monkeypatch, raw)

    parsed = SyncApi(state_no_sim).measurements(job_id=job.job_id, case_id="selected")

    assert parsed["measurements"]["vfinal"]["values"] == [LTSPICE_TRAN_RC_VFINAL]
    assert set(parsed) == {
        "measurements",
        "step_count",
        "errors",
        "warnings",
        "failed_measurements",
    }


@pytest.mark.asyncio
async def test_resolved_analysis_relay_uses_captured_diagnostics_without_parent_reads(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    with raw.with_suffix(".log").open("a", encoding="utf-8") as handle:
        handle.write("\nIteration limit reached; no convergence.\n")
    forbid_parent_reads(monkeypatch, raw)

    captured = await analyze._create_result_set(
        analyze.AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [{"metric": "measurements", "key": "meas"}],
            }
        ),
        state_no_sim,
        time.monotonic() + 30,
    )
    missing = captured.inputs["missing"]
    observations = [analyze.Observation.of(item) for item in captured.inputs["observations"]]

    assert missing == []
    failures = [item for item in observations if item.code == "solve_failure"]
    assert len(failures) == 1
    assert failures[0].evidence is not None
    assert failures[0].evidence["log"] == "Iteration limit reached; no convergence."
    assert failures[0].evidence["runs"] == ["dut"]


@pytest.mark.asyncio
async def test_resolved_analysis_relays_console_only_failure(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.with_suffix(".log").unlink()
    raw.with_suffix(".exe.log").write_text(
        "Iteration limit reached; no convergence.\n", encoding="utf-8"
    )

    captured = await analyze._create_result_set(
        analyze.AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "console"}],
                "recipes": [{"metric": "measurements", "key": "meas"}],
            }
        ),
        state_no_sim,
        time.monotonic() + 30,
    )
    runs = analyze._deserialize_runs(captured, state_no_sim)
    missing = captured.inputs["missing"]
    observations = [analyze.Observation.of(item) for item in captured.inputs["observations"]]

    assert missing == [] and len(runs) == 1
    assert runs[0].source.console == raw.with_suffix(".exe.log")
    failures = [item for item in observations if item.code == "solve_failure"]
    assert len(failures) == 1
    assert failures[0].evidence is not None
    assert failures[0].evidence["log"] == "Iteration limit reached; no convergence."


def test_api_measurements_preserve_section_errors(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.with_suffix(".log").write_text("unrecognizable malformed log\n", encoding="utf-8")
    job = make_experiment_job(state_no_sim, job_id="malformed-measurements", raw=raw)

    with pytest.raises(ResultError, match="Could not parse log file"):
        SyncApi(state_no_sim).measurements(job_id=job.job_id)


def test_api_measurements_preserve_absent_recorded_log_error(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = make_experiment_job(state_no_sim, job_id="missing-measurements", raw=raw)
    job.cases[0].log_file = None

    with pytest.raises(ResultError, match="Experiment case 'case-0000' has no log file"):
        SyncApi(state_no_sim).measurements(job_id=job.job_id)


def test_api_measurements_preserve_valid_empty_shape(
    state_no_sim: SessionState, work_dir: Path
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
    job = make_experiment_job(state_no_sim, job_id="empty-measurements", raw=raw)

    parsed = SyncApi(state_no_sim).measurements(job_id=job.job_id)

    assert parsed == {
        "measurements": {},
        "step_count": 0,
        "errors": None,
        "warnings": None,
        "failed_measurements": [],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["loader", "section"])
async def test_analysis_preserves_unread_coverage_for_loader_and_section_errors(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    original_load = services.load_artifacts

    async def failed_diagnostics(
        source: services.AnalysisSource, state: SessionState, *, require_raw: bool
    ):
        if failure == "loader":
            raise ResultError("Contained log parsing exceeded its deadline")
        artifacts = await original_load(source, state, require_raw=require_raw)
        facts = artifacts.logs.as_dict()
        facts["diagnostics"] = {
            "status": "error",
            "value": None,
            "error": {"type": "ValueError", "message": "Invalid diagnostics"},
            "nonfinite_count": 0,
        }
        return replace(artifacts, logs=DecodedLog(facts))

    monkeypatch.setattr(services, "load_artifacts", failed_diagnostics)
    captured = await analyze._create_result_set(
        analyze.AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "unread"}],
                "recipes": [{"metric": "measurements", "key": "meas"}],
            }
        ),
        state_no_sim,
        time.monotonic() + 30,
    )
    missing = captured.inputs["missing"]
    observations = [analyze.Observation.of(item) for item in captured.inputs["observations"]]

    assert missing == []
    unread = [item for item in observations if item.code == "log_unread"]
    assert len(unread) == 1
    assert unread[0].evidence is not None
    assert unread[0].evidence["runs"] == ["unread"]
    assert not [item for item in observations if item.code == "solve_failure"]
