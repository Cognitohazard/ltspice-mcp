"""Behavioral contracts for the bounded consolidated analysis surface."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import atomic_write, experiment_store, now, result_store
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from tests.conftest import FIXTURES_DIR, make_sim_job, stage_recorded_fixture


def _source(raw: Path, label: str = "dut") -> dict[str, Any]:
    return {"raw_path": str(raw), "label": label}


def _args(raw: Path, recipes: list[dict[str, Any]], **extra: Any) -> AnalyzeResultsInput:
    return AnalyzeResultsInput.model_validate(
        {"sources": [_source(raw)], "recipes": recipes, **extra}
    )


async def _analyze(
    state: SessionState,
    raw: Path,
    recipes: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    result = await handle_analyze_results(_args(raw, recipes, **extra), state)
    assert result.structuredContent is not None
    return result.structuredContent


EXECUTION_CASES = [
    ("summary", "ltspice_tran_rc", {}),
    ("measurements", "ltspice_tran_rc", {}),
    ("value", "ltspice_tran_rc", {"expr": "V(out)", "at": "900u"}),
    ("signal_stats", "ltspice_tran_rc", {"signal": "V(out)"}),
    ("edges", "ltspice_tran_rc", {"signal": "V(out)"}),
    (
        "timing",
        "ltspice_tran_rc",
        {"from": {"signal": "V(in)"}, "to": {"signal": "V(out)"}},
    ),
    ("periodic", "ltspice_step_tran", {"signal": "V(out)"}),
    (
        "transient_response",
        "ltspice_tran_rc",
        {"signal": "V(out)", "mode": "step"},
    ),
    ("thd", "ltspice_step_tran", {"signal": "V(out)"}),
    ("bode_filter", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("bode_point", "ltspice_ac_rc", {"signal": "V(out)", "at_hz": "1k"}),
    (
        "bode_crossing",
        "ltspice_ac_rc",
        {"signal": "V(out)", "level_db": -3.0},
    ),
    (
        "bode_slope",
        "ltspice_ac_rc",
        {"signal": "V(out)", "from_hz": "10k", "to_hz": "100k"},
    ),
    ("stability", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("ac_structure", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("resonance", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("return_loss", "ltspice_ac_rc", {"signal": "V(out)"}),
    ("noise_integral", "ltspice_noise_rc", {}),
    ("operating_point", "ltspice_dc_div", {}),
    (
        "waveform",
        "ltspice_tran_rc",
        {"signals": ["V(out)"], "max_points": 25},
    ),
    ("plot", "ltspice_tran_rc", {"signals": ["V(out)"]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("metric", "fixture_name", "fields"), EXECUTION_CASES)
async def test_every_discriminant_executes_against_recorded_raw(
    metric: str,
    fixture_name: str,
    fields: dict[str, Any],
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, fixture_name)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": metric, "metric": metric, **fields}],
    )
    # Some physical metrics legitimately find no feature in a tiny RC fixture
    # (periodicity, resonance, or a loop crossover). They still must execute
    # through their adapter and fail only their own item.
    assert metric in data["results"] or any(
        failure.get("stage") == "analyze" for failure in data["failures"]
    )


@pytest.mark.asyncio
async def test_values_and_extrema_carry_outer_and_inner_identity(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "vout",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "reduce": ["min", "mean"],
            }
        ],
        include={"per_run": {"limit": 10}},
    )
    # A fully-analyzed single recipe over a valid raw reports the ratified
    # success outcome (formerly "success").
    assert data["outcome"] == "complete"
    record = data["results"]["vout"]["per_run"]["items"][0]
    for field in ("case_id", "run_index", "step_index", "step_values", "assignments"):
        assert field in record
    reduced = data["results"]["vout"]["reduced"]
    assert {entry["stat"] for entry in reduced} == {"min", "mean"}
    assert next(entry for entry in reduced if entry["stat"] == "min")["run_index"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("allow_incomplete", "expected"),
    [(False, "indeterminate"), (True, "pass")],
)
async def test_spec_verdict_incomplete_matrix(
    allow_incomplete: bool,
    expected: str,
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [
                _source(raw),
                {"job_id": "missing-job", "label": "missing"},
            ],
            "recipes": [
                {
                    "key": "vout",
                    "metric": "value",
                    "expr": "V(out)",
                    "at": "900u",
                    "spec": {"min": 0.5, "allow_incomplete": allow_incomplete},
                }
            ],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    data = result.structuredContent
    assert data is not None
    verdict = data["results"]["vout"]["spec"]
    assert verdict["verdict"] == expected
    assert verdict["pass_count"] == 1
    assert verdict["fail_count"] == 0


@pytest.mark.asyncio
async def test_spec_fail_counts_and_pages_cases(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "vout",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "spec": {"max": 0.1},
            }
        ],
        include={"outliers": True},
    )
    verdict = data["results"]["vout"]["spec"]
    assert verdict["verdict"] == "fail"
    assert verdict["fail_count"] == 1
    assert verdict["fail_cases"]["returned"] == 1
    assert verdict["outliers"][0]["run_index"] == 0


def _defer_csv_estimate(recipe, runs):
    del runs
    return 61.0 if getattr(recipe, "metric", None) == "waveform" else 0.0


@pytest.mark.asyncio
async def test_oversized_artifact_defers_whole_after_progress_then_continues(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", _defer_csv_estimate)
    first = await _analyze(
        state_no_sim,
        raw,
        [
            {"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"},
            {
                "key": "csv",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
            },
        ],
    )
    assert set(first["results"]) == {"v"}
    assert first["next"] is not None

    continuation = AnalyzeResultsInput.model_validate({"continue": first["next"]})
    resumed = await handle_analyze_results(continuation, state_no_sim)
    assert resumed.structuredContent is not None
    assert "csv" in resumed.structuredContent["results"]
    artifact = resumed.structuredContent["results"]["csv"]["values"][0]["value"]["artifact"]
    assert artifact["sha256"]
    artifact_size = await asyncio.to_thread(lambda: Path(artifact["path"]).stat().st_size)
    assert artifact["bytes"] == artifact_size
    assert resumed.structuredContent["next"] is None


@pytest.mark.asyncio
async def test_artifact_far_beyond_safety_factor_fails_fast(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.1
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", lambda recipe, runs: 1.0)
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "csv",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
            }
        ],
    )
    assert any(failure["code"] == "artifact_too_large" for failure in data["failures"])
    assert data["next"] is None


@pytest.mark.asyncio
async def test_slow_csv_deadline_advances_cursor_and_removes_temp(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from ltspice_mcp.tools import analysis

    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.01
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", lambda recipe, runs: 0.0)

    def slow_writer(*args):
        out_path = args[8]
        should_abort = args[9]
        with atomic_write(out_path) as handle:
            handle.write("partial\n")
            while not should_abort():
                time.sleep(0.002)
            raise ResultError("CSV artifact exceeded its analysis item deadline")

    monkeypatch.setattr(analysis, "_build_and_write", slow_writer)
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "csv",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
            },
            {"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"},
        ],
    )
    assert any(failure["code"] == "analysis_deadline" for failure in data["failures"])
    assert data["next"] is not None
    assert not list((work_dir / ".ltspice-mcp" / "results").rglob("*.pending"))


@pytest.mark.asyncio
async def test_continuation_is_pure_and_simultaneous_reads_match(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", _defer_csv_estimate)
    first = await _analyze(
        state_no_sim,
        raw,
        [
            {"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"},
            {
                "key": "csv",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
            },
        ],
    )
    continuation = AnalyzeResultsInput.model_validate({"continue": first["next"]})
    left, right = await asyncio.gather(
        handle_analyze_results(continuation, state_no_sim),
        handle_analyze_results(continuation, state_no_sim),
    )
    assert left.structuredContent == right.structuredContent


@pytest.mark.asyncio
async def test_per_run_page_cursor_replays_same_immutable_request(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    request = {
        "sources": [_source(raw)],
        "recipes": [
            {
                "key": "values",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "all_steps": True,
            }
        ],
        "include": {"per_run": {"limit": 1}},
    }
    first_args = AnalyzeResultsInput.model_validate(request)
    first = await handle_analyze_results(first_args, state_no_sim)
    assert first.structuredContent is not None
    cursor = first.structuredContent["results"]["values"]["per_run"]["next_cursor"]
    assert cursor is not None

    second_args = AnalyzeResultsInput.model_validate(
        {
            **request,
            "include": {"per_run": {"limit": 1, "cursor": cursor}},
        }
    )
    second = await handle_analyze_results(second_args, state_no_sim)
    assert second.structuredContent is not None
    assert second.structuredContent["results"]["values"]["per_run"]["items"][0]["step_index"] == 1


def test_invalid_and_expired_cursor_errors(work_dir: Path):
    item = result_store.create(
        working_dir=work_dir,
        inputs={"working_dir": str(work_dir)},
        work=[{"index": 0, "recipe": {"key": "x", "metric": "summary"}}],
        source_manifests=[],
        source_jobs={},
        ttl_hours=24,
    )
    with pytest.raises(ResultError, match="Invalid"):
        result_store.decode_cursor("tampered", item)

    path = result_store.result_path(item.result_set_id, work_dir)
    data = json.loads(path.read_text())
    data["expires_at"] = (now() - timedelta(hours=1)).isoformat()
    path.write_text(json.dumps(data))
    with pytest.raises(ResultError, match="expired"):
        result_store.load(item.result_set_id, work_dir)


async def _deferred_value_set(
    state: SessionState,
    raw: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    monkeypatch.setattr(
        analyze_mod,
        "_artifact_estimate",
        lambda recipe, runs: 61.0 if getattr(recipe, "key", "") == "later" else 0.0,
    )
    return await _analyze(
        state,
        raw,
        [
            {"key": "now", "metric": "value", "expr": "V(out)", "at": "900u"},
            {
                "key": "later",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
            },
        ],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutate", ["raw", "log"])
async def test_continuation_detects_raw_or_log_only_drift(
    mutate: str,
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    first = await _deferred_value_set(state_no_sim, raw, monkeypatch)
    target = raw if mutate == "raw" else raw.with_suffix(".log")
    target.write_bytes(target.read_bytes() + b"\nchanged")
    continuation = AnalyzeResultsInput.model_validate({"continue": first["next"]})
    result = await handle_analyze_results(continuation, state_no_sim)
    assert result.structuredContent is not None
    assert any(
        failure["code"] == "source_drift" for failure in result.structuredContent["failures"]
    )


@pytest.mark.asyncio
async def test_mutation_during_adapter_read_is_caught_by_postcheck(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    original = analyze_mod._adapter_value

    async def mutate_after_read(*args, **kwargs):
        value = await original(*args, **kwargs)
        raw.write_bytes(raw.read_bytes() + b"\nchanged-during-read")
        return value

    monkeypatch.setattr(analyze_mod, "_adapter_value", mutate_after_read)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
    )
    assert "v" not in data["results"]
    assert any(failure["code"] == "source_drift" for failure in data["failures"])


@pytest.mark.asyncio
async def test_digest_deadline_records_failure_and_progresses(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.01
    original = result_store.sha256_file

    def slow_digest(path):
        time.sleep(0.2)
        return original(path)

    monkeypatch.setattr(result_store, "sha256_file", slow_digest)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
    )
    assert any(failure["code"] == "analysis_deadline" for failure in data["failures"])
    assert data["next"] is None


def test_multi_source_job_invalidation_and_raw_only_ttl(work_dir: Path):
    first_record = work_dir / "first.json"
    second_record = work_dir / "second.json"
    first_record.write_text("{}")
    second_record.write_text("{}")
    item = result_store.create(
        working_dir=work_dir,
        inputs={"working_dir": str(work_dir)},
        work=[],
        source_manifests=[],
        source_jobs={"j1": str(first_record), "j2": str(second_record)},
        ttl_hours=0.0001,
    )
    # Job ownership overrides the raw-only TTL while every source job exists.
    assert result_store.load(item.result_set_id, work_dir).result_set_id == item.result_set_id
    assert result_store.invalidate_for_job(work_dir, "j2") == 1
    with pytest.raises(ResultError, match="missing or expired"):
        result_store.load(item.result_set_id, work_dir)

    raw_only = result_store.create(
        working_dir=work_dir,
        inputs={"working_dir": str(work_dir)},
        work=[],
        source_manifests=[],
        source_jobs={},
        ttl_hours=24,
    )
    path = result_store.result_path(raw_only.result_set_id, work_dir)
    data = json.loads(path.read_text())
    data["expires_at"] = (now() - timedelta(seconds=1)).isoformat()
    path.write_text(json.dumps(data))
    assert result_store.cleanup(work_dir) == 1


def _completed_with_failures_experiment(
    work_dir: Path,
    raw: Path,
) -> ExperimentJob:
    circuit = work_dir / "experiment.cir"
    circuit.write_text(".tran 1m\n.end\n")
    produced = ExperimentCase(
        case_id="case_ok",
        run_index=0,
        circuit="experiment",
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="deck-ok",
        assignments={"R": "1k"},
        status="produced",
        raw_file=raw,
        log_file=raw.with_suffix(".log"),
    )
    failed = ExperimentCase(
        case_id="case_bad",
        run_index=1,
        circuit="experiment",
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="deck-bad",
        status="failed",
        error="simulator failed",
    )
    source = SourceRecord(
        circuit="experiment",
        path=circuit,
        sha256="source",
        staged_deck=circuit,
        manifest=[
            ManifestEntry(
                path=circuit,
                sha256="source",
                staged=True,
                live=False,
                staged_path=circuit,
            )
        ],
    )
    completeness = Completeness(declared=1, expanded=2)
    completeness.recount([produced, failed])
    return ExperimentJob(
        job_id="experiment_analysis",
        request_id="analysis-request",
        fingerprint="f" * 64,
        canonicalizer_version=experiment_store.CANONICALIZER_VERSION,
        control_token="secret",
        store_path=experiment_store.record_path("experiment_analysis", work_dir),
        cases=[produced, failed],
        sources=[source],
        simulator="LTspice",
        completeness=completeness,
        status="completed_with_failures",
        completed_at=now(),
    )


@pytest.mark.asyncio
async def test_completed_with_failures_experiment_analyzes_produced_cases(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = _completed_with_failures_experiment(work_dir, raw)
    experiment_store.save_job(job)
    state_no_sim.add_experiment_job(job, already_persisted=True)
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"job_id": job.job_id, "label": "experiment"}],
            "recipes": [
                {
                    "key": "v",
                    "metric": "value",
                    "expr": "V(out)",
                    "at": "900u",
                    "reduce": ["mean"],
                }
            ],
            "group_by": ["R"],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    data = result.structuredContent
    assert data is not None
    assert "v" in data["results"]
    assert data["results"]["v"]["groups"][0]["by"] == {"R": "1k"}
    assert data["coverage"]["runs_requested"] == 2
    assert data["coverage"]["missing_cases"]["items"][0]["case_id"] == "case_bad"


@pytest.mark.asyncio
async def test_raw_path_has_null_deck_hash_and_provenance_observation(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
    )
    value = data["results"]["v"]["values"][0]
    assert value["deck_sha256"] is None
    assert any(
        observation["code"] == "raw_path_without_deck_provenance"
        for observation in data["observations"]
    )


@pytest.mark.asyncio
async def test_raw_path_run_selection_reports_nonzero_outer_runs_missing(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"raw_path": str(raw), "label": "raw", "runs": [0, 3]}],
            "recipes": [{"key": "summary", "metric": "summary"}],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    assert result.structuredContent is not None
    coverage = result.structuredContent["coverage"]
    assert coverage["runs_requested"] == 2
    assert coverage["missing_cases"]["items"][0]["run_index"] == 3


@pytest.mark.asyncio
async def test_noncompleted_legacy_job_keeps_completed_only_gate(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    deck = work_dir / "legacy.cir"
    deck.write_text(".tran 1m\n.end\n")
    job = make_sim_job(
        "legacy_running",
        status="running",
        netlist=deck,
        raw_file=raw,
        log_file=raw.with_suffix(".log"),
    )
    state_no_sim.add_job(job)
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"job_id": job.job_id, "label": "legacy"}],
            "recipes": [{"key": "summary", "metric": "summary"}],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    data = result.structuredContent
    assert data is not None
    assert data["coverage"]["runs_analyzed"] == 0
    assert "not completed" in data["coverage"]["missing_cases"]["items"][0]["detail"]


@pytest.mark.asyncio
async def test_summary_and_measurement_slow_parsers_are_bounded_at_tool_level(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from ltspice_mcp.tools import analysis

    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.05

    def slow_summary(*args, **kwargs):
        del args, kwargs
        time.sleep(0.2)
        return {}

    monkeypatch.setattr(analysis, "build_simulation_summary", slow_summary)
    summary = await _analyze(
        state_no_sim,
        raw,
        [{"key": "summary", "metric": "summary"}],
    )
    assert any(
        failure["code"] == "analysis_deadline" and failure["stage"] == "analyze"
        for failure in summary["failures"]
    )

    # A fresh path avoids the shared cooldown from the deliberately wedged raw.
    second = work_dir / "second.raw"
    shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.raw", second)
    shutil.copy(FIXTURES_DIR / "ltspice_tran_rc.log", second.with_suffix(".log"))

    def slow_measurements(*args, **kwargs):
        del args, kwargs
        time.sleep(0.2)
        return {}, {}, "0 step(s)", {}

    monkeypatch.setattr(analysis, "_aggregate_log_measurements", slow_measurements)
    measurements = await _analyze(
        state_no_sim,
        second,
        [{"key": "measurements", "metric": "measurements"}],
    )
    assert any(
        failure["code"] == "analysis_deadline" and failure["stage"] == "analyze"
        for failure in measurements["failures"]
    )
