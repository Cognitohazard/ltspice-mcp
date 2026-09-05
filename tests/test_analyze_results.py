"""Behavioral contracts for the bounded consolidated analysis surface."""

from __future__ import annotations

import asyncio
import json
import shutil
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, NamedTuple

import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    ResultError,
    compact_validation_error,
)
from ltspice_mcp.lib import (
    atomic_write,
    cursor_codec,
    experiment_store,
    metrics,
    now,
    result_store,
)
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.recipes import RECIPE_MODELS
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze as analyze_mod
from ltspice_mcp.tools import experiments
from ltspice_mcp.tools.analyze import (
    AnalyzeResultsInput,
    evaluate_analysis_results,
    handle_analyze_results,
)
from tests.conftest import FIXTURES_DIR, make_experiment_job, stage_recorded_fixture


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
    assert result.structured_content is not None
    return result.structured_content


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


def test_every_recipe_class_has_a_metric_function():
    """The evaluator dispatches on recipe class, so a class with no entry in
    the table would raise a KeyError at the one call that needed it — after the
    sources were resolved and hashed. Naming the gap here instead means a new
    recipe cannot ship with no way to compute it."""
    missing = [
        model.__name__
        for model in RECIPE_MODELS
        if model not in metrics.METRICS and model not in metrics.ARTIFACT_RECIPES
    ]
    assert not missing, (
        f"These recipe classes have no metric function: {sorted(missing)}. "
        "Register one in lib.metrics.METRICS, or list the class in "
        "ARTIFACT_RECIPES if the evaluator produces its value itself."
    )
    # The reverse direction: a metric registered for a class the union dropped
    # is dead dispatch nothing can reach.
    unreachable = sorted(model.__name__ for model in metrics.METRICS if model not in RECIPE_MODELS)
    assert not unreachable, f"These metric functions answer no recipe: {unreachable}"


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
    # Lean default: attribution keys that carry information survive; a
    # null/empty one (no case, no steps on a standalone raw) is dropped —
    # absent and empty mean the same thing on a row with no required keys.
    assert record["run_index"] == 0
    for field in ("case_id", "step_index", "step_values", "assignments"):
        assert record.get(field) in (None, {}, []) or field in record
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
    data = result.structured_content
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
    assert resumed.structured_content is not None
    assert "csv" in resumed.structured_content["results"]
    artifact = resumed.structured_content["results"]["csv"]["values"][0]["value"]["artifact"]
    assert artifact["sha256"]
    artifact_size = await asyncio.to_thread(lambda: Path(artifact["path"]).stat().st_size)
    assert artifact["bytes"] == artifact_size
    assert resumed.structured_content["next"] is None


class _NeutralWork(NamedTuple):
    """Unprojected work produced for one recipe during one evaluator drive."""

    key: str
    position: int
    rows: tuple[analyze_mod.Record, ...]
    reductions: tuple[dict[str, Any], ...]
    facts: dict[str, Any]
    failures: tuple[analyze_mod.Failure, ...]
    observations: tuple[analyze_mod.Observation, ...]


def _neutral_work(evaluation: analyze_mod.AnalysisEvaluation) -> list[_NeutralWork]:
    """The drive's complete unprojected rows and derived facts, per recipe.

    This is the identity a resumed sequence of drives must reproduce exactly:
    the same rows, reductions and facts the one-shot evaluation computes, with
    every MCP cap lifted (spec fail cases uncapped, outliers always included).
    """
    work: list[_NeutralWork] = []
    for unit in evaluation.processed:
        recipe = unit.recipe
        rows = unit.records
        relevant_missing = [
            case
            for case in evaluation.missing
            if recipe.sources is None or case.get("label") in set(recipe.sources)
        ]
        spec = analyze_mod._spec(
            recipe,
            rows,
            incomplete=bool(unit.failures or relevant_missing),
            include_outliers=True,
            fail_case_limit=max(1, len(rows)),
        )
        if spec is not None:
            fail_cases = spec["fail_cases"]["items"]
            spec = {
                key: value for key, value in spec.items() if key not in {"fail_cases", "outliers"}
            }
            spec["fail_cases"] = fail_cases
        work.append(
            _NeutralWork(
                key=unit.key,
                position=unit.position,
                rows=tuple(rows),
                reductions=tuple(analyze_mod._reduce(recipe, rows)),
                facts={
                    "metric": recipe.metric,
                    "warnings": analyze_mod._record_warnings(rows),
                    "groups": analyze_mod._group_values(recipe, rows, evaluation.group_by),
                    "spec": spec,
                },
                failures=tuple(unit.failures),
                observations=tuple(unit.observations),
            )
        )
    return work


@pytest.mark.asyncio
async def test_neutral_evaluator_resumes_to_the_same_work_as_one_shot(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    args = _args(
        raw,
        [
            {"key": "first", "metric": "value", "expr": "V(out)", "at": "800u"},
            {"key": "second", "metric": "value", "expr": "V(out)", "at": "900u"},
            {"key": "summary", "metric": "summary"},
        ],
    )
    state_no_sim.config.analysis_budget_s = 0.01
    monkeypatch.setattr(
        analyze_mod,
        "_artifact_estimate",
        lambda recipe, _runs: 0.01 if recipe.key != "first" else 0.0,
    )

    accumulated: list[tuple[Any, ...]] = []
    position = None
    drives = 0
    while True:
        evaluation = await evaluate_analysis_results(
            args,
            state_no_sim,
            continuation=position,
        )
        accumulated.extend(_neutral_work(evaluation))
        drives += 1
        position = evaluation.continuation
        if position is None:
            break
    assert drives > 1, "the tiny drive budget must exercise internal resumption"

    state_no_sim.config.analysis_budget_s = 60.0
    one_shot = await evaluate_analysis_results(args, state_no_sim)
    assert one_shot.continuation is None
    assert accumulated == _neutral_work(one_shot)


@pytest.mark.asyncio
async def test_neutral_failures_are_uncapped_while_mcp_keeps_its_cap(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    args = _args(raw, [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}])
    failures = [
        analyze_mod.Failure(
            code="recipe_failed",
            stage="analyze",
            where=f"run-{index}",
            message=f"failure {index}",
        )
        for index in range(analyze_mod._FAILURE_CAP + 7)
    ]

    async def fail_many(*_args, **_kwargs):
        return [], list(failures), []

    monkeypatch.setattr(analyze_mod, "_evaluate_item", fail_many)
    neutral = await evaluate_analysis_results(args, state_no_sim)
    assert list(neutral.failure_inventory) == failures
    assert list(_neutral_work(neutral)[0].failures) == failures

    mcp = await handle_analyze_results(args, state_no_sim)
    assert mcp.structured_content is not None
    presented = mcp.structured_content
    assert presented["failures"] == [
        failure.wire() for failure in failures[: analyze_mod._FAILURE_CAP]
    ]
    note = next(
        observation
        for observation in presented["observations"]
        if observation["code"] == "failures_truncated"
    )
    assert f"of {len(failures)} failure records" in note["detail"]
    assert len(failures) - len(presented["failures"]) == 7


@pytest.mark.asyncio
async def test_neutral_rows_are_unprojected_before_mcp_paging_and_fields(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    args = _args(
        raw,
        [
            {
                "key": "values",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "all_steps": True,
            }
        ],
        include={"per_run": {"limit": 1}, "fields": ["step_index"]},
    )
    neutral = await evaluate_analysis_results(args, state_no_sim)
    rows = [record.wire() for record in _neutral_work(neutral)[0].rows]
    assert len(rows) == 3
    assert all("value" in row and "source" in row for row in rows)

    mcp = await handle_analyze_results(args, state_no_sim)
    assert mcp.structured_content is not None
    page = mcp.structured_content["results"]["values"]["per_run"]
    assert page["items"] == [{"step_index": rows[0]["step_index"]}]
    assert (page["total"], page["returned"], page["truncated"]) == (3, 1, True)
    assert page["next_cursor"] is not None
    assert page["total"] - page["returned"] == len(rows) - 1


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
            # What the real writer raises when its abort callback fires: the
            # deadline is carried by the exception type, not its wording.
            raise AnalysisDeadlineExceeded("CSV artifact exceeded its analysis item deadline")

    monkeypatch.setattr(analysis, "build_waveform_csv", slow_writer)
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
    assert left.structured_content == right.structured_content


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
    assert first.structured_content is not None
    cursor = first.structured_content["results"]["values"]["per_run"]["next_cursor"]
    assert cursor is not None

    second_args = AnalyzeResultsInput.model_validate(
        {
            **request,
            "include": {"per_run": {"limit": 1, "cursor": cursor}},
        }
    )
    second = await handle_analyze_results(second_args, state_no_sim)
    assert second.structured_content is not None
    assert second.structured_content["results"]["values"]["per_run"]["items"][0]["step_index"] == 1


@pytest.mark.asyncio
async def test_per_run_cursor_rejects_an_explicitly_different_fields_view(
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
        "include": {"per_run": {"limit": 1}, "fields": ["step_index"]},
    }
    first = await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state_no_sim)
    assert first.structured_content is not None
    cursor = first.structured_content["results"]["values"]["per_run"]["next_cursor"]
    assert cursor is not None

    with pytest.raises(ResultError, match="replay page 1"):
        await handle_analyze_results(
            AnalyzeResultsInput.model_validate(
                {
                    **request,
                    "include": {
                        "per_run": {"limit": 1, "cursor": cursor},
                        "fields": ["value"],
                    },
                }
            ),
            state_no_sim,
        )

    inherited = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                **request,
                "include": {"per_run": {"limit": 1, "cursor": cursor}},
            }
        ),
        state_no_sim,
    )
    assert inherited.structured_content is not None
    inherited_row = inherited.structured_content["results"]["values"]["per_run"]["items"][0]
    assert set(inherited_row) == {"step_index"}


@pytest.mark.asyncio
async def test_viewless_legacy_cursor_falls_back_to_the_stored_fields_view(
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
        "include": {"per_run": {"limit": 1}, "fields": ["step_index"]},
    }
    validated = AnalyzeResultsInput.model_validate(request)
    first = await handle_analyze_results(validated, state_no_sim)
    assert first.structured_content is not None
    cursor = first.structured_content["results"]["values"]["per_run"]["next_cursor"]
    assert cursor is not None
    result_set_id = first.structured_content["result_set_id"]

    record = result_store.result_path(result_set_id, work_dir)
    stored = json.loads(record.read_text())
    stored["inputs"]["request_hash"] = analyze_mod._request_hash(
        validated,
        include_fields=True,
    )
    snapshot = {
        key: stored[key]
        for key in (
            "result_set_id",
            "created_at",
            "expires_at",
            "inputs",
            "work",
            "source_manifests",
            "source_jobs",
        )
    }
    stored["snapshot_hash"] = result_store.canonical_hash(snapshot)
    record.write_text(json.dumps(stored))

    body = cursor_codec.decode_cursor(cursor)
    body.pop("view")
    legacy_cursor = cursor_codec.encode_cursor(body)

    resumed = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                **request,
                "include": {"per_run": {"limit": 1, "cursor": legacy_cursor}},
            }
        ),
        state_no_sim,
    )
    assert resumed.structured_content is not None
    row = resumed.structured_content["results"]["values"]["per_run"]["items"][0]
    assert set(row) == {"step_index"}


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
    assert result.structured_content is not None
    assert any(
        failure["code"] == "source_drift" for failure in result.structured_content["failures"]
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
        store_path=Store(work_dir).job_record("experiment_analysis"),
        cases=[produced, failed],
        sources=[source],
        simulator="LTspice",
        completeness=completeness,
        status="completed_with_failures",
        completed_at=now(),
    )


@pytest.mark.asyncio
async def test_solve_failure_in_the_log_is_relayed_into_observations(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """A run that wrote a raw despite a failed solve must not read as clean.

    The full profile relays this at its metric chokepoint; without the same
    relay here a caller reads a number off diverged data with nothing on the
    response to say the solve collapsed.
    """
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    log = raw.with_suffix(".log")
    log.write_text(log.read_text() + "\nIteration limit reached; no convergence.\n")

    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
    )

    relayed = [item for item in data["observations"] if item["code"] == "solve_failure"]
    assert len(relayed) == 1
    assert relayed[0]["kind"] == "relay"
    assert "iteration limit reached" in relayed[0]["evidence"]["log"].lower()
    assert relayed[0]["evidence"]["run_count"] == 1
    # The relay is a fact ABOUT the run, not a refusal to read it.
    assert "v" in data["results"]


@pytest.mark.asyncio
async def test_one_cause_relays_once_though_each_run_logged_its_own_numbers(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """A sweep that fails to converge fails at a different instant in every run.

    The simulator's line ends in that run's own time and timestep, so keying the
    relay on the verbatim line yields one observation per run in a channel the
    budget ladder may not trim.
    """
    template = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    base_log = template.with_suffix(".log").read_text()
    sources: list[dict[str, Any]] = []
    for index in range(4):
        raw = work_dir / f"corner{index}.raw"
        raw.write_bytes(template.read_bytes())
        raw.with_suffix(".log").write_text(
            base_log + f"\nTime step too small; time = {4.4e-05 + index * 1e-7:.7e}, "
            f"timestep = {1.2e-19 / (index + 1):.4e}\n"
        )
        sources.append({"raw_path": str(raw), "label": f"corner{index}"})

    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": sources,
                "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
            }
        ),
        state_no_sim,
    )
    assert result.structured_content is not None
    relayed = [
        item
        for item in result.structured_content["observations"]
        if item["code"] == "solve_failure"
    ]

    assert len(relayed) == 1
    assert relayed[0]["evidence"]["run_count"] == 4
    assert sorted(relayed[0]["evidence"]["runs"]) == [f"corner{i}" for i in range(4)]
    # The relayed line is one run's own, verbatim — not a normalized rewrite.
    assert "time = 4.4000000e-05" in relayed[0]["evidence"]["log"]


@pytest.mark.asyncio
async def test_a_clean_solve_relays_nothing(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")

    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
    )

    assert not [item for item in data["observations"] if item["code"] == "solve_failure"]


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
    data = result.structured_content
    assert data is not None
    assert "v" in data["results"]
    assert data["results"]["v"]["groups"][0]["by"] == {"R": "1k"}
    assert data["coverage"]["runs_requested"] == 2
    assert data["coverage"]["missing_cases"]["items"][0]["case_id"] == "case_bad"


@pytest.mark.asyncio
async def test_analyzing_experiment_reads_its_own_produced_cases(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """An experiment's attached analysis necessarily runs while the job sits in
    'analyzing' — a state entered only after every run reached a terminal status
    — so that state must resolve produced cases with full case identity."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = _completed_with_failures_experiment(work_dir, raw)
    job.status = "analyzing"
    experiment_store.save_job(job)
    state_no_sim.add_experiment_job(job, already_persisted=True)
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"job_id": job.job_id, "label": "experiment"}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
            "group_by": ["R"],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    data = result.structured_content
    assert data is not None
    assert "v" in data["results"]
    assert data["results"]["v"]["groups"][0]["by"] == {"R": "1k"}
    # The per-case gate still does the real work: the failed case is missing,
    # the produced one is analyzed.
    assert data["coverage"]["missing_cases"]["items"][0]["case_id"] == "case_bad"
    assert data["coverage"]["runs_analyzed"] == 1


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
    # Lean rows no longer carry the per-row deck digest at all; the
    # observation remains the channel that says this source has no deck
    # provenance to offer.
    assert "deck_sha256" not in value
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
    assert result.structured_content is not None
    coverage = result.structured_content["coverage"]
    assert coverage["runs_requested"] == 2
    assert coverage["missing_cases"]["items"][0]["run_index"] == 3


@pytest.mark.asyncio
async def test_noncompleted_experiment_keeps_the_terminal_only_gate(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """A job still running is not readable, even when a raw is already on disk.

    Half a sweep's artifacts exist long before the job is done; analyzing them
    as if they were the whole answer is the failure this gate exists for.
    """
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    job = make_experiment_job(state_no_sim, job_id="exp_running", status="running", raw=raw)
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"job_id": job.job_id, "label": "live"}],
            "recipes": [{"key": "summary", "metric": "summary"}],
        }
    )
    result = await handle_analyze_results(args, state_no_sim)
    data = result.structured_content
    assert data is not None
    assert data["coverage"]["runs_analyzed"] == 0
    (missing,) = data["coverage"]["missing_cases"]["items"]
    assert missing["code"] == "job_not_terminal"
    assert "no readable runs yet" in missing["detail"]


@pytest.mark.asyncio
async def test_summary_and_measurement_slow_parsers_are_bounded_at_tool_level(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from ltspice_mcp.lib import metrics

    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.05

    def slow_summary(*args, **kwargs):
        del args, kwargs
        time.sleep(0.2)
        return {}

    monkeypatch.setattr(metrics, "build_simulation_summary", slow_summary)
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

    monkeypatch.setattr(metrics, "aggregate_log_measurements", slow_measurements)
    measurements = await _analyze(
        state_no_sim,
        second,
        [{"key": "measurements", "metric": "measurements"}],
    )
    assert any(
        failure["code"] == "analysis_deadline" and failure["stage"] == "analyze"
        for failure in measurements["failures"]
    )


# ---------------------------------------------------------------------------
# Envelope contract: what the response promises must be reachable and legal
# ---------------------------------------------------------------------------


def _raw_with_non_finite(work_dir: Path) -> Path:
    """A .raw whose trace holds a NaN, as a diverged solve produces."""
    import numpy as np
    from spicelib.raw.raw_write import RawWrite, Trace

    n = 64
    values = np.linspace(0.0, 5.0, n)
    values[7] = np.nan
    writer = RawWrite(plot_name="Transient Analysis")
    writer.add_trace(Trace("time", np.linspace(0.0, 1e-3, n), whattype="time"))
    writer.add_trace(Trace("V(out)", values, whattype="voltage"))
    path = work_dir / "diverged.raw"
    writer.save(path)
    return path


@pytest.mark.asyncio
async def test_non_finite_sample_keeps_the_response_schema_conformant(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """A NaN sample makes sanitize_payload inject a top-level 'warnings' key.

    Undeclared under the schema's additionalProperties:false, that injection
    made a strict client reject the whole response exactly when a run diverged.
    """
    import jsonschema

    raw = _raw_with_non_finite(work_dir)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "wave", "metric": "waveform", "signals": ["V(out)"]}],
    )
    series = data["results"]["wave"]["values"][0]["value"]["series"][0]
    assert None in series["y"], "the NaN sample must be surfaced as an explicit null"
    assert any("Non-finite" in warning for warning in data["warnings"])
    jsonschema.Draft202012Validator(analyze_mod.OUTPUT_SCHEMA).validate(data)


@pytest.mark.asyncio
async def test_csv_waveform_states_that_max_points_is_inline_only(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """max_points bounds the inline series only; a csv artifact ignores it."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "csv",
                "metric": "waveform",
                "signals": ["V(out)"],
                "format": "csv",
                "max_points": 10,
            }
        ],
    )
    assert data["results"]["csv"]["values"][0]["value"]["row_count"] > 10
    assert any(
        observation["code"] == "max_points_not_applied" for observation in data["observations"]
    )


@pytest.mark.asyncio
async def test_artifact_too_large_names_only_levers_that_move_the_bound(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """The estimate reads raw size and signal count, nothing else — so telling
    the caller to lower max_points or narrow the window sends them in a loop."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    state_no_sim.config.analysis_budget_s = 0.1
    monkeypatch.setattr(analyze_mod, "_artifact_estimate", lambda recipe, runs: 1.0)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "csv", "metric": "waveform", "signals": ["V(out)"], "format": "csv"}],
    )
    message = next(
        failure["message"]
        for failure in data["failures"]
        if failure["code"] == "artifact_too_large"
    )
    # The levers by name — which ones the message offers is the contract,
    # the sentence that frames them is not.
    assert "fewer signals" in message
    assert "max_points" in message
    assert "window" in message
    # The one case with no request-side lever at all names the config exit.
    assert "analysis_budget_s" in message


@pytest.mark.parametrize(
    "extra",
    [
        {"include": {"per_run": {"limit": 100}}},
        {"include": {"outliers": True}},
        {"group_by": ["temp"]},
    ],
)
def test_continuation_rejects_request_shaping_arguments(extra: dict[str, Any]):
    """A continuation replays stored execution state plus its cursor view, so
    include/group_by passed alongside it must not be silently dropped."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="mutually exclusive"):
        AnalyzeResultsInput.model_validate(
            {"continue": {"result_set_id": "set-1", "cursor": "abc"}, **extra}
        )


@pytest.mark.asyncio
async def test_truncated_missing_cases_page_is_followable(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """coverage.missing_cases advertises truncation, so it must hand back a
    cursor that actually reaches the rest."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    args = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"raw_path": str(raw), "label": "dut", "runs": list(range(121))}],
            "recipes": [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}],
        }
    )
    first = await handle_analyze_results(args, state_no_sim)
    assert first.structured_content is not None
    page = first.structured_content["coverage"]["missing_cases"]
    assert (page["total"], page["returned"], page["truncated"]) == (120, 100, True)
    assert page["next_cursor"] is not None

    resumed = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "continue": {
                    "result_set_id": first.structured_content["result_set_id"],
                    "cursor": page["next_cursor"],
                }
            }
        ),
        state_no_sim,
    )
    assert resumed.structured_content is not None
    rest = resumed.structured_content["coverage"]["missing_cases"]
    assert rest["returned"] == 20
    assert rest["items"][0]["run_index"] == 101
    assert rest["truncated"] is False
    assert rest["next_cursor"] is None


@pytest.mark.asyncio
async def test_work_and_coverage_cursors_advance_independently(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """One call pages two lists; either cursor must advance its own view without
    resetting or discarding the other.

    They share one encoded resume point, so a cursor that drops the work
    position strands un-analyzed work, and one that drops the coverage offset
    re-serves missing cases the caller has already read.
    """
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    request = {
        "sources": [{"raw_path": str(raw), "label": "dut", "runs": list(range(121))}],
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
    first = await handle_analyze_results(AnalyzeResultsInput.model_validate(request), state_no_sim)
    assert first.structured_content is not None
    coverage_cursor = first.structured_content["coverage"]["missing_cases"]["next_cursor"]
    assert coverage_cursor is not None
    assert first.structured_content["next"] is not None
    result_set_id = first.structured_content["result_set_id"]

    # Following the coverage cursor advances the missing list AND keeps the
    # work resume point, so there is still a `next` to follow.
    by_coverage = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {"continue": {"result_set_id": result_set_id, "cursor": coverage_cursor}}
        ),
        state_no_sim,
    )
    assert by_coverage.structured_content is not None
    rest = by_coverage.structured_content["coverage"]["missing_cases"]
    assert rest["items"][0]["run_index"] == 101
    assert by_coverage.structured_content["next"] is not None

    # Following the work cursor advances the work AND carries the coverage
    # offset, so the missing cases already served are not replayed.
    by_work = await handle_analyze_results(
        AnalyzeResultsInput.model_validate({"continue": first.structured_content["next"]}),
        state_no_sim,
    )
    assert by_work.structured_content is not None
    carried = by_work.structured_content["coverage"]["missing_cases"]
    assert carried["items"][0]["run_index"] == 101
    assert carried["truncated"] is False


@pytest.mark.asyncio
async def test_truncated_fail_cases_names_the_route_to_the_rest(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """spec.fail_cases is a per-call projection with no cursor of its own, so a
    truncated page must name the view that does page every value."""
    monkeypatch.setattr(analyze_mod, "_FAIL_CASE_PAGE_CAP", 1)
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    data = await _analyze(
        state_no_sim,
        raw,
        [
            {
                "key": "vout",
                "metric": "value",
                "expr": "V(out)",
                "at": "900u",
                "all_steps": True,
                "spec": {"max": -1.0},
            }
        ],
    )
    entry = data["results"]["vout"]
    assert entry["spec"]["fail_cases"]["truncated"] is True
    assert any("include.per_run" in warning for warning in entry["warnings"])


# ---------------------------------------------------------------------------
# include.fields: per-row projection
# ---------------------------------------------------------------------------

# 15 source labels over a 3-step .AC sweep — 45 rows, the width at which an
# agent stops reading the tool's rows and writes its own parser instead.
_WIDE_SOURCES = 15

_LOOP_RECIPE: dict[str, Any] = {
    "key": "loop",
    "metric": "bode_filter",
    "signal": "V(out)",
    "all_steps": True,
}

_FULL_ROW_KEYS = {
    "source",
    "case_id",
    "run_index",
    "step_index",
    "step_values",
    "assignments",
    "circuit",
    "deck_sha256",
    "value",
}


def _wide_args(raw: Path, **include: Any) -> AnalyzeResultsInput:
    return AnalyzeResultsInput.model_validate(
        {
            "sources": [_source(raw, f"corner{index:02d}") for index in range(_WIDE_SOURCES)],
            "recipes": [_LOOP_RECIPE],
            "include": include,
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["values", "per_run"])
async def test_include_fields_projects_both_row_surfaces(
    surface: str,
    state_no_sim: SessionState,
    work_dir: Path,
):
    """Projection reaches the paged and un-paged row lists alike — a lever that
    worked on one and not the other would depend on an unrelated argument."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    include: dict[str, Any] = {"fields": ["step_index", "value.passband_gain_db"]}
    if surface == "per_run":
        include["per_run"] = {"limit": 10}
    data = await _analyze(state_no_sim, raw, [_LOOP_RECIPE], include=include)
    entry = data["results"]["loop"]
    rows = entry["per_run"]["items"] if surface == "per_run" else entry["values"]
    assert rows
    for row in rows:
        # Projection keeps the row's shape and drops keys: the nested read
        # row["value"]["passband_gain_db"] is identical projected or not.
        assert set(row) == {"step_index", "value"}
        assert set(row["value"]) == {"passband_gain_db"}


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["values", "per_run"])
async def test_default_rows_are_lean_and_fields_restores_the_whole_value(
    surface: str,
    state_no_sim: SessionState,
    work_dir: Path,
):
    """Lean-by-default: rows carry the scalar leaves of value and drop
    null/empty attribution plus the per-row deck digest; include.fields is
    the named opt-in that restores any dropped detail, up to the whole
    block via fields=["value"]. Both row surfaces render identically."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    include: dict[str, Any] = {"per_run": {"limit": 10}} if surface == "per_run" else {}
    data = await _analyze(state_no_sim, raw, [_LOOP_RECIPE], include=include)
    entry = data["results"]["loop"]
    rows = entry["per_run"]["items"] if surface == "per_run" else entry["values"]
    assert rows
    for row in rows:
        assert set(row) <= _FULL_ROW_KEYS
        assert "deck_sha256" not in row
        assert not any(isinstance(item, (dict, list)) for item in row["value"].values()), (
            "default value must be scalar leaves only"
        )

    full_include = dict(include)
    full_include["fields"] = ["value"]
    full = await _analyze(state_no_sim, raw, [_LOOP_RECIPE], include=full_include)
    full_entry = full["results"]["loop"]
    full_rows = full_entry["per_run"]["items"] if surface == "per_run" else full_entry["values"]
    assert any(
        isinstance(item, (dict, list)) for row in full_rows for item in row["value"].values()
    ), 'include.fields=["value"] must restore the nested detail'


def test_unknown_projection_path_names_the_valid_row_keys(work_dir: Path):
    """A path that cannot be rooted in a row is refused at the interface, naming the
    keys that exist — an advertised lever that silently keeps nothing is worse
    than no lever."""
    with pytest.raises(ValidationError) as excinfo:
        _args(
            work_dir / "unread.raw",
            [_LOOP_RECIPE],
            include={"fields": ["passband_gain_db"]},
        )
    message = str(excinfo.value)
    assert "passband_gain_db" in message
    for key in _FULL_ROW_KEYS:
        assert key in message


@pytest.mark.asyncio
async def test_absent_nested_path_names_the_keys_the_rows_do_carry(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """include.fields is call-global while each metric owns its value shape, so a
    path into 'value' can legitimately miss. It is reported per recipe, naming
    the keys that are there, instead of handing back rows with nothing in them."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    data = await _analyze(
        state_no_sim,
        raw,
        [_LOOP_RECIPE],
        include={"fields": ["value.phase_margin_deg"]},
    )
    entry = data["results"]["loop"]
    assert entry["values"] and all(row == {} for row in entry["values"])
    warning = next(text for text in entry["warnings"] if "phase_margin_deg" in text)
    assert "keys present at 'value'" in warning
    assert "passband_gain_db" in warning


@pytest.mark.asyncio
async def test_projection_leaves_spec_attribution_rows_whole(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """spec.fail_cases rows are an already-reduced attribution shape whose
    'value' is the failing number, not the metric's value dict, so include.fields
    paths do not address them and they are returned intact."""
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
        include={"fields": ["step_index"], "outliers": True},
    )
    entry = data["results"]["vout"]
    assert entry["values"] == [{"step_index": 0}]
    case = entry["spec"]["fail_cases"]["items"][0]
    assert set(case) == {
        "value",
        "case_id",
        "run_index",
        "step_index",
        "step_values",
        "assignments",
    }
    assert entry["spec"]["outliers"][0]["run_index"] == 0


@pytest.mark.asyncio
async def test_projection_shrinks_a_wide_sweep_payload(
    state_no_sim: SessionState,
    work_dir: Path,
    capsys: pytest.CaptureFixture[str],
):
    """The whole justification, measured: a 45-row table an agent wants three
    numbers from must not cost the full nested value dict of every row."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    full = await handle_analyze_results(_wide_args(raw), state_no_sim)
    projected = await handle_analyze_results(
        _wide_args(raw, fields=["source", "step_values", "value.passband_gain_db"]),
        state_no_sim,
    )
    assert full.structured_content is not None and projected.structured_content is not None
    full_rows = full.structured_content["results"]["loop"]["values"]
    projected_rows = projected.structured_content["results"]["loop"]["values"]
    assert len(full_rows) == len(projected_rows) == 45
    full_chars = len(json.dumps(full_rows))
    projected_chars = len(json.dumps(projected_rows))
    with capsys.disabled():
        print(
            f"\n45-row table: {full_chars} chars whole, {projected_chars} projected; "
            f"whole response {len(json.dumps(full.structured_content))} -> "
            f"{len(json.dumps(projected.structured_content))}"
        )
    # The row table is what projection governs; the rest of the envelope
    # (source hashes, aggregated warnings) is fixed overhead it does not claim.
    assert projected_chars * 5 < full_chars


def _hierarchical_op_raw(work_dir: Path) -> Path:
    """An .op raw whose trace names carry dots of their own.

    ngspice spells a subcircuit node ``v(x1.out)`` and a subcircuit device
    parameter ``@m.x1.m1[gm]``, so the values an analysis of a hierarchical
    design is after are named with the projector's own separator.
    """
    raw = work_dir / "hierarchical_op.raw"
    raw.write_text(
        "Title: * hierarchy\n"
        "Date: Thu Jul 10 12:00:00 2026\n"
        "Plotname: Operating Point\n"
        "Flags: real\n"
        "No. Variables: 3\n"
        "No. Points: 1\n"
        "Offset: 0.0000000000000000e+00\n"
        # The dialect line a real .raw carries; without it the reader refuses
        # the file rather than guessing at its number format.
        "Command: Linear Technology Corporation LTspice\n"
        "Variables:\n"
        "\t0\tV(x1.out)\tvoltage\n"
        "\t1\tV(out)\tvoltage\n"
        "\t2\t@m.x1.m1[gm]\tadmittance\n"
        "Values:\n"
        "0\t1.2500000000000000e+00\n"
        "\t9.0000000000000000e-01\n"
        "\t3.1000000000000000e-03\n"
    )
    return raw


_OP_RECIPE: dict[str, Any] = {"key": "bias", "metric": "operating_point"}


@pytest.mark.asyncio
async def test_projection_reaches_a_key_whose_own_name_contains_a_dot(
    state_no_sim: SessionState,
    work_dir: Path,
):
    r"""The one projection target these tools exist to serve must be addressable.

    Splitting on every dot addresses a nesting a subcircuit device parameter
    does not have, which left the most ordinary op-point key unreachable by the
    payload lever advertised for exactly this kind of wide result. ``\.`` says
    the dot belongs to the key.
    """
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(
        state_no_sim,
        raw,
        [_OP_RECIPE],
        include={
            "fields": [
                r"value.device_op_points.@m\.x1\.m1[gm]",
                r"value.voltages.V(x1\.out)",
            ]
        },
    )

    entry = data["results"]["bias"]
    assert entry["values"] == [
        {
            "value": {
                "device_op_points": {"@m.x1.m1[gm]": pytest.approx(3.1e-3)},
                "voltages": {"V(x1.out)": pytest.approx(1.25)},
            }
        }
    ]
    assert not entry["warnings"]


@pytest.mark.asyncio
async def test_the_bias_point_survives_the_default_answer_channel(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """An operating_point row IS its keyed buckets — leaning them away leaves a
    successful call carrying nothing but the step counter it was asked about.

    The lean row keeps a value's scalar leaves and drops its nested ones, which
    is right for a metric whose numbers are the scalars and exactly wrong for
    one whose scalars are ``step``/``step_count`` and whose numbers are all
    nested. A caller asking for the bias point got ``{step, step_count,
    device}`` back with outcome 'complete'.
    """
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(state_no_sim, raw, [_OP_RECIPE])

    value = data["results"]["bias"]["values"][0]["value"]
    assert value["voltages"]["V(x1.out)"] == pytest.approx(1.25)
    assert value["device_op_points"]["@m.x1.m1[gm]"] == pytest.approx(3.1e-3)


def _bare_device_op_raw(work_dir: Path) -> Path:
    """An .op raw proving a MOSFET is in the circuit and carrying no params for
    it — a deck run without ``.options logopinfo`` / without ``.save``."""
    raw = work_dir / "bare_device_op.raw"
    raw.write_text(
        "Title: * bare\n"
        "Date: Thu Jul 10 12:00:00 2026\n"
        "Plotname: Operating Point\n"
        "Flags: real\n"
        "No. Variables: 2\n"
        "No. Points: 1\n"
        "Offset: 0.0000000000000000e+00\n"
        "Command: Linear Technology Corporation LTspice\n"
        "Variables:\n"
        "\t0\tV(out)\tvoltage\n"
        "\t1\tId(M1)\tdevice_current\n"
        "Values:\n"
        "0\t9.0000000000000000e-01\n"
        "\t1.5200000000000000e-05\n"
    )
    return raw


@pytest.mark.asyncio
async def test_a_value_recipe_accepts_the_documented_device_param_shorthand(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """``read_device_op_points`` and the guide both tell a caller to address a
    device parameter as ``m1.gm``. Only the literal ``@m1[gm]`` resolved here —
    and for an LTspice log-sourced param this path is the only route, so the
    documented spelling was refused everywhere it was the one that works."""
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "gm", "metric": "value", "expr": "m.x1.m1.gm"}],
    )

    entry = data["results"]["gm"]["values"][0]["value"]
    assert entry["signal"] == "@m.x1.m1[gm]"
    assert entry["value"] == pytest.approx(3.1e-3)


@pytest.mark.asyncio
async def test_an_unresolvable_op_point_value_names_the_forms_that_work(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": "nope", "metric": "value", "expr": "m9.gm"}],
    )

    message = next(
        failure["message"] for failure in data["failures"] if failure["code"] == "recipe_failed"
    )
    assert "m1.gm" in message
    assert "@m.x1.m1[gm]" in message


@pytest.mark.asyncio
async def test_a_bias_point_with_no_device_params_says_so_on_observations(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """Never a bare success: the caller who asked for a device's operating point
    and got none must be told what was looked for and where it was looked."""
    raw = _bare_device_op_raw(work_dir)

    data = await _analyze(state_no_sim, raw, [_OP_RECIPE])

    observation = next(
        item for item in data["observations"] if item["code"] == "device_op_points_absent"
    )
    assert observation["evidence"]["recipe"] == "bias"
    assert "logopinfo" in observation["detail"]
    assert ".save" in observation["detail"]


@pytest.mark.asyncio
async def test_a_bias_point_that_found_device_params_stays_note_free(
    state_no_sim: SessionState,
    work_dir: Path,
):
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(state_no_sim, raw, [_OP_RECIPE])

    assert not [item for item in data["observations"] if item["code"] == "device_op_points_absent"]


@pytest.mark.asyncio
async def test_unreachable_parent_says_the_key_is_one_segment(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """The warning has to describe the reason it missed, not invent one.

    Reading 'value.voltages.V(x1' as a parent that 'holds no object' names a
    path the caller never wrote and sends them looking for a missing dict, when
    the dict is there and the key simply owns the dot.
    """
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(
        state_no_sim,
        raw,
        [_OP_RECIPE],
        include={"fields": ["value.voltages.V(x1.out)"]},
    )

    warning = next(text for text in data["results"]["bias"]["warnings"] if "V(x1.out)" in text)
    assert "holds no object" not in warning
    assert r"'\.'" in warning


@pytest.mark.asyncio
async def test_present_keys_are_reported_as_they_must_be_spelled(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """A key listed as present must be listed as addressable: echoing the raw
    name of a dotted key hands back a path that misses again."""
    raw = _hierarchical_op_raw(work_dir)

    data = await _analyze(
        state_no_sim,
        raw,
        [_OP_RECIPE],
        include={"fields": ["value.device_op_points.absent"]},
    )

    warning = next(text for text in data["results"]["bias"]["warnings"] if "absent" in text)
    assert r"@m\.x1\.m1[gm]" in warning


@pytest.mark.asyncio
async def test_identical_record_warnings_collapse_but_keep_their_reach(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """One warning raised by every record shipped 45 identical copies — 9,225
    characters of a 59,268-character response saying one thing. It collapses to
    a single line, and the line still says how many records raised it, because
    45-of-45 and 3-of-45 are different facts."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    result = await handle_analyze_results(_wide_args(raw), state_no_sim)
    assert result.structured_content is not None
    entry = result.structured_content["results"]["loop"]
    assert len(entry["values"]) == 45
    assert len(entry["warnings"]) == 1
    assert entry["warnings"][0].endswith(" (45 of 45 records)")


def _record(value: dict[str, Any]) -> analyze_mod.Record:
    """A minimal evaluated record carrying ``value``, for the row-level helpers."""
    return analyze_mod.Record(
        manifest_id="m", source="dut", identity=analyze_mod.RowIdentity(), value=value
    )


def test_differing_record_warnings_all_survive_in_first_seen_order():
    """Collapsing is by exact text: two different warnings are two facts, and
    the count is per record even when one record repeats itself."""
    records = [
        _record({"warnings": ["clamped window", "clamped window", "ambiguous edge"]}),
        _record({"warnings": ["clamped window"]}),
        _record({"warnings": []}),
    ]
    assert analyze_mod._record_warnings(records) == [
        "clamped window (2 of 3 records)",
        "ambiguous edge (1 of 3 records)",
    ]


def test_single_record_warnings_carry_no_count():
    """ "1 of 1" states nothing, so a single-run result reads exactly as before."""
    assert analyze_mod._record_warnings([_record({"warnings": ["clamped window"]})]) == [
        "clamped window"
    ]


class TestSourceHashProvenance:
    """Artifact paths and digests prove what was analyzed; they are not how a
    caller reaches it. In one measured response they were 1,174 of the 7,835
    characters, naming files the analysis tools resolve by manifest_id anyway."""

    @pytest.mark.asyncio
    async def test_paths_and_digests_are_opt_in(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        recipes = [{"key": "s", "metric": "summary"}]
        lean = await _analyze(state_no_sim, raw, recipes)
        entry = lean["source_hashes"][0]
        assert entry["manifest_id"]
        # The default is exactly the attribution legend rows join against —
        # anything past {manifest_id, label} is provenance and opt-in.
        assert set(entry) == {"manifest_id", "label"}

        full = await _analyze(state_no_sim, raw, recipes, include={"provenance": True})
        full_entry = full["source_hashes"][0]
        assert full_entry["raw_path"]
        assert full_entry["composite_sha256"] or full_entry["raw_sha256"]
        assert "log_present" in full_entry and "job_id" in full_entry
        assert len(json.dumps(full)) > len(json.dumps(lean))


class TestHeadlineLeafPromotion:
    """A sweep's table is read from per_run rows, and dotted projection cannot
    reach into lists — so a headline that lives only inside points[]/
    crossings[] is unprojectable below the whole value block (measured at
    13-22x the shell-equivalent size for the same 36 numbers). Each such
    metric promotes its headline to a flat value leaf, through the reducer's
    own extractor, so leaf and reduction cannot disagree."""

    @pytest.mark.asyncio
    async def test_crossing_and_point_rows_carry_flat_headlines(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [
                {"key": "gbw", "metric": "bode_crossing", "signal": "V(out)", "level_db": -3.0},
                {"key": "dc", "metric": "bode_point", "signal": "V(out)", "at_hz": "1"},
            ],
            # fields=["value"] fetches the WHOLE value block — this test checks
            # the promoted flat leaf agrees with the nested detail it came from.
            include={"per_run": {"limit": 5}, "fields": ["value"]},
        )
        crossing_row = data["results"]["gbw"]["per_run"]["items"][0]["value"]
        assert crossing_row["first_crossing_hz"] == crossing_row["crossings"][0]["frequency_hz"]
        assert "crossings_found" not in crossing_row, (
            "the adapter caps its crossings list without a truncation signal, "
            "so a count leaf would silently saturate — it must not exist"
        )
        point_row = data["results"]["dc"]["per_run"]["items"][0]["value"]
        assert point_row["magnitude_db"] == point_row["points"][0]["magnitude_db"]

    @pytest.mark.asyncio
    async def test_phase_crossing_answers_on_the_degree_axis(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """ "Where does phase cross -45 degrees" is a recipe, not a numpy detour.

        On a single-pole RC the -45 deg phase crossing IS the -3 dB corner, so
        the two axes have to agree on the same raw.
        """
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [
                {"key": "ph", "metric": "bode_crossing", "signal": "V(out)", "level_deg": -45.0},
                {"key": "mag", "metric": "bode_crossing", "signal": "V(out)", "level_db": -3.0},
            ],
            include={"per_run": {"limit": 5}, "fields": ["value.first_crossing_hz"]},
        )
        phase_hz = data["results"]["ph"]["per_run"]["items"][0]["value"]["first_crossing_hz"]
        mag_hz = data["results"]["mag"]["per_run"]["items"][0]["value"]["first_crossing_hz"]
        assert phase_hz == pytest.approx(mag_hz, rel=0.02)

    @pytest.mark.asyncio
    async def test_headline_is_projectable_to_a_lean_row(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The point of the promotion: a caller can now name the one number."""
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "gbw", "metric": "bode_crossing", "signal": "V(out)", "level_db": -3.0}],
            include={
                "per_run": {"limit": 5},
                "fields": ["step_values", "value.first_crossing_hz"],
            },
        )
        rows = data["results"]["gbw"]["per_run"]["items"]
        assert rows, "expected per_run rows"
        for row in rows:
            assert set(row) <= {"step_values", "value"}
            assert set(row["value"]) == {"first_crossing_hz"}
            assert isinstance(row["value"]["first_crossing_hz"], float)

    @pytest.mark.asyncio
    async def test_stability_rows_carry_the_crossover_frequency_flat(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """ "Give me the UGBW and phase margin" is one question, not two.

        The margins shipped flat and the crossover frequency did not, so the
        default row answered half the question and the other half cost a second
        call for the whole nested value.
        """
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "loop", "metric": "stability", "signal": "V(out)"}],
        )
        row = data["results"]["loop"]["values"][0]["value"]
        assert "unity_gain_hz" in row
        # This fixture's loop never reaches unity; the leaf says so with a null
        # rather than being absent, which is what makes it readable either way.
        assert row["unity_gain_hz"] is None
        assert row["stability"] == "always_below_unity"

    @pytest.mark.asyncio
    async def test_stability_reduces_dc_gain_and_specs_the_crossover(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """The per-case fields a caller reads must also be the ones it can
        reduce and spec; a spec on a field the loop never reaches is
        indeterminate, not a pass."""
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [
                {
                    "key": "gain",
                    "metric": "stability",
                    "signal": "V(out)",
                    "reduce": ["max"],
                    "reduce_field": "dc_gain_db",
                },
                {
                    "key": "ugbw",
                    "metric": "stability",
                    "signal": "V(out)",
                    "spec": {"field": "unity_gain_hz", "min": 1e3},
                },
                {"key": "row", "metric": "stability", "signal": "V(out)"},
            ],
        )
        reduced = next(
            entry for entry in data["results"]["gain"]["reduced"] if entry["stat"] == "max"
        )
        assert reduced["field"] == "dc_gain_db"
        # The reduced number is the same leaf the unreduced row shows.
        assert reduced["value"] == data["results"]["row"]["values"][0]["value"]["dc_gain_db"]
        # This fixture's loop never reaches unity, so the spec has no sample.
        assert data["results"]["ugbw"]["spec"]["verdict"] == "indeterminate"

    def test_stability_headline_is_the_first_crossover_in_sweep_order(self):
        """The recorded fixture has no crossover at all, so the end-to-end test
        cannot tell the extraction rule from a hardcoded null."""
        from ltspice_mcp.tools.analyze import _promote_headlines

        value = {
            "unity_gain_crossovers": [
                {"frequency_hz": 1.2e6, "direction": "falling"},
                {"frequency_hz": 8.0e6, "direction": "rising"},
            ]
        }
        assert _promote_headlines("stability", value)["unity_gain_hz"] == 1.2e6

    def test_promotion_never_overwrites_an_existing_key(self):
        """setdefault contract: if an adapter ever grows its own flat leaf with
        the same name, the adapter's value wins and promotion becomes a no-op —
        not a silent overwrite of fresher data."""
        from ltspice_mcp.tools.analyze import _promote_headlines

        value = {"crossings": [{"frequency_hz": 42.0}], "first_crossing_hz": 7.0}
        out = _promote_headlines("bode_crossing", value)
        assert out["first_crossing_hz"] == 7.0

    def test_first_crossing_means_first_in_sweep_order(self):
        """The recorded RC fixture has exactly one crossing, so the end-to-end
        tests cannot tell crossings[0] from crossings[-1] — a mutation swapping
        them survived. The rule is pure, so pin it on a two-crossing value."""
        from ltspice_mcp.tools.analyze import _promote_headlines

        value = {"crossings": [{"frequency_hz": 42.0}, {"frequency_hz": 99.0}]}
        out = _promote_headlines("bode_crossing", value)
        assert out["first_crossing_hz"] == 42.0


@pytest.mark.asyncio
@pytest.mark.parametrize(("metric", "fixture_name", "fields"), EXECUTION_CASES)
async def test_every_metric_exposes_a_flat_numeric_headline(
    metric: str,
    fixture_name: str,
    fields: dict[str, Any],
    state_no_sim: SessionState,
    work_dir: Path,
):
    """The class pin behind headline promotion: a metric whose numbers live
    ONLY inside nested structure is invisible to ``include.fields`` projection
    (dotted paths cannot reach into lists), which costs 13-22x the equivalent
    shell output. Every metric must expose at least one flat numeric leaf on
    its row value — or sit on this explicit exemption list,
    which fails CLOSED: removing a metric's flatness without adding it here
    breaks this test, and an exemption for a metric that IS flat is dead
    weight that also fails."""
    # Whole-row payloads whose value is a keyed BUNDLE the caller projects by
    # name (measurements: per-.meas stats), not a single measurement with a
    # headline. operating_point is NOT here only because 'step'/'step_count'
    # are numeric — bookkeeping, not its answer, which is why the answer
    # channel has to keep its buckets whole (_WHOLE_VALUE_METRICS) rather than
    # trust this test to notice their loss.
    exempt = {"measurements"}
    raw = stage_recorded_fixture(work_dir, fixture_name)
    data = await _analyze(
        state_no_sim,
        raw,
        [{"key": metric, "metric": metric, **fields}],
        include={"per_run": {"limit": 3}},
    )
    if metric not in data["results"]:
        pytest.skip("metric legitimately found no feature in this fixture")
    rows = data["results"][metric]["per_run"]["items"]
    if not rows:
        pytest.skip("no per-run rows for this fixture")
    value = rows[0]["value"]
    flat_numeric = any(
        isinstance(v, int | float) and not isinstance(v, bool) for v in value.values()
    )
    if metric in exempt:
        assert not flat_numeric, (
            f"{metric} now exposes a flat numeric leaf; remove it from the "
            "exemption list so coverage does not silently shrink"
        )
    else:
        assert flat_numeric, (
            f"{metric} rows carry no flat numeric leaf — its headline is "
            "unprojectable; add a _HEADLINE_LEAVES entry (see bode_crossing)"
        )


@pytest.mark.asyncio
async def test_measurements_recipe_bins_the_distribution_on_request(
    state_no_sim: SessionState,
    work_dir: Path,
):
    """Binning is reachable from the consolidated door, not just the legacy tool.

    A Monte Carlo's spread is read off the histogram; hard-coding zero bins
    here left one of MCP and the Python API unable to ask for it at all.
    """
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    shutil.copy(FIXTURES_DIR / "ltspice_step_when.log", raw.with_suffix(".log"))

    binned = await _analyze(
        state_no_sim,
        raw,
        [{"key": "m", "metric": "measurements", "histogram_bins": 3}],
        include={"per_run": {"limit": 3}},
    )
    unbinned = await _analyze(
        state_no_sim,
        raw,
        [{"key": "m", "metric": "measurements"}],
        include={"per_run": {"limit": 3}},
    )

    entry = binned["results"]["m"]["per_run"]["items"][0]["value"]["stats"]["vfinal"]
    assert len(entry["histogram"]) == 3
    assert sum(item["count"] for item in entry["histogram"]) == entry["valid_count"] == 3
    plain = unbinned["results"]["m"]["per_run"]["items"][0]["value"]["stats"]["vfinal"]
    assert plain["histogram"] == []


# ---------------------------------------------------------------------------
# artifact handles reach the caller
# ---------------------------------------------------------------------------


def _artifact_of(data: dict[str, Any], key: str) -> dict[str, Any]:
    """The one artifact handle a recipe's single row carries, and its file."""
    block = data["results"][key]
    rows = block.get("values") or block.get("per_run", {}).get("items") or []
    assert rows, f"{key} returned no rows: {block}"
    artifact = rows[0]["value"].get("artifact")
    assert isinstance(artifact, dict), (
        f"{key} row carries no artifact handle — the file it wrote is unreachable: "
        f"{rows[0]['value']}"
    )
    assert Path(artifact["path"]).is_file(), f"{key} handle names no file: {artifact}"
    return artifact


class TestArtifactHandleSurvivesTheDefaultRow:
    """A handle names a file this call already wrote; drop it and the file is
    unreachable. The lean row flattened value to its scalar leaves, so the plot
    recipe answered "plot this" with a series count and nothing else."""

    @pytest.mark.asyncio
    async def test_plot_returns_its_artifact_on_the_default_row(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "p", "metric": "plot", "signals": ["V(out)"], "title": "RC step"}],
        )
        assert _artifact_of(data, "p")["content_type"] == "text/html"

    @pytest.mark.asyncio
    async def test_waveform_csv_returns_its_artifact_too(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "w", "metric": "waveform", "signals": ["V(out)"], "format": "csv"}],
        )
        assert _artifact_of(data, "w")["content_type"] == "text/csv"

    @pytest.mark.asyncio
    async def test_a_projected_row_can_still_ask_for_the_handle_by_name(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _analyze(
            state_no_sim,
            raw,
            [{"key": "p", "metric": "plot", "signals": ["V(out)"]}],
            include={"fields": ["value.artifact"]},
        )
        assert _artifact_of(data, "p")["content_type"] == "text/html"


# ---------------------------------------------------------------------------
# include spellings
# ---------------------------------------------------------------------------


class TestIncludeFlagList:
    """A bare list of flag names is the spelling callers write first."""

    @staticmethod
    def _include(value: Any):
        return AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": "r.raw", "label": "dut"}],
                "recipes": [{"key": "s", "metric": "summary"}],
                "include": value,
            }
        ).include

    def test_flag_list_switches_the_named_blocks_on(self):
        include = self._include(["outliers", "signals_available"])
        assert include.outliers is True
        assert include.signals_available is True
        assert include.provenance is False
        assert include.per_run is None

    def test_flag_list_reaches_per_run_with_its_default_page(self):
        include = self._include(["per_run"])
        assert include.per_run is not None
        assert include.per_run.limit == 50
        assert include.per_run.cursor is None

    def test_object_spelling_is_unchanged(self):
        include = self._include({"per_run": {"limit": 3}, "outliers": True})
        assert include.per_run is not None
        assert include.per_run.limit == 3
        assert include.outliers is True

    def test_per_run_true_is_the_default_page(self):
        include = self._include({"per_run": True})
        assert include.per_run is not None
        assert include.per_run.limit == 50
        assert self._include({"per_run": False}).per_run is None

    def test_unknown_flag_name_is_rejected_and_enumerates(self):
        with pytest.raises(ValidationError) as excinfo:
            self._include(["signals", "outliers"])
        detail = compact_validation_error(excinfo.value)
        assert "signals" in detail
        assert "signals_available" in detail
        assert "provenance" in detail

    def test_fields_is_named_as_the_one_that_needs_values(self):
        with pytest.raises(ValidationError) as excinfo:
            self._include(["fields"])
        detail = compact_validation_error(excinfo.value)
        assert "'fields' takes row paths" in detail

    def test_attached_analysis_takes_the_same_spellings(self):
        args = experiments.RunExperimentsInput.model_validate(
            {
                "circuits": [{"path": "deck.cir"}],
                "analyze": {
                    "recipes": [{"key": "s", "metric": "summary"}],
                    "include": ["per_run", "outliers"],
                },
            }
        )
        assert args.analyze is not None
        assert args.analyze.include is not None
        assert args.analyze.include.outliers is True
        assert args.analyze.include.per_run is not None
        assert args.analyze.include.per_run.limit == 50
