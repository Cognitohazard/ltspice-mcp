"""Captured identities retain source selection while avoiding duplicate work."""

from __future__ import annotations

import asyncio
import os
import time
import weakref
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import parser_service, response_budget, result_store, services
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.result_cache import ResultCache
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analysis, analyze
from tests.conftest import LTSPICE_TRAN_RC_VFINAL, stage_recorded_fixture


@pytest.mark.asyncio
async def test_drive_does_not_retain_artifacts_evicted_from_result_cache(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    other = work_dir / "other.raw"
    other.write_bytes(raw.read_bytes())
    other.with_suffix(".log").write_bytes(raw.with_suffix(".log").read_bytes())
    other.with_suffix(".exe.log").write_bytes(b"different companion identity\n")
    state_no_sim.results = ResultCache(max_entries=1)
    references: list[weakref.ReferenceType[ParsedArtifacts]] = []
    original_metric = analyze._adapter_value
    original_publish = analyze._publish_unit_artifacts

    async def record_resident(recipe, source, step, state):
        value = await original_metric(recipe, source, step, state)
        assert source.captured is not None
        references.append(weakref.ref(source.captured))
        return value

    async def check_retention(processed, pending, deadline):
        assert len(references) == 6
        assert sum(reference() is not None for reference in references) <= 3
        await original_publish(processed, pending, deadline)

    monkeypatch.setattr(analyze, "_adapter_value", record_resident)
    monkeypatch.setattr(analyze, "_publish_unit_artifacts", check_retention)
    response = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(path)} for path in (raw, other)],
                "recipes": [{"key": "loop", "metric": "bode_filter", "signal": "V(out)"}],
                "all_steps": True,
                "include": {"per_run": True},
            }
        ),
        state_no_sim,
    )
    data = response.structured_content
    assert data is not None and data["failures"] == []
    assert data["results"]["loop"]["per_run"]["total"] == 6


@pytest.mark.asyncio
async def test_sync_readers_use_explicitly_captured_source(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    original = parser_service.run_parser_sync
    operations: list[str] = []

    def record_worker(request, **kwargs):
        operations.append(request["op"])
        return original(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", record_worker)
    source = services.source_for_raw_path(raw, state_no_sim)
    artifacts = await services.load_artifacts(source, state_no_sim, require_raw=True)
    source = replace(source, identity={"snapshot_id": artifacts.snapshot_id}, captured=artifacts)
    assert await asyncio.to_thread(services.load_raw_sync, source, state_no_sim) is artifacts.raw
    assert await asyncio.to_thread(services.load_logs_sync, source, state_no_sim) is artifacts.logs
    assert len(operations) == 1


@pytest.mark.asyncio
async def test_budget_pages_share_resident_capture_and_reverify_each_continuation(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    settled_stamps: None,
):
    """Fifteen references to one raw share one capture. Each continuation
    re-stamps the source: unchanged, it is answered without a parser process;
    rewritten, even at the same size and modification time, a worker recaptures
    it and every page reports the drift."""
    raw = stage_recorded_fixture(work_dir, "ltspice_step_ac")
    original = parser_service.run_parser_sync
    operations: list[str] = []

    def record_worker(request, **kwargs):
        operations.append(request["op"])
        return original(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", record_worker)
    request = {
        "sources": [{"raw_path": str(raw), "label": f"corner{index}"} for index in range(15)],
        "recipes": [{"key": "loop", "metric": "bode_filter", "signal": "V(out)"}],
        "all_steps": True,
        "budget": response_budget.BUDGET_MIN_TOKENS,
        "include": {"per_run": {"limit": 45}, "provenance": True, "signals_available": True},
    }
    first = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(request), state_no_sim
    )
    data = first.structured_content
    assert data is not None and data["failures"] == []
    page = data["results"]["loop"]["per_run"]
    assert page["total"] == 45 and 0 < page["returned"] < 45
    assert page["items"][0]["value"]["cutoff_high_hz"] > 0
    assert operations.count("load_raw") == 1
    assert len(operations) <= 2
    assert data["next"] is not None

    operations.clear()
    resumed = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(
            {"continue": data["next"], "budget": response_budget.BUDGET_MIN_TOKENS}
        ),
        state_no_sim,
    )
    continued = resumed.structured_content
    assert continued is not None and continued["failures"] == []
    next_page = continued["results"]["loop"]["per_run"]
    offset = page["returned"]
    assert (next_page["items"][0]["source"], next_page["items"][0]["step_index"]) == (
        f"corner{offset // 3}",
        offset % 3,
    )
    assert operations == []
    assert continued["next"] is not None

    before = raw.stat()
    contents = bytearray(raw.read_bytes())
    contents[-1] ^= 1
    raw.write_bytes(contents)
    os.utime(raw, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert raw.stat().st_size == before.st_size
    operations.clear()
    drifted = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(
            {"continue": continued["next"], "budget": response_budget.BUDGET_MIN_TOKENS}
        ),
        state_no_sim,
    )
    rejected = drifted.structured_content
    assert rejected is not None and "loop" not in rejected["results"]
    assert len(rejected["failures"]) == 15
    assert all(failure["code"] == "source_drift" for failure in rejected["failures"])
    assert len(operations) <= 2
    assert "load_raw" not in operations


@pytest.mark.asyncio
async def test_capture_and_verification_share_bytes_but_keep_plot_selection(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [
                {"raw_path": str(raw), "plot_index": 0},
                {"raw_path": str(raw), "plot_index": 1},
            ],
            "recipes": [{"metric": "measurements", "key": "meas"}],
        }
    )
    original = services.load_artifacts
    captures = []

    async def record_capture(source, state, *, require_raw):
        captured = await original(source, state, require_raw=require_raw)
        captures.append(captured)
        return captured

    monkeypatch.setattr(services, "load_artifacts", record_capture)
    item = await analyze._create_result_set(request, state_no_sim, time.monotonic() + 30)
    assert len(captures) == 1
    first, second = item.source_manifests
    assert first["label"] == raw.stem
    assert second["label"] == f"{raw.stem}-2"
    assert first["snapshot_id"] == second["snapshot_id"] == captures[0].snapshot_id
    assert (first["plot_index"], second["plot_index"]) == (0, 1)
    assert first["selection_sha256"] != second["selection_sha256"]
    manifest_ids = {row["manifest_id"] for row in item.source_manifests}
    failures = await analyze._verify_direct_sources(
        item.source_manifests, manifest_ids, time.monotonic() + 30, state=state_no_sim
    )
    assert failures == {}
    assert len(captures) == 2
    raw.with_suffix(".exe.log").write_bytes(b"new console companion\n")
    failures = await analyze._verify_direct_sources(
        item.source_manifests, manifest_ids, time.monotonic() + 30, state=state_no_sim
    )
    assert set(failures) == manifest_ids
    assert all(fault.code == "source_drift" for fault in failures.values())


@pytest.mark.asyncio
async def test_unlabelled_log_continuation_echo_keeps_the_immutable_record(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.unlink()
    request = {
        "sources": [{"log_path": str(raw.with_suffix(".log")), "runs": list(range(121))}],
        "recipes": [{"metric": "measurements", "key": "meas"}],
        "include": {"provenance": True, "per_run": True},
    }
    first = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(request), state_no_sim
    )
    data = first.structured_content
    assert data is not None and data["failures"] == []
    row = data["results"]["meas"]["per_run"]["items"][0]
    assert row["source"] == raw.stem
    assert row["value"]["measured"]["vfinal"] == LTSPICE_TRAN_RC_VFINAL
    manifest = data["source_hashes"][0]
    assert manifest["raw_present"] is False and manifest["log_sha256"]
    path = result_store.result_path(data["result_set_id"], work_dir)
    before = path.read_bytes()
    cursor = data["coverage"]["missing_cases"]["next_cursor"]
    assert cursor is not None
    continuation = {"result_set_id": data["result_set_id"], "cursor": cursor}
    resumed = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate({**request, "continue": continuation}),
        state_no_sim,
    )
    continued = resumed.structured_content
    assert continued is not None and continued["failures"] == []
    assert continued["coverage"]["missing_cases"]["returned"] == 20
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_stepped_plot_keeps_its_distinct_axes_without_union_facts(
    state_no_sim: SessionState, work_dir: Path
):
    raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
    response = await analysis.handle_plot_waveform(
        analysis.PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False),
        state_no_sim,
    )
    data = response.structured_content
    assert data is not None
    assert data["steps_plotted"] == data["series_count"] == data["n_steps"] > 1
    assert data["observations"] == []
    assert await asyncio.to_thread(Path(data["path"]).is_file)
