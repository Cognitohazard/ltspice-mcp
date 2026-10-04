"""Analysis initialization binds diagnostics and refuses temporary capture timeouts."""

from __future__ import annotations

import asyncio
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded
from ltspice_mcp.lib import result_store, services
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from tests.conftest import LTSPICE_TRAN_RC_VFINAL, stage_recorded_fixture


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("before", "after"),
    [
        (b"Iteration limit reached; no convergence.\n", b""),
        (b"", b"Iteration limit reached; no convergence.\n"),
    ],
)
async def test_console_mutation_cannot_rebind_relay_to_new_manifest(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    before: bytes,
    after: bytes,
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.unlink()
    console = raw.with_suffix(".exe.log")
    console.write_bytes(before)
    original = services.load_artifacts
    captures: list[ParsedArtifacts] = []

    async def mutate_after_capture(
        source: services.AnalysisSource, state: SessionState, *, require_raw: bool
    ) -> ParsedArtifacts:
        artifacts = await original(source, state, require_raw=require_raw)
        captures.append(artifacts)
        if len(captures) == 1:
            console.write_bytes(after)
        return artifacts

    monkeypatch.setattr(services, "load_artifacts", mutate_after_capture)
    response = await analyze.handle_analyze_results(
        analyze.AnalyzeResultsInput.model_validate(
            {
                "sources": [{"log_path": str(raw.with_suffix(".log")), "label": "log"}],
                "recipes": [{"metric": "measurements", "key": "meas"}],
                "include": {"provenance": True},
            }
        ),
        state_no_sim,
    )
    data = response.structured_content
    assert data is not None
    manifest = data["source_hashes"][0]
    assert manifest["snapshot_id"] == captures[0].snapshot_id
    assert manifest["console_sha256"] == hashlib.sha256(before).hexdigest()
    assert "meas" not in data["results"]
    assert any(failure["code"] == "source_drift" for failure in data["failures"])
    failures = [item for item in data["observations"] if item["code"] == "solve_failure"]
    assert bool(failures) == bool(before)


@pytest.mark.asyncio
async def test_elapsed_initial_capture_has_no_set_and_fresh_retry_succeeds(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.unlink()
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [{"log_path": str(raw.with_suffix(".log")), "label": "log"}],
            "recipes": [{"metric": "measurements", "key": "meas"}],
            "include": {"per_run": True},
        }
    )
    original = services.load_artifacts
    captures: list[ParsedArtifacts] = []

    async def consume_budget_after_capture(
        source: services.AnalysisSource, state: SessionState, *, require_raw: bool
    ) -> ParsedArtifacts:
        artifacts = await original(source, state, require_raw=require_raw)
        captures.append(artifacts)
        await asyncio.sleep(1.1)
        return artifacts

    state_no_sim.config.analysis_budget_s = 1.0
    existing = set(state_no_sim.store.results_dir.glob("*.json"))
    monkeypatch.setattr(services, "load_artifacts", consume_budget_after_capture)
    started = time.monotonic()
    with pytest.raises(AnalysisDeadlineExceeded, match="initialization"):
        await analyze.handle_analyze_results(request, state_no_sim)
    assert captures and captures[0].logs.value("measurements") is not None
    assert time.monotonic() - started < 3.0
    assert set(state_no_sim.store.results_dir.glob("*.json")) == existing

    monkeypatch.setattr(services, "load_artifacts", original)
    state_no_sim.config.analysis_budget_s = 60.0
    response = await analyze.handle_analyze_results(request, state_no_sim)
    data = response.structured_content
    assert data is not None and data["failures"] == []
    row = data["results"]["meas"]["per_run"]["items"][0]
    assert row["value"]["stats"]["vfinal"]["mean"] == LTSPICE_TRAN_RC_VFINAL


@pytest.mark.asyncio
async def test_optional_signal_inventory_does_not_start_after_call_budget(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [{"raw_path": str(raw), "label": "log"}],
            "recipes": [{"metric": "measurements", "key": "meas"}],
            "include": {"signals_available": True, "per_run": True},
        }
    )
    item = await analyze._create_result_set(request, state_no_sim, time.monotonic() + 30)
    cursor = result_store.encode_cursor(item, 0, view_fields=request.include.fields)
    continued = analyze.AnalyzeResultsInput.model_validate(
        {"continue": {"result_set_id": item.result_set_id, "cursor": cursor}}
    )
    original_publish = analyze._publish_unit_artifacts
    original_raw = services.load_raw
    raw_reads: list[services.AnalysisSource] = []
    real_loop = asyncio.get_running_loop()
    expired_at: float | None = None

    def drive_time() -> float:
        now = real_loop.time()
        return max(now, expired_at) if expired_at is not None else now

    async def expire_after_publish(processed, pending, deadline):
        nonlocal expired_at
        await original_publish(processed, pending, deadline)
        expired_at = deadline + 1.0

    async def record_raw_read(source: services.AnalysisSource, state: SessionState):
        raw_reads.append(source)
        return await original_raw(source, state)

    monkeypatch.setattr(analyze, "_publish_unit_artifacts", expire_after_publish)
    monkeypatch.setattr(services, "load_raw", record_raw_read)
    # Replace only this module's clock lookup, never the scheduler or parser clock.
    monkeypatch.setattr(
        analyze,
        "asyncio",
        SimpleNamespace(
            get_running_loop=lambda: SimpleNamespace(time=drive_time),
            to_thread=asyncio.to_thread,
        ),
    )
    response = await analyze.handle_analyze_results(continued, state_no_sim)
    data = response.structured_content
    assert data is not None and data["failures"] == []
    assert data["results"]["meas"]["per_run"]["items"]
    assert raw_reads == []
    assert data["signals_available"] == {"log:0": []}
