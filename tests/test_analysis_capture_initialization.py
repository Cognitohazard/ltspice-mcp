"""Analysis initialization binds diagnostics and refuses temporary capture timeouts."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import pytest

from ltspice_mcp.errors import AnalysisDeadlineExceeded
from ltspice_mcp.lib import parser_service, services
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
                "sources": [{"path": str(raw.with_suffix(".log")), "label": "log"}],
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
    # The recipe keeps its entry, with no rows and a pointer to the failure.
    meas = data["results"]["meas"]
    assert (
        not meas.get("values")
        and not meas.get("per_run", {}).get("items")
        and any(warning.startswith("No rows:") for warning in meas["warnings"])
    )
    assert any(failure["code"] == "source_drift" for failure in data["failures"])
    failures = [item for item in data.get("observations", []) if item["code"] == "solve_failure"]
    assert bool(failures) == bool(before)


@pytest.mark.asyncio
async def test_elapsed_initial_capture_has_no_set_and_fresh_retry_succeeds(
    state_no_sim: SessionState,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    settled_stamps: None,
):
    """A capture that returns after the budget is spent creates no result set.

    The log is loaded first, so the capture inside the call is answered from
    its stamp rather than a parser process the budget would have to cover;
    the fake work after it then runs until the call's own deadline has passed.
    """
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    raw.unlink()
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw.with_suffix(".log")), "label": "log"}],
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
        loop = asyncio.get_running_loop()
        deadline = services._analysis_deadline.get()
        assert deadline is not None
        # The condition is the clock itself, and a timer may fire early.
        while loop.time() < deadline:  # noqa: ASYNC110
            # timing: fake work that runs until the call's own deadline has passed
            await asyncio.sleep(deadline - loop.time())
        return artifacts

    state_no_sim.config.analysis_budget_s = 1.0
    existing = set(state_no_sim.store.results_dir.glob("*.json"))
    log_source = services.resolve_analysis_source(
        state_no_sim, log_file=str(raw.with_suffix(".log"))
    )
    await services.load_artifacts(log_source, state_no_sim, require_raw=False)
    parser_requests: list[str] = []
    run_parser = parser_service.run_parser_sync

    def counted(request, **kwargs):
        parser_requests.append(request["op"])
        return run_parser(request, **kwargs)

    monkeypatch.setattr(parser_service, "run_parser_sync", counted)
    monkeypatch.setattr(services, "load_artifacts", consume_budget_after_capture)
    with pytest.raises(AnalysisDeadlineExceeded, match="initialization"):
        await analyze.handle_analyze_results(request, state_no_sim)
    assert captures and captures[0].logs.value("measurements") is not None
    assert parser_requests == []
    assert set(state_no_sim.store.results_dir.glob("*.json")) == existing

    monkeypatch.setattr(services, "load_artifacts", original)
    state_no_sim.config.analysis_budget_s = 60.0
    response = await analyze.handle_analyze_results(request, state_no_sim)
    data = response.structured_content
    assert data is not None and "failures" not in data
    row = data["results"]["meas"]["per_run"]["items"][0]
    assert row["value"]["measured"]["vfinal"] == LTSPICE_TRAN_RC_VFINAL


@pytest.mark.asyncio
async def test_the_signal_inventory_reads_no_raw_of_a_log_source(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """The signal inventory, which has a budget of its own, reads a raw only
    for a source that has one."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    request = analyze.AnalyzeResultsInput.model_validate(
        {
            "sources": [{"path": str(raw.with_suffix(".log")), "label": "log"}],
            "recipes": [{"metric": "measurements", "key": "meas"}],
            "include": {"signals_available": True, "per_run": True},
        }
    )
    original_raw = services.load_raw
    raw_reads: list[services.AnalysisSource] = []

    async def record_raw_read(source: services.AnalysisSource, state: SessionState):
        raw_reads.append(source)
        return await original_raw(source, state)

    monkeypatch.setattr(services, "load_raw", record_raw_read)
    response = await analyze.handle_analyze_results(request, state_no_sim)
    data = response.structured_content
    assert data is not None and "failures" not in data
    assert data["results"]["meas"]["per_run"]["items"]
    assert raw_reads == []
    # The log holds no signal, and a reply leaves out what says nothing.
    assert "signals_available" not in data
