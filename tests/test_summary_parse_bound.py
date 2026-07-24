"""Completion summaries return bounded, fact-only fallbacks when parsing stalls."""

import asyncio
import contextlib
import threading
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import now, services
from ltspice_mcp.state import SessionState, SimulationJob
from ltspice_mcp.tools import get_tools_for_profile, simulation
from ltspice_mcp.tools.simulation import (
    CheckJobInput,
    RunSimulationInput,
    _format_success_response,
    handle_check_job,
    handle_run_simulation,
)
from ltspice_mcp.tools.status import ServerStatusInput, handle_server_status


class SlowSummaryParser:
    """A controllable parser that stays busy after its caller abandons it."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.calls = 0
        self.entered = asyncio.Event()
        self.release = threading.Event()
        self._loop = loop

    def __call__(
        self,
        raw_file: Path,
        log_file: Path,
        duration: float,
        **kwargs,
    ) -> dict:
        self.calls += 1
        self._loop.call_soon_threadsafe(self.entered.set)
        self.release.wait(2.0)
        return {
            "sim_type": "Transient",
            "duration": duration,
            "step_count": 1,
            "raw_file": str(raw_file),
            "log_file": str(log_file),
            "signals": ["time", "V(out)"],
            "warnings": [],
        }


def _completed_job(
    state: SessionState,
    netlist: Path,
    raw_file: Path,
    log_file: Path,
    *,
    job_id: str = "summary_job",
) -> SimulationJob:
    started_at = now()
    job = SimulationJob(
        job_id=job_id,
        netlist=netlist,
        simulator="FakeSim",
        status="completed",
        started_at=started_at,
        completed_at=started_at + timedelta(seconds=2),
        raw_file=raw_file,
        log_file=log_file,
    )
    state.add_job(job)
    return job


def _assert_fact_only_fallback(data: dict, job: SimulationJob) -> None:
    assert data["job_id"] == job.job_id
    assert data["status"] == "completed"
    assert data["raw_file"] == str(job.raw_file)
    assert data["log_file"] == str(job.log_file)
    # Duration is the caller's own measurement (elapsed on the run path,
    # timestamp-derived on the check_job path), not recomputed here.
    assert isinstance(data["duration"], float) and data["duration"] >= 0.0
    assert data["summary_available"] is False
    assert {
        "sim_type",
        "step_count",
        "signals",
        "signals_truncated",
        "range",
        "point_count",
        "measurements",
        "fourier",
    }.isdisjoint(data)
    observation = next(o for o in data["observations"] if o["code"] == "parse_deadline")
    assert str(job.raw_file) in observation["detail"]
    assert "simulation_summary" in data["hint"]


@pytest.mark.asyncio
async def test_run_simulation_bounds_inline_completion_summary(
    state_with_sim: SessionState,
    sample_netlist: Path,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_file = work_dir / "inline.raw"
    log_file = work_dir / "inline.log"
    raw_file.write_bytes(b"raw")
    log_file.write_text("completed\n")
    slow = SlowSummaryParser(asyncio.get_running_loop())

    async def complete_run(netlist_path, job, state):
        job.raw_file = raw_file
        job.log_file = log_file
        job.status = "completed"
        job.completed_at = job.started_at + timedelta(seconds=2)
        job.done_event.set()

    runner = MagicMock()
    runner.start_simulation = AsyncMock(side_effect=complete_run)
    monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(simulation, "parse_success_summary", slow)

    try:
        with patch("ltspice_mcp.tools.simulation._get_or_create_runner", return_value=runner):
            result = await asyncio.wait_for(
                handle_run_simulation(
                    RunSimulationInput(netlist=sample_netlist.name, timeout=5),
                    state_with_sim,
                ),
                timeout=0.5,
            )
        job = next(iter(state_with_sim.jobs.values()))
        data = result.structuredContent
        assert data is not None
        _assert_fact_only_fallback(data, job)
        assert slow.calls == 1
    finally:
        slow.release.set()
        services._wedged_raw_paths.pop(raw_file, None)


@pytest.mark.asyncio
async def test_check_job_bounds_summary_and_cooldown_fast_fails(
    state_with_sim: SessionState,
    sample_netlist: Path,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_file = work_dir / "polled.raw"
    log_file = work_dir / "polled.log"
    raw_file.write_bytes(b"raw")
    log_file.write_text("completed\n")
    job = _completed_job(state_with_sim, sample_netlist, raw_file, log_file)
    slow = SlowSummaryParser(asyncio.get_running_loop())
    load_raw_sync = MagicMock()

    monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.05)
    monkeypatch.setattr(simulation, "parse_success_summary", slow)
    monkeypatch.setattr(services, "load_raw_sync", load_raw_sync)

    try:
        first = await asyncio.wait_for(
            handle_check_job(CheckJobInput(job_id=job.job_id), state_with_sim),
            timeout=0.5,
        )
        first_data = first.structuredContent
        assert first_data is not None
        _assert_fact_only_fallback(first_data, job)
        assert slow.calls == 1

        second_started = time.monotonic()
        second = await handle_check_job(CheckJobInput(job_id=job.job_id), state_with_sim)
        assert time.monotonic() - second_started < 0.15
        second_data = second.structuredContent
        assert second_data is not None
        _assert_fact_only_fallback(second_data, job)
        assert slow.calls == 1

        with pytest.raises(ResultError, match="paused"):
            await services.load_raw(raw_file, state_with_sim)
        load_raw_sync.assert_not_called()
    finally:
        slow.release.set()
        services._wedged_raw_paths.pop(raw_file, None)


@pytest.mark.asyncio
async def test_light_request_stays_responsive_during_abandoned_summary_parse(
    state_with_sim: SessionState,
    sample_netlist: Path,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    raw_file = work_dir / "responsive.raw"
    log_file = work_dir / "responsive.log"
    raw_file.write_bytes(b"raw")
    log_file.write_text("completed\n")
    job = _completed_job(
        state_with_sim,
        sample_netlist,
        raw_file,
        log_file,
        job_id="responsive_job",
    )
    slow = SlowSummaryParser(asyncio.get_running_loop())
    monkeypatch.setattr(services, "RAW_PARSE_TIMEOUT_S", 0.2)
    monkeypatch.setattr(simulation, "parse_success_summary", slow)

    heavy = asyncio.create_task(handle_check_job(CheckJobInput(job_id=job.job_id), state_with_sim))
    try:
        await asyncio.wait_for(slow.entered.wait(), timeout=0.5)

        light_started = time.monotonic()
        light = await handle_server_status(ServerStatusInput(), state_with_sim)
        light_elapsed = time.monotonic() - light_started

        assert light.content
        assert light_elapsed < 0.15
        assert not heavy.done()

        result = await asyncio.wait_for(heavy, timeout=0.5)
        data = result.structuredContent
        assert data is not None
        _assert_fact_only_fallback(data, job)
    finally:
        slow.release.set()
        if not heavy.done():
            heavy.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heavy
        services._wedged_raw_paths.pop(raw_file, None)


def test_completion_tools_declare_summary_availability():
    _, dispatch = get_tools_for_profile("full")
    for tool_name in ("run_simulation", "check_job"):
        schema = dispatch[tool_name].definition.outputSchema
        assert schema is not None
        assert schema["properties"]["summary_available"]["type"] == "boolean"


def test_normal_completion_marks_summary_available():
    started_at = now()
    job = SimulationJob(
        job_id="normal_summary",
        netlist=Path("/tmp/normal.cir"),
        simulator="FakeSim",
        status="completed",
        started_at=started_at,
        completed_at=started_at,
    )
    result = _format_success_response(
        job,
        {
            "sim_type": "Transient",
            "duration": 0.1,
            "step_count": 1,
            "raw_file": "/tmp/normal.raw",
            "log_file": "/tmp/normal.log",
            "signals": ["time"],
            "warnings": [],
        },
    )
    assert result.structuredContent is not None
    assert result.structuredContent["summary_available"] is True
