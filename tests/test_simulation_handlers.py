"""Tests for simulation tool handlers using direct job state injection."""

import typing
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import ResultError, SimulationError
from ltspice_mcp.lib import now
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob
from ltspice_mcp.tools.simulation import (
    CancelJobInput,
    CheckJobInput,
    RunSimulationInput,
    handle_cancel_job,
    handle_check_job,
    handle_run_simulation,
)


def _text_of(result) -> str:
    """Extract text from a TextContent result, asserting type."""
    item = result.content[0]
    assert isinstance(item, types.TextContent)
    return item.text


class FakeSim:
    spice_exe: typing.ClassVar[list[str]] = ["/fake/path/sim.exe"]


@pytest.fixture
def state_with_sim(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"fake": FakeSim})


def _make_job(
    state: SessionState,
    *,
    job_id: str = "j1",
    status: str = "running",
    raw_file: Path | None = None,
    log_file: Path | None = None,
) -> SimulationJob:
    started = now()
    job = SimulationJob(
        job_id=job_id,
        netlist=Path("/tmp/test.cir"),
        simulator="FakeSim",
        status=status,  # type: ignore[arg-type]
        started_at=started,
        completed_at=started + timedelta(seconds=2) if status != "running" else None,
        raw_file=raw_file,
        log_file=log_file,
    )
    state.jobs[job_id] = job
    return job


@pytest.mark.asyncio
class TestCheckJob:
    async def test_running(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "still running" in result.content[0].text
        assert result.structuredContent["status"] == "running"

    async def test_failed(self, state_no_sim: SessionState):
        job = _make_job(state_no_sim, status="failed")
        job.error = "convergence failed"
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        text = result.content[0].text
        assert "failed" in text
        assert "convergence" in text

    async def test_cancelled(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="cancelled")
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "cancelled" in result.content[0].text

    async def test_timeout(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="timeout")
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "timed out" in result.content[0].text

    async def test_completed_missing_files(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="completed")
        with pytest.raises(ResultError, match="result files are missing"):
            await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)

    async def test_completed_files_removed(self, state_no_sim: SessionState, work_dir: Path):
        raw = work_dir / "x.raw"
        log = work_dir / "x.log"
        raw.write_text("d")
        log.write_text("l")
        _make_job(state_no_sim, status="completed", raw_file=raw, log_file=log)
        # Remove the files now
        raw.unlink()
        log.unlink()
        with pytest.raises(ResultError, match="have been removed"):
            await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)

    async def test_unknown_id(self, state_no_sim: SessionState):
        with pytest.raises(SimulationError):
            await handle_check_job(CheckJobInput(job_id="missing"), state_no_sim)

    async def test_list_empty_no_filter(self, state_no_sim: SessionState):
        result = await handle_check_job(CheckJobInput(), state_no_sim)
        assert "No active jobs" in result.content[0].text

    async def test_list_filter_status(self, state_no_sim: SessionState):
        _make_job(state_no_sim, job_id="r1", status="running")
        _make_job(state_no_sim, job_id="c1", status="completed")
        result = await handle_check_job(CheckJobInput(status="completed"), state_no_sim)
        text = result.content[0].text
        assert "c1" in text
        assert "r1" not in text

    async def test_list_filter_all(self, state_no_sim: SessionState):
        _make_job(state_no_sim, job_id="r1", status="running")
        _make_job(state_no_sim, job_id="c1", status="completed")
        result = await handle_check_job(CheckJobInput(status="all"), state_no_sim)
        text = result.content[0].text
        assert "r1" in text
        assert "c1" in text

    async def test_list_filter_none_match(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(status="failed"), state_no_sim)
        assert "No jobs with status" in result.content[0].text


class TestFormatSuccessResponse:
    def test_basic(self):
        from ltspice_mcp.tools.simulation import _format_success_response

        summary = {
            "sim_type": "Transient",
            "duration": 1.5,
            "step_count": 1,
            "raw_file": "/tmp/x.raw",
            "log_file": "/tmp/x.log",
            "signals": ["time", "V(out)"],
            "warnings": [],
        }
        result = _format_success_response("j1", summary, None)
        text = _text_of(result)
        assert "Transient" in text
        assert "V(out)" in text
        assert result.structuredContent is not None
        assert result.structuredContent["status"] == "completed"

    def test_with_many_signals(self):
        from ltspice_mcp.tools.simulation import _format_success_response

        signals = [f"V(n{i})" for i in range(30)]
        summary = {
            "sim_type": "Transient",
            "duration": 1.5,
            "step_count": 1,
            "raw_file": "/tmp/x.raw",
            "log_file": "/tmp/x.log",
            "signals": signals,
            "warnings": ["w1"],
            "errors": ["e1"],
        }
        result = _format_success_response("j1", summary, None)
        text = _text_of(result)
        assert "and 10 more" in text
        assert "Errors:" in text
        assert "Warnings:" in text


@pytest.mark.asyncio
class TestCancelJob:
    async def test_unknown_job(self, state_no_sim: SessionState):
        with pytest.raises(SimulationError):
            await handle_cancel_job(CancelJobInput(job_id="missing"), state_no_sim)

    async def test_already_completed(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="completed")
        with pytest.raises(SimulationError, match="not running"):
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_no_sim)

    async def test_no_simulator(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="running")
        with pytest.raises(SimulationError, match="No simulator"):
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_no_sim)


@pytest.mark.asyncio
class TestCancelJobBatch:
    """A sweep/Monte-Carlo job must be cancellable through cancel_job.

    Regression: handle_cancel_job resolved only the single-sim store, so every
    sweep/MC id returned "not found" and the batch runner's cancel (with its WSL
    process-kill) was unreachable from the tool surface. These tests drive the
    real handler — not the runner's cancel() directly — so the tool->runner
    routing is exercised, which is the seam the runner-level tests don't cover.
    """

    async def test_cancel_running_sweep_routes_to_sweep_runner(self, state_with_sim: SessionState):
        bj = BatchJob(
            job_id="sweep_live",
            job_type="sweep",
            netlist=Path("/tmp/s.cir"),
            total_runs=4,
            completed_runs=1,
            status="running",
        )
        state_with_sim.add_batch_job(bj)
        fake_runner = MagicMock()
        fake_runner.cancel = AsyncMock()
        with patch.object(
            state_with_sim.runners, "get_existing_sweep_runner", return_value=fake_runner
        ):
            result = await handle_cancel_job(CancelJobInput(job_id="sweep_live"), state_with_sim)
        assert "cancelled" in result.content[0].text.lower()
        fake_runner.cancel.assert_awaited_once()
        # The batch job itself (resolved from batch_jobs) was handed to the runner.
        assert fake_runner.cancel.await_args.args[0] is bj

    async def test_cancel_running_montecarlo_routes_to_mc_runner(
        self, state_with_sim: SessionState
    ):
        bj = BatchJob(
            job_id="mc_live",
            job_type="montecarlo",
            netlist=Path("/tmp/m.cir"),
            total_runs=10,
            completed_runs=2,
            status="running",
        )
        state_with_sim.add_batch_job(bj)
        fake_runner = MagicMock()
        fake_runner.cancel = AsyncMock()
        with patch.object(
            state_with_sim.runners, "get_existing_mc_runner", return_value=fake_runner
        ):
            result = await handle_cancel_job(CancelJobInput(job_id="mc_live"), state_with_sim)
        assert "cancelled" in result.content[0].text.lower()
        fake_runner.cancel.assert_awaited_once()

    async def test_cancel_batch_runner_gone_raises(self, state_with_sim: SessionState):
        # Job marked running but its runner is no longer live (e.g. after a
        # restart): surface a clear error instead of crashing on a None runner.
        bj = BatchJob(
            job_id="sweep_orphan",
            job_type="sweep",
            netlist=Path("/tmp/o.cir"),
            total_runs=4,
            completed_runs=0,
            status="running",
        )
        state_with_sim.add_batch_job(bj)
        with (
            patch.object(state_with_sim.runners, "get_existing_sweep_runner", return_value=None),
            pytest.raises(SimulationError, match="no longer live"),
        ):
            await handle_cancel_job(CancelJobInput(job_id="sweep_orphan"), state_with_sim)


@pytest.mark.asyncio
class TestRunSimulationStubbed:
    """Test handle_run_simulation by stubbing the runner."""

    async def test_async_returns_job_id(self, state_with_sim: SessionState, sample_netlist: Path):
        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock()
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            result = await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=60),
                state_with_sim,
            )
        text = result.content[0].text
        assert "Job ID:" in text
        assert len(state_with_sim.jobs) == 1

    async def test_sync_timeout(self, state_with_sim: SessionState, sample_netlist: Path):
        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock()
        fake_runner.kill = AsyncMock()
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            result = await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=0.05, wait=False),
                state_with_sim,
            )
        text = result.content[0].text
        assert "timed out" in text.lower()

    async def test_sync_failed(
        self, state_with_sim: SessionState, sample_netlist: Path, work_dir: Path
    ):
        log = work_dir / "out.log"
        log.write_text("Error: convergence failed\n")

        async def start_sim(netlist_path, job, state):
            job.log_file = log
            job.status = "failed"
            job.error = "Sim failed"
            job.completed_at = now()
            job.done_event.set()

        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock(side_effect=start_sim)
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            result = await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=5, wait=False),
                state_with_sim,
            )
        text = result.content[0].text
        assert "failed" in text.lower()

    async def test_sync_cancelled(self, state_with_sim: SessionState, sample_netlist: Path):
        async def start_sim(netlist_path, job, state):
            job.status = "cancelled"
            job.completed_at = now()
            job.done_event.set()

        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock(side_effect=start_sim)
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            result = await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=5, wait=False),
                state_with_sim,
            )
        assert "cancelled" in result.content[0].text.lower()


@pytest.mark.asyncio
class TestCheckJobBatchVisibility:
    """V7-P1-3: check_job must resolve/list batch (sweep/MC) jobs, not just sims."""

    async def test_check_job_resolves_batch_job(self, state_with_sim: SessionState):
        bj = BatchJob(
            job_id="mc_x",
            job_type="montecarlo",
            netlist=Path("/tmp/x.cir"),
            total_runs=6,
            completed_runs=6,
            status="completed",
        )
        state_with_sim.add_batch_job(bj)
        result = await handle_check_job(CheckJobInput(job_id="mc_x"), state_with_sim)
        text = _text_of(result)
        assert "mc_x" in text
        assert "montecarlo" in text
        assert "not found" not in text.lower()
        assert result.structuredContent is not None
        assert result.structuredContent["job_type"] == "montecarlo"

    async def test_list_jobs_includes_batch(self, state_with_sim: SessionState):
        bj = BatchJob(
            job_id="sweep_y",
            job_type="sweep",
            netlist=Path("/tmp/y.cir"),
            total_runs=4,
            completed_runs=4,
            status="completed",
        )
        state_with_sim.add_batch_job(bj)
        result = await handle_check_job(CheckJobInput(status="all"), state_with_sim)
        ids = [j["job_id"] for j in result.structuredContent["jobs"]]
        assert "sweep_y" in ids
