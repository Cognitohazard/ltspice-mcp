"""Tests for simulation tool handlers using direct job state injection."""

import asyncio
import typing
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import JobNotFoundError, ResultError, SimulationError
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

    async def test_failed_response_includes_result_file_paths(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A failed run must surface its raw/log paths so the caller can open the
        # full artifacts instead of working from the truncated log excerpt alone.
        raw = work_dir / "fail.raw"
        log = work_dir / "fail.log"
        raw.write_text("partial")
        log.write_text("Error: convergence failed\n")
        job = _make_job(state_no_sim, status="failed", raw_file=raw, log_file=log)
        job.error = "Sim failed"
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert data["log_file"] == str(log)
        assert data["raw_file"] == str(raw)
        # The human-readable footer points the caller at the full artifacts.
        assert "Result files:" in result.content[0].text

    async def test_failed_missing_model_hint_in_text_and_structured(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A missing-model failure surfaces the find_model recovery hint in BOTH
        # the text and the structured 'error' field, and the log excerpt appears
        # once (job.error already carries it — the branch must not duplicate it).
        log = work_dir / "mm.log"
        log.write_text('Error on line 2 : q1 c b e 2n2222 Undefined model "2n2222"\n')
        job = _make_job(state_no_sim, status="failed", log_file=log)
        job.error = (
            'Simulation failed (no output generated)\n\nLog excerpt:\nUndefined model "2n2222"'
        )
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        text = result.content[0].text
        data = result.structuredContent
        assert data is not None
        for blob in (text, data["error"]):
            assert "find_model" in blob
            assert "include_builtin=true" in blob
            assert "2n2222" in blob
        assert text.count("Log excerpt:") == 1

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
        with pytest.raises(JobNotFoundError):
            await handle_check_job(CheckJobInput(job_id="missing"), state_no_sim)

    async def test_list_empty_no_filter(self, state_no_sim: SessionState):
        result = await handle_check_job(CheckJobInput(), state_no_sim)
        assert "No active jobs" in result.content[0].text

    async def test_interrupted_job_list_duration_is_unknown_not_wallclock(
        self, state_no_sim: SessionState
    ):
        # A recovered/interrupted job is terminal with completed_at=None. Its
        # true runtime is unknowable after a restart, so the list row must NOT
        # report a wall-clock-to-now number labelled "(running)" — it shows
        # "unknown" and omits the numeric duration from the structured row.
        long_ago = now() - timedelta(hours=5)
        job = SimulationJob(
            job_id="interrupted1",
            netlist=Path("/tmp/test.cir"),
            simulator="FakeSim",
            status="interrupted",  # type: ignore[arg-type]
            started_at=long_ago,
            completed_at=None,
        )
        state_no_sim.jobs["interrupted1"] = job
        result = await handle_check_job(CheckJobInput(status="interrupted"), state_no_sim)
        text = result.content[0].text
        assert "interrupted1" in text
        assert "(running)" not in text
        assert "unknown" in text
        rows = result.structuredContent["jobs"]
        (row,) = [r for r in rows if r["job_id"] == "interrupted1"]
        assert row.get("duration") is None

    async def test_check_job_empty_default_mentions_status_all(self, state_no_sim: SessionState):
        # Default (no-arg) view hides terminal jobs. When the only jobs are
        # terminal, the empty message must tell the caller they exist and how to
        # widen the view, rather than reading as "nothing exists".
        _make_job(state_no_sim, job_id="done1", status="completed")
        result = await handle_check_job(CheckJobInput(), state_no_sim)
        text = result.content[0].text
        assert 'status="all"' in text
        assert "are hidden" in text or "hidden" in text

    async def test_check_job_empty_no_jobs_stays_minimal(self, state_no_sim: SessionState):
        # With zero jobs of any kind, the default message is the plain
        # "No active jobs" with no claim that finished jobs are hidden.
        result = await handle_check_job(CheckJobInput(), state_no_sim)
        text = result.content[0].text
        assert "No active jobs" in text
        assert "hidden" not in text

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
        with pytest.raises(JobNotFoundError):
            await handle_cancel_job(CancelJobInput(job_id="missing"), state_no_sim)

    async def test_already_completed(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="completed")
        with pytest.raises(SimulationError, match="not running"):
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_no_sim)

    async def test_cancel_terminal_job_show_hint_false(self, state_no_sim: SessionState):
        # Cancelling a job that already finished is a job-state error, not a
        # simulator-availability one: the generic "verify simulator" hint must
        # be suppressed and the message must point the caller at check_job.
        _make_job(state_no_sim, status="completed")
        with pytest.raises(SimulationError) as exc_info:
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_no_sim)
        exc = exc_info.value
        assert exc.show_hint is False
        assert "not running" in str(exc)
        assert "check_job" in str(exc)

    async def test_no_simulator(self, state_no_sim: SessionState):
        _make_job(state_no_sim, status="running")
        with pytest.raises(SimulationError, match="No SPICE simulator"):
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
        # Unblock the deadline watchdog the async path arms, so no task is
        # left pending at loop teardown.
        next(iter(state_with_sim.jobs.values())).done_event.set()
        await asyncio.sleep(0)

    async def test_async_timeout_watchdog_kills_overdue_job(
        self, state_with_sim: SessionState, sample_netlist: Path, monkeypatch
    ):
        # An async job (timeout above the sync threshold) must have its
        # deadline enforced by the background watchdog. Without it the
        # timeout was accepted and silently never enforced — observed live
        # as a 35s-timeout job still running at 119s elapsed.
        monkeypatch.setattr("ltspice_mcp.tools.simulation.SYNC_TIMEOUT_THRESHOLD", 0.0)

        async def hang_until_killed(netlist_path, job, state):
            job.status = "running"
            await job.done_event.wait()

        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock(side_effect=hang_until_killed)
        fake_runner.kill = AsyncMock()
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            result = await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=0.1),
                state_with_sim,
            )
            # Returned immediately (async path), reporting the live status.
            assert "Job ID:" in result.content[0].text
            job = next(iter(state_with_sim.jobs.values()))
            assert result.structuredContent is not None
            assert result.structuredContent["status"] == "running"
            assert job.status == "running"
            # Await the watchdog itself instead of sleeping past the
            # deadline — it returns the moment it has timed the job out.
            from ltspice_mcp.tools import simulation as simulation_module

            (watchdog,) = simulation_module._timeout_watchdogs
            await asyncio.wait_for(watchdog, timeout=2)
        assert job.status == "timeout"
        fake_runner.kill.assert_awaited_once_with(job.job_id)

    async def test_cancel_during_submit_log_leaves_no_orphaned_job(
        self, state_with_sim: SessionState, sample_netlist: Path, monkeypatch
    ):
        """A request cancelled at the post-submit MCP log notification must
        not leave a registered job with no task to advance it — no suspension
        point may sit between job registration and task creation (the
        submit-ordering rule in the tools/_base.py concurrency contract)."""
        entered = asyncio.Event()

        async def hanging_log(level, msg):
            entered.set()
            await asyncio.Event().wait()  # suspend until cancelled

        monkeypatch.setattr("ltspice_mcp.tools.simulation.mcp_log", hanging_log)

        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock()
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            request = asyncio.create_task(
                handle_run_simulation(
                    RunSimulationInput(netlist=sample_netlist.name, timeout=60),
                    state_with_sim,
                )
            )
            await asyncio.wait_for(entered.wait(), timeout=5)
            request.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request

            orphaned = [j.job_id for j in state_with_sim.jobs.values() if j.task is None]
            assert orphaned == [], (
                "cancellation between add_job and create_task orphaned a registered job"
            )
            # Drain the submission task(s) so nothing is pending at teardown.
            for job in state_with_sim.jobs.values():
                assert job.task is not None
                await job.task

    async def test_async_watchdog_leaves_completed_job_alone(
        self, state_with_sim: SessionState, sample_netlist: Path, monkeypatch
    ):
        # A job that finishes inside its deadline must not be touched when
        # the watchdog's timer would have fired.
        monkeypatch.setattr("ltspice_mcp.tools.simulation.SYNC_TIMEOUT_THRESHOLD", 0.0)

        async def complete_fast(netlist_path, job, state):
            job.status = "completed"
            job.completed_at = now()
            job.done_event.set()

        fake_runner = MagicMock()
        fake_runner.start_simulation = AsyncMock(side_effect=complete_fast)
        fake_runner.kill = AsyncMock()
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner",
            return_value=fake_runner,
        ):
            await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name, timeout=0.1),
                state_with_sim,
            )
            job = next(iter(state_with_sim.jobs.values()))
            # The watchdog is event-driven: once done_event is set inside
            # the deadline, it exits and its done-callback deregisters it.
            # How many event-loop turns that callback chain needs varies by
            # Python version (asyncio.wait_for was reworked in 3.12), so wait
            # for the deregistration rather than assuming a fixed turn count.
            from ltspice_mcp.tools import simulation as simulation_module

            for _ in range(1000):
                if not simulation_module._timeout_watchdogs:
                    break
                await asyncio.sleep(0)
            assert not simulation_module._timeout_watchdogs
        assert job.status == "completed"
        fake_runner.kill.assert_not_awaited()

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
    """check_job must resolve/list batch (sweep/MC) jobs, not just sims."""

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


@pytest.mark.asyncio
class TestResolveOutputFolder:
    """Simulation artifacts must not flood the project root — but a netlist with
    a relative local include must stay put so the simulator can resolve it."""

    @staticmethod
    def _force_non_wsl(monkeypatch):
        from ltspice_mcp.lib import wsl

        monkeypatch.setattr(wsl, "is_wsl", lambda: False)

    @staticmethod
    def _force_wsl_linux_fs(monkeypatch):
        # WSL with a working dir on the Linux filesystem (not /mnt/) — the branch
        # that relocates artifacts off the UNC path.
        from ltspice_mcp.lib import wsl

        monkeypatch.setattr(wsl, "is_wsl", lambda: True)
        monkeypatch.setattr(wsl, "is_windows_native_path", lambda p: False)

    async def test_self_contained_routes_to_sidecar(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # No includes → safe to relocate → tidy .ltspice-mcp/runs sidecar.
        from ltspice_mcp.tools._base import resolve_output_folder

        self._force_non_wsl(monkeypatch)
        nl = work_dir / "sc.cir"
        nl.write_text("* sc\nR1 in 0 1k\nV1 in 0 1\n.op\n.end\n")
        out = await resolve_output_folder(state_no_sim, nl)
        assert out == work_dir / ".ltspice-mcp" / "runs"
        assert out.is_dir()
        assert out in state_no_sim.config.allowed_paths

    async def test_relative_local_include_stays_in_working_dir(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # A relative .include pointing at a sibling file can't be relocated —
        # the simulator would resolve it against the moved netlist's dir.
        from ltspice_mcp.tools._base import resolve_output_folder

        self._force_non_wsl(monkeypatch)
        (work_dir / "models").mkdir()
        (work_dir / "models" / "r.lib").write_text(".subckt RMOD a b\nR1 a b 1k\n.ends\n")
        nl = work_dir / "wl.cir"
        nl.write_text("* wl\nX1 in 0 RMOD\nV1 in 0 1\n.include models/r.lib\n.op\n.end\n")
        out = await resolve_output_folder(state_no_sim, nl)
        assert out == work_dir

    async def test_wsl_linux_fs_local_include_stays_in_working_dir(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # On WSL Linux-fs the deck would normally relocate off the UNC path, but a
        # relative include must veto that too — else the moved deck can't find it.
        from ltspice_mcp.tools._base import resolve_output_folder

        self._force_wsl_linux_fs(monkeypatch)
        (work_dir / "models").mkdir()
        (work_dir / "models" / "r.lib").write_text(".subckt RMOD a b\nR1 a b 1k\n.ends\n")
        nl = work_dir / "wl.cir"
        nl.write_text("* wl\nX1 in 0 RMOD\nV1 in 0 1\n.include models/r.lib\n.op\n.end\n")
        out = await resolve_output_folder(state_no_sim, nl)
        assert out == work_dir

    async def test_wsl_linux_fs_self_contained_relocates(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # A self-contained deck still relocates off the UNC path on WSL Linux-fs.
        from ltspice_mcp.tools._base import resolve_output_folder

        self._force_wsl_linux_fs(monkeypatch)
        nl = work_dir / "sc.cir"
        nl.write_text("* sc\nR1 in 0 1k\nV1 in 0 1\n.op\n.end\n")
        out = await resolve_output_folder(state_no_sim, nl)
        assert out != work_dir
        assert out != work_dir / ".ltspice-mcp" / "runs"

    async def test_libpath_name_still_sidecars(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # A bare .lib NAME (resolved via the simulator's lib path, no local file)
        # survives relocation, so it still gets the sidecar.
        from ltspice_mcp.tools._base import resolve_output_folder

        self._force_non_wsl(monkeypatch)
        nl = work_dir / "lp.cir"
        nl.write_text("* lp\nM1 d g s b NMOS\nV1 d 0 1\n.lib LTC.lib\n.op\n.end\n")
        out = await resolve_output_folder(state_no_sim, nl)
        assert out == work_dir / ".ltspice-mcp" / "runs"
