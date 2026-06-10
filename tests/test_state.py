"""Tests for SessionState lifecycle, the union job store, and dataclass defaults."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.state import BatchJob, MonteCarloConfig, SessionState, SimulationJob
from ltspice_mcp.tools import get_tools_for_profile


class _RecordingRunner:
    """Stub for a cached sim/sweep/MC runner.

    Records each ``cancel()`` call and honours the runner cancel contract:
    a still-live job is transitioned to terminal ``cancelled``.
    """

    def __init__(self) -> None:
        self.cancel_calls: list[tuple[SimulationJob | BatchJob, SessionState | None]] = []

    async def cancel(
        self, job: SimulationJob | BatchJob, state: SessionState | None = None
    ) -> None:
        self.cancel_calls.append((job, state))
        if job.status in ("queued", "running"):
            transition(job, "cancelled", state=state)


class TestSessionStateCreate:
    def test_create_no_simulators(self, config: ServerConfig):
        state = SessionState.create(config, {})
        assert state.default_simulator is None
        assert state.available_simulators == {}

    def test_create_initializes_empty_caches(self, config: ServerConfig):
        state = SessionState.create(config, {})
        assert isinstance(state.editors, FileCache)
        assert isinstance(state.results, FileCache)
        assert len(state.editors) == 0
        assert len(state.results) == 0

    def test_create_initializes_runner_manager(self, config: ServerConfig):
        state = SessionState.create(config, {})
        assert isinstance(state.runners, RunnerManager)

    def test_create_populates_tool_defs_full(self, config: ServerConfig):
        state = SessionState.create(config, {})
        assert len(state.tool_defs) > 0
        assert len(state.tool_dispatch) > 0
        # Full profile — all tools
        assert len(state.tool_defs) == len(state.tool_dispatch)

    def test_create_populates_tool_defs_agentic(self, work_dir: Path):
        config = ServerConfig(
            working_dir=work_dir,
            allowed_paths=[work_dir],
            tool_profile="agentic",
        )
        state = SessionState.create(config, {})
        agentic_defs, _ = get_tools_for_profile("agentic")
        agentic_names = {tool_def.name for tool_def in agentic_defs}
        assert len(state.tool_defs) == len(agentic_names)
        assert set(state.tool_dispatch.keys()) == agentic_names


class TestSessionStateShutdown:
    async def test_shutdown_clears_caches(self, config: ServerConfig, tmp_path: Path):
        state = SessionState.create(config, {})
        p = tmp_path / "dummy.txt"
        p.write_text("data")
        state.editors.get(p, lambda path: path.read_text())
        assert len(state.editors) == 1

        await state.shutdown()
        assert len(state.editors) == 0
        assert len(state.results) == 0

    async def test_shutdown_cancels_running_sim_jobs(self, config: ServerConfig):
        state = SessionState.create(config, {})
        job = SimulationJob(
            job_id="sim1",
            netlist=Path("/tmp/test.cir"),
            simulator="ltspice",
            status="running",
            started_at=now(),
        )
        state.jobs["sim1"] = job

        await state.shutdown()
        assert job.status == "cancelled"
        assert job.done_event.is_set()

    async def test_shutdown_cancels_running_batch_jobs(self, config: ServerConfig):
        state = SessionState.create(config, {})
        batch = BatchJob(
            job_id="batch1",
            job_type="sweep",
            netlist=Path("/tmp/test.cir"),
            total_runs=10,
            status="running",
        )
        state.batch_jobs["batch1"] = batch

        await state.shutdown()
        assert batch.status == "cancelled"
        assert batch.done_event.is_set()

    async def test_shutdown_routes_sim_cancel_through_cached_runner(
        self, config: ServerConfig, tmp_path: Path
    ):
        """With a cached SimulationRunner, shutdown must delegate to its
        cancel() (the path that kills live simulator processes) rather than
        only flipping the job status."""
        state = SessionState.create(config, {})
        sim_runner = _RecordingRunner()
        state.runners._runners["sim"] = sim_runner

        job = SimulationJob(
            job_id="sim-live",
            netlist=tmp_path / "test.cir",
            simulator="ltspice",
            status="running",
            started_at=now(),
        )
        state.add_job(job)

        await state.shutdown()

        assert len(sim_runner.cancel_calls) == 1
        called_job, called_state = sim_runner.cancel_calls[0]
        assert called_job is job
        assert called_state is state
        assert job.status == "cancelled"
        assert job.done_event.is_set()

    async def test_shutdown_routes_sweep_batch_cancel_through_sweep_runner(
        self, config: ServerConfig, tmp_path: Path
    ):
        state = SessionState.create(config, {})
        sweep_runner = _RecordingRunner()
        mc_runner = _RecordingRunner()
        state.runners._runners["sweep"] = sweep_runner
        state.runners._runners["mc"] = mc_runner

        batch = BatchJob(
            job_id="sweep-live",
            job_type="sweep",
            netlist=tmp_path / "test.cir",
            total_runs=3,
            status="running",
        )
        state.add_batch_job(batch)

        await state.shutdown()

        assert len(sweep_runner.cancel_calls) == 1
        called_job, called_state = sweep_runner.cancel_calls[0]
        assert called_job is batch
        assert called_state is state
        assert mc_runner.cancel_calls == []
        assert batch.status == "cancelled"
        assert batch.done_event.is_set()

    async def test_shutdown_routes_montecarlo_batch_cancel_through_mc_runner(
        self, config: ServerConfig, tmp_path: Path
    ):
        state = SessionState.create(config, {})
        sweep_runner = _RecordingRunner()
        mc_runner = _RecordingRunner()
        state.runners._runners["sweep"] = sweep_runner
        state.runners._runners["mc"] = mc_runner

        batch = BatchJob(
            job_id="mc-live",
            job_type="montecarlo",
            netlist=tmp_path / "test.cir",
            total_runs=5,
            status="running",
        )
        state.add_batch_job(batch)

        await state.shutdown()

        assert len(mc_runner.cancel_calls) == 1
        called_job, called_state = mc_runner.cancel_calls[0]
        assert called_job is batch
        assert called_state is state
        assert sweep_runner.cancel_calls == []
        assert batch.status == "cancelled"
        assert batch.done_event.is_set()

    async def test_shutdown_ignores_completed_jobs(self, config: ServerConfig):
        state = SessionState.create(config, {})
        job = SimulationJob(
            job_id="done1",
            netlist=Path("/tmp/test.cir"),
            simulator="ltspice",
            status="completed",
            started_at=now(),
            completed_at=now(),
        )
        state.jobs["done1"] = job

        await state.shutdown()
        assert job.status == "completed"


def _sim_job(job_id: str, *, status: str = "completed", started_at=None) -> SimulationJob:
    start = started_at or now()
    return SimulationJob(
        job_id=job_id,
        netlist=Path("/tmp/test.cir"),
        simulator="ltspice",
        status=status,  # type: ignore[arg-type]
        started_at=start,
        completed_at=start + timedelta(seconds=1) if status == "completed" else None,
    )


def _batch_job(job_id: str, *, status: str = "completed") -> BatchJob:
    return BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=Path("/tmp/test.cir"),
        total_runs=2,
        status=status,  # type: ignore[arg-type]
    )


class TestUnionJobStoreViews:
    """``state.jobs`` / ``state.batch_jobs`` are type-filtered writable views
    over the single union store (``state.all_jobs``): lookups surface only the
    view's job type, writes go straight through to the union dict."""

    def test_batch_job_invisible_through_sim_view(self, config: ServerConfig):
        state = SessionState.create(config, {})
        batch = _batch_job("b1")
        state.batch_jobs["b1"] = batch

        assert state.jobs.get("b1") is None
        assert "b1" not in state.jobs
        assert len(state.jobs) == 0
        assert list(state.jobs.values()) == []
        # ...but it exists in the union store and its own view.
        assert state.all_jobs["b1"] is batch
        assert state.batch_jobs["b1"] is batch

    def test_sim_job_invisible_through_batch_view(self, config: ServerConfig):
        state = SessionState.create(config, {})
        sim = _sim_job("j1")
        state.jobs["j1"] = sim

        assert state.batch_jobs.get("j1") is None
        assert "j1" not in state.batch_jobs
        assert len(state.batch_jobs) == 0
        assert list(state.batch_jobs.values()) == []
        assert state.all_jobs["j1"] is sim
        assert state.jobs["j1"] is sim

    def test_views_write_through_to_union_store(self, config: ServerConfig):
        state = SessionState.create(config, {})
        sim = _sim_job("j1")
        batch = _batch_job("b1")
        state.jobs["j1"] = sim
        state.batch_jobs["b1"] = batch

        assert state.all_jobs == {"j1": sim, "b1": batch}
        assert len(state.jobs) == 1
        assert len(state.batch_jobs) == 1
        assert set(state.jobs) == {"j1"}
        assert set(state.batch_jobs) == {"b1"}

    def test_view_write_rejects_wrong_job_type(self, config: ServerConfig):
        """Writing a job of the wrong type through a typed view must fail
        loudly: silently accepting it would store a job that is invisible
        through the view that wrote it."""
        state = SessionState.create(config, {})

        with pytest.raises(
            TypeError, match=r"SimulationJob view cannot store BatchJob \(key 'b1'\)"
        ):
            state.jobs["b1"] = _batch_job("b1")  # type: ignore[assignment]
        with pytest.raises(
            TypeError, match=r"BatchJob view cannot store SimulationJob \(key 'j1'\)"
        ):
            state.batch_jobs["j1"] = _sim_job("j1")  # type: ignore[assignment]
        # Nothing leaked into the union store.
        assert state.all_jobs == {}


class TestCancelRunningSnapshotsViews:
    """``cancel_running`` awaits runner ``cancel()`` mid-loop; jobs registered
    during that suspension must not invalidate the iteration over the live
    union store (lazy view iteration would raise ``RuntimeError: dictionary
    changed size during iteration``)."""

    async def test_sim_job_registered_during_cancel_does_not_break_iteration(
        self, config: ServerConfig
    ):
        state = SessionState.create(config, {})
        registry = state.job_registry

        class _RegisteringRunner(_RecordingRunner):
            async def cancel(
                self, job: SimulationJob | BatchJob, state: SessionState | None = None
            ) -> None:
                registry.add_sim_job(_sim_job(f"late-{job.job_id}"))
                await super().cancel(job, state)

        state.runners._runners["sim"] = _RegisteringRunner()
        for i in range(3):
            registry.jobs[f"run{i}"] = _sim_job(f"run{i}", status="running")

        await registry.cancel_running(state.runners, state)

        for i in range(3):
            assert registry.jobs[f"run{i}"].status == "cancelled"
            # Jobs registered mid-cancel land in the store untouched.
            assert registry.jobs[f"late-run{i}"].status == "completed"

    async def test_batch_job_registered_during_cancel_does_not_break_iteration(
        self, config: ServerConfig
    ):
        state = SessionState.create(config, {})
        registry = state.job_registry

        class _RegisteringRunner(_RecordingRunner):
            async def cancel(
                self, job: SimulationJob | BatchJob, state: SessionState | None = None
            ) -> None:
                registry.add_batch_job(_batch_job(f"late-{job.job_id}"))
                await super().cancel(job, state)

        state.runners._runners["sweep"] = _RegisteringRunner()
        for i in range(3):
            registry.jobs[f"bat{i}"] = _batch_job(f"bat{i}", status="running")

        await registry.cancel_running(state.runners, state)

        for i in range(3):
            assert registry.jobs[f"bat{i}"].status == "cancelled"
            assert registry.jobs[f"late-bat{i}"].status == "completed"


class TestPerTypeEvictionCap:
    def test_each_job_type_capped_at_200_finished(self):
        """The registry keeps at most 200 finished jobs PER TYPE in the union
        store: 205 finished sims plus 205 finished batches leave 200 of each
        (the oldest five of each type evicted), 400 jobs total."""
        registry = JobRegistry(persist_enabled=False)
        base = now()
        for i in range(205):
            registry.add_sim_job(_sim_job(f"sim{i:03d}", started_at=base + timedelta(seconds=i)))
        for i in range(205):
            batch = _batch_job(f"bat{i:03d}")
            batch.started_at = base + timedelta(seconds=i)
            registry.add_batch_job(batch)

        assert len(registry.sim_jobs) == 200
        assert len(registry.batch_jobs) == 200
        assert len(registry.jobs) == 400
        for i in range(5):
            assert f"sim{i:03d}" not in registry.jobs
            assert f"bat{i:03d}" not in registry.jobs
        assert "sim005" in registry.sim_jobs
        assert "sim204" in registry.sim_jobs
        assert "bat005" in registry.batch_jobs
        assert "bat204" in registry.batch_jobs


class TestDataclassDefaults:
    def test_montecarlo_config_defaults(self):
        mc = MonteCarloConfig(netlist=Path("/tmp/test.cir"))
        assert mc.num_runs == 100

    def test_batchjob_defaults(self):
        bj = BatchJob(
            job_id="j1",
            job_type="sweep",
            netlist=Path("/tmp/test.cir"),
            total_runs=5,
        )
        assert bj.status == "running"
        assert bj.completed_runs == 0
        assert bj.failed_runs == 0

    def test_simulation_job_done_event(self):
        job = SimulationJob(
            job_id="s1",
            netlist=Path("/tmp/test.cir"),
            simulator="ltspice",
            status="queued",
            started_at=now(),
        )
        assert isinstance(job.done_event, asyncio.Event)
        assert not job.done_event.is_set()
