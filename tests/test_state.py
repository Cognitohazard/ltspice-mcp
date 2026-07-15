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
from tests.conftest import make_batch_job, make_sim_job


class _RecordingRunner:
    """Stub for a cached sim/sweep/MC runner.

    Records each ``cancel()`` call and honours the runner cancel contract:
    a still-live job is transitioned to terminal ``cancelled``.
    """

    def __init__(self) -> None:
        self.cancel_calls: list[tuple[SimulationJob | BatchJob, SessionState | None]] = []

    def owns_batch_job(self, job_id: str) -> bool:
        # No recorded ownership: batch-cancel routing falls back to
        # most-recent-of-kind, which is what these tests pin.
        return False

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
        # Full profile — every advertised tool dispatches; the dispatch map may
        # also carry deprecated aliases absent from tool_defs (RegisteredTool.aliases).
        def_names = {t.name for t in state.tool_defs}
        assert def_names <= set(state.tool_dispatch)
        alias_only = set(state.tool_dispatch) - def_names
        assert all(name in state.tool_dispatch[name].aliases for name in alias_only)

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
        # tool_dispatch = advertised names plus any deprecated aliases.
        assert agentic_names <= set(state.tool_dispatch)
        alias_only = set(state.tool_dispatch) - agentic_names
        assert all(name in state.tool_dispatch[name].aliases for name in alias_only)


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
        state.runners._runners[("sim", _RecordingRunner, tmp_path)] = sim_runner

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

    @pytest.mark.parametrize(
        ("job_type", "active_runner_key", "idle_runner_key"),
        [
            pytest.param("sweep", "sweep", "mc", id="sweep-batch-routes-to-sweep-runner"),
            pytest.param("montecarlo", "mc", "sweep", id="mc-batch-routes-to-mc-runner"),
        ],
    )
    async def test_shutdown_routes_batch_cancel_through_matching_runner(
        self,
        config: ServerConfig,
        tmp_path: Path,
        job_type: str,
        active_runner_key: str,
        idle_runner_key: str,
    ):
        state = SessionState.create(config, {})
        active_runner = _RecordingRunner()
        idle_runner = _RecordingRunner()
        state.runners._runners[(active_runner_key, _RecordingRunner, tmp_path)] = active_runner
        state.runners._runners[(idle_runner_key, _RecordingRunner, tmp_path)] = idle_runner

        batch = make_batch_job(
            f"{job_type}-live",
            status="running",
            job_type=job_type,
            netlist=tmp_path / "test.cir",
            total_runs=3,
        )
        state.add_batch_job(batch)

        await state.shutdown()

        assert len(active_runner.cancel_calls) == 1
        called_job, called_state = active_runner.cancel_calls[0]
        assert called_job is batch
        assert called_state is state
        assert idle_runner.cancel_calls == []
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


class TestUnionJobStoreViews:
    """``state.jobs`` / ``state.batch_jobs`` are type-filtered writable views
    over the single union store (``state.all_jobs``): lookups surface only the
    view's job type, writes go straight through to the union dict."""

    @pytest.mark.parametrize(
        ("job_id", "make_job", "own_view_name", "other_view_name"),
        [
            pytest.param(
                "b1", make_batch_job, "batch_jobs", "jobs", id="batch-job-invisible-in-sim-view"
            ),
            pytest.param(
                "j1", make_sim_job, "jobs", "batch_jobs", id="sim-job-invisible-in-batch-view"
            ),
        ],
    )
    def test_job_invisible_through_other_type_view(
        self, config: ServerConfig, job_id: str, make_job, own_view_name: str, other_view_name: str
    ):
        state = SessionState.create(config, {})
        job = make_job(job_id)
        getattr(state, own_view_name)[job_id] = job

        other_view = getattr(state, other_view_name)
        assert other_view.get(job_id) is None
        assert job_id not in other_view
        assert len(other_view) == 0
        assert list(other_view.values()) == []
        # ...but it exists in the union store and its own view.
        assert state.all_jobs[job_id] is job
        assert getattr(state, own_view_name)[job_id] is job

    def test_views_write_through_to_union_store(self, config: ServerConfig):
        state = SessionState.create(config, {})
        sim = make_sim_job("j1")
        batch = make_batch_job("b1")
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
            state.jobs["b1"] = make_batch_job("b1")  # type: ignore[assignment]
        with pytest.raises(
            TypeError, match=r"BatchJob view cannot store SimulationJob \(key 'j1'\)"
        ):
            state.batch_jobs["j1"] = make_sim_job("j1")  # type: ignore[assignment]
        # Nothing leaked into the union store.
        assert state.all_jobs == {}


class TestCancelRunningSnapshotsViews:
    """``cancel_running`` awaits runner ``cancel()`` mid-loop; jobs registered
    during that suspension must not invalidate the iteration over the live
    union store (lazy view iteration would raise ``RuntimeError: dictionary
    changed size during iteration``)."""

    @pytest.mark.parametrize(
        ("runner_key", "make_job", "register_attr"),
        [
            pytest.param("sim", make_sim_job, "add_sim_job", id="sim-job-registered-mid-cancel"),
            pytest.param(
                "sweep", make_batch_job, "add_batch_job", id="batch-job-registered-mid-cancel"
            ),
        ],
    )
    async def test_job_registered_during_cancel_does_not_break_iteration(
        self, config: ServerConfig, runner_key: str, make_job, register_attr: str
    ):
        state = SessionState.create(config, {})
        registry = state.job_registry
        register_late = getattr(registry, register_attr)

        class _RegisteringRunner(_RecordingRunner):
            async def cancel(
                self, job: SimulationJob | BatchJob, state: SessionState | None = None
            ) -> None:
                register_late(make_job(f"late-{job.job_id}"))
                await super().cancel(job, state)

        state.runners._runners[(runner_key, _RegisteringRunner, Path("."))] = _RegisteringRunner()
        for i in range(3):
            registry.jobs[f"run{i}"] = make_job(f"run{i}", status="running")

        await registry.cancel_running(state.runners, state)

        for i in range(3):
            assert registry.jobs[f"run{i}"].status == "cancelled"
            # Jobs registered mid-cancel land in the store untouched.
            assert registry.jobs[f"late-run{i}"].status == "completed"


class TestPerTypeEvictionCap:
    def test_each_job_type_capped_at_200_finished(self):
        """The registry keeps at most 200 finished jobs PER TYPE in the union
        store: 205 finished sims plus 205 finished batches leave 200 of each
        (the oldest five of each type evicted), 400 jobs total."""
        registry = JobRegistry(persist_enabled=False)
        base = now()
        for i in range(205):
            registry.add_sim_job(
                make_sim_job(f"sim{i:03d}", started_at=base + timedelta(seconds=i))
            )
        for i in range(205):
            registry.add_batch_job(
                make_batch_job(f"bat{i:03d}", started_at=base + timedelta(seconds=i))
            )

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
