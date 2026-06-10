"""Tests for SessionState lifecycle and dataclass defaults."""

import asyncio
from pathlib import Path

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.job_lifecycle import transition
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
