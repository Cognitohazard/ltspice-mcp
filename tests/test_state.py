"""Tests for SessionState lifecycle and dataclass defaults."""

import asyncio
from pathlib import Path

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.state import BatchJob, MonteCarloConfig, SessionState, SimulationJob
from ltspice_mcp.tools import get_tools_for_profile


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
