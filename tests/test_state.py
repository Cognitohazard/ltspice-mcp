"""Tests for SessionState lifecycle and the union job store."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.cache import FileCache
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.runner_manager import RunnerManager
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools


def _experiment(
    working_dir: Path,
    circuit: Path,
    *,
    job_id: str,
    status: str = "completed",
) -> ExperimentJob:
    """A minimal ExperimentJob — one produced case over one deck."""
    case = ExperimentCase(
        case_id="case_0000",
        run_index=0,
        circuit="dut",
        circuit_path=circuit,
        staged_deck=circuit,
        deck_sha256="a" * 64,
        assignments={},
        status="produced" if status == "completed" else "queued",
    )
    return ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=Store(working_dir).job_record(job_id),
        cases=[case],
        sources=[
            SourceRecord(
                circuit="dut",
                path=circuit,
                sha256="b" * 64,
                staged_deck=circuit,
                manifest=[],
                simulator="FakeSim",
                dialect="ltspice",
            )
        ],
        simulator="FakeSim",
        completeness=Completeness(declared=1, expanded=1),
        status=status,  # type: ignore[arg-type]
    )


@pytest.fixture
def config(tmp_path: Path) -> ServerConfig:
    return ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])


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

    def test_create_populates_tool_defs(self, config: ServerConfig):
        state = SessionState.create(config, {})
        assert len(state.tool_defs) > 0
        # Every advertised tool dispatches, and nothing else does.
        def_names = {t.name for t in state.tool_defs}
        assert def_names == set(state.tool_dispatch)
        # The consolidated profile is the only one; state mirrors it exactly,
        # minus run_code, which a default config does not turn on.
        consolidated_defs, _ = get_tools(exclude=("run_code",))
        assert def_names == {tool_def.name for tool_def in consolidated_defs}


@pytest.mark.asyncio
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


class TestPersistDuringInterpreterTeardown:
    """Once the interpreter's default executor is gone, ``asyncio.to_thread``
    raises ``RuntimeError: cannot schedule new futures after shutdown`` — and
    the async persist's task exception was never retrieved, so the caller saw
    an irrelevant traceback while the status change it carried was LOST.
    Observed live three times: a script exiting while its job settled. The
    persist must fall back to the synchronous write (blocking is fine during
    teardown) so the record lands instead of the noise.
    """

    def test_persist_completes_synchronously_when_the_executor_is_gone(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        job = _experiment(work_dir, circuit, job_id="exp_teardown", status="running")

        async def executor_gone(fn, *args, **kwargs):
            raise RuntimeError("cannot schedule new futures after shutdown")

        monkeypatch.setattr(asyncio, "to_thread", executor_gone)

        async def go():
            registry.persist_job(job)
            await registry.drain_pending()

        asyncio.run(go())
        assert experiment_store.load_job("exp_teardown", work_dir) is not None


class TestPerTypeEvictionCap:
    def test_finished_jobs_capped_per_type(self, work_dir: Path):
        """The registry keeps at most 200 finished jobs PER TYPE in the union
        store: 205 finished experiments leave 200 (the oldest five evicted)."""
        circuit = work_dir / "deck.cir"
        circuit.write_text(".op\n.end\n")
        registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
        base = now()
        for i in range(205):
            job = _experiment(work_dir, circuit, job_id=f"exp{i:03d}", status="completed")
            job.started_at = base + timedelta(seconds=i)
            registry.add_experiment_job(job, already_persisted=True)

        assert len(registry.jobs) == 200
        assert len(registry.jobs) == 200
        for i in range(5):
            assert f"exp{i:03d}" not in registry.jobs
        assert "exp005" in registry.jobs
        assert "exp204" in registry.jobs
