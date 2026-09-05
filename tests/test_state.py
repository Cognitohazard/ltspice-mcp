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
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools_for_profile
from tests.conftest import make_legacy_record


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
        store_path=experiment_store.record_path(job_id, working_dir),
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
        # The consolidated profile is the only one; state mirrors it exactly.
        consolidated_defs, _ = get_tools_for_profile("consolidated")
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

    async def test_shutdown_leaves_a_legacy_record_alone(self, config: ServerConfig):
        # Nothing in this process launched it, so there is no simulator to kill
        # and no status of ours to write over the one its release persisted.
        state = SessionState.create(config, {})
        record = make_legacy_record("sim_old", status="interrupted")
        state.job_registry.jobs["sim_old"] = record

        await state.shutdown()
        assert state.all_jobs["sim_old"].status == "interrupted"


class TestUnionJobStoreViews:
    """``state.legacy_records`` / ``state.experiment_jobs`` are type-filtered
    writable views over the single union store (``state.all_jobs``): lookups
    surface only the view's job type, writes go through to the union dict."""

    def test_record_invisible_through_the_experiment_view(self, config: ServerConfig):
        state = SessionState.create(config, {})
        record = make_legacy_record("j1")
        state.legacy_records["j1"] = record

        other_view = state.experiment_jobs
        assert other_view.get("j1") is None
        assert "j1" not in other_view
        assert len(other_view) == 0
        assert list(other_view.values()) == []
        # ...but it exists in the union store and its own view.
        assert state.all_jobs["j1"] is record
        assert state.legacy_records["j1"] is record

    def test_views_write_through_to_union_store(self, config: ServerConfig):
        state = SessionState.create(config, {})
        record = make_legacy_record("j1")
        state.legacy_records["j1"] = record

        assert state.all_jobs == {"j1": record}
        assert set(state.legacy_records) == {"j1"}
        assert len(state.experiment_jobs) == 0

    def test_view_write_rejects_wrong_job_type(self, config: ServerConfig):
        """Writing a job of the wrong type through a typed view must fail
        loudly: silently accepting it would store a job that is invisible
        through the view that wrote it."""
        state = SessionState.create(config, {})

        with pytest.raises(TypeError, match=r"ExperimentJob view cannot store LegacyJobRecord"):
            state.experiment_jobs["j1"] = make_legacy_record("j1")  # type: ignore[assignment]
        assert state.all_jobs == {}


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

        assert len(registry.experiment_jobs) == 200
        assert len(registry.jobs) == 200
        for i in range(5):
            assert f"exp{i:03d}" not in registry.jobs
        assert "exp005" in registry.experiment_jobs
        assert "exp204" in registry.experiment_jobs
