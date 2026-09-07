"""Integration tests for SessionState persistence hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from tests.conftest import persist_experiment_record


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    config = ServerConfig(
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        log_level="DEBUG",
    )
    return SessionState.create(config, {})


class TestEnsureJobsLoadedFor:
    """Loading a circuit's persisted jobs: once per session, dedup against
    what is already in memory, and nothing at all with persistence off."""

    def test_loads_persisted_records_once(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        persist_experiment_record(tmp_path, circuit, "exp_prior")

        assert "exp_prior" not in state.all_jobs
        state.ensure_jobs_loaded_for(circuit)
        assert "exp_prior" in state.all_jobs
        # Second call is a no-op (tracked in _loaded_circuits).
        state.ensure_jobs_loaded_for(circuit)
        assert len(state.all_jobs) == 1

    @pytest.mark.asyncio
    async def test_async_loader_matches_sync(self, state: SessionState, tmp_path: Path) -> None:
        # ensure_loaded_for_async offloads the record read but must apply the
        # same registry state as the sync path (the loop-freeze fix on the common
        # dispatch path), and dedup on second call the same way.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        persist_experiment_record(tmp_path, circuit, "exp_async")

        assert "exp_async" not in state.all_jobs
        await state.ensure_jobs_loaded_for_async(circuit)
        assert "exp_async" in state.all_jobs
        await state.ensure_jobs_loaded_for_async(circuit)
        assert len(state.all_jobs) == 1

    def test_reload_dedupes_against_the_job_store(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        """A job id already in memory is skipped on load — the in-memory object
        is never replaced by its persisted copy."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        persist_experiment_record(tmp_path, circuit, "exp_dupe")

        state.ensure_jobs_loaded_for(circuit)
        live = state.all_jobs["exp_dupe"]

        # Release the once-per-session claim so the load actually runs a second
        # time; without that the dedup under test is never reached.
        state.job_registry._loaded_circuits.clear()
        state.ensure_jobs_loaded_for(circuit)

        assert state.all_jobs["exp_dupe"] is live
        assert len(state.all_jobs) == 1

    def test_disabled_persistence_skips_load(self, tmp_path: Path) -> None:
        config = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path], persist_jobs=False)
        state = SessionState.create(config, available={})
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        persist_experiment_record(tmp_path, circuit, "exp_off")

        state.ensure_jobs_loaded_for(circuit)
        assert state.all_jobs == {}


class TestRecentDebounce:
    async def test_note_recent_circuit_only_writes_once_per_session(
        self,
        state: SessionState,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from ltspice_mcp.lib import recent

        monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        resolved = circuit.resolve()

        calls: list[Path] = []
        real_touch = recent.touch

        def spy(p: Path, **kwargs: object) -> None:
            calls.append(p)
            real_touch(p, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(recent, "touch", spy)

        for _ in range(5):
            await state.note_recent_circuit(resolved)
        assert len(calls) == 1


class TestPreloadRecent:
    def test_preload_loads_records_for_recent_circuits(
        self, state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Route recent.json to tmp_path so we don't touch the user's home dir.
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))

        from ltspice_mcp.lib import recent

        for idx in range(2):
            sub = tmp_path / f"proj{idx}"
            sub.mkdir()
            circuit = sub / "rc.cir"
            circuit.write_text("")
            persist_experiment_record(tmp_path, circuit, f"exp_pre_{idx}")
            recent.touch(circuit)

        # Fresh registry should see zero jobs before preload.
        fresh = type(state.job_registry)(persist_enabled=True, working_dir=tmp_path)
        assert not fresh.jobs
        loaded = fresh.preload_recent(max_circuits=10)
        assert loaded == 2
        assert {"exp_pre_0", "exp_pre_1"} <= set(fresh.jobs)

    def test_preload_bounded_by_max_circuits(
        self, state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
        from ltspice_mcp.lib import recent

        for idx in range(5):
            sub = tmp_path / f"proj{idx}"
            sub.mkdir()
            circuit = sub / "rc.cir"
            circuit.write_text("")
            persist_experiment_record(tmp_path, circuit, f"exp_bound_{idx}")
            recent.touch(circuit)

        fresh = type(state.job_registry)(persist_enabled=True, working_dir=tmp_path)
        loaded = fresh.preload_recent(max_circuits=2)
        assert loaded == 2
        assert len(fresh.jobs) == 2

    def test_preload_zero_is_noop(self, state: SessionState) -> None:
        assert state.job_registry.preload_recent(max_circuits=0) == 0

    def test_preload_disabled_persistence_is_noop(self, tmp_path: Path) -> None:
        config = ServerConfig(
            working_dir=tmp_path,
            allowed_paths=[tmp_path],
            persist_jobs=False,
            log_level="DEBUG",
        )
        state = SessionState.create(config, {})
        assert state.job_registry.preload_recent(max_circuits=10) == 0
