"""Integration tests for SessionState persistence hooks."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from tests.conftest import make_legacy_record, write_legacy_sidecar


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    config = ServerConfig(
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        log_level="DEBUG",
    )
    return SessionState.create(config, {})


class TestEnsureJobsLoadedFor:
    """Loading a circuit's sidecar directory: once per session, dedup against
    what is already in memory, and nothing at all with persistence off."""

    def test_loads_persisted_records_once(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        write_legacy_sidecar(circuit, "sim_prior")

        assert "sim_prior" not in state.all_jobs
        state.ensure_jobs_loaded_for(circuit)
        assert "sim_prior" in state.all_jobs
        # Second call is a no-op (tracked in _loaded_circuits).
        state.ensure_jobs_loaded_for(circuit)
        assert len(state.all_jobs) == 1

    @pytest.mark.asyncio
    async def test_async_loader_matches_sync(self, state: SessionState, tmp_path: Path) -> None:
        # ensure_loaded_for_async offloads the sidecar read but must apply the
        # same registry state as the sync path (the loop-freeze fix on the common
        # dispatch path), and dedup on second call the same way.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        write_legacy_sidecar(circuit, "sim_async")

        assert "sim_async" not in state.all_jobs
        await state.ensure_jobs_loaded_for_async(circuit)
        assert "sim_async" in state.all_jobs
        await state.ensure_jobs_loaded_for_async(circuit)
        assert len(state.all_jobs) == 1

    def test_reload_dedupes_against_union_store(self, state: SessionState, tmp_path: Path) -> None:
        """A job id already present in the union store is skipped on load —
        the in-memory object is never replaced by its persisted copy."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        write_legacy_sidecar(circuit, "sim_dupe", status="completed")

        live = make_legacy_record("sim_dupe", status="interrupted", netlist=circuit)
        state.job_registry.jobs["sim_dupe"] = live

        state.ensure_jobs_loaded_for(circuit)

        assert state.all_jobs["sim_dupe"] is live
        assert len(state.all_jobs) == 1

    def test_a_record_keeps_the_status_its_release_wrote(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        # No recovery promotion: this version has no runner that could confirm
        # or resume the run, so the status stands as persisted rather than being
        # upgraded from an artifact nothing here produced.
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        raw = circuit.with_suffix(".raw")
        raw.write_bytes(b"Title: * rc\nPlotname: Transient Analysis\n")
        write_legacy_sidecar(circuit, "sim_int", status="interrupted")

        state.ensure_jobs_loaded_for(circuit)
        assert state.all_jobs["sim_int"].status == "interrupted"

    def test_disabled_persistence_skips_load(self, tmp_path: Path) -> None:
        config = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path], persist_jobs=False)
        state = SessionState.create(config, available={})
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        write_legacy_sidecar(circuit, "sim_off")

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

        # Two circuits in separate parent dirs so each has its own sidecar.
        for idx in range(2):
            sub = tmp_path / f"proj{idx}"
            sub.mkdir()
            circuit = sub / "rc.cir"
            circuit.write_text("")
            write_legacy_sidecar(circuit, f"sim_pre_{idx}")
            recent.touch(circuit)

        # Fresh registry should see zero jobs before preload.
        fresh = type(state.job_registry)(persist_enabled=True)
        assert not fresh.jobs
        loaded = fresh.preload_recent(max_circuits=10)
        assert loaded == 2
        assert {"sim_pre_0", "sim_pre_1"} <= set(fresh.jobs)

    def test_preload_bounded_by_max_circuits(
        self, state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))
        from ltspice_mcp.lib import recent

        # Each circuit needs its own parent dir — sidecars are stored at
        # ``<parent>/.ltspice-mcp/jobs/``, so siblings share one sidecar
        # directory and loading any one of them would fetch all jobs.
        for idx in range(5):
            sub = tmp_path / f"proj{idx}"
            sub.mkdir()
            circuit = sub / "rc.cir"
            circuit.write_text("")
            write_legacy_sidecar(circuit, f"sim_bound_{idx}")
            recent.touch(circuit)

        fresh = type(state.job_registry)(persist_enabled=True)
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
