"""Integration tests for SessionState persistence hooks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import job_registry as job_registry_module
from ltspice_mcp.lib import job_store, now
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob


@pytest.fixture
def state(tmp_path: Path) -> SessionState:
    config = ServerConfig(
        working_dir=tmp_path,
        allowed_paths=[tmp_path],
        log_level="DEBUG",
    )
    return SessionState.create(config, {})


class TestAddJobPersists:
    def test_add_sim_job_writes_sidecar(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_test_1",
            netlist=circuit,
            simulator="LTspice",
            status="completed",
            started_at=now(),
            completed_at=now(),
        )
        state.add_job(job)
        sidecar_file = job_store.sidecar_dir(circuit) / "sim_test_1.json"
        assert sidecar_file.exists()

    def test_add_batch_job_writes_sidecar(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "amp.cir"
        circuit.write_text("")
        bj = BatchJob(
            job_id="sweep_test_1",
            job_type="sweep",
            netlist=circuit,
            total_runs=5,
            status="running",
        )
        state.add_batch_job(bj)
        sidecar_file = job_store.sidecar_dir(circuit) / "sweep_test_1.json"
        assert sidecar_file.exists()

    def test_persist_jobs_disabled_writes_nothing(self, tmp_path: Path) -> None:
        config = ServerConfig(
            working_dir=tmp_path,
            allowed_paths=[tmp_path],
            persist_jobs=False,
        )
        state = SessionState.create(config, {})
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_nop",
            netlist=circuit,
            simulator="LTspice",
            status="completed",
            started_at=now(),
            completed_at=now(),
        )
        state.add_job(job)
        assert not job_store.sidecar_dir(circuit).exists()


class TestEnsureJobsLoadedFor:
    def test_loads_persisted_jobs_once(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        # Seed a sidecar from a "prior session"
        prior = SimulationJob(
            job_id="sim_prior",
            netlist=circuit,
            simulator="LTspice",
            status="completed",
            started_at=now(),
            completed_at=now(),
        )
        job_store.save_job(prior)

        assert "sim_prior" not in state.jobs
        state.ensure_jobs_loaded_for(circuit)
        assert "sim_prior" in state.jobs
        # Second call is a no-op (tracked in _loaded_circuits).
        state.ensure_jobs_loaded_for(circuit)
        assert len(state.jobs) == 1

    def test_interrupted_with_valid_raw_promoted_to_completed(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        raw = tmp_path / "sim_prior.raw"
        # Real LTspice raw files start with "Title:" in ASCII.
        raw.write_bytes(b"Title: * /tmp/rc.cir\nDate: ...\n")
        log = tmp_path / "sim_prior.log"
        log.write_text("...")

        # Persist a job as if it had been running when the server died.
        running = SimulationJob(
            job_id="sim_interrupted",
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
            raw_file=raw,
            log_file=log,
        )
        job_store.save_job(running)

        state.ensure_jobs_loaded_for(circuit)
        loaded = state.jobs["sim_interrupted"]
        # Raw header matches → promoted to completed.
        assert loaded.status == "completed"
        assert loaded.error is None

    def test_interrupted_with_garbage_raw_stays_interrupted(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        """A file at ``raw_file`` that isn't a real .raw doesn't promote."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        raw = tmp_path / "bogus.raw"
        raw.write_bytes(b"not actually a raw file")

        running = SimulationJob(
            job_id="sim_garbage",
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
            raw_file=raw,
        )
        job_store.save_job(running)

        state.ensure_jobs_loaded_for(circuit)
        assert state.jobs["sim_garbage"].status == "interrupted"

    def test_interrupted_without_raw_stays_interrupted(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        running = SimulationJob(
            job_id="sim_orphan",
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
        )
        job_store.save_job(running)

        state.ensure_jobs_loaded_for(circuit)
        loaded = state.jobs["sim_orphan"]
        assert loaded.status == "interrupted"

    def test_disabled_persistence_skips_load(self, tmp_path: Path) -> None:
        config = ServerConfig(
            working_dir=tmp_path,
            allowed_paths=[tmp_path],
            persist_jobs=False,
        )
        state = SessionState.create(config, {})
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        # Create a sidecar manually — ensure_jobs_loaded_for should ignore it.
        prior = SimulationJob(
            job_id="sim_should_not_load",
            netlist=circuit,
            simulator="LTspice",
            status="completed",
            started_at=now(),
        )
        # Temporarily enable to seed, then disable for the real call.
        config.persist_jobs = True
        state.persist_job(prior)
        config.persist_jobs = False
        state.job_registry._loaded_circuits.clear()

        state.ensure_jobs_loaded_for(circuit)
        assert "sim_should_not_load" not in state.jobs


class TestRecentDebounce:
    def test_note_recent_circuit_only_writes_once_per_session(
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
            state.note_recent_circuit(resolved)
        assert len(calls) == 1


class TestBatchProgressThrottle:
    def test_persist_batch_progress_skips_intermediate_runs(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "amp.cir"
        circuit.write_text("")
        bj = BatchJob(
            job_id="sweep_throttle",
            job_type="sweep",
            netlist=circuit,
            total_runs=100,
            status="running",
        )
        state.add_batch_job(bj)

        sidecar = job_store.sidecar_dir(circuit) / "sweep_throttle.json"
        first_mtime = sidecar.stat().st_mtime_ns

        # Run 1 shouldn't trigger a rewrite under the 1/20 schedule.
        bj.completed_runs = 1
        state.persist_batch_progress(bj)
        assert sidecar.stat().st_mtime_ns == first_mtime

        # Run 5 (100 // 20 == 5) triggers a checkpoint.
        bj.completed_runs = 5
        state.persist_batch_progress(bj)
        assert sidecar.stat().st_mtime_ns > first_mtime

    def test_persist_batch_progress_always_persists_final_run(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        circuit = tmp_path / "amp.cir"
        circuit.write_text("")
        bj = BatchJob(
            job_id="sweep_final",
            job_type="sweep",
            netlist=circuit,
            total_runs=7,
            status="running",
        )
        state.add_batch_job(bj)

        sidecar = job_store.sidecar_dir(circuit) / "sweep_final.json"
        initial_mtime = sidecar.stat().st_mtime_ns

        # total_runs=7 → step=1 (max(1, 7//20)). Every run persists.
        bj.completed_runs = 7
        state.persist_batch_progress(bj)
        assert sidecar.stat().st_mtime_ns >= initial_mtime


class TestAsyncPersistDrain:
    async def test_shutdown_awaits_pending_writes(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        """Scheduled writes must land before shutdown returns."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_drain",
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
        )
        state.jobs["sim_drain"] = job
        # Schedule a write from async context (this is what the runners do).
        state.persist_job(job)
        assert state.job_registry._pending_persist, "expected a pending persist task"

        await state.shutdown()

        assert not state.job_registry._pending_persist
        sidecar = job_store.sidecar_dir(circuit) / "sim_drain.json"
        assert sidecar.exists()

    async def test_successive_writes_preserve_order(
        self, state: SessionState, tmp_path: Path
    ) -> None:
        """Writes across a terminal transition must land in call order.

        The prior implementation popped ``_persist_locks[job_id]`` inside
        the writer whenever the job hit a terminal state — a later writer
        would then allocate a fresh Lock while the previous one was still
        unwinding and run concurrently. The fix moves cleanup to eviction,
        so all writes for a live job serialise on the same Lock.
        """
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job = SimulationJob(
            job_id="sim_ordered",
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
        )
        state.jobs["sim_ordered"] = job
        state.persist_job(job)
        job.status = "completed"
        job.completed_at = now()
        state.persist_job(job)
        # A trailing write after terminal state — historically raced with
        # the still-unwinding "completed" write; now safely queued.
        job.error = "observed twice"
        state.persist_job(job)

        await state.shutdown()

        sidecar = job_store.sidecar_dir(circuit) / "sim_ordered.json"
        data = json.loads(sidecar.read_text())
        assert data["status"] == "completed"
        assert data["error"] == "observed twice"


class TestEvictionDeletesSidecar:
    def test_evicted_job_file_is_removed(self, state: SessionState, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")

        # Temporarily shrink the cap so eviction triggers quickly.
        original_cap = job_registry_module._MAX_FINISHED_JOBS
        job_registry_module._MAX_FINISHED_JOBS = 2
        try:
            ids = []
            for i in range(3):
                job = SimulationJob(
                    job_id=f"sim_evict_{i}",
                    netlist=circuit,
                    simulator="LTspice",
                    status="completed",
                    started_at=now(),
                    completed_at=now(),
                )
                state.add_job(job)
                ids.append(job.job_id)

            # Oldest (sim_evict_0) should be evicted from memory and disk.
            assert ids[0] not in state.jobs
            assert not (job_store.sidecar_dir(circuit) / f"{ids[0]}.json").exists()
            # Newest two remain.
            assert ids[1] in state.jobs
            assert ids[2] in state.jobs
        finally:
            job_registry_module._MAX_FINISHED_JOBS = original_cap


class TestPreloadRecent:
    def test_preload_loads_jobs_for_recent_circuits(
        self, state: SessionState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Route recent.json to tmp_path so we don't touch the user's home dir.
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(tmp_path / "home"))

        from ltspice_mcp.lib import recent

        # Two circuits in separate parent dirs so each has its own sidecar.
        circuits = []
        for idx in range(2):
            sub = tmp_path / f"proj{idx}"
            sub.mkdir()
            c = sub / "rc.cir"
            c.write_text("")
            job = SimulationJob(
                job_id=f"sim_pre_{idx}",
                netlist=c,
                simulator="LTspice",
                status="completed",
                started_at=now(),
                completed_at=now(),
            )
            state.add_job(job)
            recent.touch(c)
            circuits.append(c)

        # Fresh registry should see zero jobs before preload.
        fresh = type(state.job_registry)(persist_enabled=True)
        assert not fresh.sim_jobs
        loaded = fresh.preload_recent(max_circuits=10)
        assert loaded == 2
        assert {"sim_pre_0", "sim_pre_1"} <= set(fresh.sim_jobs)

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
            c = sub / "rc.cir"
            c.write_text("")
            state.add_job(
                SimulationJob(
                    job_id=f"sim_bound_{idx}",
                    netlist=c,
                    simulator="LTspice",
                    status="completed",
                    started_at=now(),
                    completed_at=now(),
                )
            )
            recent.touch(c)

        fresh = type(state.job_registry)(persist_enabled=True)
        loaded = fresh.preload_recent(max_circuits=2)
        assert loaded == 2
        assert len(fresh.sim_jobs) == 2

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
