"""Multi-process and multi-thread safety for the persistent job queue.

Parallel MCP sessions sharing a machine (common when a user has several
circuits open across clients) must not corrupt ``recent.json`` or lose
job records. These tests exercise the file-lock and atomic-write paths
directly with real processes/threads.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
from pathlib import Path

import pytest

from ltspice_mcp.lib import job_store, now, recent
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.state import SimulationJob

# ---------------------------------------------------------------------------
# Cross-process: recent.json
# ---------------------------------------------------------------------------


def _worker_touch_recent(home_dir: str, circuit_dir: str, index: int, passes: int) -> None:
    """Subprocess helper: touch a unique circuit path ``passes`` times."""
    os.environ["LTSPICE_MCP_HOME"] = home_dir
    circuit = Path(circuit_dir) / f"c{index}.cir"
    circuit.write_text("")
    from ltspice_mcp.lib import recent as _recent

    for _ in range(passes):
        _recent.touch(circuit, cap=200)


class TestRecentConcurrentProcesses:
    def test_parallel_touches_preserve_all_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        circuits = tmp_path / "circuits"
        circuits.mkdir()
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(home))

        ctx = mp.get_context("spawn")
        n_workers = 6
        passes = 8
        procs = [
            ctx.Process(
                target=_worker_touch_recent,
                args=(str(home), str(circuits), i, passes),
            )
            for i in range(n_workers)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0, f"worker {p.pid} exited {p.exitcode}"

        entries = recent.load()
        paths = {Path(e["path"]).name for e in entries}
        assert paths == {f"c{i}.cir" for i in range(n_workers)}
        # File is still valid JSON (no torn writes).
        raw = (home / "recent.json").read_text()
        assert json.loads(raw)["circuits"]


# ---------------------------------------------------------------------------
# Cross-thread: job_store atomic writes
# ---------------------------------------------------------------------------


class TestJobStoreConcurrentThreads:
    def test_many_threads_saving_different_jobs(self, tmp_path: Path) -> None:
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")

        def save(job_id: str) -> None:
            job = SimulationJob(
                job_id=job_id,
                netlist=circuit,
                simulator="LTspice",
                status="completed",
                started_at=now(),
                completed_at=now(),
            )
            job_store.save_job(job)

        threads = [
            threading.Thread(target=save, args=(f"sim_thread_{i}",))
            for i in range(32)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive()

        sim_jobs, _ = job_store.load_jobs_for_circuit(circuit)
        assert {j.job_id for j in sim_jobs} == {f"sim_thread_{i}" for i in range(32)}

    def test_same_job_rewritten_from_many_threads_stays_valid(
        self, tmp_path: Path
    ) -> None:
        """Concurrent writes to the same file must never leave a torn JSON."""
        circuit = tmp_path / "rc.cir"
        circuit.write_text("")
        job_id = "sim_same"
        # Initial save so the file exists.
        base = SimulationJob(
            job_id=job_id,
            netlist=circuit,
            simulator="LTspice",
            status="running",
            started_at=now(),
        )
        job_store.save_job(base)
        path = job_store.sidecar_dir(circuit) / f"{job_id}.json"

        def rewrite(status: str) -> None:
            job = SimulationJob(
                job_id=job_id,
                netlist=circuit,
                simulator="LTspice",
                status=status,  # type: ignore[arg-type]
                started_at=now(),
                completed_at=now() if status != "running" else None,
            )
            for _ in range(10):
                job_store.save_job(job)

        threads = [
            threading.Thread(target=rewrite, args=(s,))
            for s in ("running", "completed", "failed", "cancelled")
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # File must always be parseable; last-writer-wins semantics mean the
        # exact status is non-deterministic but the file never corrupts.
        data = json.loads(path.read_text())
        assert data["job_id"] == job_id
        assert data["status"] in {"running", "completed", "failed", "cancelled"}


# ---------------------------------------------------------------------------
# file_lock semantics
# ---------------------------------------------------------------------------


class TestFileLock:
    def test_exclusive_lock_serialises_writers(self, tmp_path: Path) -> None:
        target = tmp_path / "counter.txt"
        target.write_text("0")

        def increment() -> None:
            for _ in range(50):
                with file_lock(target):
                    current = int(target.read_text())
                    target.write_text(str(current + 1))

        threads = [threading.Thread(target=increment) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        # Without serialisation this would be dramatically less than 200.
        assert int(target.read_text()) == 200

    def test_lock_timeout_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "blocked.txt"
        target.touch()
        held = threading.Event()
        release = threading.Event()

        def hold_lock() -> None:
            with file_lock(target):
                held.set()
                release.wait(timeout=5)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        try:
            assert held.wait(timeout=5)
            with pytest.raises(TimeoutError), file_lock(target, timeout=0.1):
                pass
        finally:
            release.set()
            holder.join(timeout=5)
