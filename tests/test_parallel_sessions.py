"""Cross-process coordination between parallel server sessions.

Independent MCP server processes can share one working directory. Three
mechanisms keep them out of each other's way:

- the cross-process circuit-file lock — concurrent edits of the same file
  serialize (edit-on-latest) instead of last-writer-wins;
- owner-pid liveness in job sidecars — a live sibling's running job isn't
  mislabeled ``interrupted``, shutdown only cancels this process's own jobs,
  and a foreign job's status refreshes from disk at resolution time;
- the token-scoped simulator kill — cancel/timeout can only ever hit the
  job's own simulator process, never a sibling session's.

The "peer" in the lock tests is a thread holding the real ``file_lock`` on a
separate fd — flock/msvcrt contention is per open file description, so this
exercises the exact cross-process semantics without spawning a process.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import job_store, now
from ltspice_mcp.lib import proc_kill as proc_kill_mod
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.job_types import SimulationJob
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import circuit_lock_target
from ltspice_mcp.tools.circuit import handle_set_component_value
from tests.conftest import make_batch_job, make_sim_job


def _hold_lock_then_write(target: Path, content: bytes, hold_s: float) -> threading.Thread:
    """Peer session stand-in: grab the file's cross-process lock, write the
    file just before releasing. Returns the thread once the lock is held."""
    held = threading.Event()

    def peer() -> None:
        with file_lock(circuit_lock_target(target)):
            held.set()
            time.sleep(hold_s)
            target.write_bytes(content)

    t = threading.Thread(target=peer, daemon=True)
    t.start()
    if not held.wait(5):
        raise RuntimeError("peer thread failed to take the lock")
    return t


def _hold_lock_until_released(target: Path) -> tuple[threading.Thread, threading.Event]:
    """Peer session stand-in: hold the file's cross-process lock until the
    returned event is set. Returns (thread, release_event) once held."""
    held = threading.Event()
    release = threading.Event()

    def peer() -> None:
        with file_lock(circuit_lock_target(target)):
            held.set()
            release.wait(10)

    t = threading.Thread(target=peer, daemon=True)
    t.start()
    if not held.wait(5):
        raise RuntimeError("peer thread failed to take the lock")
    return t, release


@pytest.mark.asyncio
class TestCircuitFileLock:
    async def test_cir_edit_waits_for_peer_and_keeps_both_edits(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Without the cross-process lock this is last-writer-wins: our edit
        # reads the pre-peer bytes immediately and the peer's later write
        # erases it. With the lock, our edit blocks until the peer releases,
        # re-reads, and lands on top of the peer's version.
        cir = work_dir / "shared.cir"
        cir.write_text("* shared\nR1 in 0 1k\n.END\n")
        peer_version = b"* shared\nR1 in 0 1k\nC1 out 0 1n\n.END\n"
        t = _hold_lock_then_write(cir, peer_version, hold_s=0.4)

        await handle_set_component_value(
            {"path": cir.name, "reference": "R1", "value": "2k"}, state_no_sim
        )
        t.join(5)
        text = cir.read_text()
        assert "2k" in text, "our edit must survive"
        assert "C1 out 0 1n" in text, "the peer session's edit must survive too"

    async def test_asc_edit_waits_for_peer_and_keeps_both_edits(
        self, asc_state: SessionState, asc_file: Path, work_dir: Path
    ):
        # Same scenario through the cached-AscEditor path: the editor fetch
        # stats the file INSIDE the guard, so the peer's completed write
        # forces a reload instead of saving a stale in-memory editor.
        from ltspice_mcp.tools.circuit import handle_list_components

        await handle_list_components({"path": asc_file.name}, asc_state)  # warm the cache
        peer_version = asc_file.read_bytes() + b"TEXT -48 320 Left 2 ;external marker\n"  # noqa: ASYNC240
        t = _hold_lock_then_write(asc_file, peer_version, hold_s=0.4)

        await handle_set_component_value(
            {"path": asc_file.name, "reference": "R1", "value": "2k2"}, asc_state
        )
        t.join(5)
        data = asc_file.read_bytes()  # noqa: ASYNC240
        assert b"2k2" in data, "our edit must survive"
        assert b"external marker" in data, "the peer session's edit must survive too"

    async def test_contended_lock_times_out_with_clear_error(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        import ltspice_mcp.tools._base as base_mod

        cir = work_dir / "busy.cir"
        cir.write_text("* busy\nR1 in 0 1k\n.END\n")
        # Shrink the acquisition window so the test doesn't sit out the
        # full default timeout.
        monkeypatch.setattr(base_mod, "file_lock", lambda target: file_lock(target, timeout=0.2))
        t, release = _hold_lock_until_released(cir)
        try:
            with pytest.raises(NetlistError, match="locked by another ltspice-mcp process"):
                await handle_set_component_value(
                    {"path": cir.name, "reference": "R1", "value": "2k"}, state_no_sim
                )
        finally:
            release.set()
            t.join(5)

    async def test_pin_geometry_resolved_under_the_lock(
        self, asc_state: SessionState, asc_file: Path, work_dir: Path
    ):
        # A peer session moves R1 while holding the lock. Our add_net_label
        # by pin reference must resolve R1's position AFTER acquiring the
        # lock (post-move), not from the editor cached before it — otherwise
        # the label lands at the old, now-empty coordinate.
        from ltspice_mcp.tools.circuit import (
            NetLabelInput,
            _get_asc_editor,
            _resolve_pin,
            handle_add_net_label,
            handle_list_components,
        )

        await handle_list_components({"path": asc_file.name}, asc_state)  # warm the cache
        original = asc_file.read_bytes()  # noqa: ASYNC240
        moved = original.replace(b"SYMBOL res 128 112 R90", b"SYMBOL res 128 240 R90")
        assert moved != original, "fixture layout changed — update the SYMBOL line above"
        t = _hold_lock_then_write(asc_file, moved, hold_s=0.4)

        await handle_add_net_label(
            NetLabelInput(path=asc_file.name, net="probe", pin="R1.1"), asc_state
        )
        t.join(5)
        x, y = _resolve_pin("R1.1", _get_asc_editor(asc_file, asc_state))
        text = asc_file.read_text(errors="replace")  # noqa: ASYNC240
        assert f"FLAG {x} {y} probe" in text, "label must sit at R1's post-move pin position"

    async def test_export_guard_locks_the_net_sidecar(
        self, asc_state: SessionState, asc_file: Path, monkeypatch
    ):
        # LTspice's export overwrites the sibling .net; a peer session editing
        # that .net holds ITS file lock, so the export guard must contend on
        # the .net lock too — not just the .asc.
        import ltspice_mcp.tools._base as base_mod
        from ltspice_mcp.tools._base import asc_export_lock

        monkeypatch.setattr(base_mod, "file_lock", lambda target: file_lock(target, timeout=0.2))
        t, release = _hold_lock_until_released(asc_file.with_suffix(".net"))
        try:
            with pytest.raises(NetlistError, match="locked by another ltspice-mcp process"):
                async with asc_export_lock(asc_file):
                    pass
        finally:
            release.set()
            t.join(5)

    async def test_lock_file_lives_in_sidecar_dir_not_next_to_circuit(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        cir = work_dir / "tidy.cir"
        cir.write_text("* tidy\nR1 in 0 1k\n.END\n")
        await handle_set_component_value(
            {"path": cir.name, "reference": "R1", "value": "2k"}, state_no_sim
        )
        assert (work_dir / ".ltspice-mcp" / "locks" / "tidy.cir.lock").exists()
        assert not (work_dir / "tidy.cir.lock").exists()


def _make_running_job(work_dir: Path, job_id: str, pid: int) -> SimulationJob:
    return make_sim_job(job_id, status="running", netlist=work_dir / "deck.cir", owner_pid=pid)


@pytest.fixture(scope="module")
def live_peer_pid():
    """A real, live process that is not this one (a parallel session stand-in)."""
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    yield proc.pid
    proc.kill()
    proc.wait()


class TestOwnerPidLiveness:
    def test_running_job_with_live_owner_stays_running(self, work_dir: Path, live_peer_pid: int):
        job = _make_running_job(work_dir, "sim_1_livepeer", live_peer_pid)
        job_store.save_job(job)
        sim_jobs, _ = job_store.load_jobs_for_circuit(work_dir / "deck.cir")
        assert len(sim_jobs) == 1
        assert sim_jobs[0].status == "running"
        assert sim_jobs[0].owner_pid == live_peer_pid
        assert sim_jobs[0].error is None

    def test_running_job_with_dead_owner_loads_interrupted(self, work_dir: Path):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()  # pid is now dead
        job = _make_running_job(work_dir, "sim_1_deadpeer", proc.pid)
        job_store.save_job(job)
        sim_jobs, _ = job_store.load_jobs_for_circuit(work_dir / "deck.cir")
        assert sim_jobs[0].status == "interrupted"

    def test_running_record_without_pid_loads_interrupted(self, work_dir: Path):
        # Records from builds that predate the pid field: no liveness signal,
        # so the pre-pid behavior (interrupted) stands.
        job = _make_running_job(work_dir, "sim_1_legacy", os.getpid())
        data = job_store.serialize_job(job)
        del data["pid"]
        target = job_store.sidecar_dir(job.netlist) / f"{job.job_id}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        import json

        target.write_text(json.dumps(data, default=str))
        sim_jobs, _ = job_store.load_jobs_for_circuit(work_dir / "deck.cir")
        assert sim_jobs[0].status == "interrupted"
        assert sim_jobs[0].owner_pid == 0

    def test_own_pid_in_record_counts_as_dead(self, work_dir: Path):
        # A record carrying OUR pid can't be ours (it isn't in our registry),
        # so it must be a recycled pid — treated as a dead owner.
        job = _make_running_job(work_dir, "sim_1_recycled", os.getpid())
        job_store.save_job(job)
        sim_jobs, _ = job_store.load_jobs_for_circuit(work_dir / "deck.cir")
        assert sim_jobs[0].status == "interrupted"

    @pytest.mark.asyncio
    async def test_shutdown_cancels_only_own_jobs(self, work_dir: Path, live_peer_pid: int):
        registry = JobRegistry(persist_enabled=False)
        own = _make_running_job(work_dir, "sim_1_own", os.getpid())
        foreign = _make_running_job(work_dir, "sim_1_foreign", live_peer_pid)
        registry.jobs[own.job_id] = own
        registry.jobs[foreign.job_id] = foreign

        cancelled: list[str] = []

        class _StubRunners:
            def get_existing_sim_runner(self):
                return self

            def get_existing_sweep_runner(self):
                return None

            def get_existing_mc_runner(self):
                return None

            async def cancel(self, job, state):
                cancelled.append(job.job_id)

        await registry.cancel_running(_StubRunners(), None)
        assert cancelled == ["sim_1_own"]
        assert foreign.status == "running", "a parallel session's live job must be left alone"

    @pytest.mark.asyncio
    async def test_refresh_foreign_job_picks_up_owner_completion(
        self, work_dir: Path, live_peer_pid: int
    ):
        # Async test: the registry swap is loop-only, so this runs on a loop.
        registry = JobRegistry(persist_enabled=True)
        stale = _make_running_job(work_dir, "sim_1_refresh", live_peer_pid)
        registry.jobs[stale.job_id] = stale
        # The owner finishes the job and persists the terminal state.
        done = _make_running_job(work_dir, "sim_1_refresh", live_peer_pid)
        done.status = "completed"
        done.completed_at = now()
        job_store.save_job(done)

        fresh = registry.refresh_foreign_job(stale)
        assert fresh.status == "completed"
        assert registry.jobs["sim_1_refresh"] is fresh

    def test_refresh_off_loop_returns_fresh_without_registry_swap(
        self, work_dir: Path, live_peer_pid: int
    ):
        # Sync test = no running loop, the worker-thread situation (resource
        # reads run there). The caller still gets the owner's latest state,
        # but the loop-owned registry must not be mutated off-loop.
        registry = JobRegistry(persist_enabled=True)
        stale = _make_running_job(work_dir, "sim_1_threadview", live_peer_pid)
        registry.jobs[stale.job_id] = stale
        done = _make_running_job(work_dir, "sim_1_threadview", live_peer_pid)
        done.status = "completed"
        done.completed_at = now()
        job_store.save_job(done)

        fresh = registry.refresh_foreign_job(stale)
        assert fresh.status == "completed"
        assert registry.jobs["sim_1_threadview"] is stale

    def test_refresh_foreign_job_leaves_own_jobs_alone(self, work_dir: Path):
        registry = JobRegistry(persist_enabled=True)
        own = _make_running_job(work_dir, "sim_1_mine", os.getpid())
        registry.jobs[own.job_id] = own
        assert registry.refresh_foreign_job(own) is own

    def test_batch_jobs_roundtrip_owner_pid(self, work_dir: Path, live_peer_pid: int):
        bj = make_batch_job(
            "sweep_1_peer",
            status="running",
            netlist=work_dir / "deck.cir",
            owner_pid=live_peer_pid,
        )
        job_store.save_job(bj)
        _, batch_jobs = job_store.load_jobs_for_circuit(work_dir / "deck.cir")
        assert batch_jobs[0].status == "running"
        assert batch_jobs[0].owner_pid == live_peer_pid


class _FakeProc:
    def __init__(self, pid: int, name: str, cmdline: list[str]):
        self.pid = pid
        self.info = {"name": name, "cmdline": cmdline}
        self.killed = False

    def kill(self):
        self.killed = True


class TestScopedKill:
    def _iter(self, monkeypatch, procs):
        monkeypatch.setattr(proc_kill_mod.psutil, "process_iter", lambda attrs: iter(procs))

    def test_kills_only_name_and_token_match(self, monkeypatch):
        token = "sim_1751000000_deadbeef"
        target = _FakeProc(101, "ngspice", ["ngspice", "-b", f"/tmp/runs/{token}.cir"])
        other_job = _FakeProc(102, "ngspice", ["ngspice", "-b", "/tmp/runs/sim_x_other.cir"])
        token_wrong_name = _FakeProc(103, "python3", ["python3", f"analyze_{token}.py"])
        self._iter(monkeypatch, [target, other_job, token_wrong_name])

        assert kill_simulator_by_token(token, {"ngspice"}) == 1
        assert target.killed
        assert not other_job.killed, "a sibling session's simulator must not be touched"
        assert not token_wrong_name.killed, "token match without the simulator name is spared"

    def test_wine_process_matched_via_argv_basename(self, monkeypatch):
        # Under Wine the psutil name can be the loader, not the simulator —
        # the executable basename in argv is what identifies it.
        token = "sim_1751000000_cafe0123"
        wine = _FakeProc(
            201,
            "wine-preloader",
            ["/usr/bin/wine", "/opt/lt/LTspice.exe", "-Run", "-b", f"Z:\\runs\\{token}.net"],
        )
        self._iter(monkeypatch, [wine])
        assert kill_simulator_by_token(token, {"wine", "ltspice.exe"}) == 1
        assert wine.killed

    def test_token_matches_only_at_filename_boundary(self, monkeypatch):
        # A job id must not match a longer id it happens to prefix; it must
        # still match its own single-run file ({id}.cir) and its batch
        # sub-runs ({id}_{n}.cir).
        token = "sim_1_ab"
        longer_id = _FakeProc(401, "ngspice", ["ngspice", "-b", "/runs/sim_1_abc.cir"])
        own_single = _FakeProc(402, "ngspice", ["ngspice", "-b", "/runs/sim_1_ab.cir"])
        own_subrun = _FakeProc(403, "ngspice", ["ngspice", "-b", "/runs/sim_1_ab_3.cir"])
        self._iter(monkeypatch, [longer_id, own_single, own_subrun])

        assert kill_simulator_by_token(token, {"ngspice"}) == 2
        assert not longer_id.killed, "a different job whose id extends ours must be spared"
        assert own_single.killed
        assert own_subrun.killed

    def test_vanished_process_is_skipped(self, monkeypatch):
        token = "sim_1751000000_feedf00d"
        ghost = _FakeProc(301, "ngspice", ["ngspice", "-b", f"{token}.cir"])

        def _gone():
            raise psutil.NoSuchProcess(ghost.pid)

        ghost.kill = _gone  # type: ignore[method-assign]
        self._iter(monkeypatch, [ghost])
        assert kill_simulator_by_token(token, {"ngspice"}) == 0

    def test_empty_inputs_kill_nothing(self, monkeypatch):
        def _must_not_scan(attrs):
            raise AssertionError("process table must not be scanned")

        monkeypatch.setattr(proc_kill_mod.psutil, "process_iter", _must_not_scan)
        assert kill_simulator_by_token("", {"ngspice"}) == 0
        assert kill_simulator_by_token("sim_1_x", set()) == 0

    def test_simulator_executable_names(self):
        from typing import ClassVar

        class FakeLTspice:
            spice_exe: ClassVar[list[str]] = ["C:/Program Files/ADI/LTspice/LTspice.exe"]
            process_name = "LTspice.exe"

        class FakeWineLTspice:
            spice_exe: ClassVar[list[str]] = ["wine", "/opt/lt/LTspice.exe"]
            process_name = ""

        class Bare:
            pass

        assert simulator_executable_names(FakeLTspice) == frozenset({"ltspice.exe"})
        assert simulator_executable_names(FakeWineLTspice) == frozenset({"wine", "ltspice.exe"})
        assert simulator_executable_names(Bare) == frozenset()
