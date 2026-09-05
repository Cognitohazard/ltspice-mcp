"""Cross-process coordination between parallel server sessions.

Independent MCP server processes can share one working directory. Three
mechanisms keep them out of each other's way:

- the cross-process circuit-file lock — concurrent edits of the same file
  serialize, and the revision check runs inside the lock so a peer's committed
  write is seen instead of silently overwritten;
- owner-pid liveness in job sidecars — a live sibling's running job isn't
  mislabeled ``interrupted``, shutdown only cancels this process's own jobs,
  and a foreign job's status refreshes from disk at resolution time;
- the token-scoped simulator kill — cancel/timeout can only ever hit the
  job's own simulator process, never a sibling session's.

The "peer" in the lock tests is a thread holding the real ``file_lock`` on a
separate fd — flock/msvcrt contention is per open file description, so this
exercises the exact cross-process semantics without spawning a process.
"""

import hashlib
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib import proc_kill as proc_kill_mod
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.filelock import circuit_lock_target, file_lock
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.proc_kill import kill_simulator_by_token, simulator_executable_names
from ltspice_mcp.lib.schematic_ops import (
    get_asc_editor,
    resolve_pin,
)
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.state import SessionState
from tests._asc_ops import apply_ops, sha_of

#: The line a peer session appends while holding the lock.
_PEER_MARKER = b"TEXT -48 320 Left 2 ;external marker\n"


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
    async def test_asc_edit_sees_a_peer_write_instead_of_overwriting_it(
        self, asc_state: SessionState, asc_file: Path
    ):
        # The editor fetch and the revision check both run INSIDE the guard, so
        # a peer's completed write is seen. Without that ordering our edit would
        # read the pre-peer bytes, pass its own stale sha, and erase the peer's
        # work; with it, the stale revision is refused and nothing is written.
        sha_before = sha_of(asc_file)
        peer_version = asc_file.read_bytes() + _PEER_MARKER  # noqa: ASYNC240
        t = _hold_lock_then_write(asc_file, peer_version, hold_s=0.4)

        data = await apply_ops(
            asc_state,
            asc_file.name,
            [{"op": "set_component_value", "reference": "R1", "value": "2k2"}],
            expected_sha256=sha_before,
        )
        t.join(5)
        assert data["outcome"] == "failed"
        assert data["error"]["code"] == "revision_conflict"
        assert data["commit_state"] == "not_committed"
        payload = asc_file.read_bytes()  # noqa: ASYNC240
        assert _PEER_MARKER in payload, "the peer session's edit must survive"
        assert b"2k2" not in payload, "a refused edit must write nothing"

    async def test_asc_edit_on_the_peers_revision_keeps_both_edits(
        self, asc_state: SessionState, asc_file: Path
    ):
        # Same race, but the caller submits the peer's revision: our edit blocks
        # on the lock, re-reads inside it, and lands on top of the peer's bytes.
        peer_version = asc_file.read_bytes() + _PEER_MARKER  # noqa: ASYNC240
        peer_sha = hashlib.sha256(peer_version).hexdigest()
        t = _hold_lock_then_write(asc_file, peer_version, hold_s=0.4)

        data = await apply_ops(
            asc_state,
            asc_file.name,
            [{"op": "set_component_value", "reference": "R1", "value": "2k2"}],
            expected_sha256=peer_sha,
        )
        t.join(5)
        assert data["outcome"] == "complete"
        payload = asc_file.read_bytes()  # noqa: ASYNC240
        assert b"2k2" in payload, "our edit must survive"
        assert _PEER_MARKER in payload, "the peer session's edit must survive too"

    async def test_contended_lock_times_out_with_clear_error(
        self, asc_state: SessionState, asc_file: Path, monkeypatch
    ):
        import ltspice_mcp.lib.filelock as lock_mod

        # Shrink the acquisition window so the test doesn't sit out the
        # full default timeout.
        monkeypatch.setattr(lock_mod, "file_lock", lambda target: file_lock(target, timeout=0.2))
        t, release = _hold_lock_until_released(asc_file)
        try:
            with pytest.raises(NetlistError, match="locked by another ltspice-mcp process"):
                await apply_ops(
                    asc_state,
                    asc_file.name,
                    [{"op": "set_component_value", "reference": "R1", "value": "2k"}],
                )
        finally:
            release.set()
            t.join(5)

    async def test_pin_geometry_resolved_under_the_lock(
        self, asc_state: SessionState, asc_file: Path
    ):
        # A peer session moves R1 while holding the lock. Our add_net_label by
        # pin reference must resolve R1's position AFTER acquiring the lock
        # (post-move), not from the editor cached before it — otherwise the
        # label lands at the old, now-empty coordinate.
        original = asc_file.read_bytes()  # noqa: ASYNC240
        moved = original.replace(b"SYMBOL res 128 112 R90", b"SYMBOL res 128 240 R90")
        assert moved != original, "fixture layout changed — update the SYMBOL line above"
        moved_sha = hashlib.sha256(moved).hexdigest()
        t = _hold_lock_then_write(asc_file, moved, hold_s=0.4)

        data = await apply_ops(
            asc_state,
            asc_file.name,
            [{"op": "add_net_label", "net": "probe", "pin": "R1.1"}],
            expected_sha256=moved_sha,
        )
        t.join(5)
        assert data["outcome"] == "complete"
        x, y = resolve_pin("R1.1", get_asc_editor(asc_file, asc_state))
        text = asc_file.read_text(errors="replace")  # noqa: ASYNC240
        assert f"FLAG {x} {y} probe" in text, "label must sit at R1's post-move pin position"

    async def test_export_guard_locks_the_net_sidecar(
        self, asc_state: SessionState, asc_file: Path, monkeypatch
    ):
        # LTspice's export overwrites the sibling .net; a peer session editing
        # that .net holds ITS file lock, so the export guard must contend on
        # the .net lock too — not just the .asc.
        import ltspice_mcp.lib.filelock as lock_mod
        from ltspice_mcp.tools._base import asc_export_lock

        monkeypatch.setattr(lock_mod, "file_lock", lambda target: file_lock(target, timeout=0.2))
        t, release = _hold_lock_until_released(asc_file.with_suffix(".net"))
        try:
            with pytest.raises(NetlistError, match="locked by another ltspice-mcp process"):
                async with asc_export_lock(asc_file):
                    pass
        finally:
            release.set()
            t.join(5)

    async def test_lock_file_lives_in_sidecar_dir_not_next_to_circuit(
        self, asc_state: SessionState, asc_file: Path, work_dir: Path
    ):
        await apply_ops(
            asc_state,
            asc_file.name,
            [{"op": "set_component_value", "reference": "R1", "value": "2k"}],
        )
        assert (work_dir / ".ltspice-mcp" / "locks" / f"{asc_file.name}.lock").exists()
        assert not (work_dir / f"{asc_file.name}.lock").exists()


def _running_experiment(work_dir: Path, job_id: str, pid: int) -> ExperimentJob:
    """A running experiment recorded as owned by ``pid``."""
    circuit = work_dir / "deck.cir"
    if not circuit.exists():
        circuit.write_text(".op\n.end\n", encoding="utf-8")
    job = ExperimentJob(
        job_id=job_id,
        request_id=f"request-{job_id}",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="control-secret",
        store_path=experiment_store.record_path(job_id, work_dir),
        cases=[
            ExperimentCase(
                case_id="case_0000",
                run_index=0,
                circuit="dut",
                circuit_path=circuit,
                staged_deck=circuit,
                deck_sha256="a" * 64,
                assignments={},
                status="queued",
            )
        ],
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
        status="running",
    )
    job.owner_pid = pid
    return job


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
        job = _running_experiment(work_dir, "exp_livepeer", live_peer_pid)
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)
        assert loaded is not None
        assert loaded.status == "running"
        assert loaded.owner_pid == live_peer_pid

    def test_running_job_with_dead_owner_loads_interrupted(self, work_dir: Path):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()  # pid is now dead
        job = _running_experiment(work_dir, "exp_deadpeer", proc.pid)
        experiment_store.save_job(job)

        loaded = experiment_store.load_job(job.job_id, work_dir)
        assert loaded is not None
        assert loaded.status == "interrupted"

    @pytest.mark.asyncio
    async def test_shutdown_cancels_only_own_jobs(self, work_dir: Path, live_peer_pid: int):
        registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
        own = _running_experiment(work_dir, "exp_own", os.getpid())
        foreign = _running_experiment(work_dir, "exp_foreign", live_peer_pid)
        registry.jobs[own.job_id] = own
        registry.jobs[foreign.job_id] = foreign

        cancelled: list[str] = []

        class _StubRunners:
            def get_experiment_runner_for(self, job):
                return self

            async def cancel(self, job, **kwargs):
                cancelled.append(job.job_id)
                job.status = "cancelled"
                job.done_event.set()
                return []

        await registry.cancel_running(_StubRunners(), None)
        assert cancelled == ["exp_own"]
        assert foreign.status == "running", "a parallel session's live job must be left alone"

    def test_refresh_foreign_job_picks_up_owner_completion(
        self, work_dir: Path, live_peer_pid: int
    ):
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        stale = _running_experiment(work_dir, "exp_refresh", live_peer_pid)
        registry.jobs[stale.job_id] = stale
        # The owner finishes the job and persists the terminal state.
        done = _running_experiment(work_dir, "exp_refresh", live_peer_pid)
        done.status = "completed"
        done.cases[0].status = "produced"
        done.completeness.produced = 1
        experiment_store.save_job(done)

        fresh = registry.refresh_foreign_job(stale)
        assert fresh.status == "completed"
        # Off the loop (the worker-thread situation a resource read runs in):
        # the caller gets the owner's latest state, but the loop-owned registry
        # must not be mutated from a thread.
        assert registry.jobs["exp_refresh"] is stale

    @pytest.mark.asyncio
    async def test_refresh_on_the_loop_swaps_the_registry_entry(
        self, work_dir: Path, live_peer_pid: int
    ):
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        stale = _running_experiment(work_dir, "exp_async_refresh", live_peer_pid)
        registry.jobs[stale.job_id] = stale
        done = _running_experiment(work_dir, "exp_async_refresh", live_peer_pid)
        done.status = "completed"
        done.cases[0].status = "produced"
        done.completeness.produced = 1
        experiment_store.save_job(done)

        fresh = await registry.refresh_foreign_job_async(stale)
        assert fresh.status == "completed"
        assert registry.jobs["exp_async_refresh"] is fresh

    def test_refresh_foreign_job_leaves_own_jobs_alone(self, work_dir: Path):
        registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
        own = _running_experiment(work_dir, "exp_mine", os.getpid())
        registry.jobs[own.job_id] = own
        assert registry.refresh_foreign_job(own) is own


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

    def test_stemmed_id_kills_its_own_case_runs(self, monkeypatch):
        # Experiment ids carry the deck's name; the token must still match the
        # per-case run files staged as {job_id}_case_{n}.
        token = generate_id("exp", "RC Filter.v2")
        own_case = _FakeProc(501, "ngspice", ["ngspice", "-b", f"/runs/{token}_case_2.net"])
        self._iter(monkeypatch, [own_case])

        assert kill_simulator_by_token(token, {"ngspice"}) == 1
        assert own_case.killed

    def test_a_deck_named_after_an_older_job_id_does_not_cross_match(self, monkeypatch):
        # The adversarial stem: a deck named after an artifact of an earlier
        # job, so the older id appears verbatim inside the newer one. Folding
        # the stem's underscores away is what keeps the older job's cancel from
        # killing the newer job's simulator.
        older = generate_id("exp", "amp")
        newer = generate_id("exp", older)
        victim = _FakeProc(502, "ngspice", ["ngspice", "-b", f"/runs/{newer}_case_0.net"])
        self._iter(monkeypatch, [victim])

        assert kill_simulator_by_token(older, {"ngspice"}) == 0
        assert not victim.killed, "a later job whose deck was named after this id must be spared"

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
