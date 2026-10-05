"""Multi-process and multi-thread safety for the persistent job queue.

Parallel MCP sessions sharing a machine (common when a user has several
circuits open across clients) must not corrupt ``recent.json`` or lose
job records. These tests exercise the file-lock and atomic-write paths
directly with real processes/threads.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import threading
from pathlib import Path

import pytest

from ltspice_mcp.lib import filelock, recent
from ltspice_mcp.lib.filelock import file_lock
from tests.conftest import LIVENESS_S

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
                release.wait()

        holder = threading.Thread(target=hold_lock)
        holder.start()
        try:
            assert held.wait(timeout=LIVENESS_S)
            # timing: asserts the lock times out while the holder has it
            with pytest.raises(TimeoutError), file_lock(target, timeout=0.1):
                pass
        finally:
            release.set()
            holder.join(timeout=LIVENESS_S)


class TestAsyncFileLock:
    """The coroutine-side lock: it must not strand a flock when cancelled."""

    async def test_a_cancelled_waiter_leaves_no_lock_behind(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel can land while a worker thread's attempt is still in flight.

        The thread goes on to take the lock, and by then the coroutine that
        asked for it is gone — so the lock has to be handed back rather than
        held until the process exits. The attempt is held open, with the flock
        taken, until the cancel has landed, so the window is entered every time
        rather than raced for.
        """
        real_file_lock = filelock.file_lock
        real_try_acquire = filelock._try_acquire
        mid_attempt = threading.Event()
        cancelled = threading.Event()
        attempt_done = threading.Event()

        @contextlib.contextmanager
        def held_open_lock(target: Path, **kwargs: object):
            with real_file_lock(target, **kwargs):  # type: ignore[arg-type]
                mid_attempt.set()
                cancelled.wait(LIVENESS_S)
                yield

        # The attempt's arguments, the hand-off among them, stay referenced so
        # a release cannot be left to the collector.
        attempts: list[tuple[object, ...]] = []

        def recorded_attempt(*args: object) -> bool:
            attempts.append(args)
            try:
                return real_try_acquire(*args)  # type: ignore[arg-type]
            finally:
                attempt_done.set()

        monkeypatch.setattr(filelock, "file_lock", held_open_lock)
        monkeypatch.setattr(filelock, "_try_acquire", recorded_attempt)
        target = tmp_path / "gate.txt"
        target.touch()

        async def waiter() -> None:
            async with filelock.async_file_lock(target):
                pass

        task = asyncio.create_task(waiter())
        assert await asyncio.to_thread(mid_attempt.wait, 10), "the attempt never took the lock"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        cancelled.set()
        assert await asyncio.to_thread(attempt_done.wait, 10), "the attempt never finished"

        monkeypatch.undo()
        with file_lock(target, timeout=0):
            pass

    def test_a_lock_published_after_the_waiter_gave_up_is_released(self, tmp_path: Path) -> None:
        """The two halves of the hand-off, run the wrong way round on purpose.

        A worker thread can finish taking the lock only after the coroutine
        that asked for it has already abandoned the wait. Releasing it then has
        to be the code's doing: a flock left for the collector to notice is one
        another process waits on for as long as that takes. The ``held`` stack
        below stays referenced here precisely so nothing can be blamed on the
        collector.
        """
        target = tmp_path / "gate.txt"
        target.touch()
        handoff = filelock._LockHandoff()
        held = contextlib.ExitStack()
        held.enter_context(file_lock(target, timeout=0))

        handoff.abandon()  # the waiting coroutine was cancelled
        handoff.publish(held)  # and only then did the worker win the lock

        with file_lock(target, timeout=LIVENESS_S):
            pass
