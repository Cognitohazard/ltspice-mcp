"""Cross-process advisory file locking using stdlib primitives.

Parallel MCP sessions can target the same circuit directory or the same
global recent-circuits index. Without coordination, two processes doing
read-modify-write on ``recent.json`` would race and drop entries. This
module provides a context-managed lock backed by ``fcntl.flock`` on POSIX
and ``msvcrt.locking`` on Windows.

The lock is advisory — processes that don't cooperate can still clobber
the file — but every caller in this package goes through ``file_lock``.

Acquisition BLOCKS the calling thread: the poll loop below waits with
``time.sleep`` for up to ``timeout`` seconds (default 10). Never call
``file_lock`` from a coroutine on the event loop — wrap the whole
lock-and-write operation in ``asyncio.to_thread`` so a contended lock
parks a worker thread instead of freezing every in-flight request.
``TimeoutError`` still surfaces to the caller; don't retry around it.

A coroutine that has to *hold* the lock across its own awaits takes
``async_file_lock`` instead, which polls on the event loop and parks
nothing at all.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

from ltspice_mcp.lib.store import SIDECAR_DIRNAME

logger = logging.getLogger(__name__)

_LOCK_POLL_INTERVAL = 0.05  # seconds between retry attempts
DEFAULT_TIMEOUT = 10.0


def _lock_path_for(target: Path) -> Path:
    """Return the sibling ``.lock`` file path for ``target``."""
    return target.with_name(target.name + ".lock")


if sys.platform == "win32":
    import msvcrt

    # msvcrt.locking has no shared/exclusive distinction — every lock is
    # exclusive. Call sites today only need exclusive locks; ``file_lock``
    # documents this platform quirk.
    _LOCK_TRY_EXCS: tuple[type[BaseException], ...] = (OSError,)

    def _try_lock(fd: int, exclusive: bool) -> None:
        del exclusive  # Windows has no shared-lock mode.
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _release(fd: int) -> None:
        with contextlib.suppress(OSError):
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    _LOCK_TRY_EXCS = (BlockingIOError,)

    def _try_lock(fd: int, exclusive: bool) -> None:
        mode = (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
        fcntl.flock(fd, mode)

    def _release(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def _acquire(fd: int, *, exclusive: bool, timeout: float) -> None:
    """Poll-with-timeout loop shared by both platform back-ends."""
    start = time.monotonic()
    while True:
        try:
            _try_lock(fd, exclusive)
            return
        except _LOCK_TRY_EXCS:
            if time.monotonic() - start >= timeout:
                raise TimeoutError(f"Lock not acquired within {timeout:.1f}s") from None
            time.sleep(_LOCK_POLL_INTERVAL)


@contextlib.contextmanager
def file_lock(
    target: Path,
    *,
    exclusive: bool = True,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[None]:
    """Acquire a cross-process advisory lock associated with ``target``.

    The lock is held on a sibling ``{name}.lock`` file so the target's
    own lifecycle (create/delete/replace) is independent of the lock.

    Args:
        target: Path whose modification should be serialised.
        exclusive: If True (default), acquire an exclusive write lock.
            If False, acquire a shared read lock on POSIX. On Windows
            ``msvcrt.locking`` has no shared mode, so this flag is
            silently ignored and an exclusive lock is taken instead.
        timeout: Seconds to wait before raising ``TimeoutError``.

    Raises:
        TimeoutError: Lock was not available within ``timeout``.
        OSError: Lock file could not be opened/created.
    """
    lock_path = _lock_path_for(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        _acquire(fd, exclusive=exclusive, timeout=timeout)
        try:
            yield
        finally:
            _release(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Per-circuit locking: the in-process lock and the cross-process one, layered
# ---------------------------------------------------------------------------
#
# These live here rather than in the tool layer because the schematic edit
# engine (a lib module) takes them, and a core module must not import the layer
# that imports it. ``tools/_base`` re-exports them.
#
# The sidecar directory name comes from lib/store.py, which owns the layout:
# a second declaration here is a second thing to change when it moves.


def path_lock(registry: dict[Path, asyncio.Lock], path: Path, cap: int = 64) -> asyncio.Lock:
    """Get or create a per-path lock in ``registry``, LRU-bounded at ``cap``.

    Shared mechanism behind every per-file lock registry (schematic edits,
    ``.asc`` exports): refresh recency on hit; at capacity evict the oldest
    *unheld* lock — if all are held, overshoot temporarily rather than break
    mutual exclusion by evicting a lock someone is inside.
    """
    if path in registry:
        registry[path] = registry.pop(path)
        return registry[path]
    if len(registry) >= cap:
        for candidate in list(registry):
            if not registry[candidate].locked():
                del registry[candidate]
                break
    registry[path] = asyncio.Lock()
    return registry[path]


def circuit_lock_target(path: Path) -> Path:
    """Anchor for the cross-process lock on one circuit file.

    Lives under the circuit's ``.ltspice-mcp/locks/`` sidecar directory
    (``file_lock`` appends ``.lock``) so user directories aren't littered
    with lock files next to their circuits.
    """
    return path.parent / SIDECAR_DIRNAME / "locks" / path.name


class _LockHandoff:
    """One flock, passed from the worker that took it to the coroutine that asked.

    A worker cannot just return the lock. The coroutine awaiting it can be
    cancelled while an attempt is in flight, and a result nobody receives is a
    flock held until the process exits. So the worker publishes what it took
    and then asks whether the coroutine still wants it, while the coroutine
    withdraws its interest and then looks for a lock. The two orders are
    opposite, so whichever runs second sees the other's mark and releases —
    and a double release is a no-op, because closing an exhausted
    ``ExitStack`` unwinds nothing.
    """

    def __init__(self) -> None:
        self._held: contextlib.ExitStack | None = None
        self._abandoned = False

    def publish(self, held: contextlib.ExitStack) -> None:
        self._held = held
        if self._abandoned:
            self.release()

    def abandon(self) -> None:
        self._abandoned = True
        self.release()

    def release(self) -> None:
        held, self._held = self._held, None
        if held is not None:
            held.close()


def _try_acquire(target: Path, handoff: _LockHandoff, exclusive: bool) -> bool:
    """One non-blocking attempt, in a worker thread. True if the lock is ours.

    ``timeout=0`` makes ``file_lock`` try once and raise rather than poll, so
    this returns in the time of two syscalls whether or not it succeeded.
    """
    held = contextlib.ExitStack()
    try:
        held.enter_context(file_lock(target, exclusive=exclusive, timeout=0))
    except TimeoutError:
        return False
    handoff.publish(held)
    return True


@contextlib.asynccontextmanager
async def async_file_lock(
    target: Path,
    *,
    exclusive: bool = True,
    acquire_timeout: float = DEFAULT_TIMEOUT,
) -> AsyncIterator[None]:
    """``file_lock`` for a coroutine that has to hold it across its own awaits.

    One non-blocking attempt per hop: the attempt itself runs in a worker
    thread, per this module's contract, but the *waiting* happens here on
    ``asyncio.sleep`` and the deadline is enforced here too. So a contended
    lock — the request gate waits minutes for one — parks neither the event
    loop nor a thread from the pool every other offloaded read shares.

    That also removes the leak a single long blocking acquisition had: there
    is no worker sitting inside a poll loop that can take the lock after the
    coroutine has given up on it. A cancel arriving mid-attempt is handed to
    ``_LockHandoff``, which releases whatever that attempt won.

    ``TimeoutError`` surfaces to the caller, which decides what a timeout
    means for its operation.
    """
    handoff = _LockHandoff()
    try:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + acquire_timeout
        while not await asyncio.to_thread(_try_acquire, target, handoff, exclusive):
            if loop.time() >= deadline:
                raise TimeoutError(f"Lock not acquired within {acquire_timeout:.1f}s")
            await asyncio.sleep(_LOCK_POLL_INTERVAL)
        yield
    finally:
        # Release is two fast syscalls and must not await: a cancel arriving
        # here would abandon the lock at the first suspension point.
        handoff.abandon()


@contextlib.asynccontextmanager
async def circuit_file_lock(path: Path) -> AsyncIterator[None]:
    """Cross-process lock for mutations/exports of one circuit file.

    Parallel MCP server processes editing the same circuit serialize here —
    without it, the whole-file read-modify-write saves are last-writer-wins
    and a concurrent session's edit is silently lost. This is
    ``async_file_lock`` on the circuit's lock anchor, plus the one thing that
    is specific to a circuit: a wait that runs out says so in the caller's own
    terms. The translation is scoped to the acquire, so a ``TimeoutError`` the
    guarded work raises for its own reasons still reads as itself.

    Acquire this BEFORE fetching a cached editor: the editor cache re-stats
    the file on every fetch, so taking the lock first guarantees the stat
    sees a concurrent writer's completed save rather than a mid-edit state.
    (Residual: on coarse-mtime filesystems like WSL's /mnt/c a same-size
    rewrite within one mtime tick can still go undetected — see FileCache.)
    """
    from ltspice_mcp.errors import NetlistError

    # Read at call time rather than through a default argument, so the wait
    # this states and the wait it takes cannot disagree.
    timeout = DEFAULT_TIMEOUT
    async with contextlib.AsyncExitStack() as stack:
        try:
            await stack.enter_async_context(
                async_file_lock(circuit_lock_target(path), acquire_timeout=timeout)
            )
        except TimeoutError as e:
            raise NetlistError(
                f"{path.name} is locked by another ltspice-mcp process "
                f"(waited {timeout:.0f}s). Retry once its edit finishes."
            ) from e
        yield
