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


@contextlib.asynccontextmanager
async def circuit_file_lock(path: Path) -> AsyncIterator[None]:
    """Cross-process lock for mutations/exports of one circuit file.

    Parallel MCP server processes editing the same circuit serialize here —
    without it, the whole-file read-modify-write saves are last-writer-wins
    and a concurrent session's edit is silently lost. Acquisition polls in a
    worker thread (per this module's contract, so a contended lock never
    stalls the event loop); release is two fast syscalls, done inline.

    Acquire this BEFORE fetching a cached editor: the editor cache re-stats
    the file on every fetch, so taking the lock first guarantees the stat
    sees a concurrent writer's completed save rather than a mid-edit state.
    (Residual: on coarse-mtime filesystems like WSL's /mnt/c a same-size
    rewrite within one mtime tick can still go undetected — see FileCache.)
    """
    # Acquire INSIDE the try so stack.close() always runs: a cancel landing at
    # the await boundary right after the worker thread took the flock would
    # otherwise leak it until process exit. (Residual: if the cancel lands
    # while the worker is still blocked acquiring, the thread can register the
    # lock after close() already ran — inherent to to_thread, not fixable
    # without a cancel-aware lock; the narrow window is cancel-only.)
    from ltspice_mcp.errors import NetlistError

    stack = contextlib.ExitStack()
    try:
        try:
            await asyncio.to_thread(stack.enter_context, file_lock(circuit_lock_target(path)))
        except TimeoutError as e:
            raise NetlistError(
                f"{path.name} is locked by another ltspice-mcp process "
                f"(waited {DEFAULT_TIMEOUT:.0f}s). Retry once its edit finishes."
            ) from e
        yield
    finally:
        stack.close()
