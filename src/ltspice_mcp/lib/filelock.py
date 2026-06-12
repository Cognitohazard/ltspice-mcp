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

import contextlib
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path

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
