"""Generic file cache with (mtime, size)-based invalidation."""

import threading
from collections.abc import Callable
from pathlib import Path
from typing import Generic, TypeVar

T = TypeVar("T")


class FileCache(Generic[T]):
    """Generic cache for file-derived data with mtime-based invalidation.

    The cache stores values keyed by file path and automatically invalidates
    entries when the file's modification time or size changes. (Size is part
    of the stamp because an in-place rewrite can land within the same mtime
    tick — e.g. an external process overwriting a ``.raw``/``.log`` in the
    shared store — which an mtime-only check would miss.) This is useful for
    caching parsed netlists, simulation results, or other file-derived data.

    Known limitation — ``(mtime, size)`` is a heuristic (the same one CPython's
    import system and most build tools use): a rewrite that preserves byte
    length *and* lands in the same mtime tick is not detected. In this server
    that residual is narrow — human-timescale external re-runs differ in mtime
    (seconds apart exceeds even coarse-granularity filesystems), and programmatic
    rapid rewrites typically change size. It is accepted rather than closed with
    a content digest, because hashing file *content* on every lookup would
    re-read the large ``.raw`` artifacts this cache exists to avoid re-reading
    (slow across the WSL ``/mnt/c`` boundary). For files the server mutates
    itself, the robust complement is explicit invalidation at the write boundary
    — e.g. the ``.asc`` editor cache is invalidated after every edit.

    Thread safety — handlers offload heavy factories (RawRead parses) to
    worker threads via ``asyncio.to_thread``, so ``get``/``set``/``invalidate``
    can run concurrently. All ``_entries`` operations are guarded by the main
    lock; the stat+factory runs under a per-path lock instead (see ``get``).
    The locks make the *table* safe, not the values — mutable editor
    instances are loop-only; see the concurrency contract in tools/_base.py.

    Type parameter T is the type of cached value.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._entries: dict[Path, tuple[tuple[int, int], T]] = {}
        self._lock = threading.Lock()
        # Per-path single-flight locks. Pruned together with their cache
        # entries (invalidate/clear), so the map is bounded by the live
        # entries plus paths probed but never cached this session — a few
        # dozen small Lock objects at worst for this server's workloads.
        self._key_locks: dict[Path, threading.Lock] = {}

    def get(self, path: Path, factory: Callable[[Path], T]) -> T:
        """Get cached value or create it via factory function.

        Single-flight per path: concurrent cold-cache readers of the SAME
        file serialise on a per-path lock, so one thread runs the
        multi-second parse and the rest reuse its stored value instead of
        each re-parsing. Different paths still build concurrently — the
        factory never runs under the main table lock.

        Args:
            path: File path to cache
            factory: Function to create value from path if not cached or stale

        Returns:
            Cached value if the file's (mtime, size) stamp is unchanged,
            otherwise a newly created value
        """
        with self._lock:
            key_lock = self._key_locks.setdefault(path, threading.Lock())

        with key_lock:
            try:
                st = path.stat()
                stamp = (st.st_mtime_ns, st.st_size)
            except OSError:
                return factory(path)

            # Re-check under the per-path lock: a follower that waited on
            # the leader's parse finds the freshly stored entry here.
            with self._lock:
                entry = self._entries.get(path)
            if entry is not None and entry[0] == stamp:
                return entry[1]

            value = factory(path)
            with self._lock:
                self._entries[path] = (stamp, value)
            return value

    def peek(self, path: Path) -> T | None:
        """Return the cached value if it is still fresh, else ``None``.

        Never runs a factory and never blocks on the per-path single-flight
        lock — one ``stat`` plus a stamp compare. Lets async callers skip
        the ``asyncio.to_thread`` hop on a guaranteed cache hit.
        """
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return None
        with self._lock:
            entry = self._entries.get(path)
            if entry is not None and entry[0] == stamp:
                return entry[1]
        return None

    def set(self, path: Path, value: T) -> None:
        """Store a value in the cache with the file's current mtime.

        Args:
            path: File path to associate the value with
            value: The value to cache
        """
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            stamp = (0, -1)
        with self._lock:
            self._entries[path] = (stamp, value)

    def invalidate(self, path: Path) -> None:
        """Remove a specific entry from cache.

        Also prunes the path's single-flight lock. A get() holding that
        lock keeps its own reference; the next get() simply creates a
        fresh one, which at worst lets two threads parse the same path
        once — benign, both derive from the same stamped bytes.

        Args:
            path: File path to invalidate
        """
        with self._lock:
            self._entries.pop(path, None)
            self._key_locks.pop(path, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._entries.clear()
            self._key_locks.clear()

    def items(self) -> list[tuple[Path, tuple[tuple[int, int], T]]]:
        """Return all cached entries as (path, (stamp, value)) pairs."""
        with self._lock:
            return list(self._entries.items())

    def keys(self) -> list[Path]:
        """Return all cached paths."""
        with self._lock:
            return list(self._entries.keys())

    def __contains__(self, path: Path) -> bool:
        """Check if a path is in the cache."""
        with self._lock:
            return path in self._entries

    def __len__(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._entries)
