"""Generic file cache with (mtime, size)-based invalidation."""

import threading
from collections.abc import Callable
from pathlib import Path


class FileCache[T]:
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
    can run concurrently. All ``_entries`` operations are guarded by a lock;
    the factory deliberately runs *outside* it (see ``get``). The lock makes
    the *cache* safe, not the cached values: a value shared across threads
    must be treated as immutable after construction (true for ``RawRead``).
    Mutable values — the netlist/schematic editor instances — must only be
    created and used from the event-loop thread.

    Type parameter T is the type of cached value.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._entries: dict[Path, tuple[tuple[int, int], T]] = {}
        self._lock = threading.Lock()

    def get(self, path: Path, factory: Callable[[Path], T]) -> T:
        """Get cached value or create it via factory function.

        Args:
            path: File path to cache
            factory: Function to create value from path if not cached or stale

        Returns:
            Cached value if the file's (mtime, size) stamp is unchanged,
            otherwise a newly created value
        """
        try:
            st = path.stat()
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return factory(path)

        with self._lock:
            entry = self._entries.get(path)
            if entry is not None and entry[0] == stamp:
                return entry[1]

        # The factory runs outside the lock: two threads can both miss and
        # both build the value (benign — both derive from the same stamped
        # bytes, last store wins), which beats serialising multi-second
        # parses behind one mutex. A torn entry is impossible: the value is
        # fully constructed before the locked store.
        value = factory(path)
        with self._lock:
            self._entries[path] = (stamp, value)
        return value

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

        Args:
            path: File path to invalidate
        """
        with self._lock:
            self._entries.pop(path, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        with self._lock:
            self._entries.clear()

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
