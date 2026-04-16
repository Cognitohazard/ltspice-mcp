"""Generic file cache with mtime-based invalidation."""

from collections.abc import Callable
from pathlib import Path


class FileCache[T]:
    """Generic cache for file-derived data with mtime-based invalidation.

    The cache stores values keyed by file path and automatically invalidates
    entries when the file's modification time changes. This is useful for
    caching parsed netlists, simulation results, or other file-derived data.

    Type parameter T is the type of cached value.
    """

    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._entries: dict[Path, tuple[float, T]] = {}

    def get(self, path: Path, factory: Callable[[Path], T]) -> T:
        """Get cached value or create it via factory function.

        Args:
            path: File path to cache
            factory: Function to create value from path if not cached or stale

        Returns:
            Cached value if mtime matches, otherwise newly created value
        """
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return factory(path)

        entry = self._entries.get(path)
        if entry is not None and entry[0] == mtime:
            return entry[1]

        value = factory(path)
        self._entries[path] = (mtime, value)
        return value

    def set(self, path: Path, value: T) -> None:
        """Store a value in the cache with the file's current mtime.

        Args:
            path: File path to associate the value with
            value: The value to cache
        """
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        self._entries[path] = (mtime, value)

    def invalidate(self, path: Path) -> None:
        """Remove a specific entry from cache.

        Args:
            path: File path to invalidate
        """
        self._entries.pop(path, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._entries.clear()

    def items(self) -> list[tuple[Path, tuple[float, T]]]:
        """Return all cached entries as (path, (mtime, value)) pairs."""
        return list(self._entries.items())

    def keys(self) -> list[Path]:
        """Return all cached paths."""
        return list(self._entries.keys())

    def __contains__(self, path: Path) -> bool:
        """Check if a path is in the cache."""
        return path in self._entries

    def __len__(self) -> int:
        """Return number of cached entries."""
        return len(self._entries)
