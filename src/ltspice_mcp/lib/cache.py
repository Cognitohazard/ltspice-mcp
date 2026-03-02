"""Generic file cache with mtime-based invalidation."""

from pathlib import Path
from typing import Callable, Generic, TypeVar

T = TypeVar("T")


class FileCache(Generic[T]):
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
        # Get current modification time
        try:
            mtime = path.stat().st_mtime
        except OSError:
            # File doesn't exist or can't be accessed - call factory and don't cache
            return factory(path)

        # Check cache
        entry = self._entries.get(path)
        if entry is not None and entry[0] == mtime:
            # Cache hit - return cached value
            return entry[1]

        # Cache miss or stale - create new value
        value = factory(path)
        self._entries[path] = (mtime, value)
        return value

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
