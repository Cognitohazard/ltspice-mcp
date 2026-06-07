"""Tests for FileCache (mtime, size)-based invalidation."""

import os
from pathlib import Path

from ltspice_mcp.lib.cache import FileCache


class TestFileCache:
    def test_cache_hit_avoids_factory(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "a.txt"
        p.write_text("v1")

        call_count = 0

        def factory(path: Path) -> str:
            nonlocal call_count
            call_count += 1
            return path.read_text()

        r1 = cache.get(p, factory)
        r2 = cache.get(p, factory)
        assert r1 == r2 == "v1"
        assert call_count == 1

    def test_mtime_change_invalidates(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "b.txt"
        p.write_text("v1")

        cache.get(p, lambda path: path.read_text())

        # Force mtime change without sleeping (avoids flaky timing)
        p.write_text("v2")
        future = p.stat().st_mtime + 2.0
        os.utime(p, (future, future))

        result = cache.get(p, lambda path: path.read_text())
        assert result == "v2"

    def test_same_mtime_size_change_invalidates(self, tmp_path: Path):
        # An in-place rewrite can land within the same mtime tick (observed on
        # ext4/tmpfs and the /mnt/c Windows mount under WSL — exactly where a
        # re-run sim or the shared .ltspice-mcp store gets overwritten). Size
        # is part of the stamp, so a same-tick content change still invalidates.
        cache: FileCache[str] = FileCache()
        p = tmp_path / "same_tick.txt"
        p.write_text("v1")

        call_count = 0

        def factory(path: Path) -> str:
            nonlocal call_count
            call_count += 1
            return path.read_text()

        first = cache.get(p, factory)
        original_mtime_ns = p.stat().st_mtime_ns

        # Rewrite with longer, different content, then restore the EXACT mtime
        # (nanoseconds) so ONLY the size differs — isolating the size component.
        p.write_text("v2-different-and-longer")
        os.utime(p, ns=(original_mtime_ns, original_mtime_ns))
        assert p.stat().st_mtime_ns == original_mtime_ns

        second = cache.get(p, factory)
        assert first == "v1"
        assert second == "v2-different-and-longer"
        assert call_count == 2

    def test_nonexistent_file_bypasses_cache(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "missing.txt"

        call_count = 0

        def factory(_path: Path) -> str:
            nonlocal call_count
            call_count += 1
            return "default"

        cache.get(p, factory)
        cache.get(p, factory)
        assert call_count == 2

    def test_invalidate_forces_reload(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "c.txt"
        p.write_text("v1")

        call_count = 0

        def factory(path: Path) -> str:
            nonlocal call_count
            call_count += 1
            return path.read_text()

        cache.get(p, factory)
        assert call_count == 1

        cache.invalidate(p)
        cache.get(p, factory)
        assert call_count == 2

    def test_clear_empties_all(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        for name in ("x.txt", "y.txt"):
            p = tmp_path / name
            p.write_text(name)
            cache.get(p, lambda path: path.read_text())

        assert len(cache) == 2
        cache.clear()
        assert len(cache) == 0

    def test_contains_tracks_state(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "d.txt"
        p.write_text("data")

        assert p not in cache
        cache.get(p, lambda path: path.read_text())
        assert p in cache
        cache.invalidate(p)
        assert p not in cache

    def test_concurrent_paths_isolated(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p1 = tmp_path / "one.txt"
        p2 = tmp_path / "two.txt"
        p1.write_text("alpha")
        p2.write_text("beta")

        r1 = cache.get(p1, lambda path: path.read_text())
        r2 = cache.get(p2, lambda path: path.read_text())
        assert r1 == "alpha"
        assert r2 == "beta"

        cache.invalidate(p1)
        assert p1 not in cache
        assert p2 in cache
