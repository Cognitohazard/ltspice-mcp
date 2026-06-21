"""Tests for FileCache (mtime, size)-based invalidation."""

import os
import threading
import time
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

    def test_concurrent_threads_get_consistent_values(self, tmp_path: Path):
        """Handlers offload heavy cache factories to worker threads via
        asyncio.to_thread, so get/invalidate/items can interleave across
        threads. Every get() must return a fully-constructed value and no
        compound dict operation may raise (e.g. dict-changed-during-iteration
        from items() racing an insert)."""
        cache: FileCache[tuple[str, int]] = FileCache()
        p = tmp_path / "data.raw"
        p.write_text("payload")

        def factory(path: Path) -> tuple[str, int]:
            text = path.read_text()
            return (text, len(text))

        errors: list[BaseException] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    value = cache.get(p, factory)
                    if value != ("payload", 7):
                        errors.append(AssertionError(f"torn value: {value!r}"))
                        return
            except BaseException as e:  # surface any thread crash, including asserts
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        try:
            for _ in range(200):
                cache.invalidate(p)
                cache.items()
                len(cache)
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)
        assert not any(t.is_alive() for t in threads)
        assert not errors

    def test_single_flight_concurrent_cold_readers_parse_once(self, tmp_path: Path):
        """Two threads racing a cold cache on the SAME path must run the
        factory exactly once: the per-path lock serialises them, and the
        follower's re-check finds the leader's stored value."""
        cache: FileCache[str] = FileCache()
        p = tmp_path / "big.raw"
        p.write_text("payload")

        call_count = 0
        in_factory = threading.Event()
        release = threading.Event()

        def factory(path: Path) -> str:
            nonlocal call_count
            call_count += 1
            in_factory.set()
            assert release.wait(timeout=5)  # hold the leader mid-parse
            return path.read_text()

        results: list[str] = []

        def reader() -> None:
            results.append(cache.get(p, factory))

        leader = threading.Thread(target=reader)
        follower = threading.Thread(target=reader)
        leader.start()
        assert in_factory.wait(timeout=5)  # leader is inside the factory
        follower.start()
        time.sleep(0.05)  # let the follower reach (and block on) the per-path lock
        release.set()
        leader.join(timeout=5)
        follower.join(timeout=5)

        assert results == ["payload", "payload"]
        assert call_count == 1

    def test_peek_hits_only_when_cached_and_fresh(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "peek.txt"
        p.write_text("v1")

        assert cache.peek(p) is None  # cold cache

        cache.get(p, lambda path: path.read_text())
        assert cache.peek(p) == "v1"  # fresh hit, no factory involved

    def test_peek_misses_on_stale_or_missing_file(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        p = tmp_path / "stale.txt"
        p.write_text("v1")
        cache.get(p, lambda path: path.read_text())

        # Stale: content rewritten with a forced mtime bump.
        p.write_text("v2-longer")
        future = p.stat().st_mtime + 2.0
        os.utime(p, (future, future))
        assert cache.peek(p) is None

        # Missing: stat failure also reads as a miss.
        p.unlink()
        assert cache.peek(p) is None

    def test_unbounded_by_default_keeps_all(self, tmp_path: Path):
        cache: FileCache[str] = FileCache()
        for i in range(50):
            p = tmp_path / f"f{i}"
            p.write_text(str(i))
            cache.get(p, lambda path: path.read_text())
        assert len(cache) == 50

    def test_maxsize_evicts_least_recently_used(self, tmp_path: Path):
        cache: FileCache[str] = FileCache(maxsize=2)
        calls: dict[str, int] = {}

        def factory(p: Path) -> str:
            calls[p.name] = calls.get(p.name, 0) + 1
            return p.read_text()

        a, b, c = (tmp_path / n for n in ("a", "b", "c"))
        for f in (a, b, c):
            f.write_text(f.name)

        cache.get(a, factory)  # [a]
        cache.get(b, factory)  # [a, b]
        cache.get(c, factory)  # over cap -> evict a -> [b, c]
        assert len(cache) == 2
        assert a not in cache and b in cache and c in cache

        cache.get(a, factory)  # a was evicted -> factory runs again
        assert calls["a"] == 2
        assert calls["b"] == 1 and calls["c"] == 1

    def test_get_hit_refreshes_recency(self, tmp_path: Path):
        cache: FileCache[str] = FileCache(maxsize=2)
        a, b, c = (tmp_path / n for n in ("a", "b", "c"))
        for f in (a, b, c):
            f.write_text(f.name)

        cache.get(a, lambda p: p.read_text())  # [a]
        cache.get(b, lambda p: p.read_text())  # [a, b]
        cache.get(a, lambda p: p.read_text())  # touch a -> b is now LRU
        cache.get(c, lambda p: p.read_text())  # evict b -> [a, c]
        assert a in cache and c in cache and b not in cache
