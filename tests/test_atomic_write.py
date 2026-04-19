"""Tests for atomic write helpers in ``ltspice_mcp.lib``."""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from ltspice_mcp.lib import (
    atomic_write,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
)

# ---------------------------------------------------------------------------
# atomic_write_text — user-facing convenience wrapper
# ---------------------------------------------------------------------------


class TestAtomicWriteText:
    def test_writes_content(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        atomic_write_text(path, "hello\n")
        assert path.read_text(encoding="utf-8") == "hello\n"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        path.write_text("old")
        atomic_write_text(path, "new")
        assert path.read_text(encoding="utf-8") == "new"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "deep" / "out.txt"
        atomic_write_text(path, "data")
        assert path.read_text(encoding="utf-8") == "data"

    def test_cleans_up_tempfile_on_rename_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        with (
            patch("ltspice_mcp.lib.os.replace", side_effect=OSError("boom")),
            pytest.raises(OSError, match="boom"),
        ):
            atomic_write_text(path, "data")
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_no_partial_file_on_crash_mid_write(self, tmp_path: Path) -> None:
        """A crash during write must leave the destination untouched."""
        path = tmp_path / "out.txt"
        path.write_text("original")

        real_fdopen = os.fdopen

        def boom_fdopen(fd, *args, **kwargs):
            handle = real_fdopen(fd, *args, **kwargs)
            handle.close()  # drop the fd so the write never completes
            raise OSError("disk full")

        with (
            patch("ltspice_mcp.lib.os.fdopen", side_effect=boom_fdopen),
            pytest.raises(OSError, match="disk full"),
        ):
            atomic_write_text(path, "replacement")

        assert path.read_text(encoding="utf-8") == "original"
        assert list(tmp_path.iterdir()) == [path]


# ---------------------------------------------------------------------------
# atomic_write_json — thin JSON wrapper
# ---------------------------------------------------------------------------


class TestAtomicWriteJson:
    def test_writes_json(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        atomic_write_json(path, {"a": 1, "b": [2, 3]})
        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": [2, 3]}

    def test_uses_default_serializer(self, tmp_path: Path) -> None:
        path = tmp_path / "out.json"
        atomic_write_json(path, {"x": {1, 2}}, default=list)
        assert json.loads(path.read_text()) == {"x": [1, 2]}


# ---------------------------------------------------------------------------
# atomic_write — primary context-manager primitive
# ---------------------------------------------------------------------------


class TestAtomicWriteContextManager:
    def test_yields_writable_handle(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        with atomic_write(path) as f:
            f.write("hello ")
            f.write("world")
        assert path.read_text(encoding="utf-8") == "hello world"

    def test_streaming_write_without_materialization(self, tmp_path: Path) -> None:
        """Caller can write in chunks — no need to build the full string first."""
        path = tmp_path / "big.txt"
        with atomic_write(path) as f:
            for i in range(1000):
                f.write(f"line {i}\n")
        lines = path.read_text().splitlines()
        assert len(lines) == 1000
        assert lines[0] == "line 0"
        assert lines[-1] == "line 999"

    def test_exception_in_block_cleans_up(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        path.write_text("original")

        def write_then_fail() -> None:
            with atomic_write(path) as f:
                f.write("partial")
                raise RuntimeError("caller error")

        with pytest.raises(RuntimeError, match="caller error"):
            write_then_fail()
        assert path.read_text() == "original"
        assert list(tmp_path.iterdir()) == [path]

    def test_keyboard_interrupt_cleans_up(self, tmp_path: Path) -> None:
        """KeyboardInterrupt must unlink the tempfile (BaseException handling)."""
        path = tmp_path / "out.txt"

        def write_then_interrupt() -> None:
            with atomic_write(path) as f:
                f.write("partial")
                raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            write_then_interrupt()
        assert not path.exists()
        assert list(tmp_path.iterdir()) == []

    def test_encoding_honored(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        with atomic_write(path, encoding="latin-1") as f:
            f.write("café")
        assert path.read_bytes() == "café".encode("latin-1")


# ---------------------------------------------------------------------------
# Durability — fsync must be called in the right order when durable=True
# ---------------------------------------------------------------------------


class TestDurability:
    def test_fsync_called_on_file_by_default(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        real_fsync = os.fsync
        fsync_calls: list[int] = []

        def tracking_fsync(fd: int) -> None:
            fsync_calls.append(fd)
            real_fsync(fd)

        with patch("ltspice_mcp.lib.os.fsync", side_effect=tracking_fsync):
            atomic_write_text(path, "data")

        # File fd fsync plus (on POSIX) parent-dir fsync.
        expected = 1 if sys.platform == "win32" else 2
        assert len(fsync_calls) == expected

    def test_fsync_skipped_when_durable_false(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        with patch("ltspice_mcp.lib.os.fsync") as mock_fsync:
            atomic_write_text(path, "data", durable=False)
        mock_fsync.assert_not_called()
        assert path.read_text() == "data"

    def test_fsync_file_before_rename(self, tmp_path: Path) -> None:
        """File must be fsync'd before the rename, not after — otherwise
        power loss between rename and fsync leaves a committed-but-empty file.
        """
        path = tmp_path / "out.txt"
        events: list[str] = []
        real_fsync = os.fsync
        real_replace = os.replace

        def tracking_fsync(fd: int) -> None:
            events.append("fsync")
            real_fsync(fd)

        def tracking_replace(src, dst):
            events.append("replace")
            real_replace(src, dst)

        with (
            patch("ltspice_mcp.lib.os.fsync", side_effect=tracking_fsync),
            patch("ltspice_mcp.lib.os.replace", side_effect=tracking_replace),
        ):
            atomic_write_text(path, "data")

        # Expected order on POSIX: fsync(file) → replace → fsync(dir).
        # On Windows: fsync(file) → replace (no dir fsync).
        assert events[0] == "fsync"
        assert events[1] == "replace"
        if sys.platform != "win32":
            assert events[2] == "fsync"

    def test_file_fsync_failure_aborts_and_cleans(self, tmp_path: Path) -> None:
        """If file fsync fails, the rename must not happen."""
        path = tmp_path / "out.txt"
        path.write_text("original")

        with (
            patch("ltspice_mcp.lib.os.fsync", side_effect=OSError("io error")),
            pytest.raises(OSError, match="io error"),
        ):
            atomic_write_text(path, "replacement")

        assert path.read_text() == "original"
        assert list(tmp_path.iterdir()) == [path]

    @pytest.mark.skipif(sys.platform == "win32", reason="no dir fsync on Windows")
    def test_dir_fsync_failure_swallowed(self, tmp_path: Path) -> None:
        """Dir fsync happens AFTER rename — data is already on disk, so a
        dir-fsync failure only affects rename durability, not correctness.
        Swallow it rather than surfacing a confusing error to the caller.
        """
        path = tmp_path / "out.txt"
        real_fsync = os.fsync

        def selective_fsync(fd: int) -> None:
            # Fail only on the dir fsync. Distinguish via fstat: file fsync
            # targets a regular file, dir fsync targets a directory.
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise OSError("dir sync failed")
            real_fsync(fd)

        with patch("ltspice_mcp.lib.os.fsync", side_effect=selective_fsync):
            atomic_write_text(path, "data")  # must not raise

        assert path.read_text() == "data"

    def test_dir_fsync_skipped_on_windows(self, tmp_path: Path) -> None:
        """When sys.platform is 'win32', the dir fsync path is skipped."""
        path = tmp_path / "out.txt"
        real_fsync = os.fsync
        fsync_count = 0

        def counting_fsync(fd: int) -> None:
            nonlocal fsync_count
            fsync_count += 1
            real_fsync(fd)

        with (
            patch("ltspice_mcp.lib.sys.platform", "win32"),
            patch("ltspice_mcp.lib.os.fsync", side_effect=counting_fsync),
        ):
            atomic_write_text(path, "data")

        # Only the file fsync ran; dir fsync is a no-op on Windows.
        assert fsync_count == 1
        assert path.read_text() == "data"


# ---------------------------------------------------------------------------
# Concurrency / atomicity — observable behavior
# ---------------------------------------------------------------------------


class TestAtomicityObservable:
    def test_rename_is_the_commit_point(self, tmp_path: Path) -> None:
        """Before os.replace runs, the destination still has old contents
        (or does not exist). The tempfile is a separate path.
        """
        path = tmp_path / "out.txt"
        path.write_text("old")

        observed: dict[str, str] = {}
        real_replace = os.replace

        def observe_before_replace(src, dst):
            observed["before"] = path.read_text()
            real_replace(src, dst)
            observed["after"] = path.read_text()

        with patch("ltspice_mcp.lib.os.replace", side_effect=observe_before_replace):
            atomic_write_text(path, "new")

        assert observed == {"before": "old", "after": "new"}


# ---------------------------------------------------------------------------
# Binary mode
# ---------------------------------------------------------------------------


class TestBinaryMode:
    def test_context_manager_binary(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bin"
        with atomic_write(path, mode="wb") as f:
            f.write(b"\x00\x01\x02\xff")
        assert path.read_bytes() == b"\x00\x01\x02\xff"

    def test_atomic_write_bytes_helper(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bin"
        atomic_write_bytes(path, b"binary \x00 data")
        assert path.read_bytes() == b"binary \x00 data"

    def test_binary_mode_ignores_encoding(self, tmp_path: Path) -> None:
        """In binary mode the encoding kwarg must not be passed to fdopen."""
        path = tmp_path / "out.bin"
        with atomic_write(path, mode="wb", encoding="utf-8") as f:
            f.write(b"\xc3\xa9")  # utf-8 encoding of "é"
        assert path.read_bytes() == b"\xc3\xa9"


# ---------------------------------------------------------------------------
# Permission preservation on overwrite
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission model")
class TestPermissionPreservation:
    def test_overwrite_preserves_mode(self, tmp_path: Path) -> None:
        path = tmp_path / "out.txt"
        path.write_text("old")
        os.chmod(path, 0o644)

        atomic_write_text(path, "new")

        assert stat.S_IMODE(path.stat().st_mode) == 0o644

    def test_fresh_file_has_tempfile_default_mode(self, tmp_path: Path) -> None:
        """New files inherit mkstemp's 0600 — documented behavior."""
        path = tmp_path / "new.txt"
        atomic_write_text(path, "data")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    def test_chmod_failure_does_not_abort_write(self, tmp_path: Path) -> None:
        """A chmod failure is logged-and-swallowed — the write still commits."""
        path = tmp_path / "out.txt"
        path.write_text("old")
        os.chmod(path, 0o644)

        with patch("ltspice_mcp.lib.os.chmod", side_effect=OSError("no chmod")):
            atomic_write_text(path, "new")

        assert path.read_text() == "new"


# ---------------------------------------------------------------------------
# overwrite=False — exclusive-create semantics
# ---------------------------------------------------------------------------


class TestExclusiveCreate:
    def test_succeeds_when_dst_absent(self, tmp_path: Path) -> None:
        path = tmp_path / "new.txt"
        atomic_write_text(path, "data", overwrite=False)
        assert path.read_text() == "data"

    def test_raises_when_dst_exists(self, tmp_path: Path) -> None:
        path = tmp_path / "existing.txt"
        path.write_text("original")

        with pytest.raises(FileExistsError):
            atomic_write_text(path, "new", overwrite=False)

        # Destination untouched; no leftover tempfile.
        assert path.read_text() == "original"
        assert list(tmp_path.iterdir()) == [path]

    def test_raises_when_dst_exists_binary(self, tmp_path: Path) -> None:
        path = tmp_path / "existing.bin"
        path.write_bytes(b"original")

        with pytest.raises(FileExistsError):
            atomic_write_bytes(path, b"new", overwrite=False)

        assert path.read_bytes() == b"original"
        assert list(tmp_path.iterdir()) == [path]

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX os.link path")
    def test_posix_uses_os_link(self, tmp_path: Path) -> None:
        """On POSIX, overwrite=False uses os.link (race-free) not os.rename."""
        path = tmp_path / "new.txt"
        with (
            patch("ltspice_mcp.lib.os.link", wraps=os.link) as mock_link,
            patch("ltspice_mcp.lib.os.rename") as mock_rename,
        ):
            atomic_write_text(path, "data", overwrite=False)
        mock_link.assert_called_once()
        mock_rename.assert_not_called()


# ---------------------------------------------------------------------------
# macOS F_FULLFSYNC — platter-level flush on Darwin
# ---------------------------------------------------------------------------


class TestFullFsyncMacOS:
    def test_uses_f_fullfsync_when_available(self, tmp_path: Path) -> None:
        """When _F_FULLFSYNC is set, _fsync_fd routes through fcntl and
        does NOT call os.fsync on the file."""
        from ltspice_mcp import lib as _lib

        path = tmp_path / "out.txt"
        fsync_fds: list[int] = []
        fcntl_calls: list[tuple[int, int]] = []
        real_fsync = os.fsync

        def tracking_fsync(fd: int) -> None:
            fsync_fds.append(fd)
            real_fsync(fd)

        def fake_fcntl(fd: int, cmd: int) -> int:
            fcntl_calls.append((fd, cmd))
            return 0

        fake_fcntl_mod = type("M", (), {"fcntl": fake_fcntl})

        with (
            patch.object(_lib, "_F_FULLFSYNC", 51),
            patch.object(_lib, "_fcntl", fake_fcntl_mod),
            patch("ltspice_mcp.lib.os.fsync", side_effect=tracking_fsync),
        ):
            atomic_write_text(path, "data")

        # File-level flush went through F_FULLFSYNC, not fsync.
        assert len(fcntl_calls) == 1
        assert fcntl_calls[0][1] == 51
        # Dir fsync (POSIX) still uses os.fsync on the dir fd.
        expected_fsync_count = 0 if sys.platform == "win32" else 1
        assert len(fsync_fds) == expected_fsync_count

    def test_falls_back_to_fsync_on_f_fullfsync_error(self, tmp_path: Path) -> None:
        """If F_FULLFSYNC raises (e.g., filesystem doesn't support it),
        _fsync_fd falls back to plain fsync."""
        from ltspice_mcp import lib as _lib

        path = tmp_path / "out.txt"
        real_fsync = os.fsync
        fsync_fds: list[int] = []

        def tracking_fsync(fd: int) -> None:
            fsync_fds.append(fd)
            real_fsync(fd)

        def failing_fcntl(fd: int, cmd: int) -> int:
            raise OSError("unsupported on this FS")

        fake_fcntl_mod = type("M", (), {"fcntl": failing_fcntl})

        with (
            patch.object(_lib, "_F_FULLFSYNC", 51),
            patch.object(_lib, "_fcntl", fake_fcntl_mod),
            patch("ltspice_mcp.lib.os.fsync", side_effect=tracking_fsync),
        ):
            atomic_write_text(path, "data")

        # File fsync fallback + (on POSIX) dir fsync.
        expected = 1 if sys.platform == "win32" else 2
        assert len(fsync_fds) == expected
        assert path.read_text() == "data"

    def test_non_darwin_does_not_use_f_fullfsync(self) -> None:
        """On non-macOS platforms, the module-level _F_FULLFSYNC is None."""
        from ltspice_mcp import lib as _lib

        if sys.platform != "darwin":
            assert _lib._F_FULLFSYNC is None
            assert _lib._fcntl is None
