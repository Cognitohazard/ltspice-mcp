"""Unit tests for path security sandbox."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.pathutil import resolve_safe_path


class TestResolveSafePath:
    """Tests for resolve_safe_path()."""

    def test_relative_path_within_sandbox(self, work_dir: Path):
        result = resolve_safe_path("file.cir", [work_dir])
        assert result == work_dir / "file.cir"

    def test_relative_nested_path(self, work_dir: Path):
        result = resolve_safe_path("subdir/file.cir", [work_dir])
        assert result == work_dir / "subdir" / "file.cir"

    def test_absolute_path_within_sandbox(self, work_dir: Path):
        abs_path = str(work_dir / "file.cir")
        result = resolve_safe_path(abs_path, [work_dir])
        assert result == work_dir / "file.cir"

    def test_traversal_rejected(self, work_dir: Path):
        with pytest.raises(PathSecurityError, match="traversal"):
            resolve_safe_path("../../etc/passwd", [work_dir])

    def test_dotdot_in_middle_rejected(self, work_dir: Path):
        with pytest.raises(PathSecurityError, match="traversal"):
            resolve_safe_path("subdir/../../../etc/passwd", [work_dir])

    def test_absolute_path_outside_sandbox(self, work_dir: Path):
        with pytest.raises(PathSecurityError, match="outside allowed"):
            resolve_safe_path("/etc/passwd", [work_dir])

    def test_empty_allowed_dirs(self):
        with pytest.raises(PathSecurityError, match="No allowed directories"):
            resolve_safe_path("file.cir", [])

    def test_embedded_nul_byte_rejected(self, work_dir: Path):
        # A NUL byte makes path.resolve raise ValueError (not OSError); it must
        # surface as a PathSecurityError, not escape the security boundary.
        with pytest.raises(PathSecurityError, match="Failed to resolve"):
            resolve_safe_path("file\x00.cir", [work_dir])

    def test_multiple_allowed_dirs(self, work_dir: Path, tmp_path: Path):
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        abs_path = str(other_dir / "file.cir")
        result = resolve_safe_path(abs_path, [work_dir, other_dir])
        assert result == other_dir / "file.cir"

    def test_symlink_escape_blocked(self, tmp_path: Path):
        """Symlink pointing outside sandbox should be blocked."""
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        secret = outside / "secret.txt"
        secret.write_text("secret")

        link = sandbox / "sneaky_link"
        link.symlink_to(secret)

        with pytest.raises(PathSecurityError, match="outside allowed"):
            resolve_safe_path("sneaky_link", [sandbox])
