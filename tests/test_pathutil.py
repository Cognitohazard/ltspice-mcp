"""Unit tests for path security sandbox."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.pathutil import relative_paths_from, resolve_safe_path
from tests.conftest import symlink_or_skip


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
        # A NUL byte makes path.resolve raise ValueError on most platforms, and
        # return a path holding the NUL on Windows with Python 3.13; either way
        # it must surface as a PathSecurityError, not pass the sandbox check.
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
        symlink_or_skip(link, secret)

        with pytest.raises(PathSecurityError, match="outside allowed"):
            resolve_safe_path("sneaky_link", [sandbox])


class TestDeclaredRelativeBase:
    """A host may declare where relative paths are taken from; unset, nothing
    about the old resolution changes."""

    def test_unset_base_still_uses_the_first_allowed_dir(self, tmp_path: Path, monkeypatch):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        sandbox = tmp_path / "sandbox"
        sandbox.mkdir()
        monkeypatch.chdir(elsewhere)
        assert resolve_safe_path("file.cir", [sandbox]) == sandbox / "file.cir"

    def test_unset_base_resolves_a_relative_sandbox_against_the_cwd(
        self, tmp_path: Path, monkeypatch
    ):
        here = tmp_path / "here"
        here.mkdir()
        monkeypatch.chdir(here)
        assert resolve_safe_path("file.cir", [Path(".")]) == here / "file.cir"

    def test_declared_base_anchors_the_user_path(self, tmp_path: Path, monkeypatch):
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        base = tmp_path / "designs"
        base.mkdir()
        monkeypatch.chdir(elsewhere)
        with relative_paths_from(base):
            assert resolve_safe_path("file.cir", [base]) == base / "file.cir"

    def test_declared_base_also_anchors_a_relative_sandbox_root(self, tmp_path: Path, monkeypatch):
        # The generated TOML ships allowed_paths = ["."]. Left pinned to the
        # process cwd, rebasing the user path onto the working dir would put it
        # outside the sandbox and turn the fix into a security refusal.
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        base = tmp_path / "designs"
        base.mkdir()
        monkeypatch.chdir(elsewhere)
        with relative_paths_from(base):
            assert resolve_safe_path("file.cir", [Path(".")]) == base / "file.cir"

    def test_absolute_paths_are_untouched_by_a_base(self, tmp_path: Path):
        base = tmp_path / "designs"
        base.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        target = other / "file.cir"
        with relative_paths_from(base):
            assert resolve_safe_path(str(target), [base, other]) == target

    def test_the_base_is_restored_on_exit(self, tmp_path: Path, monkeypatch):
        here = tmp_path / "here"
        here.mkdir()
        base = tmp_path / "designs"
        base.mkdir()
        monkeypatch.chdir(here)
        with relative_paths_from(base):
            pass
        assert resolve_safe_path("file.cir", [Path(".")]) == here / "file.cir"
