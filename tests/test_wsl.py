"""Unit tests for WSL detection and path conversion."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from ltspice_mcp.lib.wsl import (
    _resolve_win_env,
    find_windows_ltspice_exe,
    get_ltspice_lib_paths,
    get_windows_output_dir,
    is_windows_native_path,
    is_wsl,
    to_windows_path,
)


class TestIsWsl:
    """Tests for WSL detection."""

    def setup_method(self):
        """Reset cached WSL detection between tests."""
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = None

    def test_detects_via_env_var(self):
        with patch.dict("os.environ", {"WSL_DISTRO_NAME": "Ubuntu"}):
            assert is_wsl() is True

    def test_detects_via_proc_version(self):
        with patch.dict("os.environ", {}, clear=True):
            microsoft_version = "Linux version 5.15.0-microsoft-standard-WSL2"
            with patch("builtins.open", mock_open(read_data=microsoft_version)):
                import ltspice_mcp.lib.wsl as wsl_mod

                wsl_mod._is_wsl_cached = None
                assert is_wsl() is True

    def test_not_wsl_when_neither(self):
        with patch.dict("os.environ", {"WSL_DISTRO_NAME": ""}, clear=True):
            plain_linux = "Linux version 6.1.0-generic"
            with patch("builtins.open", mock_open(read_data=plain_linux)):
                import ltspice_mcp.lib.wsl as wsl_mod

                wsl_mod._is_wsl_cached = None
                assert is_wsl() is False

    def test_caches_result(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        assert is_wsl() is True
        wsl_mod._is_wsl_cached = False
        assert is_wsl() is False


class TestToWindowsPath:
    """Tests for WSL path conversion."""

    def test_passthrough_when_not_wsl(self):
        """When not in WSL, path should pass through unchanged."""
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = False

        path = Path("/tmp/test.cir")
        assert to_windows_path(path) == "/tmp/test.cir"

    def test_relative_path_passthrough(self):
        """Relative paths should pass through regardless of WSL status."""
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = False

        path = Path("circuit.net")
        assert to_windows_path(path) == "circuit.net"

    def test_wsl_path_conversion(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        fake_result = MagicMock(stdout="C:\\Users\\test\\file.cir\n", stderr="", returncode=0)
        with patch("subprocess.run", return_value=fake_result):
            result = to_windows_path(Path("/mnt/c/Users/test/file.cir"))
            assert "C:" in result

    def test_wslpath_not_found(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = to_windows_path(Path("/tmp/foo"))
            assert result == "/tmp/foo"

    def test_wslpath_failure(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        err = subprocess.CalledProcessError(1, "wslpath", stderr="bad path")
        with patch("subprocess.run", side_effect=err):
            result = to_windows_path(Path("/tmp/foo"))
            assert result == "/tmp/foo"


class TestResolveWinEnv:
    def test_failure_returns_none(self):
        with patch("subprocess.run", side_effect=Exception("boom")):
            assert _resolve_win_env("TEMP") is None


class TestGetWindowsOutputDir:
    def test_not_wsl(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = False
        wsl_mod._win_temp_dir = None
        assert get_windows_output_dir() is None

    def test_cached(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        cached = Path("/tmp/cached_dir")
        wsl_mod._win_temp_dir = cached
        assert get_windows_output_dir() == cached
        # Reset
        wsl_mod._win_temp_dir = None
        wsl_mod._is_wsl_cached = None


class TestIsWindowsNativePath:
    def test_mnt_path(self, tmp_path: Path):
        # tmp_path is not under /mnt
        assert is_windows_native_path(tmp_path) is False

    def test_oserror(self, monkeypatch):
        def boom(self):
            raise OSError("denied")

        monkeypatch.setattr(Path, "resolve", boom)
        assert is_windows_native_path(Path("/foo")) is False


class TestGetLtspiceLibPaths:
    def test_not_wsl(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = False
        assert get_ltspice_lib_paths() == []

    def test_wsl_no_localappdata(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        with patch("ltspice_mcp.lib.wsl._resolve_win_env", return_value=None):
            assert get_ltspice_lib_paths() == []
        wsl_mod._is_wsl_cached = None


class TestFindWindowsLtspiceExe:
    """WSL auto-detection of the Windows-side LTspice executable (Fix B)."""

    def test_not_wsl(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = False
        assert find_windows_ltspice_exe() is None
        wsl_mod._is_wsl_cached = None

    def test_localappdata_adi_hit(self, tmp_path: Path):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        exe = tmp_path / "Programs" / "ADI" / "LTspice" / "LTspice.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub")

        def fake_env(var: str):
            return tmp_path if var == "LOCALAPPDATA" else None

        with patch("ltspice_mcp.lib.wsl._resolve_win_env", side_effect=fake_env):
            assert find_windows_ltspice_exe() == exe
        wsl_mod._is_wsl_cached = None

    def test_program_files_legacy_hit(self, tmp_path: Path):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        exe = tmp_path / "LTC" / "LTspiceXVII" / "XVIIx64.exe"
        exe.parent.mkdir(parents=True)
        exe.write_text("stub")

        def fake_env(var: str):
            return tmp_path if var == "ProgramFiles" else None

        with patch("ltspice_mcp.lib.wsl._resolve_win_env", side_effect=fake_env):
            assert find_windows_ltspice_exe() == exe
        wsl_mod._is_wsl_cached = None

    def test_no_install_found(self, tmp_path: Path):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        # Bases resolve and exist, but no LTspice executable lives under them.
        with patch("ltspice_mcp.lib.wsl._resolve_win_env", return_value=tmp_path):
            assert find_windows_ltspice_exe() is None
        wsl_mod._is_wsl_cached = None

    def test_env_unresolvable(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        wsl_mod._is_wsl_cached = True
        with patch("ltspice_mcp.lib.wsl._resolve_win_env", return_value=None):
            assert find_windows_ltspice_exe() is None
        wsl_mod._is_wsl_cached = None
