"""Unit tests for WSL detection and path conversion."""

from pathlib import Path
from unittest.mock import patch, mock_open

import pytest

from ltspice_mcp.lib.wsl import is_wsl, to_windows_path


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
