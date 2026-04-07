"""Tests for ltspice_mcp.main entry point."""

import os
import sys
from unittest.mock import patch

from ltspice_mcp.main import main


class TestMain:
    def test_main_sets_env_var(self, tmp_path, monkeypatch):
        # Avoid actually running the server
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text("")

        monkeypatch.setattr(sys, "argv", ["ltspice-mcp", "--config", str(cfg_path)])
        with patch("asyncio.run") as mock_run, \
             patch("os.dup", return_value=99), \
             patch("os.open", return_value=98), \
             patch("os.dup2"), \
             patch("os.close"):
            main()
            assert os.environ.get("LTSPICE_MCP_CONFIG") == str(cfg_path)
            mock_run.assert_called_once()

    def test_main_no_config_arg(self, monkeypatch):
        monkeypatch.delenv("LTSPICE_MCP_CONFIG", raising=False)
        monkeypatch.setattr(sys, "argv", ["ltspice-mcp"])
        with patch("asyncio.run") as mock_run, \
             patch("os.dup", return_value=99), \
             patch("os.open", return_value=98), \
             patch("os.dup2"), \
             patch("os.close"):
            main()
            mock_run.assert_called_once()
