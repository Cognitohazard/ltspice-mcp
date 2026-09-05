"""Tests for ltspice_mcp.main entry point."""

import os
import sys
from unittest.mock import patch

from ltspice_mcp.main import main


def _serve_never_started(mock_run) -> None:
    """Close the server coroutine the patched ``asyncio.run`` was handed.

    Nothing awaits it, and a coroutine collected un-awaited raises a
    RuntimeWarning from wherever the collector happens to run — which lands the
    complaint on an unrelated test.
    """
    mock_run.assert_called_once()
    mock_run.call_args.args[0].close()


class TestMain:
    def test_main_sets_env_var(self, tmp_path, monkeypatch):
        # Avoid actually running the server
        cfg_path = tmp_path / "test.toml"
        cfg_path.write_text("")

        monkeypatch.setattr(sys, "argv", ["ltspice-mcp", "--config", str(cfg_path)])
        with patch("asyncio.run") as mock_run:
            main()
            assert os.environ.get("LTSPICE_MCP_CONFIG") == str(cfg_path)
            _serve_never_started(mock_run)

    def test_main_no_config_arg(self, monkeypatch):
        monkeypatch.delenv("LTSPICE_MCP_CONFIG", raising=False)
        monkeypatch.setattr(sys, "argv", ["ltspice-mcp"])
        with patch("asyncio.run") as mock_run:
            main()
            _serve_never_started(mock_run)
