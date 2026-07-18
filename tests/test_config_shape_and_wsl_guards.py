"""Config-shape validation and WSL interop guards.

Covers three robustness fixes:
- TOML list-of-strings fields (``allowed_paths``, ``symbol_paths``) reject a
  bare scalar string instead of iterating it character-wise.
- ``_resolve_win_env`` retries after a transient failure (only successes are
  cached).
- ``kill_windows_ltspice_by_token`` anchors the job-id token at a filename
  boundary, mirroring the Linux ``proc_kill`` twin.
"""

import logging
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.wsl import _resolve_win_env, kill_windows_ltspice_by_token


class TestAllowedPathsShape:
    """[security] allowed_paths must be a list of strings, not a scalar."""

    def test_scalar_string_rejected_and_defaults(
        self, work_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        # The common TOML mistake: a bare string instead of a list. Iterating it
        # would yield one Path per character (including Path("/") per slash),
        # silently widening the sandbox to the whole filesystem.
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[security]\nallowed_paths = "/home/me/circuits"\n')
        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.config"):
            config = ServerConfig.load(toml_path)
        # Falls back to the working-dir default, not a per-character explosion.
        assert config.allowed_paths == [config.working_dir]
        assert Path("/") not in config.allowed_paths
        assert "allowed_paths" in caplog.text
        assert "list of strings" in caplog.text

    def test_list_of_strings_accepted(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[security]\nallowed_paths = ["/opt/a", "/opt/b"]\n')
        config = ServerConfig.load(toml_path)
        assert config.allowed_paths == [Path("/opt/a"), Path("/opt/b")]


class TestSymbolPathsShape:
    """[schematic] symbol_paths shares the allowed_paths guard."""

    def test_scalar_string_rejected(self, work_dir: Path, caplog: pytest.LogCaptureFixture):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[schematic]\nsymbol_paths = "/lib/sym"\n')
        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.config"):
            config = ServerConfig.load(toml_path)
        # Kept at its empty default rather than one Path per character.
        assert config.symbol_paths == []
        assert "symbol_paths" in caplog.text
        assert "list of strings" in caplog.text

    def test_list_of_strings_accepted(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[schematic]\nsymbol_paths = ["/lib/sym", "/lib/sym2"]\n')
        config = ServerConfig.load(toml_path)
        assert config.symbol_paths == [Path("/lib/sym"), Path("/lib/sym2")]


class TestResolveWinEnvRetries:
    """A transient interop failure must not be memoized as permanent."""

    def test_transient_failure_is_not_memoized(self):
        import ltspice_mcp.lib.wsl as wsl_mod

        var = "SHAPE_GUARD_RETRY_TEST"
        wsl_mod._win_env_cache.pop(var, None)

        cmd_calls = {"n": 0}

        def fake_run(cmd, **kwargs):
            # The cmd.exe echo times out the first time, then succeeds; wslpath
            # always succeeds. First _resolve_win_env call fails, second retries.
            if cmd[0] == "cmd.exe":
                cmd_calls["n"] += 1
                if cmd_calls["n"] == 1:
                    raise subprocess.TimeoutExpired("cmd.exe", 15)
                return MagicMock(stdout="C:\\Users\\me\\AppData\\Local\n", stderr="", returncode=0)
            return MagicMock(stdout="/mnt/c/Users/me/AppData/Local\n", stderr="", returncode=0)

        with patch("ltspice_mcp.lib.wsl.subprocess.run", side_effect=fake_run):
            assert _resolve_win_env(var) is None
            assert var not in wsl_mod._win_env_cache
            second = _resolve_win_env(var)

        assert second == Path("/mnt/c/Users/me/AppData/Local")
        wsl_mod._win_env_cache.pop(var, None)


class TestKillTokenFilterAnchored:
    """The PowerShell command-line match anchors the job-id token."""

    def test_powershell_filter_is_boundary_anchored(self):
        token = "sim_1234567890_abcdef01"
        captured: dict[str, str] = {}

        def fake_run(cmd, **kwargs):
            if cmd[0].endswith("powershell.exe"):
                captured["ps"] = cmd[-1]
                return MagicMock(stdout="", stderr="", returncode=0)
            return MagicMock(stdout="SUCCESS", stderr="", returncode=0)

        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=True),
            patch("ltspice_mcp.lib.wsl.subprocess.run", side_effect=fake_run),
        ):
            kill_windows_ltspice_by_token(token)

        ps = captured["ps"]
        # Staged decks are ``{token}.{ext}`` or ``{token}_{n}.{ext}``, so the id
        # is matched only when followed by '.' or '_'.
        assert f"*{token}.*" in ps
        assert f"*{token}_*" in ps
        # The old unanchored bare-substring form must be gone.
        assert f"'*{token}*'" not in ps
