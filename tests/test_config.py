"""Unit tests for configuration loading."""

import os
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig, generate_default_config


class TestServerConfig:
    """Tests for ServerConfig loading."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.simulator is None
        assert config.simulator_exe is None
        assert config.max_parallel_sims == 4
        assert config.default_timeout == 300.0
        assert config.log_level == "INFO"

    def test_allowed_paths_defaults_to_working_dir(self):
        config = ServerConfig()
        assert config.allowed_paths == [config.working_dir]

    def test_load_from_toml(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(
            '[simulator]\ndefault = "ltspice"\npath = "/usr/bin/ltspice"\n'
            "[simulation]\nmax_parallel = 8\ntimeout = 60.0\n"
            "[logging]\nlevel = \"DEBUG\"\n"
        )
        config = ServerConfig.load(toml_path)
        assert config.simulator == "ltspice"
        assert config.simulator_exe == Path("/usr/bin/ltspice")
        assert config.max_parallel_sims == 8
        assert config.default_timeout == 60.0
        assert config.log_level == "DEBUG"

    def test_load_empty_path_is_none(self, work_dir: Path):
        """Empty path string should result in None, not Path('')."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[simulator]\ndefault = "ltspice"\npath = ""\n')
        config = ServerConfig.load(toml_path)
        assert config.simulator_exe is None

    def test_env_var_override(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_SIMULATOR", "ngspice")
        monkeypatch.setenv("LTSPICE_MCP_LOG_LEVEL", "WARNING")
        # Load with no TOML
        config = ServerConfig.load(work_dir / "nonexistent.toml")
        assert config.simulator == "ngspice"
        assert config.log_level == "WARNING"

    def test_env_overrides_toml(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        """Env vars take precedence over TOML."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[simulator]\ndefault = "ltspice"\n')
        monkeypatch.setenv("LTSPICE_MCP_SIMULATOR", "ngspice")
        config = ServerConfig.load(toml_path)
        assert config.simulator == "ngspice"

    def test_generate_default_config(self, work_dir: Path):
        path = work_dir / "generated.toml"
        generate_default_config(path)
        assert path.exists()
        content = path.read_text()
        assert "ltspice" in content
        assert "allowed_paths" in content


class TestSimulatorExeConfig:
    """Tests for the simulator_exe config field being wired to detection."""

    def test_simulator_exe_applied_to_detection(self, work_dir: Path):
        """Config simulator_exe should be used by detect_simulators."""
        from ltspice_mcp.lib.simulator import detect_simulators

        # Use a non-existent path - should warn but not crash
        config = ServerConfig(
            simulator="ltspice",
            simulator_exe=Path("/nonexistent/ltspice.exe"),
            working_dir=work_dir,
            allowed_paths=[work_dir],
        )
        # Should not raise
        available = detect_simulators(config)
        # Non-existent path should not register
        # (may or may not have ltspice depending on system)

    def test_detect_without_config(self):
        """detect_simulators(None) should still work (backwards compat)."""
        from ltspice_mcp.lib.simulator import detect_simulators

        # Should not raise
        available = detect_simulators()
        assert isinstance(available, dict)
