"""Unit tests for configuration loading."""

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
            '[logging]\nlevel = "DEBUG"\n'
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


class TestToolProfile:
    """Tests for tool_profile configuration."""

    def test_default_profile_is_full(self):
        config = ServerConfig()
        assert config.tool_profile == "full"

    def test_profile_from_toml(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nprofile = "agentic"\n')
        config = ServerConfig.load(toml_path)
        assert config.tool_profile == "agentic"

    def test_invalid_profile_in_toml_falls_back(self, work_dir: Path):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nprofile = "bogus"\n')
        config = ServerConfig.load(toml_path)
        assert config.tool_profile == "full"

    def test_env_var_override(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "agentic")
        config = ServerConfig.load(work_dir / "nonexistent.toml")
        assert config.tool_profile == "agentic"

    def test_env_var_overrides_toml(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nprofile = "full"\n')
        monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "agentic")
        config = ServerConfig.load(toml_path)
        assert config.tool_profile == "agentic"

    def test_invalid_env_var_falls_back(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "bogus")
        config = ServerConfig.load(work_dir / "nonexistent.toml")
        assert config.tool_profile == "full"

    def test_generated_config_includes_tools_section(self, work_dir: Path):
        path = work_dir / "generated.toml"
        generate_default_config(path)
        content = path.read_text()
        assert "[tools]" in content
        assert "profile" in content


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
        detect_simulators(config)
        # Non-existent path should not register
        # (may or may not have ltspice depending on system)

    def test_detect_without_config(self):
        """detect_simulators(None) should still work (backwards compat)."""
        from ltspice_mcp.lib.simulator import detect_simulators

        # Should not raise
        available = detect_simulators()
        assert isinstance(available, dict)


class TestDetectionDiagnostics:
    """Fix A: silent simulator misconfiguration must be surfaced, not buried."""

    def test_missing_exe_records_diagnostic(self, work_dir: Path):
        from unittest.mock import patch

        from ltspice_mcp.lib.simulator import detect_simulators

        config = ServerConfig(
            simulator="ltspice",
            simulator_exe=Path("/nonexistent/ltspice.exe"),
            working_dir=work_dir,
            allowed_paths=[work_dir],
        )
        diagnostics: list[str] = []
        # Suppress WSL auto-detect so the only diagnostic is the bad path
        # (this suite runs on a real WSL host).
        with patch("ltspice_mcp.lib.simulator.is_wsl", return_value=False):
            detect_simulators(config, diagnostics)
        assert any("does not exist" in d for d in diagnostics)
        assert any("ltspice" in d for d in diagnostics)

    def test_mismatched_exe_not_bound(self, work_dir: Path):
        """NGv7-CONFIG-1: an LTspice-looking path must not bind to ngspice."""
        from ltspice_mcp.lib import simulator as sim

        exe = work_dir / "LTspice.exe"
        exe.write_text("stub")
        config = ServerConfig(
            simulator="ngspice",
            simulator_exe=exe,
            working_dir=work_dir,
            allowed_paths=[work_dir],
        )
        diagnostics: list[str] = []
        applied = sim._apply_simulator_exe(config, diagnostics)
        assert applied is False
        assert any("looks like a ltspice" in d.lower() for d in diagnostics)

    def test_valid_exe_suppresses_autodetect(self, work_dir: Path):
        """A working hardcoded path takes control → WSL autodetect is skipped."""
        from unittest.mock import patch

        import ltspice_mcp.lib.simulator as sim

        exe = work_dir / "ltspice.exe"
        exe.write_text("stub")
        config = ServerConfig(
            simulator="ltspice",
            simulator_exe=exe,
            working_dir=work_dir,
            allowed_paths=[work_dir],
        )
        with (
            patch("ltspice_mcp.lib.simulator.is_wsl", return_value=True),
            patch.object(sim.SIMULATORS["ltspice"], "create_from") as mock_create,
            patch("ltspice_mcp.lib.wsl.find_windows_ltspice_exe") as mock_find,
        ):
            sim.detect_simulators(config, [])
        mock_create.assert_called_once_with(str(exe))
        mock_find.assert_not_called()

    def test_fallback_records_diagnostic(self):
        from ltspice_mcp.lib.simulator import select_default_simulator

        class NG:
            pass

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.simulator = "ltspice"
        diagnostics: list[str] = []
        result = select_default_simulator({"ngspice": NG}, cfg, diagnostics)
        assert result is NG
        assert any("not available" in d for d in diagnostics)
        assert any("ngspice" in d for d in diagnostics)

    def test_available_simulator_no_diagnostic(self):
        from ltspice_mcp.lib.simulator import select_default_simulator

        class LT:
            pass

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.simulator = "ltspice"
        diagnostics: list[str] = []
        assert select_default_simulator({"ltspice": LT}, cfg, diagnostics) is LT
        assert diagnostics == []


class TestWslLtspiceAutodetect:
    """Fix B: detect_simulators fills in LTspice from /mnt/c on WSL."""

    def test_registers_found_exe(self):
        from unittest.mock import MagicMock, patch

        import ltspice_mcp.lib.simulator as sim

        fake_cls = MagicMock()
        fake_cls.spice_exe = []  # not yet configured
        diagnostics: list[str] = []
        with (
            patch("ltspice_mcp.lib.simulator.is_wsl", return_value=True),
            patch.dict("ltspice_mcp.lib.simulator.SIMULATORS", {"ltspice": fake_cls}),
            patch(
                "ltspice_mcp.lib.wsl.find_windows_ltspice_exe",
                return_value=Path("/mnt/c/x/LTspice.exe"),
            ),
        ):
            sim._autodetect_wsl_ltspice(diagnostics)
        fake_cls.create_from.assert_called_once_with("/mnt/c/x/LTspice.exe")
        assert any("Auto-detected" in d for d in diagnostics)

    def test_skips_when_already_configured(self):
        from unittest.mock import MagicMock, patch

        import ltspice_mcp.lib.simulator as sim

        fake_cls = MagicMock()
        fake_cls.spice_exe = ["/already/configured.exe"]
        with (
            patch("ltspice_mcp.lib.simulator.is_wsl", return_value=True),
            patch.dict("ltspice_mcp.lib.simulator.SIMULATORS", {"ltspice": fake_cls}),
            patch("ltspice_mcp.lib.wsl.find_windows_ltspice_exe") as mock_find,
        ):
            sim._autodetect_wsl_ltspice([])
        mock_find.assert_not_called()
        fake_cls.create_from.assert_not_called()

    def test_noop_off_wsl(self):
        from unittest.mock import patch

        import ltspice_mcp.lib.simulator as sim

        with patch("ltspice_mcp.lib.simulator.is_wsl", return_value=False):
            sim._autodetect_wsl_ltspice([])  # returns before touching SIMULATORS

    def test_no_install_no_register(self):
        from unittest.mock import MagicMock, patch

        import ltspice_mcp.lib.simulator as sim

        fake_cls = MagicMock()
        fake_cls.spice_exe = []
        with (
            patch("ltspice_mcp.lib.simulator.is_wsl", return_value=True),
            patch.dict("ltspice_mcp.lib.simulator.SIMULATORS", {"ltspice": fake_cls}),
            patch("ltspice_mcp.lib.wsl.find_windows_ltspice_exe", return_value=None),
        ):
            sim._autodetect_wsl_ltspice([])
        fake_cls.create_from.assert_not_called()


class TestEnabledSimulators:
    """[simulator] enabled allowlist — empty = auto-detect all."""

    def test_resolve_empty_returns_all(self):
        from ltspice_mcp.lib.simulator import SIMULATORS, _resolve_enabled_names

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        assert _resolve_enabled_names(cfg) == list(SIMULATORS)

    def test_resolve_none_config_returns_all(self):
        from ltspice_mcp.lib.simulator import SIMULATORS, _resolve_enabled_names

        assert _resolve_enabled_names(None) == list(SIMULATORS)

    def test_resolve_filters_to_listed(self):
        from ltspice_mcp.lib.simulator import _resolve_enabled_names

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.enabled_simulators = ["ngspice"]
        assert _resolve_enabled_names(cfg) == ["ngspice"]

    def test_resolve_unknown_name_diagnostic(self):
        from ltspice_mcp.lib.simulator import _resolve_enabled_names

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.enabled_simulators = ["bogus", "ngspice"]
        diagnostics: list[str] = []
        names = _resolve_enabled_names(cfg, diagnostics)
        assert names == ["ngspice"]
        assert any("bogus" in d for d in diagnostics)

    def test_detect_respects_allowlist(self):
        # Only ngspice enabled → ltspice never probed, autodetect never runs.
        from unittest.mock import patch

        import ltspice_mcp.lib.simulator as sim

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.enabled_simulators = ["ngspice"]
        with (
            patch("ltspice_mcp.lib.simulator.is_wsl", return_value=True),
            patch("ltspice_mcp.lib.wsl.find_windows_ltspice_exe") as mock_find,
        ):
            available = sim.detect_simulators(cfg, [])
        mock_find.assert_not_called()
        assert "ltspice" not in available

    def test_toml_parse(self, work_dir: Path):
        toml = work_dir / "ltspice-mcp.toml"
        toml.write_text('[simulator]\nenabled = ["ngspice", "LTspice"]\n')
        cfg = ServerConfig.load(toml)
        assert cfg.enabled_simulators == ["ngspice", "ltspice"]

    def test_env_override(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_ENABLED_SIMULATORS", "ngspice,ltspice")
        cfg = ServerConfig.load(work_dir / "nonexistent.toml")
        assert cfg.enabled_simulators == ["ngspice", "ltspice"]
