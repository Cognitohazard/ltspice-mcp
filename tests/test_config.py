"""Unit tests for configuration loading."""

import dataclasses
import logging
import os
from pathlib import Path

import pytest

import ltspice_mcp.config as config_module
from ltspice_mcp.config import ServerConfig, generate_default_config


class TestServerConfig:
    """Tests for ServerConfig loading."""

    def test_defaults(self):
        config = ServerConfig()
        assert config.simulator is None
        assert config.simulator_exe is None
        assert config.max_parallel_sims == min(os.cpu_count() or 4, 8)
        assert config.default_timeout == 300.0
        assert config.analysis_budget_s == 60.0
        assert config.result_set_ttl_hours == 24.0
        # WARNING, not INFO: the server's stderr is the caller's stderr on the
        # in-process and per-script doors, and a startup banner there gets
        # answered with a blanket 2>/dev/null that also hides real tracebacks.
        assert config.log_level == "WARNING"

    def test_max_parallel_defaults_to_capped_core_count(self, monkeypatch: pytest.MonkeyPatch):
        # Core-aware default: use the host's cores, but cap so a many-core box
        # doesn't spawn dozens of cold simulator processes.
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: 64)
        assert ServerConfig().max_parallel_sims == 8
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: 2)
        assert ServerConfig().max_parallel_sims == 2
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: None)
        assert ServerConfig().max_parallel_sims == 4

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

    def test_ngbehavior_unset_is_none(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LTSPICE_MCP_NGBEHAVIOR", raising=False)
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[simulator]\ndefault = "ngspice"\n')
        assert ServerConfig.load(toml_path).ngbehavior is None

    def test_ngbehavior_from_toml(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("LTSPICE_MCP_NGBEHAVIOR", raising=False)
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[simulator]\nngbehavior = "kipsa"\n')
        assert ServerConfig.load(toml_path).ngbehavior == "kipsa"

    def test_ngbehavior_env_overrides_toml(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[simulator]\nngbehavior = "kipsa"\n')
        monkeypatch.setenv("LTSPICE_MCP_NGBEHAVIOR", "hsa")
        assert ServerConfig.load(toml_path).ngbehavior == "hsa"

    def test_ngbehavior_non_string_toml_ignored(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("LTSPICE_MCP_NGBEHAVIOR", raising=False)
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text("[simulator]\nngbehavior = 42\n")
        assert ServerConfig.load(toml_path).ngbehavior is None

    def test_unknown_toml_section_is_ignored(self, work_dir: Path):
        """Sections the server no longer reads (e.g. the retired [plotting])
        must be silently skipped, not crash the load."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(
            '[plotting]\ndpi = 150\nstyle = "seaborn-v0_8-darkgrid"\n'
            "[simulation]\nmax_parallel = 8\n"
        )
        config = ServerConfig.load(toml_path)
        assert config.max_parallel_sims == 8
        assert not hasattr(config, "plot_dpi")

    def test_generated_config_has_no_plotting_section(self, work_dir: Path):
        path = work_dir / "generated.toml"
        generate_default_config(path)
        assert "[plotting]" not in path.read_text()

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
        assert "analysis_budget_s" in content
        assert "result_set_ttl_hours" in content

    def test_analysis_budget_and_result_ttl_load_from_toml(
        self,
        work_dir: Path,
    ):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text("[analysis]\nanalysis_budget_s = 12.5\nresult_set_ttl_hours = 48\n")
        config = ServerConfig.load(toml_path)
        assert config.analysis_budget_s == 12.5
        assert config.result_set_ttl_hours == 48.0

    def test_generated_config_does_not_pin_max_parallel(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A freshly generated config must ship max_parallel commented out so the
        dynamic min(CPU cores, 8) default survives. A live ``max_parallel = 4``
        would silently cap every auto-generated install at 4."""
        path = work_dir / "generated.toml"
        generate_default_config(path)
        content = path.read_text()
        # The key appears only as a commented example, never as a live assignment.
        assert not any(ln.strip().startswith("max_parallel") for ln in content.splitlines())
        assert "# max_parallel" in content
        # Loading the generated file fresh leaves the core-aware default intact.
        monkeypatch.setattr(config_module.os, "cpu_count", lambda: 64)
        monkeypatch.delenv("LTSPICE_MCP_MAX_PARALLEL", raising=False)
        config = ServerConfig.load(path)
        assert config.max_parallel_sims == 8


class TestToolProfile:
    """There is one tool surface, and no setting names it."""

    @pytest.mark.parametrize("value", ["consolidated", "full", "agentic", "bogus"])
    def test_a_profile_key_in_toml_is_not_read(self, work_dir: Path, value: str):
        """``[tools] profile`` is no longer a key this loader knows. Whatever it
        names, the file still loads and the rest of the section is read — an
        unknown key is ignored like any other."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(f'[tools]\nprofile = "{value}"\nlisting = "compact"\n')
        config = ServerConfig.load(toml_path)
        assert not hasattr(config, "tool_profile")
        assert config.tool_listing == "compact"

    def test_a_profile_env_var_is_not_read(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("LTSPICE_MCP_TOOL_PROFILE", "full")
        config = ServerConfig.load(work_dir / "nonexistent.toml")
        assert not hasattr(config, "tool_profile")

    def test_generated_config_includes_tools_section(self, work_dir: Path):
        path = work_dir / "generated.toml"
        generate_default_config(path)
        content = path.read_text()
        assert "[tools]" in content
        assert "listing" in content
        assert "profile" not in content


class TestToolListing:
    """[tools] listing selects how the tool list is served."""

    def test_default_listing_is_full(self):
        assert ServerConfig().tool_listing == "full"

    @pytest.mark.parametrize("mode", ["full", "compact"])
    def test_listing_from_toml(self, work_dir: Path, mode: str):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(f'[tools]\nlisting = "{mode}"\n')
        assert ServerConfig.load(toml_path).tool_listing == mode

    @pytest.mark.parametrize("mode", ["full", "compact"])
    def test_listing_from_env(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch, mode: str):
        monkeypatch.setenv("LTSPICE_MCP_TOOL_LISTING", mode)
        assert ServerConfig.load(work_dir / "nonexistent.toml").tool_listing == mode

    def test_env_overrides_toml(self, work_dir: Path, monkeypatch: pytest.MonkeyPatch):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nlisting = "full"\n')
        monkeypatch.setenv("LTSPICE_MCP_TOOL_LISTING", "compact")
        assert ServerConfig.load(toml_path).tool_listing == "compact"

    def test_unknown_value_in_toml_falls_back_to_full(
        self, work_dir: Path, caplog: pytest.LogCaptureFixture
    ):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nlisting = "sparse"\n')
        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.config"):
            config = ServerConfig.load(toml_path)
        assert config.tool_listing == "full"
        message = "\n".join(record.getMessage() for record in caplog.records)
        assert "sparse" in message
        assert "compact" in message, "the warning must enumerate the valid values"

    def test_unknown_env_value_does_not_clobber_a_valid_toml_listing(
        self, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text('[tools]\nlisting = "compact"\n')
        monkeypatch.setenv("LTSPICE_MCP_TOOL_LISTING", "sparse")
        assert ServerConfig.load(toml_path).tool_listing == "compact"

    def test_generated_config_documents_the_listing_key(self, work_dir: Path):
        path = work_dir / "generated.toml"
        generate_default_config(path)
        content = path.read_text()
        assert 'listing = "full"' in content
        assert "compact" in content


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
        """An LTspice-looking path must not bind to ngspice."""
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
        # Must point at how to make the requested simulator appear, not dead-end.
        assert any("[simulator] enabled" in d for d in diagnostics)

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


class TestSimulatorRemediation:
    """The capabilities remediation names the keys the loader actually reads,
    and leads with the allowlist when the allowlist is the cause."""

    def test_names_the_loader_keys(self):
        from ltspice_mcp.config import SIM_PATH_ENV
        from ltspice_mcp.lib.simulator import simulator_remediation

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        remediation = simulator_remediation("ltspice", cfg)
        assert remediation["config_key"] == "simulator.path"
        assert remediation["env_var"] == SIM_PATH_ENV
        assert remediation["config_file"] == str(cfg.config_path)
        assert "restart" in str(remediation["action"])
        # On this project's platforms an example executable exists for ltspice.
        assert "LTspice" in str(remediation.get("example_value", "LTspice"))

    def test_allowlist_exclusion_is_the_first_fact(self):
        """A simulator turned off by [simulator] enabled must not be answered
        with an install hint — the install is not the cause."""
        from ltspice_mcp.lib.simulator import simulator_remediation

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.enabled_simulators = ["ngspice"]
        remediation = simulator_remediation("ltspice", cfg)
        assert remediation["excluded_by_allowlist"] is True
        action = str(remediation["action"])
        assert "simulator.enabled" in action
        assert "Install" not in action

    def test_loader_and_remediation_share_one_key_spelling(self):
        """The constant is used at the loader's read site: a config written
        with the remediation's key names must actually load."""
        import textwrap

        from ltspice_mcp.lib.simulator import simulator_remediation

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        remediation = simulator_remediation("ltspice", cfg)
        section, key = str(remediation["config_key"]).split(".")
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            toml = Path(tmp) / "ltspice-mcp.toml"
            toml.write_text(
                textwrap.dedent(f"""
                    [{section}]
                    {key} = "/opt/fake/LTspice.exe"
                """)
            )
            loaded = ServerConfig.load(toml)
        assert loaded.simulator_exe == Path("/opt/fake/LTspice.exe")


# Every LTSPICE_MCP_* variable the loader reads, with a value distinct from
# both the dataclass default and the TOML below.
ENV_OVERRIDES: dict[str, str] = {
    "LTSPICE_MCP_SIMULATOR": "ltspice",
    "LTSPICE_MCP_ENABLED_SIMULATORS": "LTspice, xyce",
    "LTSPICE_MCP_SIMULATOR_EXE": "/opt/env/ltspice",
    "LTSPICE_MCP_NGBEHAVIOR": "  hsa  ",
    "LTSPICE_MCP_WORKING_DIR": "/tmp/env-working-dir",
    "LTSPICE_MCP_ALLOWED_PATHS": f"/tmp/env-a{os.pathsep}/tmp/env-b",
    "LTSPICE_MCP_MAX_PARALLEL": "9",
    "LTSPICE_MCP_MAX_EXPERIMENT_CASES": "88",
    "LTSPICE_MCP_TIMEOUT": "99.5",
    "LTSPICE_MCP_MAX_POINTS": "777",
    "LTSPICE_MCP_ANALYSIS_BUDGET_S": "21.5",
    "LTSPICE_MCP_DEFAULT_BUDGET": "3300",
    "LTSPICE_MCP_RESULT_SET_TTL_HOURS": "72",
    "LTSPICE_MCP_MAX_ESTIMATED_POINTS": "7654321",
    "LTSPICE_MCP_MAX_RAW_MB": "256",
    "LTSPICE_MCP_LOG_LEVEL": "error",
    "LTSPICE_MCP_SYMBOL_PATHS": f"/tmp/env-sym-a{os.pathsep}/tmp/env-sym-b",
    "LTSPICE_MCP_TOOL_LISTING": "full",
    "LTSPICE_MCP_PERSIST_JOBS": "on",
    "LTSPICE_MCP_PRELOAD_RECENT_COUNT": "7",
}

# One TOML naming every key the loader reads, with values distinct from both
# the dataclass defaults and the env values above, so a dropped key shows up
# as a default and a swapped precedence shows up as the wrong source.
FULL_TOML = """
[simulator]
default = "ngspice"
path = "/opt/toml/ngspice"
enabled = ["NGspice", " LTspice "]
ngbehavior = "  kipsa  "

[security]
allowed_paths = ["/tmp/toml-a", "/tmp/toml-b"]

[simulation]
max_parallel = 6
max_experiment_cases = 77
timeout = 45.5
max_estimated_points = 1234567
max_raw_mb = 512

[analysis]
max_points = 555
analysis_budget_s = 12.5
default_budget = 2500
result_set_ttl_hours = 48

[logging]
level = "debug"

[schematic]
symbol_paths = ["/tmp/sym-a", "/tmp/sym-b"]

[tools]
listing = "compact"

[state]
persist_jobs = false
preload_recent_count = 3
"""


@pytest.fixture
def no_ltspice_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop every LTSPICE_MCP_* variable so a load sees only TOML + defaults."""
    for name in list(os.environ):
        if name.startswith("LTSPICE_MCP_"):
            monkeypatch.delenv(name, raising=False)


def _snapshot(cfg: ServerConfig) -> dict[str, object]:
    """Every field of a loaded config, as a plain dict."""
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)}


class TestLoadCoversEveryKey:
    """Whole-surface snapshots of ``ServerConfig.load``.

    Each key is read from TOML, overridden from the environment, and rejected
    on a bad value in one place, so a change to how the loader is wired shows
    up as a diff in a snapshot rather than as one silently dropped setting.
    """

    def test_every_toml_key_lands_on_its_field(self, work_dir: Path, no_ltspice_env: None):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(FULL_TOML)
        assert _snapshot(ServerConfig.load(toml_path)) == {
            "simulator": "ngspice",
            "enabled_simulators": ["ngspice", "ltspice"],
            "simulator_exe": Path("/opt/toml/ngspice"),
            "ngbehavior": "kipsa",
            "working_dir": Path.cwd(),
            "allowed_paths": [Path("/tmp/toml-a"), Path("/tmp/toml-b")],
            "max_parallel_sims": 6,
            "max_experiment_cases": 77,
            "default_timeout": 45.5,
            "max_estimated_points": 1234567,
            "max_raw_mb": 512,
            "max_points_returned": 555,
            "analysis_budget_s": 12.5,
            "result_set_ttl_hours": 48.0,
            "default_budget": 2500,
            "log_level": "DEBUG",
            "symbol_paths": [Path("/tmp/sym-a"), Path("/tmp/sym-b")],
            "tool_listing": "compact",
            "persist_jobs": False,
            "preload_recent_count": 3,
            "config_path": toml_path,
        }

    def test_every_env_var_overrides_its_toml_key(
        self, work_dir: Path, no_ltspice_env: None, monkeypatch: pytest.MonkeyPatch
    ):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(FULL_TOML)
        for name, value in ENV_OVERRIDES.items():
            monkeypatch.setenv(name, value)
        assert _snapshot(ServerConfig.load(toml_path)) == {
            "simulator": "ltspice",
            "enabled_simulators": ["ltspice", "xyce"],
            "simulator_exe": Path("/opt/env/ltspice"),
            "ngbehavior": "hsa",
            "working_dir": Path("/tmp/env-working-dir"),
            "allowed_paths": [Path("/tmp/env-a"), Path("/tmp/env-b")],
            "max_parallel_sims": 9,
            "max_experiment_cases": 88,
            "default_timeout": 99.5,
            "max_estimated_points": 7654321,
            "max_raw_mb": 256,
            "max_points_returned": 777,
            "analysis_budget_s": 21.5,
            "result_set_ttl_hours": 72.0,
            "default_budget": 3300,
            "log_level": "ERROR",
            "symbol_paths": [Path("/tmp/env-sym-a"), Path("/tmp/env-sym-b")],
            "tool_listing": "full",
            "persist_jobs": True,
            "preload_recent_count": 7,
            "config_path": toml_path,
        }

    def test_env_overrides_apply_without_any_toml(
        self, work_dir: Path, no_ltspice_env: None, monkeypatch: pytest.MonkeyPatch
    ):
        """With no TOML file at all, every env override still applies."""
        missing = work_dir / "nonexistent.toml"
        for name, value in ENV_OVERRIDES.items():
            monkeypatch.setenv(name, value)
        snapshot = _snapshot(ServerConfig.load(missing))
        assert snapshot["max_parallel_sims"] == 9
        assert snapshot["default_timeout"] == 99.5
        assert snapshot["log_level"] == "ERROR"
        assert snapshot["persist_jobs"] is True
        assert snapshot["config_path"] == missing

    def test_a_rejected_env_value_leaves_the_toml_value_standing(
        self,
        work_dir: Path,
        no_ltspice_env: None,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ):
        """Every validating key: a bad env value warns and is dropped, so the
        TOML value survives instead of being clobbered."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(FULL_TOML)
        rejected = {
            "LTSPICE_MCP_ENABLED_SIMULATORS": "",  # falsy: no override at all
            "LTSPICE_MCP_NGBEHAVIOR": "   ",  # blank after strip
            "LTSPICE_MCP_MAX_PARALLEL": "9999",  # above the bound
            "LTSPICE_MCP_MAX_EXPERIMENT_CASES": "0",  # below the bound
            "LTSPICE_MCP_TIMEOUT": "0",  # exclusive minimum
            "LTSPICE_MCP_MAX_POINTS": "not-a-number",
            "LTSPICE_MCP_ANALYSIS_BUDGET_S": "0",  # exclusive minimum
            "LTSPICE_MCP_DEFAULT_BUDGET": "-1",
            "LTSPICE_MCP_RESULT_SET_TTL_HOURS": "0",  # exclusive minimum
            "LTSPICE_MCP_MAX_ESTIMATED_POINTS": "0",
            "LTSPICE_MCP_MAX_RAW_MB": "0",
            "LTSPICE_MCP_LOG_LEVEL": "LOUD",
            "LTSPICE_MCP_TOOL_LISTING": "sparse",
            "LTSPICE_MCP_PERSIST_JOBS": "maybe",
            "LTSPICE_MCP_PRELOAD_RECENT_COUNT": "-2",
        }
        for name, value in rejected.items():
            monkeypatch.setenv(name, value)
        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.config"):
            snapshot = _snapshot(ServerConfig.load(toml_path))
        assert snapshot["enabled_simulators"] == ["ngspice", "ltspice"]
        assert snapshot["ngbehavior"] == "kipsa"
        assert snapshot["max_parallel_sims"] == 6
        assert snapshot["max_experiment_cases"] == 77
        assert snapshot["default_timeout"] == 45.5
        assert snapshot["max_points_returned"] == 555
        assert snapshot["analysis_budget_s"] == 12.5
        assert snapshot["default_budget"] == 2500
        assert snapshot["result_set_ttl_hours"] == 48.0
        assert snapshot["max_estimated_points"] == 1234567
        assert snapshot["max_raw_mb"] == 512
        assert snapshot["log_level"] == "DEBUG"
        assert snapshot["tool_listing"] == "compact"
        assert snapshot["persist_jobs"] is False
        assert snapshot["preload_recent_count"] == 3
        message = "\n".join(record.getMessage() for record in caplog.records)
        for named in (
            "LTSPICE_MCP_MAX_PARALLEL",
            "LTSPICE_MCP_MAX_EXPERIMENT_CASES",
            "LTSPICE_MCP_TIMEOUT",
            "LTSPICE_MCP_MAX_POINTS",
            "LTSPICE_MCP_ANALYSIS_BUDGET_S",
            "LTSPICE_MCP_DEFAULT_BUDGET",
            "LTSPICE_MCP_RESULT_SET_TTL_HOURS",
            "LTSPICE_MCP_MAX_ESTIMATED_POINTS",
            "LTSPICE_MCP_MAX_RAW_MB",
            "LTSPICE_MCP_LOG_LEVEL",
            "LTSPICE_MCP_PERSIST_JOBS",
            "LTSPICE_MCP_PRELOAD_RECENT_COUNT",
        ):
            assert named in message, f"no warning named {named}: {message!r}"

    def test_a_rejected_toml_value_falls_back_to_the_default(
        self, work_dir: Path, no_ltspice_env: None, caplog: pytest.LogCaptureFixture
    ):
        """The TOML side of the same contract: an out-of-range or wrongly typed
        value is dropped with a warning, never half-applied."""
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(
            "[simulator]\n"
            "ngbehavior = 42\n"
            'enabled = "ngspice"\n'
            "[security]\n"
            'allowed_paths = "/tmp/scalar"\n'
            "[simulation]\n"
            "max_parallel = 0\n"
            "max_experiment_cases = 99999999\n"
            "timeout = 0\n"
            "max_estimated_points = 0\n"
            "max_raw_mb = 0\n"
            "[analysis]\n"
            'max_points = "lots"\n'
            "analysis_budget_s = 0\n"
            "default_budget = -1\n"
            "result_set_ttl_hours = 0\n"
            "[logging]\n"
            'level = "LOUD"\n'
            "[schematic]\n"
            'symbol_paths = "/tmp/scalar-sym"\n'
            "[state]\n"
            'persist_jobs = "yes"\n'
            "preload_recent_count = -1\n"
        )
        with caplog.at_level(logging.WARNING, logger="ltspice_mcp.config"):
            config = ServerConfig.load(toml_path)
        defaults = ServerConfig(working_dir=config.working_dir)
        assert config.ngbehavior is None
        assert config.enabled_simulators == []
        # A scalar string must NOT be expanded character-wise into paths.
        assert config.allowed_paths == [config.working_dir]
        assert config.symbol_paths == []
        assert config.max_parallel_sims == defaults.max_parallel_sims
        assert config.max_experiment_cases == defaults.max_experiment_cases
        assert config.default_timeout == defaults.default_timeout
        assert config.max_estimated_points == defaults.max_estimated_points
        assert config.max_raw_mb == defaults.max_raw_mb
        assert config.max_points_returned == defaults.max_points_returned
        assert config.analysis_budget_s == defaults.analysis_budget_s
        assert config.default_budget == defaults.default_budget
        assert config.result_set_ttl_hours == defaults.result_set_ttl_hours
        assert config.log_level == defaults.log_level
        assert config.persist_jobs is True
        assert config.preload_recent_count == defaults.preload_recent_count

    def test_explicit_overrides_win_over_environment(
        self, work_dir: Path, no_ltspice_env: None, monkeypatch: pytest.MonkeyPatch
    ):
        toml_path = work_dir / "ltspice-mcp.toml"
        toml_path.write_text(FULL_TOML)
        monkeypatch.setenv("LTSPICE_MCP_MAX_PARALLEL", "9")
        config = ServerConfig.load(toml_path, overrides={"max_parallel_sims": 3})
        assert config.max_parallel_sims == 3


def test_checked_in_example_matches_the_generator(tmp_path: Path) -> None:
    """ltspice-mcp.example.toml is the generator's output plus a two-line
    header; it once drifted to a removed profile for a whole release cycle."""
    from ltspice_mcp.config import generate_default_config

    generated = tmp_path / "generated.toml"
    generate_default_config(generated)
    example = (Path(__file__).resolve().parent.parent / "ltspice-mcp.example.toml").read_text()
    body = "\n".join(example.splitlines()[2:]) + "\n"
    expected = "\n".join(generated.read_text().splitlines()[1:]) + "\n"
    assert body == expected
