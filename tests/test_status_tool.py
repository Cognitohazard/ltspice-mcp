"""Tests for ltspice_server_status tool handler."""

import typing
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import recent
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.status import (
    RecentInput,
    ServerStatusInput,
    handle_recent,
    handle_server_status,
)


@pytest.mark.asyncio
class TestGetServerStatus:
    async def test_no_simulator_text(self, state_no_sim: SessionState):
        result = await handle_server_status(ServerStatusInput(), state_no_sim)
        text = result.content[0].text
        assert "LTSpice MCP Server Status" in text
        assert "degraded mode" in text
        assert "Tool profile" in text
        assert "Allowed paths" in text
        assert result.structuredContent is not None
        assert result.structuredContent["default_simulator"] is None

    async def test_with_simulator(self, state_with_sim: SessionState):
        result = await handle_server_status(ServerStatusInput(), state_with_sim)
        text = result.content[0].text
        assert "fake: available" in text
        assert "default" in text
        assert "/fake/path/sim.exe" in text
        assert result.structuredContent["default_simulator"] == "FakeSim"
        assert result.structuredContent["simulators"]["fake"]["default"] is True

    async def test_json_format(self, state_no_sim: SessionState):
        result = await handle_server_status(ServerStatusInput(format="json"), state_no_sim)
        assert result.structuredContent is not None
        assert "configuration" in result.structuredContent
        assert "runtime" in result.structuredContent

    async def test_config_file_present(self, state_no_sim: SessionState, work_dir):
        # Deliberately NOT working_dir/ltspice-mcp.toml: server_status must
        # report the path config actually resolved (config_path, which honors
        # LTSPICE_MCP_CONFIG), never a working_dir guess.
        cfg = work_dir / "custom-config.toml"
        cfg.write_text("[simulator]\nname = 'ltspice'\n")
        state_no_sim.config.config_path = cfg
        result = await handle_server_status(ServerStatusInput(), state_no_sim)
        assert "Config file:" in result.content[0].text
        sc = result.structuredContent
        # Config file path and the simulator-selection knob must reach the
        # structured surface an agent parses, not just the human text.
        assert sc["configuration"]["config_file"] == str(cfg)
        assert sc["configuration"]["config_file_exists"] is True
        assert "[simulator] default" in sc["simulator_select"]
        assert "LTSPICE_MCP_CONFIG" in sc["simulator_select"]

    async def test_persist_jobs_surfaced(self, state_no_sim: SessionState):
        # An agent must be able to tell "persistence off" from "nothing run yet".
        state_no_sim.config.persist_jobs = False
        result = await handle_server_status(ServerStatusInput(), state_no_sim)
        sc = result.structuredContent
        assert sc["configuration"]["persist_jobs"] is False
        assert "preload_recent_count" in sc["configuration"]
        assert "in-memory only" in result.content[0].text

    async def test_no_diagnostics_clean_status(self, state_no_sim: SessionState):
        result = await handle_server_status(ServerStatusInput(format="json"), state_no_sim)
        sc = result.structuredContent
        assert sc["diagnostics"] == []
        assert "Startup diagnostics" not in result.content[0].text

    async def test_simulator_fallback_surfaced(self, config: ServerConfig):
        """Fix A: requested≠active fallback must appear in server_status."""

        class NG:
            spice_exe: typing.ClassVar[list[str]] = ["ngspice"]

        config.simulator = "ltspice"
        state = SessionState.create(config, available={"ngspice": NG})
        # Default format → human text in content + structuredContent both present.
        result = await handle_server_status(ServerStatusInput(), state)
        sc = result.structuredContent
        assert sc["requested_simulator"] == "ltspice"
        assert sc["default_simulator"] == "NG"
        assert any("ngspice" in d for d in sc["diagnostics"])
        text = result.content[0].text
        assert "Startup diagnostics" in text
        assert "Requested simulator: ltspice" in text


@pytest.mark.asyncio
class TestRecent:
    @pytest.fixture
    def recent_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Redirect the user-global recent index to a per-test temp dir."""
        home = tmp_path / "ltspice-mcp-home"
        monkeypatch.setenv("LTSPICE_MCP_HOME", str(home))
        return home

    async def test_empty_mirrors_hint_into_structured(
        self, state_no_sim: SessionState, recent_home: Path
    ):
        result = await handle_recent(RecentInput(), state_no_sim)
        text = result.content[0].text
        assert "No recent circuits" in text
        sc = result.structuredContent
        assert sc["circuits"] == []
        assert sc["count"] == 0
        # Structured-aware clients only see the data dict — the empty-state
        # guidance must be mirrored there, not live only in the text channel.
        assert sc["hint"] == text
        assert "run_simulation" in sc["hint"]

    async def test_populated_omits_hint(
        self, state_no_sim: SessionState, recent_home: Path, tmp_path: Path
    ):
        circuit = tmp_path / "rc.cir"
        circuit.write_text("* rc\n.end\n")
        recent.touch(circuit)
        result = await handle_recent(RecentInput(), state_no_sim)
        sc = result.structuredContent
        assert sc["count"] == 1
        assert sc["circuits"][0]["path"] == str(circuit.resolve())
        # hint is empty-state guidance only.
        assert "hint" not in sc
