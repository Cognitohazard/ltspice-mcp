"""Tests for ltspice_get_server_status tool handler."""

import typing

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.status import GetServerStatusInput, handle_get_server_status


class FakeSim:
    """Stub simulator class for tests that need a default simulator."""

    spice_exe: typing.ClassVar[list[str]] = ["/fake/path/sim.exe"]


@pytest.fixture
def state_with_sim(config: ServerConfig) -> SessionState:
    return SessionState.create(config, available={"fake": FakeSim})


@pytest.mark.asyncio
class TestGetServerStatus:
    async def test_no_simulator_text(self, state_no_sim: SessionState):
        result = await handle_get_server_status(GetServerStatusInput(), state_no_sim)
        text = result.content[0].text
        assert "LTSpice MCP Server Status" in text
        assert "degraded mode" in text
        assert "Tool profile" in text
        assert "Allowed paths" in text
        assert result.structuredContent is not None
        assert result.structuredContent["default_simulator"] is None

    async def test_with_simulator(self, state_with_sim: SessionState):
        result = await handle_get_server_status(GetServerStatusInput(), state_with_sim)
        text = result.content[0].text
        assert "fake: available" in text
        assert "default" in text
        assert "/fake/path/sim.exe" in text
        assert result.structuredContent["default_simulator"] == "FakeSim"
        assert result.structuredContent["simulators"]["fake"]["default"] is True

    async def test_json_format(self, state_no_sim: SessionState):
        result = await handle_get_server_status(
            GetServerStatusInput(format="json"), state_no_sim
        )
        assert result.structuredContent is not None
        assert "configuration" in result.structuredContent
        assert "runtime" in result.structuredContent

    async def test_config_file_present(
        self, state_no_sim: SessionState, work_dir
    ):
        cfg = work_dir / "ltspice-mcp.toml"
        cfg.write_text("[simulator]\nname = 'ltspice'\n")
        result = await handle_get_server_status(GetServerStatusInput(), state_no_sim)
        assert "Config file:" in result.content[0].text
