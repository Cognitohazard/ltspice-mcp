"""Tests for server.py — error hints, asc editor configuration, and dispatch."""

from pathlib import Path
from unittest.mock import patch

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import (
    LibraryError,
    LTSpiceMCPError,
    NetlistError,
    PathSecurityError,
)
from ltspice_mcp.server import (
    _configure_asc_editor,
    _get_error_hint,
    call_tool,
    list_resources,
    list_tools,
    read_resource,
)
from ltspice_mcp.state import SessionState


class TestGetErrorHint:
    def test_known_full(self):
        hint = _get_error_hint(NetlistError, "full")
        assert hint is not None
        assert "ltspice_read_circuit" in hint

    def test_known_agentic(self):
        hint = _get_error_hint(LibraryError, "agentic")
        assert hint is not None

    def test_unknown_returns_none(self):
        class FakeErr(LTSpiceMCPError):
            pass

        assert _get_error_hint(FakeErr, "full") is None


class TestConfigureAscEditor:
    def test_explicit_symbol_paths(self, tmp_path: Path):
        symdir = tmp_path / "syms"
        symdir.mkdir()
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = [symdir]
        with patch("spicelib.editor.asc_editor.AscEditor") as mock_cls:
            mock_cls.custom_lib_paths = []
            _configure_asc_editor(cfg, available={})
            assert str(symdir) in mock_cls.custom_lib_paths

    def test_explicit_symbol_paths_invalid(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = [tmp_path / "nonexistent"]
        # Falls through to other checks; with no LTspice → no-op
        _configure_asc_editor(cfg, available={})

    def test_no_ltspice(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        _configure_asc_editor(cfg, available={})

    def test_wsl_no_lib_paths(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with patch("ltspice_mcp.lib.wsl.is_wsl", return_value=True), \
             patch("ltspice_mcp.lib.wsl.get_ltspice_lib_paths", return_value=[]):
            _configure_asc_editor(cfg, available={"ltspice": object})

    def test_wsl_with_lib_paths(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        symdir = tmp_path / "wslsyms"
        symdir.mkdir()
        with patch("ltspice_mcp.lib.wsl.is_wsl", return_value=True), \
             patch("ltspice_mcp.lib.wsl.get_ltspice_lib_paths", return_value=[str(symdir)]):
            _configure_asc_editor(cfg, available={"ltspice": object})


class _FakeSession:
    """Stub MCP session — log/progress calls are no-ops."""

    async def send_log_message(self, **kwargs):
        pass

    async def send_progress_notification(self, **kwargs):
        pass


class _FakeRequestContext:
    def __init__(self, state: SessionState):
        self.lifespan_context = {"state": state}
        self.session = _FakeSession()
        self.meta = None


class _FakeServer:
    def __init__(self, state: SessionState):
        self.request_context = _FakeRequestContext(state)


@pytest.mark.asyncio
class TestServerDispatch:
    """Test list_tools / call_tool / list_resources / read_resource via patched server."""

    async def test_list_tools(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            tools = await list_tools()  # type: ignore[call-arg]
            assert len(tools) > 0

    async def test_call_unknown_tool(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)), \
             pytest.raises(ValueError, match="Unknown tool"):
            await call_tool("ltspice_nonexistent", {})

    async def test_call_validation_error(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)), \
             pytest.raises(ValueError, match="Invalid arguments"):
            await call_tool("ltspice_create_netlist", {"missing": "field"})

    async def test_call_path_security_error(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)), \
             pytest.raises(PathSecurityError, match="Allowed paths"):
            await call_tool("ltspice_read_circuit", {"path": "/etc/passwd"})

    async def test_call_ltspice_error_with_hint(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)), \
             pytest.raises(NetlistError) as excinfo:
            await call_tool("ltspice_read_circuit", {"path": "missing.cir"})
        msg = str(excinfo.value)
        assert "ltspice_read_circuit" in msg or "ltspice_list_components" in msg

    async def test_list_resources(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            resources = await list_resources()  # type: ignore[call-arg]
            assert len(resources) > 0

    async def test_read_resource_invalid_uri(self, state_no_sim: SessionState):
        from pydantic import AnyUrl

        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)), \
             pytest.raises(ValueError, match="Unknown"):
            await read_resource(AnyUrl("ltspice://nonexistent"))

    async def test_read_resource_valid(self, state_no_sim: SessionState):
        from pydantic import AnyUrl

        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            result = await read_resource(AnyUrl("ltspice://config"))
            result = list(result)
            assert len(result) > 0
