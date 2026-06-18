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
    SERVER_INSTRUCTIONS,
    _configure_asc_editor,
    _get_error_hint,
    build_instructions,
    call_tool,
    list_resources,
    list_tools,
    read_resource,
    server,
)
from ltspice_mcp.state import SessionState
from tests.conftest import _FakeServer


class TestGetErrorHint:
    def test_known_full(self):
        hint = _get_error_hint(NetlistError, "full")
        assert hint is not None
        assert "read_circuit" in hint

    def test_known_agentic(self):
        hint = _get_error_hint(LibraryError, "agentic")
        assert hint is not None

    def test_unknown_returns_none(self):
        class FakeErr(LTSpiceMCPError):
            pass

        assert _get_error_hint(FakeErr, "full") is None


class TestServerInstructions:
    def test_instructions_forwarded_to_init_options(self):
        # The block must reach the client at the MCP initialize handshake.
        opts = server.create_initialization_options()
        assert opts.instructions == SERVER_INSTRUCTIONS
        assert opts.instructions  # non-empty

    def test_instructions_cover_key_workflow_guidance(self):
        text = SERVER_INSTRUCTIONS
        # netlist-first default + the asc-build doctrine
        assert "netlist" in text.lower()
        assert "apply_schematic_ops" in text
        # analysis-tool/run-type pairing + the result-trust guardrail
        assert "operating_point" in text and "bode_metrics" in text
        assert "completed" in text
        # must name only tools present in BOTH profiles (create_netlist is full-only)
        assert "create_netlist" not in text


class _LT:
    pass


class _NG:
    pass


class TestBuildInstructions:
    """The runtime-prepended line must name the actually-detected simulators."""

    def test_includes_static_body(self):
        assert SERVER_INSTRUCTIONS in build_instructions({"ngspice": _NG}, _NG)

    def test_none_detected(self):
        text = build_instructions({}, None)
        assert "No SPICE simulator detected" in text

    def test_ngspice_only_notes_ltspice_absence(self):
        text = build_instructions({"ngspice": _NG}, _NG)
        assert "Active simulator: ngspice." in text
        assert "(LTspice not detected.)" in text
        assert "(default)" not in text  # no default marker for a single engine

    def test_ltspice_only(self):
        text = build_instructions({"ltspice": _LT}, _LT)
        assert "Active simulator: LTspice." in text
        assert "LTspice not detected" not in text

    def test_both_marks_default(self):
        text = build_instructions({"ltspice": _LT, "ngspice": _NG}, _LT)
        assert "Active simulators: LTspice (default), ngspice." in text
        assert "LTspice not detected" not in text


class TestConfigureAscEditor:
    """Symbol-path resolution. Every test mocks ``is_wsl`` (the suite runs on a
    real WSL host) and patches ``AscEditor`` so no test mutates the shared
    class-level ``custom_lib_paths`` global."""

    def test_explicit_symbol_paths(self, tmp_path: Path):
        symdir = tmp_path / "syms"
        symdir.mkdir()
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = [symdir]
        with patch("spicelib.editor.asc_editor.AscEditor") as mock_cls:
            mock_cls.custom_lib_paths = []
            _configure_asc_editor(cfg, available={})
            assert str(symdir) in mock_cls.custom_lib_paths

    def test_explicit_symbol_paths_invalid_non_wsl(self, tmp_path: Path):
        # Invalid override + non-WSL + no LTspice → disabled, nothing set.
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = [tmp_path / "nonexistent"]
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=False),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = []
            mock_cls.simulator_lib_paths = []
            _configure_asc_editor(cfg, available={})
            assert mock_cls.custom_lib_paths == []

    def test_non_wsl_no_ltspice_disabled(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=False),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = []
            _configure_asc_editor(cfg, available={})
            assert mock_cls.custom_lib_paths == []

    def test_non_wsl_prepare_for_simulator(self, tmp_path: Path):
        # Windows-native / Wine path: needs the detected LTspice class.
        class FakeLT:
            pass

        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=False),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = ["/x/lib/sym"]
            mock_cls.simulator_lib_paths = []
            _configure_asc_editor(cfg, available={"ltspice": FakeLT})
            mock_cls.prepare_for_simulator.assert_called_once_with(FakeLT)

    def test_wsl_no_lib_paths_disabled(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=True),
            patch("ltspice_mcp.lib.wsl.get_ltspice_lib_paths", return_value=[]),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = []
            _configure_asc_editor(cfg, available={})
            assert mock_cls.custom_lib_paths == []

    def test_wsl_symbols_decoupled_from_simulator(self, tmp_path: Path):
        # Fix D regression: on WSL the symbols resolve even when NO LTspice
        # simulator was detected (available is empty). Schematic editing must
        # not be gated on the simulator executable being found.
        symdir = tmp_path / "wslsyms"
        symdir.mkdir()
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=True),
            patch("ltspice_mcp.lib.wsl.get_ltspice_lib_paths", return_value=[str(symdir)]),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = []
            _configure_asc_editor(cfg, available={})  # empty: no simulator at all
            assert str(symdir) in mock_cls.custom_lib_paths


@pytest.mark.asyncio
class TestServerDispatch:
    """Test list_tools / call_tool / list_resources / read_resource via patched server."""

    async def test_list_tools(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            tools = await list_tools()  # type: ignore[call-arg]
            assert len(tools) > 0

    async def test_call_unknown_tool(self, state_no_sim: SessionState):
        with (
            patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)),
            pytest.raises(ValueError, match="Unknown tool"),
        ):
            await call_tool("ltspice_nonexistent", {})

    async def test_call_validation_error(self, state_no_sim: SessionState):
        with (
            patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)),
            pytest.raises(ValueError, match="Invalid arguments"),
        ):
            await call_tool("create_netlist", {"missing": "field"})

    async def test_call_path_security_error(self, state_no_sim: SessionState):
        with (
            patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)),
            pytest.raises(PathSecurityError, match="Allowed paths"),
        ):
            await call_tool("read_circuit", {"path": "/etc/passwd"})

    async def test_call_ltspice_error_with_hint(self, state_no_sim: SessionState):
        with (
            patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)),
            pytest.raises(NetlistError) as excinfo,
        ):
            await call_tool("read_circuit", {"path": "missing.cir"})
        msg = str(excinfo.value)
        assert "read_circuit" in msg or "list_components" in msg

    async def test_list_resources(self, state_no_sim: SessionState):
        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            resources = await list_resources()  # type: ignore[call-arg]
            assert len(resources) > 0

    async def test_read_resource_invalid_uri(self, state_no_sim: SessionState):
        from pydantic import AnyUrl

        with (
            patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)),
            pytest.raises(ValueError, match="Unknown"),
        ):
            await read_resource(AnyUrl("ltspice://nonexistent"))

    async def test_read_resource_valid(self, state_no_sim: SessionState):
        from pydantic import AnyUrl

        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            result = await read_resource(AnyUrl("ltspice://config"))
            result = list(result)
            assert len(result) > 0

    async def test_error_with_suggestions_returns_structured_result(
        self, state_no_sim: SessionState, tmp_path
    ):
        """LibraryError with suggestions should surface as isError=True + structuredContent.

        ``model_info`` was folded into ``find_model``, so this exercises the
        same fuzzy-match suggestion path through ``find_model`` instead.
        """
        from mcp import types as mcp_types

        lib = state_no_sim.working_dir / "mini.lib"
        lib.write_text(".MODEL 2N2222 NPN(BF=200)\n")
        state_no_sim.libraries.load_library(lib)

        with patch("ltspice_mcp.server.server", _FakeServer(state_no_sim)):
            result = await call_tool("find_model", {"name": "2N2223"})
        assert isinstance(result, mcp_types.CallToolResult)
        # find_model returns success with fuzzy matches rather than an error
        # — assert the candidate is still surfaced.
        assert result.isError is False
        assert result.structuredContent is not None
        names = [r["name"] for r in result.structuredContent["results"]]
        assert "2N2222" in names
