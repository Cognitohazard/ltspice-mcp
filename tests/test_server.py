"""Tests for server.py — error hints, asc editor configuration, and dispatch."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from mcp import types as mcp_types
from mcp.shared.exceptions import MCPError

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.engine import configure_asc_editor
from ltspice_mcp.errors import (
    LibraryError,
    LTSpiceMCPError,
    NetlistError,
    PathSecurityError,
)
from ltspice_mcp.server import (
    CONSOLIDATED_INSTRUCTIONS,
    _get_error_hint,
    build_instructions,
    call_tool,
    list_resources,
    list_tools,
    read_resource,
    server,
)
from ltspice_mcp.state import SessionState
from tests.conftest import call_tool_params, fake_request_context, tool_text


def _ctx(state: SessionState):
    """The per-request context a server handler reads its session state from."""
    return fake_request_context(state)


def _read_params(uri: str) -> mcp_types.ReadResourceRequestParams:
    """The ``resources/read`` params a dispatch-level test hands to the handler."""
    return mcp_types.ReadResourceRequestParams(uri=uri)


class TestGetErrorHint:
    def test_known_types_have_hints(self):
        hint = _get_error_hint(NetlistError)
        assert hint is not None
        assert "verify_circuit" in hint
        assert _get_error_hint(LibraryError) is not None

    def test_unknown_returns_none(self):
        class FakeErr(LTSpiceMCPError):
            pass

        assert _get_error_hint(FakeErr) is None


class TestServerInstructions:
    def test_instructions_forwarded_to_init_options(self):
        # The block must reach the client at the MCP initialize handshake.
        # The options are built per request off the server's live instructions,
        # which the lifespan rewrites to name the detected simulators — so
        # compare against that attribute, not against the static default a
        # process that never booted still carries.
        opts = server.create_initialization_options()
        assert opts.instructions == server.instructions
        assert opts.instructions
        assert CONSOLIDATED_INSTRUCTIONS in opts.instructions

    def test_instructions_cover_key_workflow_guidance(self):
        text = CONSOLIDATED_INSTRUCTIONS
        # deck-authoring default + the three planes + the result-trust tail
        assert "deck" in text.lower()
        assert "run_experiments" in text and "analyze_results" in text
        assert "edit_schematic" in text
        assert "status completed and still hold a degenerate result" in text
        # must name no tool the surface does not expose
        for dead in ("run_simulation", "check_job", "bode_metrics", "create_netlist"):
            assert dead not in text


class _LT:
    pass


class _NG:
    pass


class TestBuildInstructions:
    """The runtime-prepended line must name the actually-detected simulators."""

    def test_includes_static_body(self):
        assert CONSOLIDATED_INSTRUCTIONS in build_instructions({"ngspice": _NG}, _NG)

    def test_none_detected(self):
        text = build_instructions({}, None)
        assert "No SPICE simulator detected" in text
        # actionable, not a dead end: how to get a simulator + the restart caveat
        assert "ngspice" in text
        assert "restart" in text

    def test_ngspice_only_notes_ltspice_absence(self):
        text = build_instructions({"ngspice": _NG}, _NG)
        assert "ngspice" in text
        assert "LTspice not detected" in text
        # Accurate: .asc editing depends on LTspice symbol files, not the
        # executable — don't over-claim a flat "unavailable".
        assert ".asc" in text and "symbol" in text
        assert "(default)" not in text  # no default marker for a single engine

    def test_ltspice_only(self):
        text = build_instructions({"ltspice": _LT}, _LT)
        assert "LTspice" in text
        assert "LTspice not detected" not in text

    def test_both_marks_default(self):
        text = build_instructions({"ltspice": _LT, "ngspice": _NG}, _LT)
        # Both engines named, and the default marker on the one that is it.
        assert "LTspice (default)" in text
        assert "ngspice" in text
        assert "LTspice not detected" not in text

    def test_the_instructions_fit_the_client_budget(self):
        """Claude Code truncates server instructions at 2048 chars; the tail
        (the result-trust guidance) must survive under every prefix shape — a
        prefix left out of this list ships silently truncated."""
        from ltspice_mcp.server import _INSTRUCTIONS_BUDGET

        worst_cases = [
            build_instructions({}, None),
            build_instructions({"ngspice": _NG}, _NG),
            build_instructions({"ltspice": _LT, "ngspice": _NG, "qspice": _LT, "xyce": _NG}, _LT),
            # Multiple simulators WITHOUT LTspice: the longest active-line list
            # PLUS the LTspice-not-detected note stack on the same edition — the
            # one combination the three cases above never form, and the branch
            # that shipped truncated in v0.5.0.
            build_instructions({"ngspice": _NG, "qspice": _LT, "xyce": _NG}, _NG),
        ]
        for text in worst_cases:
            assert len(text) <= _INSTRUCTIONS_BUDGET, (
                f"instructions {len(text)} chars > {_INSTRUCTIONS_BUDGET} client truncation budget"
            )
            # The Python API discovery pointer must ride every shape: the
            # instructions are the one surface an agent sees without asking,
            # and an agent that never learns the API exists can never choose it
            # (the Python API has no other advertised definition at handshake time).
            assert "from ltspice_mcp.api import Api" in text, (
                "an instruction shape lost the Python API discovery line"
            )


class TestInstructionHints:
    def test_every_hint_names_only_exposed_tools(self):
        from ltspice_mcp.server import _ERROR_HINTS
        from ltspice_mcp.tools import get_tools

        exposed = {t.name for t in get_tools()[0]}
        # Hints may reference an exposed tool by name; they must never
        # reference a removed one (the shape of the stale-hint bug).
        removed = {
            "check_job",
            "server_status",
            "list_libraries",
            "load_library",
            "read_circuit",
            "list_components",
            "simulation_summary",
            "find_model",
        }
        for err_type, hint in _ERROR_HINTS.items():
            assert isinstance(hint, str) and hint
            assert not (set(hint.replace("(", " ").replace('"', " ").split()) & removed), (
                f"{err_type.__name__} hint names a removed tool: {hint}"
            )
            assert any(tool in hint for tool in exposed), (
                f"{err_type.__name__} hint names no exposed tool: {hint}"
            )


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
            configure_asc_editor(cfg, available={})
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
            configure_asc_editor(cfg, available={})
            assert mock_cls.custom_lib_paths == []

    def test_non_wsl_no_ltspice_disabled(self, tmp_path: Path):
        cfg = ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path])
        cfg.symbol_paths = []
        with (
            patch("ltspice_mcp.lib.wsl.is_wsl", return_value=False),
            patch("spicelib.editor.asc_editor.AscEditor") as mock_cls,
        ):
            mock_cls.custom_lib_paths = []
            configure_asc_editor(cfg, available={})
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
            configure_asc_editor(cfg, available={"ltspice": FakeLT})
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
            configure_asc_editor(cfg, available={})
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
            configure_asc_editor(cfg, available={})  # empty: no simulator at all
            assert str(symdir) in mock_cls.custom_lib_paths


@pytest.mark.asyncio
class TestServerDispatch:
    """Test list_tools / call_tool / list_resources / read_resource via patched server."""

    async def test_list_tools(self, state_no_sim: SessionState):
        result = await list_tools(_ctx(state_no_sim), None)
        assert len(result.tools) > 0

    async def test_call_unknown_tool(self, state_no_sim: SessionState):
        result = await call_tool(_ctx(state_no_sim), call_tool_params("ltspice_nonexistent", {}))
        assert result.is_error
        assert "Unknown tool" in tool_text(result)

    async def test_call_removed_tool_is_unknown(self, state_no_sim: SessionState):
        """A 0.5-era tool name is gone from the registry entirely — the wire
        answers 'Unknown tool', the same as any other unknown name (migration
        guidance lives in the config warning and the docs)."""
        result = await call_tool(
            _ctx(state_no_sim), call_tool_params("run_simulation", {"netlist": "x.cir"})
        )
        assert result.is_error
        assert "Unknown tool" in tool_text(result)

    async def test_call_validation_error(self, state_no_sim: SessionState):
        result = await call_tool(
            _ctx(state_no_sim), call_tool_params("run_experiments", {"missing": "field"})
        )
        assert result.is_error
        assert "Invalid arguments" in tool_text(result)

    async def test_call_path_security_error(self, state_no_sim: SessionState):
        result = await call_tool(
            _ctx(state_no_sim), call_tool_params("plot_waveform", {"raw_file": "/etc/passwd"})
        )
        assert result.is_error
        msg = tool_text(result)
        assert "Allowed paths" in msg
        # The agent can't self-widen the sandbox, so the message must name the
        # knob AND the human-escalation / move-the-file fallback.
        assert "LTSPICE_MCP_ALLOWED_PATHS" in msg
        assert "ask the user" in msg

    async def test_call_ltspice_error_with_hint(self, state_no_sim: SessionState):
        result = await call_tool(
            _ctx(state_no_sim), call_tool_params("plot_waveform", {"raw_file": "missing.raw"})
        )
        assert result.is_error
        msg = tool_text(result)
        # The appended recovery hint names only exposed tools.
        assert "jobs" in msg or "analyze_results" in msg

    async def test_list_resources(self, state_no_sim: SessionState):
        result = await list_resources(_ctx(state_no_sim), None)
        assert len(result.resources) > 0

    async def test_read_resource_path_security_enriched(self, state_no_sim: SessionState):
        # The resource-read boundary must enrich a sandbox rejection with the
        # same recovery guidance as the tool-call path (covers spice://netlists
        # /{outside} etc.), not leak a bare "outside allowed directories".
        boom = PathSecurityError("Path /etc/x.cir is outside allowed directories [/work]")
        with (
            patch("ltspice_mcp.server.handle_read_resource", side_effect=boom),
            pytest.raises(MCPError) as excinfo,
        ):
            await read_resource(_ctx(state_no_sim), _read_params("spice://netlists/x.cir"))
        msg = excinfo.value.message
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        assert "outside allowed directories" in msg
        assert "LTSPICE_MCP_ALLOWED_PATHS" in msg
        assert "ask the user" in msg

    async def test_read_resource_invalid_uri(self, state_no_sim: SessionState):
        # 2026-07-28 dropped the resource-not-found code; an unserved URI is an
        # invalid parameter, which is what a client keys its recovery on.
        with pytest.raises(MCPError) as excinfo:
            await read_resource(_ctx(state_no_sim), _read_params("spice://nonexistent"))
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        assert "Unknown" in excinfo.value.message

    async def test_read_resource_valid(self, state_no_sim: SessionState):
        result = await read_resource(_ctx(state_no_sim), _read_params("spice://config"))
        assert len(result.contents) > 0

    async def test_error_with_suggestions_returns_structured_result(
        self, state_no_sim: SessionState, tmp_path
    ):
        """LibraryError with suggestions should surface as is_error=True + structuredContent.

        The fuzzy-match suggestion path now lives behind inspect's model
        search query.
        """
        lib = state_no_sim.working_dir / "mini.lib"
        lib.write_text(".MODEL 2N2222 NPN(BF=200)\n")
        state_no_sim.libraries.load_library(lib)

        result = await call_tool(
            _ctx(state_no_sim),
            call_tool_params(
                "inspect",
                {"queries": [{"kind": "model", "mode": "search", "query": "2N2223"}]},
            ),
        )
        assert isinstance(result, mcp_types.CallToolResult)
        # The model search returns success with fuzzy matches rather than an
        # error — assert the near-miss candidate is still surfaced.
        assert result.is_error is False
        assert result.structured_content is not None
        item = result.structured_content["results"][0]
        assert item["ok"] is True
        assert "2N2222" in str(item["data"])


class TestClientLogLevelFilter:
    def test_no_level_set_sends_everything(self):
        from ltspice_mcp.server import _below_client_log_level

        assert _below_client_log_level("debug", None) is False
        assert _below_client_log_level("emergency", None) is False

    def test_below_floor_filtered_at_and_above_sent(self):
        from ltspice_mcp.server import _below_client_log_level

        assert _below_client_log_level("debug", "warning") is True
        assert _below_client_log_level("info", "warning") is True
        assert _below_client_log_level("warning", "warning") is False
        assert _below_client_log_level("error", "warning") is False

    def test_unknown_levels_never_filtered(self):
        from ltspice_mcp.server import _below_client_log_level

        assert _below_client_log_level("verbose", "warning") is False
        assert _below_client_log_level("info", "chatty") is False

    def test_set_level_handler_registered_declares_capability(self):
        # Registering the logging/setLevel handler is what makes the SDK
        # declare the logging capability in the initialize result — without
        # it, spec-conforming clients drop notifications/message entirely.
        from ltspice_mcp.server import server

        assert server.get_request_handler("logging/setLevel") is not None
        assert server.get_capabilities().logging is not None


class TestStderrIsQuietByDefault:
    """The server's stderr is the caller's stderr on the in-process and
    per-script doors. A startup banner there gets answered with a blanket
    2>/dev/null, which then hides the tracebacks that mattered — measured, that
    cost two turns in one session. So INFO is opt-in, not the default."""

    @staticmethod
    def _emit(level: str | None) -> str:
        import logging as stdlib_logging

        from ltspice_mcp.server import _configure_server_logging

        config = ServerConfig() if level is None else ServerConfig(log_level=level)
        stream = io.StringIO()
        _configure_server_logging(config)
        for handler in stdlib_logging.getLogger().handlers:
            if isinstance(handler, stdlib_logging.StreamHandler):
                handler.setStream(stream)  # type: ignore[attr-defined]
        stdlib_logging.getLogger("ltspice_mcp.server").info("=== LTSpice MCP Server Starting ===")
        stdlib_logging.getLogger("ltspice_mcp.server").warning("a real problem")
        return stream.getvalue()

    def test_the_default_config_keeps_info_off_stderr(self):
        emitted = self._emit(None)
        assert "Server Starting" not in emitted
        assert "a real problem" in emitted

    def test_an_explicit_info_level_brings_the_banner_back(self):
        emitted = self._emit("INFO")
        assert "Server Starting" in emitted


class TestToolAnnotationHonesty:
    def test_mutating_tools_not_marked_idempotent(self):
        # edit_schematic mutates a sheet (revision-guarded, but each call
        # advances it) and plot_waveform mints a fresh artifact — an
        # auto-retrying client must not treat either as idempotent. The
        # request_id-keyed and read-only tools ARE idempotent and say so.
        from ltspice_mcp.tools._base import registry

        dispatch = registry.get_tools()[1]
        expected = {
            "edit_schematic": False,
            "plot_waveform": False,
            # Without a caller request_id the same arguments start new work.
            "run_experiments": False,
            "jobs": True,
            "analyze_results": True,
            "verify_circuit": True,
            "inspect": True,
        }
        assert set(expected) == set(dispatch)
        for tool_name, idempotent in expected.items():
            annotations = dispatch[tool_name].definition.annotations
            assert annotations is not None
            assert annotations.idempotent_hint is idempotent, tool_name
