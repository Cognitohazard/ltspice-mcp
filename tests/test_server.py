"""Tests for server.py — error hints, asc editor configuration, and dispatch."""

import io
import re
from dataclasses import replace
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
from tests.conftest import (
    REGISTERED_TOOLS,
    SERVED_WITHOUT_RUN_CODE,
    call_tool_params,
    fake_request_context,
    removed_tool_names,
    tool_text,
)


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
        """The block must reach the client at the MCP initialize handshake.

        The lifespan rewrites ``server.instructions`` after the Server was
        constructed, to name the simulators it detected. So the options have to
        be built off the live attribute every time; a snapshot taken at
        construction would hand every client the pre-boot default. Rewrite the
        attribute the way the lifespan does and check the change comes through
        — comparing the options against the attribute as it stands proves
        nothing, because before a boot the two agree either way.
        """
        original = server.instructions
        rewritten = f"{original}\n\nDetected simulators: none."
        try:
            server.instructions = rewritten
            opts = server.create_initialization_options()
        finally:
            server.instructions = original
        assert opts.instructions == rewritten
        assert CONSOLIDATED_INSTRUCTIONS in (opts.instructions or "")
        assert server.create_initialization_options().instructions == original

    def test_instructions_name_no_removed_tool(self):
        # What the instructions must name (every envelope tool, the result-trust
        # warning) is pinned in test_guide_delivery.py; this is the other half.
        named = sorted(
            name
            for name in removed_tool_names()
            if re.search(rf"\b{re.escape(name)}\b", CONSOLIDATED_INSTRUCTIONS)
        )
        assert not named, f"the instructions name removed tools: {named}"


def _says(text: str, *facts: str) -> bool:
    """True when ``text`` states every fact, ignoring case and line wrapping."""
    flat = " ".join(text.split()).lower()
    return all(" ".join(fact.split()).lower() in flat for fact in facts)


# The Python API's import line: the discovery route every instruction edition
# carries, whatever the sentence around it says.
_API_IMPORT = re.compile(r"from\s+ltspice_mcp\.api\s+import\s+Api\b")


class _LT:
    pass


class _NG:
    pass


class TestBuildInstructions:
    """The runtime-prepended line must name the actually-detected simulators."""

    def test_includes_static_body(self):
        assert CONSOLIDATED_INSTRUCTIONS in build_instructions({"ngspice": _NG}, _NG)

    def test_none_detected(self):
        from ltspice_mcp.lib.simulator import no_simulator_message

        text = build_instructions({}, None)
        assert no_simulator_message(short=True) in text
        # actionable, not a dead end: how to get a simulator + the restart caveat
        assert _says(text, "ngspice", "restart")

    def test_ngspice_only_notes_ltspice_absence(self):
        text = build_instructions({"ngspice": _NG}, _NG)
        assert _says(text, "ngspice", "LTspice not detected")
        # Accurate: .asc editing depends on LTspice symbol files, not the
        # executable — don't over-claim a flat "unavailable".
        assert _says(text, ".asc", "symbol")
        assert not _says(text, "(default)")  # no default marker for a single engine

    def test_ltspice_only(self):
        text = build_instructions({"ltspice": _LT}, _LT)
        assert _says(text, "LTspice")
        assert not _says(text, "not detected")

    def test_both_marks_default(self):
        text = build_instructions({"ltspice": _LT, "ngspice": _NG}, _LT)
        # Both engines named, and the default marker on the one that is it.
        assert _says(text, "LTspice (default)", "ngspice")
        assert not _says(text, "ngspice (default)")
        assert not _says(text, "not detected")

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
            # The run_code edition swaps the code-loop clause for a longer one.
            build_instructions(
                {"ngspice": _NG, "qspice": _LT, "xyce": _NG}, _NG, served={"run_code"}
            ),
            build_instructions(
                {"ltspice": _LT, "ngspice": _NG, "qspice": _LT, "xyce": _NG},
                _LT,
                served={"run_code"},
            ),
        ]
        # The no-simulator prefix embeds a platform-specific install hint, and
        # the longest of those is not the one this test happens to run on.
        from ltspice_mcp.lib import simulator as simulator_module

        for wsl, system in (
            (True, "Linux"),
            (False, "Linux"),
            (False, "Darwin"),
            (False, "Windows"),
        ):
            with pytest.MonkeyPatch.context() as mp:
                mp.setattr(simulator_module, "is_wsl", lambda wsl=wsl: wsl)
                mp.setattr(simulator_module.platform, "system", lambda system=system: system)
                worst_cases.append(build_instructions({}, None))
        for text in worst_cases:
            assert len(text) <= _INSTRUCTIONS_BUDGET, (
                f"instructions {len(text)} chars > {_INSTRUCTIONS_BUDGET} client truncation budget"
            )
            # The Python API discovery pointer must ride every shape: the
            # instructions are the one surface an agent sees without asking,
            # and an agent that never learns the API exists can never choose it
            # (the Python API has no other advertised definition at handshake time).
            assert _API_IMPORT.search(text), (
                "an instruction shape lost the Python API discovery line"
            )

    def test_run_code_is_named_only_when_it_is_served(self):
        default = build_instructions({"ltspice": _LT}, _LT)
        silent = build_instructions({"ltspice": _LT}, _LT, served=())
        assert re.search(r"\brun_code\b", default)
        assert not re.search(r"\brun_code\b", silent)
        # Both editions keep the library door, and route trace math to code.
        for text in (default, silent):
            assert _API_IMPORT.search(text)
            assert _says(text, "trace math")


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

    @pytest.mark.parametrize(
        ("run_code", "served"),
        [(True, REGISTERED_TOOLS), (False, SERVED_WITHOUT_RUN_CODE)],
        ids=["run_code-on", "run_code-off"],
    )
    async def test_every_listed_tool_is_callable_and_nothing_else_is(
        self, config: ServerConfig, run_code: bool, served: tuple[str, ...]
    ):
        """What tools/list advertises is exactly what tools/call answers to.

        Each listed name reaches its own handler (a strict input model rejects
        the stray argument and names the tool), and the tool a session turned
        off is neither listed nor callable.
        """
        state = SessionState.create(
            replace(config, run_code=run_code, write_config=False), available={}
        )
        ctx = fake_request_context(state)
        listed = [tool.name for tool in (await list_tools(ctx, None)).tools]
        assert set(listed) == set(served)
        for name in listed:
            result = await call_tool(ctx, call_tool_params(name, {"__not_an_argument__": 1}))
            assert result.is_error
            assert tool_text(result).startswith(f"Invalid arguments for {name}:")
        for name in set(REGISTERED_TOOLS) - set(listed):
            with pytest.raises(MCPError, match=f"Unknown tool: {name}"):
                await call_tool(ctx, call_tool_params(name, {}))

    async def test_call_unknown_tool(self, state_no_sim: SessionState):
        """A name the server does not serve is a protocol error, not a tool
        result: there is no tool to attribute a result to, so the lookup
        failure answers invalid-params the way an unknown resource URI does.
        The message lists the names that do exist."""
        with pytest.raises(MCPError) as excinfo:
            await call_tool(
                fake_request_context(state_no_sim), call_tool_params("ltspice_nonexistent", {})
            )
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        message = excinfo.value.message
        assert "ltspice_nonexistent" in message
        for name in state_no_sim.tool_dispatch:
            assert name in message

    async def test_call_removed_tool_is_unknown(self, state_no_sim: SessionState):
        """A 0.5-era tool name is gone from the registry entirely — the wire
        answers the same invalid-params error as any other unknown name
        (migration guidance lives in the config warning and the docs)."""
        with pytest.raises(MCPError) as excinfo:
            await call_tool(
                fake_request_context(state_no_sim),
                call_tool_params("run_simulation", {"netlist": "x.cir"}),
            )
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        assert "run_simulation" in excinfo.value.message

    async def test_call_validation_error(self, state_no_sim: SessionState):
        result = await call_tool(
            fake_request_context(state_no_sim),
            call_tool_params("run_experiments", {"missing": "field"}),
        )
        assert result.is_error
        assert "Invalid arguments" in tool_text(result)

    async def test_call_path_security_error(self, state_no_sim: SessionState):
        result = await call_tool(
            fake_request_context(state_no_sim),
            call_tool_params("plot_waveform", {"raw_file": "/etc/passwd"}),
        )
        assert result.is_error
        msg = tool_text(result)
        assert "Allowed paths" in msg
        # The message names the config line that widens the sandbox and says the
        # file is re-read on the next call, so the agent can act on it itself.
        assert "[security] allowed_paths" in msg
        assert "next call" in msg
        assert "LTSPICE_MCP_ALLOWED_PATHS" in msg
        # A structured-aware client drops the text channel, so the refusal and
        # its guidance ride structuredContent too.
        data = result.structured_content
        assert data is not None
        assert data["code"] == "path_denied"
        assert "outside allowed directories" in data["error"]
        assert "[security] allowed_paths" in data["hint"]
        assert str(state_no_sim.config.config_path) in data["hint"]

    async def test_call_ltspice_error_with_hint(self, state_no_sim: SessionState):
        result = await call_tool(
            fake_request_context(state_no_sim),
            call_tool_params("plot_waveform", {"raw_file": "missing.raw"}),
        )
        assert result.is_error
        msg = tool_text(result)
        # The appended recovery hint names only exposed tools.
        assert "jobs" in msg or "analyze_results" in msg

    async def test_list_resources(self, state_no_sim: SessionState):
        result = await list_resources(fake_request_context(state_no_sim), None)
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
            await read_resource(
                fake_request_context(state_no_sim), _read_params("spice://netlists/x.cir")
            )
        msg = excinfo.value.message
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        assert "outside allowed directories" in msg
        assert "LTSPICE_MCP_ALLOWED_PATHS" in msg
        assert "[security] allowed_paths" in msg
        assert "next call" in msg

    async def test_read_resource_invalid_uri(self, state_no_sim: SessionState):
        # 2026-07-28 dropped the resource-not-found code; an unserved URI is an
        # invalid parameter, which is what a client keys its recovery on.
        with pytest.raises(MCPError) as excinfo:
            await read_resource(
                fake_request_context(state_no_sim), _read_params("spice://nonexistent")
            )
        assert excinfo.value.code == mcp_types.INVALID_PARAMS
        assert "Unknown" in excinfo.value.message

    async def test_read_resource_crash_is_reported_as_a_server_fault(
        self, state_no_sim: SessionState
    ):
        """An unhandled exception in a resource handler is this server's bug.

        Reported as invalid-params it would tell a client that keys recovery on
        that code to keep trying other URIs, when no URI it can send avoids a
        fault in the router.
        """
        with (
            patch("ltspice_mcp.server.handle_read_resource", side_effect=TypeError("boom")),
            pytest.raises(MCPError) as excinfo,
        ):
            await read_resource(fake_request_context(state_no_sim), _read_params("spice://config"))
        assert excinfo.value.code == mcp_types.INTERNAL_ERROR
        assert "TypeError" in excinfo.value.message

    async def test_read_resource_valid(self, state_no_sim: SessionState):
        result = await read_resource(
            fake_request_context(state_no_sim), _read_params("spice://config")
        )
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
            fake_request_context(state_no_sim),
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


def _capabilities_call() -> mcp_types.CallToolRequestParams:
    return call_tool_params("inspect", {"queries": [{"kind": "capabilities"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("write_config", [True, False])
async def test_the_first_tool_call_writes_a_default_config_unless_switched_off(
    config: ServerConfig, work_dir: Path, write_config: bool
):
    """Switched off, a server leaves the directory it was started in untouched."""
    config_path = work_dir / "ltspice-mcp.toml"
    state = SessionState.create(
        replace(config, config_path=config_path, write_config=write_config), available={}
    )
    result = await call_tool(fake_request_context(state), _capabilities_call())
    assert not result.is_error
    written = [p.name for p in work_dir.iterdir()]  # noqa: ASYNC240
    assert written == (["ltspice-mcp.toml"] if write_config else [])


class TestLoggingCapabilityDropped:
    """The 2026-07-28 revision deprecates the whole logging capability — the
    `logging/setLevel` request, the `logging` capability, and the
    server-to-client `notifications/message` delivery — with no replacement.
    The server serves none of it, so nothing is advertised and nothing is
    sent."""

    def test_no_set_level_handler_and_no_capability(self):
        from ltspice_mcp.server import server

        assert server.get_request_handler("logging/setLevel") is None
        assert server.get_capabilities().logging is None

    def test_no_protocol_log_delivery_module(self):
        # The delivery itself is gone, not just its advertisement: no module
        # for it, and no import of one left behind.
        import importlib

        import ltspice_mcp.server as server_module

        assert not hasattr(server_module, "mcp_log")
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module("ltspice_mcp.lib.mcp_logging")


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
