"""MCP server instance with lifespan management and tool dispatch."""

import logging
import sys
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import NamedTuple

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from pydantic import AnyUrl, ValidationError

from ltspice_mcp import errors as _err
from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.errors import LTSpiceMCPError, PathSecurityError
from ltspice_mcp.lib.mcp_logging import mcp_log, set_log_fn, set_progress_fn
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.resources import (
    get_resource_templates,
    get_static_resources,
    handle_read_resource,
)
from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


def _get_state(server_: Server) -> SessionState:
    """Extract session state from the lifespan context."""
    try:
        return server_.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
        raise RuntimeError(f"Session state not available: {e}") from e


def _configure_asc_editor(config: ServerConfig, available: dict) -> bool:
    """Configure AscEditor library paths for .asc schematic support.

    Handles four platform scenarios:
    1. Windows native — spicelib auto-detects LTspice lib paths
    2. WSL + LTspice on Windows — resolve via Windows %LOCALAPPDATA%
    3. WSL + other simulator (no LTspice) — no .asc support
    4. Linux native (no LTspice) — no .asc support

    Users can override auto-detection via config.symbol_paths or
    LTSPICE_MCP_SYMBOL_PATHS env var for non-standard installs.

    Returns:
        True if AscEditor is usable (symbol library paths found).
    """
    from spicelib.editor.asc_editor import AscEditor

    # 1. Explicit config override takes priority on all platforms
    if config.symbol_paths:
        valid = [str(p) for p in config.symbol_paths if p.is_dir()]
        if valid:
            AscEditor.custom_lib_paths = valid
            logger.info(f"AscEditor symbol paths from config: {valid}")
            return True
        logger.warning(f"Configured symbol_paths do not exist: {config.symbol_paths}")

    # 2. No LTspice detected → no .asc support (Linux native, WSL + ngspice only)
    ltspice_cls = available.get("ltspice")
    if ltspice_cls is None:
        logger.info("No LTspice detected — .asc schematic editing disabled")
        return False

    # 3. WSL + LTspice on Windows — spicelib can't auto-detect via /mnt/c/
    from ltspice_mcp.lib.wsl import get_ltspice_lib_paths, is_wsl

    if is_wsl():
        lib_paths = get_ltspice_lib_paths()
        if lib_paths:
            AscEditor.custom_lib_paths = lib_paths
            logger.info(f"AscEditor WSL library paths: {lib_paths}")
            return True
        logger.warning(
            "LTspice detected but symbol library not found on WSL. "
            "Set [schematic] symbol_paths in ltspice-mcp.toml or "
            "LTSPICE_MCP_SYMBOL_PATHS env var."
        )
        return False

    # 4. Windows native (or Linux with Wine) — spicelib handles it
    try:
        AscEditor.prepare_for_simulator(ltspice_cls)
        if AscEditor.simulator_lib_paths or AscEditor.custom_lib_paths:
            logger.info("AscEditor configured via prepare_for_simulator()")
            return True
        logger.warning("prepare_for_simulator() found no library paths")
        return False
    except Exception as e:
        logger.warning(f"AscEditor prepare_for_simulator failed: {e}")
        return False


class _ErrorHint(NamedTuple):
    """Profile-aware error hint: full references MCP tools, agentic gives direct guidance."""

    full: str
    agentic: str


# Error type → profile-aware hint appended to error messages.
# PathSecurityError is handled separately (needs dynamic allowed_paths).
_ERROR_HINTS: dict[type[LTSpiceMCPError], _ErrorHint] = {
    _err.MissingModelError: _ErrorHint(
        full=(
            "Try ltspice_search_library to find the model, or "
            "ltspice_load_library to load a library file containing it."
        ),
        agentic=(
            "Try ltspice_search_library to find the model, then add "
            "the .lib directive to the netlist manually."
        ),
    ),
    _err.ConvergenceError: _ErrorHint(
        full=(
            "Suggestions:\n"
            "  - Add .OPTIONS (e.g., .OPTIONS reltol=0.003 or .OPTIONS method=gear)\n"
            "  - Use ltspice_edit_directive to add a .OPTIONS directive\n"
            "  - Check component values for very large/small ratios"
        ),
        agentic=(
            "Suggestions:\n"
            "  - Add a .OPTIONS directive to the netlist "
            "(e.g., .OPTIONS reltol=0.003 or .OPTIONS method=gear)\n"
            "  - Check component values for very large/small ratios"
        ),
    ),
    _err.SingularMatrixError: _ErrorHint(
        full=(
            "This usually means a floating node or short circuit.\n"
            "Use ltspice_read_circuit to inspect the netlist for connectivity issues."
        ),
        agentic=(
            "This usually means a floating node or short circuit.\n"
            "Inspect the netlist for connectivity issues."
        ),
    ),
    _err.SimulationError: _ErrorHint(
        full="Use ltspice_get_server_status to verify simulator availability.",
        agentic="Use ltspice_get_server_status to verify simulator availability.",
    ),
    _err.NetlistError: _ErrorHint(
        full=(
            "Use ltspice_read_circuit to inspect the file, or "
            "ltspice_list_components to verify component references."
        ),
        agentic=(
            "Inspect the netlist file directly, or use "
            "ltspice_list_components to verify component references."
        ),
    ),
    _err.ResultError: _ErrorHint(
        full=(
            "Verify the simulation completed successfully with ltspice_check_job, "
            "and check signal names with ltspice_get_simulation_summary."
        ),
        agentic=(
            "Verify the simulation completed successfully with ltspice_check_job, "
            "and check signal names with ltspice_get_simulation_summary."
        ),
    ),
    _err.LibraryError: _ErrorHint(
        full=(
            "Use ltspice_list_libraries to see loaded libraries, or "
            "ltspice_load_library to load a new one."
        ),
        agentic=(
            "Use ltspice_search_library to find models, or add "
            ".lib directives to the netlist manually."
        ),
    ),
}


def _get_error_hint(err_type: type[LTSpiceMCPError], profile: str) -> str | None:
    """Get the appropriate error hint for the active tool profile."""
    hint = _ERROR_HINTS.get(err_type)
    if hint is None:
        return None
    return hint.agentic if profile == "agentic" else hint.full


@asynccontextmanager
async def server_lifespan(server: Server) -> AsyncIterator[dict]:
    """Initialize session state on startup, clean up on shutdown.

    Loads configuration, sets up logging, detects simulators, creates session state.
    Logs a verbose startup summary to stderr for diagnostics.

    Yields:
        dict containing "state" key with SessionState instance

    Raises:
        Various exceptions during config/simulator setup (allowed to propagate)
    """
    # 1. Load config (generates default TOML if missing)
    # Config path resolution is handled by ServerConfig.load():
    #   --config CLI arg sets $LTSPICE_MCP_CONFIG → env var → CWD fallback
    config = ServerConfig.load()
    config_file = config.config_path

    if config_file.exists():
        config_source = str(config_file)
    else:
        # Generate default config in CWD
        config_file = Path.cwd() / "ltspice-mcp.toml"
        generate_default_config(config_file)
        config_source = f"{config_file} (generated)"

    # 2. Setup logging to stderr
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,  # Override any existing config
    )
    logger = logging.getLogger("ltspice_mcp.server")

    # 3. Detect simulators (pass config so simulator_exe override is applied)
    available = detect_simulators(config)

    # 4. Configure AscEditor library paths for .asc schematic support
    asc_available = _configure_asc_editor(config, available)

    # 5. Create session state
    state = SessionState.create(config, available)
    state.asc_editor_available = asc_available

    # 6. Log verbose startup summary
    logger.info("=== LTSpice MCP Server Starting ===")
    logger.info("Server name: ltspice-mcp")
    logger.info(f"Config source: {config_source}")
    logger.info(f"Working directory: {state.working_dir}")
    logger.info(f"Tool profile: {config.tool_profile} ({len(state.tool_defs)} tools)")
    logger.info(f"Log level: {config.log_level}")

    logger.info("Detected simulators:")
    if available:
        for name, cls in available.items():
            is_default = cls == state.default_simulator
            default_marker = " (default)" if is_default else ""
            logger.info(f"  - {name}{default_marker}")
            try:
                # Try to get executable path if available
                if hasattr(cls, "spice_exe"):
                    exe_path = (
                        cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                    )
                    logger.info(f"    Executable: {exe_path}")
            except Exception:
                pass
    else:
        logger.warning(
            "No simulators detected. Circuit editing will work but simulation tools will return errors."
        )

    logger.info(
        f"Default simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
    )

    logger.info("Allowed paths (sandbox):")
    for allowed_path in config.allowed_paths:
        logger.info(f"  - {allowed_path.resolve()}")

    logger.info("Startup complete. Server ready for MCP connections.")

    # 7. Yield state to server
    try:
        yield {"state": state}
    finally:
        # 8. Cleanup on shutdown
        await state.shutdown()
        logger.info("Server shutdown complete")


# Create server instance with lifespan
server = Server("ltspice-mcp")
server.lifespan = server_lifespan


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return MCP tools filtered by the active tool profile."""
    return _get_state(server).tool_defs


@server.call_tool()
async def call_tool(name: str, arguments: dict | None):
    """Dispatch tool calls to registered handlers.

    All handlers return types.CallToolResult (the MCP protocol's canonical
    response type). Data-returning tools populate structuredContent.
    """
    state = _get_state(server)

    # Look up handler in profile-filtered dispatch table
    registered = state.tool_dispatch.get(name)
    if registered is None:
        raise ValueError(f"Unknown tool: {name}")

    # Set up MCP protocol logging and progress for this request.
    # Handlers and services call mcp_log() / mcp_progress() which read
    # these ContextVars — no server/session reference needed downstream.
    session = server.request_context.session

    async def _log(level: str, msg: str) -> None:
        await session.send_log_message(level=level, data=msg, logger="ltspice-mcp")  # type: ignore[arg-type]

    set_log_fn(_log)

    # Progress: only set up if client provided a progressToken
    meta = server.request_context.meta
    progress_token = meta.progressToken if meta else None
    if progress_token is not None:

        async def _progress(completed: float, total: float | None = None, msg: str | None = None) -> None:
            await session.send_progress_notification(
                progress_token=progress_token,  # type: ignore[arg-type]
                progress=completed,
                total=total,
                message=msg,
            )

        set_progress_fn(_progress)
    else:
        set_progress_fn(None)

    # Invoke handler — enrich known errors with actionable guidance.
    # Exceptions propagate to the MCP SDK which sets isError=True.
    # Input validation (Pydantic model_validate) is handled by the registry
    # wrapper in _base.py — no need to validate here.
    try:
        return await registered.handler(arguments or {}, state)
    except ValidationError as e:
        raise ValueError(f"Invalid arguments for {name}: {e}") from None
    except PathSecurityError as e:
        await mcp_log("warning", f"Path security violation in {name}: {e}")
        allowed = ", ".join(str(p) for p in state.config.allowed_paths)
        raise PathSecurityError(
            f"{e}\n\nAllowed paths: {allowed}\n"
            f"Use ltspice_get_server_status to see full sandbox configuration."
        ) from None
    except LTSpiceMCPError as e:
        hint = _get_error_hint(type(e), state.config.tool_profile)
        if hint:
            raise type(e)(f"{e}\n\n{hint}") from None
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in tool {name}")
        raise RuntimeError(f"Internal error in {name}. Check server logs for details.") from e


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Return all static MCP resources."""
    return get_static_resources()


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """Return all dynamic MCP resource templates."""
    return get_resource_templates()


@server.read_resource()
async def read_resource(uri: AnyUrl) -> Iterable[ReadResourceContents]:
    """Read a specific resource by URI.

    Dispatches to appropriate handler based on URI scheme and path.
    Converts internal TextResourceContents/BlobResourceContents to the
    SDK's ReadResourceContents format (which uses .content instead of .text).

    Args:
        uri: Resource URI to read (ltspice://...)

    Returns:
        Iterable of ReadResourceContents entries

    Raises:
        ValueError: If URI is unknown or resource not found
    """
    state = _get_state(server)

    try:
        result = handle_read_resource(str(uri), state)
    except LTSpiceMCPError as e:
        raise ValueError(str(e)) from None
    except Exception as e:
        logger.exception(f"Unexpected error reading resource {uri}")
        raise ValueError(f"Internal error reading resource: {type(e).__name__}: {e}") from e

    # Convert from types.TextResourceContents/BlobResourceContents
    # to the SDK's ReadResourceContents (which has .content not .text)
    converted = []
    for item in result.contents:
        if isinstance(item, types.TextResourceContents):
            converted.append(
                ReadResourceContents(
                    content=item.text,
                    mime_type=item.mimeType,
                )
            )
        elif isinstance(item, types.BlobResourceContents):
            converted.append(
                ReadResourceContents(
                    content=item.blob,
                    mime_type=item.mimeType,
                )
            )
    return converted
