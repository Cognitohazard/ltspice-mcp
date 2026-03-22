"""MCP server instance with lifespan management and tool dispatch."""

from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any
import logging
import sys
from pathlib import Path

from mcp.server.lowlevel import Server
from mcp import types

from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.state import SessionState
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.tools import ALL_MODULES
from mcp.server.lowlevel.helper_types import ReadResourceContents
from ltspice_mcp.resources import get_static_resources, get_resource_templates, handle_read_resource
from pydantic import AnyUrl

logger = logging.getLogger(__name__)


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
    from ltspice_mcp.lib.wsl import is_wsl, get_ltspice_lib_paths

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
        # Verify it actually found paths
        if AscEditor.simulator_lib_paths or AscEditor.custom_lib_paths:
            logger.info("AscEditor configured via prepare_for_simulator()")
            return True
        logger.warning("prepare_for_simulator() found no library paths")
        return False
    except Exception as e:
        logger.warning(f"AscEditor prepare_for_simulator failed: {e}")
        return False


# Build unified dispatch table at module level
_DISPATCH: dict[str, Any] = {}
for _mod in ALL_MODULES:
    _DISPATCH.update(_mod.TOOL_HANDLERS)


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
    logger.info(f"Server name: ltspice-mcp")
    logger.info(f"Config source: {config_source}")
    logger.info(f"Working directory: {state.working_dir}")
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
                    exe_path = cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                    logger.info(f"    Executable: {exe_path}")
            except Exception:
                pass
    else:
        logger.warning("No simulators detected. Circuit editing will work but simulation tools will return errors.")

    logger.info(f"Default simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}")

    logger.info("Allowed paths (sandbox):")
    for allowed_path in config.allowed_paths:
        logger.info(f"  - {allowed_path.resolve()}")

    logger.info("Startup complete. Server ready for MCP connections.")

    # 7. Yield state to server
    try:
        yield {"state": state}
    finally:
        # 8. Cleanup on shutdown
        state.shutdown()
        logger.info("Server shutdown complete")


# Create server instance with lifespan
server = Server("ltspice-mcp")
server.lifespan = server_lifespan


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Return all registered MCP tools from all tool modules."""
    tools = []
    for mod in ALL_MODULES:
        tools.extend(mod.TOOL_DEFS)
    return tools


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Dispatch tool calls to registered handlers.

    Args:
        name: Tool name to invoke
        arguments: Tool-specific arguments dict

    Returns:
        List of TextContent responses from the tool handler

    Raises:
        ValueError: If tool name is unknown
        LTSpiceMCPError: Tool-specific errors (returned as error text)
    """
    # Look up handler in dispatch table
    handler = _DISPATCH.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name}")

    # Get session state from lifespan context
    try:
        state = server.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
        raise RuntimeError(f"Session state not available: {e}")

    # Invoke handler — exceptions propagate to the MCP SDK, which
    # wraps them in CallToolResult(isError=True) automatically.
    return await handler(arguments, state)


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
    # Get session state from lifespan context
    try:
        state = server.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
        raise ValueError(f"Internal error: Session state not available ({e})")

    result = await handle_read_resource(str(uri), state)

    # Convert from types.TextResourceContents/BlobResourceContents
    # to the SDK's ReadResourceContents (which has .content not .text)
    converted = []
    for item in result.contents:
        if isinstance(item, types.TextResourceContents):
            converted.append(ReadResourceContents(
                content=item.text,
                mime_type=item.mimeType,
            ))
        elif isinstance(item, types.BlobResourceContents):
            converted.append(ReadResourceContents(
                content=item.blob,
                mime_type=item.mimeType,
            ))
    return converted


