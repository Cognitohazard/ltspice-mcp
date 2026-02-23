"""MCP server instance with lifespan management and tool dispatch."""

from collections.abc import AsyncIterator
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
from ltspice_mcp.errors import LTSpiceMCPError
from ltspice_mcp.resources import get_static_resources, get_resource_templates, handle_read_resource
from ltspice_mcp.prompts import get_prompt_definitions, handle_get_prompt
from pydantic import AnyUrl

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
    config = ServerConfig.load()
    config_file = Path.cwd() / "ltspice-mcp.toml"
    config_source = str(config_file) if config_file.exists() else "defaults only"

    # If no config file exists, generate one
    if not config_file.exists():
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

    # 3. Detect simulators
    available = detect_simulators()

    # 4. Create session state
    state = SessionState.create(config, available)

    # 5. Log verbose startup summary
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

    # 6. Yield state to server
    try:
        yield {"state": state}
    finally:
        # 7. Cleanup on shutdown
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
        error_msg = f"Unknown tool: {name}"
        return [types.TextContent(type="text", text=f"ERROR: {error_msg}")]

    # Get session state from lifespan context
    try:
        state = server.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
        error_msg = f"Internal error: Session state not available ({e})"
        return [types.TextContent(type="text", text=f"ERROR: {error_msg}")]

    # Invoke handler with error handling
    try:
        return await handler(arguments, state)
    except LTSpiceMCPError as e:
        # Known error types - return as simple error text
        return [types.TextContent(type="text", text=f"ERROR: {e}")]
    except ValueError as e:
        # Validation errors
        return [types.TextContent(type="text", text=f"ERROR: {e}")]
    # Let other exceptions propagate - MCP SDK handles them


@server.list_resources()
async def list_resources() -> list[types.Resource]:
    """Return all static MCP resources."""
    return get_static_resources()


@server.list_resource_templates()
async def list_resource_templates() -> list[types.ResourceTemplate]:
    """Return all dynamic MCP resource templates."""
    return get_resource_templates()


@server.read_resource()
async def read_resource(uri: AnyUrl) -> types.ReadResourceResult:
    """Read a specific resource by URI.

    Dispatches to appropriate handler based on URI scheme and path.

    Args:
        uri: Resource URI to read (ltspice://...)

    Returns:
        ReadResourceResult with resource contents

    Raises:
        ValueError: If URI is unknown or resource not found
    """
    # Get session state from lifespan context
    try:
        state = server.request_context.lifespan_context["state"]
    except (AttributeError, KeyError) as e:
        raise ValueError(f"Internal error: Session state not available ({e})")

    return await handle_read_resource(str(uri), state)


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    """Return all available MCP prompts."""
    return get_prompt_definitions()


@server.get_prompt()
async def get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
    """Get a specific prompt by name with arguments.

    Args:
        name: Prompt name to retrieve
        arguments: Prompt-specific arguments dict

    Returns:
        GetPromptResult with prompt messages

    Raises:
        ValueError: If prompt name is unknown
    """
    return await handle_get_prompt(name, arguments)
