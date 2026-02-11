"""Server status and diagnostics tools."""

from mcp import types

from ltspice_mcp.state import SessionState


async def handle_get_server_status(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get comprehensive server status information.

    Returns detailed information about detected simulators, configuration,
    sandbox settings, and runtime state. This tool allows the LLM to check
    what capabilities are available before attempting operations.

    Args:
        arguments: Empty dict (no parameters required)
        state: Current session state

    Returns:
        List containing a single TextContent with formatted status information
    """
    lines = ["=== LTSpice MCP Server Status ===\n"]

    # Detected simulators
    lines.append("Simulators:")
    if state.available_simulators:
        for name, cls in state.available_simulators.items():
            is_default = cls == state.default_simulator
            default_marker = " (default)" if is_default else ""
            lines.append(f"  - {name}: available{default_marker}")
            try:
                # Try to get executable path if available
                if hasattr(cls, "spice_exe"):
                    exe_path = cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                    lines.append(f"    Executable: {exe_path}")
            except Exception:
                pass
    else:
        lines.append("  No simulators detected (server running in degraded mode)")

    # Default simulator
    lines.append(f"\nDefault simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}")

    # Configuration
    lines.append("\nConfiguration:")
    lines.append(f"  Working directory: {state.working_dir}")
    lines.append(f"  Max parallel simulations: {state.config.max_parallel_sims}")
    lines.append(f"  Default timeout: {state.config.default_timeout}s")
    lines.append(f"  Max points returned: {state.config.max_points_returned}")
    lines.append(f"  Log level: {state.config.log_level}")

    # Security sandbox
    lines.append("\nSecurity (Sandbox):")
    lines.append("  Allowed paths:")
    for allowed_path in state.config.allowed_paths:
        lines.append(f"    - {allowed_path}")

    # Config source
    config_file = state.working_dir / "ltspice-mcp.toml"
    if config_file.exists():
        lines.append(f"\n  Config file: {config_file}")
    else:
        lines.append("\n  Config file: Not found (using defaults)")

    # Runtime state
    lines.append("\nRuntime State:")
    lines.append(f"  Active jobs: {len(state.jobs)}")
    lines.append(f"  Cached editors: {len(state.editors)}")
    lines.append(f"  Cached results: {len(state.results)}")
    lines.append(f"  Loaded libraries: {len(state.libraries)}")

    status_text = "\n".join(lines)
    return [types.TextContent(type="text", text=status_text)]


# Tool definitions
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="get_server_status",
        description=(
            "Get comprehensive server status including detected simulators, "
            "configuration settings, security sandbox paths, and runtime state. "
            "Use this to check what capabilities are available before attempting operations."
        ),
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    )
]

# Tool handler mapping
TOOL_HANDLERS: dict[str, object] = {
    "get_server_status": handle_get_server_status,
}
