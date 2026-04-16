"""Server status and diagnostics tools."""

from typing import Literal

from pydantic import Field

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    RO_ANNOTATIONS,
    ToolInput,
    format_response,
    registry,
)


class GetServerStatusInput(ToolInput):
    """Inputs for the server status tool."""

    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_get_server_status",
    description=(
        "Get comprehensive server status including detected simulators, "
        "configuration settings, security sandbox paths, and runtime state. "
        "Use this to check what capabilities are available before attempting operations."
    ),
    input_model=GetServerStatusInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "simulators": {"type": "object"},
            "default_simulator": {"type": ["string", "null"]},
            "tool_profile": {"type": "string"},
            "tool_count": {"type": "integer"},
            "configuration": {"type": "object"},
            "allowed_paths": {"type": "array", "items": {"type": "string"}},
            "runtime": {"type": "object"},
        },
    },
)
async def handle_get_server_status(arguments: GetServerStatusInput, state: SessionState):
    """Get comprehensive server status information."""
    fmt = arguments.format

    lines = ["=== LTSpice MCP Server Status ===\n"]

    simulators_data = {}

    lines.append("Simulators:")
    if state.available_simulators:
        for name, cls in state.available_simulators.items():
            is_default = cls == state.default_simulator
            default_marker = " (default)" if is_default else ""
            lines.append(f"  - {name}: available{default_marker}")
            sim_info: dict = {"available": True, "default": is_default}
            try:
                if hasattr(cls, "spice_exe"):
                    exe_path = (
                        cls.spice_exe[0] if isinstance(cls.spice_exe, list) else cls.spice_exe
                    )
                    lines.append(f"    Executable: {exe_path}")
                    sim_info["executable"] = str(exe_path)
            except Exception:
                pass
            simulators_data[name] = sim_info
    else:
        lines.append("  No simulators detected (server running in degraded mode)")

    lines.append(
        f"\nDefault simulator: {state.default_simulator.__name__ if state.default_simulator else 'None'}"
    )

    lines.append("\nConfiguration:")
    lines.append(f"  Tool profile: {state.config.tool_profile}")
    lines.append(f"  Tools exposed: {len(state.tool_defs)}")
    lines.append(f"  Working directory: {state.working_dir}")
    lines.append(f"  Max parallel simulations: {state.config.max_parallel_sims}")
    lines.append(f"  Default timeout: {state.config.default_timeout}s")
    lines.append(f"  Max points returned: {state.config.max_points_returned}")
    lines.append(f"  Log level: {state.config.log_level}")

    allowed_paths_list = [str(p) for p in state.config.allowed_paths]
    lines.append("\nSecurity (Sandbox):")
    lines.append("  Allowed paths:")
    for allowed_path in state.config.allowed_paths:
        lines.append(f"    - {allowed_path}")

    config_file = state.working_dir / "ltspice-mcp.toml"
    if config_file.exists():
        lines.append(f"\n  Config file: {config_file}")
    else:
        lines.append("\n  Config file: Not found (using defaults)")

    lines.append("\nRuntime State:")
    lines.append(f"  Active jobs: {len(state.jobs)}")
    lines.append(f"  Cached editors: {len(state.editors)}")
    lines.append(f"  Cached results: {len(state.results)}")
    lines.append(f"  Loaded libraries: {len(state.libraries)}")

    data = {
        "simulators": simulators_data,
        "default_simulator": state.default_simulator.__name__ if state.default_simulator else None,
        "tool_profile": state.config.tool_profile,
        "tool_count": len(state.tool_defs),
        "configuration": {
            "working_directory": str(state.working_dir),
            "max_parallel_sims": state.config.max_parallel_sims,
            "default_timeout": state.config.default_timeout,
            "max_points_returned": state.config.max_points_returned,
            "log_level": state.config.log_level,
        },
        "allowed_paths": allowed_paths_list,
        "runtime": {
            "active_jobs": len(state.jobs),
            "cached_editors": len(state.editors),
            "cached_results": len(state.results),
            "loaded_libraries": len(state.libraries),
        },
    }
    return format_response("\n".join(lines), data, fmt)
