"""Tool module collection for ltspice-mcp server.

Each tool module exports:
- TOOL_DEFS: list[types.Tool] - MCP tool definitions
- TOOL_HANDLERS: dict[str, object] - Mapping from tool name to handler function

The server iterates ALL_MODULES to build the complete tool dispatch table.
Tool profiles filter which tools are exposed to the client.
"""

from typing import Any

from mcp import types

from . import advanced, analysis, circuit, library, simulation, status

ALL_MODULES = [circuit, simulation, analysis, advanced, library, status]

# Tools kept in "agentic" profile — binary parsing, orchestration, discovery.
# Everything else (netlist editing, sweep/MC config, niche schematic ops,
# library session management) is handled natively by capable LLM agents.
AGENTIC_TOOLS: frozenset[str] = frozenset(
    {
        # Simulation lifecycle
        "ltspice_run_simulation",
        "ltspice_check_job",
        "ltspice_cancel_job",
        # Analysis (binary .raw parsing, context compression)
        "ltspice_get_signal_stats",
        "ltspice_query_value",
        "ltspice_get_operating_point",
        "ltspice_get_simulation_summary",
        "ltspice_get_measurements",
        # Batch orchestration
        "ltspice_run_sweep",
        "ltspice_run_montecarlo",
        "ltspice_get_batch_results",
        # Schematic (AscEditor dependency)
        "ltspice_list_components",
        "ltspice_export_netlist",
        # Library (context compression on large dirs)
        "ltspice_search_library",
        "ltspice_get_model_info",
        # Server discovery
        "ltspice_get_server_status",
    }
)


def get_tools_for_profile(
    profile: str,
) -> tuple[list[types.Tool], dict[str, Any]]:
    """Build tool definitions and dispatch table filtered by profile.

    Args:
        profile: "full" (all tools) or "agentic" (subset for capable agents).

    Returns:
        (tool_defs, dispatch) — list of Tool objects and name→handler dict.
    """
    allowed = AGENTIC_TOOLS if profile == "agentic" else None
    defs: list[types.Tool] = []
    handlers: dict[str, Any] = {}
    for mod in ALL_MODULES:
        for td in mod.TOOL_DEFS:
            if allowed is None or td.name in allowed:
                defs.append(td)
        for name, handler in mod.TOOL_HANDLERS.items():
            if allowed is None or name in allowed:
                handlers[name] = handler
    return defs, handlers
