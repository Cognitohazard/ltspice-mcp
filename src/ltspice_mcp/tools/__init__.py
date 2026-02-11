"""Tool module collection for ltspice-mcp server.

Each tool module exports:
- TOOL_DEFS: list[types.Tool] - MCP tool definitions
- TOOL_HANDLERS: dict[str, object] - Mapping from tool name to handler function

The server iterates ALL_MODULES to build the complete tool dispatch table.
"""

from . import advanced, analysis, circuit, library, simulation, status, visualization

ALL_MODULES = [circuit, simulation, analysis, visualization, advanced, library, status]
