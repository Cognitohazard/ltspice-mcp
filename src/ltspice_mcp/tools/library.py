"""Component library management tools. (Phase 5)"""

from mcp import types

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import run_sync, safe_path


async def handle_search_library(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Search component libraries by name.

    Args:
        arguments: Contains 'query' (string), 'source' ('user' | 'builtin'),
                   'offset' (int), 'limit' (int)
        state: Session state with library manager

    Returns:
        List with single TextContent containing search results

    Raises:
        LibraryError: Search failed
    """
    query = arguments["query"]
    source = arguments.get("source", "user")
    offset = arguments.get("offset", 0)
    limit = min(arguments.get("limit", 50), 50)  # Cap at 50

    try:
        if source == "user":
            result = await run_sync(
                state.libraries.search_user_libraries, query, offset, limit
            )
        elif source == "builtin":
            result = await run_sync(
                state.libraries.search_builtin_libraries, query, offset, limit
            )
        else:
            raise LibraryError(f"Invalid source: {source}. Must be 'user' or 'builtin'")
    except LibraryError:
        raise
    except Exception as e:
        raise LibraryError(f"Search failed: {e}")

    results = result["results"]
    total = result["total"]

    # Format response
    if not results:
        return [types.TextContent(type="text", text=f"No models found matching '{query}'")]

    lines = [f"Found {total} model(s) matching '{query}'"]
    lines.append(f"Showing {offset + 1}-{offset + len(results)} of {total}")
    lines.append("")

    for r in results:
        lines.append(f"  {r['name']} ({r['type']}) - {r['source_path']}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def handle_get_model_info(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get SPICE model/subcircuit details.

    Args:
        arguments: Contains 'name' (string), 'full' (bool)
        state: Session state with library manager

    Returns:
        List with single TextContent containing model details

    Raises:
        LibraryError: Model not found or lookup failed
    """
    name = arguments["name"]
    full = arguments.get("full", False)

    try:
        info = await run_sync(state.libraries.get_model_info, name, full)
    except Exception as e:
        raise LibraryError(f"Failed to get model info: {e}")

    if info is None:
        raise LibraryError(
            f"Model '{name}' not found in loaded or built-in libraries. "
            "Use search_library to find models."
        )

    # Format response
    lines = [
        f"Model: {info['name']}",
        f"Type: {info['type']}",
        f"Source: {info['source_path']}",
        "",
        "Include directive:",
        f"  {info['include_directive']}",
        "",
    ]

    if info["parameters"]:
        lines.append("Parameters:")
        for param in info["parameters"]:
            lines.append(f"  {param}")
        lines.append("")

    if full and "raw_text" in info:
        lines.append("Full SPICE definition:")
        lines.append(info["raw_text"])

    return [types.TextContent(type="text", text="\n".join(lines))]


async def handle_load_library(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Load a SPICE library file or directory.

    Args:
        arguments: Contains 'path' (string)
        state: Session state with library manager

    Returns:
        List with single TextContent containing load summary

    Raises:
        PathSecurityError: Path outside sandbox
        LibraryError: Load failed
    """
    path = safe_path(arguments["path"], state)

    try:
        summary = await run_sync(state.libraries.load_library, path)
    except LibraryError:
        raise
    except Exception as e:
        raise LibraryError(f"Failed to load library: {e}")

    result = (
        f"Loaded {summary['path']}: "
        f"{summary['models']} models, {summary['subcircuits']} subcircuits "
        f"from {summary['files_loaded']} file(s)"
    )

    return [types.TextContent(type="text", text=result)]


async def handle_unload_library(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Unload a library from the session.

    Args:
        arguments: Contains 'path' (string)
        state: Session state with library manager

    Returns:
        List with single TextContent confirming unload

    Raises:
        PathSecurityError: Path outside sandbox
        LibraryError: Library not loaded
    """
    path = safe_path(arguments["path"], state)

    try:
        result = await run_sync(state.libraries.unload_library, path)
    except Exception as e:
        raise LibraryError(f"Failed to unload library: {e}")

    if not result["removed"]:
        raise LibraryError(f"Library not loaded: {path}")

    return [types.TextContent(type="text", text=f"Unloaded library: {path}")]


async def handle_list_libraries(
    _arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """List all loaded libraries.

    Args:
        _arguments: Empty dict
        state: Session state with library manager

    Returns:
        List with single TextContent containing library paths
    """
    libs = await run_sync(state.libraries.list_libraries)

    if not libs:
        return [types.TextContent(type="text", text="No libraries loaded")]

    lines = [f"Loaded libraries ({len(libs)}):"]
    for lib_path in libs:
        lines.append(f"  {lib_path}")

    return [types.TextContent(type="text", text="\n".join(lines))]


async def handle_list_subcircuits(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """List subcircuit models from loaded libraries.

    Args:
        arguments: Optional 'path' (string) to filter to specific library
        state: Session state with library manager

    Returns:
        List with single TextContent containing subcircuit names

    Raises:
        PathSecurityError: Path outside sandbox (if path provided)
    """
    filter_path = None
    if "path" in arguments:
        filter_path = safe_path(arguments["path"], state)

    # Search for all subcircuits (empty query, filter to .SUBCKT type)
    try:
        result = await run_sync(
            state.libraries.search_user_libraries, "", 0, 999999
        )
    except Exception as e:
        raise LibraryError(f"Failed to list subcircuits: {e}")

    # Filter to .SUBCKT type only
    subcircuits = [
        r for r in result["results"]
        if r["type"] == ".SUBCKT"
    ]

    # Apply path filter if specified
    if filter_path:
        subcircuits = [
            r for r in subcircuits
            if str(filter_path) in r["source_path"]
        ]

    if not subcircuits:
        msg = "No subcircuits found"
        if filter_path:
            msg += f" in {filter_path}"
        return [types.TextContent(type="text", text=msg)]

    lines = [f"Subcircuits ({len(subcircuits)}):"]
    for sc in subcircuits:
        lines.append(f"  {sc['name']} - {sc['source_path']}")

    return [types.TextContent(type="text", text="\n".join(lines))]


# Tool definitions
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="search_library",
        description="Search component libraries for models and subcircuits by name (case-insensitive substring match). Search 'user' for loaded libraries or 'builtin' for simulator built-in libraries. Results include model name, type (.MODEL/.SUBCKT), and source file.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search term (case-insensitive substring match)",
                },
                "source": {
                    "type": "string",
                    "enum": ["user", "builtin"],
                    "description": "Library source: 'user' for loaded libraries, 'builtin' for simulator built-in libraries (default: 'user')",
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of results to skip for pagination (default: 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results to return (default: 50, max: 50)",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="get_model_info",
        description="Get SPICE model/subcircuit details including parameters and ready-to-use .include directive. Set full=true to get the complete SPICE definition text. Searches both user-loaded and built-in libraries.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Model or subcircuit name (case-insensitive)",
                },
                "full": {
                    "type": "boolean",
                    "description": "Include full SPICE definition text (default: false)",
                },
            },
            "required": ["name"],
        },
    ),
    types.Tool(
        name="load_library",
        description="Load a SPICE library file (.lib, .mod) or directory of library files into the session. Loaded libraries are searchable via search_library. Accepts file or directory path — directories are scanned recursively for .lib/.mod files.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to library file or directory",
                },
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="unload_library",
        description="Unload a previously loaded library from the session. The library will no longer appear in search results.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to library file or directory to unload",
                },
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="list_libraries",
        description="List all user-loaded library file paths in the current session.",
        inputSchema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
    types.Tool(
        name="list_subcircuits",
        description="List available subcircuit models from loaded libraries. Optionally filter to a specific library file path.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional: filter to specific library file path",
                },
            },
            "required": [],
        },
    ),
]

# Handler mapping
TOOL_HANDLERS: dict[str, object] = {
    "search_library": handle_search_library,
    "get_model_info": handle_get_model_info,
    "load_library": handle_load_library,
    "unload_library": handle_unload_library,
    "list_libraries": handle_list_libraries,
    "list_subcircuits": handle_list_subcircuits,
}
