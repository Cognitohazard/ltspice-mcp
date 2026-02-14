"""Circuit management tools. (Phase 2)"""

from mcp import types
from spicelib import SpiceEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import run_sync, safe_path


async def handle_create_netlist(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Create a new SPICE netlist file from content string.

    Args:
        arguments: Contains 'name' (filename without extension) and 'content' (netlist text)
        state: Session state with working directory

    Returns:
        List with single TextContent containing file path and component count

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File already exists or invalid netlist syntax
    """
    name = arguments["name"]
    content = arguments["content"]

    # Build target path and validate sandbox
    target_path = safe_path(f"{name}.cir", state)

    # Check if file already exists
    if await run_sync(target_path.exists):
        raise NetlistError(f"File already exists: {target_path}")

    # Ensure content ends with .END for portability (Pitfall 4)
    if not content.strip().upper().endswith(".END"):
        content = content.rstrip() + "\n.END\n"

    # Write content to file
    await run_sync(target_path.write_text, content)

    # Parse to verify syntax and get component count
    try:
        editor = await run_sync(SpiceEditor, str(target_path))
        components = await run_sync(editor.get_components)
        comp_count = len(components)
    except Exception as e:
        # Clean up invalid file
        await run_sync(lambda: target_path.unlink(missing_ok=True))
        raise NetlistError(f"Invalid netlist syntax: {e}")

    result = f"Created netlist: {target_path}\nComponents: {comp_count}"
    return [types.TextContent(type="text", text=result)]


async def handle_read_netlist(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Read and parse a SPICE netlist file.

    Args:
        arguments: Contains 'path' to .net/.cir file
        state: Session state with editors cache

    Returns:
        List with single TextContent containing raw netlist and component summary

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found or parse error
    """
    netlist_path = safe_path(arguments["path"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Read raw content
    content = await run_sync(netlist_path.read_text)

    # Get parsed component list via cached editor
    # SpiceEditor constructor is blocking (file I/O), so we wrap the cache access
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    # Get component references (in-memory operation, fast)
    # get_components() returns list of strings (component references)
    components = editor.get_components()

    # Format component summary by getting each value
    if components:
        comp_lines = []
        for comp_ref in components:
            value = editor.get_component_value(comp_ref)
            comp_lines.append(f"{comp_ref}  {value}")
        comp_summary = "\n".join(comp_lines)
    else:
        comp_summary = "(no components)"

    # Build response
    result = f"=== {netlist_path.name} ===\n\n{content}\n\n=== Components ({len(components)}) ===\n{comp_summary}"

    return [types.TextContent(type="text", text=result)]


async def handle_list_components(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """List all components in a netlist, optionally filtered by prefix.

    Args:
        arguments: Contains 'netlist' (path) and optional 'prefix' (component type filter)
        state: Session state with editors cache

    Returns:
        List with single TextContent containing component list with values

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found or parse error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    # Build prefix filter
    prefix = arguments.get("prefix")
    if prefix:
        # get_components() accepts list of prefixes
        components = await run_sync(editor.get_components, [prefix])
    else:
        # None or empty list returns all components
        components = await run_sync(editor.get_components)

    # Format component list
    if components:
        comp_lines = []
        for comp_ref in components:
            value = editor.get_component_value(comp_ref)
            comp_lines.append(f"{comp_ref}  {value}")
        result = "\n".join(comp_lines)
    else:
        if prefix:
            result = f"No components matching prefix '{prefix}' found"
        else:
            result = "No components found"

    return [types.TextContent(type="text", text=result)]


async def handle_get_component_value(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get the value of a specific component by reference.

    Args:
        arguments: Contains 'netlist' (path) and 'reference' (component designator)
        state: Session state with editors cache

    Returns:
        List with single TextContent containing component value

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, component not found, or parse error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    reference = arguments["reference"]

    # Try to get component value
    try:
        value = await run_sync(editor.get_component_value, reference)
        result = f"{reference} = {value}"
    except Exception:
        # Component not found - try case-insensitive search for suggestions (Pitfall 1)
        all_components = await run_sync(editor.get_components)
        matches = [c for c in all_components if c.upper() == reference.upper()]
        if matches:
            raise NetlistError(
                f"Component '{reference}' not found. Did you mean '{matches[0]}'? "
                f"(Component names are case-preserving)"
            )
        else:
            raise NetlistError(f"Component '{reference}' not found in netlist")

    return [types.TextContent(type="text", text=result)]


async def handle_get_parameters(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get all .PARAM names and values from a netlist.

    Args:
        arguments: Contains 'netlist' (path)
        state: Session state with editors cache

    Returns:
        List with single TextContent containing parameter list

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found or parse error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    # Get parameter names
    param_names = await run_sync(editor.get_all_parameter_names)

    # Format parameter list
    if param_names:
        param_lines = []
        for name in param_names:
            value = await run_sync(editor.get_parameter, name)
            param_lines.append(f".PARAM {name} = {value}")
        result = "\n".join(param_lines)
    else:
        result = "No .PARAM directives found"

    return [types.TextContent(type="text", text=result)]


# Tool definitions for MCP
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="create_netlist",
        description="Create a new SPICE netlist file from content string. Automatically appends .END if missing. File is created in working directory with .cir extension.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "File name without extension (e.g., 'rc_filter')",
                },
                "content": {
                    "type": "string",
                    "description": "Complete SPICE netlist content including components and directives",
                },
            },
            "required": ["name", "content"],
        },
    ),
    types.Tool(
        name="read_netlist",
        description="Read and parse a SPICE netlist file. Returns raw netlist content and component list with values.",
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .net or .cir file (relative to working directory or absolute within sandbox)",
                },
            },
            "required": ["path"],
        },
    ),
    types.Tool(
        name="list_components",
        description="List all components in a netlist, optionally filtered by component type prefix (R, C, L, Q, M, X, V, I, etc.). Component names are case-preserving.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "prefix": {
                    "type": "string",
                    "description": "Optional component prefix filter (e.g., 'R' for resistors, 'C' for capacitors, 'Q' for transistors)",
                },
            },
            "required": ["netlist"],
        },
    ),
    types.Tool(
        name="get_component_value",
        description="Get the value of a specific component by reference designator. Supports hierarchical references (e.g., 'X1:C2' for component inside subcircuit). Component names are case-preserving.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator (e.g., 'R1', 'C2', 'X1:R5')",
                },
            },
            "required": ["netlist", "reference"],
        },
    ),
    types.Tool(
        name="get_parameters",
        description="Get all .PARAM names and values from a netlist. Parameters defined inside subcircuits are scoped locally to that subcircuit.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
            },
            "required": ["netlist"],
        },
    ),
]

TOOL_HANDLERS: dict[str, object] = {
    "create_netlist": handle_create_netlist,
    "read_netlist": handle_read_netlist,
    "list_components": handle_list_components,
    "get_component_value": handle_get_component_value,
    "get_parameters": handle_get_parameters,
}
