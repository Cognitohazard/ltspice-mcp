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


async def handle_set_component_value(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Set the value of a single component.

    Args:
        arguments: Contains 'netlist' (path), 'reference' (component), 'value' (new value)
        state: Session state with editors cache

    Returns:
        List with single TextContent confirming the change

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, component not found, or write error
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
    new_value = arguments["value"]

    # Get old value first for confirmation
    try:
        old_value = editor.get_component_value(reference)
    except Exception:
        # Component not found - try case-insensitive search for suggestions
        all_components = editor.get_components()
        matches = [c for c in all_components if c.upper() == reference.upper()]
        if matches:
            raise NetlistError(
                f"Component '{reference}' not found. Did you mean '{matches[0]}'? "
                f"(Component names are case-preserving)"
            )
        else:
            raise NetlistError(f"Component '{reference}' not found in netlist")

    # Set new value (in-memory operation)
    editor.set_component_value(reference, new_value)

    # Save to disk
    await run_sync(editor.save_netlist, str(netlist_path))

    # Invalidate cache to prevent stale reads
    state.editors.invalidate(netlist_path)

    result = f"Set {reference}: {old_value} -> {new_value}"
    return [types.TextContent(type="text", text=result)]


async def handle_set_component_values(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Set values for multiple components in one call.

    Args:
        arguments: Contains 'netlist' (path) and 'values' (dict of reference->value)
        state: Session state with editors cache

    Returns:
        List with single TextContent listing all changes

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, invalid values format, or write error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    values = arguments["values"]

    # Validate that values is a dict
    if not isinstance(values, dict):
        raise NetlistError("Values must be an object mapping component references to new values")

    # Set component values using spicelib's batch API
    editor.set_component_values(**values)

    # Save to disk
    await run_sync(editor.save_netlist, str(netlist_path))

    # Invalidate cache
    state.editors.invalidate(netlist_path)

    # Format result
    changes = []
    for ref, val in values.items():
        changes.append(f"{ref}: {val}")
    result = f"Updated {len(values)} component(s):\n" + "\n".join(changes)

    return [types.TextContent(type="text", text=result)]


async def handle_set_parameter(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Set a .PARAM directive value.

    Args:
        arguments: Contains 'netlist' (path), 'name' (parameter), 'value' (new value)
        state: Session state with editors cache

    Returns:
        List with single TextContent confirming the change

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found or write error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    param_name = arguments["name"]
    param_value = arguments["value"]

    # Set parameter (in-memory operation)
    editor.set_parameter(param_name, param_value)

    # Save to disk
    await run_sync(editor.save_netlist, str(netlist_path))

    # Invalidate cache
    state.editors.invalidate(netlist_path)

    result = f"Set .PARAM {param_name} = {param_value}"
    return [types.TextContent(type="text", text=result)]


async def handle_add_instruction(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Add a SPICE directive to a netlist.

    Args:
        arguments: Contains 'netlist' (path) and 'instruction' (directive text)
        state: Session state with editors cache

    Returns:
        List with single TextContent confirming the addition

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, invalid directive, or write error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    instruction = arguments["instruction"]

    # Validate directive starts with dot
    if not instruction.strip().startswith("."):
        raise NetlistError(
            "SPICE directives must start with '.' (e.g. .tran, .ac, .param)"
        )

    # Add directive (in-memory operation)
    # Note: spicelib automatically replaces unique directives of the same type
    editor.add_instruction(instruction)

    # Save to disk
    await run_sync(editor.save_netlist, str(netlist_path))

    # Invalidate cache
    state.editors.invalidate(netlist_path)

    result = f"Added directive: {instruction}"
    return [types.TextContent(type="text", text=result)]


async def handle_remove_instruction(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Remove a SPICE directive from a netlist.

    Args:
        arguments: Contains 'netlist' (path) and 'instruction' (directive text or regex)
        state: Session state with editors cache

    Returns:
        List with single TextContent confirming the removal

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, directive not found, or write error
    """
    netlist_path = safe_path(arguments["netlist"], state)

    # Verify file exists
    if not await run_sync(netlist_path.exists):
        raise NetlistError(f"Netlist not found: {netlist_path}")

    # Get cached editor
    editor = await run_sync(
        state.editors.get, netlist_path, lambda p: SpiceEditor(str(p))
    )

    instruction = arguments["instruction"]

    # Determine removal strategy
    if instruction.startswith("regex:"):
        # Regex pattern removal
        pattern = instruction[6:]  # Strip "regex:" prefix
        editor.remove_Xinstruction(pattern)
    elif any(char in instruction for char in r"\[]().*+?^${}|"):
        # Contains regex metacharacters, use regex removal
        editor.remove_Xinstruction(instruction)
    else:
        # Exact match removal
        editor.remove_instruction(instruction)

    # Save to disk
    await run_sync(editor.save_netlist, str(netlist_path))

    # Invalidate cache
    state.editors.invalidate(netlist_path)

    result = f"Removed directive: {instruction}"
    return [types.TextContent(type="text", text=result)]


async def handle_convert_schematic(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Convert a .asc schematic to .net netlist using LTSpice.

    Args:
        arguments: Contains 'asc_path' (path to .asc schematic file)
        state: Session state with available_simulators

    Returns:
        List with single TextContent containing conversion results

    Raises:
        PathSecurityError: Path outside sandbox
        NetlistError: File not found, not .asc file, LTSpice unavailable, or conversion failed
    """
    asc_path = safe_path(arguments["asc_path"], state)

    # Verify file exists
    if not await run_sync(asc_path.exists):
        raise NetlistError(f"Schematic file not found: {asc_path}")

    # Validate .asc extension (case-insensitive)
    if asc_path.suffix.lower() != ".asc":
        raise NetlistError(f"Expected .asc file, got {asc_path.suffix}")

    # Check LTSpice availability
    if "ltspice" not in state.available_simulators:
        available = list(state.available_simulators.keys())
        raise NetlistError(
            f"convert_schematic requires LTSpice. Available simulators: {available}"
        )

    # Convert via SpiceEditor (LTSpice auto-converts .asc to .net)
    try:
        editor = await run_sync(SpiceEditor, str(asc_path))
    except Exception as e:
        raise NetlistError(f"LTSpice conversion failed: {e}")

    # Calculate expected output path
    net_path = asc_path.with_suffix(".net")

    # Save netlist to ensure .net file is written
    await run_sync(editor.save_netlist, str(net_path))

    # Verify .net file was created
    if not await run_sync(net_path.exists):
        raise NetlistError("Conversion failed: .net file not created")

    # Get component summary
    components = editor.get_components()
    comp_count = len(components)

    result = f"Converted {asc_path.name} -> {net_path.name}\nComponents: {comp_count}\nOutput: {net_path}"
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
    types.Tool(
        name="set_component_value",
        description="Set the value of a single component. Changes are persisted to disk immediately. SPICE notation: k=1e3, M=1e-3 (milli, NOT mega), Meg=1e6, u=1e-6, n=1e-9, p=1e-12.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator e.g. R1, XU1:C2",
                },
                "value": {
                    "type": "string",
                    "description": "New value e.g. 10k, 100n, LM358. In SPICE: M=milli and Meg=mega",
                },
            },
            "required": ["netlist", "reference", "value"],
        },
    ),
    types.Tool(
        name="set_component_values",
        description="Set values for multiple components in one call. Changes are persisted to disk immediately. SPICE notation: k=1e3, M=1e-3 (milli, NOT mega), Meg=1e6, u=1e-6, n=1e-9, p=1e-12.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "values": {
                    "type": "object",
                    "description": 'Map of reference to new value e.g. {"R1": "10k", "C1": "100n"}. In SPICE: M=milli and Meg=mega',
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["netlist", "values"],
        },
    ),
    types.Tool(
        name="set_parameter",
        description="Set a .PARAM directive value. If the parameter does not exist, it is created. Changes are persisted to disk immediately.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "name": {
                    "type": "string",
                    "description": "Parameter name",
                },
                "value": {
                    "type": "string",
                    "description": "Parameter value",
                },
            },
            "required": ["netlist", "name", "value"],
        },
    ),
    types.Tool(
        name="add_instruction",
        description="Add a SPICE directive to a netlist (e.g., .tran, .ac, .param). If a unique directive of the same type already exists (e.g., .tran), it is replaced automatically. Changes are persisted to disk immediately.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "instruction": {
                    "type": "string",
                    "description": "SPICE directive to add e.g. '.tran 0 10m 0 1u', '.ac dec 100 1 1Meg', '.param R_val=1k'. Must start with a dot.",
                },
            },
            "required": ["netlist", "instruction"],
        },
    ),
    types.Tool(
        name="remove_instruction",
        description="Remove a SPICE directive from a netlist. Uses exact text match by default (whitespace-sensitive). For regex patterns, prefix with 'regex:' or include regex metacharacters. Use read_netlist first to see exact directive text. Changes are persisted to disk immediately.",
        inputSchema={
            "type": "object",
            "properties": {
                "netlist": {
                    "type": "string",
                    "description": "Path to .net or .cir file",
                },
                "instruction": {
                    "type": "string",
                    "description": "Directive text to remove (exact match) or regex pattern prefixed with 'regex:' e.g. 'regex:\\.tran\\s+.*'",
                },
            },
            "required": ["netlist", "instruction"],
        },
    ),
    types.Tool(
        name="convert_schematic",
        description="Convert a .asc schematic file to .net netlist using LTSpice. LTSpice must be available on the system. The .net file is created in the same directory as the .asc file.",
        inputSchema={
            "type": "object",
            "properties": {
                "asc_path": {
                    "type": "string",
                    "description": "Path to .asc schematic file. LTSpice must be available for conversion.",
                },
            },
            "required": ["asc_path"],
        },
    ),
]

TOOL_HANDLERS: dict[str, object] = {
    "create_netlist": handle_create_netlist,
    "read_netlist": handle_read_netlist,
    "list_components": handle_list_components,
    "get_component_value": handle_get_component_value,
    "get_parameters": handle_get_parameters,
    "set_component_value": handle_set_component_value,
    "set_component_values": handle_set_component_values,
    "set_parameter": handle_set_parameter,
    "add_instruction": handle_add_instruction,
    "remove_instruction": handle_remove_instruction,
    "convert_schematic": handle_convert_schematic,
}
