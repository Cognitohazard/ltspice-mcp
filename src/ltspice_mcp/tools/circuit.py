"""Unified circuit editing tools for .cir/.net netlists and .asc schematics.

Extension-based dispatch: the file extension determines which spicelib editor
is used (SpiceEditor for .cir/.net, AscEditor for .asc).  Schematic-only
operations (position, rotation, attributes, export) validate the extension
and raise NetlistError if given a non-.asc file.
"""

import asyncio
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, Union

from mcp import types
from spicelib import AscEditor, SpiceEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_PROP, PAGINATION_SCHEMA, format_response, paginate,
    pagination_metadata, run_sync, safe_path, text_response,
)

# Per-file locks to prevent concurrent edits to the same circuit file
_edit_locks: dict[Path, asyncio.Lock] = defaultdict(asyncio.Lock)

# Type alias for the union returned by _make_editor / _get_editor.
# Schematic-only handlers narrow this to AscEditor after _require_asc.
Editor = Union[AscEditor, SpiceEditor]


# ---------------------------------------------------------------------------
# Editor factory — extension-based dispatch
# ---------------------------------------------------------------------------

def _make_editor(path: Path) -> Editor:
    """Create an AscEditor or SpiceEditor based on file extension.

    Raises NetlistError if file not found or .asy symbol files are missing.
    """
    try:
        if path.suffix.lower() == ".asc":
            return AscEditor(str(path))
        return SpiceEditor(str(path))
    except FileNotFoundError as e:
        if ".asy" in str(e):
            raise NetlistError(
                f"Cannot open .asc schematic: {e}\n\n"
                "LTspice symbol libraries (.asy files) are required. "
                "Set [schematic] symbol_paths in ltspice-mcp.toml or "
                "LTSPICE_MCP_SYMBOL_PATHS environment variable."
            ) from e
        raise NetlistError(f"File not found: {path}") from e


async def _get_editor(path: Path, state: SessionState) -> Editor:
    """Get a cached editor instance, creating via _make_editor if needed."""
    return await run_sync(state.editors.get, path, lambda p: _make_editor(p))


async def _get_asc_editor(path: Path, state: SessionState) -> AscEditor:
    """Get a cached AscEditor. Caller must have validated _require_asc first."""
    editor = await _get_editor(path, state)
    assert isinstance(editor, AscEditor)
    return editor


def _is_asc(path: Path) -> bool:
    return path.suffix.lower() == ".asc"


def _require_asc(path: Path) -> None:
    """Raise if path is not an .asc file (for schematic-only operations)."""
    if not _is_asc(path):
        raise NetlistError(
            f"This operation requires an .asc schematic, got '{path.suffix}'. "
        )


@asynccontextmanager
async def _editing(path: Path, state: SessionState) -> AsyncIterator[Editor]:
    """Get a cached editor, yield it, then save and invalidate on success.

    If the caller raises, changes are not saved (fail-safe).
    Uses per-file locking to prevent concurrent edits to the same file.
    """
    async with _edit_locks[path]:
        editor = await _get_editor(path, state)
        yield editor
        await run_sync(editor.save_netlist, str(path))
        state.editors.invalidate(path)


@asynccontextmanager
async def _editing_asc(path: Path, state: SessionState) -> AsyncIterator[AscEditor]:
    """Get a cached AscEditor, yield it, then save and invalidate on success.

    Caller must have validated _require_asc first.
    Uses per-file locking to prevent concurrent edits to the same file.
    """
    async with _edit_locks[path]:
        editor = await _get_asc_editor(path, state)
        yield editor
        await run_sync(editor.save_netlist, str(path))
    state.editors.invalidate(path)


# ---------------------------------------------------------------------------
# Handlers — shared operations (work on .cir/.net and .asc)
# ---------------------------------------------------------------------------


async def handle_create_netlist(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Create a new SPICE netlist file from content string."""
    name = arguments["name"]
    content = arguments["content"]
    target_path = safe_path(f"{name}.cir", state)

    if await run_sync(target_path.exists):
        raise NetlistError(f"File already exists: {target_path}")

    if not content.strip().upper().endswith(".END"):
        content = content.rstrip() + "\n.END\n"

    await run_sync(target_path.write_text, content)

    try:
        editor = await run_sync(SpiceEditor, str(target_path))
        components = await run_sync(editor.get_components)
        comp_count = len(components)
    except Exception as e:
        await run_sync(lambda: target_path.unlink(missing_ok=True))
        raise NetlistError(f"Invalid netlist syntax: {e}")

    return text_response(f"Created netlist: {target_path}\nComponents: {comp_count}")


async def handle_read_circuit(
    arguments: dict, state: SessionState
):
    """Read and parse a circuit file. For .asc schematics, returns component
    positions, net labels, wires, and directives. For .cir/.net, returns raw
    content and component list with values.
    """
    file_path = safe_path(arguments["path"], state)
    fmt = arguments.get("format")

    if _is_asc(file_path):
        # Schematic info path (formerly get_schematic_info)
        editor = await _get_asc_editor(file_path, state)

        components = await run_sync(editor.get_components)

        lines = [f"=== {file_path.name} ===", ""]

        comp_data = []
        lines.append(f"Components ({len(components)}):")
        for ref in components:
            value = editor.get_component_value(ref)
            pos, rot = editor.get_component_position(ref)
            rot_str = f"R{rot.value}" if rot.value < 360 else f"M{rot.value - 360}"
            lines.append(f"  {ref:<8} {value:<20} pos=({pos.X},{pos.Y}) {rot_str}")
            comp_data.append({"reference": ref, "value": value, "x": pos.X, "y": pos.Y, "rotation": rot_str})

        label_data = []
        if editor.labels:
            lines.append("")
            lines.append(f"Net Labels ({len(editor.labels)}):")
            for label in editor.labels:
                lines.append(f"  {label.text:<16} at ({label.coord.X},{label.coord.Y})")
                label_data.append({"text": label.text, "x": label.coord.X, "y": label.coord.Y})

        directive_data = []
        lines.append("")
        lines.append(f"Wires: {len(editor.wires)}")
        lines.append(f"Directives: {len(editor.directives)}")

        if editor.directives:
            lines.append("")
            lines.append("SPICE Directives:")
            for d in editor.directives:
                lines.append(f"  {d.text}")
                directive_data.append(d.text)

        data = {
            "file": str(file_path),
            "type": "asc",
            "components": comp_data,
            "labels": label_data,
            "wire_count": len(editor.wires),
            "directives": directive_data,
        }
        return format_response("\n".join(lines), data, fmt)
    else:
        # Netlist read path — editor load catches FileNotFoundError
        editor = await _get_editor(file_path, state)
        content = await run_sync(file_path.read_text)
        components = editor.get_components()

        comp_list = []
        if components:
            comp_lines = []
            for comp_ref in components:
                value = editor.get_component_value(comp_ref)
                comp_lines.append(f"{comp_ref}  {value}")
                comp_list.append({"reference": comp_ref, "value": value})
            comp_summary = "\n".join(comp_lines)
        else:
            comp_summary = "(no components)"

        result = f"=== {file_path.name} ===\n\n{content}\n\n=== Components ({len(components)}) ===\n{comp_summary}"
        data = {
            "file": str(file_path),
            "type": "netlist",
            "content": content,
            "components": comp_list,
        }
        return format_response(result, data, fmt)


async def handle_list_components(
    arguments: dict, state: SessionState
):
    """List all components, optionally filtered by prefix. If a single
    reference is provided, return just that component's value.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments["path"], state)
    fmt = arguments.get("format")

    editor = await _get_editor(file_path, state)

    # Single-component lookup mode (absorbed from get_component_value)
    reference = arguments.get("reference")
    if reference is not None:
        try:
            value = await run_sync(editor.get_component_value, reference)
        except Exception:
            raise NetlistError(f"Component '{reference}' not found")
        data = {"reference": reference, "value": value}
        return format_response(f"{reference} = {value}", data, fmt)

    # List mode
    prefix = arguments.get("prefix")
    if prefix:
        components = await run_sync(editor.get_components, [prefix])
    else:
        components = await run_sync(editor.get_components)

    if not components:
        msg = f"No components matching prefix '{prefix}' found" if prefix else "No components found"
        return format_response(msg, {"components": [], "pagination": pagination_metadata(0, 0, 50)}, fmt)

    page, total, offset, limit = paginate(components, arguments)

    comp_list = []
    comp_lines = []
    for comp_ref in page:
        value = editor.get_component_value(comp_ref)
        comp_lines.append(f"{comp_ref}  {value}")
        comp_list.append({"reference": comp_ref, "value": value})

    header = f"Showing {offset + 1}-{offset + len(page)} of {total} components"
    if prefix:
        header += f" (prefix '{prefix}')"
    result = header + "\n\n" + "\n".join(comp_lines)

    if offset + len(page) < total:
        result += f"\n\nNext page: ltspice_list_components(path=..., offset={offset + limit})"

    data = {
        "components": comp_list,
        "pagination": pagination_metadata(total, offset, limit),
    }
    if prefix:
        data["prefix"] = prefix
    return format_response(result, data, fmt)


async def handle_set_component_value(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Set component value(s). Accepts single or batch mode.

    Single mode: provide 'reference' and 'value'.
    Batch mode: provide 'values' dict mapping references to new values.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments["path"], state)

    values_dict = arguments.get("values")
    reference = arguments.get("reference")
    value = arguments.get("value")

    async with _editing(file_path, state) as editor:
        if values_dict is not None:
            # Batch mode
            if not isinstance(values_dict, dict):
                raise NetlistError("'values' must be an object mapping references to new values")
            editor.set_component_values(**values_dict)
            changes = [f"{ref}: {val}" for ref, val in values_dict.items()]
            result = f"Updated {len(values_dict)} component(s):\n" + "\n".join(changes)
        elif reference is not None and value is not None:
            # Single mode
            try:
                old_value = editor.get_component_value(reference)
            except Exception:
                raise NetlistError(f"Component '{reference}' not found")
            editor.set_component_value(reference, value)
            result = f"Set {reference}: {old_value} -> {value}"
        else:
            raise NetlistError(
                "Provide either 'reference'+'value' (single) or 'values' dict (batch)"
            )

    return text_response(result)


async def handle_parameter(
    arguments: dict, state: SessionState
):
    """Get or set .PARAM directive values. Without name/value: returns all
    parameters. With name and value: sets the parameter.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments["path"], state)
    fmt = arguments.get("format")

    param_name = arguments.get("name")
    param_value = arguments.get("value")

    if param_name is not None and param_value is not None:
        # Set mode — confirmation only, no structured data needed
        async with _editing(file_path, state) as editor:
            editor.set_parameter(param_name, param_value)
        return text_response(f"Set .PARAM {param_name} = {param_value}")

    # Get mode (formerly get_parameters) — read-only, no _editing needed
    editor = await _get_editor(file_path, state)
    param_names = await run_sync(editor.get_all_parameter_names)

    params = {}
    if param_names:
        param_lines = []
        for name in param_names:
            value = await run_sync(editor.get_parameter, name)
            param_lines.append(f".PARAM {name} = {value}")
            params[name] = value
        result = "\n".join(param_lines)
    else:
        result = "No .PARAM directives found"

    return format_response(result, {"parameters": params}, fmt)


async def handle_edit_directive(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Add or remove a SPICE directive. Works on .cir/.net and .asc."""
    file_path = safe_path(arguments["path"], state)

    action = arguments["action"]
    instruction = arguments["instruction"]

    async with _editing(file_path, state) as editor:
        if action == "add":
            if not instruction.strip().startswith("."):
                raise NetlistError(
                    "SPICE directives must start with '.' (e.g. .tran, .ac, .param)"
                )
            editor.add_instruction(instruction)
            result = f"Added directive: {instruction}"

        elif action == "remove":
            if instruction.startswith("regex:"):
                pattern = instruction[6:]
                editor.remove_Xinstruction(pattern)
            elif any(char in instruction for char in r"\[]().*+?^${}|"):
                editor.remove_Xinstruction(instruction)
            else:
                editor.remove_instruction(instruction)
            result = f"Removed directive: {instruction}"

        else:
            raise NetlistError(f"Invalid action '{action}'. Must be 'add' or 'remove'.")

    return text_response(result)


# ---------------------------------------------------------------------------
# Handlers — schematic-only operations (.asc only)
# ---------------------------------------------------------------------------


async def handle_remove_component(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Remove a component from a schematic by reference designator."""
    asc_path = safe_path(arguments["path"], state)
    _require_asc(asc_path)


    reference = arguments["reference"]

    async with _editing_asc(asc_path, state) as editor:
        components = await run_sync(editor.get_components)
        if reference not in components:
            raise NetlistError(
                f"Component '{reference}' not found. "
                f"Available: {', '.join(components)}"
            )
        await run_sync(editor.remove_component, reference)

    return text_response(f"Removed {reference} from {asc_path.name}")


async def handle_move_component(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Move or rotate a component in a schematic."""
    from spicelib.editor.base_schematic import ERotation, Point

    asc_path = safe_path(arguments["path"], state)
    _require_asc(asc_path)


    reference = arguments["reference"]
    x = int(arguments["x"])
    y = int(arguments["y"])
    rotation = arguments.get("rotation", None)

    async with _editing_asc(asc_path, state) as editor:
        old_pos, old_rot = editor.get_component_position(reference)

        if rotation is not None:
            rot_map = {
                "R0": ERotation.R0, "R90": ERotation.R90,
                "R180": ERotation.R180, "R270": ERotation.R270,
                "M0": ERotation.M0, "M90": ERotation.M90,
                "M180": ERotation.M180, "M270": ERotation.M270,
            }
            new_rot = rot_map.get(rotation)
            if new_rot is None:
                raise NetlistError(
                    f"Invalid rotation '{rotation}'. "
                    f"Valid: {', '.join(rot_map.keys())}"
                )
        else:
            new_rot = old_rot

        new_pos = Point(x, y)
        editor.set_component_position(reference, new_pos, new_rot)

    rot_str = f"R{new_rot.value}" if new_rot.value < 360 else f"M{new_rot.value - 360}"
    return text_response(f"Moved {reference}: ({old_pos.X},{old_pos.Y}) -> ({x},{y}) {rot_str}")


async def handle_set_component_attribute(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Set an attribute on a schematic component (e.g., SpiceLine, SpiceModel)."""
    asc_path = safe_path(arguments["path"], state)
    _require_asc(asc_path)


    reference = arguments["reference"]
    attribute = arguments["attribute"]
    value = arguments["value"]

    async with _editing_asc(asc_path, state) as editor:
        await run_sync(editor.set_component_attribute, reference, attribute, value)

    return text_response(f"Set {reference}.{attribute} = {value}")


async def handle_export_netlist(
    arguments: dict, state: SessionState
) -> types.CallToolResult:
    """Export an .asc schematic to a SPICE netlist (.net) using LTspice."""
    asc_path = safe_path(arguments["path"], state)
    _require_asc(asc_path)


    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        raise NetlistError(
            "export_netlist requires LTspice to convert .asc to netlist. "
            "Available simulators: " + str(list(state.available_simulators.keys()))
        )

    try:
        net_path = await run_sync(ltspice_cls.create_netlist, str(asc_path))
        net_path = Path(net_path)
    except Exception as e:
        raise NetlistError(f"LTspice netlist export failed: {e}")

    if not await run_sync(net_path.exists):
        raise NetlistError("Export failed: .net file not created")

    content = await run_sync(net_path.read_text)

    return text_response(f"=== {net_path.name} ===\n\n{content}")


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

_PATH_DESC = "Path to circuit file (.cir, .net, or .asc schematic)"

TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="ltspice_create_netlist",
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
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_read_circuit",
        description=(
            "Read and parse a circuit file (.cir/.net or .asc). "
            "For .cir/.net: returns raw netlist content and component list with values. "
            "For .asc: returns component positions, net labels, wire count, and SPICE directives."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": _PATH_DESC,
                },
                "format": FORMAT_PROP,
            },
            "required": ["path"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_list_components",
        description=(
            "List components in a circuit file (.cir/.net or .asc), optionally filtered by type prefix "
            "(R, C, L, Q, M, X, V, I, etc.). If 'reference' is provided, returns just that component's "
            "value (e.g., 'R1 = 10k'). Component lookups are case-insensitive."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": _PATH_DESC,
                },
                "prefix": {
                    "type": "string",
                    "description": "Optional component prefix filter (e.g., 'R' for resistors, 'C' for capacitors, 'Q' for transistors)",
                },
                "reference": {
                    "type": "string",
                    "description": "Optional: get a single component's value by reference designator (e.g., 'R1', 'C2', 'X1:R5'). Supports hierarchical references. Case-insensitive.",
                },
                "offset": {
                    "type": "integer",
                    "description": "Number of components to skip for pagination (default: 0)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum components to return (default: 50, max: 50)",
                },
                "format": FORMAT_PROP,
            },
            "required": ["path"],
        },
        outputSchema={
            "type": "object",
            "properties": {
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "reference": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                },
                "pagination": PAGINATION_SCHEMA,
            },
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_set_component_value",
        description=(
            "Set component value(s) in a circuit file (.cir/.net or .asc). "
            "Single mode: provide 'reference' and 'value'. "
            "Batch mode: provide 'values' dict. "
            "SPICE notation: k=1e3, M=1e-3 (milli, NOT mega), Meg=1e6, u=1e-6, n=1e-9, p=1e-12. "
            "Changes are saved immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": _PATH_DESC,
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator (e.g., 'R1'). Use with 'value' for single mode.",
                },
                "value": {
                    "type": "string",
                    "description": "New value (e.g., '10k', '100n', 'LM358'). Use with 'reference' for single mode.",
                },
                "values": {
                    "type": "object",
                    "description": 'Batch mode: map of reference to value, e.g. {"R1": "10k", "C1": "100n"}.',
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["path"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_parameter",
        description=(
            "Read or write .PARAM directive values in a circuit file. "
            "Without name/value: lists all parameters and their current values. "
            "With name and value: sets the parameter (creates directive if missing). "
            "Changes saved immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": _PATH_DESC,
                },
                "name": {
                    "type": "string",
                    "description": "Parameter name (required for set mode)",
                },
                "value": {
                    "type": "string",
                    "description": "Parameter value (required for set mode)",
                },
                "format": FORMAT_PROP,
            },
            "required": ["path"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_edit_directive",
        description=(
            "Add or remove a SPICE directive (.tran, .ac, .param, etc.). "
            "Action 'add' replaces existing unique directives of the same type. "
            "Action 'remove' supports exact match or regex (prefix with 'regex:')."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": _PATH_DESC,
                },
                "action": {
                    "type": "string",
                    "description": "Whether to 'add' or 'remove' the directive",
                    "enum": ["add", "remove"],
                },
                "instruction": {
                    "type": "string",
                    "description": "SPICE directive (e.g. '.tran 0 10m 0 1u', '.ac dec 100 1 1Meg'). Must start with '.' for add. For remove, use exact text or prefix with 'regex:' for pattern match.",
                },
            },
            "required": ["path", "action", "instruction"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    # --- Schematic-only operations (.asc) ---
    types.Tool(
        name="ltspice_remove_component",
        description=(
            "Remove a component from an .asc schematic by reference designator "
            "(e.g., R1, C2, U1). The component and symbol are removed; wires remain. "
            "Changes are saved immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .asc schematic file",
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator to remove (e.g., 'R1', 'C2')",
                },
            },
            "required": ["path", "reference"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=True,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_move_component",
        description=(
            "Move and/or rotate a component in an .asc schematic. "
            "Coordinates are in LTspice's grid units (typically multiples of 16). "
            "Changes are saved immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .asc schematic file",
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator (e.g., 'R1')",
                },
                "x": {
                    "type": "integer",
                    "description": "New X position in grid units",
                },
                "y": {
                    "type": "integer",
                    "description": "New Y position in grid units",
                },
                "rotation": {
                    "type": "string",
                    "description": "Rotation/mirror: R0, R90, R180, R270, M0, M90, M180, M270. If omitted, keeps current rotation.",
                    "enum": ["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"],
                },
            },
            "required": ["path", "reference", "x", "y"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_set_component_attribute",
        description=(
            "Set an attribute on a component in an .asc schematic. "
            "Common attributes: Value, Value2, SpiceLine, SpiceLine2, SpiceModel. "
            "Use this for advanced configuration beyond simple value changes "
            "(e.g., setting MOSFET model parameters via SpiceLine). "
            "Changes are saved immediately."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .asc schematic file",
                },
                "reference": {
                    "type": "string",
                    "description": "Component reference designator (e.g., 'M1', 'U1')",
                },
                "attribute": {
                    "type": "string",
                    "description": "Attribute name (e.g., 'Value', 'Value2', 'SpiceLine', 'SpiceModel')",
                },
                "value": {
                    "type": "string",
                    "description": "Attribute value to set",
                },
            },
            "required": ["path", "reference", "attribute", "value"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=False,
        ),
    ),
    types.Tool(
        name="ltspice_export_netlist",
        description=(
            "Export an .asc schematic to a SPICE netlist (.net) using the LTspice binary. "
            "Returns the generated netlist content. LTspice must be available. "
            "Use this when you need the SPICE netlist for a schematic, "
            "or before running simulation on a schematic."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to .asc schematic file",
                },
            },
            "required": ["path"],
        },
        annotations=types.ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=False,
        ),
    ),
]

TOOL_HANDLERS: dict[str, object] = {
    "ltspice_create_netlist": handle_create_netlist,
    "ltspice_read_circuit": handle_read_circuit,
    "ltspice_list_components": handle_list_components,
    "ltspice_set_component_value": handle_set_component_value,
    "ltspice_parameter": handle_parameter,
    "ltspice_edit_directive": handle_edit_directive,
    "ltspice_remove_component": handle_remove_component,
    "ltspice_move_component": handle_move_component,
    "ltspice_set_component_attribute": handle_set_component_attribute,
    "ltspice_export_netlist": handle_export_netlist,
}
