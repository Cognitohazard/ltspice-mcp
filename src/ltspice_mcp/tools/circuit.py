"""Unified circuit editing tools for .cir/.net netlists and .asc schematics.

Extension-based dispatch: the file extension determines which spicelib editor
is used (SpiceEditor for .cir/.net, AscEditor for .asc).  Schematic-only
operations (position, rotation, attributes, export) validate the extension
and raise NetlistError if given a non-.asc file.
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from mcp import types
from pydantic import Field
from spicelib import AscEditor, SpiceEditor
from spicelib.editor.base_schematic import (
    ERotation,
    Line,
    Point,
    SchematicComponent,
    Text,
    TextTypeEnum,
)

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, get_symbol_info
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    PAGINATION_SCHEMA,
    RO_ANNOTATIONS,
    StrictModel,
    ToolInput,
    format_response,
    paginate,
    pagination_metadata,
    registry,
    safe_path,
    text_response,
)

# Per-file locks to prevent concurrent edits to the same circuit file.
# Bounded to avoid unbounded growth; only evicts *unheld* locks.
_MAX_EDIT_LOCKS = 64
_edit_locks: dict[Path, asyncio.Lock] = {}


def _get_edit_lock(path: Path) -> asyncio.Lock:
    """Get or create a per-file edit lock, evicting oldest unheld lock if at capacity."""
    if path in _edit_locks:
        # Move to end (most recently used) for LRU ordering
        _edit_locks[path] = _edit_locks.pop(path)
        return _edit_locks[path]
    if len(_edit_locks) >= _MAX_EDIT_LOCKS:
        # Evict the oldest *unheld* lock to avoid breaking mutual exclusion
        for candidate in list(_edit_locks):
            if not _edit_locks[candidate].locked():
                del _edit_locks[candidate]
                break
        # If all locks are held, allow temporary overshoot rather than break safety
    _edit_locks[path] = asyncio.Lock()
    return _edit_locks[path]

# Rotation string -> ERotation enum mapping (shared by move/add handlers)
_ROTATION_MAP: dict[str, ERotation] = {
    "R0": ERotation.R0,
    "R90": ERotation.R90,
    "R180": ERotation.R180,
    "R270": ERotation.R270,
    "M0": ERotation.M0,
    "M90": ERotation.M90,
    "M180": ERotation.M180,
    "M270": ERotation.M270,
}


def _parse_rotation(rotation: str) -> ERotation:
    """Parse a rotation string to ERotation enum. Raises NetlistError if invalid."""
    erot = _ROTATION_MAP.get(rotation)
    if erot is None:
        raise NetlistError(
            f"Invalid rotation '{rotation}'. Valid: {', '.join(_ROTATION_MAP.keys())}"
        )
    return erot


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """AABB overlap test between two bounding boxes with {x, y, width, height}."""
    return (
        a["x"] < b["x"] + b["width"]
        and a["x"] + a["width"] > b["x"]
        and a["y"] < b["y"] + b["height"]
        and a["y"] + a["height"] > b["y"]
    )


def _collect_component_geometry(editor: AscEditor) -> list[dict]:
    """Collect bounding boxes and pin positions for all components."""
    result: list[dict] = []
    for ref in editor.get_components():
        comp = editor.components[ref]
        sym = comp.symbol
        sym_info = get_symbol_info(sym) if sym else None
        if sym_info is None:
            continue
        pos, erot = editor.get_component_position(ref)
        rot_str = erot.name if erot else "R0"
        geo = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
        result.append({"ref": ref, **geo["bounding_box"], "pins": geo["pins"]})
    return result

# Type alias for the union returned by _make_editor / _get_editor.
# Schematic-only handlers narrow this to AscEditor after _require_asc.
Editor = AscEditor | SpiceEditor


class CreateNetlistInput(ToolInput):
    name: str = Field(description="File name without extension")
    content: str = Field(description="Complete SPICE netlist content")


class CircuitReadInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc schematic)")
    format: Literal["json", "text"] | None = Field(default=None)


class ListComponentsInput(ToolInput):
    path: str
    prefix: str | None = None
    reference: str | None = None
    offset: int = 0
    limit: int = 50
    format: Literal["json", "text"] | None = None


class SetComponentValueInput(ToolInput):
    path: str
    reference: str | None = None
    value: str | None = None
    values: dict[str, str] | None = None


class ParameterInput(ToolInput):
    path: str
    name: str | None = None
    value: str | None = None
    format: Literal["json", "text"] | None = None


class EditDirectiveInput(ToolInput):
    path: str
    action: Literal["add", "remove"]
    instruction: str


class RemoveComponentInput(ToolInput):
    path: str
    reference: str


class MoveComponentInput(ToolInput):
    path: str
    reference: str
    x: int
    y: int
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] | None = None


class SetComponentAttributeInput(ToolInput):
    path: str
    reference: str
    attribute: str
    value: str


class ExportNetlistInput(ToolInput):
    path: str


class AddComponentInput(ToolInput):
    path: str
    reference: str
    symbol: str
    x: int
    y: int
    value: str | None = None
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] = "R0"
    attributes: dict[str, str] | None = Field(
        default=None,
        description="Optional attributes to set (e.g., {'SpiceLine': 'W=10u L=0.5u', 'Value2': '...'})",
    )


class NetLabelInput(ToolInput):
    path: str
    net: str = Field(description="Net name ('0' for ground, or a name like 'VDD', 'outp')")
    x: int | None = Field(default=None, description="X coordinate (required unless pin is specified)")
    y: int | None = Field(default=None, description="Y coordinate (required unless pin is specified)")
    pin: str | None = Field(
        default=None,
        description="Component pin reference (e.g., 'M3.S') — places label at the pin's coordinates",
    )
    action: Literal["add", "remove"] = "add"


class AddTextInput(ToolInput):
    path: str
    text: str = Field(description="Text content to display on the schematic")
    x: int
    y: int
    size: int = Field(default=2, description="Font size (1=small, 2=normal, 3=large)")


class WaypointInput(StrictModel):
    x: int
    y: int


class ConnectInput(ToolInput):
    path: str
    from_pin: str = Field(
        description="Source pin as 'Reference.Pin' (e.g., 'M1.D', 'VDD.+') or 'net:name' for a net label"
    )
    to_pin: str = Field(
        description="Target pin as 'Reference.Pin' (e.g., 'M4a.D', 'VDD.+') or 'net:name' for a net label"
    )
    waypoints: list[WaypointInput] = Field(
        default_factory=list,
        description=(
            "Intermediate points for wire routing. For L-shaped routes, provide the "
            "corner point. For straight connections (same x or same y), omit."
        ),
    )


class SymbolInfoInput(ToolInput):
    symbol: str = Field(description="Symbol name (e.g., 'nmos', 'pmos', 'res', 'cap', 'voltage')")
    x: int = Field(default=0, description="Placement X coordinate (for computing absolute positions)")
    y: int = Field(default=0, description="Placement Y coordinate (for computing absolute positions)")
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] = "R0"


class ComponentInfoInput(ToolInput):
    path: str
    reference: str = Field(description="Component reference (e.g., 'M1', 'R1')")


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


def _get_editor(path: Path, state: SessionState) -> Editor:
    """Get a cached editor instance, creating via _make_editor if needed."""
    return state.editors.get(path, lambda p: _make_editor(p))


def _get_asc_editor(path: Path, state: SessionState) -> AscEditor:
    """Get a cached AscEditor. Caller must have validated _require_asc first."""
    editor = _get_editor(path, state)
    if not isinstance(editor, AscEditor):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")
    return editor


def _is_asc(path: Path) -> bool:
    return path.suffix.lower() == ".asc"


def _require_asc(path: Path) -> None:
    """Raise if path is not an .asc file (for schematic-only operations)."""
    if not _is_asc(path):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")


@asynccontextmanager
async def _editing(path: Path, state: SessionState) -> AsyncIterator[Editor]:
    """Get a cached editor, yield it, then save and invalidate on success.

    If the caller raises, changes are not saved (fail-safe).
    Uses per-file locking to prevent concurrent edits to the same file.
    """
    async with _get_edit_lock(path):
        editor = _get_editor(path, state)
        yield editor
        editor.save_netlist(str(path))
        state.editors.invalidate(path)


@asynccontextmanager
async def _editing_asc(path: Path, state: SessionState) -> AsyncIterator[AscEditor]:
    """Get a cached AscEditor, yield it, then save and invalidate on success.

    Caller must have validated _require_asc first.
    Uses per-file locking to prevent concurrent edits to the same file.
    """
    async with _get_edit_lock(path):
        editor = _get_asc_editor(path, state)
        yield editor
        editor.save_netlist(str(path))
        state.editors.invalidate(path)


# ---------------------------------------------------------------------------
# Handlers — shared operations (work on .cir/.net and .asc)
# ---------------------------------------------------------------------------


@registry.tool(
    name="ltspice_create_netlist",
    description=(
        "Create a new SPICE netlist file from content string. Automatically appends .END if missing."
    ),
    input_model=CreateNetlistInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_create_netlist(arguments: CreateNetlistInput, state: SessionState) -> types.CallToolResult:
    """Create a new SPICE netlist file from content string."""
    name = arguments.name
    content = arguments.content
    target_path = safe_path(f"{name}.cir", state)

    if target_path.exists():
        raise NetlistError(f"File already exists: {target_path}")

    if not content.strip().upper().endswith(".END"):
        content = content.rstrip() + "\n.END\n"

    target_path.write_text(content)

    try:
        editor = SpiceEditor(str(target_path))
        components = editor.get_components()
        comp_count = len(components)
    except Exception as e:
        target_path.unlink(missing_ok=True)
        raise NetlistError(f"Invalid netlist syntax: {e}") from e

    return text_response(f"Created netlist: {target_path}\nComponents: {comp_count}")


@registry.tool(
    name="ltspice_read_circuit",
    description=(
        "Read and parse a circuit file (.cir/.net or .asc). For netlists: returns content "
        "and component values. For schematics: returns layout and directives."
    ),
    input_model=CircuitReadInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_read_circuit(arguments: CircuitReadInput, state: SessionState):
    """Read and parse a circuit file. For .asc schematics, returns component
    positions, net labels, wires, and directives. For .cir/.net, returns raw
    content and component list with values.
    """
    file_path = safe_path(arguments.path, state)
    fmt = arguments.format

    if _is_asc(file_path):
        data = services.extract_asc_info(_get_asc_editor(file_path, state), file_path)
    else:
        data = services.extract_netlist_info(_get_editor(file_path, state), file_path)
    return format_response(_format_circuit_text(file_path, data), data, fmt)


def _format_circuit_text(file_path: Path, data: dict) -> str:
    """Build the human-readable circuit summary from structured data."""
    if data["type"] == "asc":
        lines = [f"=== {file_path.name} ===", ""]
        components = data["components"]
        lines.append(f"Components ({len(components)}):")
        for comp in components:
            lines.append(
                f"  {comp['reference']:<8} {comp['value']:<20} "
                f"pos=({comp['x']},{comp['y']}) {comp['rotation']}"
            )

        labels = data["labels"]
        if labels:
            lines.append("")
            lines.append(f"Net Labels ({len(labels)}):")
            for label in labels:
                lines.append(f"  {label['text']:<16} at ({label['x']},{label['y']})")

        lines.append("")
        lines.append(f"Wires: {data['wire_count']}")
        lines.append(f"Directives: {len(data['directives'])}")
        if data["directives"]:
            lines.append("")
            lines.append("SPICE Directives:")
            for directive in data["directives"]:
                lines.append(f"  {directive}")
        return "\n".join(lines)

    components = data["components"]
    if components:
        comp_summary = "\n".join(f"{comp['reference']}  {comp['value']}" for comp in components)
    else:
        comp_summary = "(no components)"
    return (
        f"=== {file_path.name} ===\n\n{data['content']}\n\n"
        f"=== Components ({len(components)}) ===\n{comp_summary}"
    )


@registry.tool(
    name="ltspice_list_components",
    description=(
        "List components in a circuit file, optionally filtered by type prefix, or "
        "return a single component value by reference."
    ),
    input_model=ListComponentsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
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
)
async def handle_list_components(arguments: ListComponentsInput, state: SessionState):
    """List all components, optionally filtered by prefix. If a single
    reference is provided, return just that component's value.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments.path, state)
    fmt = arguments.format

    editor = _get_editor(file_path, state)

    # Single-component lookup mode (absorbed from get_component_value)
    reference = arguments.reference
    if reference is not None:
        try:
            value = editor.get_component_value(reference)
        except Exception:
            raise NetlistError(f"Component '{reference}' not found") from None
        data = {"reference": reference, "value": value}
        return format_response(f"{reference} = {value}", data, fmt)

    # List mode
    prefix = arguments.prefix
    components = editor.get_components(prefix) if prefix else editor.get_components()

    if not components:
        msg = (
            f"No components matching prefix '{prefix}' found" if prefix else "No components found"
        )
        return format_response(
            msg, {"components": [], "pagination": pagination_metadata(0, 0, 50)}, fmt
        )

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


@registry.tool(
    name="ltspice_set_component_value",
    description="Set component value(s) in a circuit file. Supports single or batch mode.",
    input_model=SetComponentValueInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_set_component_value(arguments: SetComponentValueInput, state: SessionState) -> types.CallToolResult:
    """Set component value(s). Accepts single or batch mode.

    Single mode: provide 'reference' and 'value'.
    Batch mode: provide 'values' dict mapping references to new values.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments.path, state)

    values_dict = arguments.values
    reference = arguments.reference
    value = arguments.value

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
                raise NetlistError(f"Component '{reference}' not found") from None
            editor.set_component_value(reference, value)
            result = f"Set {reference}: {old_value} -> {value}"
        else:
            raise NetlistError(
                "Provide either 'reference'+'value' (single) or 'values' dict (batch)"
            )

    return text_response(result)


@registry.tool(
    name="ltspice_parameter",
    description="Read or write .PARAM directive values in a circuit file.",
    input_model=ParameterInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_parameter(arguments: ParameterInput, state: SessionState):
    """Get or set .PARAM directive values. Without name/value: returns all
    parameters. With name and value: sets the parameter.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(arguments.path, state)
    fmt = arguments.format

    param_name = arguments.name
    param_value = arguments.value

    if param_name is not None and param_value is not None:
        # Set mode — confirmation only, no structured data needed
        async with _editing(file_path, state) as editor:
            editor.set_parameter(param_name, param_value)
        return text_response(f"Set .PARAM {param_name} = {param_value}")

    # Get mode (formerly get_parameters) — read-only, no _editing needed
    editor = _get_editor(file_path, state)
    param_names = editor.get_all_parameter_names()

    params = {}
    if param_names:
        param_lines = []
        for name in param_names:
            value = editor.get_parameter(name)
            param_lines.append(f".PARAM {name} = {value}")
            params[name] = value
        result = "\n".join(param_lines)
    else:
        result = "No .PARAM directives found"

    return format_response(result, {"parameters": params}, fmt)


@registry.tool(
    name="ltspice_edit_directive",
    description="Add or remove a SPICE directive (.tran, .ac, .param, etc.).",
    input_model=EditDirectiveInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_edit_directive(arguments: EditDirectiveInput, state: SessionState) -> types.CallToolResult:
    """Add or remove a SPICE directive. Works on .cir/.net and .asc."""
    file_path = safe_path(arguments.path, state)

    action = arguments.action
    instruction = arguments.instruction

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


@registry.tool(
    name="ltspice_remove_component",
    description="Remove a component from an .asc schematic by reference designator.",
    input_model=RemoveComponentInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_remove_component(arguments: RemoveComponentInput, state: SessionState) -> types.CallToolResult:
    """Remove a component from a schematic by reference designator."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    reference = arguments.reference

    # Collect pin positions before removal to check for orphaned wires
    editor_pre = _get_asc_editor(asc_path, state)
    if reference not in editor_pre.get_components():
        raise NetlistError(
            f"Component '{reference}' not found. "
            f"Available: {', '.join(editor_pre.get_components())}"
        )
    comp = editor_pre.components[reference]
    sym_info = get_symbol_info(comp.symbol) if comp.symbol else None
    pin_coords: set[tuple[int, int]] = set()
    if sym_info is not None:
        pos, erot = editor_pre.get_component_position(reference)
        rot_str = erot.name if erot else "R0"
        geo = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
        pin_coords = {(p["x"], p["y"]) for p in geo["pins"]}

    async with _editing_asc(asc_path, state) as editor:
        editor.remove_component(reference)

    # Check for wires that touch the removed component's pin positions
    result = f"Removed {reference} from {asc_path.name}"
    if pin_coords:
        editor_post = _get_asc_editor(asc_path, state)
        orphaned_at: list[str] = []
        for w in editor_post.wires:
            for coord in [(int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))]:
                if coord in pin_coords:
                    orphaned_at.append(f"({coord[0]},{coord[1]})")
                    pin_coords.discard(coord)
        if orphaned_at:
            result += f"\n\nWarning: orphaned wires remain at: {', '.join(orphaned_at)}"

    return text_response(result)


@registry.tool(
    name="ltspice_move_component",
    description="Move and/or rotate a component in an .asc schematic.",
    input_model=MoveComponentInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_move_component(arguments: MoveComponentInput, state: SessionState) -> types.CallToolResult:
    """Move or rotate a component in a schematic."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    reference = arguments.reference
    x = arguments.x
    y = arguments.y
    rotation = arguments.rotation

    async with _editing_asc(asc_path, state) as editor:
        old_pos, old_rot = editor.get_component_position(reference)

        new_rot = _parse_rotation(rotation) if rotation is not None else old_rot

        new_pos = Point(x, y)
        editor.set_component_position(reference, new_pos, new_rot)

    rot_str = f"R{new_rot.value}" if new_rot.value < 360 else f"M{new_rot.value - 360}"
    return text_response(f"Moved {reference}: ({old_pos.X},{old_pos.Y}) -> ({x},{y}) {rot_str}")


@registry.tool(
    name="ltspice_set_component_attribute",
    description="Set a schematic-only component attribute such as SpiceLine or SpiceModel.",
    input_model=SetComponentAttributeInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_set_component_attribute(
    arguments: SetComponentAttributeInput, state: SessionState
) -> types.CallToolResult:
    """Set an attribute on a schematic component (e.g., SpiceLine, SpiceModel)."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    reference = arguments.reference
    attribute = arguments.attribute
    value = arguments.value

    async with _editing_asc(asc_path, state) as editor:
        editor.set_component_attribute(reference, attribute, value)

    return text_response(f"Set {reference}.{attribute} = {value}")


@registry.tool(
    name="ltspice_add_component",
    description="Add a new component to an .asc schematic at a specified grid position.",
    input_model=AddComponentInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_add_component(arguments: AddComponentInput, state: SessionState) -> types.CallToolResult:
    """Add a new component to an .asc schematic."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    reference = arguments.reference
    symbol = arguments.symbol
    x = arguments.x
    y = arguments.y
    value = arguments.value
    rotation = arguments.rotation
    erot = _parse_rotation(rotation)

    async with _editing_asc(asc_path, state) as editor:
        if reference in editor.components:
            raise NetlistError(
                f"Component '{reference}' already exists. "
                "Use ltspice_set_component_value to modify it, "
                "or ltspice_remove_component to remove it first."
            )

        comp = SchematicComponent(editor, "")
        comp.reference = reference
        comp.symbol = symbol  # pyright: ignore[reportAttributeAccessIssue]
        comp.position = Point(x, y)
        comp.rotation = erot
        if value is not None:
            comp.attributes["Value"] = value
        if arguments.attributes:
            for attr_name, attr_val in arguments.attributes.items():
                comp.attributes[attr_name] = attr_val

        editor.add_component(comp)

    result = f"Added {reference} ({symbol}) at ({x},{y})"
    if value is not None:
        result += f" = {value}"

    # Compute pin positions and bounding box
    sym_info = get_symbol_info(symbol)
    if sym_info is None:
        return text_response(result)

    geometry = compute_placed_geometry(sym_info, x, y, rotation)
    for pin in geometry["pins"]:
        result += f"\n  {pin['name']}: ({pin['x']}, {pin['y']}) [{pin['dir']}]"
    bb = geometry["bounding_box"]
    result += f"\n  bbox: ({bb['x']},{bb['y']}) {bb['width']}x{bb['height']}"

    # Check for overlap with existing components
    warnings: list[str] = []
    for ebb in _collect_component_geometry(_get_asc_editor(asc_path, state)):
        if ebb["ref"] == reference:
            continue
        if _bboxes_overlap(bb, ebb):
            warnings.append(f"Overlaps {ebb['ref']} bounding box")

    if warnings:
        result += "\n\nWarnings:"
        for w in warnings:
            result += f"\n  {w}"

    data: dict = {
        "reference": reference,
        "symbol": symbol,
        "position": {"x": x, "y": y},
        "rotation": rotation,
        "pins": geometry["pins"],
        "bounding_box": geometry["bounding_box"],
    }
    if warnings:
        data["warnings"] = warnings

    return format_response(result, data, None)


_previous_exports: dict[Path, list[str]] = {}


@registry.tool(
    name="ltspice_export_netlist",
    description="Export an .asc schematic to a SPICE netlist (.net) using LTspice.",
    input_model=ExportNetlistInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_export_netlist(arguments: ExportNetlistInput, state: SessionState) -> types.CallToolResult:
    """Export an .asc schematic to a SPICE netlist (.net) using LTspice."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    ltspice_cls = state.available_simulators.get("ltspice")
    if ltspice_cls is None:
        raise NetlistError(
            "export_netlist requires LTspice to convert .asc to netlist. "
            "Available simulators: " + str(list(state.available_simulators.keys()))
        )

    try:
        net_path = ltspice_cls.create_netlist(str(asc_path))
        net_path = Path(net_path)
    except Exception as e:
        raise NetlistError(f"LTspice netlist export failed: {e}") from e

    if not net_path.exists():
        raise NetlistError("Export failed: .net file not created")

    content = net_path.read_text()
    current_lines = content.splitlines()

    result = f"=== {net_path.name} ===\n\n{content}"

    # Show diff if a previous export exists for this file
    prev = _previous_exports.get(asc_path)
    if prev is not None:
        added = [ln for ln in current_lines if ln not in prev and not ln.startswith("*")]
        removed = [ln for ln in prev if ln not in current_lines and not ln.startswith("*")]
        if added or removed:
            result += "\n\n--- Changes since last export ---"
            for ln in removed:
                result += f"\n- {ln}"
            for ln in added:
                result += f"\n+ {ln}"

    _previous_exports[asc_path] = current_lines

    return text_response(result)


@registry.tool(
    name="ltspice_get_symbol_info",
    description=(
        "Get symbol pin positions, bounding box, and description. "
        "Optionally compute absolute positions for a given placement and rotation."
    ),
    input_model=SymbolInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
)
async def handle_get_symbol_info(
    arguments: SymbolInfoInput, state: SessionState
) -> types.CallToolResult:
    """Get symbol geometry info for schematic layout planning."""
    symbol = arguments.symbol
    sym_info = get_symbol_info(symbol)
    if sym_info is None:
        raise NetlistError(
            f"Symbol '{symbol}' not found. Ensure LTspice symbol libraries are configured."
        )

    geometry = compute_placed_geometry(sym_info, arguments.x, arguments.y, arguments.rotation)
    data = {
        **sym_info.to_dict(),
        "placement": {"x": arguments.x, "y": arguments.y, "rotation": arguments.rotation},
        "absolute_pins": geometry["pins"],
        "absolute_bounding_box": geometry["bounding_box"],
    }

    lines = [f"Symbol: {sym_info.name}"]
    if sym_info.description:
        lines.append(f"Description: {sym_info.description}")
    lines.append(f"Size: {sym_info.bbox_width}x{sym_info.bbox_height}")
    lines.append(f"Pins (at {arguments.rotation}, origin ({arguments.x},{arguments.y})):")
    for pin in geometry["pins"]:
        lines.append(f"  {pin['name']}: ({pin['x']}, {pin['y']})")
    bb = geometry["bounding_box"]
    lines.append(f"Bounding box: ({bb['x']},{bb['y']}) {bb['width']}x{bb['height']}")

    return format_response("\n".join(lines), data, None)


@registry.tool(
    name="ltspice_get_component_info",
    description=(
        "Get a placed component's pin positions, bounding box, value, and attributes "
        "from an .asc schematic."
    ),
    input_model=ComponentInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
)
async def handle_get_component_info(
    arguments: ComponentInfoInput, state: SessionState
) -> types.CallToolResult:
    """Get full info about a placed component including computed pin positions."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)
    reference = arguments.reference

    editor = _get_asc_editor(asc_path, state)
    component_refs = editor.get_components()
    if reference not in component_refs:
        raise NetlistError(
            f"Component '{reference}' not found. "
            f"Available: {', '.join(sorted(component_refs))}"
        )

    pos, erot = editor.get_component_position(reference)
    rot_str = erot.name if erot else "R0"
    # Access the SchematicComponent object for symbol and attributes
    comp = editor.components[reference]
    symbol = comp.symbol

    value = None
    with contextlib.suppress(Exception):
        value = editor.get_component_value(reference)

    data: dict = {
        "reference": reference,
        "symbol": symbol,
        "position": {"x": pos.X, "y": pos.Y},
        "rotation": rot_str,
        "value": value,
    }

    lines = [f"{reference} ({symbol}) at ({pos.X},{pos.Y}) {rot_str}"]
    if value:
        lines.append(f"Value: {value}")

    # Compute pin positions from symbol geometry
    sym_info = get_symbol_info(symbol) if symbol else None
    if sym_info is not None:
        geometry = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
        data["pins"] = geometry["pins"]
        data["bounding_box"] = geometry["bounding_box"]
        lines.append("Pins:")
        for pin in geometry["pins"]:
            lines.append(f"  {pin['name']}: ({pin['x']}, {pin['y']})")
        bb = geometry["bounding_box"]
        lines.append(f"Bounding box: ({bb['x']},{bb['y']}) {bb['width']}x{bb['height']}")

    # Include non-trivial attributes
    for attr_name, attr_val in comp.attributes.items():
        if attr_name not in ("Value", "InstName") and attr_val:
            data.setdefault("attributes", {})[attr_name] = attr_val
            lines.append(f"{attr_name}: {attr_val}")

    return format_response("\n".join(lines), data, None)


def _resolve_pin(
    pin_ref: str, editor: AscEditor
) -> tuple[int, int]:
    """Resolve a pin reference ('M1.D' or 'net:VDD') to absolute (x, y) coordinates.

    Raises NetlistError if the reference cannot be resolved.
    """
    if pin_ref.startswith("net:"):
        # Look up a FLAG/net label position in the .asc
        net_name = pin_ref[4:]
        matches = [
            (int(lbl.coord.X), int(lbl.coord.Y))
            for lbl in editor.labels
            if lbl.text == net_name
        ]
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. "
                "Add it with ltspice_add_net_label first."
            )
        if len(matches) > 1:
            coords = ", ".join(f"({x},{y})" for x, y in matches)
            raise NetlistError(
                f"Multiple '{net_name}' labels found at: {coords}. "
                "Use a unique net label, or connect directly to a component pin."
            )
        return matches[0]

    # Component.Pin format
    if "." not in pin_ref:
        raise NetlistError(
            f"Invalid pin reference '{pin_ref}'. "
            "Use 'Reference.Pin' (e.g., 'M1.D') or 'net:name' (e.g., 'net:VDD')."
        )

    ref, pin_name = pin_ref.rsplit(".", 1)
    component_refs = editor.get_components()
    if ref not in component_refs:
        raise NetlistError(
            f"Component '{ref}' not found. Available: {', '.join(sorted(component_refs))}"
        )

    pos, erot = editor.get_component_position(ref)
    rot_str = erot.name if erot else "R0"
    comp = editor.components[ref]
    symbol = comp.symbol

    sym_info = get_symbol_info(symbol) if symbol else None
    if sym_info is None:
        raise NetlistError(f"Cannot resolve pins for '{ref}': symbol '{symbol}' not found.")

    geometry = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
    for pin in geometry["pins"]:
        if pin["name"].upper() == pin_name.upper():
            return pin["x"], pin["y"]

    available = [p["name"] for p in geometry["pins"]]
    raise NetlistError(
        f"Pin '{pin_name}' not found on {ref} ({symbol}). Available: {', '.join(available)}"
    )


@registry.tool(
    name="ltspice_add_net_label",
    description="Add a net label or ground flag to an .asc schematic at a wire junction.",
    input_model=NetLabelInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_add_net_label(
    arguments: NetLabelInput, state: SessionState
) -> types.CallToolResult:
    """Add or remove a FLAG (net label or ground) in a schematic."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    net = arguments.net
    label_desc = "ground" if net == "0" else f"net '{net}'"

    # Resolve coordinates from pin reference or explicit x/y
    if arguments.pin is not None:
        editor = _get_asc_editor(asc_path, state)
        x, y = _resolve_pin(arguments.pin, editor)
    elif arguments.x is not None and arguments.y is not None:
        x, y = arguments.x, arguments.y
    else:
        raise NetlistError("Either pin or both x and y coordinates are required.")

    if arguments.action == "remove":
        async with _editing_asc(asc_path, state) as editor:
            for i, lbl in enumerate(editor.labels):
                if lbl.text == net and int(lbl.coord.X) == x and int(lbl.coord.Y) == y:
                    editor.labels.pop(i)
                    return text_response(f"Removed {label_desc} at ({x},{y})")
            raise NetlistError(f"No {label_desc} found at ({x},{y})")

    # Add mode
    result = ""
    async with _editing_asc(asc_path, state) as editor:
        # Warn on duplicate non-ground labels
        if net != "0":
            for lbl in editor.labels:
                if lbl.text == net:
                    result = (
                        f"Warning: '{net}' already exists at "
                        f"({int(lbl.coord.X)},{int(lbl.coord.Y)}). "
                        "Multiple labels with the same name will cause "
                        "ltspice_connect to error on ambiguity.\n"
                    )
                    break

        label = Text(coord=Point(x, y), text=net, type=TextTypeEnum.LABEL)
        editor.labels.append(label)

    result += f"Added {label_desc} at ({x},{y})"
    return text_response(result)


@registry.tool(
    name="ltspice_add_text",
    description="Add a comment text annotation to an .asc schematic.",
    input_model=AddTextInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_add_text(
    arguments: AddTextInput, state: SessionState
) -> types.CallToolResult:
    """Add a comment text to a schematic."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    async with _editing_asc(asc_path, state) as editor:
        comment = Text(
            coord=Point(arguments.x, arguments.y),
            text=arguments.text,
            type=TextTypeEnum.COMMENT,
            size=arguments.size,
        )
        editor.directives.append(comment)

    return text_response(f"Added text at ({arguments.x},{arguments.y}): {arguments.text}")


@registry.tool(
    name="ltspice_connect",
    description=(
        "Connect two component pins with wire(s). Resolves pin positions automatically. "
        "Waypoints define the wire route through intermediate points. "
        "For a straight horizontal or vertical connection, waypoints can be omitted."
    ),
    input_model=ConnectInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
)
async def handle_connect(
    arguments: ConnectInput, state: SessionState
) -> types.CallToolResult:
    """Connect two pins with auto-routed or waypoint-guided wires."""
    asc_path = safe_path(arguments.path, state)
    _require_asc(asc_path)

    # Collect component bounding boxes and existing wires for crossing detection
    pre_editor = _get_asc_editor(asc_path, state)
    component_bboxes = _collect_component_geometry(pre_editor)
    existing_wires = [
        (int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in pre_editor.wires
    ]

    async with _editing_asc(asc_path, state) as ed:
        x1, y1 = _resolve_pin(arguments.from_pin, ed)
        x2, y2 = _resolve_pin(arguments.to_pin, ed)

        # Build list of points: from → [waypoints] → to (dedup consecutive)
        raw_points = [(x1, y1)]
        for wp in arguments.waypoints:
            raw_points.append((wp.x, wp.y))
        raw_points.append((x2, y2))
        points = [raw_points[0]]
        for pt in raw_points[1:]:
            if pt != points[-1]:
                points.append(pt)

        # Create wire segments between consecutive points
        segments: list[tuple[int, int, int, int]] = []
        for i in range(len(points) - 1):
            px1, py1 = points[i]
            px2, py2 = points[i + 1]
            if px1 == px2 and py1 == py2:
                continue  # skip zero-length
            ed.wires.append(Line(Point(px1, py1), Point(px2, py2)))
            segments.append((px1, py1, px2, py2))

    # Compute warnings
    warnings: list[str] = []

    for sx1, sy1, sx2, sy2 in segments:
        if sx1 != sx2 and sy1 != sy2:
            warnings.append(f"Diagonal wire ({sx1},{sy1})->({sx2},{sy2}): not orthogonal")

    total_length = sum(abs(sx2 - sx1) + abs(sy2 - sy1) for sx1, sy1, sx2, sy2 in segments)
    if total_length > 400:
        warnings.append(
            f"Long wire run ({total_length} units): consider placing components closer "
            "or adding a local net label"
        )

    # Check bounding box crossings (skip components that own the from/to pins)
    skip_refs = {
        ref.rsplit(".", 1)[0]
        for ref in (arguments.from_pin, arguments.to_pin)
        if "." in ref and not ref.startswith("net:")
    }

    for sx1, sy1, sx2, sy2 in segments:
        for bb in component_bboxes:
            if bb["ref"] in skip_refs:
                continue
            bx, by, bw, bh = bb["x"], bb["y"], bb["width"], bb["height"]
            # Check if wire segment intersects bounding box interior
            # For horizontal wire (sy1 == sy2)
            if sy1 == sy2:
                wy = sy1
                wx_min, wx_max = min(sx1, sx2), max(sx1, sx2)
                if by < wy < by + bh and wx_min < bx + bw and wx_max > bx:
                    warnings.append(
                        f"Wire at y={wy} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )
            # For vertical wire (sx1 == sx2)
            elif sx1 == sx2:
                wx = sx1
                wy_min, wy_max = min(sy1, sy2), max(sy1, sy2)
                if bx < wx < bx + bw and wy_min < by + bh and wy_max > by:
                    warnings.append(
                        f"Wire at x={wx} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )

    # Check if wire passes through any component pin (not from/to endpoints)
    endpoints = {(x1, y1), (x2, y2)}
    for cg in component_bboxes:
        if cg["ref"] in skip_refs:
            continue
        for pin in cg["pins"]:
            px, py = pin["x"], pin["y"]
            if (px, py) in endpoints:
                continue
            for sx1, sy1, sx2, sy2 in segments:
                on_wire = False
                if sy1 == sy2 and py == sy1:
                    on_wire = min(sx1, sx2) <= px <= max(sx1, sx2)
                elif sx1 == sx2 and px == sx1:
                    on_wire = min(sy1, sy2) <= py <= max(sy1, sy2)
                if on_wire:
                    warnings.append(
                        f"Wire passes through {cg['ref']}.{pin['name']} at ({px},{py}): "
                        "will create unintended connection"
                    )

    # Check for unintended junctions with existing wires
    for sx1, sy1, sx2, sy2 in segments:
        for ex1, ey1, ex2, ey2 in existing_wires:
            # Check if any interior point of the new segment lies on an existing wire
            # (or vice versa). Only flag points that aren't the intended endpoints.
            if sx1 == sx2 and ex1 == ex2 and sx1 == ex1:
                # Both vertical, same x — check y overlap
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_y = max(new_min, ext_min)
                    if (sx1, overlap_y) not in endpoints:
                        warnings.append(
                            f"Wire overlap at x={sx1} between y={max(new_min, ext_min)} "
                            f"and y={min(new_max, ext_max)}: may create unintended junction"
                        )
                        break
            elif sy1 == sy2 and ey1 == ey2 and sy1 == ey1:
                # Both horizontal, same y — check x overlap
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_x = max(new_min, ext_min)
                    if (overlap_x, sy1) not in endpoints:
                        warnings.append(
                            f"Wire overlap at y={sy1} between x={max(new_min, ext_min)} "
                            f"and x={min(new_max, ext_max)}: may create unintended junction"
                        )
                        break
            elif sx1 == sx2 and ey1 == ey2:
                # New vertical, existing horizontal — check cross point
                cross_x, cross_y = sx1, ey1
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if (
                    new_min < cross_y < new_max
                    and ext_min < cross_x < ext_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    warnings.append(
                        f"Wire crosses existing wire at ({cross_x},{cross_y}): "
                        "may create unintended junction"
                    )
            elif sy1 == sy2 and ex1 == ex2:
                # New horizontal, existing vertical — check cross point
                cross_x, cross_y = ex1, sy1
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if (
                    ext_min < cross_y < ext_max
                    and new_min < cross_x < new_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    warnings.append(
                        f"Wire crosses existing wire at ({cross_x},{cross_y}): "
                        "may create unintended junction"
                    )

    result_lines = [f"Connected {arguments.from_pin} to {arguments.to_pin}"]
    result_lines.append(f"  From: ({x1},{y1})  To: ({x2},{y2})")
    for sx1, sy1, sx2, sy2 in segments:
        result_lines.append(f"  Wire: ({sx1},{sy1})->({sx2},{sy2})")

    if warnings:
        result_lines.append("")
        result_lines.append("Warnings:")
        for w in warnings:
            result_lines.append(f"  {w}")

    data: dict = {
        "from": {"ref": arguments.from_pin, "x": x1, "y": y1},
        "to": {"ref": arguments.to_pin, "x": x2, "y": y2},
        "wire_count": len(segments),
        "points": [{"x": p[0], "y": p[1]} for p in points],
    }
    if warnings:
        data["warnings"] = warnings

    return format_response("\n".join(result_lines), data, None)
