"""Unified circuit editing tools for .cir/.net netlists and .asc schematics.

Extension-based dispatch: the file extension determines which spicelib editor
is used (SpiceEditor for .cir/.net, AscEditor for .asc).  Schematic-only
operations (position, rotation, attributes, export) validate the extension
and raise NetlistError if given a non-.asc file.
"""

import asyncio
import bisect
import contextlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

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
from ltspice_mcp.lib import atomic_write_text, services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.spice_validator import validate_directive
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, get_symbol_info
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    BBOX_SCHEMA,
    PAGINATION_SCHEMA,
    PIN_SCHEMA,
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


def _create_component(
    editor: AscEditor,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    rotation: ERotation,
    *,
    value: str | None = None,
    attributes: dict[str, str] | None = None,
) -> None:
    """Create and add a SchematicComponent to an AscEditor.

    Wraps the fragile pattern of constructing a blank SchematicComponent
    then manually setting .reference, .symbol, .position, .rotation.
    """
    comp = SchematicComponent(editor, "")
    comp.reference = reference
    comp.symbol = symbol  # pyright: ignore[reportAttributeAccessIssue]
    comp.position = Point(x, y)
    comp.rotation = rotation
    if value is not None:
        comp.attributes["Value"] = value
    if attributes:
        for attr_name, attr_val in attributes.items():
            comp.attributes[attr_name] = attr_val
    editor.add_component(comp)


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


# Matches one ``KEY=VALUE`` token (value may have braces, parens, sign, etc.).
# Used to peel trailing parameters off a multi-token component value like
# ``"NMOS1 W=10u L=1u"`` so we can route each piece through the right
# spicelib API (model name → ``set_component_value``; W/L → ``set_component_parameters``).
_PARAM_TOKEN_RE = re.compile(r"(\w+)\s*=\s*([^\s=]+)")


def _validate_component_value(reference: str, value: str) -> None:
    """Reject values that would corrupt the netlist line on write.

    spicelib writes the value verbatim into the component line; spaces in
    a non-parameterised, non-quoted value bleed into a phantom node and
    irrecoverably break the netlist (Bug L). The check is permissive of:
    - SPICE expressions in braces (``{1/(2*pi*RC)}``) — braces protect spaces
    - quoted strings (``"a b"``)
    - ``KEY=VALUE`` parameter lists (handled by ``_apply_component_value``)
    """
    if not isinstance(value, str):  # type: ignore[reportUnnecessaryIsInstance]
        # Pydantic should have rejected non-strings already, but guard
        # anyway since this writes to disk verbatim.
        raise NetlistError(
            f"Component '{reference}' value must be a string, got {type(value).__name__}"
        )
    stripped = value.strip()
    if not stripped:
        raise NetlistError(f"Component '{reference}' value must not be empty")
    if "\n" in stripped or "\r" in stripped:
        raise NetlistError(
            f"Component '{reference}' value must be a single line; "
            f"got embedded newline in {value!r}"
        )
    # Brace-balanced expression or quoted literal — spaces are safe.
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return
    # ``[MODEL_NAME] KEY=VALUE [KEY=VALUE ...]`` is valid: at most one bare
    # head token (the model name) followed by a non-empty list of KEY=VALUE
    # tokens. The pure-params and head+params forms collapse into one rule.
    if "=" in stripped:
        tokens = stripped.split()
        head_tokens: list[str] = []
        for tok in tokens:
            if "=" in tok:
                break
            head_tokens.append(tok)
        rest = tokens[len(head_tokens) :]
        if (
            len(head_tokens) <= 1
            and rest
            and all(bool(_PARAM_TOKEN_RE.fullmatch(tok)) for tok in rest)
        ):
            return
    if any(c.isspace() for c in stripped):
        raise NetlistError(
            f"Component '{reference}' value {value!r} contains whitespace. "
            "Wrap SPICE expressions in braces ({...}) or use the parameter "
            "form (e.g. 'NMOS1 W=10u L=1u'). A bare space-separated value "
            "would corrupt the netlist line."
        )


def _apply_component_value(editor, reference: str, value: str) -> None:
    """Set a component's value, splitting trailing ``KEY=VALUE`` tokens off.

    spicelib's ``set_component_value`` writes only the model/value field of
    the element line — it does NOT touch the trailing parameter section.
    Calling it with ``"NMOS1 W=10u L=1u"`` against an existing ``M1 ... NMOS1 W=20u L=1u``
    leaves both sets in place (``... NMOS1 W=10u L=1u W=20u L=1u``), which
    LTspice may parse either way. To DWIM, we split off any ``KEY=VALUE``
    tokens and route them through ``set_component_parameters`` (which edits
    the params section via the same regex), keeping the model/value field
    for ``set_component_value``.
    """
    _validate_component_value(reference, value)
    if "=" not in value:
        editor.set_component_value(reference, value)
        return
    params: dict[str, str] = {}
    head_end = len(value)
    for m in _PARAM_TOKEN_RE.finditer(value):
        params[m.group(1)] = m.group(2)
        head_end = min(head_end, m.start())
    head = value[:head_end].strip()
    if head:
        editor.set_component_value(reference, head)
    if params:
        editor.set_component_parameters(reference, **params)


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """AABB overlap test between two bounding boxes with {x, y, width, height}."""
    return (
        a["x"] < b["x"] + b["width"]
        and a["x"] + a["width"] > b["x"]
        and a["y"] < b["y"] + b["height"]
        and a["y"] + a["height"] > b["y"]
    )


def _format_available_refs(refs: list[str] | set[str], cap: int = 20) -> str:
    """Format a component-reference list for "Available: ..." error messages.

    Caps the displayed list so errors on large schematics don't explode into
    hundreds of refs.
    """
    sorted_refs = sorted(refs)
    if len(sorted_refs) > cap:
        return ", ".join(sorted_refs[:cap]) + f", ... ({len(sorted_refs)} total)"
    return ", ".join(sorted_refs)


def _require_component(editor: "AscEditor | SpiceEditor", reference: str) -> list[str]:
    """Verify a component reference exists in the editor.

    Calls ``editor.get_components()`` exactly once and reuses the result for
    both the membership check and the "Available: ..." error message, avoiding
    the redundant scans that several handlers used to do.

    Returns the component list so callers can reuse it.
    """
    comps = editor.get_components()
    if reference not in comps:
        raise NetlistError(
            f"Component '{reference}' not found. Available: {_format_available_refs(comps)}"
        )
    return comps


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
    overwrite: bool = Field(
        default=False,
        description="Overwrite an existing file at this path. Default is to refuse.",
    )


class CircuitReadInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class ListComponentsInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    prefix: str | None = Field(
        default=None, description="Filter by reference prefix (e.g., 'R', 'M', 'C')"
    )
    reference: str | None = Field(
        default=None, description="Look up a single component by reference (e.g., 'R1')"
    )
    offset: int = Field(default=0, description="Pagination offset")
    limit: int = Field(default=50, description="Max results to return")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class SetComponentValueInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    reference: str | None = Field(
        default=None, description="Component reference for single mode (e.g., 'R1')"
    )
    value: str | None = Field(
        default=None, description="New value for single mode (e.g., '10k', '100n')"
    )
    values: dict[str, str] | None = Field(
        default=None,
        description="Batch mode: {reference: value} dict (e.g., {'R1': '10k', 'C1': '100n'})",
    )


class ParameterInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    name: str | None = Field(
        default=None, description="Parameter name to set (omit to read all params)"
    )
    value: str | None = Field(
        default=None, description="Parameter value (required when name is specified)"
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class EditDirectiveInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    action: Literal["add", "remove"] = Field(description="Whether to add or remove the directive")
    instruction: str = Field(
        description=(
            "SPICE directive text (e.g., '.tran 10m', '.ac dec 100 1 1G'). "
            "For ``kind='comment'`` this is the comment text instead. "
            "For remove: exact match or regex with 'regex:' prefix; the "
            "match is run against directives AND comments so callers can "
            "remove either kind without knowing which it is."
        ),
    )
    kind: Literal["directive", "comment"] = Field(
        default="directive",
        description=(
            "``directive`` (default) — emit a SPICE directive line. "
            "``comment`` — emit a free-text annotation. .asc-only; the "
            "tool refuses ``kind='comment'`` on .cir/.net since plain "
            "netlists already accept ``*`` / ``;`` comments inline."
        ),
    )
    x: int | None = Field(
        default=None,
        description=(
            "Optional X coordinate when adding to an .asc schematic. "
            "Default places the directive in the lower-left corner."
        ),
    )
    y: int | None = Field(
        default=None,
        description="Optional Y coordinate (see ``x``).",
    )
    size: int = Field(
        default=2,
        description="Font size (.asc only). 1=small, 2=normal, 3=large.",
    )


class RemoveComponentInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Component reference to remove (e.g., 'R1', 'M3')")


class MoveComponentInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Component reference to move (e.g., 'R1', 'M3')")
    x: int = Field(description="New X coordinate (LTspice grid units)")
    y: int = Field(description="New Y coordinate (LTspice grid units)")
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] | None = Field(
        default=None, description="New rotation (omit to keep current)"
    )


class SetComponentAttributeInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Component reference (e.g., 'M1', 'R1')")
    attribute: str = Field(
        description="Attribute name (e.g., 'SpiceLine', 'SpiceModel', 'Value2')"
    )
    value: str = Field(description="Attribute value (e.g., 'W=10u L=0.5u')")


class ExportNetlistInput(ToolInput):
    path: str = Field(description="Path to .asc schematic to export")


class AddComponentInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Reference designator (e.g., 'M1', 'R3', 'VDD')")
    symbol: str = Field(description="Symbol name (e.g., 'nmos', 'pmos', 'res', 'cap', 'voltage')")
    x: int = Field(description="X coordinate (LTspice grid units)")
    y: int = Field(description="Y coordinate (LTspice grid units)")
    value: str | None = Field(
        default=None, description="Component value (e.g., '10k', 'NMOS_3V3')"
    )
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] = Field(
        default="R0", description="Rotation/mirror (PMOS typically M180, NMOS typically R0)"
    )
    attributes: dict[str, str] | None = Field(
        default=None,
        description="Optional attributes to set (e.g., {'SpiceLine': 'W=10u L=0.5u', 'Value2': '...'})",
    )


class NetLabelInput(ToolInput):
    path: str
    net: str = Field(description="Net name ('0' for ground, or a name like 'VDD', 'outp')")
    x: int | None = Field(
        default=None, description="X coordinate (required unless pin is specified)"
    )
    y: int | None = Field(
        default=None, description="Y coordinate (required unless pin is specified)"
    )
    pin: str | None = Field(
        default=None,
        description="Component pin reference (e.g., 'M3.S') — places label at the pin's coordinates",
    )
    action: Literal["add", "remove"] = "add"


class WaypointInput(StrictModel):
    x: int = Field(description="X coordinate of waypoint")
    y: int = Field(description="Y coordinate of waypoint")


class ConnectInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
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
    x: int = Field(
        default=0, description="Placement X coordinate (for computing absolute positions)"
    )
    y: int = Field(
        default=0, description="Placement Y coordinate (for computing absolute positions)"
    )
    rotation: Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"] = "R0"
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class ComponentInfoInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Component reference (e.g., 'M1', 'R1')")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


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
async def handle_create_netlist(
    args: CreateNetlistInput, state: SessionState
) -> types.CallToolResult:
    """Create a new SPICE netlist file from content string."""
    name = args.name
    content = args.content
    target_path = safe_path(f"{name}.cir", state)

    # Friction N: pre-flight every directive line through the same Layer-A
    # validator that ``edit_directive`` uses, so known-bad patterns
    # (vdb()/phase()/group_delay() inside .MEAS, etc.) are refused at
    # write time rather than only after a wasted simulation run.
    bad_directives: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line.startswith("."):
            continue
        err = validate_directive(line, simulator="LTspice")
        if err is not None:
            bad_directives.append(f"  {line}\n    {err.message}\n    Suggestion: {err.suggestion}")
    if bad_directives:
        joined = "\n".join(bad_directives)
        raise NetlistError(
            "Refusing to create netlist; one or more directives are known "
            "to fail in LTspice:\n" + joined
        )

    if not content.strip().upper().endswith(".END"):
        content = content.rstrip() + "\n.END\n"

    try:
        atomic_write_text(target_path, content, overwrite=args.overwrite, durable=False)
    except FileExistsError as e:
        raise NetlistError(
            f"File already exists: {target_path}. Pass overwrite=true to replace it."
        ) from e

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
    output_schema={
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "type": {"type": "string", "enum": ["asc", "netlist"]},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string"},
                        "value": {"type": "string"},
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "rotation": {"type": "string"},
                    },
                },
            },
            "content": {"type": "string"},
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
            "wire_count": {"type": "integer"},
            "directives": {"type": "array", "items": {"type": "string"}},
        },
    },
)
async def handle_read_circuit(args: CircuitReadInput, state: SessionState):
    """Read and parse a circuit file. For .asc schematics, returns component
    positions, net labels, wires, and directives. For .cir/.net, returns raw
    content and component list with values.
    """
    file_path = safe_path(args.path, state)
    fmt = args.format

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
async def handle_list_components(args: ListComponentsInput, state: SessionState):
    """List all components, optionally filtered by prefix. If a single
    reference is provided, return just that component's value.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(args.path, state)
    fmt = args.format

    if args.reference is not None and args.prefix is not None:
        raise NetlistError(
            "'reference' (single lookup) and 'prefix' (filter) are mutually "
            "exclusive — provide one, not both."
        )

    editor = _get_editor(file_path, state)

    # Single-component lookup mode (absorbed from get_component_value)
    reference = args.reference
    if reference is not None:
        try:
            value = editor.get_component_value(reference)
        except Exception:
            raise NetlistError(f"Component '{reference}' not found") from None
        data = {"reference": reference, "value": value}
        return format_response(f"{reference} = {value}", data, fmt)

    # A prefix containing regex metacharacters or more than one character
    # would otherwise reach spicelib's parser which raises a raw
    # NotImplementedError out of our error hierarchy.
    prefix = args.prefix
    if prefix is not None and (len(prefix) != 1 or not prefix.isalpha()):
        raise NetlistError(
            f"Component prefix must be a single letter (e.g. 'R', 'C'), got {prefix!r}"
        )

    try:
        components = editor.get_components(prefix) if prefix else editor.get_components()
    except Exception as e:
        raise NetlistError(f"Failed to list components: {e}") from e

    if not components:
        msg = (
            f"No components matching prefix '{prefix}' found" if prefix else "No components found"
        )
        return format_response(
            msg, {"components": [], "pagination": pagination_metadata(0, 0, 50)}, fmt
        )

    page, total, offset, limit = paginate(components, args)

    comp_list = []
    comp_lines = []
    for comp_ref in page:
        try:
            value = editor.get_component_value(comp_ref)
        except Exception:
            # spicelib's component-line regex chokes on B-sources with
            # commas in if(...) expressions; degrade gracefully rather
            # than abort the whole listing (Bug K).
            value = "<unparseable>"
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
async def handle_set_component_value(
    args: SetComponentValueInput, state: SessionState
) -> types.CallToolResult:
    """Set component value(s). Accepts single or batch mode.

    Single mode: provide 'reference' and 'value'.
    Batch mode: provide 'values' dict mapping references to new values.
    Works on .cir/.net and .asc.
    """
    file_path = safe_path(args.path, state)

    values_dict = args.values
    reference = args.reference
    value = args.value

    # Reject ambiguous input: single and batch mode are mutually exclusive.
    single_mode_args = reference is not None or value is not None
    if values_dict is not None and single_mode_args:
        raise NetlistError(
            "Single mode ('reference'+'value') and batch mode ('values') "
            "are mutually exclusive — provide one, not both."
        )

    async with _editing(file_path, state) as editor:
        if values_dict is not None:
            # Batch mode — validate every (ref, value) pair BEFORE writing
            # anything, so a bad ref or unparseable value doesn't corrupt
            # earlier successful writes (Bug J).
            if not isinstance(values_dict, dict):
                raise NetlistError("'values' must be an object mapping references to new values")
            if not values_dict:
                raise NetlistError("'values' dict must not be empty")
            unknown_refs: list[str] = []
            for ref in values_dict:
                try:
                    editor.get_component_value(ref)
                except Exception:
                    unknown_refs.append(ref)
            if unknown_refs:
                raise NetlistError(
                    "Component(s) not found: " + ", ".join(repr(r) for r in unknown_refs)
                )
            # Validate value syntax up-front so we don't half-apply.
            for ref, val in values_dict.items():
                _validate_component_value(ref, val)
            for ref, val in values_dict.items():
                _apply_component_value(editor, ref, val)
            changes = [f"{ref}: {val}" for ref, val in values_dict.items()]
            result = f"Updated {len(values_dict)} component(s):\n" + "\n".join(changes)
        elif reference is not None and value is not None:
            # Single mode
            try:
                old_value = editor.get_component_value(reference)
            except Exception:
                raise NetlistError(f"Component '{reference}' not found") from None
            _apply_component_value(editor, reference, value)
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
    output_schema={
        "type": "object",
        "properties": {
            "parameters": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
    },
)
async def handle_parameter(args: ParameterInput, state: SessionState):
    """Get or set .PARAM directive values.

    Modes:
      - no args         → list every .PARAM in the file
      - name only       → read a single parameter's value
      - name and value  → set a parameter (creates it if missing)

    Providing value without name is an error. Works on .cir/.net and .asc.
    """
    file_path = safe_path(args.path, state)
    fmt = args.format

    param_name = args.name
    param_value = args.value

    if param_name is not None and not param_name.strip():
        raise NetlistError("Parameter name must not be empty")

    if param_value is not None and param_name is None:
        raise NetlistError("'value' requires 'name' — cannot set a parameter without a name")

    if param_name is not None and param_value is not None:
        async with _editing(file_path, state) as editor:
            editor.set_parameter(param_name, param_value)
        return format_response(
            f"Set .PARAM {param_name} = {param_value}",
            {"parameters": {param_name: param_value}},
            fmt,
        )

    editor = _get_editor(file_path, state)

    if param_name is not None:
        value = None
        with contextlib.suppress(Exception):
            value = editor.get_parameter(param_name)
        if value is None:
            raise NetlistError(f"Parameter '{param_name}' not found in {file_path.name}")
        return format_response(
            f".PARAM {param_name} = {value}",
            {"parameters": {param_name: value}},
            fmt,
        )

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
    description=(
        "Add or remove a SPICE directive or .asc free-text comment. Set "
        "``kind=comment`` for annotation text; default is a SPICE directive. "
        "Works on .cir/.net and .asc; ``kind=comment`` is .asc-only. "
        "``remove`` matches against directives AND comments, so callers can "
        "delete either kind without knowing which it is."
    ),
    input_model=EditDirectiveInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_edit_directive(
    args: EditDirectiveInput, state: SessionState
) -> types.CallToolResult:
    """Add or remove a SPICE directive (or .asc comment). Works on .cir/.net and .asc."""
    file_path = safe_path(args.path, state)

    action = args.action
    instruction = args.instruction
    kind = args.kind

    if not instruction.strip():
        raise NetlistError("Directive instruction must not be empty")

    async with _editing(file_path, state) as editor:
        if action == "add":
            if kind == "comment":
                if not _is_asc(file_path):
                    raise NetlistError(
                        "kind='comment' is .asc-only — for .cir/.net files "
                        "add a literal ``*`` or ``;`` comment in the file directly."
                    )
                # ``_is_asc`` above guarantees AscEditor; cast for the type checker.
                asc_editor = cast(AscEditor, editor)
                comment = Text(
                    coord=Point(
                        args.x if args.x is not None else 0, args.y if args.y is not None else 0
                    ),
                    text=instruction,
                    type=TextTypeEnum.COMMENT,
                    size=args.size,
                )
                asc_editor.directives.append(comment)
                result = f"Added comment: {instruction}"
            else:
                stripped = instruction.strip()
                # Friction D guard: a leading ``!`` / ``.`` was the giveaway
                # that the old ``add_text`` user actually wanted a directive.
                if not stripped.startswith("."):
                    raise NetlistError(
                        "SPICE directives must start with '.' (e.g. .tran, "
                        ".ac, .param). For free-text annotations on .asc "
                        "schematics, set kind='comment'."
                    )
                # Pre-flight validation: catch known-bad patterns (e.g. vdb()
                # in .MEAS) before they reach the simulator and fail post-hoc
                # inside the .log.
                err = validate_directive(instruction, simulator="LTspice")
                if err is not None:
                    raise NetlistError(f"{err.message}\n  Suggestion: {err.suggestion}")
                editor.add_instruction(instruction)
                result = f"Added directive: {instruction}"

        elif action == "remove":
            removed = _remove_directive_or_comment(editor, instruction)
            result = f"Removed {removed.label}: {instruction}"

        else:
            raise NetlistError(f"Invalid action '{action}'. Must be 'add' or 'remove'.")

    return text_response(result)


@dataclass(frozen=True)
class _RemoveResult:
    """Tag describing what kind of entry was removed."""

    label: str


def _remove_directive_or_comment(editor, instruction: str) -> "_RemoveResult":
    """Remove a directive or comment matching ``instruction`` (regex- or literal).

    spicelib distinguishes "directives" (``.foo``) from "comments" (free
    text in TEXT entries). The user-facing ``edit_directive remove`` should
    not require them to know which kind they're targeting; this helper
    hits both when applicable.
    """
    if instruction.startswith("regex:"):
        pattern = instruction[6:]
        if not pattern.strip():
            raise NetlistError(
                "Empty regex pattern would match every directive; "
                "provide an explicit regex after 'regex:'."
            )
        editor.remove_Xinstruction(pattern)
        # Best-effort: also remove TEXT-COMMENT entries whose body matches.
        _strip_matching_comments(editor, re.compile(pattern))
        return _RemoveResult(label="directive(s)/comment(s)")
    if any(char in instruction for char in r"\[]().*+?^${}|"):
        editor.remove_Xinstruction(instruction)
        _strip_matching_comments(editor, re.compile(instruction))
        return _RemoveResult(label="directive(s)/comment(s)")
    editor.remove_instruction(instruction)
    _strip_matching_comments(editor, instruction)
    return _RemoveResult(label="directive")


def _asc_directive_lines(editor: AscEditor) -> list[str]:
    """Return the SPICE-directive text bodies from an .asc editor.

    Free-text COMMENT TEXT entries are filtered out — only DIRECTIVE-type
    entries flow through. Used by both ``edit_directive`` and
    ``validate_netlist`` so the ``.asc`` directive boundary is defined in
    exactly one place.
    """
    return [
        d.text
        for d in editor.directives
        if getattr(d, "type", None) == TextTypeEnum.DIRECTIVE and isinstance(d.text, str)
    ]


def _strip_matching_comments(editor, matcher) -> None:
    """Best-effort removal of TEXT-COMMENT entries whose body matches.

    ``matcher`` is either a literal string (exact match) or a compiled
    regex. ``editor.directives`` only exists on AscEditor — silently
    no-op for netlist-mode editors.
    """
    directives = getattr(editor, "directives", None)
    if directives is None:
        return
    keep = []
    for entry in directives:
        body = getattr(entry, "text", None)
        entry_kind = getattr(entry, "type", None)
        if entry_kind == TextTypeEnum.COMMENT and isinstance(body, str):
            if isinstance(matcher, str):
                if body.strip() == matcher.strip():
                    continue
            else:
                if matcher.search(body):
                    continue
        keep.append(entry)
    if len(keep) != len(directives):
        directives[:] = keep


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
async def handle_remove_component(
    args: RemoveComponentInput, state: SessionState
) -> types.CallToolResult:
    """Remove a component from a schematic by reference designator."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    reference = args.reference

    # Collect pin positions before removal to check for orphaned wires
    editor_pre = _get_asc_editor(asc_path, state)
    _require_component(editor_pre, reference)
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
async def handle_move_component(
    args: MoveComponentInput, state: SessionState
) -> types.CallToolResult:
    """Move or rotate a component in a schematic."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    reference = args.reference
    x = args.x
    y = args.y
    rotation = args.rotation

    async with _editing_asc(asc_path, state) as editor:
        _require_component(editor, reference)
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
    args: SetComponentAttributeInput, state: SessionState
) -> types.CallToolResult:
    """Set an attribute on a schematic component (e.g., SpiceLine, SpiceModel)."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    reference = args.reference
    attribute = args.attribute
    value = args.value

    if not attribute.strip():
        raise NetlistError("Attribute name must not be empty")

    async with _editing_asc(asc_path, state) as editor:
        _require_component(editor, reference)
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
    output_schema={
        "type": "object",
        "properties": {
            "reference": {"type": "string"},
            "symbol": {"type": "string"},
            "position": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "rotation": {"type": "string"},
            "pins": {"type": "array", "items": PIN_SCHEMA},
            "bounding_box": BBOX_SCHEMA,
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    },
)
async def handle_add_component(
    args: AddComponentInput, state: SessionState
) -> types.CallToolResult:
    """Add a new component to an .asc schematic."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    reference = args.reference
    symbol = args.symbol
    x = args.x
    y = args.y
    value = args.value
    rotation = args.rotation
    erot = _parse_rotation(rotation)

    # Validate the symbol exists BEFORE touching the file. Saving a .asc with
    # a dangling symbol name corrupts it — spicelib's AscEditor refuses to
    # re-open such a file because it can't find the .asy on reset_netlist().
    if get_symbol_info(symbol) is None:
        raise NetlistError(
            f"Symbol '{symbol}' not found in any configured symbol library. "
            "Use ltspice_symbol_info to verify the symbol name, or "
            "configure [schematic] symbol_paths in ltspice-mcp.toml."
        )

    async with _editing_asc(asc_path, state) as editor:
        if reference in editor.components:
            raise NetlistError(
                f"Component '{reference}' already exists. "
                "Use ltspice_set_component_value to modify it, "
                "or ltspice_remove_component to remove it first."
            )

        _create_component(
            editor,
            reference,
            symbol,
            x,
            y,
            erot,
            value=value,
            attributes=args.attributes,
        )

    result = f"Added {reference} ({symbol}) at ({x},{y})"
    if value is not None:
        result += f" = {value}"

    sym_info = get_symbol_info(symbol)
    if sym_info is None:
        fallback_data = {
            "reference": reference,
            "symbol": symbol,
            "position": {"x": x, "y": y},
            "rotation": rotation,
        }
        return format_response(result, fallback_data, None)

    geometry = compute_placed_geometry(sym_info, x, y, rotation)
    for pin in geometry["pins"]:
        result += f"\n  {pin['name']}: ({pin['x']}, {pin['y']}) [{pin['dir']}]"
    bb = geometry["bounding_box"]
    result += f"\n  bbox: ({bb['x']},{bb['y']}) {bb['width']}x{bb['height']}"

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
async def handle_export_netlist(
    args: ExportNetlistInput, state: SessionState
) -> types.CallToolResult:
    """Export an .asc schematic to a SPICE netlist (.net) using LTspice."""
    asc_path = safe_path(args.path, state)
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
    name="ltspice_symbol_info",
    description=(
        "Get symbol pin positions, bounding box, and description. "
        "Optionally compute absolute positions for a given placement and rotation."
    ),
    input_model=SymbolInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "bbox_width": {"type": "integer"},
            "bbox_height": {"type": "integer"},
            "pins": {"type": "array", "items": PIN_SCHEMA},
            "placement": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "rotation": {"type": "string"},
                },
            },
            "absolute_pins": {"type": "array", "items": PIN_SCHEMA},
            "absolute_bounding_box": BBOX_SCHEMA,
        },
    },
)
async def handle_symbol_info(args: SymbolInfoInput, state: SessionState) -> types.CallToolResult:
    """Get symbol geometry info for schematic layout planning."""
    symbol = args.symbol
    sym_info = get_symbol_info(symbol)
    if sym_info is None:
        raise NetlistError(
            f"Symbol '{symbol}' not found. Ensure LTspice symbol libraries are configured."
        )

    geometry = compute_placed_geometry(sym_info, args.x, args.y, args.rotation)
    data = {
        **sym_info.to_dict(),
        "placement": {"x": args.x, "y": args.y, "rotation": args.rotation},
        "absolute_pins": geometry["pins"],
        "absolute_bounding_box": geometry["bounding_box"],
    }

    lines = [f"Symbol: {sym_info.name}"]
    if sym_info.description:
        lines.append(f"Description: {sym_info.description}")
    lines.append(f"Size: {sym_info.bbox_width}x{sym_info.bbox_height}")
    lines.append(f"Pins (at {args.rotation}, origin ({args.x},{args.y})):")
    for pin in geometry["pins"]:
        lines.append(f"  {pin['name']}: ({pin['x']}, {pin['y']})")
    bb = geometry["bounding_box"]
    lines.append(f"Bounding box: ({bb['x']},{bb['y']}) {bb['width']}x{bb['height']}")

    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_component_info",
    description=(
        "Get a placed component's pin positions, bounding box, value, and attributes "
        "from an .asc schematic."
    ),
    input_model=ComponentInfoInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "reference": {"type": "string"},
            "symbol": {"type": "string"},
            "position": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "rotation": {"type": "string"},
            "value": {"type": ["string", "null"]},
            "pins": {"type": "array", "items": PIN_SCHEMA},
            "bounding_box": BBOX_SCHEMA,
            "attributes": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
        },
    },
)
async def handle_component_info(
    args: ComponentInfoInput, state: SessionState
) -> types.CallToolResult:
    """Get full info about a placed component including computed pin positions."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)
    reference = args.reference

    editor = _get_asc_editor(asc_path, state)
    _require_component(editor, reference)

    pos, erot = editor.get_component_position(reference)
    rot_str = erot.name if erot else "R0"
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

    return format_response("\n".join(lines), data, args.format)


def _resolve_pin(pin_ref: str, editor: AscEditor) -> tuple[int, int]:
    """Resolve a pin reference ('M1.D' or 'net:VDD') to absolute (x, y) coordinates.

    Raises NetlistError if the reference cannot be resolved.
    """
    if pin_ref.startswith("net:"):
        # Look up a FLAG/net label position in the .asc
        net_name = pin_ref[4:]
        matches = [
            (int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels if lbl.text == net_name
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
                "Use a unique net label, connect directly to a component pin, "
                "or place the label at a pin with add_net_label(net='0', pin='M3.S')."
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
async def handle_add_net_label(args: NetLabelInput, state: SessionState) -> types.CallToolResult:
    """Add or remove a FLAG (net label or ground) in a schematic."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    net = args.net
    label_desc = "ground" if net == "0" else f"net '{net}'"

    # Resolve coordinates from pin reference or explicit x/y
    if args.pin is not None:
        editor = _get_asc_editor(asc_path, state)
        x, y = _resolve_pin(args.pin, editor)
    elif args.x is not None and args.y is not None:
        x, y = args.x, args.y
    else:
        raise NetlistError("Either pin or both x and y coordinates are required.")

    if args.action == "remove":
        async with _editing_asc(asc_path, state) as editor:
            for i, lbl in enumerate(editor.labels):
                if lbl.text == net and int(lbl.coord.X) == x and int(lbl.coord.Y) == y:
                    editor.labels.pop(i)
                    return text_response(f"Removed {label_desc} at ({x},{y})")
            raise NetlistError(f"No {label_desc} found at ({x},{y})")

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

        editor.labels.append(Text(coord=Point(x, y), text=net, type=TextTypeEnum.LABEL))

    result += f"Added {label_desc} at ({x},{y})"
    return text_response(result)


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
    output_schema={
        "type": "object",
        "properties": {
            "from": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
            "to": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
            },
            "wire_count": {"type": "integer"},
            "points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    },
)
async def handle_connect(args: ConnectInput, state: SessionState) -> types.CallToolResult:
    """Connect two pins with auto-routed or waypoint-guided wires."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    # Collect component geometry and existing wires for validation
    pre_editor = _get_asc_editor(asc_path, state)
    component_geo = _collect_component_geometry(pre_editor)
    existing_wires = [
        (int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in pre_editor.wires
    ]

    # Resolve pins (read-only — no edits yet)
    x1, y1 = _resolve_pin(args.from_pin, pre_editor)
    x2, y2 = _resolve_pin(args.to_pin, pre_editor)

    # Reject zero-length connections — they would emit no wires and silently
    # report success, which is almost always a user error.
    if (x1, y1) == (x2, y2) and not args.waypoints:
        raise NetlistError(
            f"Cannot connect {args.from_pin} to {args.to_pin}: "
            f"both endpoints resolve to the same coordinate ({x1},{y1})."
        )

    # Build list of points: from → [waypoints] → to (dedup consecutive)
    raw_points = [(x1, y1)]
    for wp in args.waypoints:
        raw_points.append((wp.x, wp.y))
    raw_points.append((x2, y2))
    points = [raw_points[0]]
    for pt in raw_points[1:]:
        if pt != points[-1]:
            points.append(pt)

    # Compute segments (not yet added to editor)
    segments: list[tuple[int, int, int, int]] = []
    for i in range(len(points) - 1):
        px1, py1 = points[i]
        px2, py2 = points[i + 1]
        if px1 != px2 or py1 != py2:
            segments.append((px1, py1, px2, py2))

    # If after dedup we ended up with no segments at all (e.g. waypoints all
    # collapse onto one of the endpoints), reject as a no-op.
    if not segments:
        raise NetlistError(
            f"Cannot connect {args.from_pin} to {args.to_pin}: "
            "the requested route has zero length after deduplicating waypoints."
        )

    # --- Validate before adding wires ---
    errors: list[str] = []
    warnings: list[str] = []
    endpoints = {(x1, y1), (x2, y2)}
    skip_refs = {
        ref.rsplit(".", 1)[0]
        for ref in (args.from_pin, args.to_pin)
        if "." in ref and not ref.startswith("net:")
    }

    # Diagonal wires (error — never valid)
    for sx1, sy1, sx2, sy2 in segments:
        if sx1 != sx2 and sy1 != sy2:
            errors.append(f"Diagonal wire ({sx1},{sy1})->({sx2},{sy2}): not orthogonal")

    # Pin collision check (error — will create unintended connection)
    # A pin is safe if it's already wired to the same net as our target
    # (i.e., an existing wire connects both the pin and one of our endpoints).
    def _pin_on_target_net(px: int, py: int) -> bool:
        for ex1, ey1, ex2, ey2 in existing_wires:
            wire_pts = {(ex1, ey1), (ex2, ey2)}
            if (px, py) in wire_pts and wire_pts & endpoints:
                return True
        return False

    for cg in component_geo:
        if cg["ref"] in skip_refs:
            continue
        for pin in cg["pins"]:
            px, py = pin["x"], pin["y"]
            if (px, py) in endpoints:
                continue
            if _pin_on_target_net(px, py):
                continue
            for sx1, sy1, sx2, sy2 in segments:
                on_wire = False
                if sy1 == sy2 and py == sy1:
                    on_wire = min(sx1, sx2) <= px <= max(sx1, sx2)
                elif sx1 == sx2 and px == sx1:
                    on_wire = min(sy1, sy2) <= py <= max(sy1, sy2)
                if on_wire:
                    errors.append(
                        f"Wire passes through {cg['ref']}.{pin['name']} at ({px},{py}): "
                        "will create unintended connection"
                    )

    # Wire junction check (error — will create unintended junction)
    # Allow overlaps where the existing wire connects to one of our endpoints
    # (e.g., T-junction onto a power rail that reaches the target net label)
    for sx1, sy1, sx2, sy2 in segments:
        for ex1, ey1, ex2, ey2 in existing_wires:
            ext_endpoints = {(ex1, ey1), (ex2, ey2)}
            if ext_endpoints & endpoints:
                continue  # existing wire shares an endpoint — intentional junction
            if sx1 == sx2 and ex1 == ex2 and sx1 == ex1:
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_y = max(new_min, ext_min)
                    if (sx1, overlap_y) not in endpoints:
                        errors.append(
                            f"Wire overlap at x={sx1} between y={max(new_min, ext_min)} "
                            f"and y={min(new_max, ext_max)}: will create unintended junction"
                        )
                        break
            elif sy1 == sy2 and ey1 == ey2 and sy1 == ey1:
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_x = max(new_min, ext_min)
                    if (overlap_x, sy1) not in endpoints:
                        errors.append(
                            f"Wire overlap at y={sy1} between x={max(new_min, ext_min)} "
                            f"and x={min(new_max, ext_max)}: will create unintended junction"
                        )
                        break
            elif sx1 == sx2 and ey1 == ey2:
                cross_x, cross_y = sx1, ey1
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if (
                    new_min < cross_y < new_max
                    and ext_min < cross_x < ext_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    errors.append(
                        f"Wire crosses existing wire at ({cross_x},{cross_y}): "
                        "will create unintended junction"
                    )
            elif sy1 == sy2 and ex1 == ex2:
                cross_x, cross_y = ex1, sy1
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if (
                    ext_min < cross_y < ext_max
                    and new_min < cross_x < new_max
                    and (cross_x, cross_y) not in endpoints
                ):
                    errors.append(
                        f"Wire crosses existing wire at ({cross_x},{cross_y}): "
                        "will create unintended junction"
                    )

    # Refuse to add wires if any errors detected
    if errors:
        error_lines = [f"Refused to connect {args.from_pin} to {args.to_pin}:"]
        for e in errors:
            error_lines.append(f"  {e}")
        error_lines.append("\nFix the route with different waypoints to avoid these issues.")
        raise NetlistError("\n".join(error_lines))

    # Non-blocking warnings
    total_length = sum(abs(sx2 - sx1) + abs(sy2 - sy1) for sx1, sy1, sx2, sy2 in segments)
    if total_length > 400:
        warnings.append(
            f"Long wire run ({total_length} units): consider placing components closer "
            "or adding a local net label"
        )

    for sx1, sy1, sx2, sy2 in segments:
        for bb in component_geo:
            if bb["ref"] in skip_refs:
                continue
            bx, by, bw, bh = bb["x"], bb["y"], bb["width"], bb["height"]
            if sy1 == sy2:
                wy = sy1
                wx_min, wx_max = min(sx1, sx2), max(sx1, sx2)
                if by < wy < by + bh and wx_min < bx + bw and wx_max > bx:
                    warnings.append(
                        f"Wire at y={wy} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )
            elif sx1 == sx2:
                wx = sx1
                wy_min, wy_max = min(sy1, sy2), max(sy1, sy2)
                if bx < wx < bx + bw and wy_min < by + bh and wy_max > by:
                    warnings.append(
                        f"Wire at x={wx} crosses {bb['ref']} bounding box "
                        f"({bx},{by})-({bx + bw},{by + bh})"
                    )

    # All checks passed — now add the wires
    async with _editing_asc(asc_path, state) as ed:
        for sx1, sy1, sx2, sy2 in segments:
            ed.wires.append(Line(Point(sx1, sy1), Point(sx2, sy2)))

    result_lines = [f"Connected {args.from_pin} to {args.to_pin}"]
    result_lines.append(f"  From: ({x1},{y1})  To: ({x2},{y2})")
    for sx1, sy1, sx2, sy2 in segments:
        result_lines.append(f"  Wire: ({sx1},{sy1})->({sx2},{sy2})")

    if warnings:
        result_lines.append("")
        result_lines.append("Warnings:")
        for w in warnings:
            result_lines.append(f"  {w}")

    data: dict = {
        "from": {"ref": args.from_pin, "x": x1, "y": y1},
        "to": {"ref": args.to_pin, "x": x2, "y": y2},
        "wire_count": len(segments),
        "points": [{"x": p[0], "y": p[1]} for p in points],
    }
    if warnings:
        data["warnings"] = warnings

    return format_response("\n".join(result_lines), data, None)


# ---------------------------------------------------------------------------
# New tools: schematic seeding, netlist validation, .step querying, diff
# ---------------------------------------------------------------------------


class CreateSchematicInput(ToolInput):
    name: str = Field(description="File name without the .asc extension")
    width: int = Field(
        default=880,
        description="Sheet width (LTspice grid units). 880 matches LTspice's default.",
    )
    height: int = Field(
        default=680,
        description="Sheet height (LTspice grid units). 680 matches LTspice's default.",
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite an existing file at this path. Default is to refuse.",
    )


@registry.tool(
    name="ltspice_create_schematic",
    description=(
        "Create an empty .asc schematic ready for incremental editing via "
        "ltspice_add_component / ltspice_connect / ltspice_add_net_label. "
        "Tip: prefer ``ltspice_create_netlist`` + .cir for design iteration; "
        "use this only when a visual schematic is the deliverable."
    ),
    input_model=CreateSchematicInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full",),
)
async def handle_create_schematic(
    args: CreateSchematicInput, state: SessionState
) -> types.CallToolResult:
    """Create an empty .asc schematic file."""
    target_path = safe_path(f"{args.name}.asc", state)
    if args.width <= 0 or args.height <= 0:
        raise NetlistError(
            f"Sheet dimensions must be positive; got width={args.width}, height={args.height}"
        )
    body = f"Version 4\nSHEET 1 {args.width} {args.height}\n"
    try:
        atomic_write_text(target_path, body, overwrite=args.overwrite, durable=False)
    except FileExistsError as e:
        raise NetlistError(
            f"File already exists: {target_path}. Pass overwrite=true to replace it."
        ) from e
    return text_response(
        f"Created schematic: {target_path}\n  Sheet: {args.width} x {args.height}"
    )


class ValidateNetlistInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_validate_netlist",
    description=(
        "Run static checks over a netlist or schematic before simulation: "
        "rejects known-bad .MEAS patterns (vdb()/phase()/group_delay()), "
        "flags spicelib-unparseable B-source lines, and surfaces directives "
        "that the LTspice runner is known to reject. Returns a structured "
        "list of issues; an empty list means the file passes the static gate."
    ),
    input_model=ValidateNetlistInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "issue_count": {"type": "integer"},
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "severity": {"type": "string", "enum": ["error", "warning"]},
                        "line": {"type": ["integer", "null"]},
                        "directive": {"type": "string"},
                        "message": {"type": "string"},
                        "suggestion": {"type": ["string", "null"]},
                    },
                },
            },
        },
    },
)
async def handle_validate_netlist(
    args: ValidateNetlistInput, state: SessionState
) -> types.CallToolResult:
    """Static validation pass over a netlist / schematic."""
    file_path = safe_path(args.path, state)
    fmt = args.format

    if _is_asc(file_path):
        try:
            content = "\n".join(_asc_directive_lines(_get_asc_editor(file_path, state)))
        except Exception as e:
            raise NetlistError(f"Failed to open .asc: {e}") from e
    else:
        content = file_path.read_text(encoding="utf-8", errors="replace")

    issues: list[dict] = []
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line.startswith("."):
            continue
        err = validate_directive(line, simulator="LTspice")
        if err is not None:
            issues.append(
                {
                    "severity": "error",
                    "line": lineno,
                    "directive": line,
                    "message": err.message,
                    "suggestion": err.suggestion,
                }
            )

    # Sniff for B-sources whose value field contains commas inside if(...)
    # — those defeat spicelib's component-line regex (Bug K).
    if not _is_asc(file_path):
        for lineno, raw_line in enumerate(content.splitlines(), 1):
            line = raw_line.lstrip()
            if line[:1].upper() != "B":
                continue
            if "if(" in line.lower() and "," in line:
                issues.append(
                    {
                        "severity": "warning",
                        "line": lineno,
                        "directive": line.rstrip(),
                        "message": (
                            "Behavioural source uses an ``if(...)`` expression "
                            "with commas — spicelib's component-line regex "
                            "rejects this shape, so ``read_circuit`` and "
                            "``list_components`` will report ``<unparseable>`` "
                            "for this ref. The LTspice simulator parses it fine."
                        ),
                        "suggestion": (
                            "If you need spicelib to introspect the value, "
                            "rewrite as ``limit(...)`` or split into multiple "
                            "B-sources without commas."
                        ),
                    }
                )

    summary = {"file": str(file_path), "issue_count": len(issues), "issues": issues}
    if not issues:
        return format_response(f"OK: no issues in {file_path.name}", summary, fmt)
    lines = [f"{file_path.name}: {len(issues)} issue(s)"]
    for issue in issues:
        loc = f":{issue['line']}" if issue.get("line") else ""
        lines.append(f"  [{issue['severity']}] line{loc}: {issue['message']}")
        if issue.get("directive"):
            lines.append(f"    {issue['directive']}")
        if issue.get("suggestion"):
            lines.append(f"    Suggestion: {issue['suggestion']}")
    return format_response("\n".join(lines), summary, fmt)


class DiffCircuitInput(ToolInput):
    path_a: str = Field(description="Path to the first circuit file (.cir, .net, or .asc)")
    path_b: str = Field(description="Path to the second circuit file (.cir, .net, or .asc)")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


def _components_and_directives(path: Path) -> tuple[dict[str, str], set[str]]:
    """Return (components, directive_lines) for a circuit file in one read.

    Reuses ``services.extract_{asc,netlist}_info`` so unparseable B-sources,
    AscEditor dispatch, and directive collection all flow through the
    canonical path. No second disk read.
    """
    try:
        ed = _make_editor(path)
    except Exception:
        return {}, set()
    if _is_asc(path):
        assert isinstance(ed, AscEditor)  # _make_editor dispatches on extension
        info = services.extract_asc_info(ed, path)
        components = {comp["reference"]: str(comp["value"]) for comp in info["components"]}
        directives = {d.strip() for d in info.get("directives", []) if d.strip().startswith(".")}
        return components, directives
    info = services.extract_netlist_info(ed, path)
    components = {comp["reference"]: str(comp["value"]) for comp in info["components"]}
    directives = {
        line.strip()
        for line in info.get("content", "").splitlines()
        if line.strip().startswith(".")
    }
    return components, directives


@registry.tool(
    name="ltspice_diff_circuit",
    description=(
        "Structural diff between two circuit files: reports added/removed "
        "components, components whose value changed, and added/removed "
        ".PARAM/.MEAS/.MODEL directives. Use after ``set_component_value`` "
        "or ``edit_directive`` to confirm that the intended change "
        "actually landed."
    ),
    input_model=DiffCircuitInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
)
async def handle_diff_circuit(args: DiffCircuitInput, state: SessionState) -> types.CallToolResult:
    """Structural diff between two circuit files."""
    path_a = safe_path(args.path_a, state)
    path_b = safe_path(args.path_b, state)

    a, da = _components_and_directives(path_a)
    b, db = _components_and_directives(path_b)

    added = sorted(set(b) - set(a))
    removed = sorted(set(a) - set(b))
    changed: list[dict[str, str]] = []
    for ref in sorted(set(a) & set(b)):
        if a[ref] != b[ref]:
            changed.append({"reference": ref, "before": a[ref], "after": b[ref]})

    directive_added = sorted(db - da)
    directive_removed = sorted(da - db)

    data = {
        "path_a": str(path_a),
        "path_b": str(path_b),
        "components_added": added,
        "components_removed": removed,
        "components_changed": changed,
        "directives_added": directive_added,
        "directives_removed": directive_removed,
    }

    lines = [f"Diff: {path_a.name} -> {path_b.name}"]
    if added:
        lines.append("Components added:")
        for r in added:
            lines.append(f"  + {r}: {b[r]}")
    if removed:
        lines.append("Components removed:")
        for r in removed:
            lines.append(f"  - {r}: {a[r]}")
    if changed:
        lines.append("Components changed:")
        for c in changed:
            lines.append(f"  ~ {c['reference']}: {c['before']} -> {c['after']}")
    if directive_added:
        lines.append("Directives added:")
        for d in directive_added:
            lines.append(f"  + {d}")
    if directive_removed:
        lines.append("Directives removed:")
        for d in directive_removed:
            lines.append(f"  - {d}")
    if not (added or removed or changed or directive_added or directive_removed):
        lines.append("(no structural differences)")

    return format_response("\n".join(lines), data, args.format)


class StepGetInput(ToolInput):
    raw_file: str = Field(description="Path to a stepped .raw result")
    axis: str = Field(
        description=(
            "Step parameter name to query (e.g. ``temp``, ``RS``). For .DC "
            "sweeps the axis is the swept variable; for .step parametric "
            "runs it's the parameter that was stepped."
        ),
    )
    value: str = Field(
        description="SPICE-notation target value (e.g. ``27``, ``1k``, ``100u``).",
    )
    signal: str = Field(description="Signal to read at the chosen step (e.g. ``V(out)``).")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_step_get",
    description=(
        "Look up a signal at a chosen value of a .step / .DC sweep axis "
        "(e.g. ``axis='temp', value='27'``). Avoids the manual run_index → "
        "params lookup users had to do via ltspice_batch_results."
    ),
    input_model=StepGetInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
)
async def handle_step_get(args: StepGetInput, state: SessionState) -> types.CallToolResult:
    """Query a signal at a specific axis value of a stepped .raw result."""
    raw_path = safe_path(args.raw_file, state)
    raw = services.load_raw(raw_path, state)

    try:
        target = parse_spice_value(args.value)
    except ValueError as e:
        raise NetlistError(f"Invalid value {args.value!r}: {e}") from e

    signal = services.validate_signal(raw, args.signal)

    # Strategy: if ``axis`` matches the .raw's axis name (case-insensitive),
    # use the axis values directly. Otherwise fall back to .step parameter
    # lookup via spicelib's ``get_steps``.
    raw_axis_name = ""
    try:
        plot = raw.get_raw_property("Plotname")
        if plot:
            # Plotname doesn't carry the axis name; pull from trace 0.
            raw_axis_name = raw.get_trace_names()[0]
    except Exception:
        pass

    axis_lower = args.axis.lower()
    if raw_axis_name and axis_lower == raw_axis_name.lower():
        try:
            axis_vals = list(raw.get_axis(step=0))
        except Exception as e:
            raise NetlistError(
                f"Cannot read axis values: {e}. Use ltspice_query_value if "
                "the raw doesn't have an explicit axis."
            ) from e
        # nearest neighbour
        ins = bisect.bisect_left(axis_vals, target)
        if ins == 0:
            idx = 0
        elif ins == len(axis_vals):
            idx = len(axis_vals) - 1
        else:
            idx = (
                ins - 1
                if abs(axis_vals[ins - 1] - target) <= abs(axis_vals[ins] - target)
                else ins
            )
        wave = raw.get_wave(signal, step=0)
        actual = float(axis_vals[idx])
        value = float(wave[idx])
        data = {
            "signal": signal,
            "axis": args.axis,
            "requested_value": target,
            "actual_value": actual,
            "value": value,
        }
        return format_response(f"{signal} at {args.axis}={actual:g}: {value:g}", data, args.format)

    # Fallback: spicelib step lookup.
    try:
        steps = raw.get_steps()
    except Exception as e:
        raise NetlistError(f"Raw file has no .step iterations: {e}") from e

    best_idx = None
    best_actual: float | None = None
    for i, step_record in enumerate(steps):
        # spicelib returns dicts like {"temp": -40, "RS": 1000}.
        if isinstance(step_record, dict):
            v = step_record.get(args.axis)
            if v is None:
                # try case-insensitive match
                for k, val in step_record.items():
                    if k.lower() == axis_lower:
                        v = val
                        break
            if v is None:
                continue
            try:
                v_f = float(v)
            except (TypeError, ValueError):
                continue
            if best_actual is None or abs(v_f - target) < abs(best_actual - target):
                best_actual = v_f
                best_idx = i

    if best_idx is None:
        raise NetlistError(
            f"Step axis {args.axis!r} not found in this raw file. "
            "Available axes: "
            + (", ".join(steps[0].keys()) if steps and isinstance(steps[0], dict) else "<none>")
        )

    wave = raw.get_wave(signal, step=best_idx)
    point = float(wave[0]) if len(wave) else float("nan")
    data = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": best_actual,
        "step_index": best_idx,
        "value": point,
    }
    return format_response(
        f"{signal} at {args.axis}={best_actual:g} (step {best_idx}): {point:g}",
        data,
        args.format,
    )
