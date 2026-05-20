"""Unified circuit editing tools for .cir/.net netlists and .asc schematics.

Extension-based dispatch: the file extension determines which spicelib editor
is used (SpiceEditor for .cir/.net, AscEditor for .asc).  Schematic-only
operations (position, rotation, attributes, export) validate the extension
and raise NetlistError if given a non-.asc file.
"""

import asyncio
import bisect
import contextlib
import io
import re
from collections.abc import AsyncIterator
from collections.abc import Set as AbstractSet
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, NamedTuple, cast

import numpy as np
from mcp import types
from pydantic import Field
from spicelib import AscEditor, SpiceEditor
from spicelib.editor.asc_editor import LTSPICE_ATTRIBUTES, LTSPICE_PARAMETERS
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
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.raw_parser import nearest_index, real_axis, sample_to_dict
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_validator import (
    ANALYSIS_KINDS,
    MEAS_KINDS,
    validate_directive,
    validate_netlist_arity,
)
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, get_symbol_info
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    BBOX_SCHEMA,
    PAGINATION_SCHEMA,
    PIN_SCHEMA,
    RO_ANNOTATIONS,
    VALIDATION_WARNINGS_SCHEMA,
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
            # Spicelib's parser fails on subsequent reads of a SYMATTR line
            # with no value, leaving the .asc permanently unreadable until
            # manually edited. Reject up front (P-N1).
            if attr_val == "":
                raise NetlistError(
                    f"Attribute {attr_name!r} on {reference!r}: empty value "
                    "would corrupt the schematic. Omit the key to leave the "
                    "attribute unset."
                )
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


# Standard LTspice SYMATTR slot names. Anything outside this set is
# silently ignored at netlist-export time, so we reject up-front rather
# than letting a typo no-op silently. Sourced from spicelib so a future
# release that adds a slot is picked up without a code change here.
_LTSPICE_ATTR_NAMES: frozenset[str] = frozenset(LTSPICE_PARAMETERS + LTSPICE_ATTRIBUTES)
_LTSPICE_ATTR_CANONICAL: dict[str, str] = {n.lower(): n for n in _LTSPICE_ATTR_NAMES}


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
    # Independent-source waveform spec: ``PULSE(...)``, ``SIN(...)``,
    # ``EXP(...)``, ``PWL(...)``, ``SFFM(...)``, ``TABLE(...)``, ``AM(...)``,
    # ``NOISE(...)``. The keyword is followed by a balanced parenthetical
    # group whose parens protect the embedded whitespace. Optionally
    # preceded by a DC magnitude (``"1 PULSE(...)"``) and followed by an
    # ``AC <mag>`` annotation (``"PULSE(...) AC 1"``).
    try:
        toks = tokenize_body(stripped)
    except SpiceLexError:
        toks = []
    if toks and any(t.kind == TokenKind.PARENED for t in toks):
        # If the body is a sequence of BARE/PARENED tokens (no stray
        # equals signs, no unbalanced quotes), the parens protect their
        # internal whitespace from corrupting the netlist line.
        ok_kinds = (TokenKind.BARE, TokenKind.PARENED, TokenKind.QUOTED, TokenKind.BRACED)
        if all(t.kind in ok_kinds for t in toks):
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
    tokens and route them through ``set_component_parameters``, keeping
    the model/value field for ``set_component_value``.

    Token-based split via ``spice_lex.tokenize_body``: head is every
    ``BARE`` / ``QUOTED`` / ``BRACED`` token before any ``KEY_VALUE``
    token; params are the ``KEY_VALUE`` tokens. The classified-token
    layer knows model-name vs param-name by construction, so adversarial
    cases like ``M1 d g s b "NMOS lvt" W=10u`` and
    ``R1 n1 n2 {1/(2*pi*RC)}`` route correctly.
    """
    _validate_component_value(reference, value)
    if "=" not in value:
        editor.set_component_value(reference, value)
        return
    try:
        tokens = tokenize_body(value)
    except SpiceLexError as e:
        raise NetlistError(f"Component '{reference}' value {value!r} failed to parse: {e}") from e
    params: dict[str, str] = {}
    head_parts: list[str] = []
    for tok in tokens:
        if tok.kind == TokenKind.KEY_VALUE:
            assert tok.key is not None
            assert tok.value is not None
            params[tok.key] = tok.value
        elif tok.kind in (TokenKind.BARE, TokenKind.QUOTED, TokenKind.BRACED):
            head_parts.append(tok.text)
        # COMMENT_TRAIL / EQUALS / PARENED outside KEY_VALUE: ignore
        # for value-setting purposes — _validate_component_value
        # already rejected the shapes that would corrupt the netlist.
    head = " ".join(head_parts)
    if head:
        editor.set_component_value(reference, head)
    if params:
        editor.set_component_parameters(reference, **params)


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """AABB overlap test between two bounding boxes with {x, y, width, height}."""
    return BBox.from_origin_size(a["x"], a["y"], a["width"], a["height"]).overlaps(
        BBox.from_origin_size(b["x"], b["y"], b["width"], b["height"])
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


def _component_pin_coords(editor: AscEditor, reference: str) -> set[tuple[int, int]]:
    """Pin coordinates for a single component, ``set()`` if symbol unknown."""
    if reference not in editor.components:
        return set()
    comp = editor.components[reference]
    if not comp.symbol:
        return set()
    sym_info = get_symbol_info(comp.symbol)
    if sym_info is None:
        return set()
    pos, erot = editor.get_component_position(reference)
    rot_str = erot.name if erot else "R0"
    geo = compute_placed_geometry(sym_info, int(pos.X), int(pos.Y), rot_str)
    return {(p["x"], p["y"]) for p in geo["pins"]}


def _other_components_pin_coords(editor: AscEditor, exclude_ref: str) -> set[tuple[int, int]]:
    """Union of pin coordinates for every component except ``exclude_ref``.

    Used by remove/move handlers to filter orphaned-wire warnings: a wire
    endpoint that coincides with another component's pin isn't actually
    orphaned, it's that component's wire.
    """
    coords: set[tuple[int, int]] = set()
    for ref in editor.get_components():
        if ref == exclude_ref:
            continue
        coords.update(_component_pin_coords(editor, ref))
    return coords


def _point_on_segment(point: tuple[int, int], v1: tuple[int, int], v2: tuple[int, int]) -> bool:
    """True iff ``point`` lies on the orthogonal wire segment ``v1 → v2``."""
    px, py = point
    x1, y1 = v1
    x2, y2 = v2
    if x1 == x2:
        return px == x1 and min(y1, y2) <= py <= max(y1, y2)
    if y1 == y2:
        return py == y1 and min(x1, x2) <= px <= max(x1, x2)
    # Diagonal wire — shouldn't happen in LTspice, but if it does, fall
    # back to endpoint-only matching.
    return point in (v1, v2)


def _trace_nets(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
) -> dict[tuple[int, int], frozenset[str]]:
    """Map each pin/label/wire coordinate to the labels on its net.

    Segment-aware: a label or pin lying anywhere ON a wire (not just at
    an endpoint) is unioned with that wire — endpoint-only matching
    misses FLAGs placed mid-segment.

    ``extra_segments`` lets the caller include not-yet-committed wire
    segments (e.g. the route ``connect`` is about to add) so the
    short-detection check operates on the post-route net layout.
    """
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(p: tuple[int, int]) -> tuple[int, int]:
        if p not in parent:
            parent[p] = p
            return p
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Collect every "interest point": pin coords + label coords + wire
    # endpoints. A wire that touches one of these in its interior pulls
    # it into the same connected component as its endpoints.
    interest_points: set[tuple[int, int]] = set()
    for ref in editor.get_components():
        for coord in _component_pin_coords(editor, ref):
            interest_points.add(coord)
            find(coord)
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        interest_points.add(coord)
        find(coord)

    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for w in editor.wires:
        segments.append(((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))))
    if extra_segments:
        for sx1, sy1, sx2, sy2 in extra_segments:
            segments.append(((sx1, sy1), (sx2, sy2)))

    # Wire endpoints are interest points themselves.
    for v1, v2 in segments:
        interest_points.add(v1)
        interest_points.add(v2)
        union(v1, v2)

    # For each segment, union every interest point lying on it with the
    # segment's endpoints. This is O(segments * interest_points) — fine
    # for typical schematics (a few hundred of each).
    for v1, v2 in segments:
        for pt in interest_points:
            if pt in (v1, v2):
                continue
            if _point_on_segment(pt, v1, v2):
                union(pt, v1)

    labels_by_root: dict[tuple[int, int], set[str]] = {}
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        labels_by_root.setdefault(find(coord), set()).add(lbl.text)

    return {p: frozenset(labels_by_root.get(find(p), set())) for p in parent}


def _net_label_at(
    nets: dict[tuple[int, int], frozenset[str]], coord: tuple[int, int]
) -> frozenset[str]:
    """Labels on the net at ``coord``; empty when net is unnamed."""
    return nets.get(coord, frozenset())


def _named_labels(labels: frozenset[str]) -> set[str]:
    """Strip ground ('0') from a label set so 'short to ground' isn't
    flagged as a conflict by detect-multi-label net checks."""
    return {lbl for lbl in labels if lbl != "0"}


def _post_op_warnings(editor: AscEditor) -> list[dict]:
    """Schematic-state advisories surfaced after a mutating op succeeds.

    Returns structured warnings the agent can act on without a follow-up
    inspection turn:

    - ``floating_pin`` — a component pin with no wire passing through,
      no net label sitting on it, and no other component pin sharing
      the coordinate.
    - ``duplicate_wire`` — two wire segments sharing the same endpoints
      (in either order). Pure noise, costs nothing to drop.
    - ``dangling_label`` — a net label whose coordinate is neither on a
      wire nor at any component pin.

    Read-only on the editor. Cheap to compute during an existing edit
    session; intended for callers to surface in their response payload.
    """
    pins: list[tuple[str, str, int, int]] = []
    for entry in _collect_component_geometry(editor):
        ref = entry["ref"]
        for p in entry["pins"]:
            pins.append((ref, p["name"], p["x"], p["y"]))

    pin_count_at: dict[tuple[int, int], int] = {}
    for _, _, x, y in pins:
        pin_count_at[(x, y)] = pin_count_at.get((x, y), 0) + 1

    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    def _on_any_wire(coord: tuple[int, int]) -> bool:
        return any(coord in (v1, v2) or _point_on_segment(coord, v1, v2) for v1, v2 in segments)

    warnings: list[dict] = []

    for ref, name, x, y in pins:
        coord = (x, y)
        if pin_count_at[coord] > 1:
            continue
        if coord in label_coords:
            continue
        if _on_any_wire(coord):
            continue
        pin_label = f"{ref}.{name}" if name else ref
        warnings.append(
            {
                "kind": "floating_pin",
                "ref": ref,
                "pin": name,
                "x": x,
                "y": y,
                "message": f"Floating pin: {pin_label} at ({x},{y})",
            }
        )

    seen_segments: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for v1, v2 in segments:
        if v1 == v2:
            continue
        key = (v1, v2) if v1 <= v2 else (v2, v1)
        seen_segments[key] = seen_segments.get(key, 0) + 1
    for (a, b), count in seen_segments.items():
        if count > 1:
            warnings.append(
                {
                    "kind": "duplicate_wire",
                    "from": {"x": a[0], "y": a[1]},
                    "to": {"x": b[0], "y": b[1]},
                    "count": count,
                    "message": (f"Duplicate wire ({count}×): ({a[0]},{a[1]})->({b[0]},{b[1]})"),
                }
            )

    pin_coords = pin_count_at.keys()
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        if coord in pin_coords:
            continue
        if _on_any_wire(coord):
            continue
        warnings.append(
            {
                "kind": "dangling_label",
                "label": lbl.text,
                "x": coord[0],
                "y": coord[1],
                "message": f"Dangling label '{lbl.text}' at ({coord[0]},{coord[1]})",
            }
        )

    return warnings


def _validation_warnings_lines(warnings: list[dict]) -> list[str]:
    """Format ``_post_op_warnings`` output as message lines for a text response.

    Returns ``[]`` when ``warnings`` is empty so callers can ``lines.extend``
    unconditionally without an outer guard.
    """
    if not warnings:
        return []
    return ["", "Schematic warnings:", *(f"  {w['message']}" for w in warnings)]


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
            "For remove: literal exact match by default — copy the line "
            "verbatim from ``read_circuit``. Pass ``regex:<pattern>`` to "
            "use a regex (matches against directives AND comments). Raises "
            "an error when nothing matched, so a typo can't silently leave "
            "the directive in place."
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
    cleanup_wires: bool = Field(
        default=False,
        description=(
            "When true, also delete every wire whose endpoint touches one of "
            "the removed component's pins (Fr7). Default false keeps the v2 "
            "behaviour of leaving wires in place and surfacing a warning, so "
            "callers can opt in once they've confirmed the removal is clean."
        ),
    )


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


def _atomic_save_editor(editor: Editor, target: Path) -> None:
    """Render editor to a buffer, then atomically rename onto target.

    Avoids partial-write corruption (P-N1): if rendering or writing fails,
    the original file is untouched. Spicelib's save_netlist accepts an
    io.StringIO sink (verified for AscEditor and SpiceEditor), so we can
    skip the temp-file dance and reuse atomic_write_text's tested rename.
    """
    buf = io.StringIO()
    editor.save_netlist(buf)
    atomic_write_text(
        target,
        buf.getvalue(),
        encoding=getattr(editor, "encoding", "utf-8") or "utf-8",
        durable=False,
    )


@asynccontextmanager
async def _editing(path: Path, state: SessionState) -> AsyncIterator[Editor]:
    """Get a cached editor, yield it, save atomically on success.

    Cache is invalidated unconditionally — whether the yield body raised,
    the save raised, or both succeeded. The cached editor is dirty after
    any mutation; a failed save doesn't roll back the in-memory state.
    """
    async with _get_edit_lock(path):
        editor = _get_editor(path, state)
        try:
            yield editor
            _atomic_save_editor(editor, path)
        finally:
            state.editors.invalidate(path)


@asynccontextmanager
async def _editing_asc(path: Path, state: SessionState) -> AsyncIterator[AscEditor]:
    """``_editing`` narrowed to ``AscEditor`` — caller must have validated
    ``_require_asc`` first. Same save/rollback contract."""
    async with _editing(path, state) as editor:
        assert isinstance(editor, AscEditor)
        yield editor


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

    # Pre-flight every directive line through the same validator that
    # ``edit_directive`` uses, so known-bad patterns (vdb()/phase()/
    # group_delay() inside .MEAS, etc.) are refused at write time rather
    # than only after a wasted simulation run.
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
                        # Optional non-default SYMATTR map (.asc only).
                        "attributes": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
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
        data = services.extract_netlist_info(file_path)
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
                        # Optional per-component SYMATTR map for .asc files
                        # (omitted for .cir/.net and for .asc components
                        # without non-default attributes).
                        "attributes": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
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

    .cir/.net reads use the spice_lex pipeline (no spicelib editor) so
    BOM/UTF-16 files and unclosed-``.SUBCKT`` files don't crash.
    """
    file_path = safe_path(args.path, state)
    fmt = args.format

    if args.reference is not None and args.prefix is not None:
        raise NetlistError(
            "'reference' (single lookup) and 'prefix' (filter) are mutually "
            "exclusive — provide one, not both."
        )

    if not _is_asc(file_path):
        return await _list_components_netlist(args, file_path, fmt)

    editor = _get_asc_editor(file_path, state)

    # Single-component lookup mode (absorbed from get_component_value).
    # For .asc, read the Value SYMATTR directly so Value2 doesn't get
    # concatenated into the displayed value (Fr3).
    reference = args.reference
    if reference is not None:
        if reference not in editor.components:
            raise NetlistError(f"Component '{reference}' not found")
        value = services.asc_component_value(editor, reference)
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

    is_asc = isinstance(editor, AscEditor)
    comp_list: list[dict] = []
    comp_lines = []
    for comp_ref in page:
        try:
            # For .asc, read Value SYMATTR directly (Fr3 — avoids the
            # Value+Value2 concat). For .cir/.net this branch is not
            # reached; netlists go through _list_components_netlist.
            if is_asc:
                value = services.asc_component_value(editor, comp_ref)
            else:
                value = editor.get_component_value(comp_ref)
        except Exception:
            # spicelib's component-line regex chokes on B-sources with
            # commas in if(...) expressions; degrade gracefully rather
            # than abort the whole listing (Bug K).
            value = "<unparseable>"
        entry: dict = {"reference": comp_ref, "value": value}
        # Surface non-default SYMATTRs (SpiceLine, SpiceModel, …) for
        # .asc components so callers don't need a per-component
        # component_info round-trip to spot W=10u/L=0.5u-style overrides.
        if is_asc and comp_ref in editor.components:
            attrs = {
                k: v
                for k, v in (editor.components[comp_ref].attributes or {}).items()
                if k not in ("Value", "InstName") and v
            }
            if attrs:
                entry["attributes"] = attrs
        line = f"{comp_ref}  {value}"
        if entry.get("attributes"):
            line += "  " + " ".join(f"{k}={v}" for k, v in entry["attributes"].items())
        comp_lines.append(line)
        comp_list.append(entry)

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


async def _list_components_netlist(
    args: ListComponentsInput, file_path: Path, fmt
) -> types.CallToolResult:
    """List components in a .cir/.net file via spice_lex (no editor).

    Mirrors the editor-based path's response shape: single-ref lookup
    returns ``{reference, value}``; multi-ref returns a paginated list
    with ``{components, pagination}``.
    """
    from ltspice_mcp.lib.encoding import read_spice_text
    from ltspice_mcp.lib.spice_lex import lex
    from ltspice_mcp.lib.spice_lex_views import (
        InstanceLine,
        body_has_stray_kv_remnant,
        instances_by_ref,
    )

    try:
        content = read_spice_text(file_path)
    except FileNotFoundError as e:
        raise NetlistError(f"File not found: {file_path}") from e
    cards = lex(content).cards
    refs_to_card = instances_by_ref(cards)

    def _value_of(card: SpiceCard) -> str:
        if body_has_stray_kv_remnant(card.body):
            return "<unparseable>"
        try:
            return InstanceLine.from_card(card).display_value()
        except Exception:
            return "<unparseable>"

    reference = args.reference
    if reference is not None:
        match = refs_to_card.get(reference.lower())
        if match is None:
            raise NetlistError(f"Component '{reference}' not found")
        value = _value_of(match)
        data = {"reference": reference, "value": value}
        return format_response(f"{reference} = {value}", data, fmt)

    prefix = args.prefix
    if prefix is not None and (len(prefix) != 1 or not prefix.isalpha()):
        raise NetlistError(
            f"Component prefix must be a single letter (e.g. 'R', 'C'), got {prefix!r}"
        )

    if prefix:
        upper = prefix.upper()
        components = [
            c.name for c in refs_to_card.values() if c.name and c.name[:1].upper() == upper
        ]
    else:
        components = [c.name for c in refs_to_card.values() if c.name]

    if not components:
        msg = (
            f"No components matching prefix '{prefix}' found" if prefix else "No components found"
        )
        return format_response(
            msg, {"components": [], "pagination": pagination_metadata(0, 0, 50)}, fmt
        )

    page, total, offset, limit = paginate(components, args)
    comp_list: list[dict] = []
    comp_lines: list[str] = []
    for ref in page:
        card = refs_to_card.get(ref.lower())
        value = _value_of(card) if card is not None else ""
        comp_list.append({"reference": ref, "value": value})
        comp_lines.append(f"{ref}  {value}")

    header = f"Showing {offset + 1}-{offset + len(page)} of {total} components"
    if prefix:
        header += f" (prefix '{prefix}')"
    body = header + "\n\n" + "\n".join(comp_lines)
    if offset + len(page) < total:
        body += f"\n\nNext page: ltspice_list_components(path=..., offset={offset + limit})"

    data = {
        "components": comp_list,
        "pagination": pagination_metadata(total, offset, limit),
    }
    if prefix:
        data["prefix"] = prefix
    return format_response(body, data, fmt)


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

    For .cir/.net, edits route through the typed spice_lex dispatcher
    (``lib.component_value.apply_value_to_instance``) which knows the
    body shape per element class — preserves B-source ``V=``/``I=``
    prefixes, replaces only the trailing gain on E/G positional forms,
    accepts multi-token V/I source specs (PULSE/SIN/...).
    """
    file_path = safe_path(args.path, state)

    values_dict = args.values
    reference = args.reference
    value = args.value

    single_mode_args = reference is not None or value is not None
    if values_dict is not None and single_mode_args:
        raise NetlistError(
            "Single mode ('reference'+'value') and batch mode ('values') "
            "are mutually exclusive — provide one, not both."
        )
    if values_dict is None and not single_mode_args:
        raise NetlistError("Provide either 'reference'+'value' (single) or 'values' dict (batch)")
    if single_mode_args and (reference is None or value is None):
        raise NetlistError("Single mode requires both 'reference' and 'value'")

    if values_dict is not None:
        if not isinstance(values_dict, dict):
            raise NetlistError("'values' must be an object mapping references to new values")
        if not values_dict:
            raise NetlistError("'values' dict must not be empty")
        pairs: list[tuple[str, str]] = list(values_dict.items())
    else:
        assert reference is not None and value is not None
        pairs = [(reference, value)]

    if _is_asc(file_path):
        return await _set_component_value_asc(file_path, pairs, state)
    return await _set_component_value_netlist(file_path, pairs, state)


async def _set_component_value_netlist(
    file_path: Path,
    pairs: list[tuple[str, str]],
    state: SessionState,
) -> types.CallToolResult:
    """Apply value changes to a .cir/.net via spice_lex typed dispatch.

    Validates every (ref, value) pair before mutating. The typed
    dispatch in ``lib/component_value`` rejects shape-mismatched values
    (e.g. a brace expression on a B-source with no V=/I= prefix) before
    any card is touched, so partial writes can't happen.
    """
    from ltspice_mcp.lib.component_value import apply_value_to_instance
    from ltspice_mcp.lib.encoding import read_spice_text
    from ltspice_mcp.lib.spice_lex import lex, write_cards
    from ltspice_mcp.lib.spice_lex_views import instances_by_ref

    async with _get_edit_lock(file_path):
        cards = lex(read_spice_text(file_path)).cards
        refs_to_card = instances_by_ref(cards)

        unknown_refs = [ref for ref, _ in pairs if ref.lower() not in refs_to_card]
        if unknown_refs:
            available = sorted(refs_to_card.values(), key=lambda c: c.name or "")
            available_names = [c.name for c in available if c.name]
            raise NetlistError(
                f"Component(s) not found: {', '.join(repr(r) for r in unknown_refs)}. "
                f"Available: {_format_available_refs(available_names)}"
            )

        changes: list[str] = []
        for ref, val in pairs:
            applied = apply_value_to_instance(refs_to_card[ref.lower()], val)
            changes.append(f"{applied.reference}: {applied.old_summary} -> {applied.new_summary}")

        write_cards(cards, file_path)
        state.editors.invalidate(file_path)

    if len(pairs) == 1:
        return text_response(f"Set {changes[0]}")
    return text_response(f"Updated {len(pairs)} component(s):\n" + "\n".join(changes))


async def _set_component_value_asc(
    file_path: Path,
    pairs: list[tuple[str, str]],
    state: SessionState,
) -> types.CallToolResult:
    """Apply value changes to an .asc via the cached AscEditor.

    .asc components carry separate Value / Value2 / SpiceLine slots that
    AscEditor knows how to address; we keep that path. The new validator
    still gates whitespace shapes through.
    """
    async with _editing(file_path, state) as editor:
        unknown_refs: list[str] = []
        for ref, _ in pairs:
            try:
                editor.get_component_value(ref)
            except Exception:
                unknown_refs.append(ref)
        if unknown_refs:
            raise NetlistError(
                "Component(s) not found: " + ", ".join(repr(r) for r in unknown_refs)
            )
        for ref, val in pairs:
            _validate_component_value(ref, val)
        changes: list[str] = []
        for ref, val in pairs:
            old_value = editor.get_component_value(ref)
            _apply_component_value(editor, ref, val)
            changes.append(f"{ref}: {old_value} -> {val}")

    if len(pairs) == 1:
        return text_response(f"Set {changes[0]}")
    return text_response(f"Updated {len(pairs)} component(s):\n" + "\n".join(changes))


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
                # Fr5: refuse comment text that *looks* like a directive —
                # that's almost always a mis-typed ``kind`` and would silently
                # render as ``* !.tran 1m`` (a comment of a directive).
                stripped_comment = instruction.lstrip()
                if stripped_comment.startswith("!") or stripped_comment.startswith("."):
                    raise NetlistError(
                        f"Comment text starts with {stripped_comment[:1]!r}, which "
                        "looks like a SPICE directive (e.g. '.tran', '!.ac'). "
                        "Use kind='directive' to add a directive, or rephrase the "
                        "comment so it doesn't begin with '!' or '.'."
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
                # A leading ``!`` / ``.`` is the giveaway that the user
                # actually wanted a directive, not free text.
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
            label = _remove_directive_or_comment(editor, instruction)
            result = f"Removed {label}: {instruction}"

        else:
            raise NetlistError(f"Invalid action '{action}'. Must be 'add' or 'remove'.")

    return text_response(result)


def _remove_directive_or_comment(editor, instruction: str) -> str:
    """Remove a directive or comment matching ``instruction``.

    Treats the input as a literal exact-match by default — common SPICE
    directives contain ``(`` and ``)`` (every ``.meas``/``.four``/``.print``
    referencing ``V(node)``) which would silently turn into regex capture
    groups under any "metachar means regex" heuristic. Pass an explicit
    ``regex:`` prefix to opt in to regex matching.

    Returns a label describing what was removed. Raises NetlistError when
    nothing matched, so the user can't think they cleaned a directive
    that's still in the file.
    """
    if instruction.startswith("regex:"):
        pattern = instruction[6:]
        if not pattern.strip():
            raise NetlistError(
                "Empty regex pattern would match every directive; "
                "provide an explicit regex after 'regex:'."
            )
        try:
            compiled = re.compile(pattern)
        except re.error as e:
            raise NetlistError(f"Invalid regex {pattern!r}: {e}") from e
        directive_hit = bool(editor.remove_Xinstruction(pattern))
        comment_hit = _strip_matching_comments(editor, compiled)
        if not (directive_hit or comment_hit):
            raise NetlistError(
                f"No directive or comment matched regex {pattern!r}. "
                "Use ltspice_read_circuit to see what's actually in the file."
            )
        return "directive(s)/comment(s)"

    directive_hit = bool(editor.remove_instruction(instruction))
    comment_hit = _strip_matching_comments(editor, instruction)
    if not (directive_hit or comment_hit):
        raise NetlistError(
            f"No directive or comment matched {instruction!r} exactly. "
            "Match is literal by default — pass 'regex:<pattern>' for regex "
            "matching, or copy the line verbatim from ltspice_read_circuit."
        )
    return "directive"


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


def _strip_matching_comments(editor, matcher) -> bool:
    """Best-effort removal of TEXT-COMMENT entries whose body matches.

    ``matcher`` is either a literal string (exact match) or a compiled
    regex. ``editor.directives`` only exists on AscEditor — silently
    no-op for netlist-mode editors. Returns True when at least one
    comment was removed so the caller can decide whether the overall
    remove operation hit anything.
    """
    directives = getattr(editor, "directives", None)
    if directives is None:
        return False
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
        return True
    return False


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

    # Collect pin positions before removal to check for orphaned wires.
    # Other components' pins are excluded so we don't blame this remove
    # for wires that legitimately belong to a neighbour.
    editor_pre = _get_asc_editor(asc_path, state)
    _require_component(editor_pre, reference)
    pin_coords = _component_pin_coords(editor_pre, reference)
    other_pins = _other_components_pin_coords(editor_pre, reference)
    target_only_pins = pin_coords - other_pins

    async with _editing_asc(asc_path, state) as editor:
        editor.remove_component(reference)
        deleted_wires = 0
        if args.cleanup_wires and target_only_pins:
            kept_wires = []
            for w in editor.wires:
                v1 = (int(w.V1.X), int(w.V1.Y))
                v2 = (int(w.V2.X), int(w.V2.Y))
                if v1 in target_only_pins or v2 in target_only_pins:
                    deleted_wires += 1
                    continue
                kept_wires.append(w)
            editor.wires = kept_wires
        validation_warnings = _post_op_warnings(editor)

    result = f"Removed {reference} from {asc_path.name}"
    if args.cleanup_wires and deleted_wires:
        result += f" (also deleted {deleted_wires} attached wire(s))"
    elif target_only_pins and not args.cleanup_wires:
        editor_post = _get_asc_editor(asc_path, state)
        orphaned_at: list[str] = []
        for w in editor_post.wires:
            for coord in [(int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))]:
                if coord in target_only_pins:
                    orphaned_at.append(f"({coord[0]},{coord[1]})")
                    target_only_pins.discard(coord)
        if orphaned_at:
            result += (
                f"\n\nWarning: orphaned wires remain at: {', '.join(orphaned_at)}. "
                "Re-run with cleanup_wires=true to delete them."
            )
    extra_lines = _validation_warnings_lines(validation_warnings)
    if extra_lines:
        result += "\n" + "\n".join(extra_lines)

    return text_response(result)


@registry.tool(
    name="ltspice_move_component",
    description=(
        "Move and/or rotate a component in an .asc schematic. Warns if the "
        "new position overlaps another component's bounding box, and lists "
        "wire endpoints orphaned by the move (the component's old pin "
        "coordinates are no longer connected to anything)."
    ),
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

        # Snapshot OLD pin coordinates and OTHER components' pins before
        # the position change — used after to detect orphaned wires
        # without blaming this move for neighbours' wires.
        old_pin_coords = _component_pin_coords(editor, reference)
        other_pins = _other_components_pin_coords(editor, reference)

        new_rot = _parse_rotation(rotation) if rotation is not None else old_rot
        new_pos = Point(x, y)
        editor.set_component_position(reference, new_pos, new_rot)

        new_pin_coords = _component_pin_coords(editor, reference)

        # Overlap check mirrors add_component's: any other component whose
        # bounding box intersects the moved one's NEW bounding box.
        moved_bb: dict[str, int] | None = None
        comp = editor.components[reference]
        if comp.symbol:
            moved_sym = get_symbol_info(comp.symbol)
            if moved_sym is not None:
                rot_str = new_rot.name
                moved_bb = compute_placed_geometry(moved_sym, x, y, rot_str)["bounding_box"]
        overlap_warnings: list[str] = []
        if moved_bb is not None:
            for ebb in _collect_component_geometry(editor):
                if ebb["ref"] == reference:
                    continue
                if _bboxes_overlap(moved_bb, ebb):
                    overlap_warnings.append(f"Overlaps {ebb['ref']} bounding box")

        # Wires whose endpoint sat on a pin that moved — and that pin's
        # coordinate isn't now another component's pin — are orphaned by
        # the move. Don't auto-delete; surface so the caller can fix routing.
        abandoned_pins = old_pin_coords - new_pin_coords - other_pins
        orphan_coords: list[tuple[int, int]] = []
        if abandoned_pins:
            for w in editor.wires:
                for coord in (
                    (int(w.V1.X), int(w.V1.Y)),
                    (int(w.V2.X), int(w.V2.Y)),
                ):
                    if coord in abandoned_pins and coord not in orphan_coords:
                        orphan_coords.append(coord)

        validation_warnings = _post_op_warnings(editor)

    rot_str = f"R{new_rot.value}" if new_rot.value < 360 else f"M{new_rot.value - 360}"
    msg = f"Moved {reference}: ({old_pos.X},{old_pos.Y}) -> ({x},{y}) {rot_str}"
    if overlap_warnings:
        msg += "\n\nWarnings:"
        for w_msg in overlap_warnings:
            msg += f"\n  {w_msg}"
    if orphan_coords:
        coord_str = ", ".join(f"({cx},{cy})" for cx, cy in orphan_coords)
        msg += (
            f"\n\nWarning: wires left at old pin coordinates with no "
            f"connection: {coord_str}. Re-route or delete these wires."
        )
    extra_lines = _validation_warnings_lines(validation_warnings)
    if extra_lines:
        msg += "\n" + "\n".join(extra_lines)
    return text_response(msg)


@registry.tool(
    name="ltspice_set_component_attribute",
    description=(
        "Set a schematic-only component attribute. The standard LTspice slots "
        "are ``Value``, ``Value2``, ``SpiceLine``, ``SpiceLine2``, "
        "``SpiceModel``, ``InstName`` — anything else is rejected, since "
        "LTspice silently ignores unknown SYMATTR keys at netlist time. To "
        "set arbitrary KEY=val pairs (e.g. ``W=10u L=0.5u``), pass them as "
        "the ``SpiceLine`` value."
    ),
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

    if attribute not in _LTSPICE_ATTR_NAMES:
        canonical = _LTSPICE_ATTR_CANONICAL.get(attribute.lower())
        if canonical:
            raise NetlistError(
                f"Unknown attribute {attribute!r}. Did you mean {canonical!r}? "
                f"LTspice attribute names are case-sensitive."
            )
        raise NetlistError(
            f"Unknown attribute {attribute!r}. LTspice silently ignores "
            f"unrecognised SYMATTR keys at netlist time. Valid attributes: "
            f"{', '.join(sorted(_LTSPICE_ATTR_NAMES))}. For arbitrary KEY=val "
            "pairs, set them through 'SpiceLine' instead."
        )

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
            "validation_warnings": VALIDATION_WARNINGS_SCHEMA,
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
        validation_warnings = _post_op_warnings(editor)

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
    if validation_warnings:
        data["validation_warnings"] = validation_warnings
    extra_lines = _validation_warnings_lines(validation_warnings)
    if extra_lines:
        result += "\n" + "\n".join(extra_lines)

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
    lines.append(f"Size: {sym_info.bbox.width}x{sym_info.bbox.height}")
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

        # Floating-label detection: a FLAG placed at a coordinate with
        # no wire endpoint and no component pin is silently ignored at
        # netlist time. Surface a warning so the caller can verify; don't
        # refuse outright since a common workflow is "place labels first,
        # wire them up later".
        wire_endpoints = {(int(w.V1.X), int(w.V1.Y)) for w in editor.wires} | {
            (int(w.V2.X), int(w.V2.Y)) for w in editor.wires
        }
        all_pin_coords: set[tuple[int, int]] = set()
        for ref in editor.get_components():
            all_pin_coords.update(_component_pin_coords(editor, ref))
        is_floating = (x, y) not in wire_endpoints and (x, y) not in all_pin_coords

        # Net-label conflict: adding a label to a coord that's already
        # on a different named net would short the two nets at netlist
        # time. Catch deliberately, with a clear message.
        if net != "0":
            nets = _trace_nets(editor)
            existing = _net_label_at(nets, (x, y))
            other_labels = {n for n in existing if n != net and n != "0"}
            if other_labels:
                raise NetlistError(
                    f"Refused to add net '{net}' at ({x},{y}): the wire "
                    f"network at this coordinate already carries the label(s) "
                    f"{sorted(other_labels)}. Adding '{net}' would short "
                    f"those nets together. Remove the existing label(s) "
                    f"first or pick a different coordinate."
                )

        editor.labels.append(Text(coord=Point(x, y), text=net, type=TextTypeEnum.LABEL))

    if is_floating:
        result = (
            f"Warning: ({x},{y}) has no wire endpoint or component pin — "
            f"LTspice will ignore this floating label until you wire it up.\n"
        ) + result
    result += f"Added {label_desc} at ({x},{y})"
    return text_response(result)


class _ConnectPlan(NamedTuple):
    """Validated connect route ready to commit to the editor."""

    x1: int
    y1: int
    x2: int
    y2: int
    points: list[tuple[int, int]]
    segments: list[tuple[int, int, int, int]]
    warnings: list[str]


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
            "validation_warnings": VALIDATION_WARNINGS_SCHEMA,
        },
    },
)
async def handle_connect(args: ConnectInput, state: SessionState) -> types.CallToolResult:
    """Connect two pins with auto-routed or waypoint-guided wires."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    pre_editor = _get_asc_editor(asc_path, state)
    plan = _plan_connect_route(pre_editor, args.from_pin, args.to_pin, args.waypoints)

    async with _editing_asc(asc_path, state) as ed:
        for sx1, sy1, sx2, sy2 in plan.segments:
            ed.wires.append(Line(Point(sx1, sy1), Point(sx2, sy2)))
        validation_warnings = _post_op_warnings(ed)

    x1, y1, x2, y2, segments, warnings = (
        plan.x1,
        plan.y1,
        plan.x2,
        plan.y2,
        plan.segments,
        plan.warnings,
    )
    points = plan.points

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
    if validation_warnings:
        data["validation_warnings"] = validation_warnings
    result_lines.extend(_validation_warnings_lines(validation_warnings))

    return format_response("\n".join(result_lines), data, None)


def _plan_connect_route(
    editor: AscEditor,
    from_pin: str,
    to_pin: str,
    waypoints: list[WaypointInput],
) -> _ConnectPlan:
    """Resolve, route, and validate a wire path between two pins.

    Returns a :class:`_ConnectPlan` whose ``segments`` are ready to append
    to ``editor.wires`` directly. Raises ``NetlistError`` for any
    validation failure (zero-length route, diagonal segment, pin
    collision, wire-junction overlap, named-net short).

    Shared by ``handle_connect`` and the ``connect`` op of
    ``apply_schematic_ops`` so both paths apply identical safety checks.
    """
    component_geo = _collect_component_geometry(editor)
    existing_wires = [(int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in editor.wires]

    x1, y1 = _resolve_pin(from_pin, editor)
    x2, y2 = _resolve_pin(to_pin, editor)

    if (x1, y1) == (x2, y2) and not waypoints:
        raise NetlistError(
            f"Cannot connect {from_pin} to {to_pin}: "
            f"both endpoints resolve to the same coordinate ({x1},{y1})."
        )

    raw_points: list[tuple[int, int]] = [(x1, y1)]
    raw_points.extend((wp.x, wp.y) for wp in waypoints)
    raw_points.append((x2, y2))
    points: list[tuple[int, int]] = [raw_points[0]]
    for pt in raw_points[1:]:
        if pt != points[-1]:
            points.append(pt)

    segments: list[tuple[int, int, int, int]] = []
    for i in range(len(points) - 1):
        px1, py1 = points[i]
        px2, py2 = points[i + 1]
        if px1 != px2 or py1 != py2:
            segments.append((px1, py1, px2, py2))

    if not segments:
        raise NetlistError(
            f"Cannot connect {from_pin} to {to_pin}: "
            "the requested route has zero length after deduplicating waypoints."
        )

    endpoints = {(x1, y1), (x2, y2)}
    skip_refs = {
        ref.rsplit(".", 1)[0]
        for ref in (from_pin, to_pin)
        if "." in ref and not ref.startswith("net:")
    }
    errors: list[str] = []
    warnings: list[str] = []

    # Net-label conflict — checked first because it's a "wrong intent"
    # error: rejecting it gives the user a clearer signal than a route
    # geometry complaint. Skip when either side uses ``net:`` form (those
    # are already named explicitly). Two-phase check:
    #   1) BEFORE state — endpoints resolve to two different already-named
    #      nets (the standard short).
    #   2) AFTER state — proposed route drags a mid-segment label into
    #      the union, merging an additional named net.
    if not from_pin.startswith("net:") and not to_pin.startswith("net:"):
        nets_before = _trace_nets(editor)
        from_labels_before = _named_labels(_net_label_at(nets_before, (x1, y1)))
        to_labels_before = _named_labels(_net_label_at(nets_before, (x2, y2)))
        if (
            from_labels_before
            and to_labels_before
            and from_labels_before.isdisjoint(to_labels_before)
        ):
            raise NetlistError(
                f"Refused to connect {from_pin} to {to_pin}: "
                f"Net-label conflict — {from_pin} is on net "
                f"{sorted(from_labels_before)} and {to_pin} is on net "
                f"{sorted(to_labels_before)}. Connecting them would short "
                f"the two named nets. Pick one labelling and rewire, or "
                f"use add_net_label to merge them deliberately."
            )
        nets_after = _trace_nets(editor, extra_segments=segments)
        from_labels_after = _named_labels(_net_label_at(nets_after, (x1, y1)))
        to_labels_after = _named_labels(_net_label_at(nets_after, (x2, y2)))
        unioned = from_labels_after | to_labels_after
        if len(unioned) >= 2:
            # Some labels seen post-route weren't there pre-route on
            # either endpoint — that's the mid-segment case.
            unioned_before = from_labels_before | to_labels_before
            new_labels = unioned - unioned_before
            if new_labels:
                raise NetlistError(
                    f"Refused to connect {from_pin} to {to_pin}: "
                    f"Net-label conflict — the proposed route would "
                    f"merge named nets {sorted(unioned)} (a label on a "
                    f"mid-segment of the wire path adds "
                    f"{sorted(new_labels)} to the merged net). Reroute "
                    "to avoid the labelled wire."
                )

    for sx1, sy1, sx2, sy2 in segments:
        if sx1 != sx2 and sy1 != sy2:
            errors.append(f"Diagonal wire ({sx1},{sy1})->({sx2},{sy2}): not orthogonal")

    # Pin-collision check: a pin is safe if it's already wired to the
    # same net as our target (an existing wire reaches both that pin and
    # one of our endpoints), e.g. T-junction onto a power rail.
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
                if _point_on_segment((px, py), (sx1, sy1), (sx2, sy2)):
                    errors.append(
                        f"Wire passes through {cg['ref']}.{pin['name']} at ({px},{py}): "
                        "will create unintended connection"
                    )

    # Wire-junction check: forbid overlaps with existing wires unless the
    # existing wire already terminates at one of our endpoints (intended
    # T-junction).
    for sx1, sy1, sx2, sy2 in segments:
        for ex1, ey1, ex2, ey2 in existing_wires:
            ext_endpoints = {(ex1, ey1), (ex2, ey2)}
            if ext_endpoints & endpoints:
                continue
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

    if errors:
        error_lines = [f"Refused to connect {from_pin} to {to_pin}:"]
        for e in errors:
            error_lines.append(f"  {e}")
        error_lines.append("\nFix the route with different waypoints to avoid these issues.")
        raise NetlistError("\n".join(error_lines))

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

    return _ConnectPlan(x1, y1, x2, y2, points, segments, warnings)


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


def _analysis_kind(line_lower: str) -> str | None:
    """Return the analysis-kind token (``tran``/``ac``/``dc``/``op``/``noise``)
    if ``line_lower`` is a directive of that kind, else None.

    ``.option`` looks like ``.op`` to a naive prefix match, so it gets an
    explicit guard here.
    """
    for kind in ANALYSIS_KINDS:
        if line_lower.startswith(f".{kind}"):
            if kind == "op" and line_lower.startswith(".option"):
                return None
            return kind
    return None


def _b_source_unparseable_issue(lineno: int, line: str) -> dict:
    """Issue dict for a B-source whose ``if(...)`` body breaks spicelib's
    component regex (Bug K). The simulator parses the line fine — only
    spicelib introspection (read_circuit, list_components) is affected.
    """
    return {
        "severity": "warning",
        "line": lineno,
        "directive": line,
        "message": (
            "Behavioural source uses an ``if(...)`` expression with commas — "
            "spicelib's component-line regex rejects this shape, so "
            "``read_circuit`` and ``list_components`` will report "
            "``<unparseable>`` for this ref. The LTspice simulator parses it fine."
        ),
        "suggestion": (
            "If you need spicelib to introspect the value, rewrite as "
            "``limit(...)`` or split into multiple B-sources without commas."
        ),
    }


def _multiple_analyses_issue(
    analysis_lines: dict[str, list[tuple[int, str]]],
    duplicate_kinds: list[str],
) -> dict:
    """Build the issue surfaced when a netlist has more than one analysis
    directive (LTspice rejects with "More than one analysis specified")."""
    flat = sorted((ln, body, k) for k, entries in analysis_lines.items() for ln, body in entries)
    first_lineno, first_line, _ = flat[0] if flat else (None, "", None)
    if duplicate_kinds:
        kind_str = ", ".join(f".{k}" for k in duplicate_kinds)
        message = (
            f"Duplicate analysis directive ({kind_str}). LTspice rejects "
            "this with 'More than one analysis specified.'"
        )
    else:
        kind_str = ", ".join(f".{k}" for k in sorted(analysis_lines))
        message = (
            f"Multiple distinct analysis directives ({kind_str}). LTspice "
            "rejects this with 'More than one analysis specified.'"
        )
    return {
        "severity": "error",
        "line": first_lineno,
        "directive": first_line,
        "message": message,
        "suggestion": "Keep exactly one analysis directive; remove the others.",
    }


def _meas_mismatch_issue(
    lineno: int, line: str, kind: str, active_kinds: "AbstractSet[str]"
) -> dict:
    """Issue for a ``.meas <kind>`` whose analysis isn't in the file
    (LTspice silently drops these)."""
    active = ", ".join(sorted(active_kinds)) or "none"
    if len(active_kinds) == 1:
        target = next(iter(active_kinds))
        suggestion = (
            f"Use ``.meas {target} ...`` to match the analysis, "
            f"or add a separate .{kind} simulation."
        )
    elif active_kinds:
        options = ", ".join(f".meas {k}" for k in sorted(active_kinds))
        suggestion = (
            f"Use one of the analysis-matching forms ({options}), or add "
            f"a separate .{kind} simulation."
        )
    else:
        suggestion = f"Add a .{kind} directive, or remove the .meas {kind} lines."
    return {
        "severity": "error",
        "line": lineno,
        "directive": line,
        "message": (
            f".meas {kind} requires a .{kind} analysis but the active "
            f"analysis directives are: {active}. LTspice silently drops "
            f".meas {kind} on non-.{kind} runs, so this measurement "
            "won't appear in the log."
        ),
        "suggestion": suggestion,
    }


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
    # Single pass: validate directives, collect each analysis directive,
    # bookmark every ``.meas <kind>`` line, and (on netlists) sniff
    # B-sources whose ``if(...)`` body breaks spicelib's component regex.
    sniff_b_source = not _is_asc(file_path)
    analysis_lines: dict[str, list[tuple[int, str]]] = {}
    meas_lines: list[tuple[int, str, str]] = []
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if sniff_b_source and line[:1].upper() == "B":
            lower_b = line.lower()
            if "if(" in lower_b and "," in line:
                issues.append(_b_source_unparseable_issue(lineno, line))
            continue
        if not line.startswith("."):
            continue
        lower = line.lower()
        analysis_kind = _analysis_kind(lower)
        if analysis_kind is not None:
            analysis_lines.setdefault(analysis_kind, []).append((lineno, line))
        elif lower.startswith(".meas"):
            tokens = lower.split()
            if len(tokens) >= 2 and tokens[1] in MEAS_KINDS:
                meas_lines.append((lineno, line, tokens[1]))
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

    # LTspice rejects more than one analysis directive with "More than one
    # analysis specified."
    duplicate_kinds = sorted(k for k, entries in analysis_lines.items() if len(entries) > 1)
    if duplicate_kinds or len(analysis_lines) > 1:
        issues.append(_multiple_analyses_issue(analysis_lines, duplicate_kinds))

    # ``.meas <kind>`` runs only when the matching analysis is present.
    # LTspice silently drops mismatched .meas lines from the log.
    active_kinds = analysis_lines.keys()
    for lineno, line, kind in meas_lines:
        if kind in active_kinds:
            continue
        issues.append(_meas_mismatch_issue(lineno, line, kind, active_kinds))

    # Element-arity pass (C-N4): walk lexer cards, consult ELEMENT_SPECS,
    # flag instances with too few positional nodes or B-sources missing
    # the V=/I= prefix. LTspice's "Expected 2 node names here" / "Unknown
    # parameter" errors at runtime become up-front issues here.
    try:
        arity_cards = lex(content).cards
    except SpiceLexError:
        arity_cards = []
    for arity_issue in validate_netlist_arity(arity_cards):
        issues.append({"severity": "error", **arity_issue})

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
    if _is_asc(path):
        try:
            ed = _make_editor(path)
        except Exception:
            return {}, set()
        assert isinstance(ed, AscEditor)
        info = services.extract_asc_info(ed, path)
        components = {comp["reference"]: str(comp["value"]) for comp in info["components"]}
        directives = {d.strip() for d in info.get("directives", []) if d.strip().startswith(".")}
        return components, directives
    try:
        info = services.extract_netlist_info(path)
    except Exception:
        return {}, set()
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
    at: str | None = Field(
        default=None,
        description=(
            "Optional inner-axis position to query within the chosen step "
            "(time for .tran, frequency for .ac). Defaults to the first "
            "sample, which is the only useful answer for stepped .op runs "
            "but rarely the right one for .ac/.tran. SPICE notation."
        ),
    )
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

    # Fallback: spicelib step lookup, falling back to .log parsing if
    # spicelib returns nothing (which it does for ``.step param NAME``
    # runs — the parameter map is in the log, not the .raw header).
    try:
        steps = list(raw.get_steps() or [])
    except Exception:
        steps = []

    if not any(isinstance(s, dict) and s for s in steps):
        # parse_step_iterations swallows OSError, so no .exists() guard.
        steps = list(parse_step_iterations(raw_path.with_suffix(".log")))

    best_idx = None
    best_actual: float | None = None
    for i, step_record in enumerate(steps):
        if not isinstance(step_record, dict):
            continue
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
        # Build the axis listing only on the error path.
        available_axes: list[str] = []
        for step_record in steps:
            if isinstance(step_record, dict):
                for k in step_record:
                    if k not in available_axes:
                        available_axes.append(k)
        raise NetlistError(
            f"Step axis {args.axis!r} not found in this raw file. "
            "Available axes: " + (", ".join(available_axes) if available_axes else "<none>")
        )

    wave = raw.get_wave(signal, step=best_idx)
    if len(wave) == 0:
        raise NetlistError(
            f"Step {best_idx} of {signal!r} contains no samples; "
            "verify the simulation completed and the signal exists in this step."
        )

    # Pick the inner-axis sample. Default is index 0 (correct for .op
    # results); when ``at=`` is given, find the nearest neighbour on the
    # per-step axis (frequency for .AC, time for .TRAN).
    inner_idx = 0
    target_at: float | None = None
    actual_at: float | None = None
    if args.at is not None:
        try:
            target_at = parse_spice_value(args.at)
        except ValueError as e:
            raise NetlistError(f"Invalid at {args.at!r}: {e}") from e
        try:
            inner_axis = real_axis(np.asarray(raw.get_axis(step=best_idx)))
        except Exception as e:
            raise NetlistError(
                f"Cannot read inner axis for at={args.at!r}: {e}. "
                "Drop the ``at`` argument for .op-style raws."
            ) from e
        if inner_axis.size == 0:
            raise NetlistError(f"Step {best_idx} has an empty axis; ``at`` cannot be applied.")
        inner_idx = nearest_index(inner_axis, target_at)
        actual_at = float(inner_axis[inner_idx])

    sample_dict = sample_to_dict(wave[inner_idx])
    data: dict = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": best_actual,
        "step_index": best_idx,
        **sample_dict,
    }
    if target_at is not None:
        data["requested_at"] = target_at
        data["actual_at"] = actual_at

    sample_str = (
        f"{sample_dict['value']:g}"
        if "value" in sample_dict
        else f"{sample_dict['magnitude_db']:.3f} dB / {sample_dict['phase_deg']:.2f}°"
    )
    at_str = f", at={actual_at:g}" if actual_at is not None else ""
    summary = f"{signal} at {args.axis}={best_actual:g} (step {best_idx}){at_str}: {sample_str}"
    return format_response(summary, data, args.format)


# ---------------------------------------------------------------------------
# Batch-transaction op (Fr1) — apply many edits to one .asc atomically.
# ---------------------------------------------------------------------------


_RotationLiteral = Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"]


class _OpAddComponent(StrictModel):
    op: Literal["add_component"]
    reference: str
    symbol: str
    x: int
    y: int
    rotation: _RotationLiteral = "R0"
    value: str | None = None
    attributes: dict[str, str] | None = None


class _OpSetComponentValue(StrictModel):
    op: Literal["set_component_value"]
    reference: str
    value: str


class _OpSetComponentAttribute(StrictModel):
    op: Literal["set_component_attribute"]
    reference: str
    attribute: str
    value: str


class _OpRemoveComponent(StrictModel):
    op: Literal["remove_component"]
    reference: str
    cleanup_wires: bool = False


class _OpMoveComponent(StrictModel):
    op: Literal["move_component"]
    reference: str
    x: int
    y: int
    rotation: _RotationLiteral | None = None


class _OpAddNetLabel(StrictModel):
    op: Literal["add_net_label"]
    net: str
    pin: str | None = None
    x: int | None = None
    y: int | None = None


class _OpConnect(StrictModel):
    op: Literal["connect"]
    from_pin: str
    to_pin: str
    waypoints: list[WaypointInput] = Field(default_factory=list)


class _OpAddDirective(StrictModel):
    op: Literal["add_directive"]
    instruction: str
    kind: Literal["directive", "comment"] = "directive"
    x: int | None = None
    y: int | None = None
    size: int = 2


SchematicOp = (
    _OpAddComponent
    | _OpSetComponentValue
    | _OpSetComponentAttribute
    | _OpRemoveComponent
    | _OpMoveComponent
    | _OpAddNetLabel
    | _OpConnect
    | _OpAddDirective
)


class ApplySchematicOpsInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    ops: list[SchematicOp] = Field(
        description=(
            "List of edit operations applied in order against a single in-memory "
            "AscEditor. The file is saved once at the end iff every op succeeded "
            "(or stop_on_error=false). Each op is tagged by its ``op`` field; see "
            "the schema for per-op fields."
        )
    )
    stop_on_error: bool = Field(
        default=True,
        description=(
            "When true (default), the first op that raises aborts the transaction "
            "and nothing is saved. When false, every op runs and per-op errors "
            "are recorded in ``results``; the file IS saved with whatever ops "
            "did succeed — set false only when failures are recoverable."
        ),
    )


def _apply_op_inplace(editor: AscEditor, op: SchematicOp, asc_path: Path) -> dict[str, object]:
    """Apply one schematic op against ``editor`` in place, return its result.

    Mirrors the validation done by the per-op tools but skips the load /
    save / lock dance — the caller (``handle_apply_schematic_ops``) holds
    the lock and saves once at the end.

    Raises ``NetlistError`` on any per-op validation failure; the caller
    decides whether to abort or continue based on ``stop_on_error``.
    """
    if isinstance(op, _OpAddComponent):
        if get_symbol_info(op.symbol) is None:
            raise NetlistError(f"Symbol '{op.symbol}' not found in any configured symbol library.")
        if op.reference in editor.components:
            raise NetlistError(f"Component '{op.reference}' already exists in {asc_path.name}.")
        erot = _parse_rotation(op.rotation)
        _create_component(
            editor,
            op.reference,
            op.symbol,
            op.x,
            op.y,
            erot,
            value=op.value,
            attributes=op.attributes,
        )
        return {"op": "add_component", "reference": op.reference}

    if isinstance(op, _OpSetComponentValue):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        _apply_component_value(editor, op.reference, op.value)
        return {"op": "set_component_value", "reference": op.reference, "value": op.value}

    if isinstance(op, _OpSetComponentAttribute):
        if op.attribute not in _LTSPICE_ATTR_NAMES:
            raise NetlistError(
                f"Unknown attribute {op.attribute!r}. Valid: "
                f"{', '.join(sorted(_LTSPICE_ATTR_NAMES))}."
            )
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        editor.set_component_attribute(op.reference, op.attribute, op.value)
        return {
            "op": "set_component_attribute",
            "reference": op.reference,
            "attribute": op.attribute,
        }

    if isinstance(op, _OpRemoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        target_only = _component_pin_coords(editor, op.reference) - _other_components_pin_coords(
            editor, op.reference
        )
        editor.remove_component(op.reference)
        if op.cleanup_wires and target_only:
            editor.wires = [
                w
                for w in editor.wires
                if (int(w.V1.X), int(w.V1.Y)) not in target_only
                and (int(w.V2.X), int(w.V2.Y)) not in target_only
            ]
        return {"op": "remove_component", "reference": op.reference}

    if isinstance(op, _OpMoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        new_rot = (
            _parse_rotation(op.rotation)
            if op.rotation is not None
            else editor.get_component_position(op.reference)[1]
        )
        editor.set_component_position(op.reference, Point(op.x, op.y), new_rot)
        return {"op": "move_component", "reference": op.reference}

    if isinstance(op, _OpAddNetLabel):
        if op.pin is not None:
            x, y = _resolve_pin(op.pin, editor)
        elif op.x is not None and op.y is not None:
            x, y = op.x, op.y
        else:
            raise NetlistError("add_net_label needs either pin or both x and y.")
        editor.labels.append(Text(coord=Point(x, y), text=op.net, type=TextTypeEnum.LABEL))
        return {"op": "add_net_label", "net": op.net, "x": x, "y": y}

    if isinstance(op, _OpConnect):
        plan = _plan_connect_route(editor, op.from_pin, op.to_pin, op.waypoints)
        for sx1, sy1, sx2, sy2 in plan.segments:
            editor.wires.append(Line(Point(sx1, sy1), Point(sx2, sy2)))
        return {
            "op": "connect",
            "from_pin": op.from_pin,
            "to_pin": op.to_pin,
            "wire_count": len(plan.segments),
        }

    if isinstance(op, _OpAddDirective):
        # Inline the minimal directive-validation that edit_directive does.
        # Comments allow any text; SPICE directives go through validate_directive.
        if op.kind == "directive":
            err = validate_directive(op.instruction, simulator="LTspice")
            if err is not None:
                raise NetlistError(
                    f"Refusing directive {op.instruction!r}: {err.message} ({err.suggestion})"
                )
        coord = Point(op.x if op.x is not None else 16, op.y if op.y is not None else 16)
        text_type = TextTypeEnum.DIRECTIVE if op.kind == "directive" else TextTypeEnum.COMMENT
        editor.directives.append(
            Text(coord=coord, text=op.instruction, type=text_type, size=op.size)
        )
        return {"op": "add_directive", "instruction": op.instruction}

    raise NetlistError(f"Unknown op type: {type(op).__name__}")


@registry.tool(
    name="ltspice_apply_schematic_ops",
    description=(
        "Apply many .asc edits in one transaction. Loads the schematic once, "
        "runs each op against the in-memory editor in order, and saves once at "
        "the end. Cuts the typical 25+ tool calls to build a real circuit "
        "(add_component × N + connect × N + add_net_label × N + edit_directive "
        "× N) down to a single round-trip.\n\n"
        "Supported ops (each tagged via the ``op`` field): ``add_component``, "
        "``set_component_value``, ``set_component_attribute``, "
        "``remove_component``, ``move_component``, ``add_net_label``, "
        "``connect``, ``add_directive``.\n\n"
        "By default, the first op that raises aborts the whole transaction "
        "and nothing is written to disk. Set ``stop_on_error=false`` to run "
        "every op and persist whatever subset succeeded — useful when each op "
        "is independent and partial progress is acceptable. Errors are "
        "recorded under each op's ``error`` field; successes carry the "
        "per-op result keys (e.g. ``wire_count``)."
    ),
    input_model=ApplySchematicOpsInput,
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
            "path": {"type": "string"},
            "applied_count": {"type": "integer"},
            "failed_count": {"type": "integer"},
            "saved": {"type": "boolean"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "op": {"type": "string"},
                        "ok": {"type": "boolean"},
                        "error": {"type": ["string", "null"]},
                    },
                    "required": ["index", "op", "ok"],
                },
            },
            "validation_warnings": VALIDATION_WARNINGS_SCHEMA,
        },
        "required": ["path", "applied_count", "failed_count", "saved", "results"],
    },
)
async def handle_apply_schematic_ops(
    args: ApplySchematicOpsInput, state: SessionState
) -> types.CallToolResult:
    """Apply a list of schematic edits in one transaction (Fr1)."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)

    if not args.ops:
        raise NetlistError("ops list is empty — pass at least one op.")

    results: list[dict[str, object]] = []
    applied = 0
    failed = 0
    saved = False
    abort_reason: str | None = None

    validation_warnings: list[dict] = []
    async with _get_edit_lock(asc_path):
        editor = _get_asc_editor(asc_path, state)
        try:
            for i, op in enumerate(args.ops):
                entry: dict[str, object] = {"index": i, "op": op.op, "ok": True, "error": None}
                try:
                    op_result = _apply_op_inplace(editor, op, asc_path)
                    entry.update({k: v for k, v in op_result.items() if k != "op"})
                    applied += 1
                except (NetlistError, ValueError) as e:
                    entry["ok"] = False
                    entry["error"] = str(e)
                    failed += 1
                    results.append(entry)
                    if args.stop_on_error:
                        abort_reason = f"op #{i} ({op.op}) failed: {e}"
                        break
                    else:
                        continue
                results.append(entry)

            if args.stop_on_error and failed:
                # Evict from cache so the next caller re-reads from disk; the
                # in-memory mutations on ``editor`` are discarded with the
                # local reference once this scope exits.
                state.editors.invalidate(asc_path)
            else:
                validation_warnings = _post_op_warnings(editor)
                _atomic_save_editor(editor, asc_path)
                state.editors.invalidate(asc_path)
                saved = True
        except BaseException:
            # Uncaught exception (KeyboardInterrupt, CancelledError, or any
            # non-NetlistError/ValueError from _apply_op_inplace). The file
            # on disk is intact — _atomic_save_editor either hasn't run or
            # renames atomically — but the cached editor has mutations
            # from earlier ops in this batch. Evict so the next caller
            # doesn't see them.
            state.editors.invalidate(asc_path)
            raise

    summary_lines = [f"apply_schematic_ops on {asc_path.name}: {applied} ok, {failed} failed"]
    if abort_reason:
        summary_lines.append(f"Transaction aborted — {abort_reason}")
        summary_lines.append("No changes were saved.")
    elif saved:
        summary_lines.append("All changes saved.")
    for r in results:
        marker = "ok" if r["ok"] else "ERR"
        prefix = f"  [{r['index']}] {r['op']}  {marker}"
        if r["ok"]:
            summary_lines.append(prefix)
        else:
            summary_lines.append(f"{prefix}  {r['error']}")

    data: dict = {
        "path": str(asc_path),
        "applied_count": applied,
        "failed_count": failed,
        "saved": saved,
        "results": results,
    }
    if validation_warnings:
        data["validation_warnings"] = validation_warnings
    summary_lines.extend(_validation_warnings_lines(validation_warnings))
    return format_response("\n".join(summary_lines), data, None)
