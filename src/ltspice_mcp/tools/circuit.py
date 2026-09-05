"""Unified circuit editing tools for .cir/.net netlists and .asc schematics.

Extension-based dispatch: the file extension determines which spicelib editor
is used (SpiceEditor for .cir/.net, AscEditor for .asc).  Schematic-only
operations (position, rotation, attributes, export) validate the extension
and raise NetlistError if given a non-.asc file.
"""

import asyncio
import bisect
import importlib
import itertools
import re
from collections import defaultdict
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, NamedTuple

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
from spicelib.raw.raw_read import RawRead

# The concrete class to instantiate for a from-scratch .asc component.
# spicelib 1.6 introduced ``AscComponent`` (the type its own .asc parser
# builds) and turned ``Component.attributes`` into a lazy property that raises
# ``NotImplementedError`` for a bare ``SchematicComponent`` whose attribute
# store is still empty — so building one from scratch detonates on first
# attribute access. ``AscComponent`` implements the contract; on spicelib < 1.6
# (no ``AscComponent``) the plain ``SchematicComponent`` already works.
# Resolved dynamically because the symbol does not exist on the pinned (<1.6)
# spicelib, so a static import would not type-check.
try:
    _SchematicComponentClass: type = importlib.import_module(
        "spicelib.editor.asc_editor"
    ).AscComponent
except (ImportError, AttributeError):  # spicelib < 1.6 (the currently pinned range)
    _SchematicComponentClass = SchematicComponent

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.raw_parser import nearest_index, real_axis, sample_to_dict
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, tokenize_body
from ltspice_mcp.lib.spice_validator import (
    validate_directive,
)
from ltspice_mcp.lib.symbol_geometry import SymbolInfo, compute_placed_geometry, get_symbol_info
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    WARNINGS_SCHEMA,
    StrictModel,
    ToolInput,
    circuit_file_lock,
    declare_output_schema,
    format_response,
    path_lock,
    safe_path,
)


def _reject_empty_attr_value(reference: str, attribute: str, value: str) -> None:
    """Refuse an empty SYMATTR value on any .asc write path.

    LTspice writes ``SYMATTR <attr>`` (keyword + name, no value) for an empty
    value; spicelib's parser then can't unpack that 2-token line on the NEXT
    read, leaving the .asc permanently unreadable until hand-edited. Reject up
    front rather than emit an .asc the server's own reader rejects.
    """
    if not value.strip():
        raise NetlistError(
            f"Attribute {attribute!r} on {reference!r}: an empty value writes a "
            "2-token SYMATTR line the schematic parser cannot read back, "
            "corrupting the .asc. Provide a non-empty value or omit the attribute."
        )


def _require_clearable_attr(reference: str, attribute: str) -> None:
    """Refuse clearing an attribute the component cannot live without.

    An empty value on set_component_attribute means "remove the SYMATTR line"
    (LTspice's format has no empty-value representation — a 2-token SYMATTR
    line is unreadable on the next parse). InstName is the instance's
    identity and must never be removed.
    """
    if attribute == "InstName":
        raise NetlistError(
            f"InstName on {reference!r} cannot be cleared — it is the "
            "component's identity. Use remove_component to delete the "
            "instance, or set a new non-empty name."
        )


def _reject_unknown_attr(attribute: str) -> None:
    """Refuse a SYMATTR name outside LTspice's fixed slot set.

    Any name outside the known slots is dropped at netlist-export time, so a
    typo (``Val`` for ``Value``) would silently no-op. Reject up-front with a
    'did you mean' hint. Shared by set_component_attribute (handler + op) and
    the add_component create path so all three refuse identically.
    """
    if attribute in _LTSPICE_ATTR_NAMES:
        return
    canonical = _LTSPICE_ATTR_CANONICAL.get(attribute.lower())
    if canonical:
        raise NetlistError(
            f"Unknown attribute {attribute!r}. Did you mean {canonical!r}? "
            "LTspice attribute names are case-sensitive."
        )
    if attribute.lower() == "prefix":
        # A SpiceLine referral here misdirects: Prefix is a SYMBOL property,
        # not an instance attribute, and SpiceLine can't change it.
        raise NetlistError(
            "Attribute 'Prefix' is a symbol property, not an instance "
            "attribute — the element prefix comes from the symbol's .asy "
            "file ('SYMATTR Prefix ...'). To place a subcircuit (X-prefix) "
            "device, use the edit_schematic add_component op with a symbol whose .asy declares "
            "'SYMATTR Prefix X', then bind the subckt name via Value or "
            "SpiceModel."
        )
    raise NetlistError(
        f"Unknown attribute {attribute!r}. LTspice silently ignores "
        f"unrecognised SYMATTR keys at netlist time. Valid attributes: "
        f"{', '.join(sorted(_LTSPICE_ATTR_NAMES))}. For arbitrary KEY=val "
        "pairs, set them through 'SpiceLine' instead."
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
    comp = _SchematicComponentClass(editor, "")
    comp.reference = reference
    comp.symbol = symbol  # pyright: ignore[reportAttributeAccessIssue]
    comp.position = Point(x, y)
    comp.rotation = rotation
    if value is not None:
        # An empty Value writes the same 2-token SYMATTR line that corrupts the
        # .asc on the next read, so guard this path too (not just attributes).
        _reject_empty_attr_value(reference, "Value", value)
        comp.attributes["Value"] = value
    if attributes:
        for attr_name, attr_val in attributes.items():
            _reject_unknown_attr(attr_name)
            _reject_empty_attr_value(reference, attr_name, attr_val)
            comp.attributes[attr_name] = attr_val
    editor.add_component(comp)


# Per-file locks to prevent concurrent edits to the same circuit file.
# Bounded to avoid unbounded growth; only evicts *unheld* locks.
_MAX_EDIT_LOCKS = 64
_edit_locks: dict[Path, asyncio.Lock] = {}


def _get_edit_lock(path: Path) -> asyncio.Lock:
    """Get or create a per-file edit lock (shared LRU mechanism: ``path_lock``)."""
    return path_lock(_edit_locks, path, _MAX_EDIT_LOCKS)


@asynccontextmanager
async def _edit_guard(path: Path) -> AsyncIterator[None]:
    """Serialize a mutation of one circuit file, in-process and cross-process.

    Layering: the per-path asyncio lock first (tasks in this session), then
    the cross-process file lock (parallel server sessions in the same
    directory). Every mutation is a whole-file read-modify-write, so an
    unserialized concurrent edit is last-writer-wins; this guard plus the
    editor cache's stat-on-fetch — which must happen INSIDE the guard —
    turn that into edit-on-latest.
    """
    async with _get_edit_lock(path), circuit_file_lock(path):
        yield


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

# A GUI opamp complexity-level label (e.g. ``Level.2``) that belongs in the
# SpiceModel/level selector, NOT the Value slot. On a subcircuit (X) symbol
# LTspice emits the Value as an extra positional token on the instance line,
# so netlisting fails downstream with "sub-circuit name is not defined".
_LEVEL_LABEL_RE = re.compile(r"^\s*level\.\d+\s*$", re.IGNORECASE)


def _validate_component_value(reference: str, value: str) -> None:
    """Reject values that would corrupt the netlist line on write.

    spicelib writes the value verbatim into the component line; spaces in
    a non-parameterised, non-quoted value bleed into a phantom node and
    irrecoverably break the netlist. The check is permissive of:
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


def _asc_component_value(editor, reference: str) -> str | None:
    """Current Value of an .asc component, or ``None`` if it has no Value slot.

    ``AscEditor.get_component_value`` raises for a component added without a
    Value (e.g. ``add_component`` with no ``value=``); callers distinguish that
    from a missing component via ``editor.components`` membership.
    """
    try:
        return str(editor.get_component_value(reference))
    except Exception:
        return None


def _level_label_lint(editor, reference: str, value: str) -> str | None:
    """Warn when a subcircuit (X) symbol's Value slot will corrupt the netlist.

    Two shapes, both ending in "sub-circuit name is not defined" at netlist
    time because LTspice appends the Value as a stray positional token on the X
    instance line:

    - a GUI opamp complexity label (``Level.2``) written to Value; or
    - any Value set on a symbol whose model is selected via the ``SpiceModel``
      attribute (e.g. UniversalOpamp2), whose Value slot must stay empty.

    The subcircuit gate is an OR of signals — the InstName may be ``U1`` while
    the netlist prefix is X (from the .asy Prefix), so the X-prefix test alone
    is not enough. A symbol that carries its subckt name IN Value (the common
    library part, no SpiceModel) is left alone. Returns the warning text, or
    ``None`` when neither shape applies.
    """
    if not value.strip():
        return None
    comp = editor.components.get(reference)
    attrs = getattr(comp, "attributes", None) or {}
    has_spicemodel = "SpiceModel" in attrs
    is_subckt = reference[:1].upper() == "X" or has_spicemodel or "_SUBCKT" in attrs
    if not is_subckt:
        return None
    if _LEVEL_LABEL_RE.match(value):
        return (
            f"{reference}: Value {value!r} is a GUI opamp complexity-level label, not a "
            "subcircuit value. LTspice emits it as an extra positional token on the X "
            "instance line, so netlisting fails with 'sub-circuit name is not defined'. "
            "Select the level via the SpiceModel attribute (or the symbol's default), "
            "not the Value slot."
        )
    if has_spicemodel:
        return (
            f"{reference}: this symbol's model is selected via its SpiceModel attribute, "
            f"so its Value slot must stay empty. LTspice emits Value {value!r} as an "
            "extra positional token on the X instance line, and netlisting fails with "
            "'sub-circuit name is not defined'. Clear the Value; pick the model through "
            "SpiceModel instead."
        )
    return None


def _set_or_create_value(editor, reference: str, value: str) -> None:
    """Set the Value slot, creating the ``SYMATTR Value`` line if it has none.

    ``set_component_value`` only updates an EXISTING Value slot; a component
    added without one (``add_component`` with no ``value=``) needs the line
    written directly via ``set_component_attribute`` — symmetric with
    ``add_component(value=)``. Callers pre-validate the value non-empty.
    """
    if _asc_component_value(editor, reference) is None:
        editor.set_component_attribute(reference, "Value", value)
    else:
        editor.set_component_value(reference, value)


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
    cases like ``M1 d g s b "NMOS_lvt" W=10u`` and
    ``R1 n1 n2 {1/(2*pi*RC)}`` route correctly.
    """
    _validate_component_value(reference, value)
    # Behavioral sources: the whole value IS an equation whose first token is
    # V=/I=/R=... — not a model name with trailing parameters. The KEY=VALUE
    # split below would route it to set_component_parameters, which the .asc
    # editor writes into SpiceLine while the stale expression stays in Value:
    # the netlisted B-line then carries two expressions ("No such node")
    # behind a success message.
    if reference[:1].upper() == "B" or "=" not in value:
        _set_or_create_value(editor, reference, value)
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
        _set_or_create_value(editor, reference, head)
    if params:
        editor.set_component_parameters(reference, **params)


def _bboxes_overlap(a: dict, b: dict) -> bool:
    """AABB overlap test between two bounding boxes with {x, y, width, height}."""
    return BBox.from_origin_size(a["x"], a["y"], a["width"], a["height"]).overlaps(
        BBox.from_origin_size(b["x"], b["y"], b["width"], b["height"])
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


def _overlap_warnings(editor: AscEditor, reference: str, bbox: dict[str, int]) -> list[str]:
    """Warn for each other component whose bounding box overlaps ``bbox``.

    Shared by add_component placement and move_component reposition — both flag
    where a just-placed or just-moved part lands on top of another.
    """
    return [
        f"Overlaps {existing['ref']} bounding box"
        for existing in _collect_component_geometry(editor)
        if existing["ref"] != reference and _bboxes_overlap(bbox, existing)
    ]


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


def _build_on_wire_predicate(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> "Callable[[tuple[int, int]], bool]":
    """Return an ``on_wire(coord)`` predicate with the same semantics as
    ``_point_on_segment`` but O(1)-amortised per query.

    The naive ``any(_point_on_segment(coord, *seg) for seg in segments)``
    scan is O(segments) per coord; calling it once per pin makes
    ``_post_op_warnings`` O(pins × segments), which becomes the dominant
    cost during a long ``add_component`` build. Bucketing
    horizontal segments by row and vertical by column collapses each query
    to the handful of segments sharing that row/column.
    """
    endpoints: set[tuple[int, int]] = set()
    horiz: dict[int, list[tuple[int, int]]] = {}
    vert: dict[int, list[tuple[int, int]]] = {}
    for (x1, y1), (x2, y2) in segments:
        endpoints.add((x1, y1))
        endpoints.add((x2, y2))
        if y1 == y2 and x1 != x2:
            horiz.setdefault(y1, []).append((min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            vert.setdefault(x1, []).append((min(y1, y2), max(y1, y2)))
        # Diagonal / zero-length segments contribute via endpoints only,
        # matching _point_on_segment's diagonal fallback.

    def on_wire(coord: tuple[int, int]) -> bool:
        if coord in endpoints:
            return True
        px, py = coord
        if any(xmin <= px <= xmax for xmin, xmax in horiz.get(py, ())):
            return True
        return any(ymin <= py <= ymax for ymin, ymax in vert.get(px, ()))

    return on_wire


class _NetPartition(NamedTuple):
    """Connected-component view of a schematic's nets.

    ``root`` maps any interest coordinate to its net's canonical
    representative; ``members`` maps a root to every coordinate on that net;
    ``pin_owners`` maps a coordinate to the ``(ref, pin_name)`` pairs sitting
    there; ``label_texts`` maps a coordinate to the FLAG texts placed there.
    """

    root: "Callable[[tuple[int, int]], tuple[int, int]]"
    members: dict[tuple[int, int], set[tuple[int, int]]]
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]]
    label_texts: dict[tuple[int, int], set[str]]


def _net_partition(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
) -> _NetPartition:
    """Union-find over pins, labels, and wires → a connected-net partition.

    Segment-aware: a label or pin lying anywhere ON a wire (not just at an
    endpoint) is unioned with that wire — endpoint-only matching misses
    FLAGs placed mid-segment.

    ``extra_segments`` lets the caller include not-yet-committed wire
    segments (e.g. the route ``wire_pins`` is about to add) so checks operate
    on the post-route net layout. Shared by ``_trace_nets`` (labels-per-net)
    and ``trace_net`` (full net membership).
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
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for entry in _collect_component_geometry(editor):
        ref = entry["ref"]
        for pin in entry["pins"]:
            coord = (pin["x"], pin["y"])
            interest_points.add(coord)
            find(coord)
            pin_owners.setdefault(coord, []).append((ref, pin["name"]))
    label_texts: dict[tuple[int, int], set[str]] = {}
    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        interest_points.add(coord)
        find(coord)
        label_texts.setdefault(coord, set()).add(lbl.text)

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

    members: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for p in parent:
        members.setdefault(find(p), set()).add(p)

    return _NetPartition(
        root=find, members=members, pin_owners=pin_owners, label_texts=label_texts
    )


def _trace_nets(
    editor: AscEditor,
    extra_segments: list[tuple[int, int, int, int]] | None = None,
) -> dict[tuple[int, int], frozenset[str]]:
    """Map each pin/label/wire coordinate to the labels on its net.

    Thin labels-per-coordinate view over :func:`_net_partition`. See it for
    the segment-aware semantics and ``extra_segments`` contract.
    """
    part = _net_partition(editor, extra_segments)
    labels_by_root: dict[tuple[int, int], set[str]] = {}
    for coord, texts in part.label_texts.items():
        labels_by_root.setdefault(part.root(coord), set()).update(texts)

    return {
        p: frozenset(labels_by_root.get(part.root(p), set()))
        for members in part.members.values()
        for p in members
    }


def _net_label_at(
    nets: dict[tuple[int, int], frozenset[str]], coord: tuple[int, int]
) -> frozenset[str]:
    """Labels on the net at ``coord``; empty when net is unnamed."""
    return nets.get(coord, frozenset())


def _named_labels(labels: frozenset[str]) -> set[str]:
    """Strip ground ('0') from a label set so 'short to ground' isn't
    flagged as a conflict by detect-multi-label net checks."""
    return {lbl for lbl in labels if lbl != "0"}


def _segment_key(
    a: tuple[int, int], b: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """One segment's identity, orientation-independent."""
    return (a, b) if a <= b else (b, a)


def _wire_segment_keys(editor: AscEditor) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [
        _segment_key((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires
    ]


def _append_wire_segments(
    editor: AscEditor,
    segments: Sequence[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Draw the planned segments, skipping any the sheet already carries.

    Returns the ones that were already there. A second copy of a segment draws
    nothing and connects nothing, but a caller cannot tell it from a real second
    wire — and asking to delete "the duplicate" deletes every copy, so the
    request that meant "tidy up" cut the connection instead. Not creating the
    duplicate is what makes that sequence impossible; the alternative, warning
    about it afterwards, is what led the caller into it.
    """
    present = set(_wire_segment_keys(editor))
    already: list[tuple[int, int, int, int]] = []
    for sx1, sy1, sx2, sy2 in segments:
        key = _segment_key((sx1, sy1), (sx2, sy2))
        if key in present:
            already.append((sx1, sy1, sx2, sy2))
            continue
        present.add(key)
        editor.wires.append(Line(Point(sx1, sy1), Point(sx2, sy2)))
    return already


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
    - ``label_over_component`` — a net label whose coordinate falls strictly
      inside a component's bounding box while sitting on no component's pin.
      Surfaces the anchor-in-box fact only: the axis-aligned box also spans
      leads and empty corners, so this is not a guarantee the rendered glyph
      overlaps the drawn symbol. A label on a pin (any component's) — the normal
      ground-flag pattern — is on a box boundary and is excluded.
    - ``stacked_directive`` — two or more directive/comment text objects at
      the exact same anchor, rendering on top of each other. Exact-coordinate
      match only (no font-metric guessing), so this never fires on a
      deliberately tight-but-offset directive block.

    Read-only on the editor. Cheap to compute during an existing edit
    session; intended for callers to surface in their response payload.
    """
    pins: list[tuple[str, str, int, int]] = []
    comp_boxes: list[tuple[str, BBox]] = []
    for entry in _collect_component_geometry(editor):
        ref = entry["ref"]
        comp_boxes.append(
            (ref, BBox.from_origin_size(entry["x"], entry["y"], entry["width"], entry["height"]))
        )
        for p in entry["pins"]:
            pins.append((ref, p["name"], p["x"], p["y"]))

    pin_count_at: dict[tuple[int, int], int] = {}
    for _, _, x, y in pins:
        pin_count_at[(x, y)] = pin_count_at.get((x, y), 0) + 1

    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    _on_any_wire = _build_on_wire_predicate(segments)

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

    for lbl in editor.labels:
        coord = (int(lbl.coord.X), int(lbl.coord.Y))
        # A label on ANY component's pin is the normal flag pattern (pins sit on
        # symbol outlines) — never report it, even when it also lands inside a
        # different, overlapping component's box.
        if coord in pin_count_at:
            continue
        for ref, box in comp_boxes:
            # Strict interior only: a coordinate on the box boundary — where pins
            # and leads sit — is not "inside". No break: with overlapping boxes a
            # label can be inside more than one, and each is a distinct fact.
            if box.x1 < coord[0] < box.x2 and box.y1 < coord[1] < box.y2:
                warnings.append(
                    {
                        "kind": "label_over_component",
                        "label": lbl.text,
                        "ref": ref,
                        "x": coord[0],
                        "y": coord[1],
                        "message": (
                            f"Label '{lbl.text}' at ({coord[0]},{coord[1]}) is inside "
                            f"{ref}'s bounding box"
                        ),
                    }
                )

    directive_anchor_count: dict[tuple[int, int], int] = {}
    for d in editor.directives:
        anchor = (int(d.coord.X), int(d.coord.Y))
        directive_anchor_count[anchor] = directive_anchor_count.get(anchor, 0) + 1
    for (dx, dy), count in directive_anchor_count.items():
        if count > 1:
            warnings.append(
                {
                    "kind": "stacked_directive",
                    "x": dx,
                    "y": dy,
                    "count": count,
                    "message": (
                        f"{count} directives/comments share anchor ({dx},{dy}) — "
                        "they render on top of each other"
                    ),
                }
            )

    return warnings


def _wiring_profile(editor: AscEditor) -> dict[str, int]:
    """Neutral whole-schematic connectivity counts: are connections drawn as
    wires or carried by net-labels?

    Lets a caller tell a routed schematic from net-label soup (every pin
    tagged with a flag, no wires drawn — simulates identically but reads as a
    wiring list, not a schematic). Facts only, no verdict (see
    ``lib/result_observations.py``): a high ``pins_label_only`` with
    ``wire_segments`` near zero is the soup signature, but net-labels are also
    the right tool for ground/power/genuinely distant nets — the model judges,
    this only counts. A pin counts as wired if a wire passes through it, else
    label-only if a flag sits on it; pins that are floating or directly
    abutting another pin fall into neither. ``pins_total`` is every pin over
    components with resolvable symbol geometry (unknown symbols are skipped) —
    the denominator that makes the two classified counts interpretable.
    """
    segments = [((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) for w in editor.wires]
    on_wire = _build_on_wire_predicate(segments)
    label_coords = {(int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels}

    pins_total = 0
    pins_wired = 0
    pins_label_only = 0
    for entry in _collect_component_geometry(editor):
        for p in entry["pins"]:
            pins_total += 1
            coord = (p["x"], p["y"])
            if on_wire(coord):
                pins_wired += 1
            elif coord in label_coords:
                pins_label_only += 1

    return {
        "wire_segments": len(segments),
        "pins_total": pins_total,
        "pins_wired": pins_wired,
        "pins_label_only": pins_label_only,
    }


# Type alias for the union returned by _make_editor / _get_editor.
# Schematic-only handlers narrow this to AscEditor after _require_asc.
Editor = AscEditor | SpiceEditor


class WaypointInput(StrictModel):
    x: int = Field(description="X coordinate of waypoint")
    y: int = Field(description="Y coordinate of waypoint")


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


# ---------------------------------------------------------------------------
# Handlers — shared operations (work on .cir/.net and .asc)
# ---------------------------------------------------------------------------


def netlist_card_value(card: SpiceCard) -> str:
    """Display value for one netlist instance card, or ``"<unparseable>"``.

    Rejects a card whose body survived lexing but carries a broken ``k=v``
    remnant; otherwise the typed ``InstanceLine`` view yields the element-class
    display value. Shared by the netlist ``list_components`` path and the
    ``inspect`` component queries so the two agree on what a value is.
    """
    from ltspice_mcp.lib.spice_lex_views import InstanceLine, body_has_stray_kv_remnant

    if body_has_stray_kv_remnant(card.body):
        return "<unparseable>"
    try:
        return InstanceLine.from_card(card).display_value()
    except Exception:
        return "<unparseable>"


_ASC_TEXT_DEFAULT_SIZE = 2
"""LTspice's normal font size — the fallback when a caller doesn't pick one."""

_ASC_TEXT_LINE_PITCH = 16
"""One grid cell — the downward step used to declutter exactly-stacked text.

Multiple directives added without explicit coordinates all default to the same
anchor (16,16) and render on top of each other. Nudging each new one down by a
grid cell until its anchor is free keeps them readable. Only exact-anchor
collisions move — no font-metric guesswork, so no false shifts."""


def _append_asc_text(
    editor: AscEditor,
    text: str,
    text_type: TextTypeEnum,
    x: int | None,
    y: int | None,
    size: int | None,
    *,
    default_x: int,
    default_y: int,
) -> None:
    """Append one TEXT record (directive or comment) to an ``.asc``.

    The single producer of placed schematic text — ``edit_directive``'s two
    kinds and ``edit_schematic``'s ``add_directive`` op all come through
    here, so placement defaulting can't drift between them. The per-site
    ``default_x``/``default_y`` differ deliberately (comments default to the
    sheet origin; directives to (16,16)).

    Auto-declutter: if the resolved anchor is already occupied by another text
    object (the common case when several directives take the (16,16) default),
    step it down one grid cell at a time until it's free, so stacked directives
    don't render on top of each other.
    """
    # LTspice's on-disk TEXT record is one physical line; embedded newlines
    # are stored as literal "\n" escapes (LTspice's own multi-line text
    # convention). A raw newline would split the record and corrupt the file
    # — the downstream symptom is an unrelated-looking "Primitive not
    # supported" parse error.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")
    px = x if x is not None else default_x
    py = y if y is not None else default_y
    occupied = {(int(d.coord.X), int(d.coord.Y)) for d in editor.directives}
    while (px, py) in occupied:
        py += _ASC_TEXT_LINE_PITCH
    editor.directives.append(
        Text(
            coord=Point(px, py),
            text=text,
            type=text_type,
            size=size if size is not None else _ASC_TEXT_DEFAULT_SIZE,
        )
    )


def _remove_exact_directive(editor, instruction: str) -> bool:
    """Remove one DIRECTIVE-type record matching ``instruction`` by FULL-TEXT
    equality (not substring), returning whether one was removed.

    spicelib's ``remove_instruction`` matches by substring and removes the first
    hit, so an undo of ``.tran 1m`` could instead delete ``.tran 10m`` (or a
    pre-existing duplicate) and silently corrupt the simulation setup. On an
    AscEditor (``directives`` is a list of typed Text records) we filter by exact
    text and drop a single record. Netlist editors (SpiceEditor — no
    ``directives`` list) keep spicelib's substring matcher; the ``.cir``/``.net``
    edit path is out of scope here.
    """
    directives = getattr(editor, "directives", None)
    if directives is None:
        return bool(editor.remove_instruction(instruction))
    for idx, d in enumerate(directives):
        dtype = getattr(d, "type", None)
        dtext = getattr(d, "text", None)
        if dtype == TextTypeEnum.DIRECTIVE and dtext == instruction:
            del directives[idx]
            return True
    return False


def _remove_directive_or_comment(editor, instruction: str) -> str:
    """Remove a directive or comment matching ``instruction``.

    The default path is an exact, literal match: a DIRECTIVE is removed only when
    its full text equals ``instruction`` (see ``_remove_exact_directive``), and a
    comment only when its body equals it. ``instruction`` is never treated as a
    regex by default — common SPICE directives contain ``(`` and ``)`` (every
    ``.meas``/``.four``/``.print`` referencing ``V(node)``) which would silently
    turn into regex capture groups under any "metachar means regex" heuristic.
    Pass an explicit ``regex:`` prefix to opt in to regex matching.

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
                "Use inspect(kind='components') to see what's actually in the file."
            )
        return "directive(s)/comment(s)"

    directive_hit = _remove_exact_directive(editor, instruction)
    comment_hit = _strip_matching_comments(editor, instruction)
    if not (directive_hit or comment_hit):
        raise NetlistError(
            f"No directive or comment matched {instruction!r} exactly. "
            "Match is literal by default — pass 'regex:<pattern>' for regex "
            "matching, or copy the line verbatim from inspect(kind='components')."
        )
    return "directive"


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


def _orphaned_wire_coords(editor: AscEditor, target_only_pins: set[tuple[int, int]]) -> list[str]:
    """``(x,y)`` strings for wires still ending on a removed/moved component's
    former pins (its pins minus other components' pins). Shared by the standalone
    handlers and the edit_schematic ops; call AFTER the mutation."""
    orphaned: list[str] = []
    for w in editor.wires:
        for coord in ((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))):
            label = f"({coord[0]},{coord[1]})"
            if coord in target_only_pins and label not in orphaned:
                orphaned.append(label)
    return orphaned


def _drop_wires_at(editor: AscEditor, coords: set[tuple[int, int]]) -> int:
    """Remove every wire with an endpoint in ``coords``; return how many were
    dropped. Shared by remove_component's wire cleanup and the remove_wire op."""
    if not coords:
        return 0
    kept = [
        w
        for w in editor.wires
        if (int(w.V1.X), int(w.V1.Y)) not in coords and (int(w.V2.X), int(w.V2.Y)) not in coords
    ]
    dropped = len(editor.wires) - len(kept)
    editor.wires = kept
    return dropped


def _move_component_warnings(
    editor: AscEditor,
    reference: str,
    rot_name: str,
    x: int,
    y: int,
    old_pin_coords: set[tuple[int, int]],
    other_pins: set[tuple[int, int]],
) -> list[str]:
    """Bounding-box-overlap + orphaned-wire warnings for a just-moved component.

    Shared by the standalone handler and the apply_schematic_ops move op so both
    surface the same facts. ``old_pin_coords`` and ``other_pins`` must be
    captured BEFORE the move; call this AFTER ``set_component_position``.
    """
    warnings: list[str] = []
    new_pin_coords = _component_pin_coords(editor, reference)

    comp = editor.components[reference]
    moved_bb: dict[str, int] | None = None
    if comp.symbol:
        moved_sym = get_symbol_info(comp.symbol)
        if moved_sym is not None:
            moved_bb = compute_placed_geometry(moved_sym, x, y, rot_name)["bounding_box"]
    if moved_bb is not None:
        warnings.extend(_overlap_warnings(editor, reference, moved_bb))

    # Pins that the move abandoned (no longer this component's, not a neighbour's)
    # but that still have a wire ending on them are orphaned by the move.
    abandoned_pins = old_pin_coords - new_pin_coords - other_pins
    orphaned = _orphaned_wire_coords(editor, abandoned_pins) if abandoned_pins else []
    if orphaned:
        warnings.append(
            f"wires left at old pin coordinates with no connection: {', '.join(orphaned)}. "
            "Re-route or delete these wires."
        )
    return warnings


def _placed_component_data(
    editor: AscEditor,
    reference: str,
    symbol: str,
    x: int,
    y: int,
    rotation: str,
    symbol_info: SymbolInfo,
) -> dict[str, object]:
    """Return the placed symbol geometry and any component-overlap warnings."""
    geometry = compute_placed_geometry(symbol_info, x, y, rotation)
    bounding_box = geometry["bounding_box"]
    warnings = _overlap_warnings(editor, reference, bounding_box)
    return {
        "reference": reference,
        "symbol": symbol,
        "position": {"x": x, "y": y},
        "rotation": rotation,
        "pins": geometry["pins"],
        "bounding_box": bounding_box,
        "warnings": warnings,
    }


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
                f"Net label '{net_name}' not found in schematic. Add it with the "
                "add_net_label op of edit_schematic first."
            )
        if len(matches) > 1:
            coords = ", ".join(f"({x},{y})" for x, y in matches)
            raise NetlistError(
                f"Multiple '{net_name}' labels found at: {coords}. "
                "Connect to a component pin (Ref.Pin) instead, or place the label "
                "at a specific pin with the add_net_label op of edit_schematic "
                f"(net='{net_name}', pin='<Ref.Pin>')."
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


def _add_net_label_checks(editor: AscEditor, net: str, x: int, y: int) -> list[str]:
    """Validate placing net label ``net`` at ``(x, y)``; shared by the standalone
    handler and the ``apply_schematic_ops`` add_net_label op so both enforce the
    same rules from either entry point.

    Raises ``NetlistError`` if a non-ground label would merge two different named
    nets (a short at netlist time) — a structural error, refused outright.
    Returns advisory warnings (duplicate name, floating placement) as plain
    facts for the caller to surface; placing labels before wiring them is a
    legitimate workflow, so those are warnings, not refusals.
    """
    warnings: list[str] = []
    if net != "0":
        # Duplicate non-ground label name. This is NOT a short: the netlist merges
        # same-name labels into one net, which is a valid (often simpler) way to
        # tie distant nets. The only downstream cost is that wire_pins(net=...) can't
        # disambiguate which label to route to — so surface that, not a scare.
        for lbl in editor.labels:
            if lbl.text == net:
                warnings.append(
                    f"'{net}' already labels a net at ({int(lbl.coord.X)},"
                    f"{int(lbl.coord.Y)}); the netlist merges the two into one net "
                    "(this is correct — a valid way to tie distant nets). Only a "
                    f"later wire_pins(net='{net}') is ambiguous with duplicate labels — "
                    "connect to a component pin (Ref.Pin) instead."
                )
                break
        # Net-label conflict: a non-ground label on a network that already
        # carries a different named net shorts the two at netlist time. Refuse.
        nets = _trace_nets(editor)
        other_labels = {n for n in _net_label_at(nets, (x, y)) if n != net and n != "0"}
        if other_labels:
            raise NetlistError(
                f"Refused to add net '{net}' at ({x},{y}): the wire network at this "
                f"coordinate already carries the label(s) {sorted(other_labels)}. Adding "
                f"'{net}' would short those nets together. Remove the existing label(s) "
                f"first or pick a different coordinate."
            )
    # Floating label: a FLAG at a coordinate with no wire endpoint and no
    # component pin is silently ignored by LTspice at netlist time.
    wire_endpoints = {(int(w.V1.X), int(w.V1.Y)) for w in editor.wires} | {
        (int(w.V2.X), int(w.V2.Y)) for w in editor.wires
    }
    all_pin_coords: set[tuple[int, int]] = set()
    for ref in editor.get_components():
        all_pin_coords.update(_component_pin_coords(editor, ref))
    if (x, y) not in wire_endpoints and (x, y) not in all_pin_coords:
        warnings.append(
            f"({x},{y}) has no wire endpoint or component pin — LTspice will ignore "
            "this floating label until you wire it up."
        )
    return warnings


class _ConnectPlan(NamedTuple):
    """Validated connect route ready to commit to the editor."""

    x1: int
    y1: int
    x2: int
    y2: int
    points: list[tuple[int, int]]
    segments: list[tuple[int, int, int, int]]
    warnings: list[str]


def _merge_collinear_runs(
    segments: list[tuple[int, int, int, int]],
    node_coords: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Collapse straight runs of collinear wire segments into single segments,
    mirroring LTspice's netlist-time wire merge.

    A vertex breaks a run — stays its own node — when it is a pin coordinate
    (``node_coords``) or a corner/junction (its incident segment ends are not
    exactly two ends of one orientation). Only pure pass-through vertices
    (degree-2, both ends collinear, not a pin) are merged across. This is what
    makes a *collinear* waypoint disappear: an in-line bend leaves only bare
    pass-through vertices, so the run collapses back to one segment; a bend that
    turns a corner leaves the corner vertices as breaks, so its segments stay
    split. Verticals (``x1==x2``) and horizontals (``y1==y2``) are merged
    per-line by interval union split at breaks; any diagonal passes through
    unchanged.
    """
    ends: dict[tuple[int, int], list[str]] = defaultdict(list)
    verticals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    horizontals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    merged: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in segments:
        if (x1, y1) == (x2, y2):
            continue  # zero-length record (hand-corrupted WIRE) — nothing to merge
        if x1 == x2:
            verticals[x1].append((min(y1, y2), max(y1, y2)))
            ends[(x1, y1)].append("V")
            ends[(x2, y2)].append("V")
        elif y1 == y2:
            horizontals[y1].append((min(x1, x2), max(x1, x2)))
            ends[(x1, y1)].append("H")
            ends[(x2, y2)].append("H")
        else:
            merged.append((x1, y1, x2, y2))  # diagonal — passed through as-is

    def _breaks(coord: tuple[int, int]) -> bool:
        es = ends.get(coord, [])
        return coord in node_coords or len(es) != 2 or len(set(es)) != 1

    def _emit(intervals: list[tuple[int, int]], cuts: set[int], vertical: bool, line: int) -> None:
        intervals.sort()
        runs: list[list[int]] = []
        for lo, hi in intervals:
            if runs and lo <= runs[-1][1]:
                runs[-1][1] = max(runs[-1][1], hi)
            else:
                runs.append([lo, hi])
        for lo, hi in runs:
            pts = sorted({lo, hi} | {c for c in cuts if lo < c < hi})
            for a, b in itertools.pairwise(pts):
                merged.append((line, a, line, b) if vertical else (a, line, b, line))

    # A component pin on the interior of a run is a junction too — LTspice splits
    # the wire there — but it is not a wire endpoint, so it never appears in
    # ``ends`` and a break test over ``ends`` alone would miss it. Add every pin
    # sitting on the line as a cut candidate; ``_emit`` keeps only those strictly
    # inside a run's span, so a pin at a run end (already a natural node) or off
    # any run adds nothing.
    for x, iv in verticals.items():
        cuts = {y for (cx, y) in ends if cx == x and _breaks((cx, y))}
        cuts |= {py for (px, py) in node_coords if px == x}
        _emit(iv, cuts, vertical=True, line=x)
    for y, iv in horizontals.items():
        cuts = {x for (x, cy) in ends if cy == y and _breaks((x, cy))}
        cuts |= {px for (px, py) in node_coords if py == y}
        _emit(iv, cuts, vertical=False, line=y)
    return merged


def _pin_owners(
    component_geo: list[dict],
) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """Map each pin coordinate to its ``(ref, pin_name)`` owners.

    The same shape :func:`_net_partition` builds (available there as
    ``part.pin_owners``); use this where only the pin-owner view is needed and
    no full net partition is on hand. ``component_geo`` is
    :func:`_collect_component_geometry` output.
    """
    owners: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for cg in component_geo:
        for pin in cg["pins"]:
            owners.setdefault((pin["x"], pin["y"]), []).append((cg["ref"], pin["name"]))
    return owners


def _wire_segments(editor: AscEditor) -> list[tuple[int, int, int, int]]:
    """Every wire as a flat ``(x1, y1, x2, y2)`` integer tuple."""
    return [(int(w.V1.X), int(w.V1.Y), int(w.V2.X), int(w.V2.Y)) for w in editor.wires]


def _same_instance_dropped_segments(
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]],
    segments: list[tuple[int, int, int, int]],
) -> list[dict]:
    """Wire segments LTspice discards from the exported netlist.

    LTspice drops a wire run whose two ends both land exactly on pins of the
    SAME single component instance (verified against LTspice 26 ``-netlist``:
    such a run never reaches the netlist, so the two pins stay on separate nodes
    and the drawn tie has no electrical effect). Two routes still get kept, and
    both were confirmed against LTspice 26: a run spanning two *different*
    instances, and a same-instance tie that turns a corner OUT OF LINE with the
    two pins. A waypoint that stays *collinear* with the pins does NOT survive —
    LTspice merges the in-line segments back into one and drops it — so the
    segments are collinear-merged (:func:`_merge_collinear_runs`) before this
    check, which is what catches an all-in-line waypoint route as well as the
    bare direct wire. A net label on one pin does not rescue the run either.

    Returns one dict per dropped run with ``segment`` (the merged ``(x1, y1, x2,
    y2)`` tuple), ``ref`` (the shared instance), and ``pins`` (the two pin
    names on that instance), ordered deterministically. ``pin_owners`` maps each
    pin coordinate to its ``(ref, pin_name)`` owners (:func:`_pin_owners` and
    :func:`_net_partition` both build this shape).
    """
    dropped: list[dict] = []
    for seg in _merge_collinear_runs(list(segments), set(pin_owners)):
        sx1, sy1, sx2, sy2 = seg
        if (sx1, sy1) == (sx2, sy2):
            continue  # defensive: a collapsed/zero-length run is not a tie
        owners_a = pin_owners.get((sx1, sy1), [])
        owners_b = pin_owners.get((sx2, sy2), [])
        if not owners_a or not owners_b:
            continue  # an end is a bare vertex/waypoint, not a pin — kept
        refs_a = {r for r, _ in owners_a}
        refs_b = {r for r, _ in owners_b}
        if len(refs_a | refs_b) != 1:
            continue  # the run bridges two distinct instances — kept
        ref = next(iter(refs_a))
        pin_a, pin_b = owners_a[0][1], owners_b[0][1]
        dropped.append({"segment": seg, "ref": ref, "pins": (pin_a, pin_b)})
    dropped.sort(key=lambda d: (d["ref"], d["segment"]))
    return dropped


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

    Shared by ``handle_wire_pins`` and the ``wire_pins`` op of
    ``apply_schematic_ops`` (``connect`` accepted as a deprecated alias for
    both) so both paths apply identical safety checks.
    """
    component_geo = _collect_component_geometry(editor)
    existing_wires = _wire_segments(editor)

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
                f"use the edit_schematic add_net_label op to merge them "
                f"deliberately."
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

    # Same-instance self-loop — refused first because, like the net-label
    # conflict above, it's a "wrong intent" error: a wire tying two pins of one
    # component (directly, or through a collinear waypoint that merges back into
    # a straight wire) is dropped by LTspice at netlist time (see
    # _same_instance_dropped_segments), so reporting the tie as connected would
    # be a lie. The fix is a route change, so surface it before route geometry.
    dropped = _same_instance_dropped_segments(_pin_owners(component_geo), segments)
    if dropped:
        ref = dropped[0]["ref"]
        pin_a, pin_b = dropped[0]["pins"]
        raise NetlistError(
            f"Refused to connect {from_pin} to {to_pin}: this is a same-instance "
            f"wire — it ties two pins of the same component {ref} ({ref}.{pin_a} "
            f"and {ref}.{pin_b}). LTspice drops such a wire from the exported "
            f"netlist, so the tie would have no electrical effect. To tie two pins "
            f"of one component: route the wire so it bends OUT OF LINE with the two "
            f"pins (a waypoint that stays collinear with them is merged back into a "
            f"straight wire and still dropped; the bend must leave that line), or "
            f"drop the wire and give both pins the same net label via the "
            f"edit_schematic add_net_label op (same-name labels merge into one "
            f"net)."
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

    # Pin-collision exemption is by exact endpoint *coordinate* (the
    # ``(px, py) in endpoints`` check below), NOT by whole component: the
    # OTHER pin of an endpoint component still lies on the route and must be
    # flagged — otherwise a waypoint landing on it silently shorts the
    # component while wire_pins reports success. ``skip_refs`` stays in the
    # bbox-crossing *warning* loop, where exempting an endpoint component is
    # reasonable.
    for cg in component_geo:
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
                    # A pin at the shared corner of two consecutive segments
                    # satisfies _point_on_segment for both — report it once.
                    break

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


def blank_sheet(width: int = 880, height: int = 680) -> str:
    """The .asc body of an empty sheet (880x680 = LTspice's default extent)."""
    return f"Version 4\nSHEET 1 {width} {height}\n"


class TraceNetInput(ToolInput):
    path: str = Field(description="Path to an .asc schematic")
    pin: str | None = Field(
        default=None,
        description=(
            "Pin or net reference to start from: 'Ref.Pin' (e.g. 'M1.D'), "
            "'net:NAME' (e.g. 'net:VDD'), or omit and pass x/y."
        ),
    )
    x: int | None = Field(default=None, description="X coordinate (with y) to trace from")
    y: int | None = Field(default=None, description="Y coordinate (with x) to trace from")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "start": {
                "type": "object",
                "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            },
            "labels": {"type": "array", "items": {"type": "string"}},
            "pins": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string"},
                        "pin": {"type": "string"},
                        "x": {"type": "integer"},
                        "y": {"type": "integer"},
                    },
                },
            },
            "coordinates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
                },
            },
            "is_shorted": {"type": "boolean"},
            "warnings": WARNINGS_SCHEMA,
        },
    }
)
async def handle_trace_net(args: TraceNetInput, state: SessionState) -> types.CallToolResult:
    """Trace every pin/label/wire vertex on the net at a pin, label, or (x,y)."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)
    editor = _get_asc_editor(asc_path, state)

    if args.pin is not None and args.pin.startswith("net:"):
        # A net: reference legitimately matches many same-name FLAGs — the
        # normal case on label-wired schematics (one FLAG per pin).
        # _resolve_pin refuses ambiguous net labels, but trace_net's own
        # name-merge step below absorbs duplicates, so just seed from any
        # matching label coordinate (lowest, for determinism).
        net_name = args.pin[4:]
        matches = sorted(
            (int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels if lbl.text == net_name
        )
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. Add it with the "
                "add_net_label op of edit_schematic first, or trace a "
                "component pin / coordinate."
            )
        x, y = matches[0]
    elif args.pin is not None:
        x, y = _resolve_pin(args.pin, editor)
    elif args.x is not None and args.y is not None:
        x, y = args.x, args.y
    else:
        raise NetlistError("trace_net needs either 'pin' or both 'x' and 'y'.")

    part = _net_partition(editor)
    start = (x, y)
    physical_members = part.members.get(part.root(start), set())
    if start not in physical_members and start not in part.pin_owners:
        # The coordinate isn't on any pin/label/wire endpoint — an empty point
        # (or a bare mid-wire span carrying nothing).
        raise NetlistError(
            f"Nothing found at ({x},{y}): no component pin, net label, or wire "
            "vertex sits there. Use inspect(kind='components') to inspect the layout."
        )

    # The physical partition connects by wire only; LTspice also makes FLAGs
    # with the same NAME electrically common. Fold physical nets that share a
    # label name together (a second union-find over physical roots) so
    # trace_net answers "what's on net X" on label-wired schematics,
    # not just wire-routed ones.
    root_parent: dict[tuple[int, int], tuple[int, int]] = {}

    def _rfind(r: tuple[int, int]) -> tuple[int, int]:
        root_parent.setdefault(r, r)
        while root_parent[r] != r:
            root_parent[r] = root_parent[root_parent[r]]
            r = root_parent[r]
        return r

    label_first: dict[str, tuple[int, int]] = {}
    for root, coords in part.members.items():
        for coord in coords:
            for lbl in part.label_texts.get(coord, ()):
                if lbl in label_first:
                    ra, rb = _rfind(label_first[lbl]), _rfind(root)
                    if ra != rb:
                        root_parent[ra] = rb
                else:
                    label_first[lbl] = root

    target_root = _rfind(part.root(start))
    member_coords: set[tuple[int, int]] = set()
    for root, coords in part.members.items():
        if _rfind(root) == target_root:
            member_coords |= coords
    if not member_coords:
        member_coords = {start}

    labels: set[str] = set()
    pins: list[dict] = []
    for coord in member_coords:
        labels.update(part.label_texts.get(coord, set()))
        for ref, pin_name in part.pin_owners.get(coord, []):
            pins.append({"reference": ref, "pin": pin_name, "x": coord[0], "y": coord[1]})

    named = sorted(_named_labels(frozenset(labels)))
    is_shorted = len(named) > 1
    pins.sort(key=lambda p: (p["reference"], p["pin"]))
    coords = sorted(member_coords)

    # LTspice drops a wire segment joining two pins of one component instance,
    # so a net whose shape here depends on such a segment over-reports what
    # LTspice will actually netlist. Surface each dropped segment on this net as
    # a fact (not a verdict) — the model decides whether the tie was intended.
    net_segments = [
        s
        for s in _wire_segments(editor)
        if (s[0], s[1]) in member_coords and (s[2], s[3]) in member_coords
    ]
    warnings = [
        f"{d['ref']}.{d['pins'][0]} and {d['ref']}.{d['pins'][1]} are joined by a "
        f"wire between two pins of the same component; LTspice drops that wire from "
        f"the netlist, so this connection may not exist in simulation. Reroute the "
        f"wire to bend out of line with the two pins, or label both pins with the "
        f"same net name."
        for d in _same_instance_dropped_segments(part.pin_owners, net_segments)
    ]

    data: dict = {
        "start": {"x": x, "y": y},
        "labels": sorted(labels),
        "pins": pins,
        "coordinates": [{"x": cx, "y": cy} for cx, cy in coords],
        "is_shorted": is_shorted,
    }
    if warnings:
        data["warnings"] = warnings

    net_name = ", ".join(sorted(labels)) if labels else "<unnamed>"
    lines = [f"Net at ({x},{y}): {net_name}"]
    if pins:
        lines.append("  Pins:")
        for p in pins:
            lines.append(f"    {p['reference']}.{p['pin']} at ({p['x']},{p['y']})")
    else:
        lines.append("  (no component pins on this net)")
    if is_shorted:
        lines.append(f"  WARNING: net carries multiple labels {named} — likely a short.")
    for w in warnings:
        lines.append(f"  WARNING: {w}")
    return format_response("\n".join(lines), data, args.format)


def _norm_micro(s: str) -> str:
    """Map both micro codepoints (µ U+00B5, μ U+03BC) to ASCII 'u' so a value
    LTspice renders with the micro sign compares equal to the same value
    authored as 'u' (e.g. 1µ vs 1u). Used ONLY for diff equality, never on the
    displayed strings — a real magnitude change like 1u vs 2u still differs."""
    return s.replace("µ", "u").replace("μ", "u")


def _component_signature(comp: dict) -> str:
    """Comparable string for a component: its Value plus any extra SYMATTR
    attributes (Value2/SpiceLine/SpiceModel). ``set_component_attribute`` edits
    land in these attributes and change the exported netlist, so diff_circuit
    must compare them too — otherwise such an edit reads as 'no differences'."""
    value = str(comp["value"])
    attrs = comp.get("attributes") or {}
    if not attrs:
        return value
    attr_str = "; ".join(f"{k}={attrs[k]}" for k in sorted(attrs))
    return f"{value} | {attr_str}"


def _components_and_directives(path: Path) -> tuple[dict[str, str], set[str], str | None]:
    """Return (components, directive_lines, parse_error) for a circuit file.

    Reuses ``services.extract_{asc,netlist}_info`` so unparseable component
    values, AscEditor dispatch, and directive collection all flow through the
    canonical path. No second disk read. ``parse_error`` is None on success,
    or a short message when the file could not be parsed — so the diff can
    flag an unreadable file rather than treat it as an empty circuit (which
    would report every component of the other file as a removal).
    """
    if _is_asc(path):
        try:
            ed = _make_editor(path)
        except Exception as e:
            return {}, set(), f"{path.name} could not be parsed ({e})"
        assert isinstance(ed, AscEditor)
        info = services.extract_asc_info(ed, path)
        components = {comp["reference"]: _component_signature(comp) for comp in info["components"]}
        directives = {d.strip() for d in info.get("directives", []) if d.strip().startswith(".")}
        return components, directives, None
    try:
        info = services.extract_netlist_info(path)
    except Exception as e:
        return {}, set(), f"{path.name} could not be parsed ({e})"
    components = {comp["reference"]: _component_signature(comp) for comp in info["components"]}
    directives = {
        line.strip()
        for line in info.get("content", "").splitlines()
        if line.strip().startswith(".")
    }
    return components, directives, None


# The added/removed/changed delta both structural comparisons in this codebase
# produce from ``_components_and_directives``: diff_circuit's own payload and
# verify_circuit's structural_diff / sidecar-export diff. One payload, one schema
# — declared here, beside the function whose output it describes.
#
# "baseline" is the first deck given (diff_circuit's ``path_a``, verify's
# reference); "compared" is the second (``path_b``, the circuit under test).
STRUCTURAL_DELTA_PROPS: dict[str, Any] = {
    "components_added": {
        "type": "array",
        "items": {"type": "string"},
        "description": "References present in the compared deck but absent from the baseline.",
    },
    "components_removed": {
        "type": "array",
        "items": {"type": "string"},
        "description": "References present in the baseline but absent from the compared deck.",
    },
    "components_changed": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "reference": {"type": "string", "description": "Component reference, e.g. 'R1'."},
                "before": {"type": "string", "description": "Its signature in the baseline."},
                "after": {"type": "string", "description": "Its signature in the compared deck."},
            },
            "required": ["reference", "before", "after"],
        },
        "description": "References in both decks whose type/value signature differs.",
    },
    "directives_added": {
        "type": "array",
        "items": {"type": "string"},
        "description": "SPICE directives in the compared deck and not the baseline.",
    },
    "directives_removed": {
        "type": "array",
        "items": {"type": "string"},
        "description": "SPICE directives in the baseline and not the compared deck.",
    },
}


def parse_failure_warnings(pairs: Sequence[tuple[str, str | None]]) -> list[str]:
    """Structured warnings for the decks a structural diff could not parse.

    ``pairs`` is ``(display name, parse error or None)`` per side. An unparsed
    deck is diffed as an EMPTY circuit, so the other side's whole content reads
    as added or removed. That interpretation has to ride in the structured
    channel, not only the text one: structured-aware clients never see the text
    caveat, and the bogus added/removed lists look trustworthy without it.

    Returns the interpretation first, then one message per unparsed deck.
    """
    errors = [err for _name, err in pairs if err]
    if not errors:
        return []
    unparsed = " and ".join(name for name, err in pairs if err)
    return [
        f"{unparsed} could not be parsed; the diff treats it as empty, so its "
        "components/directives appear as added/removed. Fix the file before "
        "trusting this comparison.",
        *errors,
    ]


def snap_match(requested: float, actual: float, *, rtol: float = 1e-3) -> bool:
    """True iff ``actual`` is within ``rtol`` (relative) of ``requested``.

    A query snaps to the nearest available sample (a discrete step value or the
    nearest point on a sweep axis); a legitimate lookup lands on (or extremely
    near) one. A large gap means the request fell outside the range and was
    silently clamped to the nearest endpoint — worth flagging rather than
    presenting the clamp as an exact answer. Shared by the step-axis lookup and
    ``query_value``'s direct ``at`` path so their snap flags can't drift.
    """
    scale = max(abs(actual), abs(requested), 1e-30)
    return abs(requested - actual) <= rtol * scale


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
        description=FORMAT_DESCRIPTION,
    )


def _step_get_native_axis(
    raw: RawRead, args: StepGetInput, signal: str, target: float
) -> types.CallToolResult:
    """Query on the .raw's native axis (DC sweep variable / AC frequency).

    The queried axis IS the inner axis, so this is a nearest-neighbour lookup
    on the axis values; a request beyond the axis ends is a clamp worth flagging.
    """
    # On the native-axis branch the queried axis IS the inner axis, so
    # there is no second position for ``at`` to select. Silently
    # ignoring it would return a value at ``value`` while the caller
    # believes the ``at`` slice was applied — refuse loudly instead.
    if args.at is not None:
        raise NetlistError(
            f"'at' does not apply here: {args.axis!r} is the raw file's "
            "native axis, so the query position is 'value' itself. "
            "'at' selects the inner-axis point only when 'axis' names a "
            ".step parameter."
        )
    try:
        axis_vals = real_axis(np.asarray(raw.get_axis(step=0))).tolist()
    except Exception as e:
        raise NetlistError(
            f"Cannot read axis values: {e}. Use query_value if "
            "the raw doesn't have an explicit axis."
        ) from e
    if not axis_vals:
        raise NetlistError(f"Axis {args.axis!r} has no samples in this raw file.")
    # nearest neighbour
    ins = bisect.bisect_left(axis_vals, target)
    if ins == 0:
        idx = 0
    elif ins == len(axis_vals):
        idx = len(axis_vals) - 1
    else:
        idx = ins - 1 if abs(axis_vals[ins - 1] - target) <= abs(axis_vals[ins] - target) else ins
    wave = raw.get_wave(signal, step=0)
    actual = float(axis_vals[idx])
    # This is a continuous native axis (DC sweep variable / AC frequency),
    # not a discrete step list: an off-grid interior request is a normal
    # nearest-neighbour lookup, and only a request beyond the axis ends is
    # genuinely clamped. sample_to_dict keeps complex AC samples intact
    # (magnitude/phase) instead of float() silently dropping the imag part.
    sample_dict = sample_to_dict(wave[idx])
    exact = snap_match(target, actual)
    lo, hi = min(axis_vals[0], axis_vals[-1]), max(axis_vals[0], axis_vals[-1])
    out_of_range = target < lo or target > hi
    data = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": actual,
        "exact_match": exact,
        **sample_dict,
    }
    sample_str = (
        f"{sample_dict['value']:g}"
        if "value" in sample_dict
        else f"{sample_dict['magnitude_db']:.3f} dB / {sample_dict['phase_deg']:.2f}°"
    )
    summary = f"{signal} at {args.axis}={actual:g}: {sample_str}"
    if out_of_range:
        warning = (
            f"Requested {args.axis}={target:g} is outside the swept range "
            f"[{lo:g}, {hi:g}]; clamped to the nearest end {actual:g}."
        )
        data["warnings"] = [warning]
        summary += f"\nWarning: {warning}"
    return format_response(summary, data, args.format)


def _step_get_param_lookup(
    raw: RawRead,
    raw_path: Path,
    args: StepGetInput,
    signal: str,
    target: float,
    axis_lower: str,
) -> types.CallToolResult:
    """Query by .step parameter value, using the nearest stepped run.

    Falls back to .log parsing when spicelib's ``get_steps`` returns nothing
    (which it does for ``.step param NAME`` runs — the parameter map lives in
    the log, not the .raw header).
    """
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
        if available_axes:
            raise NetlistError(
                f"Step axis {args.axis!r} not found in this raw file. "
                "Available axes: " + ", ".join(available_axes)
            )
        # No .step parameters at all — the caller likely meant the primary sweep
        # axis of a bare .dc/.ac sweep, which isn't a step. Point at the direct
        # route instead of a bare "not found".
        raise NetlistError(
            f"This raw file has no .step parameters, so {args.axis!r} is not a step "
            f"axis. If {args.axis!r} is the primary sweep variable of a bare .dc/.ac "
            f"sweep, query it directly: query_value(at='{args.value}')."
        )

    assert best_actual is not None  # set in lockstep with best_idx above
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
    warnings: list[str] = []
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
    else:
        # No inner coordinate requested. For .op raws index 0 is the only
        # sample; for .ac/.tran it's the first (passband / t=0) bin, whose
        # value is uninterpretable without knowing the coordinate. Surface
        # the implied coordinate when there is a real inner axis.
        try:
            inner_axis = real_axis(np.asarray(raw.get_axis(step=best_idx)))
        except Exception:
            inner_axis = np.asarray([])
        if inner_axis.size > 1:
            actual_at = float(inner_axis[0])
            warnings.append(
                f"No 'at' given: returning the first inner sample at {actual_at:g}. "
                "Pass 'at' (frequency for .ac, time for .tran) to pick a point."
            )

    if not snap_match(target, best_actual):
        warnings.append(
            f"Requested {args.axis}={target:g} but no step matches; using the "
            f"nearest step {best_actual:g}."
        )

    sample_dict = sample_to_dict(wave[inner_idx])
    data: dict = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": best_actual,
        "exact_match": snap_match(target, best_actual),
        "step_index": best_idx,
        **sample_dict,
    }
    if target_at is not None:
        data["requested_at"] = target_at
    if actual_at is not None:
        data["actual_at"] = actual_at
    if warnings:
        data["warnings"] = warnings

    sample_str = (
        f"{sample_dict['value']:g}"
        if "value" in sample_dict
        else f"{sample_dict['magnitude_db']:.3f} dB / {sample_dict['phase_deg']:.2f}°"
    )
    at_str = f", at={actual_at:g}" if actual_at is not None else ""
    summary = f"{signal} at {args.axis}={best_actual:g} (step {best_idx}){at_str}: {sample_str}"
    for warning in warnings:
        summary += f"\nWarning: {warning}"
    return format_response(summary, data, args.format)


# Internal compute adapter — exposed publicly via query_value(step_axis=, step_value=).
# Operates on a SINGLE multi-step .raw (as produced by .step/.dc). An external
# sweep job (configure_sweep/run_sweep) emits N single-point raws with no step
# axis instead — use batch_results for those.
async def handle_step_get(args: StepGetInput, state: SessionState) -> types.CallToolResult:
    """Query a signal at a specific axis value of a stepped .raw result."""
    raw_path = safe_path(args.raw_file, state)
    raw = await services.load_raw(raw_path, state)

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
        return _step_get_native_axis(raw, args, signal, target)
    return _step_get_param_lookup(raw, raw_path, args, signal, target, axis_lower)


# ---------------------------------------------------------------------------
# Batch-transaction op — apply many edits to one .asc atomically.
# ---------------------------------------------------------------------------


_RotationLiteral = Literal["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"]

# Shared op-field descriptions. The op models are published in three profiles
# (full, agentic and the consolidated edit_schematic), so each string is paid
# for three times over — keep them to the fact the caller cannot infer. The
# coordinate convention is stated once on the ``ops`` field instead of on the
# dozen x/y pairs below.
_ROTATION_DESCRIPTION = (
    "'R<deg>' rotates clockwise by that many degrees; 'M<deg>' mirrors "
    "horizontally and then rotates. Pins move with the body."
)
_REFERENCE_DESCRIPTION = "Reference designator of an existing component, e.g. 'R1', 'M3'."
_COORDINATE_DESCRIPTION = (
    "All x/y are LTspice grid units, with x increasing to the right and y increasing DOWNWARD."
)


class _OpAddComponent(StrictModel):
    """Place a new component from its symbol at a coordinate."""

    op: Literal["add_component"]
    reference: str = Field(
        description="Reference designator to give the new part, e.g. 'R1', 'M3'."
    )
    symbol: str = Field(
        description=(
            "Symbol name without the .asy extension, e.g. 'res', 'nmos4'. It must be "
            "one the active symbol libraries resolve."
        )
    )
    x: int
    y: int
    rotation: _RotationLiteral = Field(default="R0", description=_ROTATION_DESCRIPTION)
    value: str | None = Field(
        default=None, description="Value or model name, e.g. '10k', '1u', 'BSS123'."
    )
    attributes: dict[str, str] | None = Field(
        default=None,
        description="Further symbol attributes by name, e.g. SpiceLine, SpiceModel.",
    )


class _OpSetComponentValue(StrictModel):
    """Set an existing component's primary value."""

    op: Literal["set_component_value"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    value: str = Field(description="New value or model name, e.g. '10k', '1u', 'BSS123'.")


class _OpSetComponentAttribute(StrictModel):
    """Set one named symbol attribute on an existing component."""

    op: Literal["set_component_attribute"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    attribute: str = Field(
        description="Attribute name, e.g. Value, Value2, SpiceLine, SpiceModel."
    )
    value: str = Field(description="New value for that attribute.")


class _OpRemoveComponent(StrictModel):
    """Delete a component, optionally taking its dangling wires with it."""

    op: Literal["remove_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    cleanup_wires: bool = Field(
        default=False,
        description=(
            "Also delete wires left dangling at the removed component's pins. "
            "These wire deletions are NOT restored by a later add_component, and "
            "the edit commits — keep your own copy of the sheet if you may need to "
            "undo removing the wrong component."
        ),
    )


class _OpMoveComponent(StrictModel):
    """Move an existing component, optionally re-rotating it."""

    op: Literal["move_component"]
    reference: str = Field(description=_REFERENCE_DESCRIPTION)
    x: int
    y: int
    rotation: _RotationLiteral | None = Field(
        default=None, description=f"Omit to keep the current rotation. {_ROTATION_DESCRIPTION}"
    )


class _OpAddNetLabel(StrictModel):
    """Name a net by placing a label, either at a pin or at a coordinate."""

    op: Literal["add_net_label"]
    net: str = Field(description="Net name the label declares, e.g. 'VDD', 'out'.")
    pin: str | None = Field(
        default=None,
        description="Place at this pin, e.g. 'M1.D'. Give this or x/y, not both.",
    )
    x: int | None = None
    y: int | None = None


class _OpWirePins(StrictModel):
    """Draw an orthogonal wire between two pins, refusing a diagonal run, a pin
    collision, or an overlapping wire junction rather than drawing them."""

    # "connect" is the deprecated former name, still accepted.
    op: Literal["wire_pins", "connect"]
    from_pin: str = Field(
        description="Source pin as 'Reference.Pin', e.g. 'M1.D', or 'net:NAME' for a label."
    )
    to_pin: str = Field(
        description="Target pin as 'Reference.Pin', e.g. 'M4a.D', or 'net:NAME' for a label."
    )
    waypoints: list[WaypointInput] = Field(
        default_factory=list,
        description=(
            "Corner points the route must pass through, in order. Omit to let the "
            "router pick the elbow; supply them to steer around other parts."
        ),
    )


class _OpRemoveNetLabel(StrictModel):
    """Delete a net label, addressed by its pin or its coordinate."""

    op: Literal["remove_net_label"]
    pin: str | None = Field(
        default=None, description="The pin the label sits on. Give this or x/y, not both."
    )
    x: int | None = None
    y: int | None = None


class _OpRemoveWire(StrictModel):
    """Delete wires, addressed either as one exact segment or as every segment
    incident on a point. Prefer the segment form to undo one wire_pins call."""

    op: Literal["remove_wire"]
    x1: int | None = Field(
        default=None,
        description=(
            "Segment form: with y1/x2/y2, removes only this segment (either "
            "direction). Byte-identical duplicates all go at once; 'removed' counts them."
        ),
    )
    y1: int | None = None
    x2: int | None = None
    y2: int | None = None
    pin: str | None = Field(
        default=None,
        description=(
            "Incident-point form: removes EVERY segment touching this pin, including "
            "wires belonging to other connections at a shared node."
        ),
    )
    x: int | None = Field(default=None, description="Incident-point form, as a coordinate.")
    y: int | None = None


class _OpAddDirective(StrictModel):
    """Add a SPICE directive or a comment to the sheet."""

    op: Literal["add_directive"]
    instruction: str = Field(
        description="Directive or comment text, e.g. '.tran 1m' or '.model NMOS ...'."
    )
    kind: Literal["directive", "comment"] = Field(
        default="directive",
        description="'directive' is simulated; 'comment' is annotation LTspice ignores.",
    )
    x: int | None = Field(
        default=None,
        description=(
            "Omit x/y to use the default anchor, stepped down past any text already "
            "there so directives do not stack on top of each other."
        ),
    )
    y: int | None = None
    size: int = Field(default=2, description="LTspice text size index.")


class _OpRemoveDirective(StrictModel):
    """Delete a directive or comment by its text."""

    op: Literal["remove_directive"]
    instruction: str = Field(
        description=(
            "Directive or comment text to remove. Matched literally (exact) by "
            "default; prefix with 'regex:' to match by pattern. Inverse of "
            "add_directive."
        )
    )


SchematicOp = (
    _OpAddComponent
    | _OpSetComponentValue
    | _OpSetComponentAttribute
    | _OpRemoveComponent
    | _OpMoveComponent
    | _OpAddNetLabel
    | _OpRemoveNetLabel
    | _OpRemoveWire
    | _OpWirePins
    | _OpAddDirective
    | _OpRemoveDirective
)


def _resolve_op_xy(
    op: "_OpAddNetLabel | _OpRemoveNetLabel | _OpRemoveWire", editor: AscEditor
) -> tuple[int, int]:
    """Resolve an op's ``pin`` | ``x,y`` point locator to a coordinate."""
    if op.pin is not None:
        return _resolve_pin(op.pin, editor)
    if op.x is not None and op.y is not None:
        return op.x, op.y
    raise NetlistError(f"{op.op} needs either pin or both x and y.")


def _apply_op_inplace(editor: AscEditor, op: SchematicOp, asc_path: Path) -> dict[str, object]:
    """Apply one schematic op against ``editor`` in place, return its result.

    Mirrors the validation done by the per-op tools but skips the load /
    save / lock dance — the caller (``handle_apply_schematic_ops``) holds
    the lock and saves once at the end.

    Raises ``NetlistError`` on any per-op validation failure; the caller
    decides whether to abort or continue based on ``stop_on_error``.
    """
    if isinstance(op, _OpAddComponent):
        symbol_info = get_symbol_info(op.symbol)
        if symbol_info is None:
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
        return {
            "op": "add_component",
            **_placed_component_data(
                editor,
                op.reference,
                op.symbol,
                op.x,
                op.y,
                op.rotation,
                symbol_info,
            ),
        }

    if isinstance(op, _OpSetComponentValue):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        lint = _level_label_lint(editor, op.reference, op.value)
        _apply_component_value(editor, op.reference, op.value)
        result = {"op": "set_component_value", "reference": op.reference, "value": op.value}
        if lint:
            result["warnings"] = [lint]
        return result

    if isinstance(op, _OpSetComponentAttribute):
        _reject_unknown_attr(op.attribute)
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        if not op.value.strip():
            # Empty value = clear. LTspice's format has no "empty value"
            # representation (a 2-token SYMATTR line is unreadable), so the
            # line is removed instead.
            _require_clearable_attr(op.reference, op.attribute)
            editor.get_component(op.reference).attributes.pop(op.attribute, None)
        else:
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
        rm_result: dict[str, object] = {"op": "remove_component", "reference": op.reference}
        if op.cleanup_wires and target_only:
            rm_result["deleted_wires"] = _drop_wires_at(editor, target_only)
        elif target_only:
            # Same orphaned-wire warning the standalone handler surfaces.
            orphaned = _orphaned_wire_coords(editor, target_only)
            if orphaned:
                rm_result["warnings"] = [
                    f"orphaned wires remain at: {', '.join(orphaned)}. "
                    "Re-run with cleanup_wires=true to delete them."
                ]
        return rm_result

    if isinstance(op, _OpMoveComponent):
        if op.reference not in editor.components:
            raise NetlistError(f"Component '{op.reference}' not found.")
        new_rot = (
            _parse_rotation(op.rotation)
            if op.rotation is not None
            else editor.get_component_position(op.reference)[1]
        )
        old_pins = _component_pin_coords(editor, op.reference)
        other_pins = _other_components_pin_coords(editor, op.reference)
        editor.set_component_position(op.reference, Point(op.x, op.y), new_rot)
        # Same bbox-overlap + orphaned-wire warnings as the standalone handler.
        mv_warnings = _move_component_warnings(
            editor, op.reference, new_rot.name, op.x, op.y, old_pins, other_pins
        )
        mv_result: dict[str, object] = {"op": "move_component", "reference": op.reference}
        if mv_warnings:
            mv_result["warnings"] = mv_warnings
        return mv_result

    if isinstance(op, _OpAddNetLabel):
        x, y = _resolve_op_xy(op, editor)
        # Same short-refusal + duplicate/floating warnings as the standalone
        # handler — the op is the public path, so it enforces the same rules.
        warnings = _add_net_label_checks(editor, op.net, x, y)
        editor.labels.append(Text(coord=Point(x, y), text=op.net, type=TextTypeEnum.LABEL))
        result: dict[str, object] = {"op": "add_net_label", "net": op.net, "x": x, "y": y}
        if warnings:
            result["warnings"] = warnings
        return result

    if isinstance(op, _OpRemoveNetLabel):
        x, y = _resolve_op_xy(op, editor)
        before = len(editor.labels)
        editor.labels = [
            lbl for lbl in editor.labels if not (int(lbl.coord.X) == x and int(lbl.coord.Y) == y)
        ]
        removed = before - len(editor.labels)
        if removed == 0:
            raise NetlistError(f"No net label found at ({x},{y}).")
        return {"op": "remove_net_label", "x": x, "y": y, "removed": removed}

    if isinstance(op, _OpRemoveWire):
        before = len(editor.wires)
        if all(v is not None for v in (op.x1, op.y1, op.x2, op.y2)):
            # Exact-segment form: drop the matching segment in either direction.
            seg = ((op.x1, op.y1), (op.x2, op.y2))
            rev = ((op.x2, op.y2), (op.x1, op.y1))
            kept = [
                w
                for w in editor.wires
                if ((int(w.V1.X), int(w.V1.Y)), (int(w.V2.X), int(w.V2.Y))) not in (seg, rev)
            ]
            copies = len(editor.wires) - len(kept)
            if copies > 1:
                # A duplicated segment is one connection drawn twice, so "remove
                # the duplicate" and "remove the connection" are the same
                # request at this interface. Removing every copy is right only
                # when the connection was redundant — and redundancy is a NET
                # fact, not a pin fact: cutting a duplicated bridge between two
                # wired stubs splits the net while every pin still touches some
                # wire, so a floating-pin scan blesses exactly the cut this
                # guard exists to refuse. The oracle is the net partition.
                part_before = _net_partition(editor)
                nets_before: dict[tuple[int, int], list[tuple[int, int]]] = {}
                for coord in part_before.pin_owners:
                    nets_before.setdefault(part_before.root(coord), []).append(coord)
                original, editor.wires = editor.wires, kept
                part_after = _net_partition(editor)
                for coords in nets_before.values():
                    sides: dict[tuple[int, int], list[tuple[int, int]]] = {}
                    for coord in coords:
                        sides.setdefault(part_after.root(coord), []).append(coord)
                    if len(sides) < 2:
                        continue
                    editor.wires = original

                    def _side_names(side: list[tuple[int, int]]) -> str:
                        names = [
                            f"{ref}.{pin}" if pin else ref
                            for c in sorted(side)
                            for ref, pin in part_before.pin_owners[c]
                        ]
                        return ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")

                    first, second, *_ = sorted(sides.values(), key=len, reverse=True)
                    raise NetlistError(
                        f"Refusing to remove the {copies} copies of "
                        f"({op.x1},{op.y1})->({op.x2},{op.y2}): they are one connection "
                        f"drawn {copies} times, and removing it would split the net, "
                        f"leaving {_side_names(second)} disconnected from "
                        f"{_side_names(first)}. Remove the pin's other segments first "
                        "if the disconnection is what you want."
                    )
            else:
                editor.wires = kept
        elif op.pin is not None or (op.x is not None and op.y is not None):
            # Incident-point form: drop every segment touching the coordinate.
            _drop_wires_at(editor, {_resolve_op_xy(op, editor)})
        else:
            raise NetlistError(
                "remove_wire needs either all of x1,y1,x2,y2 (a segment) or a "
                "pin / both x and y (a point)."
            )
        removed = before - len(editor.wires)
        if removed == 0:
            raise NetlistError("No matching wire segment found to remove.")
        return {"op": "remove_wire", "removed": removed}

    if isinstance(op, _OpWirePins):
        plan = _plan_connect_route(editor, op.from_pin, op.to_pin, op.waypoints)
        already = _append_wire_segments(editor, plan.segments)
        result = {
            "op": op.op,
            "from_pin": op.from_pin,
            "to_pin": op.to_pin,
            "wire_count": len(plan.segments) - len(already),
        }
        if already:
            result["already_present"] = [
                {"from": {"x": sx1, "y": sy1}, "to": {"x": sx2, "y": sy2}}
                for sx1, sy1, sx2, sy2 in already
            ]
        return result

    if isinstance(op, _OpAddDirective):
        # Comments allow any text; SPICE directives go through validate_directive.
        if op.kind == "directive":
            err = validate_directive(op.instruction, simulator="LTspice")
            if err is not None:
                raise NetlistError(
                    f"Refusing directive {op.instruction!r}: {err.message} ({err.suggestion})"
                )
        text_type = TextTypeEnum.DIRECTIVE if op.kind == "directive" else TextTypeEnum.COMMENT
        _append_asc_text(
            editor, op.instruction, text_type, op.x, op.y, op.size, default_x=16, default_y=16
        )
        return {"op": "add_directive", "instruction": op.instruction}

    if isinstance(op, _OpRemoveDirective):
        # Literal-by-default, 'regex:' opt-in, raises if nothing matched.
        removed = _remove_directive_or_comment(editor, op.instruction)
        return {"op": "remove_directive", "instruction": op.instruction, "removed": removed}

    raise NetlistError(f"Unknown op type: {type(op).__name__}")


def _run_op_batch(
    editor: AscEditor,
    ops: Sequence[SchematicOp],
    asc_path: Path,
    *,
    stop_on_error: bool,
) -> tuple[list[dict[str, object]], str | None]:
    """Apply ``ops`` in order via ``_apply_op_inplace``; return (results, abort_reason).

    One unified entry per attempted op — ``{index, op, ok, error, **op_result}``.
    A ``NetlistError``/``ValueError`` marks that op ``ok=False`` with its message;
    when ``stop_on_error`` is set the first failure aborts (``abort_reason`` set,
    loop stops). Shared verbatim by ``apply_schematic_ops`` and ``edit_schematic``.
    """
    results: list[dict[str, object]] = []
    abort_reason: str | None = None
    for i, op in enumerate(ops):
        entry: dict[str, object] = {"index": i, "op": op.op, "ok": True, "error": None}
        try:
            op_result = _apply_op_inplace(editor, op, asc_path)
            entry.update({k: v for k, v in op_result.items() if k != "op"})
        except (NetlistError, ValueError) as e:
            entry["ok"] = False
            entry["error"] = str(e)
            results.append(entry)
            if stop_on_error:
                abort_reason = f"op #{i} ({op.op}) failed: {e}"
                break
            continue
        results.append(entry)
    return results, abort_reason
