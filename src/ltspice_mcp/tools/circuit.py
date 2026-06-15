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
import math
import re
from collections.abc import AsyncIterator, Callable
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
from ltspice_mcp.lib import atomic_write_bytes, atomic_write_text, services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.raw_parser import nearest_index, real_axis, sample_to_dict
from ltspice_mcp.lib.spice_lex import SpiceCard, SpiceLexError, TokenKind, lex, tokenize_body
from ltspice_mcp.lib.spice_validator import (
    ANALYSIS_KINDS,
    EXCLUSIVE_ANALYSIS_KINDS,
    MEAS_KINDS,
    drop_title_card,
    validate_directive,
    validate_netlist_arity,
    validate_netlist_bias_topology,
    validate_netlist_dangling_nodes,
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
            # manually edited. Reject up front.
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
    segments (e.g. the route ``connect`` is about to add) so checks operate
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

    return warnings


def _scope_floating_pin_warnings(
    warnings: list[dict], refs: set[str] | str, *, keep_other_kinds: bool
) -> list[dict]:
    """Scope ``_post_op_warnings`` floating-pin advisories to ``refs``.

    ``_post_op_warnings`` returns the whole schematic's floating pins on every
    call, so during an incremental build it re-emits every not-yet-wired pin —
    O(n²) noise that buries the one actionable warning. Each mutating handler
    keeps only the floating pins of the component(s) it touched: ``add_component``
    passes its new ref with ``keep_other_kinds=False`` (a bare placement raises
    no other advisory); ``connect`` passes its two touched refs with
    ``keep_other_kinds=True`` so the shorts / junction-overlap warnings it *can*
    create still pass through.
    """
    ref_set = {refs} if isinstance(refs, str) else refs
    result = []
    for w in warnings:
        if w.get("kind") == "floating_pin":
            if w.get("ref") in ref_set:
                result.append(w)
        elif keep_other_kinds:
            result.append(w)
    return result


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
    limit: int = Field(
        default=50, description="Max results to return (server caps at 50; page with offset)"
    )
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
            "Default: directives are auto-placed in free space near the "
            "schematic's lower-left; comments default to the sheet origin (0,0)."
        ),
    )
    y: int | None = Field(
        default=None,
        description="Optional Y coordinate (see ``x``).",
    )
    size: int | None = Field(
        default=None,
        description="Font size (.asc only). 1=small, 2=normal (default), 3=large.",
    )


class RemoveComponentInput(ToolInput):
    path: str = Field(description="Path to .asc schematic")
    reference: str = Field(description="Component reference to remove (e.g., 'R1', 'M3')")
    cleanup_wires: bool = Field(
        default=False,
        description=(
            "When true, also delete every wire whose endpoint touches one of "
            "the removed component's pins. Default false leaves the wires in "
            "place and surfaces a warning, so callers can opt in once "
            "they've confirmed the removal is clean."
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


class ResetSchematicInput(ToolInput):
    path: str = Field(description="Path to .asc schematic to revert to its pre-session state")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


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
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
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


_MAX_ASC_SNAPSHOTS = 64


def _snapshot_asc(path: Path, state: SessionState) -> None:
    """Capture the on-disk bytes of an .asc file before its first in-session edit.

    Lets ``reset_schematic`` restore the pre-session state. No-op for non-.asc
    paths, after the first capture for a path, or if the file can't be read.
    Called at the start of every mutating .asc op (the disk file is still in its
    pre-edit state at that point, so the first capture is the last good state).

    Bounded to ``_MAX_ASC_SNAPSHOTS`` distinct paths: when full, the oldest
    snapshot is evicted (FIFO). reset_schematic is a best-effort, session-scoped
    recovery hatch, so dropping the oldest restore point under heavy churn is an
    acceptable trade for bounded memory.
    """
    if not _is_asc(path):
        return
    key = str(path)
    if key in state.asc_snapshots:
        return
    with contextlib.suppress(OSError):
        data = path.read_bytes()
        while len(state.asc_snapshots) >= _MAX_ASC_SNAPSHOTS:
            state.asc_snapshots.pop(next(iter(state.asc_snapshots)))
        state.asc_snapshots[key] = data


def _require_asc(path: Path) -> None:
    """Raise if path is not an .asc file (for schematic-only operations)."""
    if not _is_asc(path):
        raise NetlistError(f"This operation requires an .asc schematic, got '{path.suffix}'. ")


# spicelib's SpiceEditor.set_parameter appends this exact boilerplate comment
# to a .PARAM line when it INSERTS a new parameter. Strip it on save so it
# doesn't leak into the netlist. End-anchored and gated to .param lines, so a
# user-authored "; note" on any line (including a real .param) is preserved.
_BATCH_INSTRUCTION_RE = re.compile(r"[ \t]*;[ \t]*Batch instruction[ \t]*$")


def _strip_batch_instruction(text: str) -> str:
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.lstrip().lower().startswith(".param"):
            stripped = line.rstrip("\r\n")
            ending = line[len(stripped) :]
            lines[i] = _BATCH_INSTRUCTION_RE.sub("", stripped) + ending
    return "".join(lines)


def _atomic_save_editor(editor: Editor, target: Path) -> None:
    """Render editor to a buffer, then atomically rename onto target.

    Avoids partial-write corruption: if rendering or writing fails,
    the original file is untouched. Spicelib's save_netlist accepts an
    io.StringIO sink (verified for AscEditor and SpiceEditor), so we can
    skip the temp-file dance and reuse atomic_write_text's tested rename.
    """
    buf = io.StringIO()
    editor.save_netlist(buf)
    text = buf.getvalue()
    if isinstance(editor, SpiceEditor):
        text = _strip_batch_instruction(text)
    atomic_write_text(
        target,
        text,
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
        _snapshot_asc(path, state)
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
    name="create_netlist",
    description=(
        "Create a new SPICE netlist file from content string. Automatically appends .END if missing."
    ),
    input_model=CreateNetlistInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
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

    # Reject empty / whitespace-only content up front. Otherwise spicelib's
    # parser surfaces a cryptic ``Expected pattern "^\*" not found`` when it
    # fails to find a title line.
    if not content.strip():
        raise NetlistError(
            "Netlist content is empty. Provide at least a title line and one "
            "element or analysis directive (e.g. 'V1 in 0 1\\n.op')."
        )

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
    name="read_circuit",
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
    name="list_components",
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
    # concatenated into the displayed value.
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
            # For .asc, read Value SYMATTR directly (avoids the
            # Value+Value2 concat). For .cir/.net this branch is not
            # reached; netlists go through _list_components_netlist.
            if is_asc:
                value = services.asc_component_value(editor, comp_ref)
            else:
                value = editor.get_component_value(comp_ref)
        except Exception:
            # spicelib's component-line regex chokes on B-sources with
            # commas in if(...) expressions; degrade gracefully rather
            # than abort the whole listing.
            value = "<unparseable>"
        entry: dict = {"reference": comp_ref, "value": value}
        # Surface non-default SYMATTRs (SpiceLine, SpiceModel, …) for
        # .asc components so callers don't need a per-component
        # component_info round-trip to spot W=10u/L=0.5u-style overrides.
        if is_asc and comp_ref in editor.components:
            attrs = services.asc_component_attributes(editor.components[comp_ref])
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
        result += f"\n\nNext page: list_components(path=..., offset={offset + limit})"

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
        body += f"\n\nNext page: list_components(path=..., offset={offset + limit})"

    data = {
        "components": comp_list,
        "pagination": pagination_metadata(total, offset, limit),
    }
    if prefix:
        data["prefix"] = prefix
    return format_response(body, data, fmt)


@registry.tool(
    name="set_component_value",
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
    name="parameter",
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
        # get_all_parameter_names() uppercases (spicelib), but on-disk names
        # keep their source casing. SPICE params are case-insensitive, so this
        # is cosmetic — recover the verbatim casing from the file text so the
        # read-all projection matches what the user wrote.
        from ltspice_mcp.lib.encoding import read_spice_text

        source_text = ""
        with contextlib.suppress(Exception):
            source_text = read_spice_text(file_path)

        param_lines = []
        for name in param_names:
            display_name = name
            match = re.search(rf"\b{re.escape(name)}\b", source_text, re.IGNORECASE)
            if match:
                display_name = match.group(0)
            value = editor.get_parameter(name)
            param_lines.append(f".PARAM {display_name} = {value}")
            params[display_name] = value
        result = "\n".join(param_lines)
    else:
        result = "No .PARAM directives found"

    return format_response(result, {"parameters": params}, fmt)


_ASC_TEXT_DEFAULT_SIZE = 2
"""LTspice's normal font size — the fallback when a caller doesn't pick one."""


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
    kinds and ``apply_schematic_ops``' ``add_directive`` op all come through
    here, so placement defaulting can't drift between them. The per-site
    ``default_x``/``default_y`` differ deliberately (comments default to the
    sheet origin; directives to (16,16)).
    """
    editor.directives.append(
        Text(
            coord=Point(
                x if x is not None else default_x,
                y if y is not None else default_y,
            ),
            text=text,
            type=text_type,
            size=size if size is not None else _ASC_TEXT_DEFAULT_SIZE,
        )
    )


@registry.tool(
    name="edit_directive",
    description=(
        "Add or remove a SPICE directive or .asc free-text comment. Set "
        "``kind=comment`` for annotation text; default is a SPICE directive. "
        "Works on .cir/.net and .asc; ``kind=comment`` is .asc-only. "
        "``remove`` matches against directives AND comments, so callers can "
        "delete either kind without knowing which it is. Adding a ``.param`` "
        "is not supported here — use the 'parameter' tool to set .PARAM values."
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
                # Refuse comment text that *looks* like a directive —
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
                _append_asc_text(
                    cast(AscEditor, editor),
                    instruction,
                    TextTypeEnum.COMMENT,
                    args.x,
                    args.y,
                    args.size,
                    default_x=0,
                    default_y=0,
                )
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
                # spicelib refuses .param via add_instruction (RuntimeError,
                # surfaced to the user as an opaque "Internal error"); route
                # them to the dedicated tool instead (F3).
                if stripped.lower().startswith(".param"):
                    raise NetlistError(
                        "edit_directive cannot add a '.param' — SPICE parameters "
                        "are managed separately. Use the 'parameter' tool to set "
                        "a .PARAM value (e.g. parameter(name='foo', value='1'))."
                    )
                # Pre-flight validation: catch known-bad patterns (e.g. vdb()
                # in .MEAS) before they reach the simulator and fail post-hoc
                # inside the .log.
                err = validate_directive(instruction, simulator="LTspice")
                if err is not None:
                    raise NetlistError(f"{err.message}\n  Suggestion: {err.suggestion}")
                if _is_asc(file_path) and (
                    args.x is not None or args.y is not None or args.size is not None
                ):
                    # Honor the documented placement params on the .asc
                    # directive branch (previously only the comment branch
                    # read them — spicelib's add_instruction picks its own
                    # spot and hardcodes size, so x/y/size were silently
                    # ignored).
                    _append_asc_text(
                        cast(AscEditor, editor),
                        instruction,
                        TextTypeEnum.DIRECTIVE,
                        args.x,
                        args.y,
                        args.size,
                        default_x=16,
                        default_y=16,
                    )
                else:
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
                "Use read_circuit to see what's actually in the file."
            )
        return "directive(s)/comment(s)"

    directive_hit = bool(editor.remove_instruction(instruction))
    comment_hit = _strip_matching_comments(editor, instruction)
    if not (directive_hit or comment_hit):
        raise NetlistError(
            f"No directive or comment matched {instruction!r} exactly. "
            "Match is literal by default — pass 'regex:<pattern>' for regex "
            "matching, or copy the line verbatim from read_circuit."
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
    name="remove_component",
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
    name="move_component",
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
    name="set_component_attribute",
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
    name="add_component",
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
            "Use symbol_info to verify the symbol name, or "
            "configure [schematic] symbol_paths in ltspice-mcp.toml."
        )

    async with _editing_asc(asc_path, state) as editor:
        if reference in editor.components:
            raise NetlistError(
                f"Component '{reference}' already exists. "
                "Use set_component_value to modify it, "
                "or remove_component to remove it first."
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
        # add_component adds no wires or labels, so the only *new* advisory it
        # can raise is a floating pin on the component it just placed.
        validation_warnings = _scope_floating_pin_warnings(
            _post_op_warnings(editor), reference, keep_other_kinds=False
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
        return format_response(result, fallback_data, args.format)

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

    return format_response(result, data, args.format)


_previous_exports: dict[Path, list[str]] = {}


@registry.tool(
    name="export_netlist",
    description="Export an .asc schematic to a SPICE netlist (.net) using LTspice.",
    input_model=ExportNetlistInput,
    # Writes (and overwrites) the sibling .net file — not read-only, and
    # destructive toward a hand-edited netlist at that path, matching the
    # annotation convention of the other overwrite-capable file writers.
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
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
    name="reset_schematic",
    description=(
        "Revert an .asc schematic to the state it had BEFORE the first edit this "
        "session — a recovery escape hatch for when a sequence of edits went wrong. "
        "The server snapshots each .asc file's bytes just before its first in-session "
        "mutation (add_component, set_component_value, move_component, connect, "
        "apply_schematic_ops, etc.); this restores that snapshot exactly and drops it "
        "(so a later edit establishes a fresh restore point). Because add_component is a "
        "trigger, the first add_component on a freshly created schematic snapshots the "
        "empty file — so reset can revert all the way back to the empty post-create "
        "state, dropping every component added this session. Returns reverted=false (not an error) "
        "when the file has no recorded in-session edits. Note: the snapshot lives only "
        "for the current server session — it does not persist across restarts, and it "
        "is not a substitute for version control."
    ),
    input_model=ResetSchematicInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "reverted": {"type": "boolean"},
            "bytes": {"type": ["integer", "null"]},
        },
    },
)
async def handle_reset_schematic(
    args: ResetSchematicInput, state: SessionState
) -> types.CallToolResult:
    """Restore an .asc to its pre-first-edit snapshot captured this session."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)
    key = str(asc_path)
    snapshot = state.asc_snapshots.get(key)

    if snapshot is None:
        return format_response(
            f"No in-session edits recorded for {asc_path.name}; nothing to revert.",
            {"path": str(asc_path), "reverted": False, "bytes": None},
            args.format,
        )

    async with _get_edit_lock(asc_path):
        atomic_write_bytes(asc_path, snapshot, durable=False)
        state.editors.invalidate(asc_path)
        del state.asc_snapshots[key]

    return format_response(
        f"Reverted {asc_path.name} to its pre-session state ({len(snapshot)} bytes).",
        {"path": str(asc_path), "reverted": True, "bytes": len(snapshot)},
        args.format,
    )


@registry.tool(
    name="symbol_info",
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
            f"Symbol '{symbol}' not found. Check the symbol name spelling (e.g. "
            f"'nmos', 'res', 'cap', 'voltage'), or configure symbol libraries via "
            f"[schematic] symbol_paths / LTSPICE_MCP_SYMBOL_PATHS.",
            show_hint=False,
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
    name="component_info",
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
    for attr_name, attr_val in services.asc_component_attributes(comp).items():
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
                f"Net label '{net_name}' not found in schematic. Add it with add_net_label first."
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
    name="add_net_label",
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
                        "connect to error on ambiguity.\n"
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
    name="connect",
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
        # Scope floating-pin advisories to the components this connect touched;
        # keep_other_kinds keeps the shorts / junction-overlap warnings connect
        # can create. rsplit matches _resolve_pin's ref/pin split convention.
        touched_refs = {
            pin.rsplit(".", 1)[0] for pin in (args.from_pin, args.to_pin) if "." in pin
        }
        validation_warnings = _scope_floating_pin_warnings(
            _post_op_warnings(ed), touched_refs, keep_other_kinds=True
        )

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

    # Pin-collision exemption is by exact endpoint *coordinate* (the
    # ``(px, py) in endpoints`` check below), NOT by whole component: the
    # OTHER pin of an endpoint component still lies on the route and must be
    # flagged — otherwise a waypoint landing on it silently shorts the
    # component while connect reports success (F1). ``skip_refs`` stays in the
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
    name="create_schematic",
    description=(
        "Create an empty .asc schematic ready for incremental editing via "
        "add_component / connect / add_net_label. "
        "Tip: prefer ``create_netlist`` + .cir for design iteration; "
        "use this only when a visual schematic is the deliverable."
        " Prefer apply_schematic_ops for multi-step builds (one transaction); "
        "wire signal nets with connect and ground via add_net_label flags at "
        "pins — don't hand-edit the .asc. Full layout guidance: the "
        "ltspice://guide resource."
    ),
    input_model=CreateSchematicInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
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
        "\n\nLayout checklist (full playbook: ltspice://guide):"
        "\n- Wire signal nets with connect — orthogonal only, waypoints for bends, "
        "route outside component bodies."
        '\n- Ground: add_net_label(net="0", pin="Ref.pin") at each ground pin; '
        "don't wire to a shared ground flag."
        "\n- Don't net-label signal nets — wire them."
        "\n- Multi-step build: use apply_schematic_ops (one transaction)."
        "\n- Matched devices (diff pairs/mirrors) share a y-tier; get pin coords "
        "from symbol_info."
    )


# SPICE element prefix → standard LTspice 2-terminal symbol. These six have
# unambiguous standard symbols and a fixed 2-node arity, so the netlist
# round-trips exactly. Multi-terminal / model-polarity devices (M/Q/J),
# subcircuit instances (X), and controlled sources are intentionally left
# for manual placement (reported as ``skipped``) — their symbol and polarity
# can't be inferred from the instance line alone.
_SYNTH_SYMBOL_MAP: dict[str, str] = {
    "R": "res",
    "C": "cap",
    "L": "ind",
    "V": "voltage",
    "I": "current",
    "D": "diode",
}

# Directive card kinds passed through verbatim into the .asc as SPICE
# directive text. ``subckt``/``ends`` block delimiters and the ``end``
# terminator are dropped (a standalone .ends in a flat schematic is
# meaningless); ``comment``/``blank`` carry no electrical meaning.
_SYNTH_PASSTHROUGH_KINDS = frozenset({"directive", "model", "param", "meas"})


class _SynthInstance(NamedTuple):
    """A 2-terminal element parsed from a netlist, ready for placement."""

    ref: str
    symbol: str
    nodes: tuple[str, str]
    value: str | None


def _parse_netlist_for_synth(
    content: str,
) -> tuple[list[_SynthInstance], list[str], list[dict], list[str]]:
    """Parse a SPICE netlist into placeable instances + directives.

    Returns ``(instances, directives, skipped, warnings)``. Pure — no symbol
    lookup or filesystem access — so it is unit-testable without an LTspice
    symbol library. Per SPICE convention the first non-blank line is the
    deck title and is dropped.
    """
    try:
        cards = lex(content).cards
    except SpiceLexError as e:
        raise NetlistError(f"Could not parse netlist: {e}") from e

    instances: list[_SynthInstance] = []
    directives: list[str] = []
    skipped: list[dict] = []
    warnings: list[str] = []
    title_consumed = False

    for card in cards:
        if card.kind == "blank":
            continue
        if not title_consumed:
            title_consumed = True
            if card.kind == "comment":
                # A leading ``*`` comment is the conventional SPICE deck
                # title — drop it.
                continue
            # No title comment present (a bare netlist fragment). Dropping the
            # first card here would silently delete a real element — e.g. the
            # source on a typical ``V1 ... / R1 ...`` fragment (F2). Keep it as
            # circuit content and warn instead of losing it.
            warnings.append(
                "No title line found (first line is not a '*' comment); kept it "
                "as circuit content. Prepend a '* title' line to suppress."
            )
            # fall through to classify this card below
        if card.kind == "instance":
            ref = card.instance_ref or ""
            prefix = ref[:1].upper()
            symbol = _SYNTH_SYMBOL_MAP.get(prefix)
            if symbol is None:
                skipped.append(
                    {
                        "ref": ref,
                        "reason": (
                            f"element type '{prefix or '?'}' is not auto-placed; "
                            "add it manually with add_component + connect"
                        ),
                    }
                )
                continue
            try:
                toks = tokenize_body(card.body)
            except SpiceLexError as e:
                # lex() classifies any letter-prefixed line as an instance by
                # prefix alone, without tokenizing — so a malformed body (e.g.
                # unbalanced parens) only fails here. Skip it cleanly with a
                # reason instead of letting the ValueError surface as a generic
                # "internal error", mirroring the guarded sites elsewhere.
                skipped.append({"ref": ref, "reason": f"could not tokenize element body: {e}"})
                continue
            if len(toks) < 3:
                skipped.append(
                    {
                        "ref": ref,
                        "reason": "fewer than two nodes; cannot place a 2-terminal symbol",
                    }
                )
                continue
            node1, node2 = toks[1].text, toks[2].text
            value = card.body[toks[2].body_end :].strip() or None
            instances.append(
                _SynthInstance(ref=ref, symbol=symbol, nodes=(node1, node2), value=value)
            )
        elif card.kind in _SYNTH_PASSTHROUGH_KINDS:
            body = card.body.strip()
            if body:
                directives.append(body)
        elif card.kind == "subckt":
            warnings.append(
                f"Subcircuit definition '{card.subckt_name}' not transferred — "
                "reference it via a .lib/.include directive or add instances manually."
            )
        # "ends", "end", "comment" are dropped.

    return instances, directives, skipped, warnings


class _PlacedSynthComponent(NamedTuple):
    """A synthesised component with absolute geometry and pin→net labels."""

    ref: str
    symbol: str
    x: int
    y: int
    value: str | None
    labels: list[tuple[str, int, int]]  # (net, x, y)


def _round_up_grid(value: int, grid: int = 16) -> int:
    """Round ``value`` up to the next multiple of ``grid``."""
    return ((value + grid - 1) // grid) * grid


def _layout_synth_components(
    instances: list[_SynthInstance],
) -> tuple[list[_PlacedSynthComponent], list[dict], list[str], set[str]]:
    """Grid-place instances and label each pin with its SPICE node.

    Connectivity is by net label (a FLAG carrying the node name at each pin),
    not by routed wires — same-named labels are electrically common in
    LTspice, so the result matches the netlist regardless of geometry. Needs
    the symbol library (``get_symbol_info``); instances whose symbol is
    unavailable or whose pin count ≠ node count are returned in ``skipped``.

    Returns ``(placed, skipped, warnings, nets)``.
    """
    placed: list[_PlacedSynthComponent] = []
    skipped: list[dict] = []
    warnings: list[str] = []
    nets: set[str] = set()

    # Resolve symbols up front so the grid step can clear the largest symbol.
    resolved: list[tuple[_SynthInstance, object]] = []
    max_dim = 96
    for inst in instances:
        sym_info = get_symbol_info(inst.symbol)
        if sym_info is None:
            skipped.append(
                {
                    "ref": inst.ref,
                    "reason": (
                        f"symbol '{inst.symbol}' not found in any configured symbol "
                        "library (no LTspice install / symbol paths?)"
                    ),
                }
            )
            continue
        resolved.append((inst, sym_info))
        max_dim = max(max_dim, sym_info.bbox.width, sym_info.bbox.height)

    if not resolved:
        return placed, skipped, warnings, nets

    step = max(192, _round_up_grid(max_dim + 96))
    ncols = max(1, math.ceil(math.sqrt(len(resolved))))
    origin = 128

    coord_owner: dict[tuple[int, int], str] = {}  # (x,y) → net, to catch collisions
    for i, (inst, sym_info) in enumerate(resolved):
        col, row = i % ncols, i // ncols
        x = origin + col * step
        y = origin + row * step
        geo = compute_placed_geometry(sym_info, x, y, "R0")  # type: ignore[arg-type]
        pins = geo["pins"]
        if len(pins) != len(inst.nodes):
            skipped.append(
                {
                    "ref": inst.ref,
                    "reason": (
                        f"symbol '{inst.symbol}' has {len(pins)} pins but the netlist "
                        f"gives {len(inst.nodes)} nodes"
                    ),
                }
            )
            continue
        labels: list[tuple[str, int, int]] = []
        ok = True
        for pin in pins:
            order = pin["order"]
            if not 1 <= order <= len(inst.nodes):
                warnings.append(
                    f"{inst.ref}: pin '{pin['name']}' has SpiceOrder {order} outside "
                    f"1..{len(inst.nodes)}; skipped (symbol/netlist arity mismatch)."
                )
                ok = False
                break
            net = inst.nodes[order - 1]
            px, py = int(pin["x"]), int(pin["y"])
            prior = coord_owner.get((px, py))
            if prior is not None and prior != net:
                warnings.append(
                    f"{inst.ref}: pin coordinate ({px},{py}) already carries net "
                    f"'{prior}' — '{net}' would short to it. Move the component."
                )
                ok = False
                break
            coord_owner[(px, py)] = net
            labels.append((net, px, py))
            nets.add(net)
        if not ok:
            # Roll back the partial coordinate registrations for this rejected
            # component so its pins don't pollute coord_owner and spuriously
            # block a later component that lands on the same coordinate.
            for _net, lx, ly in labels:
                coord_owner.pop((lx, ly), None)
            skipped.append({"ref": inst.ref, "reason": "pin layout collision; not placed"})
            continue
        placed.append(
            _PlacedSynthComponent(
                ref=inst.ref, symbol=inst.symbol, x=x, y=y, value=inst.value, labels=labels
            )
        )

    return placed, skipped, warnings, nets


class SchematicFromNetlistInput(ToolInput):
    name: str = Field(description="Output file name without the .asc extension")
    content: str = Field(
        description=(
            "SPICE netlist text. Supported elements (R/C/L/V/I/D) are placed on a "
            "grid and wired by net label; per SPICE convention the first non-blank "
            "line is treated as the deck title and ignored. Directives (.model, "
            ".tran, .ac, .param, .meas, ...) are carried over verbatim."
        )
    )
    overwrite: bool = Field(
        default=False,
        description="Overwrite an existing file at this path. Default is to refuse.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="schematic_from_netlist",
    description=(
        "Generate an .asc schematic from SPICE netlist text. Parses the netlist, "
        "grid-places each supported component (R/C/L/V/I/D) on its LTspice symbol, "
        "and connects pins by net label (FLAGs carrying the node name) so the "
        "result is electrically identical to the netlist — no manual pin-by-pin "
        "placement. Directives (.model/.tran/.ac/.param/.meas/...) are carried over. "
        "Multi-terminal / controlled / subcircuit elements (M, Q, J, X, E, G, F, H) "
        "can't have their symbol inferred from the instance line and are returned "
        "in ``skipped`` for manual placement. Round-trips through read_circuit. "
        "Connection is label-based, not routed wires, so the layout is functional "
        "rather than pretty."
        " For a circuit whose active/multi-terminal parts land in `skipped` "
        "(E/G/M/Q/X/...), build via add_component + connect or apply_schematic_ops; "
        "see the ltspice://guide resource for layout."
    ),
    input_model=SchematicFromNetlistInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "file": {"type": "string"},
            "placed": {"type": "integer"},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "reference": {"type": "string"},
                        "symbol": {"type": "string"},
                        "position": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "integer"},
                                "y": {"type": "integer"},
                            },
                        },
                        "value": {"type": ["string", "null"]},
                    },
                },
            },
            "skipped": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "ref": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
            "directive_count": {"type": "integer"},
            "nets": {"type": "array", "items": {"type": "string"}},
            "warnings": {"type": "array", "items": {"type": "string"}},
            "validation_warnings": VALIDATION_WARNINGS_SCHEMA,
        },
    },
)
async def handle_schematic_from_netlist(
    args: SchematicFromNetlistInput, state: SessionState
) -> types.CallToolResult:
    """Build an .asc schematic from SPICE netlist text (label-based wiring)."""
    target_path = safe_path(f"{args.name}.asc", state)

    instances, directives, skipped, warnings = _parse_netlist_for_synth(args.content)
    placed, place_skipped, place_warnings, nets = _layout_synth_components(instances)
    skipped = skipped + place_skipped
    warnings = warnings + place_warnings

    if not placed and not directives:
        raise NetlistError(
            "Nothing to place: no supported components (R/C/L/V/I/D) and no "
            "directives were parsed from the netlist. Skipped: "
            + (", ".join(f"{s['ref']} ({s['reason']})" for s in skipped) or "none")
        )

    body = "Version 4\nSHEET 1 880 680\n"
    # Hold the per-path edit lock across the WHOLE existence-check → snapshot →
    # stub-write → populate → save. The lock is non-reentrant, so we inline
    # _editing_asc's save/invalidate contract rather than nest it — otherwise a
    # concurrent edit/reset on the same path could interleave with the overwrite
    # and corrupt the file or the reset snapshot.
    async with _get_edit_lock(target_path):
        pre_existed = target_path.exists()
        if pre_existed:
            # overwrite=true on an existing file: snapshot the ORIGINAL bytes
            # before we clobber them, so reset_schematic restores the
            # pre-synthesis file rather than the blank stub written below.
            _snapshot_asc(target_path, state)
        try:
            atomic_write_text(target_path, body, overwrite=args.overwrite, durable=False)
        except FileExistsError as e:
            raise NetlistError(
                f"File already exists: {target_path}. Pass overwrite=true to replace it."
            ) from e
        # Drop any editor cached from the pre-overwrite content (e.g. a prior
        # read_circuit) so we populate the fresh blank stub, not stale content.
        state.editors.invalidate(target_path)
        try:
            editor = _get_asc_editor(target_path, state)
            for comp in placed:
                _create_component(
                    editor, comp.ref, comp.symbol, comp.x, comp.y, ERotation.R0, value=comp.value
                )
                for net, px, py in comp.labels:
                    editor.labels.append(
                        Text(coord=Point(px, py), text=net, type=TextTypeEnum.LABEL)
                    )
            # Stack directives below the component grid; coordinates are cosmetic.
            dir_y = 128 + (max((c.y for c in placed), default=96) - 96) + 256
            for i, directive in enumerate(directives):
                editor.directives.append(
                    Text(
                        coord=Point(128, dir_y + i * 32),
                        text=directive,
                        type=TextTypeEnum.DIRECTIVE,
                        size=2,
                    )
                )
            validation_warnings = _post_op_warnings(editor)
            _atomic_save_editor(editor, target_path)
        finally:
            state.editors.invalidate(target_path)

    if not pre_existed:
        # A freshly synthesized file has no pre-session state to revert to —
        # drop the blank-stub snapshot so reset_schematic reports reverted=False
        # (matching create_schematic).
        state.asc_snapshots.pop(str(target_path), None)

    data: dict = {
        "file": str(target_path),
        "placed": len(placed),
        "components": [
            {
                "reference": c.ref,
                "symbol": c.symbol,
                "position": {"x": c.x, "y": c.y},
                "value": c.value,
            }
            for c in placed
        ],
        "skipped": skipped,
        "directive_count": len(directives),
        "nets": sorted(nets),
        "warnings": warnings,
    }
    if validation_warnings:
        data["validation_warnings"] = validation_warnings

    lines = [
        f"Created schematic from netlist: {target_path}",
        f"  Placed {len(placed)} component(s), {len(directives)} directive(s), "
        f"{len(nets)} net(s).",
    ]
    if skipped:
        lines.append(f"  Skipped {len(skipped)} element(s):")
        for s in skipped:
            lines.append(f"    {s['ref']}: {s['reason']}")
    for w in warnings:
        lines.append(f"  Warning: {w}")
    lines.extend(_validation_warnings_lines(validation_warnings))
    return format_response("\n".join(lines), data, args.format)


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
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="trace_net",
    description=(
        "Report everything electrically connected to a net: starting from a pin "
        "('Ref.Pin'), a net label ('net:NAME'), or an (x,y) coordinate, return the "
        "net's labels and every component pin, FLAG, and wire vertex on it. "
        "Follows both wires (segment-aware — catches labels placed mid-wire) and "
        "same-name FLAGs (LTspice's name-based nets, as produced by "
        "schematic_from_netlist). Use it to answer 'what's on net X', to confirm "
        "a connect landed, or to spot an accidental short (a net carrying two "
        "different non-ground labels)."
    ),
    input_model=TraceNetInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
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
        },
    },
)
async def handle_trace_net(args: TraceNetInput, state: SessionState) -> types.CallToolResult:
    """Trace every pin/label/wire vertex on the net at a pin, label, or (x,y)."""
    asc_path = safe_path(args.path, state)
    _require_asc(asc_path)
    editor = _get_asc_editor(asc_path, state)

    if args.pin is not None and args.pin.startswith("net:"):
        # A net: reference legitimately matches many same-name FLAGs — the
        # normal case on schematic_from_netlist output (one FLAG per pin).
        # _resolve_pin refuses ambiguous net labels, but trace_net's own
        # name-merge step below absorbs duplicates, so just seed from any
        # matching label coordinate (lowest, for determinism).
        net_name = args.pin[4:]
        matches = sorted(
            (int(lbl.coord.X), int(lbl.coord.Y)) for lbl in editor.labels if lbl.text == net_name
        )
        if not matches:
            raise NetlistError(
                f"Net label '{net_name}' not found in schematic. Add it with "
                "add_net_label first, or trace a component pin / coordinate."
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
            "vertex sits there. Use read_circuit to inspect the layout."
        )

    # The physical partition connects by wire only; LTspice also makes FLAGs
    # with the same NAME electrically common. Fold physical nets that share a
    # label name together (a second union-find over physical roots) so
    # trace_net answers "what's on net X" on label-wired schematics (e.g.
    # schematic_from_netlist output), not just wire-routed ones.
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

    data = {
        "start": {"x": x, "y": y},
        "labels": sorted(labels),
        "pins": pins,
        "coordinates": [{"x": cx, "y": cy} for cx, cy in coords],
        "is_shorted": is_shorted,
    }

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
    return format_response("\n".join(lines), data, args.format)


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
            f"analysis directives are: {active}. The simulator silently "
            f"drops .meas {kind} on non-.{kind} runs, so this measurement "
            "won't appear in the log."
        ),
        "suggestion": suggestion,
    }


def _asc_topology_issues(editor: AscEditor) -> list[dict]:
    """Static schematic-graph checks for ``validate_netlist`` on ``.asc`` files.

    Surfaces the three problems the tool description promises but the directive
    lint cannot see: named-net shorts (one physical net carrying >1 distinct
    non-ground label — the same union-find model ``trace_net`` reports), plus
    floating pins, dangling labels, and duplicate wires (reused from the
    ``_post_op_warnings`` advisory pass). Read-only; deterministic ordering.
    """
    issues: list[dict] = []

    # Named-net short: a single connected net carrying more than one distinct
    # non-ground label. Mirror ``handle_trace_net``'s ``len(named) > 1`` test.
    part = _net_partition(editor)
    labels_by_root: dict[tuple[int, int], set[str]] = {}
    for coord, texts in part.label_texts.items():
        labels_by_root.setdefault(part.root(coord), set()).update(texts)
    shorts: list[str] = []
    for names in labels_by_root.values():
        named = sorted(_named_labels(frozenset(names)))
        if len(named) > 1:
            shorts.append(", ".join(named))
    for joined in sorted(shorts):  # deterministic ordering
        issues.append(
            {
                "severity": "error",
                "line": None,
                "directive": f"net labels: {joined}",
                "message": (
                    f"Named-net short: labels {joined} sit on the same physical "
                    "net — LTspice merges them into one node."
                ),
                "suggestion": (
                    "Separate the wires/labels, or use a single net name for this node."
                ),
            }
        )

    # Floating pins / dangling labels / duplicate wires (all warning-level) reuse
    # the post-op advisory pass; the map both filters to the kinds we surface and
    # supplies each one's fix hint.
    fixes = {
        "floating_pin": "Wire this pin or place a net label on it.",
        "dangling_label": "Move the label onto a wire or a component pin.",
        "duplicate_wire": "Remove the redundant wire segment.",
    }
    for w in _post_op_warnings(editor):
        fix = fixes.get(w.get("kind") or "")
        if fix is None:
            continue
        issues.append(
            {
                "severity": "warning",
                "line": None,
                "directive": "",
                "message": w["message"],
                "suggestion": fix,
            }
        )

    return issues


class ValidateNetlistInput(ToolInput):
    path: str = Field(description="Path to circuit file (.cir, .net, or .asc)")
    target_simulator: Literal["LTspice", "ngspice"] = Field(
        default="LTspice",
        description=(
            "Simulator the deck is intended to run on; selects which simulator's "
            "pre-flight rules apply. 'LTspice' runs the LTspice-specific gates "
            "(more-than-one-analysis rejection, .meas analysis-kind matching, "
            "C=/L= primary-value, viewer-only .meas functions). 'ngspice' skips "
            "those and applies ngspice-only checks instead (a zero '.tran' step "
            "time is rejected). Structural checks (element arity, dangling nodes, "
            "bias topology) apply to both. Defaults to LTspice."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="validate_netlist",
    description=(
        "Lint a netlist or schematic before simulation — the static circuit "
        "check gate. Catches: empty/whitespace-only netlist files, element "
        "arity (too few nodes, missing "
        "E/G/F/H/B value), dangling nodes in .cir/.net netlists (a node "
        "touching only one element "
        "terminal — warning, since deliberate fragments are legal), "
        "bias-topology degeneracies in .cir/.net netlists (a net with no DC "
        "path to ground — floating MOSFET gate, capacitive island, "
        "current-source-only node, or isolated domain — warning, since the "
        "operating point may still be defined by other means), "
        "duplicate/multiple analysis directives ('More than "
        "one analysis specified'), .MEAS whose analysis kind isn't present, "
        "known-bad .MEAS patterns (vdb()/phase()/group_delay()), "
        "and directives the LTspice runner is known to reject (set "
        "target_simulator='ngspice' to instead flag ngspice-only "
        "incompatibilities, e.g. a zero '.tran' step time). On .asc, "
        "also surfaces named-net shorts, "
        "floating pins, and dangling labels. Returns a structured issue list; "
        "an empty list means the file passes the static gate. Note: value "
        "tokens (e.g. a typo'd '1kk') and undefined model references are NOT "
        "checked — LTspice coerces or resolves those at run time."
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

    asc_editor: AscEditor | None = None
    if _is_asc(file_path):
        try:
            asc_editor = _get_asc_editor(file_path, state)
            content = "\n".join(_asc_directive_lines(asc_editor))
        except Exception as e:
            raise NetlistError(f"Failed to open .asc: {e}") from e
    else:
        content = file_path.read_text(encoding="utf-8", errors="replace")

    issues: list[dict] = []
    # Empty / whitespace-only deck: LTspice fails it immediately at line 1.
    # Non-.asc only — the .asc branch builds ``content`` from the schematic's
    # directive lines, and a schematic without SPICE directives is fine.
    if asc_editor is None and not content.strip():
        issues.append(
            {
                "severity": "error",
                "line": 1,
                "directive": "",
                "message": "Netlist is empty — the file has no circuit elements or directives.",
                "suggestion": (
                    "Write the netlist content (title line, elements, analysis "
                    "directive) before simulating."
                ),
            }
        )
    # Single pass: validate directives, collect each analysis directive,
    # and bookmark every ``.meas <kind>`` line.
    analysis_lines: dict[str, list[tuple[int, str]]] = {}
    meas_lines: list[tuple[int, str, str]] = []
    for lineno, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
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
        err = validate_directive(line, simulator=args.target_simulator)
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

    # The multiple-analysis and .meas-kind-matching gates below encode LTspice
    # semantics specifically (its "More than one analysis specified" rejection
    # and its silent-drop of mismatched .meas lines). They are skipped for a
    # non-LTspice target rather than emitting LTspice-worded errors under it; no
    # ngspice equivalent is invented here (ngspice's batch-mode constraints are
    # surfaced at run time by services.ngspice_preflight_warnings).
    if args.target_simulator == "LTspice":
        # LTspice rejects more than one analysis directive with "More than one
        # analysis specified."
        # ``.op`` coexists with one real analysis in LTspice, so count only the
        # mutually-exclusive kinds — counting ``.op`` false-positived the common
        # ``.op`` + ``.tran``/``.ac`` idiom. ``analysis_lines`` is left intact
        # so the ``.meas`` matching below still recognises ``.op``.
        exclusive = {k: v for k, v in analysis_lines.items() if k in EXCLUSIVE_ANALYSIS_KINDS}
        duplicate_kinds = sorted(k for k, entries in exclusive.items() if len(entries) > 1)
        if duplicate_kinds or len(exclusive) > 1:
            issues.append(_multiple_analyses_issue(exclusive, duplicate_kinds))

        # ``.meas <kind>`` runs only when the matching analysis is present.
        # LTspice silently drops mismatched .meas lines from the log.
        active_kinds = analysis_lines.keys()
        for lineno, line, kind in meas_lines:
            if kind in active_kinds:
                continue
            issues.append(_meas_mismatch_issue(lineno, line, kind, active_kinds))

    # Element-arity pass: walk lexer cards, consult ELEMENT_SPECS,
    # flag instances with too few positional nodes or B-sources missing
    # the V=/I= prefix. LTspice's "Expected 2 node names here" / "Unknown
    # parameter" errors at runtime become up-front issues here.
    try:
        arity_cards = lex(content).cards
    except SpiceLexError:
        arity_cards = []

    # Line 1 of a .cir/.net is the free-text title, not an element — drop it
    # before the arity / dangling / bias passes so a title that happens to start
    # with an element letter ("RC filter") isn't lexed as an R-element with too
    # few nodes (and its words don't leak into the dangling-node suppressor set).
    # .asc ``content`` is directive-only (no title line), so it keeps every card.
    lint_cards = drop_title_card(arity_cards) if asc_editor is None else arity_cards
    for arity_issue in validate_netlist_arity(lint_cards, simulator=args.target_simulator):
        issues.append({"severity": "error", **arity_issue})

    # Dangling-node pass: a node touched by exactly one element terminal in
    # its scope. Warning only — single-connection nodes are legal in
    # deliberate fragments. Non-.asc only: a schematic's connectivity lives
    # in wires and flags this pass cannot see, so element lines legally
    # embedded in .asc SPICE-directive text would all false-positive
    # (floating pins are the schematic topology pass's job below).
    if asc_editor is None:
        for dangling_issue in validate_netlist_dangling_nodes(lint_cards):
            issues.append({"severity": "warning", **dangling_issue})
        # Bias-topology pass: a node touched by two or more terminals that
        # still has no DC path to ground (floating gate, capacitive island,
        # current-source-only node, isolated domain). Warning only — a
        # flag is provable, but the deck may bias it by other means. Same
        # non-.asc gate: schematic connectivity lives in wires this pass
        # cannot see.
        for bias_issue in validate_netlist_bias_topology(lint_cards):
            issues.append({"severity": "warning", **bias_issue})

    # .asc schematic-graph checks (named-net shorts, floating pins, dangling
    # labels) — the directive lint above only sees embedded SPICE text, so the
    # topology problems the description promises are added here.
    if asc_editor is not None:
        issues.extend(_asc_topology_issues(asc_editor))

    summary = {"file": str(file_path), "issue_count": len(issues), "issues": issues}
    if not issues:
        return format_response(f"OK: no issues in {file_path.name}", summary, fmt)
    lines = [f"{file_path.name}: {len(issues)} issue(s)"]
    for issue in issues:
        loc = f" line:{issue['line']}" if issue.get("line") else ""
        lines.append(f"  [{issue['severity']}]{loc}: {issue['message']}")
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
    must compare them too — otherwise such an edit reads as 'no differences'
    (F6)."""
    value = str(comp["value"])
    attrs = comp.get("attributes") or {}
    if not attrs:
        return value
    attr_str = "; ".join(f"{k}={attrs[k]}" for k in sorted(attrs))
    return f"{value} | {attr_str}"


def _components_and_directives(path: Path) -> tuple[dict[str, str], set[str]]:
    """Return (components, directive_lines) for a circuit file in one read.

    Reuses ``services.extract_{asc,netlist}_info`` so unparseable component
    values, AscEditor dispatch, and directive collection all flow through the
    canonical path. No second disk read.
    """
    if _is_asc(path):
        try:
            ed = _make_editor(path)
        except Exception:
            return {}, set()
        assert isinstance(ed, AscEditor)
        info = services.extract_asc_info(ed, path)
        components = {comp["reference"]: _component_signature(comp) for comp in info["components"]}
        directives = {d.strip() for d in info.get("directives", []) if d.strip().startswith(".")}
        return components, directives
    try:
        info = services.extract_netlist_info(path)
    except Exception:
        return {}, set()
    components = {comp["reference"]: _component_signature(comp) for comp in info["components"]}
    directives = {
        line.strip()
        for line in info.get("content", "").splitlines()
        if line.strip().startswith(".")
    }
    return components, directives


@registry.tool(
    name="diff_circuit",
    description=(
        "Structural diff between two circuit files: reports added/removed "
        "components, components whose value or attributes "
        "(Value2/SpiceLine/SpiceModel) changed, and added/removed "
        ".PARAM/.MEAS/.MODEL directives. Use after ``set_component_value``, "
        "``set_component_attribute`` or ``edit_directive`` to confirm that the "
        "intended change actually landed."
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
        # Normalize the micro sign for the equality test only — a value LTspice
        # rendered as 1µ must not read as changed vs an authored 1u — while the
        # before/after shown stays exactly what each file contains.
        if _norm_micro(a[ref]) != _norm_micro(b[ref]):
            changed.append({"reference": ref, "before": a[ref], "after": b[ref]})

    # Case-insensitive directive comparison — SPICE directives are
    # case-insensitive, so .end vs .END shouldn't appear as a diff. ``.END`` (the
    # deck terminator) is dropped: a .cir carries one and an exported .asc netlist
    # doesn't, so it would show as a spurious "removed directive". NOT ``.ends``
    # (the subcircuit terminator), which IS meaningful.
    def _by_lower(directives: set[str]) -> dict[str, list[str]]:
        by_lower: dict[str, list[str]] = {}
        for d in directives:
            if d.strip().lower() == ".end":
                continue
            # Key on the micro-normalized lowercase form so ".tran 1µ" and
            # ".tran 1u" collapse; the original string is kept for display.
            by_lower.setdefault(_norm_micro(d.lower()), []).append(d)
        return by_lower

    da_by_lower = _by_lower(da)
    db_by_lower = _by_lower(db)
    directive_added = sorted(
        d for k in db_by_lower.keys() - da_by_lower.keys() for d in db_by_lower[k]
    )
    directive_removed = sorted(
        d for k in da_by_lower.keys() - db_by_lower.keys() for d in da_by_lower[k]
    )

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


def _snap_match(requested: float, actual: float, *, rtol: float = 1e-3) -> bool:
    """True iff ``actual`` is within ``rtol`` (relative) of ``requested``.

    Step axes are discrete, so a legitimate lookup lands on (or extremely
    near) a real step value. A large gap means the request fell outside the
    swept range and was silently clamped to the nearest endpoint — worth a
    warning rather than presenting the clamp as a valid answer.
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
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


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
            idx = (
                ins - 1
                if abs(axis_vals[ins - 1] - target) <= abs(axis_vals[ins] - target)
                else ins
            )
        wave = raw.get_wave(signal, step=0)
        actual = float(axis_vals[idx])
        # This is a continuous native axis (DC sweep variable / AC frequency),
        # not a discrete step list: an off-grid interior request is a normal
        # nearest-neighbour lookup, and only a request beyond the axis ends is
        # genuinely clamped. sample_to_dict keeps complex AC samples intact
        # (magnitude/phase) instead of float() silently dropping the imag part.
        sample_dict = sample_to_dict(wave[idx])
        exact = _snap_match(target, actual)
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

    if not _snap_match(target, best_actual):
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
        "exact_match": _snap_match(target, best_actual),
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


# ---------------------------------------------------------------------------
# Batch-transaction op — apply many edits to one .asc atomically.
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
        text_type = TextTypeEnum.DIRECTIVE if op.kind == "directive" else TextTypeEnum.COMMENT
        _append_asc_text(
            editor, op.instruction, text_type, op.x, op.y, op.size, default_x=16, default_y=16
        )
        return {"op": "add_directive", "instruction": op.instruction}

    raise NetlistError(f"Unknown op type: {type(op).__name__}")


@registry.tool(
    name="apply_schematic_ops",
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
    """Apply a list of schematic edits in one transaction."""
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
        _snapshot_asc(asc_path, state)
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
