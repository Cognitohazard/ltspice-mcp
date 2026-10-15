"""Schematic scene graph: an ``.asc`` as placed, absolute-coordinate
render primitives.

This module is the front half of the ``.asc`` → SVG renderer. It draws a
:class:`ltspice_mcp.lib.sheet.Sheet`, which reads the file through
``asc_document`` and finds each symbol through
:class:`ltspice_mcp.lib.symbol_library.SymbolResolver`, and transforms every
symbol-local coordinate into absolute schematic space using
:func:`ltspice_mcp.lib.symbol_geometry._apply_rotation` (the shared, tested
rotation/mirror machinery — never reimplemented here). A line the sheet's
records do not read is reported in ``Scene.diagnostics`` with its line number.

The output :class:`Scene` holds semantic groups of primitives in stable source
order, plus a content bounding box for cropping and a list of human-readable
diagnostics (e.g. an unresolved symbol). :mod:`schematic_renderer` turns a
``Scene`` into SVG; nothing here emits markup.

Why a render-oriented ``.asy`` parse instead of reusing
``symbol_geometry.parse_asy_file``: that function keeps only pins and a bounding
box and discards the drawing primitives and text — everything a renderer needs.
So :func:`proto_of` keeps the whole reading of a symbol file
(lines/rects/circles/arcs/pins/windows), which ``symbol_library.SymbolResolver``
reads once, while still reusing ``PinInfo`` and the rotation transform.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ltspice_mcp.lib.asc_document import Opaque, Symbol, Window
from ltspice_mcp.lib.connectivity import point_on_segment
from ltspice_mcp.lib.encoding import decode_spice_bytes, read_spice_text, refused_sheet_mark
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.sheet import Sheet, SheetPart, read_sheet_text
from ltspice_mcp.lib.sheet_findings import Part, SheetView
from ltspice_mcp.lib.symbol_file import (
    PinInfo,
    SymbolArc,
    read_symbol,
)
from ltspice_mcp.lib.symbol_geometry import (
    _apply_rotation,  # pyright: ignore[reportPrivateUsage]  # shared rotation/mirror transform
    library_roots,
)
from ltspice_mcp.lib.symbol_library import FoundSymbol, SymbolResolver

# Attribute-window numbers we render as text next to a symbol. LTspice assigns
# window 0 to the instance name and window 3 to the value; the rest (SpiceLine,
# SpiceModel, …) are omitted from V1 rendering.
_WINDOW_INSTNAME = 0
_WINDOW_VALUE = 3

# Fixed placeholder-box extent (symbol-local) for an unresolved symbol. The
# real body is unknown, so a stable default box is drawn at the placement
# origin and flagged as a diagnostic.
_PLACEHOLDER = BBox(0, 0, 64, 48)

# Number of straight segments used to approximate an arc as a polyline. Chosen
# so transforms (including mirror) apply to sampled points rather than to SVG
# arc-sweep flags, which keeps arc rendering deterministic under rotation.
_ARC_SEGMENTS = 24

# --- Text metrics, shared with schematic_renderer ---------------------------
# LTspice stores a text *size index*; map it to a pixel height. Index 2 is the
# schematic default.
FONT_PX = {0: 8, 1: 11, 2: 14, 3: 18, 4: 24, 5: 32, 6: 42, 7: 56}
DEFAULT_FONT_PX = 14

# Extra pixels between stacked lines of a multi-line text block.
LINE_GAP = 2

# Placement of a net-label glyph relative to its FLAG coordinate, and the
# ground glyph's extent below it. Shared so the crop box and the drawn output
# cannot drift apart.
FLAG_LABEL_DY = 4
FLAG_LABEL_SIZE = 1
GND_HALF_W = 6
GND_DEPTH = 10

# Side of the small square LTspice draws at a flag that connects to nothing.
UNCONNECTED_MARKER = 4

# Glyph-extent approximation. Text is drawn in a monospace family, so a line's
# width is estimated as (character count x font height x _ADVANCE_RATIO) and its
# vertical span as one ascent above the baseline plus one descent below. This is
# deliberately an estimate — measuring real advances would need font metrics we
# do not carry — and it feeds only the crop box, where running a few pixels wide
# is harmless while running narrow clips visible text.
_ADVANCE_RATIO = 0.62
_ASCENT_RATIO = 1.0
_DESCENT_RATIO = 0.3


def font_px(size: int) -> int:
    """Pixel height for an LTspice text-size index."""
    return FONT_PX.get(size, DEFAULT_FONT_PX)


def decode_text_lines(raw: str) -> list[str]:
    r"""Split an LTspice TEXT body into display lines.

    LTspice encodes an embedded newline as the two characters ``\`` ``n`` and
    escapes a literal backslash as ``\\``. The two must be decoded together: a
    naive split on ``\n`` would cut a Windows path such as
    ``C:\\new_models\\x.lib`` at the ``\n`` inside its escaped separator.
    """
    lines: list[str] = []
    buf: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch == "\\" and i + 1 < n:
            nxt = raw[i + 1]
            if nxt == "n":
                lines.append("".join(buf))
                buf.clear()
                i += 2
                continue
            if nxt == "\\":
                buf.append("\\")
                i += 2
                continue
        buf.append(ch)
        i += 1
    lines.append("".join(buf))
    return lines


# Which way a flag glyph points, given the direction its wire extends *away*
# from the connection point. The glyph continues the wire's direction of travel,
# i.e. it is drawn on the far side of the point from the wire.
#
# Verified against LTspice itself: a reference schematic isolates a ground and a
# net label approached from each of N/S/E/W plus the wireless and corner cases,
# and the rendered result was compared case by case with a screenshot of the
# same file open in LTspice. All four single-wire directions match this mapping
# for both flag kinds.
_GLYPH_AWAY_FROM = {"up": "down", "down": "up", "left": "right", "right": "left"}

# Resting order used when the wire geometry does not determine a side. The first
# entry is the verified no-wire placement for that flag kind.
_GROUND_PREFERENCE = ("down", "up", "left", "right")
_LABEL_PREFERENCE = ("up", "right", "left", "down")


def wire_directions_at(wires: Sequence[Wire], x: int, y: int) -> set[str]:
    """Directions in which a wire extends *away* from the point ``(x, y)``.

    A wire ending on the point claims the one direction it runs off in. A wire
    passing *through* the point claims both directions of its axis: a flag
    dropped mid-span has wire on either side of it, and a glyph drawn along
    that axis would sit on top of the wire.
    """
    out: set[str] = set()
    point = (x, y)
    for w in wires:
        a, b = (w.x1, w.y1), (w.x2, w.y2)
        if a == point:
            dx, dy = b[0] - a[0], b[1] - a[1]
        elif b == point:
            dx, dy = a[0] - b[0], a[1] - b[1]
        elif point_on_segment(point, a, b):
            # Strictly interior to the segment: both ways along its axis.
            if abs(b[0] - a[0]) >= abs(b[1] - a[1]):
                out.update(("left", "right"))
            else:
                out.update(("up", "down"))
            continue
        else:
            continue
        if dx == 0 and dy == 0:
            continue
        if abs(dx) >= abs(dy):
            out.add("right" if dx > 0 else "left")
        else:
            out.add("down" if dy > 0 else "up")
    return out


def resolve_flag_placement(
    is_ground: bool, occupied: Sequence[str] | set[str]
) -> tuple[str, bool]:
    """Return ``(orientation, text_vertical)`` for a flag.

    ``orientation`` is the side the glyph extends toward — the ground triangle's
    apex, or the side a net label's text sits on. ``text_vertical`` is True when
    a net label's text is rotated a quarter turn to run along the wire axis.

    Verified against an LTspice screenshot of the reference schematic: with a
    single attached wire the glyph always continues the wire's travel, and a net
    label on a *vertical* wire has its text rotated to read bottom-to-top while
    one on a horizontal wire stays horizontal.

    Two cases are **not** verified and are documented fallbacks, because that
    schematic yielded only one sample of each:

    * More than one wire — a corner, or a flag dropped mid-span — where no
      single wire determines a side. The code scans for the first side no wire
      occupies, in resting-first order. For the one corner sample available (a
      wire from the west meeting one running south) that yields a horizontal
      label above the point, which is what LTspice drew; note the scan can give
      a *ground* at that same corner "up" rather than its usual "down", a case
      the reference schematic does not cover. So: one observed sample, plus a
      collision-avoidance rule of ours — not an established LTspice rule.
    * The rotated text read bottom-to-top in both vertical samples available, so
      that is implemented unconditionally; whether LTspice ever flips the
      reading direction by side is untested.
    """
    dirs = set(occupied)
    if len(dirs) == 1:
        wire_side = next(iter(dirs))
        return _GLYPH_AWAY_FROM[wire_side], wire_side in ("up", "down")
    # No wire, or an ambiguous junction (a corner, or a flag mid-span). Take the
    # first side that no wire occupies, starting from the resting orientation —
    # with nothing attached that yields the verified wireless placement, and at a
    # junction it at least keeps the glyph off the wire.
    preference = _GROUND_PREFERENCE if is_ground else _LABEL_PREFERENCE
    for side in preference:
        if side not in dirs:
            return side, False
    return preference[0], False


def ground_polygon(flag: NetFlag) -> tuple[tuple[int, int], ...]:
    """The three corners of a ground glyph: a base through the connection
    point and an apex ``GND_DEPTH`` away along ``flag.orientation``."""
    x, y, o = flag.x, flag.y, flag.orientation
    if o in ("down", "up"):
        sign = 1 if o == "down" else -1
        return ((x - GND_HALF_W, y), (x + GND_HALF_W, y), (x, y + sign * GND_DEPTH))
    sign = 1 if o == "right" else -1
    return ((x, y - GND_HALF_W), (x, y + GND_HALF_W), (x + sign * GND_DEPTH, y))


def flag_label_anchor(flag: NetFlag) -> tuple[int, int, str, int]:
    """Anchor ``(x, y, svg_text_anchor, rotation_degrees)`` for a net label.

    The text is pushed ``FLAG_LABEL_DY`` clear of the connection point on the
    side given by ``flag.orientation``. When ``flag.text_vertical`` is set the
    text is rotated a quarter turn anticlockwise so it reads bottom-to-top; the
    anchor end is then chosen so the string still grows away from the point
    (SVG rotates the advance direction with the glyphs, so "start" grows upward
    and "end" grows downward once rotated).
    """
    x, y, o = flag.x, flag.y, flag.orientation
    px = font_px(FLAG_LABEL_SIZE)
    if flag.text_vertical:
        # Rotated: the glyph band grows to the left of the anchor, so shift
        # right by a third of the height to straddle the wire.
        if o == "down":
            return (x + px // 3, y + FLAG_LABEL_DY, "end", -90)
        return (x + px // 3, y - FLAG_LABEL_DY, "start", -90)
    if o == "up":
        return (x, y - FLAG_LABEL_DY, "middle", 0)
    if o == "down":
        return (x, y + FLAG_LABEL_DY + px, "middle", 0)
    # Sideways: nudge the baseline down by a third of the height so the glyph
    # box straddles the connection point rather than sitting on top of it.
    if o == "left":
        return (x - FLAG_LABEL_DY, y + px // 3, "end", 0)
    return (x + FLAG_LABEL_DY, y + px // 3, "start", 0)


def flag_label_bbox(flag: NetFlag) -> BBox:
    """Estimated extent of a net label's drawn text, rotation included."""
    ax, ay, anchor, rotation = flag_label_anchor(flag)
    if rotation == 0:
        return text_extent(ax, ay, [flag.text], FLAG_LABEL_SIZE, anchor)
    return turned_text_extent(ax, ay, flag.text, FLAG_LABEL_SIZE, anchor)


def turned_text_extent(x: int, y: int, text: str, size: int, anchor: str) -> BBox:
    """Extent of text given a quarter turn anticlockwise about ``(x, y)``.

    The advance runs along y and the glyph height along x. SVG rotates the run
    with the glyphs, so a ``start`` anchor grows upward and an ``end`` anchor
    downward.
    """
    px = font_px(size)
    run = round(len(text) * px * _ADVANCE_RATIO)
    if anchor == "start":
        y1, y2 = y - run, y
    elif anchor == "end":
        y1, y2 = y, y + run
    else:
        y1, y2 = y - run // 2, y + (run - run // 2)
    return BBox(x - round(px * _ASCENT_RATIO), y1, x + round(px * _DESCENT_RATIO), y2)


def text_extent(x: int, y: int, lines: Sequence[str], size: int, anchor: str) -> BBox:
    """Estimated bounding box of a rendered text block anchored at ``(x, y)``.

    ``(x, y)`` is the first line's baseline/anchor point, matching SVG. See the
    ratio constants above for the approximation used.
    """
    px = font_px(size)
    longest = max((len(ln) for ln in lines), default=0)
    width = round(longest * px * _ADVANCE_RATIO)
    if anchor == "middle":
        x1, x2 = x - width // 2, x + (width - width // 2)
    elif anchor == "end":
        x1, x2 = x - width, x
    else:
        x1, x2 = x, x + width
    top = y - round(px * _ASCENT_RATIO)
    bottom = y + round(px * _DESCENT_RATIO) + (len(lines) - 1) * (px + LINE_GAP)
    return BBox(x1, top, x2, bottom)


# ---------------------------------------------------------------------------
# Absolute-coordinate render primitives (consumed by schematic_renderer)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawLine:
    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class DrawRect:
    """Axis-aligned rectangle (already reduced to its AABB after transform)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class DrawEllipse:
    """Axis-aligned ellipse given by its bounding box."""

    x1: int
    y1: int
    x2: int
    y2: int

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x1, self.y1), (self.x2, self.y2))


@dataclass(frozen=True)
class DrawPolyline:
    """Open polyline (arcs are approximated this way)."""

    pts: tuple[tuple[int, int], ...]

    def points(self) -> tuple[tuple[int, int], ...]:
        return self.pts


@dataclass(frozen=True)
class DrawText:
    x: int
    y: int
    text: str
    anchor: str  # "start" | "middle" | "end"
    size: int
    role: str  # "attr" | "flag" | "directive" | "comment" | "placeholder"
    # Degrees about the anchor; negative is anticlockwise. -90 is the quarter
    # turn LTspice uses for the attribute text of a sideways symbol.
    rotation: int = 0

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),)


def drawn_text_extent(t: DrawText) -> BBox:
    """Estimated extent of a single drawn text, upright or turned.

    The one place that picks between the two extent estimates, so a caller
    reasoning about where a text lands cannot pick differently from the crop.
    """
    if t.rotation:
        return turned_text_extent(t.x, t.y, t.text, t.size, t.anchor)
    return text_extent(t.x, t.y, [t.text], t.size, t.anchor)


@dataclass(frozen=True)
class DrawPin:
    x: int
    y: int
    name: str = ""

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),)


Graphic = DrawLine | DrawRect | DrawEllipse | DrawPolyline


# ---------------------------------------------------------------------------
# Symbol prototype (parsed .asy body, symbol-local coordinates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolProto:
    """Parsed ``.asy`` symbol body in symbol-local coordinates."""

    name: str
    path: Path
    lines: tuple[DrawLine, ...]
    rects: tuple[DrawRect, ...]
    circles: tuple[DrawEllipse, ...]
    arcs: tuple[SymbolArc, ...]
    pins: tuple[PinInfo, ...]
    windows: tuple[Window, ...]
    bbox: BBox
    body: BBox | None

    def window(self, number: int) -> Window | None:
        for w in self.windows:
            if w.number == number:
                return w
        return None


# ---------------------------------------------------------------------------
# Scene (absolute coordinates, ready to render)
# ---------------------------------------------------------------------------


@dataclass
class PlacedSymbol:
    reference: str
    symbol: str
    x: int
    y: int
    rotation: str
    resolved_path: Path | None
    missing: bool
    #: The first pin line of a found symbol that does not read, which makes the
    #: symbol unusable; the part is then drawn as a placeholder.
    unreadable: str = ""
    #: The part's extent with its pins, and of what it draws alone: its
    #: symbol's two boxes as placed. A part whose symbol was not found has the
    #: placeholder it is drawn as for both.
    box: BBox | None = None
    body: BBox | None = None
    graphics: list[Graphic] = field(default_factory=list)
    pins: list[DrawPin] = field(default_factory=list)
    texts: list[DrawText] = field(default_factory=list)


@dataclass(frozen=True)
class Wire:
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class NetFlag:
    x: int
    y: int
    text: str
    is_ground: bool
    # Direction the glyph extends away from the connection point: the ground
    # triangle's apex, or the side the net-label text sits on. Derived from the
    # attached wires by :func:`resolve_flag_placement`.
    orientation: str = "down"
    # Net label only: text is rotated a quarter turn to run along a vertical wire.
    text_vertical: bool = False
    # Nothing meets this flag's coordinate — LTspice marks that with a small
    # square at the point, and so do we.
    unconnected: bool = False


@dataclass(frozen=True)
class Directive:
    """A sheet ``TEXT`` record: SPICE directive (``!``) or comment (``;``)."""

    x: int
    y: int
    text: str
    align: str
    size: int
    is_directive: bool


@dataclass
class Scene:
    source: Path
    #: SHA-256 of the bytes this scene was parsed from, so anything reporting
    #: what it drew reports the revision it actually read rather than whatever
    #: a later re-read of the path would find. ``None`` when the scene was not
    #: built from a file.
    source_sha256: str | None = None
    symbols: list[PlacedSymbol] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    flags: list[NetFlag] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    sheet_graphics: list[Graphic] = field(default_factory=list)
    diagnostics: list[str] = field(default_factory=list)
    #: The byte order mark the file starts with, when it is one neither LTspice
    #: build reads a sheet behind (``encoding.refused_sheet_mark``). The drawing
    #: decodes past it, so the scene keeps it here.
    byte_order_mark: str | None = None

    def content_bbox(self) -> BBox | None:
        """Smallest box enclosing everything drawn, including text and glyphs.

        Geometry contributes exact points; text contributes an *estimated*
        extent (see :func:`text_extent`) so a long attribute value or net label
        is not cropped away, and the ground glyph contributes the box it
        actually occupies below its flag coordinate.
        """
        pts: list[tuple[int, int]] = []
        boxes: list[BBox] = []
        for sym in self.symbols:
            for g in sym.graphics:
                pts.extend(g.points())
            for p in sym.pins:
                pts.extend(p.points())
            for t in sym.texts:
                boxes.append(drawn_text_extent(t))
        for w in self.wires:
            pts.append((w.x1, w.y1))
            pts.append((w.x2, w.y2))
        for fl in self.flags:
            pts.append((fl.x, fl.y))
            if fl.unconnected:
                half = UNCONNECTED_MARKER // 2
                pts.append((fl.x - half, fl.y - half))
                pts.append((fl.x + half, fl.y + half))
            if fl.is_ground:
                pts.extend(ground_polygon(fl))
            elif fl.text:
                boxes.append(flag_label_bbox(fl))
        for d in self.directives:
            boxes.append(text_extent(d.x, d.y, decode_text_lines(d.text), d.size, "start"))
        for g in self.sheet_graphics:
            pts.extend(g.points())

        merged = BBox.from_points(pts)
        for b in boxes:
            merged = b if merged is None else merged.union(b)
        return merged


# ---------------------------------------------------------------------------
# .asy parsing (render-oriented)
# ---------------------------------------------------------------------------


def proto_of(found: FoundSymbol) -> SymbolProto:
    """A symbol the resolver read, as the drawing places it.

    The reading is ``symbol_file.read_symbol``, the one the schematic editor's
    pin geometry comes from; a line that does not read is left out of the
    drawing.
    """
    symbol = found.file
    return SymbolProto(
        name=found.name,
        path=found.path,
        lines=tuple(DrawLine(*box) for box in symbol.lines),
        rects=tuple(DrawRect(*box) for box in symbol.rects),
        circles=tuple(DrawEllipse(*box) for box in symbol.circles),
        arcs=symbol.arcs,
        pins=symbol.pins,
        windows=symbol.windows,
        bbox=symbol.bbox,
        body=symbol.body,
    )


def parse_symbol(asy_path: Path, name: str) -> SymbolProto:
    """Read the ``.asy`` file at ``asy_path`` and draw it as :func:`proto_of` does.

    Reads via ``read_spice_text`` (BOM/UTF-16/cp1252 fallback): vendor symbols
    routinely carry cp1252 bytes in description fields.
    """
    return proto_of(FoundSymbol(name, asy_path, read_symbol(read_spice_text(asy_path))))


def editor_symbol_resolver(asc_path: Path) -> SymbolResolver:
    """A resolver that finds a symbol where the schematic editor's pin geometry does.

    The sheet's own folder, then ``symbol_geometry.library_roots``, which is
    what ``symbol_geometry.get_symbol_info`` searches, and nothing more. A
    scene built with it has a part's pins exactly when the editor has them, so
    what is said of a sheet from its scene and what the editor counts on it
    are about the same parts.
    """
    return SymbolResolver(local_dir=asc_path.parent, project_paths=library_roots())


def default_stock_paths() -> list[Path]:
    """Best-effort stock LTspice symbol directories (may be empty).

    Used by the CLI so a schematic that references stock symbols renders their
    bodies when a library is installed. Never required: tests pass their own
    local ``.asy`` fixtures and do not depend on a stock install.
    """
    paths: list[Path] = []
    try:
        from ltspice_mcp.lib.wsl import get_ltspice_lib_paths, is_wsl

        if is_wsl():
            paths.extend(Path(p) for p in get_ltspice_lib_paths())
    except Exception:
        pass
    env = os.environ.get("LTSPICE_MCP_SYMBOL_PATHS")
    if env:
        paths.extend(Path(p) for p in env.split(os.pathsep) if p)
    return [p for p in paths if p.is_dir()]


# ---------------------------------------------------------------------------
# .asc parsing → raw records
# ---------------------------------------------------------------------------


# The records an LTspice sheet holds that the drawing has nothing to take from:
# the header, a port's direction (its label is the FLAG before it) and a data
# label's expression (recorded as export/data_flags: the export is the circuit
# without them). Any other keyword it does not read is reported, a bus tap
# among them: LTspice draws one and this drawing does not, though it connects
# nothing in the netlist (export/bus_tap).
_UNDRAWN_KEYWORDS = frozenset({"Version", "SHEET", "IOPIN", "DATAFLAG"})


# ---------------------------------------------------------------------------
# Placement: symbol-local geometry → absolute scene primitives
# ---------------------------------------------------------------------------


def _place_point(px: int, py: int, ox: int, oy: int, rot: str) -> tuple[int, int]:
    rx, ry = _apply_rotation(px, py, rot)
    return (ox + rx, oy + ry)


def _place_aabb(
    x1: int, y1: int, x2: int, y2: int, ox: int, oy: int, rot: str
) -> tuple[int, int, int, int]:
    """Transform a local axis-aligned box; return the absolute AABB.

    Rects and ellipses stay axis-aligned under LTspice's 90°/mirror transforms,
    so transforming the four corners and taking their AABB is exact.
    """
    corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
    placed = [_place_point(cx, cy, ox, oy, rot) for cx, cy in corners]
    bb = BBox.from_points(placed)
    assert bb is not None  # four corners always non-empty
    return (bb.x1, bb.y1, bb.x2, bb.y2)


def _place_box(box: BBox, ox: int, oy: int, rot: str) -> BBox:
    return BBox(*_place_aabb(box.x1, box.y1, box.x2, box.y2, ox, oy, rot))


def _arc_polyline(arc: SymbolArc, ox: int, oy: int, rot: str) -> DrawPolyline | None:
    """Sample an arc into an absolute-coordinate polyline.

    Sampling in symbol-local space and transforming each point means mirror and
    rotation are handled by the shared transform — no SVG sweep-flag reasoning.
    Degenerate (zero-area) ellipses are skipped. Which way an arc turns is
    ``SymbolArc.sweep``'s to say.
    """
    swept = arc.sweep()
    if swept is None:
        return None
    start, turn = swept
    pts: list[tuple[int, int]] = []
    for k in range(_ARC_SEGMENTS + 1):
        lx, ly = arc.at(start + turn * (k / _ARC_SEGMENTS))
        pts.append(_place_point(round(lx), round(ly), ox, oy, rot))
    return DrawPolyline(tuple(pts))


def _place_symbol(part: SheetPart, proto: SymbolProto | None) -> PlacedSymbol:
    ref = part.reference
    (ox, oy), rot = part.at, part.rotation

    if proto is None:
        placed = PlacedSymbol(
            reference=ref,
            symbol=part.symbol_name,
            x=ox,
            y=oy,
            rotation=rot,
            resolved_path=part.symbol.path if part.symbol is not None else None,
            missing=True,
            unreadable=part.unreadable,
        )
        x1, y1, x2, y2 = _place_aabb(
            _PLACEHOLDER.x1, _PLACEHOLDER.y1, _PLACEHOLDER.x2, _PLACEHOLDER.y2, ox, oy, rot
        )
        placed.graphics.append(DrawRect(x1, y1, x2, y2))
        placed.box = placed.body = BBox(x1, y1, x2, y2)
        label = ref or part.symbol_name
        placed.texts.append(
            DrawText(
                x=(x1 + x2) // 2,
                y=(y1 + y2) // 2,
                text=label,
                anchor="middle",
                size=2,
                role="placeholder",
            )
        )
        return placed

    placed = PlacedSymbol(
        reference=ref,
        symbol=part.symbol_name,
        x=ox,
        y=oy,
        rotation=rot,
        resolved_path=proto.path,
        missing=False,
        box=_place_box(proto.bbox, ox, oy, rot),
        body=_place_box(proto.body, ox, oy, rot) if proto.body is not None else None,
    )
    for ln in proto.lines:
        ax1, ay1 = _place_point(ln.x1, ln.y1, ox, oy, rot)
        ax2, ay2 = _place_point(ln.x2, ln.y2, ox, oy, rot)
        placed.graphics.append(DrawLine(ax1, ay1, ax2, ay2))
    for rc in proto.rects:
        x1, y1, x2, y2 = _place_aabb(rc.x1, rc.y1, rc.x2, rc.y2, ox, oy, rot)
        placed.graphics.append(DrawRect(x1, y1, x2, y2))
    for ci in proto.circles:
        x1, y1, x2, y2 = _place_aabb(ci.x1, ci.y1, ci.x2, ci.y2, ox, oy, rot)
        placed.graphics.append(DrawEllipse(x1, y1, x2, y2))
    for ar in proto.arcs:
        poly = _arc_polyline(ar, ox, oy, rot)
        if poly is not None:
            placed.graphics.append(poly)
    for pin in proto.pins:
        ax, ay = _place_point(pin.x, pin.y, ox, oy, rot)
        placed.pins.append(DrawPin(ax, ay, pin.name))

    placed.texts.extend(_attr_texts(part, proto, ox, oy, rot))
    return placed


def _attr_texts(part: SheetPart, proto: SymbolProto, ox: int, oy: int, rot: str) -> list[DrawText]:
    """Instance-name and value text at their window anchors.

    Window anchor coordinates are symbol-local: a per-instance ``WINDOW``
    override in the ``.asc`` replaces the ``.asy`` default, then the anchor is
    rotated and offset to the placement origin. Attributes without a resolvable
    window and without a value are omitted.

    The glyphs turn with the symbol, not just the anchor — see
    :func:`_attr_text_placement`.
    """
    out: list[DrawText] = []
    overrides = {w.number: w for w in part.record.windows}

    def placed(number: int, text: str) -> DrawText | None:
        w = overrides.get(number) or proto.window(number)
        if w is None:
            return None
        ax, ay = _place_point(w.x, w.y, ox, oy, rot)
        turn, anchor = _attr_text_placement(w.align, rot, explicit=number in overrides)
        return DrawText(ax, ay, text, anchor, w.size, "attr", rotation=turn)

    for number, text in (
        (_WINDOW_INSTNAME, part.reference),
        (_WINDOW_VALUE, part.attribute("Value") or ""),
    ):
        if not text:
            continue
        drawn = placed(number, text)
        if drawn is not None:
            out.append(drawn)

    return out


# The four placements whose symbol axes are swapped relative to the sheet, and
# which way the symbol's local +x axis — the direction its attribute text runs —
# then points on the sheet. The upright four are absent because their text stays
# horizontal, so there is no run direction to derive: R180 reverses that axis and
# M0 mirrors it, but LTspice draws their text left-to-right all the same, so the
# axis rule below is scoped to the turns listed here. The direction is read off
# the shared transform rather than restated, so it cannot disagree with where
# the same placement puts the pins.
_QUARTER_TURNS = {
    rot: "down" if _apply_rotation(1, 0, rot)[1] > 0 else "up"
    for rot in ("R90", "R270", "M90", "M270")
}


def _attr_text_placement(align: str, rot: str, *, explicit: bool) -> tuple[int, str]:
    """Rotation in degrees and SVG ``text-anchor`` for a symbol attribute.

    LTspice draws the instance name and value of a sideways symbol sideways too,
    and says so with a ``V``-prefixed justification. Two things make the text
    turn: such a token, from either the ``.asc`` override or the ``.asy``
    default, and — for a plain default only — the placement's own rotation. A
    default is written for the upright symbol and cannot know how the instance
    was turned, whereas a plain token on an explicit per-instance window is the
    author saying they want it horizontal, so only the former is overridden.

    Leaving a turned symbol's text horizontal is what collapses the two windows
    of a stacked-attribute symbol onto one baseline: the defaults separate them
    along the symbol's local y, and a quarter turn maps that separation onto the
    direction the text runs, so a value longer than the gap prints straight
    through the instance name.

    Which way the run then grows is taken from the placement, not from the rest
    of the token: it follows the symbol's own +x axis after the turn, so a window
    placed beyond the upright symbol's edge stays beyond the turned one's. What
    ``VTop``/``VBottom``/``VLeft``/``VRight`` each mean for run direction is not
    decoded — deriving it from the token instead was tried and put the text back
    through the bodies on LTspice's own example schematics. The token still
    chooses the anchor end where nothing is derived, i.e. on upright placements.
    Glyphs read bottom-to-top in every case, matching net labels on vertical
    wires.
    """
    anchor = _svg_anchor(align)
    grows = _QUARTER_TURNS.get(rot)
    if align.lower().startswith("v") or (not explicit and grows is not None):
        if grows == "down":
            # A quarter turn anticlockwise makes an SVG run grow upward, so the
            # opposite anchor end is the one that sends it down the sheet.
            anchor = {"start": "end", "end": "start"}.get(anchor, anchor)
        return -90, anchor
    return 0, anchor


def _svg_anchor(align: str) -> str:
    """Map an LTspice justification to an SVG ``text-anchor``.

    Only the horizontal component maps. A ``V`` prefix says the text is turned,
    which is a rotation rather than an anchor, so it is stripped here and decided
    by :func:`_attr_text_placement` — ``VRight`` must anchor like ``Right``, not
    fall through to the default the way it would if the prefix were left on.
    Vertical-alignment tokens (Top/Bottom) name no horizontal component and take
    the left-aligned default.
    """
    a = align.lower().removeprefix("v")
    if a == "right":
        return "end"
    if a == "center":
        return "middle"
    return "start"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _unread_note(keyword: str, lines: list[int]) -> str:
    """What the drawing, and every check read from it, leaves out of a sheet."""
    if len(lines) == 1:
        return (
            f"line {lines[0]}: a {keyword} record, which the drawing does not read; it is "
            "not drawn and no check built on the drawing includes it"
        )
    return (
        f"{len(lines)} {keyword} records, the first at line {lines[0]}, which the drawing "
        "does not read; they are not drawn and no check built on the drawing includes them"
    )


def build_scene(asc_path: Path, resolver: SymbolResolver | None = None) -> Scene:
    """Parse ``asc_path`` and build a fully-placed :class:`Scene`.

    If ``resolver`` is omitted, one is created rooted at the schematic's own
    directory plus best-effort stock library paths.
    """
    asc_path = Path(asc_path)
    if resolver is None:
        resolver = SymbolResolver(local_dir=asc_path.parent, stock_paths=default_stock_paths())

    # One read: the digest and the parse describe the same bytes, so a peer
    # that commits between them cannot make the reported provenance a hash of
    # something that was never drawn.
    data = asc_path.read_bytes()
    return scene_of_text(
        decode_spice_bytes(data),
        asc_path,
        resolver,
        hashlib.sha256(data).hexdigest(),
        refused_sheet_mark(data),
    )


def scene_of_text(
    text: str,
    source: Path,
    resolver: SymbolResolver,
    source_sha256: str | None = None,
    byte_order_mark: str | None = None,
) -> Scene:
    """The fully-placed :class:`Scene` of a sheet given as text.

    ``source`` is the path the sheet has or will have; nothing is read from
    it. This is how a sheet that is not on disk yet is drawn and checked: the
    text an edit is about to write. ``byte_order_mark`` is the mark the sheet's
    bytes began with, for a sheet read from a file.
    """
    return scene_of_sheet(read_sheet_text(text, resolver), source, source_sha256, byte_order_mark)


def _unread_lines(sheet: Sheet) -> dict[str, list[int]]:
    """Each keyword of a line the drawing does not read, with the lines it is on.

    A line no record type reads, and a line under a part that does not read as
    a window or an attribute. The records with nothing to draw are left out
    (``_UNDRAWN_KEYWORDS``), and so are blank lines.
    """
    unread: dict[str, list[int]] = {}
    line = 1
    for record in sheet.document.records:
        lines = record.source or ()
        if isinstance(record, Opaque):
            if record.keyword and record.keyword not in _UNDRAWN_KEYWORDS:
                unread.setdefault(record.keyword, []).append(line)
        elif isinstance(record, Symbol) and record.unread:
            pending = list(record.unread)
            for offset, text in enumerate(lines[1:], 1):
                if pending and text.rstrip("\r\n") == pending[0]:
                    keyword = pending.pop(0).split(None, 1)[0]
                    unread.setdefault(keyword, []).append(line + offset)
        line += len(lines)
    return unread


def scene_of_sheet(
    sheet: Sheet,
    source: Path,
    source_sha256: str | None = None,
    byte_order_mark: str | None = None,
) -> Scene:
    """The fully-placed :class:`Scene` of ``sheet``, which is or will be at ``source``."""
    scene = Scene(source=source, source_sha256=source_sha256, byte_order_mark=byte_order_mark)
    for keyword, lines in _unread_lines(sheet).items():
        scene.diagnostics.append(_unread_note(keyword, lines))

    for part in sheet.parts:
        proto = proto_of(part.symbol) if part.usable and part.symbol is not None else None
        instance = f" (instance {part.reference})" if part.reference else ""
        if part.symbol is None:
            scene.diagnostics.append(
                f"symbol '{part.symbol_name}'{instance} not found in any library path; "
                "placeholder rendered"
            )
        elif part.unreadable:
            scene.diagnostics.append(
                f"symbol '{part.symbol_name}'{instance} has a pin line that does not read "
                f"({part.unreadable!r}); placeholder rendered"
            )
        scene.symbols.append(_place_symbol(part, proto))

    wires = [Wire(*wire) for wire in sheet.wires]
    scene.wires.extend(wires)
    # Placement needs the wire set, so resolve it once the wires are known.
    # A flag counts as connected if a wire ends on it, a wire runs through it,
    # or a symbol pin sits on it; only a truly isolated flag gets the marker.
    pin_points = {(p.x, p.y) for sym in scene.symbols for p in sym.pins}
    segments = [((w.x1, w.y1), (w.x2, w.y2)) for w in wires]
    flags = [NetFlag(x, y, name, is_ground=(name.strip() == "0")) for x, y, name in sheet.labels]
    for fl in flags:
        occupied = wire_directions_at(wires, fl.x, fl.y)
        orientation, text_vertical = resolve_flag_placement(fl.is_ground, occupied)
        connected = (
            bool(occupied)
            or (fl.x, fl.y) in pin_points
            or any(point_on_segment((fl.x, fl.y), a, b) for a, b in segments)
        )
        scene.flags.append(
            replace(
                fl,
                orientation=orientation,
                text_vertical=text_vertical,
                unconnected=not connected,
            )
        )
    scene.directives.extend(
        Directive(t.x, t.y, t.body, t.align, t.size, t.is_directive) for t in sheet.document.texts
    )
    # Lines, then rectangles, then circles, then arcs: the order they are drawn in.
    shapes = sheet.document.shapes
    for kind, draw in (("LINE", DrawLine), ("RECTANGLE", DrawRect), ("CIRCLE", DrawEllipse)):
        scene.sheet_graphics.extend(draw(*shape.coords) for shape in shapes if shape.kind == kind)
    for shape in shapes:
        if shape.kind == "ARC":
            poly = _arc_polyline(SymbolArc(*shape.coords), 0, 0, "R0")
            if poly is not None:
                scene.sheet_graphics.append(poly)

    return scene


# ---------------------------------------------------------------------------
# The sheet as the shared checks read it
# ---------------------------------------------------------------------------


def sheet_view(scene: Scene) -> SheetView:
    """``scene`` as the checks in ``lib/sheet_findings.py`` read it.

    Every part is in it, one whose symbol was not found as the placeholder it is
    drawn as. A part's box is what it draws together with its pins, and its
    body what it draws alone: its symbol's own two boxes, as placed, which are
    the boxes the schematic editor's geometry is given too.
    """
    return SheetView(
        parts=tuple(
            Part(
                ref=sym.reference,
                symbol=sym.symbol,
                at=(sym.x, sym.y),
                box=sym.box,
                body=sym.body,
                pins=tuple((pin.name, pin.x, pin.y) for pin in sym.pins),
                texts=tuple((t.x, t.y, decode_text_lines(t.text)[0]) for t in sym.texts),
                missing=sym.missing,
                unreadable=sym.unreadable,
            )
            for sym in scene.symbols
        ),
        wires=tuple((w.x1, w.y1, w.x2, w.y2) for w in scene.wires),
        labels=tuple((flag.x, flag.y, flag.text) for flag in scene.flags),
        texts=tuple((d.x, d.y, decode_text_lines(d.text)[0]) for d in scene.directives),
    )
