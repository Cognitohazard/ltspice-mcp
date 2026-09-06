"""Schematic scene graph: parse a ``.asc`` into placed, absolute-coordinate
render primitives.

This module is the front half of the ``.asc`` → SVG renderer. It reads a
schematic the same way :class:`spicelib.AscEditor` and
:mod:`ltspice_mcp.lib.symbol_geometry` read the record types (SYMBOL / WIRE /
FLAG / IOPIN / TEXT / WINDOW / SYMATTR), resolves each symbol's ``.asy`` body
through :class:`SymbolResolver`, and transforms every symbol-local coordinate
into absolute schematic space using
:func:`ltspice_mcp.lib.symbol_geometry._apply_rotation` (the shared, tested
rotation/mirror machinery — never reimplemented here).

The output :class:`Scene` holds semantic groups of primitives in stable source
order, plus a content bounding box for cropping and a list of human-readable
diagnostics (e.g. an unresolved symbol). :mod:`schematic_renderer` turns a
``Scene`` into SVG; nothing here emits markup.

Why a render-oriented ``.asy`` parse instead of reusing
``symbol_geometry.parse_asy_file``: that function keeps only pins and a bounding
box and discards the drawing primitives and text — everything a renderer needs.
So :func:`parse_symbol` does a full parse (lines/rects/circles/arcs/pins/
windows/text) while still reusing ``PinInfo`` and the rotation transform.
"""

from __future__ import annotations

import contextlib
import hashlib
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from ltspice_mcp.lib.cache import file_stamp
from ltspice_mcp.lib.encoding import decode_spice_bytes, read_spice_text
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.symbol_geometry import (
    PinInfo,
    _apply_rotation,  # pyright: ignore[reportPrivateUsage]  # shared rotation/mirror transform
)

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

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),)


Graphic = DrawLine | DrawRect | DrawEllipse | DrawPolyline


# ---------------------------------------------------------------------------
# Symbol prototype (parsed .asy body, symbol-local coordinates)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowDef:
    """A symbol attribute-text anchor (``WINDOW n x y align size``)."""

    number: int
    x: int
    y: int
    align: str
    size: int


@dataclass(frozen=True)
class AsyArc:
    """LTspice ARC: ellipse bbox (x1,y1)-(x2,y2) plus start/end points."""

    x1: int
    y1: int
    x2: int
    y2: int
    sx: int
    sy: int
    ex: int
    ey: int


@dataclass(frozen=True)
class SymbolProto:
    """Parsed ``.asy`` symbol body in symbol-local coordinates."""

    name: str
    path: Path
    lines: tuple[DrawLine, ...]
    rects: tuple[DrawRect, ...]
    circles: tuple[DrawEllipse, ...]
    arcs: tuple[AsyArc, ...]
    pins: tuple[PinInfo, ...]
    windows: tuple[WindowDef, ...]
    bbox: BBox

    def window(self, number: int) -> WindowDef | None:
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


def _parse_shape_coords(parts: Sequence[str], count: int) -> list[int] | None:
    """Parse ``count`` integer coordinates starting after the style token.

    ``.asy`` shape lines are ``KEYWORD <style> c0 c1 ...``; coordinates begin
    at index 2. Returns ``None`` if too few fields or any coordinate is not an
    integer (a degenerate/garbage line is skipped, never fatal).
    """
    if len(parts) < 2 + count:
        return None
    try:
        return [int(parts[2 + i]) for i in range(count)]
    except ValueError:
        return None


def parse_symbol(asy_path: Path, name: str) -> SymbolProto:
    """Parse a ``.asy`` file into a render-ready :class:`SymbolProto`.

    Reads via ``read_spice_text`` (BOM/UTF-16/cp1252 fallback) — vendor symbols
    routinely carry cp1252 bytes in description fields. Unrecognized or
    malformed lines are skipped, mirroring ``symbol_geometry.parse_asy_file``.
    """
    text = read_spice_text(asy_path)
    lines_txt = text.splitlines()

    lines: list[DrawLine] = []
    rects: list[DrawRect] = []
    circles: list[DrawEllipse] = []
    arcs: list[AsyArc] = []
    pins: list[PinInfo] = []
    windows: list[WindowDef] = []

    i = 0
    n = len(lines_txt)
    while i < n:
        raw = lines_txt[i].strip()
        parts = raw.split()
        i += 1
        if not parts:
            continue
        kw = parts[0]

        if kw == "LINE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                lines.append(DrawLine(c[0], c[1], c[2], c[3]))
        elif kw == "RECTANGLE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                rects.append(DrawRect(c[0], c[1], c[2], c[3]))
        elif kw == "CIRCLE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                circles.append(DrawEllipse(c[0], c[1], c[2], c[3]))
        elif kw == "ARC":
            c = _parse_shape_coords(parts, 8)
            if c is not None:
                arcs.append(AsyArc(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]))
        elif kw == "WINDOW":
            # WINDOW n x y align size
            if len(parts) >= 4:
                try:
                    num = int(parts[1])
                    wx = int(parts[2])
                    wy = int(parts[3])
                except ValueError:
                    continue
                align = parts[4] if len(parts) > 4 else "Left"
                try:
                    size = int(parts[5]) if len(parts) > 5 else 2
                except ValueError:
                    size = 2
                windows.append(WindowDef(num, wx, wy, align, size))
        elif kw == "PIN":
            # PIN x y <justification> <offset>, then PINATTR lines.
            if len(parts) >= 3:
                try:
                    px, py = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                pin_name = ""
                pin_order = 0
                while i < n and lines_txt[i].strip().startswith("PINATTR"):
                    attr = lines_txt[i].strip()
                    if attr.startswith("PINATTR PinName"):
                        toks = attr.split(None, 2)
                        pin_name = toks[2] if len(toks) > 2 else ""
                    elif attr.startswith("PINATTR SpiceOrder"):
                        try:
                            pin_order = int(attr.split()[-1])
                        except ValueError:
                            pin_order = 0
                    i += 1
                pins.append(PinInfo(name=pin_name, order=pin_order, x=px, y=py))

    pin_points = [(p.x, p.y) for p in pins]
    element_points: list[tuple[int, int]] = []
    for ln in lines:
        element_points.extend(ln.points())
    for rc in rects:
        element_points.extend(rc.points())
    for ci in circles:
        element_points.extend(ci.points())
    for ar in arcs:
        element_points.append((ar.x1, ar.y1))
        element_points.append((ar.x2, ar.y2))
    bbox = BBox.from_points(element_points + pin_points) or BBox(0, 0, 0, 0)

    pins.sort(key=lambda p: p.order)
    return SymbolProto(
        name=name,
        path=asy_path,
        lines=tuple(lines),
        rects=tuple(rects),
        circles=tuple(circles),
        arcs=tuple(arcs),
        pins=tuple(pins),
        windows=tuple(windows),
        bbox=bbox,
    )


# ---------------------------------------------------------------------------
# Symbol resolution (schematic-local → project → stock), content-stamp cached
# ---------------------------------------------------------------------------


class SymbolResolver:
    """Resolve a symbol name to a ``.asy`` file and parse it, with caching.

    Precedence, highest first:

    1. the schematic's own directory (``local_dir``),
    2. configured project symbol paths (``project_paths``),
    3. stock LTspice library paths (``stock_paths``).

    Each resolver owns its caches; its search roots (and therefore its
    ``active_set``) are fixed at construction. The parse cache is keyed by
    ``(resolved path, content stamp, active library-path set)`` — never by bare
    symbol name — so rewriting a local ``.asy`` bumps the content stamp and
    invalidates the entry, while an unchanged file is served from cache. The
    active-set component is what keeps entries from bleeding across resolvers
    configured with different libraries: a different library set is a different
    resolver, and because it never resolves to a foreign entry, ``resolve`` for
    a symbol whose winning file changed with the search order returns the right
    path. Resolution results (including misses) are cached per instance, so a
    repeated miss does not re-walk the stock tree.
    """

    def __init__(
        self,
        local_dir: Path | None,
        project_paths: Sequence[Path] = (),
        stock_paths: Sequence[Path] = (),
    ) -> None:
        roots: list[Path] = []
        if local_dir is not None:
            roots.append(local_dir)
        roots.extend(project_paths)
        roots.extend(stock_paths)
        # De-duplicate while preserving precedence order.
        seen: set[str] = set()
        self._roots: list[Path] = []
        for r in roots:
            key = str(r)
            if key not in seen:
                seen.add(key)
                self._roots.append(r)
        self._active_set: frozenset[str] = frozenset(str(r) for r in self._roots)
        self._resolve_cache: dict[str, Path | None] = {}
        self._parse_cache: dict[tuple[str, tuple[int, int], frozenset[str]], SymbolProto] = {}

    @property
    def active_set(self) -> frozenset[str]:
        return self._active_set

    def _candidates(self, root: Path, symbol: str) -> list[Path]:
        # LTspice writes symbol names with backslash subdir separators.
        rel = symbol.replace("\\", "/")
        stem = rel.rsplit("/", 1)[-1]
        out = [root / f"{rel}.asy"]
        if stem != rel:
            out.append(root / f"{stem}.asy")
        return out

    def resolve(self, symbol: str) -> Path | None:
        """Return the ``.asy`` path for ``symbol``, or ``None`` if unresolved."""
        if symbol in self._resolve_cache:
            return self._resolve_cache[symbol]
        result: Path | None = None
        stem = symbol.replace("\\", "/").rsplit("/", 1)[-1]
        for root in self._roots:
            if not root.is_dir():
                continue
            for cand in self._candidates(root, symbol):
                if cand.is_file():
                    result = cand
                    break
            if result is not None:
                break
            # LTspice organizes stock symbols into subfolders; fall back to a
            # recursive search by bare stem within this root.
            match = next(root.rglob(f"{stem}.asy"), None)
            if match is not None:
                result = match
                break
        self._resolve_cache[symbol] = result
        return result

    def load(self, symbol: str) -> SymbolProto | None:
        """Resolve and parse ``symbol``; ``None`` if unresolved or unparseable."""
        path = self.resolve(symbol)
        if path is None:
            return None
        try:
            stamp = file_stamp(path)
        except OSError:
            return None
        key = (str(path), stamp, self._active_set)
        cached = self._parse_cache.get(key)
        if cached is not None:
            return cached
        try:
            proto = parse_symbol(path, symbol.replace("\\", "/").rsplit("/", 1)[-1])
        except Exception:
            return None
        self._parse_cache[key] = proto
        return proto


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


@dataclass
class _RawSymbol:
    symbol: str
    x: int
    y: int
    rotation: str
    attrs: dict[str, str] = field(default_factory=dict)
    windows: list[WindowDef] = field(default_factory=list)


@dataclass
class _AscDoc:
    symbols: list[_RawSymbol] = field(default_factory=list)
    wires: list[Wire] = field(default_factory=list)
    flags: list[NetFlag] = field(default_factory=list)
    directives: list[Directive] = field(default_factory=list)
    sheet_lines: list[DrawLine] = field(default_factory=list)
    sheet_rects: list[DrawRect] = field(default_factory=list)
    sheet_circles: list[DrawEllipse] = field(default_factory=list)
    sheet_arcs: list[AsyArc] = field(default_factory=list)


_ROTATIONS = frozenset({"R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"})


def _parse_asc(text: str) -> _AscDoc:
    doc = _AscDoc()
    current: _RawSymbol | None = None

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        kw = parts[0]

        if kw == "SYMBOL":
            # SYMBOL <name> <x> <y> <ROT>. The trailing token is the rotation
            # (R0/R90/…/M270); the two before it are x and y; everything
            # between "SYMBOL" and them is the symbol name (may embed a
            # backslash subdir, never a space in practice).
            if len(parts) >= 5 and parts[-1] in _ROTATIONS:
                try:
                    x, y = int(parts[-3]), int(parts[-2])
                except ValueError:
                    current = None
                    continue
                current = _RawSymbol(symbol=" ".join(parts[1:-3]), x=x, y=y, rotation=parts[-1])
                doc.symbols.append(current)
            elif len(parts) >= 4:
                # Degenerate: no rotation token; treat last two as x,y, R0.
                try:
                    x, y = int(parts[-2]), int(parts[-1])
                except ValueError:
                    current = None
                    continue
                current = _RawSymbol(symbol=" ".join(parts[1:-2]), x=x, y=y, rotation="R0")
                doc.symbols.append(current)
            else:
                current = None
        elif kw == "WINDOW" and current is not None:
            if len(parts) >= 4:
                try:
                    num = int(parts[1])
                    wx = int(parts[2])
                    wy = int(parts[3])
                except ValueError:
                    continue
                align = parts[4] if len(parts) > 4 else "Left"
                try:
                    size = int(parts[5]) if len(parts) > 5 else 2
                except ValueError:
                    size = 2
                current.windows.append(WindowDef(num, wx, wy, align, size))
        elif kw == "SYMATTR" and current is not None:
            if len(parts) >= 2:
                key = parts[1]
                value = line.split(None, 2)[2] if len(line.split(None, 2)) > 2 else ""
                current.attrs[key] = value
        elif kw == "WIRE":
            if len(parts) >= 5:
                with contextlib.suppress(ValueError):
                    doc.wires.append(
                        Wire(int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
                    )
        elif kw == "FLAG":
            # FLAG <x> <y> <text>
            if len(parts) >= 4:
                try:
                    fx, fy = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                ftext = line.split(None, 3)[3] if len(line.split(None, 3)) > 3 else ""
                doc.flags.append(NetFlag(fx, fy, ftext, is_ground=(ftext.strip() == "0")))
        elif kw == "TEXT":
            d = _parse_text_record(line)
            if d is not None:
                doc.directives.append(d)
        elif kw == "LINE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                doc.sheet_lines.append(DrawLine(c[0], c[1], c[2], c[3]))
        elif kw == "RECTANGLE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                doc.sheet_rects.append(DrawRect(c[0], c[1], c[2], c[3]))
        elif kw == "CIRCLE":
            c = _parse_shape_coords(parts, 4)
            if c is not None:
                doc.sheet_circles.append(DrawEllipse(c[0], c[1], c[2], c[3]))
        elif kw == "ARC":
            c = _parse_shape_coords(parts, 8)
            if c is not None:
                doc.sheet_arcs.append(AsyArc(c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]))
        # IOPIN / DATAFLAG / SHEET / Version: no geometry we render in V1.

    return doc


def _parse_text_record(line: str) -> Directive | None:
    """Parse ``TEXT <x> <y> <align> <size> <body>``.

    The body starts with ``;`` (comment) or ``!`` (SPICE directive). LTspice
    encodes embedded newlines; we keep the raw body and let the renderer split.
    """
    parts = line.split(None, 5)
    if len(parts) < 5:
        return None
    try:
        x, y = int(parts[1]), int(parts[2])
    except ValueError:
        return None
    align = parts[3]
    try:
        size = int(parts[4])
    except ValueError:
        size = 2
    body = parts[5] if len(parts) > 5 else ""
    is_directive = body.startswith("!")
    if body[:1] in ("!", ";"):
        body = body[1:]
    return Directive(x, y, body, align, size, is_directive)


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


def _arc_polyline(arc: AsyArc, ox: int, oy: int, rot: str) -> DrawPolyline | None:
    """Sample an arc into an absolute-coordinate polyline.

    Sampling in symbol-local space and transforming each point means mirror and
    rotation are handled by the shared transform — no SVG sweep-flag reasoning.
    Degenerate (zero-area) ellipses are skipped.

    Sweep direction: LTspice draws an ARC counter-clockwise *as displayed*, from
    the start point to the end point. LTspice's y axis points down, and with the
    parameterization ``(cx + rx·cosθ, cy + ry·sinθ)`` the on-screen angle
    *increases clockwise*; therefore the displayed counter-clockwise sweep is a
    *decreasing* θ. Empirically confirmed against the stock ``ind.asy``: its
    three arcs form the familiar coil spring (each a major, >180° loop) only
    under a decreasing-θ sweep — the increasing-θ sweep would draw the
    complementary minor arcs and the inductor would not read as a coil.
    """
    cx = (arc.x1 + arc.x2) / 2.0
    cy = (arc.y1 + arc.y2) / 2.0
    rx = abs(arc.x2 - arc.x1) / 2.0
    ry = abs(arc.y2 - arc.y1) / 2.0
    if rx == 0 or ry == 0:
        return None

    def angle_of(sx: int, sy: int) -> float:
        return math.atan2((sy - cy) / ry, (sx - cx) / rx)

    a0 = angle_of(arc.sx, arc.sy)
    a1 = angle_of(arc.ex, arc.ey)
    # Displayed counter-clockwise = decreasing θ, so keep the sweep negative.
    sweep = a1 - a0
    while sweep >= 0:
        sweep -= 2 * math.pi

    pts: list[tuple[int, int]] = []
    for k in range(_ARC_SEGMENTS + 1):
        a = a0 + sweep * (k / _ARC_SEGMENTS)
        lx = cx + rx * math.cos(a)
        ly = cy + ry * math.sin(a)
        pts.append(_place_point(round(lx), round(ly), ox, oy, rot))
    return DrawPolyline(tuple(pts))


def _place_symbol(raw: _RawSymbol, proto: SymbolProto | None) -> PlacedSymbol:
    ref = raw.attrs.get("InstName", "")
    ox, oy, rot = raw.x, raw.y, raw.rotation

    if proto is None:
        placed = PlacedSymbol(
            reference=ref,
            symbol=raw.symbol,
            x=ox,
            y=oy,
            rotation=rot,
            resolved_path=None,
            missing=True,
        )
        x1, y1, x2, y2 = _place_aabb(
            _PLACEHOLDER.x1, _PLACEHOLDER.y1, _PLACEHOLDER.x2, _PLACEHOLDER.y2, ox, oy, rot
        )
        placed.graphics.append(DrawRect(x1, y1, x2, y2))
        label = ref or raw.symbol
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
        symbol=raw.symbol,
        x=ox,
        y=oy,
        rotation=rot,
        resolved_path=proto.path,
        missing=False,
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
        placed.pins.append(DrawPin(ax, ay))

    placed.texts.extend(_attr_texts(raw, proto, ox, oy, rot))
    return placed


def _attr_texts(raw: _RawSymbol, proto: SymbolProto, ox: int, oy: int, rot: str) -> list[DrawText]:
    """Instance-name and value text at their window anchors.

    Window anchor coordinates are symbol-local: a per-instance ``WINDOW``
    override in the ``.asc`` replaces the ``.asy`` default, then the anchor is
    rotated and offset to the placement origin. Attributes without a resolvable
    window and without a value are omitted.

    The glyphs turn with the symbol, not just the anchor — see
    :func:`_attr_text_placement`.
    """
    out: list[DrawText] = []
    overrides = {w.number: w for w in raw.windows}

    def placed(number: int, text: str) -> DrawText | None:
        w = overrides.get(number) or proto.window(number)
        if w is None:
            return None
        ax, ay = _place_point(w.x, w.y, ox, oy, rot)
        turn, anchor = _attr_text_placement(w.align, rot, explicit=number in overrides)
        return DrawText(ax, ay, text, anchor, w.size, "attr", rotation=turn)

    for number, text in (
        (_WINDOW_INSTNAME, raw.attrs.get("InstName", "")),
        (_WINDOW_VALUE, raw.attrs.get("Value", "")),
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
# axis rule below is scoped to the turns listed here.
_QUARTER_TURNS = {"R90": "down", "R270": "up", "M90": "up", "M270": "down"}


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
    doc = _parse_asc(decode_spice_bytes(data))
    scene = Scene(source=asc_path, source_sha256=hashlib.sha256(data).hexdigest())

    for raw in doc.symbols:
        proto = resolver.load(raw.symbol)
        if proto is None:
            scene.diagnostics.append(
                f"symbol '{raw.symbol}'"
                + (f" (instance {raw.attrs.get('InstName')})" if raw.attrs.get("InstName") else "")
                + " not found in any library path; placeholder rendered"
            )
        scene.symbols.append(_place_symbol(raw, proto))

    scene.wires.extend(doc.wires)
    # Placement needs the wire set, so resolve it once the wires are known.
    # A flag counts as connected if a wire ends on it, a wire runs through it,
    # or a symbol pin sits on it; only a truly isolated flag gets the marker.
    pin_points = {(p.x, p.y) for sym in scene.symbols for p in sym.pins}
    segments = [((w.x1, w.y1), (w.x2, w.y2)) for w in doc.wires]
    for fl in doc.flags:
        occupied = wire_directions_at(doc.wires, fl.x, fl.y)
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
    scene.directives.extend(doc.directives)
    scene.sheet_graphics.extend(doc.sheet_lines)
    scene.sheet_graphics.extend(doc.sheet_rects)
    scene.sheet_graphics.extend(doc.sheet_circles)
    for ar in doc.sheet_arcs:
        poly = _arc_polyline(ar, 0, 0, "R0")
        if poly is not None:
            scene.sheet_graphics.append(poly)

    return scene


# ---------------------------------------------------------------------------
# Layout issues: geometric facts about a drawn schematic
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayoutIssue:
    """One geometric observation about a scene's layout.

    Facts only — a kind names a measured geometric condition, never a severity
    or a quality verdict. Whether an issue matters is the caller's judgement.

    Callers should read ``symbol_overlap`` and ``text_in_symbol_body`` with the
    same caveat the schematic editor's advisories carry: a symbol's box is the
    axis-aligned extent of everything it draws, so it also spans leads and empty
    corners. An overlap of two boxes is therefore not proof that ink overlaps.
    """

    kind: str
    refs: tuple[str, ...]
    coords: tuple[tuple[int, int], ...]
    detail: str

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "refs": list(self.refs),
            "coords": [list(c) for c in self.coords],
            "detail": self.detail,
        }


def point_on_segment(p: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if ``p`` lies on the segment ``a``-``b`` (endpoints included)."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
    if cross != 0:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _crosses_box_interior(a: tuple[int, int], b: tuple[int, int], box: BBox) -> bool:
    """True if segment ``a``-``b`` passes through ``box``'s interior.

    Parametric (Liang-Barsky) clip, then a strict containment test on the
    midpoint of the clipped span. Both steps matter: a segment that only
    terminates on the boundary clips to zero length, and one that runs *along*
    an edge clips to a span whose midpoint is on the boundary, not inside. So
    neither the normal way a wire meets a pin nor a wire tracking an edge counts
    as crossing. A degenerate (zero-length) segment never crosses.
    """
    (x1, y1), (x2, y2) = a, b
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return False
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x1 - box.x1), (dx, box.x2 - x1), (-dy, y1 - box.y1), (dy, box.y2 - y1)):
        if p == 0:
            if q < 0:
                return False
        else:
            r = q / p
            if p < 0:
                if r > t1:
                    return False
                t0 = max(t0, r)
            else:
                if r < t0:
                    return False
                t1 = min(t1, r)
    if t1 <= t0:
        return False
    tm = (t0 + t1) / 2
    mx, my = x1 + tm * dx, y1 + tm * dy
    return box.x1 < mx < box.x2 and box.y1 < my < box.y2


def _body_bbox(sym: PlacedSymbol) -> BBox | None:
    """Axis-aligned extent of everything a placed symbol draws."""
    pts: list[tuple[int, int]] = []
    for g in sym.graphics:
        pts.extend(g.points())
    return BBox.from_points(pts)


def _label_of(sym: PlacedSymbol) -> str:
    return sym.reference or sym.symbol or "<unnamed>"


def layout_issues(scene: Scene) -> list[LayoutIssue]:
    """Geometric observations about ``scene``'s layout, in a stable order.

    Detects overlapping symbol bodies, wires crossing through a symbol body,
    pins connected to nothing, wire ends connected to nothing, and text anchored
    inside another symbol's body. See :class:`LayoutIssue` for how to read them.

    Issues are emitted grouped by kind in that order, and within a kind in the
    scene's own (source) order, so the list is reproducible for a given input.

    The text check tests a text object's *anchor point* only, so the later lines
    of a multi-line directive that runs down into a symbol body are not
    reported.
    """
    issues: list[LayoutIssue] = []
    boxes: list[tuple[PlacedSymbol, BBox]] = []
    for sym in scene.symbols:
        bb = _body_bbox(sym)
        if bb is not None:
            boxes.append((sym, bb))

    # --- symbol bodies overlapping each other -------------------------------
    for i in range(len(boxes)):
        sym_a, box_a = boxes[i]
        for j in range(i + 1, len(boxes)):
            sym_b, box_b = boxes[j]
            if not box_a.overlaps(box_b):
                continue
            ox1, oy1 = max(box_a.x1, box_b.x1), max(box_a.y1, box_b.y1)
            ox2, oy2 = min(box_a.x2, box_b.x2), min(box_a.y2, box_b.y2)
            issues.append(
                LayoutIssue(
                    kind="symbol_overlap",
                    refs=(_label_of(sym_a), _label_of(sym_b)),
                    coords=((ox1, oy1), (ox2, oy2)),
                    detail=(f"bounding boxes share a {ox2 - ox1}x{oy2 - oy1} region"),
                )
            )

    # Which symbols own a pin at each coordinate, keyed by scene index. Index
    # rather than label so two symbols sharing a reference (or having none)
    # still count as two owners, and two pins of the SAME symbol stacked on one
    # coordinate count as one — that is a floating pin, not a connection.
    pin_owners: dict[tuple[int, int], set[int]] = {}
    for sym_index, sym in enumerate(scene.symbols):
        for pin in sym.pins:
            pin_owners.setdefault((pin.x, pin.y), set()).add(sym_index)
    flag_coords = {(f.x, f.y) for f in scene.flags}
    # Each span paired with its wire, zero-length ones dropped: a degenerate
    # WIRE has no span to cross anything and no end to dangle from.
    real_wires = [
        (a, b, w)
        for (a, b), w in zip(
            [((w.x1, w.y1), (w.x2, w.y2)) for w in scene.wires], scene.wires, strict=True
        )
        if a != b
    ]
    real_segments = [(a, b) for a, b, _ in real_wires]

    # --- wires crossing through a symbol body -------------------------------
    # A wire attached to one of the symbol's own pins is NOT exempt: leaving a
    # pin and running straight back across the body is the very error this
    # looks for. Landing on a boundary pin and heading outward clips to zero
    # length, so the normal connection still does not register.
    for wa, wb, wire in real_wires:
        for sym, box in boxes:
            if _crosses_box_interior(wa, wb, box):
                issues.append(
                    LayoutIssue(
                        kind="wire_through_symbol",
                        refs=(_label_of(sym),),
                        coords=((wire.x1, wire.y1), (wire.x2, wire.y2)),
                        detail="wire segment passes through the symbol's body box",
                    )
                )

    # --- pins connected to nothing ------------------------------------------
    for sym in scene.symbols:
        for pin in sym.pins:
            coord = (pin.x, pin.y)
            if coord in flag_coords:
                continue
            if len(pin_owners.get(coord, ())) > 1:
                continue  # shares the coordinate with another symbol's pin
            if any(point_on_segment(coord, a, b) for a, b in real_segments):
                continue
            issues.append(
                LayoutIssue(
                    kind="floating_pin",
                    refs=(_label_of(sym),),
                    coords=(coord,),
                    detail="pin has no wire, net label, or mating pin on it",
                )
            )

    # --- wire ends connected to nothing -------------------------------------
    # Reported once per coordinate: two loose ends meeting nothing at the same
    # point are one place to look, not two findings.
    seen_ends: set[tuple[int, int]] = set()
    for idx, (a, b) in enumerate(real_segments):
        for end in (a, b):
            if end in seen_ends or end in pin_owners or end in flag_coords:
                continue
            touches_other = any(
                point_on_segment(end, oa, ob)
                for k, (oa, ob) in enumerate(real_segments)
                if k != idx
            )
            if touches_other:
                continue
            seen_ends.add(end)
            issues.append(
                LayoutIssue(
                    kind="dangling_wire_end",
                    refs=(),
                    coords=(end,),
                    detail="wire end meets no pin, net label, or other wire",
                )
            )

    # --- text anchored inside another symbol's body -------------------------
    anchors: list[tuple[str, tuple[int, int], PlacedSymbol | None]] = []
    for sym in scene.symbols:
        for t in sym.texts:
            anchors.append((t.text, (t.x, t.y), sym))
    for d in scene.directives:
        anchors.append((d.text, (d.x, d.y), None))
    for text, (tx, ty), owner in anchors:
        for sym, box in boxes:
            if owner is sym:
                continue
            if box.x1 < tx < box.x2 and box.y1 < ty < box.y2:
                issues.append(
                    LayoutIssue(
                        kind="text_in_symbol_body",
                        refs=(_label_of(sym),),
                        coords=((tx, ty),),
                        detail=(
                            f"text {decode_text_lines(text)[0]!r} is anchored "
                            "inside the symbol's body box"
                        ),
                    )
                )

    return issues
