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

A part's pins and boxes are the ones the sheet placed; its drawing is the
symbol file's lines, rectangles, circles and arcs, placed the same way.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from itertools import pairwise
from pathlib import Path
from typing import Any, NamedTuple

from ltspice_mcp.lib.asc_document import Window
from ltspice_mcp.lib.connectivity import point_on_segment
from ltspice_mcp.lib.encoding import refused_sheet_mark
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.sheet import (
    Sheet,
    SheetPart,
    read_sheet,
    read_sheet_text,
)
from ltspice_mcp.lib.sheet_findings import Part, SheetView
from ltspice_mcp.lib.symbol_file import SymbolArc
from ltspice_mcp.lib.symbol_geometry import (
    _apply_rotation,  # pyright: ignore[reportPrivateUsage]  # shared rotation/mirror transform
    place_box,
    place_point,
)
from ltspice_mcp.lib.symbol_library import SymbolResolver

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
# LTspice draws all schematic text (attributes, net labels, comments and
# directives) in Tahoma Bold. It stores a text *size index* and draws index i at
# a multiple of one base size, index 2 being the schematic default; a 1x line of
# text is _BASE_LINE sheet units tall. Both builds were measured drawing every
# index, by the font height each asks for and the text it draws.
_SIZE_MULTIPLES = {0: 0.625, 1: 1.0, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.5, 6: 5.0, 7: 7.0}
_BASE_LINE = 18.4
# Tahoma Bold's ascent and descent in ems, which make up the height of a line,
# and how far its glyphs reach above and below the baseline.
_ASCENT = 2049 / 2048
_DESCENT = 423 / 2048
_INK_ASCENT = 0.76
_INK_DESCENT = 0.21

#: The font size (em) in sheet units of each text size index.
FONT_PX = {
    size: round(_BASE_LINE * multiple / (_ASCENT + _DESCENT))
    for size, multiple in _SIZE_MULTIPLES.items()
}
DEFAULT_FONT_PX = FONT_PX[2]

# Tahoma Bold's advance widths in thousandths of an em for printable ASCII,
# space to tilde, and for anything else.
# fmt: off
_ADVANCES = (
    293, 343, 489, 818, 637, 1199, 781, 275, 454, 454, 637, 818, 312, 431, 312, 577,
    637, 637, 637, 637, 637, 637, 637, 637, 637, 637, 363, 363, 818, 818, 818, 566,
    920, 685, 686, 667, 757, 615, 581, 745, 764, 483, 500, 696, 572, 893, 771, 770,
    657, 770, 726, 633, 612, 739, 675, 1028, 685, 670, 623, 454, 577, 454, 818, 637,
    546, 599, 632, 527, 629, 594, 382, 629, 640, 302, 363, 603, 302, 954, 640, 617,
    629, 629, 434, 515, 416, 640, 579, 890, 604, 576, 503, 623, 637, 623, 818,
)
# fmt: on
_OTHER_ADVANCE = 637

# Placement of the ground glyph relative to its FLAG coordinate, and its
# extent. Shared so the crop box and the drawn output cannot drift apart.
# A net label is size-2 text like the rest.
FLAG_LABEL_SIZE = 2
GND_HALF_W = 6
GND_DEPTH = 10

# Side of the small square LTspice draws at a flag that connects to nothing.
UNCONNECTED_MARKER = 4


def font_px(size: int) -> int:
    """Font size (em), in sheet units, for an LTspice text-size index."""
    return FONT_PX.get(size, DEFAULT_FONT_PX)


def line_height(size: int) -> int:
    """How far apart, in sheet units, LTspice sets the lines of a text block."""
    return round(font_px(size) * (_ASCENT + _DESCENT))


def text_width(text: str, size: int) -> int:
    """Width in sheet units of one line of ``text`` set at ``size``."""
    advance = sum(
        _ADVANCES[code - 32] if 32 <= (code := ord(ch)) < 127 else _OTHER_ADVANCE for ch in text
    )
    return round(advance * font_px(size) / 1000)


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


#: How a justification lays text out along its run: which way along it the
#: text extends from its point (1 for Left, -1 for Right, 0 centred), and on
#: which side of the point it lies across the run (-1 for Top, the point being
#: its top edge; 1 for Bottom; 0 centred). A V token is the same with the run
#: up its frame rather than along the frame's x axis.
_JUSTIFICATIONS: dict[str, tuple[int, int]] = {
    "left": (1, 0),
    "right": (-1, 0),
    "center": (0, 0),
    "top": (0, -1),
    "bottom": (0, 1),
}

#: The direction text reads along and its glyphs' "up" for each turn it is
#: drawn at: upright, or a quarter turn to read bottom to top. These are also
#: a frame's run and "up" for a plain and for a V justification.
_READING = {0: ((1, 0), (0, -1)), -90: ((0, -1), (-1, 0))}


class TextLayout(NamedTuple):
    """How a text is drawn from its point: turned ``rotation`` degrees, its
    lines aligned on SVG ``anchor``, and its first line's baseline ``(dx, dy)``
    from the point. Later lines follow one :func:`line_height` apart along the
    glyphs' "down"."""

    rotation: int
    anchor: str
    dx: int
    dy: int


def text_layout(align: str, size: int, rot: str = "R0", lines: int = 1) -> TextLayout | None:
    """How LTspice lays out ``lines`` lines of text justified ``align`` in a
    frame placed ``rot`` (a symbol's placement, or ``R0`` for the sheet's own
    text). None for ``Invisible``, which LTspice does not draw.

    Read off LTspice's own drawing of every justification at every placement,
    which the suite checks against screenshots of both builds
    (``test_ltspice_screenshots.py``). The token lays the text out in its frame
    and the placement turns that frame onto the sheet; the glyphs are then
    drawn readable, left to right or bottom to top. So text is sideways
    exactly when its run ends up vertical, a ``Left`` window whose run the
    placement points left or down ends at its point instead of starting there,
    and a ``Top`` window's text lies on whichever side of its point the
    placement turns the frame's "down" to.

    Each line is set in a box one line high, which the token puts on the point
    by its edge or its middle: its top edge for ``Top``, the bottom edge of the
    last line for ``Bottom``, and the middle of the first line otherwise.
    """
    token = align.lower()
    if token == "invisible":
        return None
    along, across = _JUSTIFICATIONS.get(token.removeprefix("v"), _JUSTIFICATIONS["left"])
    frame_run, frame_up = _READING[-90 if token.startswith("v") else 0]
    run, up = _apply_rotation(*frame_run, rot), _apply_rotation(*frame_up, rot)
    rotation = -90 if run[0] == 0 else 0
    readable_run, readable_up = _READING[rotation]
    if along == 0:
        anchor = "middle"
    else:
        anchor = "start" if (run[0] * along, run[1] * along) == readable_run else "end"
    em = font_px(size)
    # How far the first baseline lies from the point along the glyphs' "down".
    if across == 0:
        drop = (_ASCENT - _DESCENT) / 2 * em
    elif (up[0] * across, up[1] * across) == readable_up:
        drop = -_DESCENT * em - (lines - 1) * line_height(size)  # the block ends at the point
    else:
        drop = _ASCENT * em  # the block starts at the point
    shift = round(drop)
    return TextLayout(rotation, anchor, -readable_up[0] * shift, -readable_up[1] * shift)


def text_extent(x: int, y: int, lines: Sequence[str], size: int, layout: TextLayout) -> BBox:
    """Where the glyphs of ``lines`` reach when laid out by ``layout`` from the
    point ``(x, y)``: the text's estimated ink, from its font's widths."""
    em = font_px(size)
    (rx, ry), (ux, uy) = _READING[layout.rotation]
    corners: list[tuple[int, int]] = []
    for index, line in enumerate(lines):
        width = text_width(line, size)
        start = {"start": 0, "middle": -width // 2, "end": -width}.get(layout.anchor, 0)
        down = index * line_height(size)
        bx, by = x + layout.dx - ux * down, y + layout.dy - uy * down
        for along in (start, start + width):
            for rise in (_INK_ASCENT * em, -_INK_DESCENT * em):
                corners.append(
                    (round(bx + rx * along + ux * rise), round(by + ry * along + uy * rise))
                )
    return BBox.from_points(corners) or BBox(x, y, x, y)


#: The justification that puts a net label's text on the side of its point
#: given by ``(orientation, text_vertical)``: a label's text starts or ends at
#: the point, across a wire's line.
_FLAG_JUSTIFICATIONS = {
    ("right", False): "Left",
    ("left", False): "Right",
    ("up", False): "Bottom",
    ("down", False): "Top",
    ("up", True): "VLeft",
    ("down", True): "VRight",
}


def flag_label_layout(flag: NetFlag) -> TextLayout:
    """How a net label's text is laid out from its point.

    A label on a vertical wire has its text turned to read bottom to top (see
    :func:`resolve_flag_placement`). LTspice starts or ends the text at the
    point, centred across the wire; with no wire to run along, the text sits
    centred above or below the point.
    """
    align = _FLAG_JUSTIFICATIONS.get((flag.orientation, flag.text_vertical), "Left")
    layout = text_layout(align, FLAG_LABEL_SIZE)
    assert layout is not None, "a label's justification is never Invisible"
    return layout


def flag_label_bbox(flag: NetFlag) -> BBox:
    """Estimated extent of a net label's drawn text, rotation included."""
    return text_extent(flag.x, flag.y, [flag.text], FLAG_LABEL_SIZE, flag_label_layout(flag))


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
    size: int
    role: str  # "attr" | "flag" | "directive" | "comment" | "placeholder"
    # How the text is drawn from (x, y), the point it is placed by: LTspice
    # places text by the edges or the middle of its line (see text_layout).
    layout: TextLayout

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),)


def drawn_text_extent(t: DrawText) -> BBox:
    """Estimated extent of a single drawn text, upright or turned: the one
    estimate both the crop and the layout checks read."""
    return text_extent(t.x, t.y, [t.text], t.size, t.layout)


@dataclass(frozen=True)
class DrawPin:
    x: int
    y: int
    name: str = ""

    def points(self) -> tuple[tuple[int, int], ...]:
        return ((self.x, self.y),)


Graphic = DrawLine | DrawRect | DrawEllipse | DrawPolyline


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
    #: Why a symbol that was found cannot be used (``FoundSymbol.problem``); the
    #: part is then drawn as a placeholder. Empty for a symbol not found.
    problem: str = ""
    #: The part's extent with its pins, and of what it draws alone: its
    #: symbol's two boxes as placed. A part whose symbol was not found has the
    #: placeholder it is drawn as for both.
    box: BBox | None = None
    body: BBox | None = None
    graphics: list[Graphic] = field(default_factory=list)
    pins: list[DrawPin] = field(default_factory=list)
    texts: list[DrawText] = field(default_factory=list)
    #: The part's parameter attributes, ``SheetPart.parameters``.
    parameters: tuple[tuple[str, str], ...] = ()
    #: The element letter it netlists as (``SheetPart.element``), and the
    #: model its ``SpiceModel`` attribute selects, if any.
    element: str = ""
    spice_model: str = ""


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

    @cached_property
    def lines(self) -> list[str]:
        return decode_text_lines(self.text)

    @cached_property
    def layout(self) -> TextLayout | None:
        return text_layout(self.align, self.size, lines=len(self.lines))

    @cached_property
    def extent(self) -> BBox | None:
        """Where its glyphs reach, or None when it is not drawn."""
        layout = self.layout
        return (
            None if layout is None else text_extent(self.x, self.y, self.lines, self.size, layout)
        )


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
        boxes.extend(extent for *_, extent in self.sheet_text_extents())
        for g in self.sheet_graphics:
            pts.extend(g.points())

        merged = BBox.from_points(pts)
        for b in boxes:
            merged = b if merged is None else merged.union(b)
        return merged

    def sheet_text_extents(self) -> tuple[tuple[int, int, str, BBox], ...]:
        """Where the net labels', comments' and directives' text is drawn, as
        ``(x, y, first line, extent)``."""
        labels = (
            (flag.x, flag.y, flag.text, flag_label_bbox(flag))
            for flag in self.flags
            if not flag.is_ground and flag.text
        )
        texts = ((d.x, d.y, d.lines[0], d.extent) for d in self.directives if d.extent is not None)
        return (*labels, *texts)


# ---------------------------------------------------------------------------
# Symbol libraries
# ---------------------------------------------------------------------------


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
# Placement: symbol-local geometry → absolute scene primitives
# ---------------------------------------------------------------------------


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
        pts.append(place_point(round(lx), round(ly), ox, oy, rot))
    return DrawPolyline(tuple(pts))


def _place_symbol(part: SheetPart) -> PlacedSymbol:
    ref = part.reference
    (ox, oy), rot = part.at, part.rotation
    # What the sheet rules read of the part, whether its symbol was found or not.
    facts: dict[str, Any] = {
        "parameters": part.parameters,
        "element": part.element,
        "spice_model": part.spice_model,
    }

    if part.symbol is None or not part.usable:
        placeholder = place_box(_PLACEHOLDER, ox, oy, rot)
        placed = PlacedSymbol(
            reference=ref,
            symbol=part.symbol_name,
            x=ox,
            y=oy,
            rotation=rot,
            resolved_path=part.symbol.path if part.symbol is not None else None,
            missing=True,
            problem=part.symbol.problem if part.symbol is not None else "",
            **facts,
        )
        placed.graphics.append(
            DrawRect(placeholder.x1, placeholder.y1, placeholder.x2, placeholder.y2)
        )
        placed.box = placed.body = placeholder
        placed.texts.append(
            DrawText(
                x=(placeholder.x1 + placeholder.x2) // 2,
                y=(placeholder.y1 + placeholder.y2) // 2,
                text=ref or part.symbol_name,
                size=2,
                role="placeholder",
                layout=TextLayout(0, "middle", 0, 0),
            )
        )
        return placed

    drawing = part.symbol.file
    placed = PlacedSymbol(
        reference=ref,
        symbol=part.symbol_name,
        x=ox,
        y=oy,
        rotation=rot,
        resolved_path=part.symbol.path,
        missing=False,
        box=part.box,
        body=part.body,
        **facts,
    )
    for x1, y1, x2, y2 in drawing.lines:
        placed.graphics.append(
            DrawLine(*place_point(x1, y1, ox, oy, rot), *place_point(x2, y2, ox, oy, rot))
        )
    for kind, boxes in ((DrawRect, drawing.rects), (DrawEllipse, drawing.circles)):
        for box in boxes:
            at = place_box(BBox(*box), ox, oy, rot)
            placed.graphics.append(kind(at.x1, at.y1, at.x2, at.y2))
    for arc in drawing.arcs:
        poly = _arc_polyline(arc, ox, oy, rot)
        if poly is not None:
            placed.graphics.append(poly)
    placed.pins.extend(DrawPin(pin.x, pin.y, pin.name) for pin in part.pins)
    placed.texts.extend(_attr_texts(part, drawing.windows, ox, oy, rot))
    return placed


def _attr_texts(
    part: SheetPart, windows: Sequence[Window], ox: int, oy: int, rot: str
) -> list[DrawText]:
    """Instance-name and value text at their window anchors.

    Window anchor coordinates are symbol-local: a per-instance ``WINDOW``
    override in the ``.asc`` replaces the ``.asy`` default, then the anchor is
    rotated and offset to the placement origin. Attributes without a resolvable
    window and without a value are omitted.

    The glyphs turn with the symbol, not just the anchor — see
    :func:`text_layout`. A window justified ``Invisible`` draws nothing.
    """
    out: list[DrawText] = []
    overrides = {w.number: w for w in part.record.windows}

    def placed(number: int, text: str) -> DrawText | None:
        w = overrides.get(number) or next((w for w in windows if w.number == number), None)
        if w is None:
            return None
        layout = text_layout(w.align, w.size, rot)
        if layout is None:
            return None
        ax, ay = place_point(w.x, w.y, ox, oy, rot)
        return DrawText(ax, ay, text, w.size, "attr", layout)

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
    return scene_of_sheet(
        read_sheet(data, resolver),
        asc_path,
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


def scene_of_sheet(
    sheet: Sheet,
    source: Path,
    source_sha256: str | None = None,
    byte_order_mark: str | None = None,
) -> Scene:
    """The fully-placed :class:`Scene` of ``sheet``, which is or will be at ``source``."""
    scene = Scene(source=source, source_sha256=source_sha256, byte_order_mark=byte_order_mark)
    scene.diagnostics.extend(
        sheet.unread_notes(
            "the drawing",
            "it is not drawn and no check built on the drawing includes it",
            "they are not drawn and no check built on the drawing includes them",
        )
    )

    for part in sheet.parts:
        instance = f" (instance {part.reference})" if part.reference else ""
        if part.symbol is None:
            scene.diagnostics.append(
                f"symbol '{part.symbol_name}'{instance} not found in any library path; "
                "placeholder rendered"
            )
        elif part.symbol.problem:
            scene.diagnostics.append(
                f"symbol '{part.symbol_name}'{instance} {part.symbol.problem}; "
                "placeholder rendered"
            )
        scene.symbols.append(_place_symbol(part))

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


def _strokes(graphics: Sequence[Graphic]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """The segments a part's lines, arcs and polylines are drawn as."""
    out: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for g in graphics:
        if isinstance(g, DrawLine):
            out.append(((g.x1, g.y1), (g.x2, g.y2)))
        elif isinstance(g, DrawPolyline):
            out.extend(pairwise(g.pts))
    return tuple(out)


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
                texts=tuple((x, y, text) for x, y, text, _extent in drawn),
                text_extents=drawn,
                strokes=_strokes(sym.graphics),
                areas=tuple(
                    BBox(g.x1, g.y1, g.x2, g.y2)
                    for g in sym.graphics
                    if isinstance(g, DrawRect | DrawEllipse)
                ),
                missing=sym.missing,
                problem=sym.problem,
                parameters=sym.parameters,
                element=sym.element,
                spice_model=sym.spice_model,
            )
            for sym in scene.symbols
            for drawn in [
                tuple(
                    (t.x, t.y, decode_text_lines(t.text)[0], drawn_text_extent(t))
                    for t in sym.texts
                )
            ]
        ),
        wires=tuple((w.x1, w.y1, w.x2, w.y2) for w in scene.wires),
        labels=tuple((flag.x, flag.y, flag.text) for flag in scene.flags),
        texts=tuple((d.x, d.y, d.lines[0]) for d in scene.directives),
        text_extents=scene.sheet_text_extents(),
    )
