"""An LTspice symbol (``.asy``) read once, into everything a caller needs of it.

``read_symbol`` takes a symbol file's text and returns a :class:`SymbolFile`:
its pins, the shapes of its body, the windows its attribute text is drawn in,
and its attributes. The pin geometry the schematic editor works with and the
body the renderer draws both come from this one reading, so the two cannot
disagree about what a symbol is.

Reading is lenient about a line that does not read: it is left out, as a
renderer wants. A pin is the exception that matters to an editor, because a
pin left out is a connection nobody sees, so the lines of a pin that did not
read are listed in ``SymbolFile.unread_pins`` for a caller that must refuse
the symbol instead.

Coordinates are the symbol's own, before any placement. This module knows no
file names or search paths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ltspice_mcp.lib.asc_document import Window
from ltspice_mcp.lib.geometry import BBox

Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class PinInfo:
    """A symbol pin with name, SPICE order, and position."""

    name: str
    order: int
    x: int
    y: int


#: Slack for an angle or a coordinate that arithmetic left a hair off the value
#: it stands for, so that an arc ending on a quarter turn reaches it.
_SLACK = 1e-9


@dataclass(frozen=True)
class SymbolArc:
    """An ``ARC``: the box of its ellipse, then a start point and an end point.

    What is drawn is the part of the ellipse from the direction of the start
    point to the direction of the end point, each seen from the centre, so
    neither point need lie on the ellipse. LTspice draws it counter-clockwise
    *as displayed*. Its y axis points down, and with the ellipse written as
    ``(cx + rx·cosθ, cy + ry·sinθ)`` the angle on screen *increases clockwise*,
    so the displayed counter-clockwise turn is a *decreasing* θ. The stock
    ``ind.asy`` shows it: its three arcs make the coil (each a loop of more
    than half a turn) only under a decreasing θ, where an increasing one draws
    the three small arcs left over.
    """

    x1: int
    y1: int
    x2: int
    y2: int
    sx: int
    sy: int
    ex: int
    ey: int

    def sweep(self) -> tuple[float, float] | None:
        """The θ the arc starts at and the angle it turns through, which is
        negative; ``None`` for an ellipse with no area, which draws nothing.

        A start and an end in one direction are the whole ellipse.
        """
        cx, cy = (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0
        rx, ry = abs(self.x2 - self.x1) / 2.0, abs(self.y2 - self.y1) / 2.0
        if rx == 0 or ry == 0:
            return None
        start = math.atan2((self.sy - cy) / ry, (self.sx - cx) / rx)
        turn = math.atan2((self.ey - cy) / ry, (self.ex - cx) / rx) - start
        while turn >= 0:
            turn -= 2 * math.pi
        return start, turn

    def at(self, angle: float) -> tuple[float, float]:
        """The point of the ellipse at θ = ``angle``."""
        cx, cy = (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0
        rx, ry = abs(self.x2 - self.x1) / 2.0, abs(self.y2 - self.y1) / 2.0
        return cx + rx * math.cos(angle), cy + ry * math.sin(angle)

    def extent(self) -> Box | None:
        """The smallest box of whole units around the arc as drawn; ``None``
        for one that draws nothing.

        The box of the ellipse bounds the arc too, and more: a shallow arc cut
        from a large circle, the curved plate of a polarized capacitor, is a
        sliver of that circle's box.
        """
        swept = self.sweep()
        if swept is None:
            return None
        start, turn = swept
        ends = [self.at(start), self.at(start + turn)]
        xs, ys = [x for x, _ in ends], [y for _, y in ends]
        # Where θ is a quarter turn the ellipse touches its box, and the box
        # says where exactly: its right, bottom, left and top, in order of θ.
        left, right = sorted((self.x1, self.x2))
        top, bottom = sorted((self.y1, self.y2))
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        touches = ((right, cy), (cx, bottom), (left, cy), (cx, top))
        for quarter, (x, y) in enumerate(touches):
            if (start - quarter * math.pi / 2) % (2 * math.pi) <= -turn + _SLACK:
                xs.append(x)
                ys.append(y)
        return (
            math.floor(min(xs) + _SLACK),
            math.floor(min(ys) + _SLACK),
            math.ceil(max(xs) - _SLACK),
            math.ceil(max(ys) - _SLACK),
        )


@dataclass(frozen=True)
class SymbolFile:
    """One symbol's contents, in its own coordinates.

    ``pins`` are in SPICE order, the order a netlist lists the terminals in.
    ``lines``, ``rects`` and ``circles`` are each two corner points; a circle's
    are the corners of its box. ``attrs`` are the ``SYMATTR`` lines in file
    order, and ``symbol_type`` is the ``SymbolType`` (``CELL``, ``BLOCK``),
    empty when the file states none. ``unread_pins`` holds each ``PIN`` or
    pin-order line that did not read.
    """

    pins: tuple[PinInfo, ...] = ()
    lines: tuple[Box, ...] = ()
    rects: tuple[Box, ...] = ()
    circles: tuple[Box, ...] = ()
    arcs: tuple[SymbolArc, ...] = ()
    windows: tuple[Window, ...] = ()
    attrs: tuple[tuple[str, str], ...] = ()
    symbol_type: str = ""
    unread_pins: tuple[str, ...] = ()

    def attr(self, name: str) -> str:
        """The value of the last ``SYMATTR name`` line, or ``""`` when there is none."""
        return next((value for key, value in reversed(self.attrs) if key == name), "")

    @property
    def description(self) -> str:
        return self.attr("Description")

    @property
    def prefix(self) -> str:
        """The netlist prefix (``R``, ``QN``, ``MN``, ``X``); its first letter is the
        element class LTspice netlists the part as."""
        return self.attr("Prefix")

    @property
    def body(self) -> BBox | None:
        """The smallest box around what the symbol draws, its pins apart;
        ``None`` for a symbol that draws nothing. An arc counts as what is
        drawn of it (``SymbolArc.extent``)."""
        drawn = [extent for arc in self.arcs if (extent := arc.extent()) is not None]
        points: list[tuple[int, int]] = []
        for x1, y1, x2, y2 in (*self.lines, *self.rects, *self.circles, *drawn):
            points += [(x1, y1), (x2, y2)]
        return BBox.from_points(points)

    @property
    def bbox(self) -> BBox:
        """The smallest box around the body and the pins; empty at the origin for
        a symbol that has neither."""
        points = [(pin.x, pin.y) for pin in self.pins]
        body = self.body
        if body is not None:
            points += [(body.x1, body.y1), (body.x2, body.y2)]
        return BBox.from_points(points) or BBox(0, 0, 0, 0)


def leading_ints(words: Sequence[str], count: int) -> list[int] | None:
    """The first ``count`` of ``words`` as integers, or ``None`` if there are
    fewer or one is not a number."""
    if len(words) < count:
        return None
    try:
        return [int(word) for word in words[:count]]
    except ValueError:
        return None


def value_of(line: str) -> str:
    """What follows the second word of ``line``: the value of an attribute line."""
    parts = line.split(None, 2)
    return parts[2].strip() if len(parts) > 2 else ""


def read_window(words: Sequence[str]) -> Window | None:
    """A ``WINDOW n x y [align [size]]`` line, given as its words.

    A line that stops short reads as left-aligned at the normal size, in a
    symbol file and under a symbol on a sheet alike.
    """
    numbers = leading_ints(words[1:], 3)
    if numbers is None:
        return None
    size = leading_ints(words[5:], 1)
    return Window(
        numbers[0],
        numbers[1],
        numbers[2],
        words[4] if len(words) > 4 else "Left",
        size[0] if size is not None else 2,
    )


def read_symbol(text: str) -> SymbolFile:
    """Read a symbol file's text. Never raises: a line that does not read is left out."""
    source = [line.strip() for line in text.splitlines()]
    pins: list[PinInfo] = []
    boxes: dict[str, list[Box]] = {"LINE": [], "RECTANGLE": [], "CIRCLE": []}
    arcs: list[SymbolArc] = []
    windows: list[Window] = []
    attrs: list[tuple[str, str]] = []
    symbol_type = ""
    unread_pins: list[str] = []

    index = 0
    while index < len(source):
        line = source[index]
        words = line.split()
        index += 1
        if not words:
            continue
        keyword = words[0]
        if keyword in boxes:
            # KEYWORD <weight> x1 y1 x2 y2: the numbers start at the third word.
            numbers = leading_ints(words[2:], 4)
            if numbers is not None:
                boxes[keyword].append((numbers[0], numbers[1], numbers[2], numbers[3]))
        elif keyword == "ARC":
            numbers = leading_ints(words[2:], 8)
            if numbers is not None:
                arcs.append(SymbolArc(*numbers))
        elif keyword == "WINDOW":
            window = read_window(words)
            if window is not None:
                windows.append(window)
        elif keyword == "SYMATTR" and len(words) > 1:
            attrs.append((words[1], value_of(line)))
        elif keyword == "SymbolType" and len(words) > 1:
            symbol_type = line.split(None, 1)[1].strip()
        elif keyword == "PIN":
            at = leading_ints(words[1:], 2)
            if at is None:
                # The lines under a pin that did not read are not read as its.
                unread_pins.append(line)
                continue
            name = ""
            order = 0
            while index < len(source) and source[index].startswith("PINATTR"):
                attribute = source[index]
                if attribute.startswith("PINATTR PinName"):
                    name = value_of(attribute)
                elif attribute.startswith("PINATTR SpiceOrder"):
                    number = leading_ints(attribute.split()[-1:], 1)
                    if number is None:
                        unread_pins.append(attribute)
                    else:
                        order = number[0]
                index += 1
            pins.append(PinInfo(name=name, order=order, x=at[0], y=at[1]))

    pins.sort(key=lambda pin: pin.order)
    return SymbolFile(
        pins=tuple(pins),
        lines=tuple(boxes["LINE"]),
        rects=tuple(boxes["RECTANGLE"]),
        circles=tuple(boxes["CIRCLE"]),
        arcs=tuple(arcs),
        windows=tuple(windows),
        attrs=tuple(attrs),
        symbol_type=symbol_type,
        unread_pins=tuple(unread_pins),
    )
