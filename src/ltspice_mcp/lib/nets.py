"""A sheet's nets in a form that holds no schematic tool's rules.

A netlister decides which pins, labels and wires of a sheet are one net; the
rules differ between tools. ``connectivity.nets_by_ltspice`` applies
LTspice's and returns a :class:`Nets`. A reader asks the ``Nets`` which net a
pin or label is on and which nets a point touches, and compares net names with
the key the netlister gave it, so it holds no rule of its own and would read
another tool's nets the same way.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

Point = tuple[int, int]
Segment = tuple[Point, Point]

_T = TypeVar("_T")
_Net = TypeVar("_Net", bound=Hashable)


def point_on_segment(point: Point, v1: Point, v2: Point) -> bool:
    """True iff ``point`` lies on the segment ``v1 → v2``, ends included,
    whatever the segment's direction."""
    px, py = point
    x1, y1 = v1
    x2, y2 = v2
    if (x2 - x1) * (py - y1) != (y2 - y1) * (px - x1):
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


class SegmentIndex(Generic[_T]):
    """Segments, each with a value, looked up by a point on them.

    Horizontal segments are kept by row and vertical ones by column, so a
    lookup reads only the segments on the point's own line; a diagonal one, or
    one of no length, is tested on its own.
    """

    def __init__(self, items: Iterable[tuple[Segment, _T]]) -> None:
        self._rows: dict[int, list[tuple[int, int, _T]]] = {}
        self._columns: dict[int, list[tuple[int, int, _T]]] = {}
        self._others: list[tuple[Segment, _T]] = []
        for ((x1, y1), (x2, y2)), value in items:
            if y1 == y2 and x1 != x2:
                self._rows.setdefault(y1, []).append((min(x1, x2), max(x1, x2), value))
            elif x1 == x2 and y1 != y2:
                self._columns.setdefault(x1, []).append((min(y1, y2), max(y1, y2), value))
            else:
                self._others.append((((x1, y1), (x2, y2)), value))

    def touches(self, point: Point) -> bool:
        """Whether ``point`` lies on any of the segments."""
        x, y = point
        return (
            any(lo <= x <= hi for lo, hi, _value in self._rows.get(y, ()))
            or any(lo <= y <= hi for lo, hi, _value in self._columns.get(x, ()))
            or any(point_on_segment(point, a, b) for (a, b), _value in self._others)
        )

    def at(self, point: Point) -> list[_T]:
        """The value of every segment ``point`` lies on, ends included."""
        x, y = point
        found = [value for lo, hi, value in self._rows.get(y, ()) if lo <= x <= hi]
        found += [value for lo, hi, value in self._columns.get(x, ()) if lo <= y <= hi]
        found += [value for (a, b), value in self._others if point_on_segment(point, a, b)]
        return found


def fold_by_name(
    named: Iterable[tuple[_Net, str]], key: Callable[[str], str]
) -> Callable[[_Net], _Net]:
    """Each net's representative once nets that carry a name in common, as
    ``key`` compares names, are one. ``named`` gives each net with a name it
    carries; a net it does not give is its own representative."""
    parent: dict[_Net, _Net] = {}

    def find(net: _Net) -> _Net:
        parent.setdefault(net, net)
        while parent[net] != net:
            parent[net] = parent[parent[net]]
            net = parent[net]
        return net

    first: dict[str, _Net] = {}
    for net, text in named:
        name = key(text)
        if name not in first:
            first[name] = net
            continue
        ra, rb = find(first[name]), find(net)
        if ra != rb:
            parent[ra] = rb
    return find


@dataclass(frozen=True)
class PinAt:
    """A pin where its part puts it.

    ``part`` tells the part from every other part on the sheet, two that share
    a reference included, and ``order`` places the pin among the part's own
    pins in SpiceOrder.
    """

    point: Point
    part: int
    order: int
    reference: str
    name: str


@dataclass(frozen=True)
class LabelAt:
    """A label, ground flags included, where it is placed. ``index`` tells it
    from another label of the same text on the same point."""

    point: Point
    index: int
    text: str


@dataclass(frozen=True)
class DroppedRun:
    """A run of wire the netlister leaves out, between the two pins at its ends."""

    segment: Segment
    ends: tuple[PinAt, PinAt]


class Nets:
    """A sheet's nets as a netlister makes them, over its pins and labels.

    A net is an ``int``. A pin may be on a net of its own where others share
    its point. ``segments`` gives each run of wire with the nets a point on it
    touches: one for a run that joins what is on it, more for one the
    netlister drops (``dropped``). Labels of one name are one net only through
    ``folded``, and ``name_key`` is the form the netlister compares names in.
    """

    def __init__(
        self,
        pins: dict[PinAt, int],
        labels: dict[LabelAt, int],
        segments: Iterable[tuple[Segment, tuple[int, ...]]],
        dropped: Sequence[DroppedRun],
        name_key: Callable[[str], str],
    ) -> None:
        self._pin_net = pins
        self._label_net = labels
        #: The runs the netlister leaves out, in the order it reports them.
        self.dropped = tuple(dropped)
        #: The form two names are compared in: one key, one name.
        self.name_key = name_key
        self._pins_at: dict[Point, list[PinAt]] = {}
        for pin in pins:
            self._pins_at.setdefault(pin.point, []).append(pin)
        self._label_points = {label.point: net for label, net in labels.items()}
        self._segments = SegmentIndex(segments)
        self._folded = fold_by_name(((net, label.text) for label, net in labels.items()), name_key)

    def of_pin(self, pin: PinAt) -> int:
        """The net ``pin`` is on."""
        return self._pin_net[pin]

    def of_label(self, label: LabelAt) -> int:
        """The net ``label`` is on, by wiring alone."""
        return self._label_net[label]

    def folded(self, net: int) -> int:
        """``net`` with every net that carries a label of one of its names as one."""
        return self._folded(net)

    def nets_at(self, point: Point) -> tuple[int, ...]:
        """Every net with a pin or label on ``point`` or a run of wire through it."""
        found = {self._pin_net[pin] for pin in self._pins_at.get(point, ())}
        if point in self._label_points:
            found.add(self._label_points[point])
        for nets in self._segments.at(point):
            found.update(nets)
        return tuple(sorted(found))
