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

from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
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

    @property
    def qualified_name(self) -> str:
        """``REF.PIN``, or the reference alone for a pin with no name."""
        return f"{self.reference}.{self.name}" if self.name else self.reference


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
    its point. ``runs`` are the runs of wire the netlister keeps, each on its
    net. A run it drops (``dropped``) joins nothing, and ``dropped_at`` tells
    which a point lies inside; anything put there keeps the run, so it touches
    the nets of the pins at both its ends (``touched_at``). Labels of one name
    are one net only through ``folded``, and ``name_key`` is the form the
    netlister compares names in. ``pins`` and ``labels`` are what the nets were
    made over, in the order given.
    """

    def __init__(
        self,
        pins: dict[PinAt, int],
        labels: dict[LabelAt, int],
        runs: Sequence[tuple[Segment, int]],
        dropped: Sequence[DroppedRun],
        name_key: Callable[[str], str],
    ) -> None:
        self._pin_net = pins
        self._label_net = labels
        self.pins: tuple[PinAt, ...] = tuple(pins)
        self.labels: tuple[LabelAt, ...] = tuple(labels)
        #: The runs the netlister leaves out, in the order it reports them.
        self.dropped = tuple(dropped)
        #: The form two names are compared in: one key, one name.
        self.name_key = name_key
        self._pins_at: dict[Point, list[PinAt]] = {}
        for pin in pins:
            self._pins_at.setdefault(pin.point, []).append(pin)
        self._label_points = {label.point: net for label, net in labels.items()}
        #: How many pins, labels and kept runs each net holds.
        self._held = Counter([*pins.values(), *labels.values(), *(net for _run, net in runs)])
        self._runs = SegmentIndex((run, (run, net)) for run, net in runs)
        self._dropped = SegmentIndex((run.segment, run) for run in self.dropped)
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

    def pins_on(self, net: int) -> tuple[PinAt, ...]:
        """Every pin on ``net`` once folded: on any net that folds with it."""
        return self._pins_by_root.get(self._folded(net), ())

    def labels_on(self, net: int) -> tuple[LabelAt, ...]:
        """Every label on ``net`` once folded: on any net that folds with it."""
        return self._labels_by_root.get(self._folded(net), ())

    @cached_property
    def _pins_by_root(self) -> dict[int, tuple[PinAt, ...]]:
        return _grouped(self._pin_net, self._folded)

    @cached_property
    def _labels_by_root(self) -> dict[int, tuple[LabelAt, ...]]:
        return _grouped(self._label_net, self._folded)

    def unconnected(self, pin: PinAt) -> bool:
        """Whether nothing but ``pin`` is on its net: no other pin, no label and
        no run of wire the netlister keeps."""
        return self._held[self._pin_net[pin]] == 1

    def pins_at(self, point: Point) -> tuple[PinAt, ...]:
        """Every pin on ``point``."""
        return tuple(self._pins_at.get(point, ()))

    def nets_at(self, point: Point) -> tuple[int, ...]:
        """Every net with a pin or label on ``point`` or a run of wire the
        netlister keeps through it."""
        found = {self._pin_net[pin] for pin in self._pins_at.get(point, ())}
        if point in self._label_points:
            found.add(self._label_points[point])
        found.update(net for _run, net in self._runs.at(point))
        return tuple(sorted(found))

    def touched_at(self, point: Point) -> tuple[int, ...]:
        """Every net something put on ``point`` would be one node with: those of
        ``nets_at``, and of the pins at both ends of each dropped run the point
        lies inside, since what is put there keeps the run."""
        found = set(self.nets_at(point))
        found.update(self._pin_net[end] for run in self.dropped_at(point) for end in run.ends)
        return tuple(sorted(found))

    def runs_through(self, point: Point) -> list[tuple[Segment, tuple[int, ...]]]:
        """Each run of wire through ``point``, with the nets something put there
        would touch on it: a run the netlister keeps, ends included, is on its
        own net; a run it drops, between its ends only (``dropped_at``), is on
        the nets of the pins at both ends."""
        found = [(run, (net,)) for run, net in self._runs.at(point)]
        found += [
            (run.segment, tuple(self._pin_net[end] for end in run.ends))
            for run in self.dropped_at(point)
        ]
        return found
        return tuple(sorted(found))

    def dropped_at(self, point: Point) -> tuple[DroppedRun, ...]:
        """Every dropped run ``point`` lies inside, its two ends left out."""
        return tuple(run for run in self._dropped.at(point) if point not in run.segment)


Member = PinAt | LabelAt
#: The members of one net, as the later of two states places them.
Group = tuple[Member, ...]


@dataclass(frozen=True)
class NetChange:
    """How a later state of a sheet's nets differs from an earlier one, over
    the pins and labels both hold (:func:`net_change`)."""

    #: Each earlier net the later state divides, as its pieces.
    split: tuple[tuple[Group, ...], ...] = ()
    #: Each later net made of nets the earlier state kept apart, as those nets.
    joined: tuple[tuple[Group, ...], ...] = ()


def net_change(
    before: Nets,
    after: Nets,
    *,
    folded: bool = True,
    leaving: Callable[[Member], bool] = lambda _member: False,
) -> NetChange:
    """Which nets an edit split and which it joined: the comparison every edit
    that changes wiring reads to ask what it did to the nets (a move, the wire
    cleanup of a removed part, a duplicated wire's removal, a replaced part).
    Placing a pin, a label or a route asks instead what the one point it lands
    on would join, before anything is drawn.

    ``before`` and ``after`` are one sheet's nets before and after the edit,
    made over its parts and labels in the same order, so a pin is the same pin
    in both by its part's place and its SpiceOrder, and a label by its place
    among the labels, wherever each now lies; two states that disagree on
    which part or label stands in a place are refused. Members only one state
    holds, and those ``leaving`` picks out of ``after``, are left out. Nets are
    compared ``folded`` (nets that share a name are one) unless asked for
    wiring alone.
    """

    def keyed(nets: Nets) -> dict[tuple[str, int, int], tuple[Member, int]]:
        root = nets.folded if folded else (lambda net: net)
        found: dict[tuple[str, int, int], tuple[Member, int]] = {}
        for pin in nets.pins:
            found["pin", pin.part, pin.order] = (pin, root(nets.of_pin(pin)))
        for label in nets.labels:
            found["label", label.index, 0] = (label, root(nets.of_label(label)))
        return found

    earlier, later = keyed(before), keyed(after)
    for key, (member, _net) in later.items():
        if key in earlier and _identity(earlier[key][0]) != _identity(member):
            raise ValueError(
                f"net_change compares one sheet's states, but {key} is "
                f"{_identity(earlier[key][0])} before and {_identity(member)} after"
            )
    shared = [
        key for key, (member, _net) in later.items() if key in earlier and not leaving(member)
    ]

    def regrouped(
        outer: dict[tuple[str, int, int], tuple[Member, int]],
        inner: dict[tuple[str, int, int], tuple[Member, int]],
    ) -> tuple[tuple[Group, ...], ...]:
        nets: dict[int, dict[int, list[Member]]] = {}
        for key in shared:
            pieces = nets.setdefault(outer[key][1], {})
            pieces.setdefault(inner[key][1], []).append(later[key][0])
        return tuple(
            tuple(tuple(piece) for piece in pieces.values())
            for pieces in nets.values()
            if len(pieces) > 1
        )

    return NetChange(split=regrouped(earlier, later), joined=regrouped(later, earlier))


def _identity(member: Member) -> str:
    """What stands in a member's place: a pin's part (whose symbol, and so its
    pins' names, an edit may change), a label's text."""
    return f"part {member.reference}" if isinstance(member, PinAt) else f"label {member.text!r}"


def _grouped(members: dict[_T, int], root: Callable[[int], int]) -> dict[int, tuple[_T, ...]]:
    """``members`` by the representative ``root`` gives each one's net."""
    grouped: dict[int, list[_T]] = {}
    for member, net in members.items():
        grouped.setdefault(root(net), []).append(member)
    return {net: tuple(group) for net, group in grouped.items()}
