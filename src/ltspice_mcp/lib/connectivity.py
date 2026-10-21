"""Which points of a sheet are one net, by the rules LTspice's netlister applies.

``partition`` takes a sheet's pins, labels and wire segments as plain
coordinates and returns a :class:`NetPartition`: every pin, label and wire end
grouped by the wiring that joins it. This is the one place those rules are
written down, for the editor that refuses a wrong connection, the checker that
reports one, and the trace that explains one.

The rules, each recorded from the ``-netlist`` export of LTspice 26 and of
LTspice XVII (the ``connectivity`` cases; ``docs/TESTING.md``, "Recorded
LTspice behaviour"; the sheets are also in ``tests/fixtures/t_junctions/``):

- a wire joins its two ends;
- a pin, a label or another wire's end anywhere **on** a wire, its interior
  included, is joined to that wire, and the wire is left whole;
- two wires that only cross, neither ending at the crossing, stay apart;
- a label at a crossing joins both wires, and so does a pin;
- the pins of two parts on one point are joined with no wire at all;
- of one part's pins on such a point, with no wire and no label there, only
  the one highest in SpiceOrder is: it joins the other parts' pins, and the
  part's other pins there are each connected to nothing;
- collinear wires that overlap are joined where an end of one lies on the other;
- a point on a diagonal wire is on that wire as on any other.

Labels that share a name are one node as well, without any wire between them.
Names that differ only in case are one name (the ``node-name-case``
recording), and a ``GND`` label, in any case, is ground as ``0`` is (the
``gnd_label`` recordings); ``name_key`` is the one form names are compared in.
The partition does not apply that: it is wiring only, because a caller often
needs to tell the two apart (a wire that would join two *named* nets is a
short, two labels of one name are not). ``label_folded_nets`` applies it on
top, and ``signature`` reduces a sheet to the pin groups and names a netlist
would show, so two drawings of one circuit compare equal.

Two of the rules are in ``signature`` alone, because a partition is by
coordinate and cannot hold them: the wire LTspice leaves out between two
pins of one part when nothing else is on it, and the pins of one part that
share a point.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

import itertools
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import NamedTuple

Point = tuple[int, int]
Segment = tuple[Point, Point]


def point_on_segment(point: Point, v1: Point, v2: Point) -> bool:
    """True iff ``point`` lies on the wire segment ``v1 → v2``, ends included.

    A wire need not be horizontal or vertical: LTspice draws diagonal ones and
    connects a pin or label that sits on one anywhere along its length, as it
    does on any other wire.
    """
    px, py = point
    x1, y1 = v1
    x2, y2 = v2
    if (x2 - x1) * (py - y1) != (y2 - y1) * (px - x1):
        return False
    return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)


def build_on_wire_predicate(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> Callable[[tuple[int, int]], bool]:
    """Return an ``on_wire(coord)`` predicate with the same semantics as
    ``point_on_segment`` but O(1)-amortised per query.

    The naive ``any(point_on_segment(coord, *seg) for seg in segments)``
    scan is O(segments) per coord; calling it once per pin makes a pass
    over a sheet's pins O(pins × segments), which becomes the dominant
    cost during a long ``add_component`` build. Bucketing
    horizontal segments by row and vertical by column collapses each query
    to the handful of segments sharing that row/column.
    """
    endpoints: set[tuple[int, int]] = set()
    horiz: dict[int, list[tuple[int, int]]] = {}
    vert: dict[int, list[tuple[int, int]]] = {}
    diagonal: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (x1, y1), (x2, y2) in segments:
        endpoints.add((x1, y1))
        endpoints.add((x2, y2))
        if y1 == y2 and x1 != x2:
            horiz.setdefault(y1, []).append((min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            vert.setdefault(x1, []).append((min(y1, y2), max(y1, y2)))
        elif x1 != x2:
            # A sheet rarely has one, so each is tested on its own. A point
            # on its interior is on the wire, as LTspice has it.
            diagonal.append(((x1, y1), (x2, y2)))
        # A wire of no length is its one point, which the ends cover.

    def on_wire(coord: tuple[int, int]) -> bool:
        if coord in endpoints:
            return True
        px, py = coord
        if any(xmin <= px <= xmax for xmin, xmax in horiz.get(py, ())):
            return True
        if any(ymin <= py <= ymax for ymin, ymax in vert.get(px, ())):
            return True
        return any(point_on_segment(coord, a, b) for a, b in diagonal)

    return on_wire


def build_wire_counter(
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
) -> Callable[[tuple[int, int]], int]:
    """Return ``count(coord)``: how many of ``segments`` the point lies on, ends included.

    The same test as ``point_on_segment``, with horizontal and vertical wires
    looked up by their row or column. A wire end that touches another wire is a
    point two wires pass through.
    """
    horiz: dict[int, list[tuple[int, int]]] = {}
    vert: dict[int, list[tuple[int, int]]] = {}
    other: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (x1, y1), (x2, y2) in segments:
        if y1 == y2 and x1 != x2:
            horiz.setdefault(y1, []).append((min(x1, x2), max(x1, x2)))
        elif x1 == x2 and y1 != y2:
            vert.setdefault(x1, []).append((min(y1, y2), max(y1, y2)))
        else:
            other.append(((x1, y1), (x2, y2)))

    def count(coord: tuple[int, int]) -> int:
        px, py = coord
        return (
            sum(1 for lo, hi in horiz.get(py, ()) if lo <= px <= hi)
            + sum(1 for lo, hi in vert.get(px, ()) if lo <= py <= hi)
            + sum(1 for a, b in other if point_on_segment(coord, a, b))
        )

    return count


class NetPartition(NamedTuple):
    """Connected-component view of a schematic's nets.

    ``root`` maps any interest coordinate to its net's canonical
    representative; ``members`` maps a root to every coordinate on that net;
    ``pin_owners`` maps a coordinate to the ``(ref, pin_name)`` pairs sitting
    there; ``label_texts`` maps a coordinate to the FLAG texts placed there.
    """

    root: Callable[[Point], Point]
    members: dict[Point, set[Point]]
    pin_owners: dict[Point, list[tuple[str, str]]]
    label_texts: dict[Point, set[str]]


def partition(
    pins: Iterable[tuple[Point, tuple[str, str]]],
    labels: Iterable[tuple[Point, str]],
    segments: Iterable[Segment],
) -> NetPartition:
    """Group pins, labels and wire ends by the wiring that joins them.

    ``pins`` are ``(coordinate, (reference, pin name))``, ``labels`` are
    ``(coordinate, text)`` and ``segments`` are wires as two points each; a
    caller checking a route it has not drawn yet lists its segments with the
    sheet's own.

    The work is in finding which points lie on which wire. Points are indexed
    by row and by column, so a horizontal or vertical wire looks only at the
    points on its own line; a diagonal one, which a sheet rarely has, looks at
    all of them.
    """
    parent: dict[Point, Point] = {}

    def find(p: Point) -> Point:
        if p not in parent:
            parent[p] = p
            return p
        while parent[p] != p:
            parent[p] = parent[parent[p]]
            p = parent[p]
        return p

    def union(a: Point, b: Point) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    interest: set[Point] = set()
    pin_owners: dict[Point, list[tuple[str, str]]] = {}
    for coord, owner in pins:
        interest.add(coord)
        find(coord)
        pin_owners.setdefault(coord, []).append(owner)
    label_texts: dict[Point, set[str]] = {}
    for coord, text in labels:
        interest.add(coord)
        find(coord)
        label_texts.setdefault(coord, set()).add(text)

    wires = list(segments)
    for v1, v2 in wires:
        interest.add(v1)
        interest.add(v2)
        union(v1, v2)

    rows: dict[int, list[int]] = {}
    columns: dict[int, list[int]] = {}
    for x, y in interest:
        rows.setdefault(y, []).append(x)
        columns.setdefault(x, []).append(y)
    for line in (*rows.values(), *columns.values()):
        line.sort()

    for v1, v2 in wires:
        (x1, y1), (x2, y2) = v1, v2
        if v1 == v2:
            continue  # a zero-length wire has no interior for anything to lie on
        if y1 == y2:
            xs = rows[y1]
            on_it: Iterable[Point] = (
                (x, y1) for x in xs[bisect_left(xs, min(x1, x2)) : bisect_right(xs, max(x1, x2))]
            )
        elif x1 == x2:
            ys = columns[x1]
            on_it = (
                (x1, y) for y in ys[bisect_left(ys, min(y1, y2)) : bisect_right(ys, max(y1, y2))]
            )
        else:
            on_it = (pt for pt in interest if point_on_segment(pt, v1, v2))
        for pt in on_it:
            if pt != v1 and pt != v2:
                # Always the point under the wire, never the wire under the
                # point: the wire's representative is then the same whatever
                # order its points are met in.
                union(pt, v1)

    members: dict[Point, set[Point]] = {}
    for p in parent:
        members.setdefault(find(p), set()).add(p)

    return NetPartition(root=find, members=members, pin_owners=pin_owners, label_texts=label_texts)


#: The name of ground. A ground flag is a label like any other to the
#: partition, and no name at all when asking whether two *named* nets are joined.
GROUND = "0"


def fold_case(name: str) -> str:
    """A node name in the form two names are compared in: names that differ
    only in case are one node (the ``node-name-case`` recording).

    A fixed mapping, the same on every platform, for a sheet's labels and a
    deck's nodes alike.
    """
    return name.lower()


def is_ground(name: str) -> bool:
    """True for a label LTspice exports as node ``0``: ``0`` itself, or ``GND``
    in any case (the ``gnd_label`` and ``gnd_label_lower_case`` recordings)."""
    return name == GROUND or fold_case(name) == "gnd"


def name_key(name: str) -> str:
    """A label's name as LTspice compares it: :data:`GROUND` for every ground
    flag, and otherwise the name with its case folded. Labels with one key are
    one node, and a net's names are told apart by their keys alone.

    The netlist net query compares a deck's node names the same way: the
    ``deck-structure`` recording has ``gnd`` as node 0 in a deck too.
    ``netlist_graph.canon_net``, which compares whole decks, also takes
    ``gnd!`` for ground and strips the name; no recording has ``gnd!``.
    """
    return GROUND if is_ground(name) else fold_case(name)


def name_keys(names: Iterable[str]) -> set[str]:
    """The :func:`name_key` of each of ``names``."""
    return {name_key(name) for name in names}


def named_labels(labels: Iterable[str]) -> set[str]:
    """``labels`` without the ground flags among them: ground is no name when
    asking whether two *named* nets are joined."""
    return {label for label in labels if not is_ground(label)}


def label_folded_nets(part: NetPartition) -> Callable[[Point], Point]:
    """Map a coordinate to its electrical net's representative.

    The partition connects by wire only; LTspice also makes every FLAG with the
    same name one node, so wired nets that share a label name fold into one
    here, names compared as :func:`name_key` has them. Two coordinates are on
    the same netlist node iff this returns the same representative for both.
    """
    parent: dict[Point, Point] = {}

    def find(r: Point) -> Point:
        parent.setdefault(r, r)
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    first_root: dict[str, Point] = {}
    for root, coords in part.members.items():
        for coord in coords:
            for text in part.label_texts.get(coord, ()):
                key = name_key(text)
                if key not in first_root:
                    first_root[key] = root
                    continue
                ra, rb = find(first_root[key]), find(root)
                if ra != rb:
                    parent[ra] = rb

    return lambda coord: find(part.root(coord))


def net_members(part: NetPartition, net_of: Callable[[Point], Point], net: Point) -> set[Point]:
    """Every pin, label and wire-end coordinate on ``net``, a ``net_of`` value."""
    return {c for root, coords in part.members.items() if net_of(root) == net for c in coords}


class Joined(NamedTuple):
    """What a pin or a label put on a point of a sheet would make one node."""

    #: The nets among them that carry a name, each as its names, ground left out.
    #: Two or more is two named nets made one: a short.
    named: tuple[frozenset[str], ...]
    #: How many separate nets the wires through the point were: more than one
    #: where wires cross and nothing has joined them.
    wired: int


class NamedNets:
    """A sheet's nets with the names each carries, for asking what a point would join.

    Made once from a partition of the sheet as it is, without the thing about
    to be placed; ``joined_at`` is then asked of each point. Nets are the
    label-folded ones, so nets that share a name, in any case, and nets that
    each carry a ground flag, are already one net here. A net's names are as
    the sheet spells them, two spellings of one name included.
    """

    def __init__(self, part: NetPartition) -> None:
        self._net_of = label_folded_nets(part)
        self._names: dict[Point, set[str]] = defaultdict(set)
        #: The one net each name is on, by its key: labels of one key are one net.
        self._net_named: dict[str, Point] = {}
        for coord, texts in part.label_texts.items():
            net = self._net_of(coord)
            self._names[net].update(named_labels(texts))
            for key in name_keys(texts):
                self._net_named[key] = net

    def has_name(self, name: str) -> bool:
        """True if a label of the sheet already carries ``name``, in any case."""
        return name_key(name) in self._net_named

    def net_named(self, name: str) -> Point | None:
        """The net a label of ``name`` is on, in any case; None if no label carries it."""
        return self._net_named.get(name_key(name))

    def joined_at(
        self,
        on_wires: Iterable[Point],
        at: Point | None = None,
        name: str | None = None,
    ) -> Joined:
        """What something put on a point would make one node.

        ``on_wires`` is a point of each wire the place lies on. A pin or a label
        there joins every one of them, both wires where two cross (the
        ``pin_at_crossing`` and ``label_at_crossing`` recordings).

        ``at`` is the place itself, given when what already sits there is
        joined too: a label is one node with a pin or a label on its point. A
        place the sheet has nothing on reads as an unnamed net of its own.

        ``name`` is a net name the thing carries, which makes it one node with
        every net already of that name, in any case. Ground is not given as
        one. Nets that each carry a ground flag are one net already, and a net
        with a name may be grounded whatever else is: that is what a ground
        flag is for, and no join of the flag's own making.
        """
        wired = {self._net_of(point) for point in on_wires}
        nets = set(wired)
        if at is not None:
            nets.add(self._net_of(at))
        if name is not None and (named_net := self._net_named.get(name_key(name))) is not None:
            nets.add(named_net)
        named = tuple(frozenset(self._names[net]) for net in sorted(nets) if self._names[net])
        return Joined(named, len(wired))


def _merge_collinear_runs(
    segments: list[tuple[int, int, int, int]],
    node_coords: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Collapse straight runs of collinear wire segments into single segments,
    mirroring LTspice's netlist-time wire merge.

    A vertex breaks a run — stays its own node — when it is a pin or a label
    coordinate (``node_coords``) or a corner/junction (its incident segment ends
    are not exactly two ends of one orientation). Only pure pass-through
    vertices (degree-2, both ends collinear, nothing on them) are merged
    across. This is what
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

    # A component pin, a label or a ground flag on the interior of a run is a
    # junction too — LTspice splits the wire there — but it is not a wire
    # endpoint, so it never appears in ``ends`` and a break test over ``ends``
    # alone would miss it. Add every such point sitting on the line as a cut
    # candidate; ``_emit`` keeps only those strictly inside a run's span, so one
    # at a run end (already a natural node) or off any run adds nothing.
    cuts_on_x: dict[int, set[int]] = defaultdict(set)
    cuts_on_y: dict[int, set[int]] = defaultdict(set)
    for cx, cy in itertools.chain((end for end in ends if _breaks(end)), node_coords):
        cuts_on_x[cx].add(cy)
        cuts_on_y[cy].add(cx)
    for x, iv in verticals.items():
        _emit(iv, cuts_on_x.get(x, set()), vertical=True, line=x)
    for y, iv in horizontals.items():
        _emit(iv, cuts_on_y.get(y, set()), vertical=False, line=y)
    return merged


def same_instance_dropped_segments(
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]],
    segments: list[tuple[int, int, int, int]],
    labels: Iterable[Point],
) -> list[dict]:
    """Wire segments LTspice discards from the exported netlist.

    LTspice drops a wire run whose two ends both land exactly on pins of the
    SAME single component instance (recorded from the ``-netlist`` export of
    LTspice 26 and of LTspice XVII: such a run never reaches the netlist, so the
    two pins stay on separate nodes and the drawn tie has no electrical
    effect). Several runs still get kept, all in the same recordings: a run
    spanning two *different* instances, a same-instance tie that turns a corner
    OUT OF LINE with the two pins, and one with something on it between the
    pins (another part's pin, another wire's end, a label or a ground flag, on
    its interior or at a vertex). A waypoint that stays *collinear* with the
    pins and has nothing on it does NOT survive — LTspice merges the in-line
    segments back into one and drops it — so the segments are collinear-merged
    (:func:`_merge_collinear_runs`) before this check, which is what catches an
    all-in-line waypoint route as well as the bare direct wire. A net label on
    one pin does not rescue the run.

    Returns one dict per dropped run with ``segment`` (the merged ``(x1, y1, x2,
    y2)`` tuple), ``ref`` (the shared instance), and ``pins`` (the two pin
    names on that instance), ordered deterministically. ``pin_owners`` maps each
    pin coordinate to its ``(ref, pin_name)`` owners, as
    ``NetPartition.pin_owners`` holds it, and ``labels`` are the points of the
    sheet's labels, ground flags among them.
    """
    return _runs_and_drops(segments, pin_owners, labels)[1]


def _runs_and_drops(
    segments: Iterable[tuple[int, int, int, int]],
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]],
    labels: Iterable[Point],
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """The collinear-merged runs of ``segments``, cut at every pin and label,
    and the runs among them LTspice drops (:func:`same_instance_dropped_segments`)."""
    runs = _merge_collinear_runs(list(segments), set(pin_owners) | set(labels))
    return runs, _dropped_among(runs, pin_owners)


def _dropped_among(
    runs: list[tuple[int, int, int, int]],
    pin_owners: dict[tuple[int, int], list[tuple[str, str]]],
) -> list[dict]:
    """:func:`same_instance_dropped_segments` for runs that are already merged."""
    dropped: list[dict] = []
    for seg in runs:
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


Signature = frozenset[tuple[frozenset[tuple[str, str]], frozenset[str]]]


def signature(
    pins: Iterable[tuple[Point, tuple[str, str]]],
    labels: Iterable[tuple[Point, str]],
    segments: Iterable[Segment],
) -> Signature:
    """The circuit a sheet draws, as a netlist would show it: each net as the
    pins on it and the names it carries.

    Every rule above is applied: wiring by :func:`partition`, same-named labels
    as one net, the wire LTspice leaves out because it runs straight between
    two pins of one part (:func:`same_instance_dropped_segments`), and a
    part's pins that share a point with no wire or label on it, of which only
    the highest in SpiceOrder touches anything. A net with no pin on it is
    left out, since nothing in a netlist shows it. Names are given as
    :func:`name_key` has them, so two spellings of one name, or of ground,
    compare equal.

    Two sheets with equal signatures connect the same pins and name the same
    nets, wherever their parts and wires are drawn, so an edit meant to change
    only the drawing can be checked by comparing the two. The arguments are
    those of :func:`partition`, with each part's pins listed in SpiceOrder,
    as a symbol gives them.
    """
    pins = list(pins)
    labels = list(labels)
    owners: dict[Point, list[tuple[str, str]]] = {}
    for coord, owner in pins:
        owners.setdefault(coord, []).append(owner)
    runs, drops = _runs_and_drops(
        [(a[0], a[1], b[0], b[1]) for a, b in segments],
        owners,
        (coord for coord, _text in labels),
    )
    dropped = {drop["segment"] for drop in drops}
    netlisted = [
        ((x1, y1), (x2, y2)) for x1, y1, x2, y2 in runs if (x1, y1, x2, y2) not in dropped
    ]
    part = partition(pins, labels, netlisted)
    net_of = label_folded_nets(part)
    on_net: dict[Point, set[tuple[str, str]]] = {}
    names: dict[Point, set[str]] = {}
    alone: list[tuple[str, str]] = []
    for coord, at in part.pin_owners.items():
        touching = at
        if coord not in part.label_texts and part.members[part.root(coord)] == {coord}:
            # Nothing but pins here. Each part's last-listed pin is the one
            # that touches the others; its other pins on the point touch
            # nothing, not even each other.
            last = {ref: (ref, name) for ref, name in at}
            touching = list(last.values())
            alone += [pin for pin in at if last[pin[0]] != pin]
        on_net.setdefault(net_of(coord), set()).update(touching)
    for coord, texts in part.label_texts.items():
        names.setdefault(net_of(coord), set()).update(name_keys(texts))
    nets = {(frozenset(group), frozenset(names.get(net, ()))) for net, group in on_net.items()}
    nets |= {(frozenset({pin}), frozenset()) for pin in alone}
    return frozenset(nets)
