"""Which points of a sheet are one net, by the rules LTspice's netlister applies.

``nets_by_ltspice`` takes a sheet's pins, labels and wires and returns its
nets as a :class:`~ltspice_mcp.lib.nets.Nets`, every rule below applied. This
is the one place those rules are written down, for the editor that refuses a
wrong connection, the checker that reports one, and the trace that explains
one. ``Nets`` holds no LTspice rule (``lib/nets.py``), so another schematic
tool's rules would go in a builder of their own.

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

``partition`` (with :class:`NetPartition`, ``label_folded_nets``,
``NamedNets`` and ``net_members``) groups the same points by coordinate. It
cannot hold two of the rules, the wire LTspice leaves out between two pins of
one part when nothing else is on it and the pins of one part that share a
point, and is kept only until its readers move to ``nets_by_ltspice``. New code
reads ``Nets``.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

import itertools
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from typing import NamedTuple

from ltspice_mcp.lib.nets import (
    DroppedRun,
    LabelAt,
    Nets,
    PinAt,
    Point,
    Segment,
    SegmentIndex,
    fold_by_name,
    point_on_segment,
)

# A point on a wire need not be at its ends, nor the wire horizontal or
# vertical: LTspice draws diagonal wires and connects a pin or label anywhere
# along one, as on any other wire (``point_on_segment``).


def build_on_wire_predicate(segments: Iterable[Segment]) -> Callable[[Point], bool]:
    """``on_wire(point)``: whether ``point`` lies on any of ``segments``, ends
    included, with each lookup reading only the wires on the point's own row
    or column (:class:`~ltspice_mcp.lib.nets.SegmentIndex`)."""
    return SegmentIndex((segment, None) for segment in segments).touches


def build_wire_counter(segments: Iterable[Segment]) -> Callable[[Point], int]:
    """``count(point)``: how many of ``segments`` the point lies on, ends
    included. A wire end that touches another wire is a point two wires pass
    through."""
    index = SegmentIndex((segment, None) for segment in segments)
    return lambda point: len(index.at(point))


def crossing_points(segments: Iterable[Segment]) -> list[Point]:
    """Where a horizontal and a vertical wire pass through each other, each
    away from its ends, in the order the pairs are met.

    Two such wires stay apart unless something on the point joins them; a
    third wire ending there, or a pin or label on it, is the caller's to ask.
    """
    wires = list(segments)
    horizontal = [(a, b) for a, b in wires if a[1] == b[1] and a[0] != b[0]]
    vertical = [(a, b) for a, b in wires if a[0] == b[0] and a[1] != b[1]]
    points: dict[Point, None] = {}
    for (hx1, y), (hx2, _y) in horizontal:
        for (x, vy1), (_x, vy2) in vertical:
            if min(hx1, hx2) < x < max(hx1, hx2) and min(vy1, vy2) < y < max(vy1, vy2):
                points[(x, y)] = None
    return list(points)


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

    The work is in finding which points lie on which wire (:func:`_join`).
    """
    pin_owners: dict[Point, list[tuple[str, str]]] = {}
    for coord, owner in pins:
        pin_owners.setdefault(coord, []).append(owner)
    label_texts: dict[Point, set[str]] = {}
    for coord, text in labels:
        label_texts.setdefault(coord, set()).add(text)
    find, members = _join([*pin_owners, *label_texts], list(segments))
    return NetPartition(root=find, members=members, pin_owners=pin_owners, label_texts=label_texts)


def _join(
    points: Iterable[Point], wires: list[Segment]
) -> tuple[Callable[[Point], Point], dict[Point, set[Point]]]:
    """Group ``points`` and the wires' ends by the wires that join them: each
    point's representative, and every point by representative.

    Points are indexed by row and by column, so a horizontal or vertical wire
    looks only at the points on its own line; a diagonal one, which a sheet
    rarely has, looks at all of them.
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
    for coord in points:
        interest.add(coord)
        find(coord)

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
    return find, members


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
    find = fold_by_name(
        (
            (root, text)
            for root, coords in part.members.items()
            for coord in coords
            for text in part.label_texts.get(coord, ())
        ),
        name_key,
    )
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
    coordinate (``node_coords``), a corner/junction (its incident segment ends
    are not exactly two ends of one orientation), or a point another wire
    passes through: the two ends there are each on that wire, and join it, as
    a single end on a wire's interior does (recorded:
    ``wire_ends_meeting_on_wire_interior``). Only pure pass-through vertices
    (degree-2, both ends collinear, nothing on them) are merged across. This is
    what makes a *collinear* waypoint disappear: an in-line bend leaves only
    bare pass-through vertices, so the run collapses back to one segment; a
    bend that turns a corner leaves the corner vertices as breaks, so its
    segments stay split. Verticals (``x1==x2``) and horizontals (``y1==y2``) are merged
    per-line by interval union split at breaks; any diagonal passes through
    unchanged.
    """
    ends: dict[tuple[int, int], list[str]] = defaultdict(list)
    verticals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    horizontals: dict[int, list[tuple[int, int]]] = defaultdict(list)
    merged: list[tuple[int, int, int, int]] = []
    wires: list[Segment] = []
    for x1, y1, x2, y2 in segments:
        if (x1, y1) == (x2, y2):
            continue  # zero-length record (hand-corrupted WIRE) — nothing to merge
        wires.append(((x1, y1), (x2, y2)))
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

    through = build_wire_counter(wires)

    def _breaks(coord: tuple[int, int]) -> bool:
        es = ends.get(coord, [])
        return (
            coord in node_coords or len(es) != 2 or len(set(es)) != 1 or through(coord) > len(es)
        )

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
    pin_owners: Mapping[Point, Sequence[tuple[Hashable, str]]],
    labels: Iterable[Point],
) -> tuple[list[tuple[int, int, int, int]], list[dict]]:
    """The collinear-merged runs of ``segments``, cut at every pin and label,
    and the runs among them LTspice drops (:func:`same_instance_dropped_segments`)."""
    runs = _merge_collinear_runs(list(segments), set(pin_owners) | set(labels))
    return runs, _dropped_among(runs, pin_owners)


def _dropped_among(
    runs: list[tuple[int, int, int, int]],
    pin_owners: Mapping[Point, Sequence[tuple[Hashable, str]]],
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

    Every rule above is applied (:func:`nets_by_ltspice`), same-named labels as
    one net among them. Parts are told apart by reference, which on any sheet
    LTspice netlists names one part. A net with no pin on it is left out, since
    nothing in a netlist shows it. Names are given as :func:`name_key` has
    them, so two spellings of one name, or of ground, compare equal.

    Two sheets with equal signatures connect the same pins and name the same
    nets, wherever their parts and wires are drawn, so an edit meant to change
    only the drawing can be checked by comparing the two. The arguments are
    those of :func:`partition`, with each part's pins listed in SpiceOrder,
    as a symbol gives them.
    """
    parts: dict[str, int] = {}
    placed = [
        PinAt(point, parts.setdefault(reference, len(parts)), order, reference, name)
        for order, (point, (reference, name)) in enumerate(pins)
    ]
    named = [LabelAt(point, index, text) for index, (point, text) in enumerate(labels)]
    nets = nets_by_ltspice(placed, named, list(segments))
    on_net: dict[int, set[tuple[str, str]]] = {}
    for pin in placed:
        on_net.setdefault(nets.folded(nets.of_pin(pin)), set()).add((pin.reference, pin.name))
    names: dict[int, set[str]] = {}
    for label in named:
        names.setdefault(nets.folded(nets.of_label(label)), set()).add(nets.name_key(label.text))
    return frozenset(
        (frozenset(group), frozenset(names.get(net, ()))) for net, group in on_net.items()
    )


# ---------------------------------------------------------------------------
# LTspice's nets
# ---------------------------------------------------------------------------


def nets_by_ltspice(
    pins: Sequence[PinAt], labels: Sequence[LabelAt], wires: Sequence[Segment]
) -> Nets:
    """The nets LTspice's netlister makes of a sheet with ``pins``, ``labels``
    and ``wires``, by every rule of this module.

    Wires are merged into runs as LTspice merges them, cut at every pin, label
    and junction, and the runs it drops join nothing; a point on one touches
    the nets of both its pins, because anything put there keeps the run. Of
    one part's pins on a point with no wire or label there, only the last in
    SpiceOrder joins what else is on the point. Names compare as
    :func:`name_key` has them.
    """
    pins_at: dict[Point, list[PinAt]] = {}
    for pin in pins:
        pins_at.setdefault(pin.point, []).append(pin)
    labelled = {label.point for label in labels}
    owners = {point: [(pin.part, pin.name) for pin in here] for point, here in pins_at.items()}
    runs, drops = _runs_and_drops([(a[0], a[1], b[0], b[1]) for a, b in wires], owners, labelled)
    gone = {drop["segment"] for drop in drops}
    kept = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in runs if (x1, y1, x2, y2) not in gone]
    find, members = _join([*pins_at, *labelled], kept)
    point_nets = {
        point: net for net, root in enumerate(sorted(members)) for point in members[root]
    }
    fresh = itertools.count(len(members))
    pin_nets: dict[PinAt, int] = {}
    for point, here in pins_at.items():
        last: dict[int, PinAt] = {}
        if len(here) > 1 and point not in labelled and len(members[find(point)]) == 1:
            # Nothing but pins here: each part's last pin in SpiceOrder touches
            # the others, and its other pins on the point touch nothing.
            last = {pin.part: pin for pin in sorted(here, key=lambda one: one.order)}
        for pin in here:
            alone = bool(last) and last[pin.part] is not pin
            pin_nets[pin] = next(fresh) if alone else point_nets[point]

    def end(point: Point, part: int, name: str) -> PinAt:
        return next(pin for pin in pins_at[point] if pin.part == part and pin.name == name)

    dropped = [
        DroppedRun(
            ((x1, y1), (x2, y2)),
            (
                end((x1, y1), drop["ref"], drop["pins"][0]),
                end((x2, y2), drop["ref"], drop["pins"][1]),
            ),
        )
        for drop in drops
        for x1, y1, x2, y2 in [drop["segment"]]
    ]
    segments = [(run, (point_nets[run[0]],)) for run in kept]
    segments += [(run.segment, tuple(pin_nets[pin] for pin in run.ends)) for run in dropped]
    label_nets = {label: point_nets[label.point] for label in labels}
    return Nets(pin_nets, label_nets, segments, dropped, name_key)
