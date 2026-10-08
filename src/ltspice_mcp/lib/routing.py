"""Whether a proposed route may be drawn on a sheet.

``judge`` takes a sheet as plain data, the view and the net partition the sheet
rules read, and a :class:`Route` someone proposes, and returns what its rules
find: the findings that refuse the route, the ones that only advise, and the
junctions the route would make on the way. Nothing here resolves an endpoint,
chooses a path or writes a wire. The caller decides the route; this says what
drawing it would do.

``RULES`` lists the rules, with what each does to a proposal. They are a
registry of their own, apart from the rules of a whole sheet in
``sheet_findings``: a route rule reads a proposal, and a sheet has none.

The rules are not all independent. Three of them, a route running over a wire,
an end's leg running along the wire it ends on, and a route touching wiring of
another net, hand on which wires have already refused the route, so that one
wire is not complained of twice; they are made in that order as one stage.

Nothing here reads a file or a symbol, and nothing depends on the event loop.
"""

from __future__ import annotations

from collections.abc import Callable, Container, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ltspice_mcp.lib.connectivity import (
    NetPartition,
    label_folded_nets,
    net_members,
    point_on_segment,
)
from ltspice_mcp.lib.sheet_findings import Finding, SheetView

Point = tuple[int, int]
Wire = tuple[int, int, int, int]


@dataclass(frozen=True)
class RouteRule:
    """One thing that can be said of a proposed route.

    ``on_a_proposal`` is what a finding of the rule does to the route: it
    ``blocks`` it, and the route is not drawn, or it ``advises``, and the
    route is drawn with the finding said.
    """

    rule_id: str
    family: Literal["electrical", "drawing"]
    on_a_proposal: Literal["blocks", "advises"]
    summary: str


_RULES = (
    RouteRule(
        "route_diagonal",
        "drawing",
        "blocks",
        "A segment of the route that is neither horizontal nor vertical.",
    ),
    RouteRule(
        "route_through_pin",
        "electrical",
        "blocks",
        "The route passes over a pin that is not on its net, and would connect it.",
    ),
    RouteRule(
        "route_over_wire",
        "electrical",
        "blocks",
        "The route runs along an existing wire, and would join it along that stretch.",
    ),
    RouteRule(
        "route_along_its_end",
        "electrical",
        "blocks",
        "An end on a wire's interior whose leg runs along that wire.",
    ),
    RouteRule(
        "route_touches_net",
        "electrical",
        "blocks",
        "The route touches wiring or a label of a net it does not end on, and would merge it.",
    ),
    RouteRule(
        "route_passes_own_pin",
        "electrical",
        "advises",
        "The route passes over a pin already wired to its net, and joins it there.",
    ),
    RouteRule(
        "route_crosses_wire",
        "drawing",
        "advises",
        "The route crosses a wire where neither ends, which LTspice leaves unjoined.",
    ),
    RouteRule(
        "route_touches_own_net",
        "electrical",
        "advises",
        "The route touches wiring or a label already on its net, and joins it there.",
    ),
    RouteRule(
        "route_long",
        "drawing",
        "advises",
        "The route is more than 400 units long.",
    ),
    RouteRule(
        "route_crosses_part",
        "drawing",
        "advises",
        "A segment of the route crosses the box of a part it does not end on.",
    ),
)

#: Every rule made of a proposed route, in the order ``judge`` makes them.
RULES: Mapping[str, RouteRule] = {rule.rule_id: rule for rule in _RULES}


@dataclass(frozen=True)
class Route:
    """A route someone proposes.

    ``points`` are its corners from one end to the other, repeats removed, and
    ``segments`` the wires between them. ``ends`` are its two end points and
    ``anchors`` a point on the net each end was on before the route, which is
    the end itself for a pin or a label and a point of the wire for an end on
    a wire's interior. ``end_names`` are the ends as they were asked for, for
    wording, and ``ends_by_coordinate`` says which were given as a coordinate
    rather than as a pin or a net. ``end_parts`` are the parts the ends are
    pins of.
    """

    points: tuple[Point, ...]
    segments: tuple[Wire, ...]
    ends: tuple[Point, Point]
    anchors: tuple[Point, Point]
    end_names: tuple[str, str]
    ends_by_coordinate: tuple[bool, bool]
    end_parts: frozenset[str] = frozenset()


@dataclass
class Judgement:
    """What the rules find of a route.

    ``refusals`` are the findings of rules that block it; with any, the route
    is not to be drawn, and the rules that only measure an accepted route have
    not been made. ``advisories`` are the findings of rules that advise.
    ``junctions`` are the points other than its two ends where the route joins
    wiring already on its net, and each end that lands on a wire's interior.
    """

    refusals: list[Finding] = field(default_factory=list)
    advisories: list[Finding] = field(default_factory=list)
    junctions: list[dict[str, object]] = field(default_factory=list)


def wires_through(coord: Point, segments: Sequence[Wire]) -> list[Wire]:
    """The segments ``coord`` lies on, ends included, in the order given.

    A point on a wire's interior touches that wire the way its end would: LTspice
    joins anything placed there (see ``connectivity.partition``). More than one
    segment comes back where wires meet, overlap or cross.
    """
    return [seg for seg in segments if point_on_segment(coord, seg[:2], seg[2:])]


def segment_text(seg: Wire) -> str:
    """A wire as a message names it: ``(x1,y1)->(x2,y2)``."""
    return f"({seg[0]},{seg[1]})->({seg[2]},{seg[3]})"


def segment_json(seg: Wire) -> dict[str, dict[str, int]]:
    """A wire as a response carries it: ``{from: {x, y}, to: {x, y}}``."""
    return {"from": {"x": seg[0], "y": seg[1]}, "to": {"x": seg[2], "y": seg[3]}}


def collinear_overlap(a: Wire, b: Wire) -> bool:
    """True iff two orthogonal segments lie on one line and share a stretch of it."""
    if a[0] == a[2] == b[0] == b[2]:
        lo, hi = max(min(a[1], a[3]), min(b[1], b[3])), min(max(a[1], a[3]), max(b[1], b[3]))
        return hi > lo
    if a[1] == a[3] == b[1] == b[3]:
        lo, hi = max(min(a[0], a[2]), min(b[0], b[2])), min(max(a[0], a[2]), max(b[0], b[2]))
        return hi > lo
    return False


def _crossing(x: int, y: int, wire: Wire) -> Finding:
    # LTspice leaves a crossing where neither wire ends unjoined (the
    # crossing_wires sheet under tests/fixtures/t_junctions), so the route
    # creates no connection there and the nets stay apart. It is reported, not
    # refused: the only cost is a reader taking the crossing for a junction.
    return Finding(
        "route_crosses_wire",
        f"The route crosses the wire {segment_text(wire)} at ({x},{y}) where neither "
        "ends; LTspice leaves a plain crossing unjoined, so the two stay separate "
        "nets. To join them, end the route on that wire instead.",
        points=((x, y),),
    )


def judge(view: SheetView, partition: NetPartition, route: Route) -> Judgement:
    """What the route rules find of ``route`` on the sheet ``view`` and ``partition`` show.

    The rules that block are all made, so a refusal names everything wrong with
    the route at once, in the order: a diagonal segment; a pin passed over; a
    wire run along, an end's leg along its own wire, and other wiring touched.
    The rules that measure an accepted route, its length and the parts it
    crosses, are made only when none blocks.
    """
    said = Judgement()
    wires = list(view.wires)
    segments = list(route.segments)
    ends = set(route.ends)
    net_of = label_folded_nets(partition)
    pin_coords = partition.pin_owners.keys()

    for sx1, sy1, sx2, sy2 in segments:
        if sx1 != sx2 and sy1 != sy2:
            said.refusals.append(
                Finding(
                    "route_diagonal",
                    f"Diagonal wire ({sx1},{sy1})->({sx2},{sy2}): not orthogonal",
                    points=((sx1, sy1), (sx2, sy2)),
                )
            )

    # A pin is safe if it is already wired to the route's net: an existing wire
    # reaches both that pin and one of the route's ends, as at a T-junction
    # onto a power rail. The test is that one wire, not the whole net.
    def pin_on_target_net(px: int, py: int) -> bool:
        for ex1, ey1, ex2, ey2 in wires:
            wire_pts = {(ex1, ey1), (ex2, ey2)}
            if (px, py) in wire_pts and wire_pts & ends:
                return True
        return False

    # The exemption is by the exact coordinate of an end, NOT by whole part:
    # the OTHER pin of an end's part still lies on the route and must be
    # refused, or a waypoint landing on it shorts the part unremarked.
    for part in view.parts:
        for name, px, py in part.pins:
            if (px, py) in ends:
                continue
            # A pin at the shared corner of two consecutive segments is on
            # both, so the route is tested once per pin.
            if not wires_through((px, py), segments):
                continue
            pin_label = f"{part.ref}.{name}"
            if pin_on_target_net(px, py):
                said.junctions.append({"x": px, "y": py, "via": "pin", "pin": pin_label})
                said.advisories.append(
                    Finding(
                        "route_passes_own_pin",
                        f"The route passes over {pin_label} at ({px},{py}), already wired "
                        "to this net; LTspice joins it there.",
                        refs=(part.ref,),
                        points=((px, py),),
                        facts={"pin": name},
                    )
                )
                continue
            said.refusals.append(
                Finding(
                    "route_through_pin",
                    f"Wire passes through {pin_label} at ({px},{py}): "
                    "will create unintended connection",
                    refs=(part.ref,),
                    points=((px, py),),
                    facts={"pin": name},
                )
            )

    _wiring_met(said, wires, segments, ends, route, partition, net_of, pin_coords)
    if said.refusals:
        return said

    total_length = sum(abs(sx2 - sx1) + abs(sy2 - sy1) for sx1, sy1, sx2, sy2 in segments)
    if total_length > 400:
        said.advisories.append(
            Finding(
                "route_long",
                f"Long wire run ({total_length} units): consider placing components closer "
                "or adding a local net label",
                facts={"length": total_length},
            )
        )

    for sx1, sy1, sx2, sy2 in segments:
        for part in view.parts:
            if part.ref in route.end_parts or part.box is None:
                continue
            box = part.box
            if sy1 == sy2:
                wy = sy1
                wx_min, wx_max = min(sx1, sx2), max(sx1, sx2)
                if box.y1 < wy < box.y2 and wx_min < box.x2 and wx_max > box.x1:
                    said.advisories.append(
                        Finding(
                            "route_crosses_part",
                            f"Wire at y={wy} crosses {part.ref} bounding box "
                            f"({box.x1},{box.y1})-({box.x2},{box.y2})",
                            refs=(part.ref,),
                            points=((sx1, sy1), (sx2, sy2)),
                        )
                    )
            elif sx1 == sx2:
                wx = sx1
                wy_min, wy_max = min(sy1, sy2), max(sy1, sy2)
                if box.x1 < wx < box.x2 and wy_min < box.y2 and wy_max > box.y1:
                    said.advisories.append(
                        Finding(
                            "route_crosses_part",
                            f"Wire at x={wx} crosses {part.ref} bounding box "
                            f"({box.x1},{box.y1})-({box.x2},{box.y2})",
                            refs=(part.ref,),
                            points=((sx1, sy1), (sx2, sy2)),
                        )
                    )
    return said


def _wiring_met(
    said: Judgement,
    wires: list[Wire],
    segments: list[Wire],
    ends: set[Point],
    route: Route,
    partition: NetPartition,
    net_of: Callable[[Point], Point],
    pin_coords: Container[Point],
) -> None:
    """The three rules about existing wiring the route meets, in their order.

    ``flagged`` is the wires that have already refused the route; a later rule
    says nothing more of one.
    """
    # Running along an existing wire is refused unless that wire already ends
    # at one of the route's ends (an intended T-junction). A plain crossing,
    # where neither wire ends, is only reported: LTspice leaves it unjoined.
    # One with a label on the point is not plain, since a label there joins
    # both wires (the label_at_crossing recording), and is left to the contact
    # rule below.
    flagged: set[int] = set()
    for sx1, sy1, sx2, sy2 in segments:
        for ext_index, (ex1, ey1, ex2, ey2) in enumerate(wires):
            ext_endpoints = {(ex1, ey1), (ex2, ey2)}
            if ext_endpoints & ends:
                continue
            if sx1 == sx2 and ex1 == ex2 and sx1 == ex1:
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_y = max(new_min, ext_min)
                    if (sx1, overlap_y) not in ends:
                        flagged.add(ext_index)
                        low, high = max(new_min, ext_min), min(new_max, ext_max)
                        said.refusals.append(
                            Finding(
                                "route_over_wire",
                                f"Wire overlap at x={sx1} between y={low} "
                                f"and y={high}: will create unintended junction",
                                points=((sx1, low), (sx1, high)),
                            )
                        )
                        break
            elif sy1 == sy2 and ey1 == ey2 and sy1 == ey1:
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if new_min < ext_max and new_max > ext_min:
                    overlap_x = max(new_min, ext_min)
                    if (overlap_x, sy1) not in ends:
                        flagged.add(ext_index)
                        low, high = max(new_min, ext_min), min(new_max, ext_max)
                        said.refusals.append(
                            Finding(
                                "route_over_wire",
                                f"Wire overlap at y={sy1} between x={low} "
                                f"and x={high}: will create unintended junction",
                                points=((low, sy1), (high, sy1)),
                            )
                        )
                        break
            elif sx1 == sx2 and ey1 == ey2:
                cross_x, cross_y = sx1, ey1
                new_min, new_max = min(sy1, sy2), max(sy1, sy2)
                ext_min, ext_max = min(ex1, ex2), max(ex1, ex2)
                if (
                    new_min < cross_y < new_max
                    and ext_min < cross_x < ext_max
                    and (cross_x, cross_y) not in ends
                    and (cross_x, cross_y) not in partition.label_texts
                ):
                    said.advisories.append(_crossing(cross_x, cross_y, wires[ext_index]))
            elif sy1 == sy2 and ex1 == ex2:
                cross_x, cross_y = ex1, sy1
                new_min, new_max = min(sx1, sx2), max(sx1, sx2)
                ext_min, ext_max = min(ey1, ey2), max(ey1, ey2)
                if (
                    ext_min < cross_y < ext_max
                    and new_min < cross_x < new_max
                    and (cross_x, cross_y) not in ends
                    and (cross_x, cross_y) not in partition.label_texts
                ):
                    said.advisories.append(_crossing(cross_x, cross_y, wires[ext_index]))

    # An end on a wire's interior is a T-junction onto that wire. Its leg must
    # leave the wire: one running along it overlaps the wire it joins, which
    # the rule above exempts because it starts at an end.
    for name, by_coordinate, coord, leg in (
        (route.end_names[0], route.ends_by_coordinate[0], route.ends[0], segments[0]),
        (route.end_names[1], route.ends_by_coordinate[1], route.ends[1], segments[-1]),
    ):
        if not by_coordinate or coord in pin_coords:
            continue
        for ext_index, wire in enumerate(wires):
            if ext_index in flagged or not point_on_segment(coord, wire[:2], wire[2:]):
                continue
            if collinear_overlap(leg, wire):
                flagged.add(ext_index)
                said.refusals.append(
                    Finding(
                        "route_along_its_end",
                        f"Wire overlap: the leg from {name} runs along "
                        f"the wire {segment_text(wire)} it ends on; leave that wire at a "
                        "right angle",
                        points=(coord, wire[:2], wire[2:]),
                    )
                )
            elif coord not in {wire[:2], wire[2:]}:
                said.junctions.append(
                    {"x": coord[0], "y": coord[1], "via": "endpoint", "wire": segment_json(wire)}
                )

    # Contact. LTspice joins a wire wherever another wire's end, a pin or a
    # label touches it (see connectivity.partition), so a waypoint on existing
    # wiring, or a route passing through an existing wire's end or a label,
    # joins the route there as an end would. Onto a net the route already joins
    # that is a redundant junction, reported; onto any other net it would merge
    # a net nobody named, so it is refused, pointing at the coordinate end that
    # makes the same T on purpose. Pins are the pin rule's, and a wire that has
    # already refused the route is not reported twice.
    end_nets = {net_of(route.anchors[0]), net_of(route.anchors[1])}
    vertices = [v for v in route.points[1:-1] if v not in ends and v not in pin_coords]
    # (point, net it touches) -> (kind, the wire touched, or None for a label).
    # One entry per point and net, the first found, in the order found.
    contacts: dict[tuple[Point, Point], tuple[str, Wire | None]] = {}
    for ext_index, wire in enumerate(wires):
        if ext_index in flagged:
            continue
        touched = net_of(wire[:2])
        for v in vertices:
            if point_on_segment(v, wire[:2], wire[2:]):
                contacts.setdefault((v, touched), ("waypoint", wire))
        for end in (wire[:2], wire[2:]):
            if end in ends or end in vertices or end in pin_coords:
                continue
            if wires_through(end, segments):
                contacts.setdefault((end, touched), ("wire_end", wire))
    for coord in sorted(partition.label_texts):
        if coord in ends or coord in pin_coords or not wires_through(coord, segments):
            continue
        # A label joins whatever passes through its point, on a wire or off
        # one. On a wire that has already refused the route it is not said twice.
        under = [
            index
            for index, wire in enumerate(wires)
            if point_on_segment(coord, wire[:2], wire[2:])
        ]
        if not under or any(index not in flagged for index in under):
            contacts.setdefault((coord, net_of(coord)), ("label", None))

    def describe_net(net: Point) -> str:
        on_net = net_members(partition, net_of, net)
        labels = sorted({t for c in on_net for t in partition.label_texts.get(c, ())})
        if labels:
            return "net " + ", ".join(f"'{t}'" for t in labels)
        pins = sorted(
            f"{ref}.{name}" for c in on_net for ref, name in partition.pin_owners.get(c, ())
        )
        if pins:
            more = " and others" if len(pins) > 3 else ""
            return "the net of " + ", ".join(pins[:3]) + more
        return "a wire no pin or label is on"

    for ((cx, cy), touched), (via, wire) in contacts.items():
        entry: dict[str, object] = {"x": cx, "y": cy, "via": via}
        if wire is None:
            texts = sorted(partition.label_texts[(cx, cy)])
            what = "the net label " + ", ".join(f"'{t}'" for t in texts) + f" at ({cx},{cy})"
            entry["label"] = texts[0]
        else:
            entry["wire"] = segment_json(wire)
            what = (
                f"the wire {segment_text(wire)} at waypoint ({cx},{cy})"
                if via == "waypoint"
                else f"the end of the wire {segment_text(wire)} at ({cx},{cy})"
            )
        if touched in end_nets:
            said.junctions.append(entry)
            said.advisories.append(
                Finding(
                    "route_touches_own_net",
                    f"The route touches {what}, already on {describe_net(touched)}; "
                    "LTspice joins them there.",
                    points=((cx, cy),),
                    facts={"via": via},
                )
            )
            continue
        remedy = (
            f"To join it on purpose, end a route there with "
            f'{{"x": {cx}, "y": {cy}}} as from_pin or to_pin; otherwise move the '
            "waypoint off it."
            if via == "waypoint"
            else "Reroute around it."
        )
        said.refusals.append(
            Finding(
                "route_touches_net",
                f"Route touches {what}, on {describe_net(touched)}: LTspice joins "
                f"wiring wherever it touches, so this would merge that net. {remedy}",
                points=((cx, cy),),
                facts={"via": via},
            )
        )
