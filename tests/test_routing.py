"""The route rules on sheets made for the purpose, as plain data.

What the planner answers on real sheets is held to a record in
``test_route_planner_record.py``. This holds each rule to what it means, with
no editor and no file: a view, a partition and a route.
"""

from __future__ import annotations

from itertools import pairwise

from ltspice_mcp.lib.connectivity import NetPartition, partition
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.routing import RULES, Judgement, Route, judge
from ltspice_mcp.lib.sheet_findings import Part, SheetView

Point = tuple[int, int]


def resistor(ref: str, x: int, y: int) -> Part:
    """An upright two-pin part: pins 48 above and below its origin, 32 wide."""
    return Part(
        ref, box=BBox(x - 16, y - 48, x + 16, y + 48), pins=(("1", x, y - 48), ("2", x, y + 48))
    )


def sheet(
    parts: tuple[Part, ...] = (),
    wires: tuple[tuple[int, int, int, int], ...] = (),
    labels: tuple[tuple[int, int, str], ...] = (),
) -> tuple[SheetView, NetPartition]:
    view = SheetView(parts=parts, wires=wires, labels=labels)
    nets = partition(
        [((x, y), (part.ref, name)) for part in parts for name, x, y in part.pins],
        [((x, y), text) for x, y, text in labels],
        [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in wires],
    )
    return view, nets


def route(
    *points: Point,
    by_coordinate: tuple[bool, bool] = (False, False),
    anchors: tuple[Point, Point] | None = None,
    end_parts: tuple[str, ...] = (),
) -> Route:
    """A route through ``points`` in the order given, from the first to the last."""
    start, end = points[0], points[-1]
    segments = tuple((a[0], a[1], b[0], b[1]) for a, b in pairwise(points))
    return Route(
        points=points,
        segments=segments,
        ends=(start, end),
        anchors=anchors or (start, end),
        end_names=("FROM", "TO"),
        ends_by_coordinate=by_coordinate,
        end_parts=frozenset(end_parts),
    )


def rules(found: Judgement) -> tuple[list[str], list[str]]:
    return [one.rule for one in found.refusals], [one.rule for one in found.advisories]


class TestTheRegistry:
    def test_a_rule_blocks_or_advises_and_says_what_it_is(self) -> None:
        assert [rule.on_a_proposal for rule in RULES.values()].count("blocks") == 5
        assert {rule.on_a_proposal for rule in RULES.values()} == {"blocks", "advises"}
        for rule in RULES.values():
            assert rule.rule_id.startswith("route_")
            assert rule.summary.endswith(".") and len(rule.summary) > 20

    def test_it_is_apart_from_the_rules_of_a_whole_sheet(self) -> None:
        from ltspice_mcp.lib.sheet_findings import RULES as SHEET_RULES

        assert not set(RULES) & set(SHEET_RULES)


class TestARouteThatMayBeDrawn:
    def test_between_two_pins_with_nothing_in_the_way(self) -> None:
        view, nets = sheet(parts=(resistor("R1", 100, 300), resistor("R2", 300, 300)))
        found = judge(view, nets, route((100, 252), (300, 252), end_parts=("R1", "R2")))
        assert rules(found) == ([], [])
        assert found.junctions == []

    def test_a_crossing_where_neither_wire_ends_is_only_said(self) -> None:
        view, nets = sheet(wires=((0, 100, 400, 100),))
        found = judge(view, nets, route((200, 0), (200, 200)))
        assert rules(found) == ([], ["route_crosses_wire"])
        assert found.advisories[0].points == ((200, 100),)

    def test_a_pin_already_wired_to_one_of_its_ends_is_a_junction(self) -> None:
        # A wire runs from the route's own start to R2's pin; the route goes on past it.
        parts = (resistor("R1", 0, 48), resistor("R2", 200, 48), resistor("R3", 400, 48))
        view, nets = sheet(parts=parts, wires=((0, 0, 200, 0),))
        found = judge(view, nets, route((0, 0), (400, 0), end_parts=("R1", "R3")))
        assert rules(found) == ([], ["route_passes_own_pin"])
        assert found.junctions == [{"x": 200, "y": 0, "via": "pin", "pin": "R2.1"}]
        assert found.advisories[0].refs == ("R2",)

    def test_a_part_it_crosses_is_said_unless_it_ends_on_that_part(self) -> None:
        view, nets = sheet(parts=(resistor("R1", 200, 100),))
        across = route((0, 100), (400, 100))
        assert rules(judge(view, nets, across)) == ([], ["route_crosses_part"])
        own = route((0, 100), (400, 100), end_parts=("R1",))
        assert rules(judge(view, nets, own)) == ([], [])

    def test_a_long_one_is_said_with_its_length(self) -> None:
        view, nets = sheet()
        found = judge(view, nets, route((0, 0), (0, 300), (200, 300)))
        assert rules(found) == ([], ["route_long"])
        assert found.advisories[0].facts == {"length": 500}


class TestARouteThatIsRefused:
    def test_a_diagonal(self) -> None:
        view, nets = sheet()
        found = judge(view, nets, route((0, 0), (64, 64)))
        assert rules(found) == (["route_diagonal"], [])
        assert found.refusals[0].detail == "Diagonal wire (0,0)->(64,64): not orthogonal"

    def test_over_a_pin_of_another_net(self) -> None:
        parts = (resistor("R1", 0, 48), resistor("R2", 200, 48), resistor("R3", 400, 48))
        view, nets = sheet(parts=parts)
        found = judge(view, nets, route((0, 0), (400, 0), end_parts=("R1", "R3")))
        assert rules(found) == (["route_through_pin"], [])
        refusal = found.refusals[0]
        assert (refusal.refs, refusal.points, refusal.facts) == (
            ("R2",),
            ((200, 0),),
            {"pin": "1"},
        )

    def test_along_a_wire_and_that_wire_is_complained_of_once(self) -> None:
        # The wire's end (300,0) lies on the route too, which the contact rule
        # would report, had the wire not already refused the route.
        view, nets = sheet(wires=((100, 0, 300, 0),))
        found = judge(view, nets, route((0, 0), (400, 0)))
        assert rules(found) == (["route_over_wire"], [])
        assert found.refusals[0].points == ((100, 0), (300, 0))

    def test_an_end_whose_leg_runs_along_the_wire_it_is_on(self) -> None:
        view, nets = sheet(wires=((0, 0, 400, 0),))
        along = route((200, 0), (500, 0), by_coordinate=(True, False), anchors=((0, 0), (500, 0)))
        assert rules(judge(view, nets, along))[0] == ["route_along_its_end"]
        away = route(
            (200, 0), (200, 200), by_coordinate=(True, False), anchors=((0, 0), (200, 200))
        )
        found = judge(view, nets, away)
        assert rules(found) == ([], [])
        assert found.junctions == [
            {
                "x": 200,
                "y": 0,
                "via": "endpoint",
                "wire": {"from": {"x": 0, "y": 0}, "to": {"x": 400, "y": 0}},
            }
        ]

    def test_through_the_end_of_a_wire_of_another_net(self) -> None:
        view, nets = sheet(wires=((200, 0, 200, 100),))
        found = judge(view, nets, route((0, 0), (400, 0)))
        assert rules(found) == (["route_touches_net"], [])
        assert found.refusals[0].facts == {"via": "wire_end"}
        assert "Reroute around it." in found.refusals[0].detail

    def test_a_waypoint_on_a_wire_of_another_net(self) -> None:
        """Straight across it is a plain crossing; with a corner point on it,
        the route touches the wire there as an end would."""
        view, nets = sheet(wires=((0, 100, 400, 100),))
        found = judge(view, nets, route((200, 0), (200, 100), (200, 200)))
        assert rules(found) == (["route_touches_net"], [])
        assert found.refusals[0].facts == {"via": "waypoint"}
        assert 'end a route there with {"x": 200, "y": 100}' in found.refusals[0].detail

    def test_across_a_wire_at_the_point_its_label_is_on(self) -> None:
        """A label where two wires cross joins them, so this is no plain crossing."""
        view, nets = sheet(wires=((0, 100, 400, 100),), labels=((200, 100, "SIG"),))
        found = judge(view, nets, route((200, 0), (200, 200)))
        assert rules(found) == (["route_touches_net"], [])
        refusal = found.refusals[0]
        assert (refusal.points, refusal.facts) == (((200, 100),), {"via": "label"})
        assert "the net label 'SIG' at (200,100), on net 'SIG'" in refusal.detail

    def test_onto_its_own_net_the_same_contact_is_a_junction(self) -> None:
        view, nets = sheet(wires=((0, 100, 400, 100),), labels=((200, 100, "SIG"),))
        found = judge(view, nets, route((200, 0), (200, 200), anchors=((0, 100), (200, 200))))
        assert rules(found) == ([], ["route_touches_own_net"])
        assert found.junctions == [{"x": 200, "y": 100, "via": "label", "label": "SIG"}]

    def test_a_refused_route_is_not_measured(self) -> None:
        view, nets = sheet(parts=(resistor("R1", 200, 600),))
        found = judge(view, nets, route((0, 0), (1000, 1000)))
        assert rules(found) == (["route_diagonal"], [])

    def test_every_finding_is_of_a_rule_that_does_what_was_done_with_it(self) -> None:
        view, nets = sheet(
            parts=(resistor("R1", 200, 248),),
            wires=((0, 100, 400, 100), (300, 0, 300, 50), (50, 300, 150, 300)),
            labels=((100, 100, "SIG"),),
        )
        found = judge(
            view, nets, route((0, 0), (0, 400), (100, 400), (100, 0), (400, 0), (500, 64))
        )
        assert found.refusals and found.advisories
        assert {RULES[one.rule].on_a_proposal for one in found.refusals} == {"blocks"}
        assert {RULES[one.rule].on_a_proposal for one in found.advisories} == {"advises"}
