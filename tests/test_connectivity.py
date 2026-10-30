"""The net partition and the signature built on it, on plain coordinates.

The joining rules themselves are held against LTspice's own exports in
``test_recorded_ltspice_schematics.py``. Here the partition is held against a
plain reference that tests every point against every wire, on every sheet the
suite holds and on generated ones, so the indexed search finds exactly what
the exhaustive one does.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path

import pytest

from ltspice_mcp.lib.asc_document import parse_asc
from ltspice_mcp.lib.connectivity import (
    NamedNets,
    NetPartition,
    Point,
    Segment,
    _runs_and_drops,  # pyright: ignore[reportPrivateUsage]
    build_on_wire_predicate,
    label_folded_nets,
    name_key,
    net_members,
    nets_by_ltspice,
    partition,
    point_on_segment,
    signature,
)
from ltspice_mcp.lib.nets import LabelAt, PinAt
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file
from ltspice_mcp.lib.symbol_library import find_symbol
from tests._schematic_fixtures import SUITE_SHEETS, TESTS, suite_name

Pins = list[tuple[Point, tuple[str, str]]]
Labels = list[tuple[Point, str]]


def reference_partition(pins: Pins, labels: Labels, segments: list[Segment]) -> NetPartition:
    """The partition worked out the plain way: every point against every wire."""
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
    for v1, v2 in segments:
        interest.add(v1)
        interest.add(v2)
        union(v1, v2)
    for v1, v2 in segments:
        for pt in interest:
            if pt in (v1, v2):
                continue
            if point_on_segment(pt, v1, v2):
                union(pt, v1)
    members: dict[Point, set[Point]] = {}
    for p in parent:
        members.setdefault(find(p), set()).add(p)
    return NetPartition(root=find, members=members, pin_owners=pin_owners, label_texts=label_texts)


def assert_same(got: NetPartition, want: NetPartition) -> None:
    # The same nets, under the same representatives, listed in the same order:
    # callers key on a representative and sort by it, so a different one would
    # reorder what they report.
    assert list(got.members.items()) == list(want.members.items())
    for point in {p for group in want.members.values() for p in group}:
        assert got.root(point) == want.root(point)
    assert got.pin_owners == want.pin_owners
    assert got.label_texts == want.label_texts


def sheet_inputs(path: Path) -> tuple[Pins, Labels, list[Segment]]:
    """A sheet's pins, labels and wires, with the symbols found beside it or in
    the suite's stand-in library. A symbol found in neither has no pins here,
    which leaves the wires and labels to compare."""
    doc = parse_asc(path.read_bytes())
    pins: Pins = []
    for symbol in doc.symbols:
        asy = find_symbol(symbol.symbol, path.parent, [TESTS / "fixtures" / "symbols"])
        if asy is None:
            continue
        placed = compute_placed_geometry(parse_asy_file(asy), symbol.x, symbol.y, symbol.rotation)
        pins += [((pin["x"], pin["y"]), (symbol.reference, pin["name"])) for pin in placed["pins"]]
    labels: Labels = [((flag.x, flag.y), flag.name) for flag in doc.flags]
    segments = [((w.x1, w.y1), (w.x2, w.y2)) for w in doc.wires]
    return pins, labels, segments


def generated_sheet(seed: int) -> tuple[Pins, Labels, list[Segment]]:
    """Wires, pins and labels thrown onto a small grid, so that ends land on
    other wires, wires overlap and cross, and some run diagonally or nowhere."""
    rng = random.Random(seed)

    def point() -> Point:
        return (rng.randrange(0, 12) * 16, rng.randrange(0, 12) * 16)

    segments: list[Segment] = []
    for _ in range(rng.randrange(5, 40)):
        start = point()
        kind = rng.random()
        if kind < 0.45:
            end = (rng.randrange(0, 12) * 16, start[1])
        elif kind < 0.9:
            end = (start[0], rng.randrange(0, 12) * 16)
        else:
            end = point()
        segments.append((start, end))
    pins: Pins = [(point(), (f"X{n // 3}", f"p{n % 3}")) for n in range(rng.randrange(0, 30))]
    labels: Labels = [(point(), rng.choice(["0", "a", "b", "c"])) for _ in range(rng.randrange(8))]
    return pins, labels, segments


class TestTheIndexedSearchFindsWhatTheExhaustiveOneDoes:
    def test_the_suite_holds_sheets_with_wires_and_pins(self) -> None:
        wired = [path for path in SUITE_SHEETS if sheet_inputs(path)[2]]
        pinned = [path for path in SUITE_SHEETS if sheet_inputs(path)[0]]
        assert len(wired) >= 15 and len(pinned) >= 15

    @pytest.mark.parametrize("path", SUITE_SHEETS, ids=suite_name)
    def test_on_every_sheet_in_the_suite(self, path: Path) -> None:
        inputs = sheet_inputs(path)
        assert_same(partition(*inputs), reference_partition(*inputs))

    @pytest.mark.parametrize("seed", range(200))
    def test_on_generated_sheets(self, seed: int) -> None:
        inputs = generated_sheet(seed)
        assert_same(partition(*inputs), reference_partition(*inputs))

    def test_the_generated_sheets_hold_the_cases_that_matter(self) -> None:
        on_an_interior = diagonal = zero_length = 0
        for seed in range(200):
            pins, labels, segments = generated_sheet(seed)
            points = [p for p, _ in pins] + [p for p, _ in labels]
            points += [end for segment in segments for end in segment]
            for v1, v2 in segments:
                zero_length += v1 == v2
                diagonal += v1[0] != v2[0] and v1[1] != v2[1]
                on_an_interior += any(
                    p not in (v1, v2) and point_on_segment(p, v1, v2) for p in points
                )
        assert on_an_interior > 500 and diagonal > 100 and zero_length > 5


class TestOnAWire:
    def test_it_is_point_on_segment_for_every_kind_of_wire(self) -> None:
        segments = [
            ((0, 0), (64, 0)),
            ((96, 0), (96, 64)),
            ((0, 96), (64, 160)),
            ((200, 200), (200, 200)),
        ]
        on_wire = build_on_wire_predicate(segments)
        for x in range(-16, 232, 8):
            for y in range(-16, 232, 8):
                expected = any(point_on_segment((x, y), a, b) for a, b in segments)
                assert on_wire((x, y)) == expected, (x, y)

    def test_the_interior_of_a_diagonal_wire_is_on_it(self) -> None:
        on_wire = build_on_wire_predicate([((0, 0), (64, 64))])
        assert on_wire((32, 32)) and not on_wire((32, 48))


def nets(part: NetPartition) -> set[frozenset[Point]]:
    return {frozenset(group) for group in part.members.values()}


def joined(part: NetPartition, a: Point, b: Point) -> bool:
    return part.root(a) == part.root(b)


class TestTheRules:
    """Each rule on the smallest sheet that shows it."""

    def test_a_wire_joins_its_ends(self) -> None:
        part = partition([], [], [((0, 0), (64, 0))])
        assert joined(part, (0, 0), (64, 0))

    def test_an_end_on_another_wires_interior_is_joined_to_it(self) -> None:
        part = partition([], [], [((0, 0), (64, 0)), ((32, 0), (32, 48))])
        assert len(nets(part)) == 1

    def test_wires_that_only_cross_stay_apart(self) -> None:
        part = partition([], [], [((0, 16), (64, 16)), ((32, 0), (32, 48))])
        assert len(nets(part)) == 2
        assert not joined(part, (0, 16), (32, 0))

    def test_a_label_at_a_crossing_joins_both_wires(self) -> None:
        part = partition([], [((32, 16), "n")], [((0, 16), (64, 16)), ((32, 0), (32, 48))])
        assert len(nets(part)) == 1

    def test_a_pin_at_a_crossing_joins_both_wires(self) -> None:
        part = partition([((32, 16), ("R1", "A"))], [], [((0, 16), (64, 16)), ((32, 0), (32, 48))])
        assert len(nets(part)) == 1

    def test_a_pin_on_a_wires_interior_is_on_that_wire(self) -> None:
        part = partition([((32, 0), ("R1", "A"))], [], [((0, 0), (64, 0))])
        assert joined(part, (32, 0), (0, 0))

    def test_two_pins_on_one_point_are_joined(self) -> None:
        part = partition([((0, 0), ("R1", "B")), ((0, 0), ("R2", "A"))], [], [])
        assert part.pin_owners[(0, 0)] == [("R1", "B"), ("R2", "A")]
        assert nets(part) == {frozenset({(0, 0)})}

    def test_collinear_wires_that_overlap_are_joined(self) -> None:
        part = partition([], [], [((0, 0), (64, 0)), ((32, 0), (96, 0))])
        assert len(nets(part)) == 1

    def test_collinear_wires_with_a_gap_are_not(self) -> None:
        part = partition([], [], [((0, 0), (32, 0)), ((48, 0), (96, 0))])
        assert len(nets(part)) == 2

    def test_a_point_on_a_diagonal_wire_is_on_it(self) -> None:
        part = partition([((32, 32), ("R1", "A"))], [((16, 48), "off")], [((0, 0), (64, 64))])
        assert joined(part, (32, 32), (0, 0))
        assert not joined(part, (16, 48), (0, 0))

    def test_a_wire_of_no_length_joins_nothing(self) -> None:
        part = partition([((0, 0), ("R1", "A"))], [], [((16, 0), (16, 0))])
        assert len(nets(part)) == 2

    def test_a_point_beside_a_wire_is_not_on_it(self) -> None:
        part = partition([((32, 16), ("R1", "A"))], [], [((0, 0), (64, 0))])
        assert not joined(part, (32, 16), (0, 0))


class TestNames:
    def test_the_partition_is_wiring_only(self) -> None:
        part = partition([], [((0, 0), "vdd"), ((96, 0), "vdd")], [])
        assert not joined(part, (0, 0), (96, 0))
        assert part.label_texts == {(0, 0): {"vdd"}, (96, 0): {"vdd"}}

    def test_labels_of_one_name_are_one_net_once_folded(self) -> None:
        part = partition(
            [((16, 0), ("R1", "A")), ((112, 0), ("R2", "A")), ((200, 0), ("R3", "A"))],
            [((0, 0), "vdd"), ((96, 0), "vdd"), ((184, 0), "out")],
            [((0, 0), (16, 0)), ((96, 0), (112, 0)), ((184, 0), (200, 0))],
        )
        net_of: Callable[[Point], Point] = label_folded_nets(part)
        assert net_of((16, 0)) == net_of((112, 0))
        assert net_of((16, 0)) != net_of((200, 0))
        assert net_members(part, net_of, net_of((16, 0))) == {(0, 0), (16, 0), (96, 0), (112, 0)}


def crossing(across: str = "", down: str = "", *more: tuple[Point, str]) -> NamedNets:
    """A wire across and a wire down that cross at (200,100) and are not joined,
    each with the label given for it, and any further labels."""
    labels = [(at, name) for at, name in (((0, 100), across), ((200, 0), down)) if name]
    return NamedNets(
        partition([], [*labels, *more], [((0, 100), (400, 100)), ((200, 0), (200, 200))])
    )


#: A point of each of the two wires that cross.
BOTH = [(0, 100), (200, 0)]


class TestWhatAPointWouldJoin:
    """Asked of a sheet as it is, before the pin or the label is on it."""

    def test_on_one_wire_it_joins_that_net(self) -> None:
        said = crossing("vdd", "vss").joined_at([(0, 100)])
        assert (said.named, said.wired) == ((frozenset({"vdd"}),), 1)

    def test_on_a_crossing_it_joins_both_wires(self) -> None:
        said = crossing("vdd", "vss").joined_at(BOTH)
        assert said.wired == 2
        assert set(said.named) == {frozenset({"vdd"}), frozenset({"vss"})}

    def test_two_wires_with_no_name_between_them_are_two_nets_and_no_names(self) -> None:
        assert crossing().joined_at(BOTH) == ((), 2)
        assert crossing("vdd").joined_at(BOTH) == ((frozenset({"vdd"}),), 2)

    def test_wires_already_one_net_are_one(self) -> None:
        looped = NamedNets(
            partition(
                [],
                [((0, 100), "vdd")],
                [
                    ((0, 100), (400, 100)),
                    ((200, 0), (200, 200)),
                    ((400, 100), (400, 200)),
                    ((400, 200), (200, 200)),
                ],
            )
        )
        assert looped.joined_at(BOTH) == ((frozenset({"vdd"}),), 1)

    def test_a_name_it_carries_joins_every_net_already_of_that_name(self) -> None:
        nets = crossing("vdd", "vss", ((900, 900), "out"))
        assert nets.joined_at([(0, 100)]).named == (frozenset({"vdd"}),)
        by_name = nets.joined_at([(0, 100)], name="out")
        assert set(by_name.named) == {frozenset({"vdd"}), frozenset({"out"})}
        assert by_name.wired == 1
        # Its own name again joins nothing more, and a new name nothing at all.
        assert nets.joined_at([(0, 100)], name="vdd").named == (frozenset({"vdd"}),)
        assert nets.joined_at([(0, 100)], name="fresh").named == (frozenset({"vdd"}),)

    def test_what_sits_on_the_place_is_joined_only_when_the_place_is_given(self) -> None:
        """A label lands on what is there; a part's pin is not asked that way."""
        nets = NamedNets(partition([((50, 50), ("R1", "A"))], [((50, 50), "out")], []))
        assert nets.joined_at([]) == ((), 0)
        assert nets.joined_at([], at=(50, 50)) == ((frozenset({"out"}),), 0)
        assert nets.joined_at([], at=(700, 700)) == ((), 0)

    def test_ground_is_no_name(self) -> None:
        """Two named nets that each carry a ground are one net, of both names,
        and a third net crossing it is the only other."""
        nets = crossing("vdd", "vss", ((400, 100), "0"), ((900, 900), "com"), ((900, 900), "0"))
        (grounded,) = nets.joined_at([(0, 100)]).named
        assert grounded == frozenset({"vdd", "com"})
        assert set(nets.joined_at(BOTH).named) == {grounded, frozenset({"vss"})}
        only_ground = NamedNets(partition([], [((0, 0), "0")], [((0, 0), (64, 0))]))
        assert only_ground.joined_at([(0, 0)]) == ((), 1)


R1 = [((0, 0), ("R1", "A")), ((0, 96), ("R1", "B"))]
R2 = [((96, 0), ("R2", "A")), ((96, 96), ("R2", "B"))]


class TestSignature:
    def test_it_is_each_net_as_its_pins_and_names(self) -> None:
        assert signature(
            R1 + R2, [((48, 0), "in"), ((0, 96), "0")], [((0, 0), (96, 0))]
        ) == frozenset(
            {
                (frozenset({("R1", "A"), ("R2", "A")}), frozenset({"in"})),
                (frozenset({("R1", "B")}), frozenset({"0"})),
                (frozenset({("R2", "B")}), frozenset()),
            }
        )

    def test_one_circuit_drawn_two_ways_has_one_signature(self) -> None:
        straight = signature(R1 + R2, [], [((0, 0), (96, 0))])
        # The same two pins joined by a detour, with the parts elsewhere.
        moved = [((160, 32), ("R1", "A")), ((160, 128), ("R1", "B"))]
        moved += [((320, 64), ("R2", "A")), ((320, 160), ("R2", "B"))]
        detour = signature(
            moved,
            [],
            [((160, 32), (160, 16)), ((160, 16), (320, 16)), ((320, 16), (320, 64))],
        )
        assert straight == detour

    def test_a_wire_swapped_for_two_labels_of_one_name_is_the_same_circuit_named(self) -> None:
        wired = signature(R1 + R2, [((0, 0), "n")], [((0, 0), (96, 0))])
        labelled = signature(R1 + R2, [((0, 0), "n"), ((96, 0), "n")], [])
        assert wired == labelled

    def test_a_different_connection_is_a_different_signature(self) -> None:
        to_the_top = signature(R1 + R2, [], [((0, 0), (96, 0))])
        to_the_bottom = signature(R1 + R2, [], [((0, 0), (96, 96))])
        assert to_the_top != to_the_bottom

    def test_a_net_with_no_pin_is_not_in_it(self) -> None:
        with_a_stub = signature(R1, [((200, 200), "spare")], [((200, 200), (264, 200))])
        assert with_a_stub == signature(R1, [], [])

    def test_a_straight_wire_between_two_pins_of_one_part_joins_nothing(self) -> None:
        # LTspice leaves that wire out of the netlist, so the pins stay apart.
        assert signature(R1, [], [((0, 0), (0, 96))]) == signature(R1, [], [])
        # And still does when the run is drawn as two pieces in one line.
        assert signature(R1, [], [((0, 0), (0, 48)), ((0, 48), (0, 96))]) == signature(R1, [], [])

    def test_a_parts_pins_alone_on_one_point_touch_nothing(self) -> None:
        both_here = [((0, 0), ("R1", "1")), ((0, 0), ("R1", "2"))]
        assert signature(both_here, [], []) == frozenset(
            {
                (frozenset({("R1", "1")}), frozenset()),
                (frozenset({("R1", "2")}), frozenset()),
            }
        )

    def test_only_its_last_pin_there_touches_another_parts(self) -> None:
        pins = [((0, 0), ("R1", "1")), ((0, 0), ("R1", "2"))]
        pins += [((0, 0), ("R2", "1")), ((0, 0), ("R3", "2"))]
        assert signature(pins, [], []) == frozenset(
            {
                (frozenset({("R1", "1")}), frozenset()),
                (frozenset({("R1", "2"), ("R2", "1"), ("R3", "2")}), frozenset()),
            }
        )

    def test_a_wire_or_a_label_on_the_point_joins_every_pin_there(self) -> None:
        both_here = [((0, 0), ("R1", "1")), ((0, 0), ("R1", "2"))]
        together = frozenset({("R1", "1"), ("R1", "2")})
        assert signature(both_here, [((0, 0), "f")], []) == frozenset(
            {(together, frozenset({"f"}))}
        )
        assert signature(both_here, [], [((0, 0), (64, 0))]) == frozenset(
            {(together, frozenset())}
        )

    def test_the_same_wire_taken_out_of_line_joins_them(self) -> None:
        around = [((0, 0), (-32, 0)), ((-32, 0), (-32, 96)), ((-32, 96), (0, 96))]
        assert signature(R1, [], around) == frozenset(
            {(frozenset({("R1", "A"), ("R1", "B")}), frozenset())}
        )


# ---------------------------------------------------------------------------
# Nets over a sheet's pins and labels
# ---------------------------------------------------------------------------


def reference_signature(pins: Pins, labels: Labels, segments: list[Segment]):
    """The signature as it was first written, on the exhaustive
    :func:`reference_partition`: every pin on its point's wired net, then,
    where a point holds nothing but pins, each part's last pin left there and
    its others on nothing.

    It checks how ``nets_by_ltspice`` puts the rules together; the rules
    themselves are held to LTspice's exports in
    ``test_recorded_ltspice_schematics.py``. It goes with ``partition`` once no
    reader uses that."""
    owners: dict[Point, list[tuple[str, str]]] = {}
    for coord, owner in pins:
        owners.setdefault(coord, []).append(owner)
    runs, drops = _runs_and_drops(
        [(a[0], a[1], b[0], b[1]) for a, b in segments], owners, [p for p, _ in labels]
    )
    dropped = {drop["segment"] for drop in drops}
    kept = [((x1, y1), (x2, y2)) for x1, y1, x2, y2 in runs if (x1, y1, x2, y2) not in dropped]
    part = reference_partition(pins, labels, kept)
    net_of = label_folded_nets(part)
    on_net: dict[Point, set[tuple[str, str]]] = {}
    alone: list[tuple[str, str]] = []
    for coord, at in part.pin_owners.items():
        touching = at
        if coord not in part.label_texts and part.members[part.root(coord)] == {coord}:
            last = {ref: (ref, name) for ref, name in at}
            touching = list(last.values())
            alone += [pin for pin in at if last[pin[0]] != pin]
        on_net.setdefault(net_of(coord), set()).update(touching)
    names: dict[Point, set[str]] = {}
    for coord, texts in part.label_texts.items():
        names.setdefault(net_of(coord), set()).update(name_key(text) for text in texts)
    nets = {(frozenset(group), frozenset(names.get(net, ()))) for net, group in on_net.items()}
    return frozenset(nets | {(frozenset({pin}), frozenset()) for pin in alone})


class TestTheSignatureIsTheReferenceCircuit:
    @pytest.mark.parametrize("path", SUITE_SHEETS, ids=suite_name)
    def test_on_every_sheet_in_the_suite(self, path: Path) -> None:
        inputs = sheet_inputs(path)
        assert signature(*inputs) == reference_signature(*inputs)

    @pytest.mark.parametrize("seed", range(200))
    def test_on_generated_sheets(self, seed: int) -> None:
        inputs = generated_sheet(seed)
        assert signature(*inputs) == reference_signature(*inputs)


#: R1's two pins, at either end of a wire straight between them, or (with
#: ``STACKED``) both on one point.
TIE_A = PinAt((0, 0), 1, 0, "R1", "1")
TIE_B = PinAt((0, 96), 1, 1, "R1", "2")
STACKED = PinAt((0, 0), 1, 1, "R1", "2")


class TestNetsOverTerminals:
    def test_a_wire_straight_between_two_pins_of_one_part_joins_nothing(self) -> None:
        nets = nets_by_ltspice([TIE_A, TIE_B], [], [((0, 0), (0, 96))])
        assert nets.of_pin(TIE_A) != nets.of_pin(TIE_B)
        assert [run.ends for run in nets.dropped] == [(TIE_A, TIE_B)]

    def test_a_point_on_a_dropped_run_is_on_the_nets_of_both_its_pins(self) -> None:
        nets = nets_by_ltspice([TIE_A, TIE_B], [], [((0, 0), (0, 48)), ((0, 48), (0, 96))])
        assert set(nets.nets_at((0, 48))) == {nets.of_pin(TIE_A), nets.of_pin(TIE_B)}

    def test_a_label_on_the_run_keeps_it(self) -> None:
        nets = nets_by_ltspice([TIE_A, TIE_B], [LabelAt((0, 48), 0, "m")], [((0, 0), (0, 96))])
        assert nets.of_pin(TIE_A) == nets.of_pin(TIE_B)
        assert nets.dropped == ()

    def test_stacked_pins_of_one_part_alone_on_a_point_are_apart(self) -> None:
        other = PinAt((0, 0), 2, 0, "R2", "1")
        nets = nets_by_ltspice([TIE_A, STACKED, other], [], [])
        assert nets.of_pin(STACKED) == nets.of_pin(other) != nets.of_pin(TIE_A)
        assert len(nets.nets_at((0, 0))) == 2

    def test_two_parts_that_share_a_reference_stay_two_parts(self) -> None:
        one, two = PinAt((0, 0), 1, 0, "R1", "1"), PinAt((0, 96), 2, 0, "R1", "1")
        nets = nets_by_ltspice([one, two], [], [((0, 0), (0, 96))])
        assert nets.of_pin(one) == nets.of_pin(two)
        assert nets.dropped == ()

    def test_a_wire_end_merged_into_a_run_is_on_its_net(self) -> None:
        """An in-line waypoint is merged away, and still on its wire's net."""
        label = LabelAt((0, 0), 0, "a")
        nets = nets_by_ltspice([], [label], [((0, 0), (0, 48)), ((0, 48), (0, 96))])
        assert nets.nets_at((0, 48)) == (nets.of_label(label),)

    def test_a_wire_of_no_length_on_a_crossing_joins_nothing(self) -> None:
        across, down = LabelAt((0, 48), 0, "a"), LabelAt((48, 0), 1, "b")
        wires = [((0, 48), (96, 48)), ((48, 0), (48, 96)), ((48, 48), (48, 48))]
        nets = nets_by_ltspice([], [across, down], wires)
        assert nets.of_label(across) != nets.of_label(down)
