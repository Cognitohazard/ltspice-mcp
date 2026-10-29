"""The registry of sheet rules, and each rule on views made for the purpose.

What both tools say of real sheets is held to a record in
``test_sheet_findings_snapshot.py``. This holds the registry to what the tools
publish, and each rule to what it means.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.sheet_findings import (
    RULES,
    Finding,
    Part,
    SheetView,
    findings,
    floating_pins,
)
from ltspice_mcp.tools._base import VALIDATION_WARNING_KINDS
from ltspice_mcp.tools.verify import CHECK_ORDER
from tests import _ltspice_recorded as rec

_RECORD = Path(__file__).parent / "fixtures" / "sheet_findings.json"


def only(view: SheetView, rule: str) -> list[Finding]:
    return findings(view, [rule])


class TestTheRegistry:
    def test_every_rule_either_tool_reports_is_in_it(self) -> None:
        """But for the byte order mark ``verify_circuit`` reports: that is read
        off the file's bytes, which a view of what is drawn does not hold."""
        record = json.loads(_RECORD.read_text(encoding="utf-8"))
        from_an_edit: set[str] = set()
        from_a_check: set[str] = set()
        for entry in record.values():
            from_an_edit |= {warning["kind"] for warning in entry["edit"].get("warnings", [])}
            from_a_check |= {f["rule_id"] for f in entry["verify"] + entry["dropped_wire"]}
        assert from_an_edit <= set(RULES)
        assert from_a_check - set(RULES) == {"byte_order_mark"}

    def test_an_edit_publishes_every_rule_in_the_order_they_are_listed(self) -> None:
        """What changes the circuit or leaves it undone comes first, how the
        sheet reads after."""
        assert (
            VALIDATION_WARNING_KINDS
            == tuple(RULES)
            == (
                "unresolved_symbol",
                "shared_reference",
                "dropped_wire",
                "duplicate_parameter",
                "subcircuit_value_slot",
                "net_with_two_names",
                "joined_at_crossing",
                "unjoined_crossing",
                "diagonal_wire",
                "floating_pin",
                "dangling_wire_end",
                "dangling_label",
                "duplicate_wire",
                "symbol_overlap",
                "wire_through_symbol",
                "attribute_text_overlap",
                "label_over_component",
                "text_in_symbol_body",
                "stacked_directive",
                "label_island",
            )
        )
        families = [rule.family for rule in RULES.values()]
        assert families.index("drawing") > max(
            position for position, family in enumerate(families) if family != "drawing"
        )

    def test_every_rule_is_reported_by_a_check_the_checker_has(self) -> None:
        by_check: dict[str, set[str]] = {}
        for rule in RULES.values():
            by_check.setdefault(rule.check, set()).add(rule.rule_id)
        assert set(by_check) <= set(CHECK_ORDER)
        assert by_check == {
            "symbols": {"unresolved_symbol"},
            "export": {"dropped_wire", "net_with_two_names"},
            "layout": {
                "joined_at_crossing",
                "unjoined_crossing",
                "diagonal_wire",
                "floating_pin",
                "dangling_wire_end",
                "dangling_label",
                "duplicate_wire",
                "symbol_overlap",
                "wire_through_symbol",
                "attribute_text_overlap",
            },
            "quality": {
                "shared_reference",
                "duplicate_parameter",
                "subcircuit_value_slot",
                "label_over_component",
                "text_in_symbol_body",
                "stacked_directive",
                "label_island",
            },
        }

    def test_a_finding_on_a_whole_sheet_is_called_what_it_always_was(self) -> None:
        called = {rule.rule_id: rule.severity for rule in RULES.values()}
        assert called.pop("dropped_wire") == "warning"
        assert called.pop("subcircuit_value_slot") == "warning"
        assert called.pop("unresolved_symbol") == "error"
        assert called.pop("shared_reference") == "error"
        assert set(called.values()) == {"observation"}

    def test_an_electrical_rule_rests_on_a_recording_and_no_other_rule_claims_one(self) -> None:
        for rule in RULES.values():
            if rule.family == "electrical":
                assert rule.provenance in rec.CASES.behaviours, rule.rule_id
                assert rec.cases_of(rule.provenance), rule.rule_id
            else:
                assert rule.provenance == "definition", rule.rule_id

    def test_each_rule_says_what_it_is_in_a_sentence(self) -> None:
        for rule in RULES.values():
            assert rule.summary.endswith(".") and len(rule.summary) > 20, rule.rule_id

    def test_findings_come_grouped_by_rule_in_that_order(self) -> None:
        view = SheetView(
            parts=(
                Part("R1", box=BBox(0, 0, 32, 96), pins=(("A", 16, 0),)),
                Part("U1", symbol="opamp", at=(200, 0), missing=True),
            ),
            wires=((300, 0, 364, 0),),
            labels=((16, 48, "inside"),),
            texts=((400, 0, ".op"), (400, 0, ".tran 1")),
        )
        order = list(RULES)
        found = [one.rule for one in findings(view)]
        assert found == sorted(found, key=order.index)
        assert set(found) == {
            "unresolved_symbol",
            "floating_pin",
            "dangling_wire_end",
            "dangling_label",
            "label_over_component",
            "stacked_directive",
        }

    def test_only_the_rules_asked_for_are_run(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("A", 0, 0),)),), wires=((64, 0, 128, 0),))
        assert {one.rule for one in findings(view)} == {"floating_pin", "dangling_wire_end"}
        assert {one.rule for one in findings(view, ["floating_pin"])} == {"floating_pin"}
        assert findings(view, []) == []


class TestAFinding:
    def test_its_sentence_stands_alone_and_says_no_more_than_the_rest_of_it(self) -> None:
        """An edit's reply shows the sentence and nothing else, and two findings
        are the same finding when their rule, parts, points and facts are."""
        view = SheetView(
            parts=(
                Part("R1", box=BBox(0, 0, 32, 96), pins=(("A", 16, 0),), texts=((16, 200, "R1"),)),
                Part("R2", box=BBox(16, 48, 48, 144), texts=((8, 8, "1k"),)),
            ),
            wires=((-16, 16, 64, 16), (400, 0, 464, 0), (464, 0, 400, 0)),
            labels=((500, 500, "loose"), (600, 0, "vdd"), (700, 0, "vdd")),
            texts=((300, 300, ".op"), (300, 300, ".tran 1")),
        )
        found = findings(view)
        assert len({one.rule for one in found}) >= 8
        for one in found:
            for ref in one.refs:
                assert ref in one.detail, one
            for fact in one.facts.values():
                assert str(fact) in one.detail, one
            x, y = one.points[0]
            assert f"({x},{y})" in one.detail or one.rule == "label_island", one
        assert len({one.identity for one in found}) == len(found)

    def test_a_different_fact_is_a_different_finding(self) -> None:
        one = Finding("floating_pin", "s", refs=("R1",), points=((0, 0),), facts={"pin": "A"})
        other = Finding("floating_pin", "s", refs=("R1",), points=((0, 0),), facts={"pin": "B"})
        again = Finding("floating_pin", "t", refs=("R1",), points=((0, 0),), facts={"pin": "A"})
        assert one.identity != other.identity
        assert one.identity == again.identity


def stacked(ref: str, x: int, y: int) -> Part:
    """A part with both of its pins on one point."""
    return Part(ref, pins=(("1", x, y), ("2", x, y)))


class TestAFloatingPin:
    """What LTspice was recorded doing."""

    def floating(self, view: SheetView) -> list[str]:
        return [f"{part.ref}.{name}" for part, (name, _x, _y) in floating_pins(view)]

    def test_a_pin_on_nothing(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("1", 0, 0), ("2", 0, 96))),))
        assert self.floating(view) == ["R1.1", "R1.2"]

    def test_a_label_or_a_wire_or_another_parts_pin_connects_it(self) -> None:
        view = SheetView(
            parts=(
                Part("R1", pins=(("1", 0, 0), ("2", 0, 96), ("3", 0, 192))),
                Part("R2", pins=(("1", 0, 192),)),
            ),
            wires=((-32, 96, 32, 96),),
            labels=((0, 0, "a"),),
        )
        assert self.floating(view) == []

    def test_a_point_on_a_diagonal_wire_is_on_it(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("1", 32, 32),)),), wires=((0, 0, 64, 64),))
        assert self.floating(view) == []

    def test_a_wire_of_no_length_connects_nothing(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("1", 0, 0),)),), wires=((0, 0, 0, 0),))
        assert self.floating(view) == ["R1.1"]

    def test_a_parts_own_pins_on_one_point_do_not_connect_one_another(self) -> None:
        assert self.floating(SheetView(parts=(stacked("R1", 0, 0),))) == ["R1.1", "R1.2"]

    def test_only_its_last_pin_there_touches_another_parts(self) -> None:
        view = SheetView(parts=(stacked("R1", 0, 0), Part("R2", pins=(("1", 0, 0),))))
        assert self.floating(view) == ["R1.1"]

    def test_a_wire_or_a_label_on_the_point_connects_them_all(self) -> None:
        wired = SheetView(parts=(stacked("R1", 0, 0),), wires=((0, 0, 64, 0),))
        labelled = SheetView(parts=(stacked("R1", 0, 0),), labels=((0, 0, "f"),))
        assert self.floating(wired) == [] and self.floating(labelled) == []

    def test_the_finding_names_the_part_the_pin_and_the_place(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("A", 16, 0), ("", 16, 96))),))
        assert only(view, "floating_pin") == [
            Finding(
                "floating_pin",
                "Floating pin: R1.A at (16,0)",
                refs=("R1",),
                points=((16, 0),),
                facts={"pin": "A"},
            ),
            Finding(
                "floating_pin",
                "Floating pin: R1 at (16,96)",
                refs=("R1",),
                points=((16, 96),),
                facts={"pin": ""},
            ),
        ]


def part(ref: str, box: tuple[int, int, int, int], *pins: tuple[str, int, int]) -> Part:
    return Part(ref, box=BBox(*box), pins=pins)


class TestWhatIsLeftUndone:
    def test_a_wire_end_on_nothing_is_one_place_however_many_end_there(self) -> None:
        view = SheetView(wires=((0, 0, 64, 0), (64, 0, 64, 64)), labels=((0, 0, "a"),))
        (found,) = only(view, "dangling_wire_end")
        assert (found.points, found.detail) == (
            ((64, 64),),
            "Wire end at (64,64) meets no pin, net label, or other wire",
        )

    def test_a_label_on_nothing(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96), ("A", 16, 0)),),
            wires=((16, 0, 16, -32),),
            labels=((200, 200, "loose"), (16, -16, "on_the_wire"), (16, 0, "on_the_pin")),
        )
        (found,) = only(view, "dangling_label")
        assert (found.points, found.facts) == (((200, 200),), {"label": "loose"})

    def test_a_wire_drawn_twice_in_either_direction(self) -> None:
        view = SheetView(wires=((0, 0, 64, 0), (64, 0, 0, 0), (0, 16, 0, 16)))
        (found,) = only(view, "duplicate_wire")
        assert (found.points, found.facts) == (((0, 0), (64, 0)), {"count": 2})

    def test_a_part_whose_symbol_was_not_found(self) -> None:
        view = SheetView(parts=(Part("U1", symbol="opamp", at=(96, 64), missing=True),))
        assert findings(view) == [
            Finding(
                "unresolved_symbol",
                "Symbol 'opamp' of U1 was not found: the part has no pins and no "
                "extent here, so nothing about them is checked",
                refs=("U1",),
                points=((96, 64),),
                facts={"symbol": "opamp"},
            )
        ]

    def test_nothing_is_said_of_the_extent_of_a_part_that_was_not_found(self) -> None:
        """It is drawn as a placeholder, and a placeholder's box is not the part's."""
        placeholder = BBox(0, 0, 64, 48)
        lost = Part(
            "U1", symbol="opamp", at=(0, 0), box=placeholder, body=placeholder, missing=True
        )
        view = SheetView(
            parts=(lost, part("R1", (16, 16, 48, 112))),
            wires=((-16, 24, 80, 24),),
            labels=((8, 8, "inside"), (-16, 24, "a"), (80, 24, "b")),
            texts=((32, 40, ".op"),),
        )
        found = findings(view)
        assert [one.refs for one in found if "U1" in one.refs] == [("U1",)]
        assert {one.rule for one in found} == {
            "unresolved_symbol",
            # The wire's two labels, a and b: one net of two names, U1 not in it.
            "net_with_two_names",
            "dangling_label",
            "wire_through_symbol",
            "text_in_symbol_body",
        }
        assert {one.refs for one in found if one.rule != "unresolved_symbol"} <= {(), ("R1",)}


class TestWhatLtspiceLeavesOut:
    def test_a_wire_straight_between_two_pins_of_one_part(self) -> None:
        view = SheetView(
            parts=(Part("R1", pins=(("A", 0, 0), ("B", 0, 96))),), wires=((0, 0, 0, 96),)
        )
        (found,) = only(view, "dropped_wire")
        assert (found.refs, found.points, found.detail) == (
            ("R1",),
            ((0, 0), (0, 96)),
            "Wire (0,0)->(0,96) joins two pins of the same instance R1 and is not exported",
        )

    @pytest.mark.parametrize("reference", ["", "R1"])
    def test_two_parts_of_one_reference_are_still_two_parts(self, reference: str) -> None:
        """A wire between a pin of each is a connection, whatever they are called."""
        view = SheetView(
            parts=(
                Part(reference, symbol="res", pins=(("A", 0, 0),)),
                Part(reference, symbol="res", pins=(("A", 0, 96),)),
            ),
            wires=((0, 0, 0, 96),),
        )
        assert only(view, "dropped_wire") == []


# Two wires through each other at (32,32), neither ending there.
_CROSS = ((0, 32, 64, 32), (32, 0, 32, 64))


class TestWhatJoinsAndWhatDoesNot:
    """What a finished sheet joins, judged once on the whole sheet, so a batch
    that passes through a state on its way says nothing of that state."""

    def test_a_net_with_labels_of_two_names(self) -> None:
        view = SheetView(wires=((0, 0, 64, 0),), labels=((0, 0, "a"), (64, 0, "b")))
        (found,) = only(view, "net_with_two_names")
        assert (found.points, found.facts) == (((0, 0), (64, 0)), {"names": ("a", "b")})

    def test_labels_of_one_name_are_one_node_with_whatever_else_they_carry(self) -> None:
        """``a`` on a wire of its own and ``a`` beside ``b`` are one node, with two names."""
        view = SheetView(
            wires=((0, 0, 64, 0), (0, 96, 64, 96)),
            labels=((0, 0, "a"), (0, 96, "a"), (64, 96, "b")),
        )
        (found,) = only(view, "net_with_two_names")
        assert found.points == ((0, 0), (0, 96), (64, 96))

    def test_a_name_in_two_cases_is_spelled_as_label_island_spells_it(self) -> None:
        """The label lowest on the sheet, and of those the one furthest right."""
        view = SheetView(
            wires=((0, 0, 64, 0),), labels=((0, 0, "out"), (64, 0, "OUT"), (32, 0, "b"))
        )
        (found,) = only(view, "net_with_two_names")
        assert found.facts == {"names": ("OUT", "b")}

    @pytest.mark.parametrize(
        "names", [("out", "OUT"), ("a", "0"), ("a", "GND")], ids=["case", "ground", "gnd"]
    )
    def test_names_differing_in_case_or_a_ground_are_no_second_name(
        self, names: tuple[str, str]
    ) -> None:
        view = SheetView(wires=((0, 0, 64, 0),), labels=((0, 0, names[0]), (64, 0, names[1])))
        assert only(view, "net_with_two_names") == []

    def test_a_label_on_a_crossing_joins_the_wires(self) -> None:
        view = SheetView(wires=_CROSS, labels=((32, 32, "x"),))
        (found,) = only(view, "joined_at_crossing")
        assert (found.points, found.facts) == (((32, 32),), {"wires": 2, "nets": 2})
        assert only(view, "unjoined_crossing") == []

    def test_a_pin_on_a_crossing_names_its_part(self) -> None:
        view = SheetView(parts=(Part("R1", pins=(("A", 32, 32), ("B", 32, 128))),), wires=_CROSS)
        (found,) = only(view, "joined_at_crossing")
        assert found.refs == ("R1",)

    def test_wires_already_joined_elsewhere_are_not_joined_by_the_crossing(self) -> None:
        view = SheetView(
            wires=(*_CROSS, (64, 32, 64, 64), (32, 64, 64, 64)), labels=((32, 32, "x"),)
        )
        assert only(view, "joined_at_crossing") == []

    @pytest.mark.parametrize(
        "names", [("a", "A"), ("0", "GND")], ids=["one-name-any-case", "ground"]
    )
    def test_wires_already_one_node_by_name_are_not_joined_by_the_crossing(
        self, names: tuple[str, str]
    ) -> None:
        """Each wire carries a label of one name at its far end, so the two are
        one node before anything sits on the crossing."""
        view = SheetView(
            parts=(Part("R1", pins=(("A", 32, 32), ("B", 32, 128))),),
            wires=_CROSS,
            labels=((64, 32, names[0]), (32, 64, names[1])),
        )
        assert only(view, "joined_at_crossing") == []

    def test_a_crossing_with_nothing_on_it_is_left_apart(self) -> None:
        (found,) = only(SheetView(wires=_CROSS), "unjoined_crossing")
        assert found.points == ((32, 32),)

    def test_a_wire_ending_on_another_is_a_junction_not_a_crossing(self) -> None:
        view = SheetView(wires=((0, 32, 64, 32), (32, 0, 32, 32)))
        assert only(view, "unjoined_crossing") == []
        assert only(view, "joined_at_crossing") == []

    def test_a_diagonal_wire(self) -> None:
        view = SheetView(wires=((0, 0, 32, 32), (0, 64, 64, 64)))
        (found,) = only(view, "diagonal_wire")
        assert found.points == ((0, 0), (32, 32))


class TestAParameterGivenTwice:
    """LTspice writes Value, Value2, SpiceLine and SpiceLine2 onto one line
    (recorded: export-attributes), so a key in two of them is in it twice."""

    def test_across_two_attributes_in_any_case(self) -> None:
        view = SheetView(
            parts=(
                Part(
                    "A1", at=(0, 0), parameters=(("Value", "Vt=2 Vh=0.5"), ("SpiceLine", "vt=3"))
                ),
            )
        )
        (found,) = only(view, "duplicate_parameter")
        assert (found.refs, found.points) == (("A1",), ((0, 0),))
        assert found.facts == {"parameter": "vt", "copies": ("Vt=2 in Value", "vt=3 in SpiceLine")}

    def test_once_each_is_nothing(self) -> None:
        view = SheetView(
            parts=(Part("M1", parameters=(("Value", "NMOS"), ("Value2", "l=1u w=1u"))),)
        )
        assert only(view, "duplicate_parameter") == []


#: A part that draws a frame one unit outside its two columns of pins, as many
#: of LTspice's own symbols of an integrated circuit do, with one pin deep inside.
FRAMED = Part(
    "U1",
    box=BBox(-1, 0, 161, 96),
    body=BBox(-1, 0, 161, 96),
    pins=(("IN", 0, 48), ("OUT", 160, 48), ("REF", 112, 64)),
)

#: An amplifier drawn as a triangle pointing right, as LTspice's opamp is: its
#: body box has empty corners above and below the slopes.
TRIANGLE = Part(
    "U2",
    box=BBox(-32, -64, 96, 64),
    body=BBox(-32, -64, 96, 64),
    pins=(("In-", -32, -32), ("In+", -32, 32), ("OUT", 96, 0)),
    strokes=(((-32, -64), (-32, 64)), ((-32, 64), (96, 0)), ((96, 0), (-32, -64))),
)


class TestHowTheSheetReads:
    def test_boxes_that_share_an_area_and_boxes_that_only_touch(self) -> None:
        view = SheetView(
            parts=(
                part("R1", (0, 0, 32, 96)),
                part("R2", (16, 48, 48, 144)),
                part("R3", (32, 0, 64, 48)),
            )
        )
        (found,) = only(view, "symbol_overlap")
        assert (found.refs, found.points, found.detail) == (
            ("R1", "R2"),
            ((16, 48), (32, 96)),
            "R1 and R2: bounding boxes share a 16x48 region at (16,48)",
        )

    def test_two_parts_overlap_by_their_boxes_with_pins(self) -> None:
        # R1 draws 32 wide and has a pin 32 further out; R2 sits in that reach.
        r1 = Part("R1", box=BBox(0, 0, 64, 96), body=BBox(0, 0, 32, 96))
        r2 = Part("R2", box=BBox(40, 0, 56, 96), body=BBox(40, 0, 56, 96))
        (found,) = only(SheetView(parts=(r1, r2)), "symbol_overlap")
        assert found.refs == ("R1", "R2")

    def test_a_wire_through_a_box_and_one_along_its_edge(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96)),),
            wires=((-16, 48, 48, 48), (0, 0, 0, 96)),
        )
        (found,) = only(view, "wire_through_symbol")
        assert (found.refs, found.points, found.detail) == (
            ("R1",),
            ((-16, 48), (48, 48)),
            "Wire (-16,48)->(48,48) passes through R1's body",
        )

    def test_a_wire_is_through_a_part_by_what_the_part_draws(self) -> None:
        # A wire through the reach between a part's body and its far pin
        # crosses nothing drawn.
        r1 = Part("R1", box=BBox(0, 0, 64, 96), body=BBox(0, 0, 32, 96))
        view = SheetView(parts=(r1,), wires=((48, -16, 48, 112), (16, -16, 16, 112)))
        (found,) = only(view, "wire_through_symbol")
        assert found.points == ((16, -16), (16, 112))

    def test_a_wire_through_an_empty_corner_of_a_body_is_not_through_the_part(self) -> None:
        """Beside an amplifier's slope is inside its box, nowhere near a line
        it draws; across the triangle is through it."""
        view = SheetView(parts=(TRIANGLE,), wires=((48, -56, 128, -56), (16, -96, 16, 96)))
        (found,) = only(view, "wire_through_symbol")
        assert found.points == ((16, -96), (16, 96))

    def test_a_wire_through_a_rectangle_a_part_draws_is_through_it(self) -> None:
        plate = Part("C1", body=BBox(0, 0, 64, 96), areas=(BBox(0, 40, 64, 56),))
        view = SheetView(parts=(plate,), wires=((32, -16, 32, 112), (48, 8, 80, 8)))
        (found,) = only(view, "wire_through_symbol")
        assert found.points == ((32, -16), (32, 112))

    def test_a_view_that_knows_no_body_uses_the_box(self) -> None:
        assert Part("R1", box=BBox(0, 0, 64, 96)).drawn == BBox(0, 0, 64, 96)

    def test_a_wire_out_from_a_pin_inside_the_body_is_not_through_the_part(self) -> None:
        """Out by the nearer edge on the wire's own line, wherever the pin is:
        just inside a frame, or deep in a box larger than what it encloses."""
        view = SheetView(
            parts=(FRAMED,),
            wires=((-64, 48, 0, 48), (160, 48, 224, 48), (112, 64, 112, 128), (112, 64, 200, 64)),
        )
        assert only(view, "wire_through_symbol") == []

    def test_a_wire_from_such_a_pin_back_across_the_body_is_through_it(self) -> None:
        view = SheetView(parts=(FRAMED,), wires=((0, 48, 224, 48), (112, 64, 112, -32)))
        assert [found.points for found in only(view, "wire_through_symbol")] == [
            ((0, 48), (224, 48)),
            ((112, 64), (112, -32)),
        ]

    def test_a_wire_from_such_a_pin_that_ends_inside_the_body_is_through_it(self) -> None:
        """It has not left the body, so it is no way out of it. Between two of
        the part's pins, and to a corner the next wire leaves from."""
        view = SheetView(parts=(FRAMED,), wires=((0, 48, 160, 48), (112, 64, 144, 64)))
        assert len(only(view, "wire_through_symbol")) == 2

    def test_a_wire_over_such_a_pin_and_on_is_through_it(self) -> None:
        view = SheetView(parts=(FRAMED,), wires=((-64, 48, 64, 48),))
        assert len(only(view, "wire_through_symbol")) == 1

    def test_a_diagonal_wire_from_such_a_pin_is_through_it(self) -> None:
        view = SheetView(parts=(FRAMED,), wires=((0, 48, -64, 16),))
        assert len(only(view, "wire_through_symbol")) == 1

    def test_only_the_parts_own_pin_makes_a_wire_its_way_in(self) -> None:
        """Another part's pin at that point is a wire into this one's body."""
        other = Part("R1", pins=(("A", 0, 48),))
        frame = Part("U1", box=FRAMED.box, body=FRAMED.body)
        view = SheetView(parts=(frame, other), wires=((-64, 48, 0, 48),))
        assert len(only(view, "wire_through_symbol")) == 1

    def test_a_label_inside_a_box_and_on_no_pin(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96), ("A", 16, 0)),),
            labels=((16, 48, "inside"), (16, 0, "on_the_pin"), (200, 200, "elsewhere")),
        )
        (found,) = only(view, "label_over_component")
        assert (found.refs, found.points, found.facts) == (
            ("R1",),
            ((16, 48),),
            {"label": "inside"},
        )

    def test_text_inside_another_parts_box_but_not_its_own(self) -> None:
        own = Part("R1", box=BBox(0, 0, 32, 96), texts=((16, 16, "R1"),))
        other = Part("R2", box=BBox(100, 0, 132, 96), texts=((16, 48, "1k"),))
        view = SheetView(parts=(own, other), texts=((110, 48, ".op"),))
        found = only(view, "text_in_symbol_body")
        assert [(one.refs, one.facts, one.detail) for one in found] == [
            (("R1",), {"text": "1k"}, "Text '1k' at (16,48) is anchored inside R1's body"),
            (("R2",), {"text": ".op"}, "Text '.op' at (110,48) is anchored inside R2's body"),
        ]

    def test_text_in_an_empty_corner_of_a_body_is_not_in_the_part(self) -> None:
        """Beside an amplifier's slope is inside its box but outside what it
        draws; inside the triangle, or on one of its lines, is in it."""
        view = SheetView(
            parts=(TRIANGLE,),
            texts=((64, -48, "corner"), (0, 0, "inside"), (32, -32, "on the slope")),
        )
        assert [one.facts["text"] for one in only(view, "text_in_symbol_body")] == [
            "inside",
            "on the slope",
        ]

    @pytest.mark.parametrize(
        ("element", "spice_model", "value", "said"),
        [
            ("X", "", "Level.2", True),
            ("X", "UniversalOpamp2", "level.1", True),
            ("X", "UniversalOpamp2", "10k", True),
            ("R", "", "Level.2", False),
            ("R", "", "10k", False),
            ("X", "", "LT1013", False),
            ("Q", "mdl", "2", False),
        ],
        ids=[
            "level-on-subcircuit",
            "level-with-spicemodel",
            "any-value-with-spicemodel",
            "level-on-resistor",
            "resistor-value",
            "subcircuit-named-in-value",
            "transistor-area-factor",
        ],
    )
    def test_a_value_slot_a_subcircuit_call_cannot_hold(
        self, element: str, spice_model: str, value: str, said: bool
    ) -> None:
        """LTspice writes a subcircuit call's Value last, where the subcircuit
        is named: an opamp's complexity label there, or any Value after a
        SpiceModel, names the wrong one. A library part naming its subcircuit
        in Value is the ordinary case, and a transistor's Value after its
        SpiceModel is its area factor."""
        part = Part("U1", element=element, spice_model=spice_model, parameters=(("Value", value),))
        found = only(SheetView(parts=(part,)), "subcircuit_value_slot")
        assert [one.facts for one in found] == ([{"value": value}] if said else [])

    def test_an_empty_value_slot_is_not_said(self) -> None:
        part = Part("U1", element="X", spice_model="UniversalOpamp2")
        assert only(SheetView(parts=(part,)), "subcircuit_value_slot") == []

    def test_attribute_text_across_a_wire_names_the_wire(self) -> None:
        value = Part("R1", text_extents=((0, 0, "10k", BBox(0, -8, 60, 14)),))
        view = SheetView(parts=(value,), wires=((30, -40, 30, 40), (100, -40, 100, 40)))
        (found,) = only(view, "attribute_text_overlap")
        assert (found.refs, found.points, found.facts) == (
            ("R1",),
            ((0, 0), (30, -40), (30, 40)),
            {"text": "10k", "across": "wire"},
        )

    def test_attribute_text_across_what_another_part_draws(self) -> None:
        """A line the part draws, or a rectangle or circle; not an empty corner
        of its body, such as the space beside an amplifier's triangle."""
        text = Part("R1", text_extents=((0, 0, "10k", BBox(0, -8, 60, 14)),))
        triangle = Part(
            "U1",
            box=BBox(-40, -60, 100, 60),
            body=BBox(-40, -60, 100, 60),
            strokes=(((-40, -60), (100, -60)), ((100, -60), (100, 60))),
        )
        assert only(SheetView(parts=(text, triangle)), "attribute_text_overlap") == []
        lead = Part("C1", body=BBox(20, -40, 40, 40), strokes=(((30, -40), (30, 40)),))
        plate = Part("C2", body=BBox(20, -40, 40, 40), areas=(BBox(20, -4, 40, 4),))
        for other in (lead, plate):
            (found,) = only(SheetView(parts=(text, other)), "attribute_text_overlap")
            assert (found.refs, found.facts) == (
                ("R1", other.ref),
                {"text": "10k", "across": "body"},
            )
        # A part with nothing but its body known reaches all of it.
        boxed = Part("X1", body=BBox(20, -40, 40, 40))
        assert len(only(SheetView(parts=(text, boxed)), "attribute_text_overlap")) == 1
        # Its own drawing is the part's to draw text over.
        own = Part(
            "R1",
            text_extents=((0, 0, "10k", BBox(0, -8, 60, 14)),),
            body=BBox(20, -40, 40, 40),
            strokes=(((30, -40), (30, 40)),),
        )
        assert only(SheetView(parts=(own,)), "attribute_text_overlap") == []

    def test_attribute_texts_that_only_graze_do_not_overlap(self) -> None:
        """Extents are estimated from the font, not the string's own glyphs, so
        one line's descender band on the next line's caps is no overlap."""
        stacked = Part(
            "C1",
            text_extents=((0, 0, "C1", BBox(0, -8, 30, 14)), (0, 20, "47n", BBox(0, 12, 40, 34))),
        )
        assert only(SheetView(parts=(stacked,)), "attribute_text_overlap") == []
        printed = Part(
            "C1",
            text_extents=((0, 0, "C1", BBox(0, -8, 30, 14)), (0, 4, "100n", BBox(0, -4, 60, 18))),
        )
        (found,) = only(SheetView(parts=(printed,)), "attribute_text_overlap")
        assert (found.points, found.facts) == (
            ((0, 0), (0, 4)),
            {"text": "C1", "across": "text", "other_text": "100n"},
        )

    def test_attribute_text_across_a_label_or_a_comment(self) -> None:
        value = Part("R1", text_extents=((0, 0, "10k", BBox(0, -8, 60, 14)),))
        view = SheetView(parts=(value,), text_extents=((40, 0, "OUT", BBox(40, -8, 90, 14)),))
        (found,) = only(view, "attribute_text_overlap")
        assert found.detail == "R1's text '10k' at (0,0) is drawn across the text 'OUT' at (40,0)"
        # A view that carries no extents has nothing for the rule to read.
        bare = Part("R1", texts=((0, 0, "10k"),))
        assert (
            only(SheetView(parts=(bare,), wires=((30, -40, 30, 40),)), "attribute_text_overlap")
            == []
        )

    def test_text_at_one_anchor(self) -> None:
        view = SheetView(texts=((16, 16, ".op"), (16, 16, ".tran 1"), (16, 32, "a note")))
        (found,) = only(view, "stacked_directive")
        assert (found.points, found.facts) == (((16, 16),), {"count": 2})

    def test_a_net_joined_by_labels_alone(self) -> None:
        view = SheetView(
            wires=((0, 0, 64, 0),),
            labels=(
                (200, 0, "vdd"),
                (200, 96, "vdd"),
                (0, 0, "out"),
                (300, 0, "out"),
                (400, 0, "0"),
                (400, 96, "0"),
                (500, 0, "alone"),
            ),
        )
        (found,) = only(view, "label_island")
        assert (found.points, found.facts) == (((200, 0), (200, 96)), {"net": "vdd"})

    def test_a_part_is_called_by_its_reference_else_by_its_symbol(self) -> None:
        nameless = Part("", symbol="res", box=BBox(0, 0, 32, 96))
        view = SheetView(parts=(nameless, Part("", box=BBox(16, 16, 48, 48))))
        (found,) = only(view, "symbol_overlap")
        assert found.refs == ("res", "<unnamed>")
