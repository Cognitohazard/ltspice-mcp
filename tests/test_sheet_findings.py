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
                "dropped_wire",
                "floating_pin",
                "dangling_wire_end",
                "dangling_label",
                "duplicate_wire",
                "symbol_overlap",
                "wire_through_symbol",
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
            "export": {"dropped_wire"},
            "layout": {
                "floating_pin",
                "dangling_wire_end",
                "dangling_label",
                "duplicate_wire",
                "symbol_overlap",
                "wire_through_symbol",
            },
            "quality": {
                "label_over_component",
                "text_in_symbol_body",
                "stacked_directive",
                "label_island",
            },
        }

    def test_a_finding_on_a_whole_sheet_is_called_what_it_always_was(self) -> None:
        called = {rule.rule_id: rule.severity for rule in RULES.values()}
        assert called.pop("dropped_wire") == "warning"
        assert called.pop("unresolved_symbol") == "error"
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

    def test_a_view_that_knows_no_body_uses_the_box(self) -> None:
        assert Part("R1", box=BBox(0, 0, 64, 96)).drawn == BBox(0, 0, 64, 96)

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
