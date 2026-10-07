"""The registry of sheet rules, and the checks on views made for the purpose.

What both tools say of real sheets is held to a record in
``test_sheet_findings_snapshot.py``. This holds the registry to what the tools
publish, and each rule to what it means, whichever tool asks.
"""

from __future__ import annotations

import json
from pathlib import Path

from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.sheet_findings import (
    RULES,
    Finding,
    Part,
    SheetView,
    checker_findings,
    dropped_wires,
    editor_findings,
    floating_pins,
    label_islands,
    unresolved_symbols,
)
from ltspice_mcp.tools._base import VALIDATION_WARNING_KINDS
from ltspice_mcp.tools.verify import CHECK_ORDER
from tests import _ltspice_recorded as rec

_RECORD = Path(__file__).parent / "fixtures" / "sheet_findings.json"


def kinds(findings: list[Finding]) -> list[str]:
    return [finding.rule for finding in findings]


class TestTheRegistry:
    def test_every_rule_either_tool_reports_is_in_it(self) -> None:
        record = json.loads(_RECORD.read_text(encoding="utf-8"))
        reported: set[str] = set()
        for entry in record.values():
            reported |= {warning["kind"] for warning in entry["edit"].get("warnings", [])}
            reported |= {f["rule_id"] for f in entry["verify"] + entry["dropped_wire"]}
        assert reported <= set(RULES)

    def test_the_editors_rules_are_the_kinds_its_reply_publishes_in_that_order(self) -> None:
        assert VALIDATION_WARNING_KINDS == (
            "floating_pin",
            "duplicate_wire",
            "dangling_label",
            "label_over_component",
            "stacked_directive",
            "unresolved_symbol",
        )
        assert (
            tuple(rule.rule_id for rule in RULES.values() if rule.editor)
            == VALIDATION_WARNING_KINDS
        )

    def test_a_rule_is_reported_by_a_check_the_checker_has(self) -> None:
        checks = {rule.check for rule in RULES.values() if rule.check is not None}
        assert checks <= set(CHECK_ORDER)
        assert {rule.rule_id for rule in RULES.values() if rule.check == "layout"} == {
            "floating_pin",
            "symbol_overlap",
            "wire_through_symbol",
            "dangling_wire_end",
        }

    def test_every_rule_is_reported_by_one_tool_at_least(self) -> None:
        assert [
            rule.rule_id for rule in RULES.values() if not rule.editor and not rule.check
        ] == []

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


def stacked(ref: str, x: int, y: int) -> Part:
    """A part with both of its pins on one point."""
    return Part(ref, pins=(("1", x, y), ("2", x, y)))


class TestAFloatingPinIsOneRule:
    """What LTspice was recorded doing, the same whichever tool asks."""

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

    def test_both_tools_report_the_same_pins(self) -> None:
        view = SheetView(
            parts=(
                Part("R1", box=BBox(0, 0, 0, 0), pins=(("1", 0, 0), ("2", 0, 0))),
                Part("R2", box=BBox(0, 0, 0, 0), pins=(("1", 0, 0),)),
                Part("R3", box=BBox(96, 0, 96, 0), pins=(("1", 96, 0),)),
            ),
        )
        from_the_editor = [f.points for f in editor_findings(view) if f.rule == "floating_pin"]
        from_the_checker = [f.points for f in checker_findings(view) if f.rule == "floating_pin"]
        assert from_the_editor == from_the_checker == [((0, 0),), ((96, 0),)]


class TestAPartsBox:
    def test_two_parts_overlap_by_their_boxes_with_pins(self) -> None:
        # R1 draws 32 wide and has a pin 32 further out; R2 sits in that reach.
        r1 = Part("R1", box=BBox(0, 0, 64, 96), body=BBox(0, 0, 32, 96))
        r2 = Part("R2", box=BBox(40, 0, 56, 96), body=BBox(40, 0, 56, 96))
        (found,) = checker_findings(SheetView(parts=(r1, r2)))
        assert (found.rule, found.refs) == ("symbol_overlap", ("R1", "R2"))

    def test_a_wire_is_through_a_part_by_what_the_part_draws(self) -> None:
        # The same part; a wire through the reach between its body and its far
        # pin crosses nothing drawn.
        r1 = Part("R1", box=BBox(0, 0, 64, 96), body=BBox(0, 0, 32, 96))
        view = SheetView(
            parts=(r1,),
            wires=((48, -16, 48, 112), (16, -16, 16, 112)),
            labels=((48, -16, "a"), (48, 112, "b"), (16, -16, "c"), (16, 112, "d")),
        )
        (found,) = checker_findings(view)
        assert (found.rule, found.points) == ("wire_through_symbol", ((16, -16), (16, 112)))

    def test_a_view_that_knows_no_body_uses_the_box(self) -> None:
        assert Part("R1", box=BBox(0, 0, 64, 96)).drawn == BBox(0, 0, 64, 96)


def part(ref: str, box: tuple[int, int, int, int], *pins: tuple[str, int, int]) -> Part:
    return Part(ref, box=BBox(*box), pins=pins)


class TestTheEditorsReading:
    def test_a_pin_on_nothing(self) -> None:
        view = SheetView(parts=(part("R1", (0, 0, 32, 96), ("A", 16, 0), ("B", 16, 96)),))
        found = editor_findings(view)
        assert kinds(found) == ["floating_pin", "floating_pin"]
        assert found[0] == Finding(
            "floating_pin",
            "Floating pin: R1.A at (16,0)",
            refs=("R1",),
            points=((16, 0),),
            facts={"pin": "A"},
        )

    def test_a_pin_on_a_wires_interior_or_under_a_label_is_not_floating(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96), ("A", 16, 0), ("B", 16, 96)),),
            wires=((0, 0, 64, 0),),
            labels=((16, 96, "0"),),
        )
        assert editor_findings(view) == []

    def test_a_wire_drawn_twice_in_either_direction(self) -> None:
        view = SheetView(wires=((0, 0, 64, 0), (64, 0, 0, 0), (0, 16, 0, 16)))
        (found,) = editor_findings(view)
        assert (found.rule, found.points, found.facts) == (
            "duplicate_wire",
            ((0, 0), (64, 0)),
            {"count": 2},
        )

    def test_a_label_on_nothing_and_a_label_inside_a_box(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96), ("A", 16, 0)),),
            wires=((16, 0, 16, -32),),
            labels=((200, 200, "loose"), (16, 48, "inside"), (16, 0, "on_the_pin")),
        )
        found = editor_findings(view)
        assert [(f.rule, f.facts["label"]) for f in found] == [
            ("dangling_label", "loose"),
            ("dangling_label", "inside"),
            ("label_over_component", "inside"),
        ]

    def test_text_at_one_anchor(self) -> None:
        view = SheetView(texts=((16, 16, ".op"), (16, 16, ".tran 1"), (16, 32, "a note")))
        (found,) = editor_findings(view)
        assert (found.rule, found.points, found.facts) == (
            "stacked_directive",
            ((16, 16),),
            {"count": 2},
        )

    def test_a_part_whose_symbol_was_not_found(self) -> None:
        view = SheetView(parts=(Part("U1", symbol="opamp", at=(96, 64), missing=True),))
        assert editor_findings(view) == [
            Finding(
                "unresolved_symbol",
                "Symbol 'opamp' of U1 was not found: the part has no pins here, so "
                "nothing at them is checked",
                refs=("U1",),
                points=((96, 64),),
                facts={"symbol": "opamp"},
            )
        ]


class TestTheCheckersReading:
    def test_boxes_that_share_an_area_and_boxes_that_only_touch(self) -> None:
        view = SheetView(
            parts=(
                part("R1", (0, 0, 32, 96)),
                part("R2", (16, 48, 48, 144)),
                part("R3", (32, 0, 64, 48)),
            )
        )
        (found,) = checker_findings(view)
        assert (found.rule, found.refs, found.points) == (
            "symbol_overlap",
            ("R1", "R2"),
            ((16, 48), (32, 96)),
        )

    def test_a_wire_through_a_box_and_one_along_its_edge(self) -> None:
        view = SheetView(
            parts=(part("R1", (0, 0, 32, 96)),),
            wires=((-16, 48, 48, 48), (0, 0, 0, 96)),
            labels=((-16, 48, "a"), (48, 48, "b"), (0, 0, "c"), (0, 96, "d")),
        )
        (found,) = checker_findings(view)
        assert (found.rule, found.refs, found.points) == (
            "wire_through_symbol",
            ("R1",),
            ((-16, 48), (48, 48)),
        )

    def test_a_wire_end_on_nothing_is_one_place_however_many_end_there(self) -> None:
        view = SheetView(wires=((0, 0, 64, 0), (64, 0, 64, 64)), labels=((0, 0, "a"),))
        (found,) = checker_findings(view)
        assert (found.rule, found.points) == ("dangling_wire_end", ((64, 64),))

    def test_text_inside_another_parts_box_but_not_its_own(self) -> None:
        own = Part("R1", box=BBox(0, 0, 32, 96), texts=((16, 16, "R1"),))
        other = Part("R2", box=BBox(100, 0, 132, 96), texts=((16, 48, "1k"),))
        view = SheetView(parts=(own, other), texts=((110, 48, ".op"),))
        found = checker_findings(view)
        assert [(f.rule, f.refs, f.detail) for f in found] == [
            ("text_in_symbol_body", ("R1",), "text '1k' is anchored inside the symbol's body box"),
            (
                "text_in_symbol_body",
                ("R2",),
                "text '.op' is anchored inside the symbol's body box",
            ),
        ]

    def test_a_part_is_called_by_its_reference_else_by_its_symbol(self) -> None:
        nameless = Part("", symbol="res", box=BBox(0, 0, 32, 96))
        view = SheetView(parts=(nameless, Part("", box=BBox(16, 16, 48, 48))))
        (found,) = checker_findings(view)
        assert found.refs == ("res", "<unnamed>")


class TestTheCheckersOtherFindings:
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
        (found,) = label_islands(view)
        assert (found.rule, found.points, found.facts) == (
            "label_island",
            ((200, 0), (200, 96)),
            {"net": "vdd"},
        )

    def test_a_wire_straight_between_two_pins_of_one_part(self) -> None:
        view = SheetView(
            parts=(Part("R1", pins=(("", 0, 0), ("", 0, 96))),), wires=((0, 0, 0, 96),)
        )
        (found,) = dropped_wires(view)
        assert (found.rule, found.refs, found.points) == (
            "dropped_wire",
            ("R1",),
            ((0, 0), (0, 96)),
        )

    def test_a_symbol_that_was_not_found_names_every_part_that_uses_it(self) -> None:
        view = SheetView(
            parts=(
                Part("U2", symbol="opamp", missing=True),
                Part("R1", symbol="res"),
                Part("U1", symbol="opamp", missing=True),
            )
        )
        (found,) = unresolved_symbols(view)
        assert (found.rule, found.refs, found.facts) == (
            "unresolved_symbol",
            ("U2", "U1"),
            {"symbol": "opamp"},
        )
