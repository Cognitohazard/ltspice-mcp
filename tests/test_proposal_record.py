"""A record of answers to proposals says which proposals moved, and how, and
its driver puts each proposal to the same sheet."""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.errors import NetlistError
from tests._proposal_record import Asked, differences, driver_of, entry_of


def _asked(*shapes: str) -> list[Asked]:
    return [
        Asked({"at": index}, {"said": f"{shape} {index}"}, shape)
        for index, shape in enumerate(shapes)
    ]


RECORDED, _EXAMPLES = entry_of(_asked("placed", "refused", "placed", "placed", "placed"))


def test_answers_as_recorded_say_nothing() -> None:
    assert differences(_asked("placed", "refused", "placed", "placed", "placed"), RECORDED) == []


def test_each_move_from_a_shape_to_a_shape_names_its_proposals_and_both_shapes() -> None:
    now = _asked("refused", "placed", "refused", "refused", "refused")
    assert differences(now, RECORDED) == [
        '4 moved, as {"at": 0}; {"at": 2}; {"at": 3} and 1 more',
        "  from: placed",
        "  to:   refused",
        '1 moved, as {"at": 1}',
        "  from: refused",
        "  to:   placed",
    ]


def test_shapes_that_differ_only_at_their_end_are_told_apart() -> None:
    """A label's shape ends with what was at the point, which is what tells
    two refusals apart."""
    long = "refused: Refused to add net '…' at (#,#): " + "x" * 200
    recorded, _examples = entry_of(_asked(f"{long} [1 named]"))
    (_count, was, now) = differences(_asked(f"{long} [2 named]"), recorded)
    assert was.endswith("[1 named]") and now.endswith("[2 named]")


def test_an_answer_changed_only_in_its_particulars_is_said_to_be() -> None:
    now = _asked("placed", "refused", "placed", "placed", "placed")
    now[2] = Asked(now[2].proposal, {"said": "placed elsewhere"}, "placed")
    (said,) = differences(now, RECORDED)
    assert said.startswith("no answer moved to another shape")


def test_other_proposals_are_not_held_to_the_recorded_answers_but_their_counts_are_shown() -> None:
    said, *counts = differences(_asked("placed", "refused"), RECORDED)
    assert said.startswith("the proposals are not those recorded (5 then, 2 now)")
    assert counts == ["  4 -> 1: placed"]


#: Two named wires that cross at (200,100) without being joined.
_CROSSING = "WIRE 0 100 400 100\nWIRE 200 0 200 200\nFLAG 0 100 VDD\nFLAG 200 0 VSS\n"


@pytest.mark.usefixtures("suite_symbols_only")
class TestTheDriver:
    def _driver(self, tmp_path: Path):
        sheet = tmp_path / "crossing.asc"
        sheet.write_text("Version 4\nSHEET 1 880 680\n" + _CROSSING, encoding="utf-8")
        return driver_of(sheet)

    def test_it_gives_the_ops_refusal_and_takes_the_label_back(self, tmp_path: Path) -> None:
        driver = self._driver(tmp_path)
        with driver.label_placed({"net": "fresh", "x": 200, "y": 100}) as refused:
            assert refused is not None and "Refused to add net 'fresh'" in refused
        with driver.label_placed({"net": "fresh", "x": 100, "y": 100}) as refused:
            assert refused is None
            assert len(driver.labels()) == 3
        assert len(driver.labels()) == 2

    def test_an_error_while_the_block_reads_the_sheet_is_no_refusal(self, tmp_path: Path) -> None:
        """Only the op's own refusal is given; anything else stops the test."""
        driver = self._driver(tmp_path)
        with (
            pytest.raises(NetlistError, match="read"),
            driver.label_placed({"net": "fresh", "x": 100, "y": 100}),
        ):
            raise NetlistError("the sheet did not read")
        assert len(driver.labels()) == 2
