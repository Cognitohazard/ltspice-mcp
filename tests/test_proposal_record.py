"""A record of answers to proposals says which proposals moved, and how."""

from __future__ import annotations

from tests._proposal_record import Asked, differences, entry_of


def _asked(*shapes: str) -> list[Asked]:
    return [
        Asked({"at": index}, {"said": f"{shape} {index}"}, shape)
        for index, shape in enumerate(shapes)
    ]


RECORDED, _EXAMPLES = entry_of(_asked("placed", "refused", "placed", "placed", "placed"))


def test_answers_as_recorded_say_nothing() -> None:
    assert differences(_asked("placed", "refused", "placed", "placed", "placed"), RECORDED) == []


def test_each_move_from_a_shape_to_a_shape_is_one_line_naming_its_proposals() -> None:
    now = _asked("refused", "placed", "refused", "refused", "refused")
    assert differences(now, RECORDED) == [
        '4 from \'placed\' to \'refused\': {"at": 0}; {"at": 2}; {"at": 3} and 1 more',
        "1 from 'refused' to 'placed': {\"at\": 1}",
    ]


def test_an_answer_changed_only_in_its_particulars_is_said_to_be() -> None:
    now = _asked("placed", "refused", "placed", "placed", "placed")
    now[2] = Asked(now[2].proposal, {"said": "placed elsewhere"}, "placed")
    (said,) = differences(now, RECORDED)
    assert said.startswith("no answer moved to another shape")


def test_other_proposals_are_not_held_to_the_recorded_answers() -> None:
    (said,) = differences(_asked("placed", "refused"), RECORDED)
    assert said.startswith("the proposals are not those recorded (5 then, 2 now)")
