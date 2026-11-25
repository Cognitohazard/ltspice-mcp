"""``remove_component`` with ``cleanup_wires``: what the cleanup takes, what it
keeps and why, and the removals of one batch that find their target already
taken by an earlier op.

The cleanup walks out from the removed part's own pins. It takes a wire left
joining nothing, carries on from that wire's far end when nothing else is
there, and takes a label or ground flag the taken wiring leaves on nothing; a
wire that still joins two pins or labels by wiring stays, and the reply says
which two. Each case runs through ``edit_schematic`` and reads the reply and
the saved sheet.
"""

from __future__ import annotations

import hashlib
from itertools import permutations
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.schematic_ops import blank_sheet
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests._asc_ops import apply_one

pytestmark = pytest.mark.asyncio


def _res(ref: str, x: int, y: int) -> str:
    """A resistor at (x, y): pin 1 at (x, y-48), pin 2 at (x, y+48)."""
    return f"SYMBOL res {x} {y} R0\nSYMATTR InstName {ref}\nSYMATTR Value 1k"


def _sheet(state: SessionState, name: str, *lines: str) -> Path:
    path = Path(state.working_dir) / name
    text = blank_sheet() + "".join(f"{line}\n" for line in lines)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _records(path: Path, kind: str) -> list[str]:
    return [
        line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(kind)
    ]


async def _edit(state: SessionState, path: Path, ops: list[dict[str, Any]], **kw: Any) -> dict:
    result = await handle_edit_schematic(
        EditSchematicInput.model_validate(
            {
                "path": path.name,
                "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),  # noqa: ASYNC240 - a small file read before the call
                "ops": ops,
                **kw,
            }
        ),
        state,
    )
    assert result.structured_content is not None
    return dict(result.structured_content)


async def _remove(state: SessionState, path: Path, ref: str = "R1", **kw: Any) -> dict:
    """The results entry of one ``remove_component`` with cleanup, or ``{}``."""
    op = {"op": "remove_component", "reference": ref, "cleanup_wires": True, **kw}
    reply = await _edit(state, path, [op])
    assert "failures" not in reply, reply
    entries = reply.get("results", [])
    return {k: v for k, v in entries[0].items() if k not in ("index", "op")} if entries else {}


def _w(x1: int, y1: int, x2: int, y2: int) -> dict[str, dict[str, int]]:
    return {"from": {"x": x1, "y": y1}, "to": {"x": x2, "y": y2}}


# R1 at (96,244): its pins are A=(96,196) and B=(96,292).
_R1 = _res("R1", 96, 244)


class TestWhatTheCleanupTakes:
    async def test_a_chain_from_the_pin_to_nothing_goes_whole(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state, "chain.asc", _R1, "WIRE 96 196 96 100", "WIRE 96 100 192 100"
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 100, 192, 100), _w(96, 196, 96, 100)]
        }
        assert _records(sheet, "WIRE") == []

    async def test_a_chain_to_a_lone_pin_goes_whole(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "chain_pin.asc",
            _R1,
            _res("R2", 288, 244),
            "WIRE 96 196 96 100",
            "WIRE 96 100 288 100",
            "WIRE 288 100 288 196",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [
                _w(96, 100, 288, 100),
                _w(96, 196, 96, 100),
                _w(288, 100, 288, 196),
            ]
        }
        assert _records(sheet, "WIRE") == []

    async def test_a_chain_to_a_lone_pin_leaves_that_pin_floating(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "chain_float.asc",
            _R1,
            _res("R2", 288, 148),
            "WIRE 96 196 96 100",
            "WIRE 96 100 288 100",
        )
        reply = await _edit(
            asc_state, sheet, [{"op": "remove_component", "reference": "R1", "cleanup_wires": True}]
        )
        assert reply["results"][0]["removed_wires"] == [
            _w(96, 100, 288, 100),
            _w(96, 196, 96, 100),
        ]
        assert "Floating pin: R2.1 at (288,100)" in str(reply)

    async def test_a_direct_wire_to_a_lone_pin_goes(self, asc_state: SessionState):
        sheet = _sheet(asc_state, "direct.asc", _R1, _res("R2", 288, 244), "WIRE 96 196 288 196")
        assert await _remove(asc_state, sheet) == {"removed_wires": [_w(96, 196, 288, 196)]}

    async def test_a_label_the_taken_wire_leaves_on_nothing_goes_with_it(
        self, asc_state: SessionState
    ):
        """The label's name is on R2.1 too; it is taken all the same, as it lies
        on nothing once its wire is gone."""
        sheet = _sheet(
            asc_state,
            "labelled.asc",
            _R1,
            _res("R2", 288, 244),
            "WIRE 96 196 96 100",
            "FLAG 96 100 a",
            "FLAG 288 196 a",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 196, 96, 100)],
            "removed_labels": [{"text": "a", "x": 96, "y": 100}],
        }
        assert _records(sheet, "FLAG") == ["FLAG 288 196 a"]

    async def test_a_wire_on_to_a_ground_flag_goes_with_the_flag(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "to_ground.asc",
            _R1,
            "WIRE 96 196 96 100",
            "WIRE 96 100 192 100",
            "FLAG 192 100 0",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 100, 192, 100), _w(96, 196, 96, 100)],
            "removed_labels": [{"text": "0", "x": 192, "y": 100}],
        }
        assert _records(sheet, "WIRE") == [] and _records(sheet, "FLAG") == []

    async def test_a_label_on_the_pin_itself_goes(self, asc_state: SessionState):
        sheet = _sheet(asc_state, "pin_label.asc", _R1, "FLAG 96 196 a")
        assert await _remove(asc_state, sheet) == {
            "removed_labels": [{"text": "a", "x": 96, "y": 196}]
        }

    async def test_a_loop_from_the_pin_goes_whole(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "loop.asc",
            _R1,
            "WIRE 96 196 96 100",
            "WIRE 96 100 192 100",
            "WIRE 192 100 192 196",
            "WIRE 192 196 96 196",
        )
        reply = await _remove(asc_state, sheet)
        assert len(reply["removed_wires"]) == 4 and set(reply) == {"removed_wires"}
        assert _records(sheet, "WIRE") == []

    async def test_a_loop_through_a_pin_of_another_part_goes_whole(self, asc_state: SessionState):
        """pin -> P -> Q -> pin with R2.1 at Q: the loop's only pin is R2.1, so
        all three wires join nothing and go, and R2.1 is left floating."""
        sheet = _sheet(
            asc_state,
            "loop_pin.asc",
            _R1,
            _res("R2", 192, 148),
            "WIRE 96 196 96 100",
            "WIRE 96 100 192 100",
            "WIRE 192 100 96 196",
        )
        reply = await _edit(
            asc_state, sheet, [{"op": "remove_component", "reference": "R1", "cleanup_wires": True}]
        )
        assert reply["results"][0]["removed_wires"] == [
            _w(96, 100, 192, 100),
            _w(96, 196, 96, 100),
            _w(192, 100, 96, 196),
        ]
        assert _records(sheet, "WIRE") == []
        assert "Floating pin: R2.1 at (192,100)" in str(reply)

    async def test_each_copy_of_a_drawn_twice_wire_is_listed(self, asc_state: SessionState):
        sheet = _sheet(asc_state, "twice.asc", _R1, "WIRE 96 196 96 100", "WIRE 96 100 96 196")
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 100, 96, 196), _w(96, 196, 96, 100)]
        }

    async def test_nothing_past_a_wire_end_that_holds_a_label_is_touched(
        self, asc_state: SessionState
    ):
        """The walk stops at a labelled end: the wire on from there is not the
        removed part's, and the label stays on it."""
        sheet = _sheet(
            asc_state,
            "stop.asc",
            _R1,
            "WIRE 96 196 96 100",
            "WIRE 96 100 192 100",
            "FLAG 96 100 a",
        )
        assert await _remove(asc_state, sheet) == {"removed_wires": [_w(96, 196, 96, 100)]}
        assert _records(sheet, "WIRE") == ["WIRE 96 100 192 100"]
        assert _records(sheet, "FLAG") == ["FLAG 96 100 a"]


class TestWhatTheCleanupKeeps:
    async def test_wires_that_join_two_pins_at_a_point_the_walk_reached_stay(
        self, asc_state: SessionState
    ):
        sheet = _sheet(
            asc_state,
            "three.asc",
            _R1,
            _res("R2", 16, 148),
            _res("R3", 176, 148),
            "WIRE 96 196 96 100",
            "WIRE 16 100 96 100",
            "WIRE 96 100 176 100",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 196, 96, 100)],
            "kept_wires": [
                {"wire": _w(16, 100, 96, 100), "joins": ["R2.1", "R3.1"]},
                {"wire": _w(96, 100, 176, 100), "joins": ["R2.1", "R3.1"]},
            ],
        }
        assert _records(sheet, "WIRE") == ["WIRE 16 100 96 100", "WIRE 96 100 176 100"]

    async def test_two_wires_that_met_at_the_pin_still_meet(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "bus.asc",
            _R1,
            _res("R2", 16, 244),
            _res("R3", 176, 244),
            "WIRE 16 196 96 196",
            "WIRE 96 196 176 196",
            "WIRE 96 292 96 340",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 292, 96, 340)],
            "kept_wires": [
                {"wire": _w(16, 196, 96, 196), "joins": ["R2.1", "R3.1"]},
                {"wire": _w(96, 196, 176, 196), "joins": ["R2.1", "R3.1"]},
            ],
        }

    async def test_a_wire_from_a_ground_flag_on_the_pin_stays(self, asc_state: SessionState):
        sheet = _sheet(
            asc_state,
            "flag_on_pin.asc",
            _R1,
            _res("R2", 288, 244),
            "WIRE 96 196 288 196",
            "FLAG 96 196 0",
        )
        assert await _remove(asc_state, sheet) == {
            "kept_wires": [{"wire": _w(96, 196, 288, 196), "joins": ["R2.1", "label '0'"]}]
        }
        assert _records(sheet, "FLAG") == ["FLAG 96 196 0"]

    async def test_wires_between_labels_of_one_name_stay_as_wiring_joins_them(
        self, asc_state: SessionState
    ):
        sheet = _sheet(
            asc_state,
            "named.asc",
            _R1,
            _res("R2", 16, 244),
            _res("R3", 176, 244),
            "WIRE 16 196 96 196",
            "WIRE 96 196 176 196",
            "FLAG 16 196 a",
            "FLAG 176 196 a",
        )
        assert await _remove(asc_state, sheet) == {
            "kept_wires": [
                {"wire": _w(16, 196, 96, 196), "joins": ["R2.1", "R3.1"]},
                {"wire": _w(96, 196, 176, 196), "joins": ["R2.1", "R3.1"]},
            ]
        }

    async def test_wires_to_two_labels_of_one_name_off_any_pin_stay(
        self, asc_state: SessionState
    ):
        """Two wires meet at the removed pin and each ends on a label 'a' that
        is on no pin: the wiring joins the two labels, so both wires stay."""
        sheet = _sheet(
            asc_state,
            "named_off_pins.asc",
            _R1,
            "WIRE 16 196 96 196",
            "WIRE 96 196 176 196",
            "FLAG 16 196 a",
            "FLAG 176 196 a",
        )
        assert await _remove(asc_state, sheet) == {
            "kept_wires": [
                {"wire": _w(16, 196, 96, 196), "joins": ["label 'a'", "label 'a'"]},
                {"wire": _w(96, 196, 176, 196), "joins": ["label 'a'", "label 'a'"]},
            ]
        }
        assert _records(sheet, "FLAG") == ["FLAG 16 196 a", "FLAG 176 196 a"]

    async def test_a_wire_on_a_pin_another_part_shares_stays_on_that_pin(
        self, asc_state: SessionState
    ):
        """R3.2 sits on R1.1: the wire there is still on R3.2, so the walk does
        not start from that point."""
        sheet = _sheet(
            asc_state,
            "stacked.asc",
            _R1,
            _res("R2", 288, 244),
            _res("R3", 96, 148),
            "WIRE 96 196 288 196",
            "WIRE 96 292 96 340",
        )
        assert await _remove(asc_state, sheet) == {
            "removed_wires": [_w(96, 292, 96, 340)],
            "kept_wires": [{"wire": _w(96, 196, 288, 196), "on_pin": "R3.2"}],
        }

    async def test_cleanup_off_takes_nothing_and_reports_nothing(self, asc_state: SessionState):
        """Without cleanup the loose ends stay, and the sheet's findings name them."""
        sheet = _sheet(
            asc_state, "off.asc", _R1, "WIRE 96 196 96 100", "FLAG 96 100 a", "FLAG 96 292 b"
        )
        reply = await _edit(asc_state, sheet, [{"op": "remove_component", "reference": "R1"}])
        assert "results" not in reply
        assert reply["rules_run"]["dangling_wire_end"] == 1
        assert reply["rules_run"]["dangling_label"] == 1
        assert reply["warnings"] == [
            "Wire end at (96,196) meets no pin, net label, or other wire",
            "Dangling label 'b' at (96,292)",
        ]
        assert _records(sheet, "WIRE") == ["WIRE 96 196 96 100"]
        assert _records(sheet, "FLAG") == ["FLAG 96 100 a", "FLAG 96 292 b"]


# R2.1 and R3.1 joined twice: through R1.1's point and through R1.2's, round a
# loop whose sides are plain wires (a wire straight from one part's pin to its
# other pin is not in LTspice's netlist). With R1 gone, either path alone keeps
# the two pins one net.
_TWO_PATHS = (
    "WIRE 16 196 96 196",
    "WIRE 96 196 176 196",
    "WIRE 16 292 96 292",
    "WIRE 96 292 176 292",
    "WIRE 16 196 16 292",
    "WIRE 176 196 176 292",
)


class TestTheOrderOfTheRecords:
    async def test_the_reply_and_the_sheet_do_not_depend_on_the_order_of_records(
        self, asc_state: SessionState
    ):
        """Which of two equal paths stays is decided by the wires' places, not
        by which record the sheet lists first."""
        parts = (
            _R1,
            _res("R2", -64, 244),
            _res("R3", 256, 244),
            "WIRE -64 196 16 196",
            "WIRE 176 196 256 196",
        )
        flags = ("FLAG 96 340 x",)
        seen: set[str] = set()
        replies = []
        for n, order in enumerate(list(permutations(range(len(_TWO_PATHS))))[::97]):
            wires = [_TWO_PATHS[i] for i in order]
            sheet = _sheet(asc_state, f"order{n}.asc", *parts, *wires, *flags)
            replies.append(await _remove(asc_state, sheet))
            seen.add(repr(sorted(_records(sheet, "WIRE"))))
        assert len(seen) == 1
        assert all(reply == replies[0] for reply in replies)
        assert replies[0]["removed_wires"] == [_w(16, 196, 96, 196), _w(96, 196, 176, 196)]

    async def test_reversed_records_give_the_same_reply(self, asc_state: SessionState):
        lines = (
            "WIRE 96 196 96 100",
            "WIRE 16 100 96 100",
            "WIRE 96 100 176 100",
            "WIRE 96 292 96 340",
            "FLAG 96 340 a",
        )
        parts = (_R1, _res("R2", 16, 148), _res("R3", 176, 148))
        forward = await _remove(asc_state, _sheet(asc_state, "fwd.asc", *parts, *lines))
        backward = await _remove(asc_state, _sheet(asc_state, "bwd.asc", *parts, *lines[::-1]))
        assert forward == backward
        assert forward["removed_labels"] == [{"text": "a", "x": 96, "y": 340}]


# R1 and R2 joined over the top through (96,100) and (288,100).
_OVER_THE_TOP = (
    _R1,
    _res("R2", 288, 244),
    "WIRE 96 196 96 100",
    "WIRE 96 100 288 100",
    "WIRE 288 100 288 196",
)


class TestRemovalsInOneBatch:
    async def test_removing_both_parts_with_cleanup_takes_the_wiring_once(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "both.asc", *_OVER_THE_TOP)
        reply = await _edit(
            asc_state,
            sheet,
            [
                {"op": "remove_component", "reference": "R1", "cleanup_wires": True},
                {"op": "remove_component", "reference": "R2", "cleanup_wires": True},
            ],
        )
        assert reply["applied"] == 2
        assert [entry["index"] for entry in reply["results"]] == [0]
        assert len(reply["results"][0]["removed_wires"]) == 3
        assert _records(sheet, "WIRE") == []

    async def test_cleanup_on_the_second_part_alone_walks_through_the_first_ones_pin(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "second.asc", *_OVER_THE_TOP)
        reply = await _edit(
            asc_state,
            sheet,
            [
                {"op": "remove_component", "reference": "R1"},
                {"op": "remove_component", "reference": "R2", "cleanup_wires": True},
            ],
        )
        assert [entry["index"] for entry in reply["results"]] == [1]
        assert reply["results"][0]["removed_wires"] == [
            _w(96, 100, 288, 100),
            _w(96, 196, 96, 100),
            _w(288, 100, 288, 196),
        ]
        assert _records(sheet, "WIRE") == []

    async def test_a_removal_of_what_the_cleanup_took_says_which_op_took_it(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "after.asc", _R1, "WIRE 96 196 96 100", "FLAG 96 100 a")
        reply = await _edit(
            asc_state,
            sheet,
            [
                {"op": "remove_component", "reference": "R1", "cleanup_wires": True},
                {"op": "remove_wire", "x1": 96, "y1": 100, "x2": 96, "y2": 196},
                {"op": "remove_wire", "x": 96, "y": 100},
                {"op": "remove_net_label", "x": 96, "y": 100},
            ],
        )
        assert reply["applied"] == 4
        assert reply["results"][1:] == [
            {"index": 1, "op": "remove_wire", "already_removed_by": 0},
            {"index": 2, "op": "remove_wire", "already_removed_by": 0},
            {"index": 3, "op": "remove_net_label", "already_removed_by": 0},
        ]

    async def test_both_forms_of_remove_wire_name_the_op_that_took_a_wire_drawn_in_the_batch(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "drawn.asc", _R1, _res("R2", 288, 244))
        reply = await _edit(
            asc_state,
            sheet,
            [
                {
                    "op": "wire_pins",
                    "from_pin": "R1.1",
                    "to_pin": "R2.1",
                    "waypoints": [{"x": 96, "y": 100}, {"x": 288, "y": 100}],
                },
                {"op": "remove_component", "reference": "R1", "cleanup_wires": True},
                {"op": "remove_wire", "x": 96, "y": 196},
                {"op": "remove_wire", "x1": 96, "y1": 196, "x2": 96, "y2": 100},
            ],
        )
        assert "failures" not in reply, reply.get("failures")
        assert _records(sheet, "WIRE") == []
        assert reply["results"][-2:] == [
            {"index": 2, "op": "remove_wire", "already_removed_by": 1},
            {"index": 3, "op": "remove_wire", "already_removed_by": 1},
        ]

    async def test_a_dry_run_keeps_the_same_record(self, asc_state: SessionState):
        sheet = _sheet(asc_state, "dry.asc", _R1, "WIRE 96 196 96 100", "FLAG 96 100 a")
        before = sheet.read_bytes()
        reply = await _edit(
            asc_state,
            sheet,
            [
                {"op": "remove_component", "reference": "R1", "cleanup_wires": True},
                {"op": "remove_net_label", "x": 96, "y": 100},
            ],
            dry_run=True,
        )
        assert reply["results"] == [
            {
                "index": 0,
                "op": "remove_component",
                "removed_wires": [_w(96, 196, 96, 100)],
                "removed_labels": [{"text": "a", "x": 96, "y": 100}],
            },
            {"index": 1, "op": "remove_net_label", "already_removed_by": 0},
        ]
        assert sheet.read_bytes() == before

    async def test_a_later_call_that_removes_what_the_cleanup_took_is_refused(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "later.asc", _R1, "WIRE 96 196 96 100", "FLAG 96 100 a")
        await _remove(asc_state, sheet)
        before = sheet.read_bytes()
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_wire", "x1": 96, "y1": 196, "x2": 96, "y2": 100})
        assert str(refused.value).startswith(
            "No wire to remove: no wire has an end at (96,100) or (96,196)."
        )
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_net_label", "x": 96, "y": 100})
        assert str(refused.value).startswith("No net label at (96,100), and nothing else is there.")
        assert sheet.read_bytes() == before

    async def test_a_later_call_names_the_wires_still_at_the_ends_it_named(
        self, asc_state: SessionState
    ):
        """The cleanup took the wire from the pin to P; the wire on from P,
        past the label there, stays and the refusal names it."""
        sheet = _sheet(
            asc_state,
            "later_named.asc",
            _R1,
            "WIRE 96 196 96 100",
            "WIRE 96 100 192 100",
            "FLAG 96 100 a",
        )
        await _remove(asc_state, sheet)
        before = sheet.read_bytes()
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_wire", "x1": 96, "y1": 196, "x2": 96, "y2": 100})
        message = str(refused.value)
        assert "the wires with an end at (96,100) or (96,196) are (96,100)->(192,100)" in message
        assert sheet.read_bytes() == before

    async def test_a_wire_moved_away_earlier_in_the_batch_is_not_said_removed(
        self, asc_state: SessionState
    ):
        """Only a removal op is said to have removed a wire: one an earlier
        move took off the point leaves a removal there refused."""
        sheet = _sheet(asc_state, "moved.asc", "WIRE 400 196 400 100")
        reply = await _edit(
            asc_state,
            sheet,
            [
                {"op": "move", "dx": 32, "dy": 0, "wires": [{"x1": 400, "y1": 196, "x2": 400, "y2": 100}]},
                {"op": "remove_wire", "x": 400, "y": 100},
            ],
            dry_run=True,
        )
        assert reply["failures"] == [
            {
                "index": 1,
                "op": "remove_wire",
                "error": (
                    "No wire to remove: no wire has an end at (400,100). To see the wires "
                    "of a net with their ends, call inspect(queries=[{\"kind\": \"net\", \"path\": "
                    "<sheet>, \"at\": [<x>, <y>]}]) on a "
                    "point of it."
                ),
            }
        ]

    async def test_a_point_no_wire_ends_on_is_refused(self, asc_state: SessionState):
        sheet = _sheet(asc_state, "nowire.asc", _R1, "WIRE 96 196 96 100")
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_wire", "x": 96, "y": 50})
        assert "no wire has an end at (96,50)" in str(refused.value)


class TestARemovalThatFindsNothing:
    async def test_a_missing_segment_is_refused_naming_the_wires_at_its_ends(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "seg.asc", _R1, _res("R2", 288, 244), "WIRE 96 196 288 196")
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_wire", "x1": 96, "y1": 196, "x2": 96, "y2": 100})
        assert "(96,196)->(288,196)" in str(refused.value)

    async def test_a_long_list_of_wires_at_the_ends_is_cut_at_ten(self, asc_state: SessionState):
        spokes = [f"WIRE 400 400 {500 + 16 * i} 400" for i in range(12)]
        sheet = _sheet(asc_state, "spokes.asc", *spokes)
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_wire", "x1": 400, "y1": 400, "x2": 400, "y2": 300})
        message = str(refused.value)
        assert message.count("->") == 10 and "and 2 more" in message

    async def test_a_missing_label_is_refused_naming_what_is_at_the_point(
        self, asc_state: SessionState
    ):
        sheet = _sheet(asc_state, "lbl.asc", _R1, _res("R2", 288, 244), "WIRE 96 196 288 196")
        with pytest.raises(NetlistError) as refused:
            apply_one(asc_state, sheet, {"op": "remove_net_label", "x": 96, "y": 196})
        message = str(refused.value)
        assert "R1.1" in message and "(96,196)->(288,196)" in message
