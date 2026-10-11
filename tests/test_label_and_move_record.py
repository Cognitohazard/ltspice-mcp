"""What placing a label and moving a part are answered on every sheet of the suite, held to a record.

``add_net_label`` refuses a label that would join two nets that each carry a
name, and says when a name is repeated and when a net gets a second one.
``move_component`` refuses a pin landing where wires of two named nets cross,
says the join a pin makes on a crossing that it does not refuse, and says the
wires a move leaves behind (``add_component`` makes the same check of its
pins). Each works the answer out from the sheet as the schematic editor holds
it. Work that moves those checks is held to this, as the route planner's is to
``test_route_planner_record.py``: a fixed set of proposals is put to each op on
every sheet in the suite the editor opens, through the op itself, and what it
answers is as it was when the record was made.

A label is asked for under each name the sheet has, under ground and under a
name it does not have: at each point where two wires cross, at each label, at
a free point, at each wire's ends and middle, and at each pin, the last by the
pin's name. A part is asked to move so that each of its pins lands on each
crossing, wire end, wire middle, label and pin of another part, and to move
to where it already is.

The record is ``fixtures/label_and_move_record.json``, in the form
``tests/_proposal_record.py`` describes. The shape of a label's answer ends
with what was there, which the sentence alone does not say: whether the name
was already on the sheet, and how many nets met at the point and how many of
those carried a name. No label is placed where two nets met without a word
of it. To make the record again, run this file with
``LTSPICE_MCP_RECORD_LABELS_AND_MOVES=1``. Like the other records it is the
same on every machine.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pytest
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.connectivity import label_folded_nets
from ltspice_mcp.lib.routing import wires_through
from ltspice_mcp.lib.schematic_ops import (
    OpAddNetLabel,
    OpMoveComponent,
    apply_op_inplace,
    collect_component_geometry,
    named_labels,
    net_partition,
    resolve_pin,
    wire_segments_of,
)
from tests._proposal_record import Asked, entry_of, open_staged, without_particulars, write_record
from tests._schematic_fixtures import SUITE_SHEETS, TESTS, suite_name

_RECORD = TESTS / "fixtures" / "label_and_move_record.json"
_RECORDING = os.environ.get("LTSPICE_MCP_RECORD_LABELS_AND_MOVES") == "1"

#: A name no sheet of the suite labels a net with.
_FRESH = "a_name_no_sheet_has"

#: The most proposals of each kind put to one sheet at the points few sheets
#: have (a crossing, a label, a free point) and at the rest. A sheet with more
#: has every n-th taken, which keeps the largest sheets from being most of the run.
_MOST_AT_RARE_POINTS = 300
_MOST_ELSEWHERE = 300

#: A label placed where two or more nets met with nothing said of it.
_SILENT_JOIN = re.compile(r"^placed \[.*; [2-9] at the point")

Point = tuple[int, int]
Wire = tuple[int, int, int, int]


def _every_nth(asked: list[dict[str, Any]], most: int) -> list[dict[str, Any]]:
    return asked[:: -(-len(asked) // most)] if len(asked) > most else asked


def _in_order(points: list[Point]) -> list[Point]:
    """``points`` once each, in the order first met."""
    return list(dict.fromkeys(points))


def _crossings(wires: list[Wire]) -> list[Point]:
    """Each point where a level wire and an upright one cross, inside both."""
    found: list[Point] = []
    for x1, y1, x2, y2 in wires:
        if y1 != y2 or x1 == x2:
            continue
        for a1, b1, a2, b2 in wires:
            if a1 != a2 or b1 == b2:
                continue
            if min(x1, x2) < a1 < max(x1, x2) and min(b1, b2) < y1 < max(b1, b2):
                found.append((a1, y1))
    return _in_order(found)


def _places(editor: AscEditor) -> tuple[list[Point], list[Point], list[tuple[str, int, int]]]:
    """The points of a sheet few sheets have, the rest of its wiring, and its pins by name."""
    wires = wire_segments_of(editor)
    labels = [(int(label.coord.X), int(label.coord.Y)) for label in editor.labels]
    pins = [
        (f"{part['ref']}.{pin['name']}", pin["x"], pin["y"])
        for part in collect_component_geometry(editor)
        for pin in part["pins"]
    ]
    every = [point for x1, y1, x2, y2 in wires for point in ((x1, y1), (x2, y2))]
    every += labels + [(x, y) for _name, x, y in pins]
    free = (max((x for x, _y in every), default=0) + 160, max((y for _x, y in every), default=0))
    rare = _in_order([*_crossings(wires), *labels, free])
    ends = [point for x1, y1, x2, y2 in wires for point in ((x1, y1), (x2, y2))]
    # A point of the wire itself, which a point snapped to the grid need not be.
    middles = [
        ((x1 + x2) // 2, (y1 + y2) // 2) for x1, y1, x2, y2 in wires if x1 == x2 or y1 == y2
    ]
    wiring = [point for point in _in_order(ends + middles) if point not in rare]
    return rare, wiring, pins


def _label_proposals(editor: AscEditor) -> list[dict[str, Any]]:
    """The labels asked of a sheet, in an order that depends on the sheet alone."""
    rare, wiring, pins = _places(editor)
    names = list(dict.fromkeys([*(label.text for label in editor.labels), "0", _FRESH]))
    at_rare = [{"net": name, "x": x, "y": y} for x, y in rare for name in names]
    elsewhere = [{"net": name, "x": x, "y": y} for x, y in wiring for name in names]
    elsewhere += [{"net": name, "pin": pin} for pin, _x, _y in pins for name in names]
    return _every_nth(at_rare, _MOST_AT_RARE_POINTS) + _every_nth(elsewhere, _MOST_ELSEWHERE)


def _move_proposals(editor: AscEditor) -> list[dict[str, Any]]:
    """The moves asked of a sheet: each pin of each part onto each place, and
    each part to where it is."""
    rare, wiring, pins = _places(editor)
    onto_rare: list[dict[str, Any]] = []
    elsewhere: list[dict[str, Any]] = []
    for part in collect_component_geometry(editor):
        ref = part["ref"]
        at, _rotation = editor.get_component_position(ref)
        here = (int(at.X), int(at.Y))
        elsewhere.append({"reference": ref, "x": here[0], "y": here[1]})
        others = [(x, y) for name, x, y in pins if name.rsplit(".", 1)[0] != ref]
        for pin in part["pins"]:
            for asked, places in ((onto_rare, rare), (elsewhere, _in_order(wiring + others))):
                asked.extend(
                    {
                        "reference": ref,
                        "x": here[0] + x - pin["x"],
                        "y": here[1] + y - pin["y"],
                    }
                    for x, y in places
                )
    return _every_nth(onto_rare, _MOST_AT_RARE_POINTS) + _every_nth(elsewhere, _MOST_ELSEWHERE)


def _what_is_there(editor: AscEditor) -> Any:
    """For a point and a name: whether the name is already on the sheet, and
    the nets that meet at the point with how many of them carry a name.

    Worked out here from the partition, apart from the op's own reading of it:
    it is what tells apart answers the op words alike.
    """
    partition = net_partition(editor)
    node_of = label_folded_nets(partition)
    wires = wire_segments_of(editor)
    names_of: dict[Point, set[str]] = defaultdict(set)
    for coord, texts in partition.label_texts.items():
        names_of[node_of(coord)].update(named_labels(frozenset(texts)))
    on_the_sheet = {text for texts in partition.label_texts.values() for text in texts}

    def there(net: str, at: Point) -> str:
        nodes = {node_of(wire[:2]) for wire in wires_through(at, wires)}
        # A pin or a label there with no wire is a net too; a bare point is none.
        if at in partition.members.get(partition.root(at), ()):
            nodes.add(node_of(at))
        named = [node for node in nodes if names_of[node]]
        if net == "0":
            name = "ground"
        elif any(net in names_of[node] for node in nodes):
            name = "a name already here"
        elif net in on_the_sheet:
            name = "a name already elsewhere"
        else:
            name = "a new name"
        return f"{name}; {len(nodes)} at the point, {len(named)} named"

    return there


def _labels_asked(editor: AscEditor) -> list[Asked]:
    sheet = Path(editor.asc_file_path)
    there = _what_is_there(editor)
    asked: list[Asked] = []
    for proposal in _label_proposals(editor):
        had = len(editor.labels)
        at = (
            resolve_pin(proposal["pin"], editor)
            if "pin" in proposal
            else (proposal["x"], proposal["y"])
        )
        try:
            placed = apply_op_inplace(editor, OpAddNetLabel(op="add_net_label", **proposal), sheet)
        except NetlistError as refused:
            answer: dict[str, Any] = {"refused": str(refused)}
            text = "refused: " + answer["refused"]
        else:
            answer = {"warnings": list(cast("list[str]", placed.get("warnings", [])))}
            text = "placed: " + " | ".join(answer["warnings"]) if answer["warnings"] else "placed"
        # The sheet is put back as it was, so each proposal is put to the same sheet.
        del editor.labels[had:]
        answer["there"] = there(proposal["net"], at)
        # Its numbers are kept: how many nets met there is what the shape is for.
        asked.append(Asked(proposal, answer, f"{without_particulars(text)} [{answer['there']}]"))
    return asked


def _moves_asked(editor: AscEditor) -> list[Asked]:
    sheet = Path(editor.asc_file_path)
    asked: list[Asked] = []
    for proposal in _move_proposals(editor):
        at, rotation = editor.get_component_position(proposal["reference"])
        try:
            moved = apply_op_inplace(
                editor, OpMoveComponent(op="move_component", **proposal), sheet
            )
        except NetlistError as refused:
            answer: dict[str, Any] = {"refused": str(refused)}
            text = "refused: " + answer["refused"]
        else:
            answer = {"warnings": list(cast("list[str]", moved.get("warnings", [])))}
            text = "moved: " + " | ".join(answer["warnings"]) if answer["warnings"] else "moved"
        editor.set_component_position(proposal["reference"], at, rotation)
        asked.append(Asked(proposal, answer, without_particulars(text)))
    return asked


_OPS = {"labels": _labels_asked, "moves": _moves_asked}


def answers_of(
    sheet: Path, sandbox: Path
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]] | None:
    """What the record holds of ``sheet`` for each op and an example of each
    shape, or ``None`` for a sheet the editor cannot open."""
    editor = open_staged(sheet, sandbox)
    if editor is None:
        return None
    entry: dict[str, Any] = {}
    examples: dict[str, dict[str, dict[str, Any]]] = {}
    for op, asked_of in _OPS.items():
        entry[op], examples[op] = entry_of(asked_of(editor))
    return entry, examples


@pytest.fixture(autouse=True)
def _only_the_suites_own_symbols(asc_symbols: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No LTspice library the machine has is searched, and spicelib's symbol
    cache, which ``open_staged`` empties, is put back afterwards."""
    monkeypatch.setattr(AscEditor, "simulator_lib_paths", [])
    monkeypatch.setattr(AscEditor, "symbol_cache", dict(AscEditor.symbol_cache))


def _recorded() -> dict[str, Any]:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _RECORDING, reason="set LTSPICE_MCP_RECORD_LABELS_AND_MOVES=1 to record")
def test_record_what_the_ops_answer(tmp_path: Path):
    sheets: dict[str, Any] = {}
    examples: dict[str, dict[str, dict[str, Any]]] = {op: {} for op in _OPS}
    for number, sheet in enumerate(SUITE_SHEETS):
        sandbox = tmp_path / f"sheet{number}"
        sandbox.mkdir()
        answered = answers_of(sheet, sandbox)
        if answered is None:
            continue
        sheets[suite_name(sheet)], seen = answered
        for op, by_shape in seen.items():
            for shape, example in by_shape.items():
                examples[op].setdefault(shape, {"sheet": suite_name(sheet), **example})
    write_record(
        _RECORD,
        {"sheets": sheets, "examples": {op: dict(sorted(examples[op].items())) for op in _OPS}},
    )


def _shapes(record: dict[str, Any], op: str) -> dict[str, int]:
    shapes: dict[str, int] = {}
    for entry in record["sheets"].values():
        for shape, count in entry[op]["shapes"].items():
            shapes[shape] = shapes.get(shape, 0) + count
    return shapes


@pytest.mark.parametrize(
    ("op", "said"),
    [
        # A label refused for joining two named nets: by a name another net
        # has, and by landing where two named nets cross.
        ("labels", ("refused: Refused to add net", "[a name already elsewhere; 1 at the point")),
        ("labels", ("refused: Refused to add net", "[a new name; 2 at the point, 2 named]")),
        ("labels", ("refused: Refused to add net", "[a name already here; 2 at the point")),
        ("labels", ("refused: Refused to add ground", "[ground; 2 at the point, 2 named]")),
        # What it says of a label it places.
        ("labels", ("already labels a net at",)),
        ("labels", ("becomes a second name for it", "1 at the point, 1 named]")),
        ("labels", ("becomes a second name for it", "is where # wires cross", "1 named]")),
        # Two nets joined at a crossing, where at most one of them has a name.
        ("labels", ("placed: (#,#) is where # wires cross", "[a new name; 2 at the point")),
        ("labels", ("placed: (#,#) is where # wires cross", "[ground; 2 at the point, 0")),
        ("labels", ("placed: (#,#) is where # wires cross", "[ground; 2 at the point, 1")),
        # What it places without a word: on nothing and on one net.
        ("labels", ("placed [a new name; 0 at the point, 0 named]",)),
        ("labels", ("placed [a new name; 1 at the point, 0 named]",)),
        ("labels", ("placed [ground; 1 at the point, 1 named]",)),
        (
            "moves",
            ("refused: Refused to move", "where wires of nets that each carry a name cross"),
        ),
        ("moves", ("moved: PIN at (#,#) is on a point where # wires cross",)),
        ("moves", ("a pin on a crossing joins them. | PIN at (#,#) is on a point where",)),
        ("moves", ("wires left at old pin coordinates", "a pin on a crossing joins them")),
        ("moves", ("moved: wires left at old pin coordinates",)),
    ],
)
def test_the_record_asks_enough_to_hold_every_answer(op: str, said: tuple[str, ...]):
    """Each refusal and each advisory the two ops have is in it more than once,
    and so is each place where an op says nothing."""
    assert not [shape for shape in _shapes(_recorded(), "labels") if _SILENT_JOIN.search(shape)]
    shapes = _shapes(_recorded(), op)
    seen = sum(count for shape, count in shapes.items() if all(part in shape for part in said))
    assert seen >= 2, f"{said!r} is in {seen} recorded answers"


def test_a_part_moved_and_nothing_said_is_most_of_the_record():
    assert _shapes(_recorded(), "moves")["moved"] >= 1000


def test_the_record_has_an_example_of_every_shape():
    record = _recorded()
    for op in _OPS:
        assert set(record["examples"][op]) == set(_shapes(record, op))
        assert sum(entry[op]["proposals"] for entry in record["sheets"].values()) >= 3000


@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=suite_name)
def test_the_ops_answer_as_the_record_says(sheet: Path, tmp_path: Path):
    answered = answers_of(sheet, tmp_path)
    recorded = _recorded()["sheets"].get(suite_name(sheet))
    if answered is None:
        assert recorded is None, "the editor opened this sheet when the record was made"
        pytest.skip("the editor cannot open this sheet")
    entry, _examples = answered
    assert recorded is not None, (
        "a sheet with no record; run this file with LTSPICE_MCP_RECORD_LABELS_AND_MOVES=1"
    )
    for op in _OPS:
        # The shapes first: they say what kind of answer changed, where the
        # digest only says that one did.
        assert entry[op]["shapes"] == recorded[op]["shapes"], op
    assert entry == recorded
