"""What the route planner answers to a broad set of proposals, held to a record of it.

``wire_pins`` draws a route only if the planner accepts it, and the planner's
refusals are a dozen checks made in one function in one order. Work that moves
those checks is held to this: on every sheet in the suite the editor opens,
a fixed set of proposals (pin to pin straight and by each corner, pin to the
middle of each wire, a detour through each other part, a waypoint on an end,
and an end given by the name of its net) is put to the planner, and what it
answers, refusal or route with its advisories, is as it was when the record
was made.

The record is ``fixtures/route_planner_record.json``. It keeps, for each sheet,
how many proposals there were, a digest of every answer in order, and how many
answers had each *shape*: the answer with its numbers and quoted names taken
out, so that one refusal worded for different pins is one shape. One whole
answer is kept for each shape as an example. A digest that differs says an
answer changed; the shapes and examples say what kind. To make the record
again, run this file with ``LTSPICE_MCP_RECORD_ROUTE_PLANNER=1``.

Like the record of sheet findings, it is the same on every machine: a symbol
is looked for beside its sheet and in the suite's stand-in library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import pytest
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib import symbol_geometry
from ltspice_mcp.lib.schematic_ops import (
    GridPoint,
    _plan_connect_route,  # pyright: ignore[reportPrivateUsage]  # the planner behind wire_pins
    collect_component_geometry,
    make_editor,
    wire_segments_of,
)
from tests._schematic_fixtures import SUITE_SHEETS, TESTS, suite_name

_RECORD = TESTS / "fixtures" / "route_planner_record.json"
_RECORDING = os.environ.get("LTSPICE_MCP_RECORD_ROUTE_PLANNER") == "1"

#: The most proposals put to one sheet between pins and coordinates, and the
#: most with an end given by net name. A sheet with more has every n-th taken,
#: which keeps the largest sheets from being most of the run.
_MOST = 600
_MOST_BY_NAME = 150

Proposal = tuple[Any, Any, list[GridPoint]]


def _proposals(editor: AscEditor) -> list[Proposal]:
    """The routes asked of a sheet, in an order that depends on the sheet alone."""
    parts = collect_component_geometry(editor)
    pins = [
        (f"{part['ref']}.{pin['name']}", pin["x"], pin["y"])
        for part in parts
        for pin in part["pins"]
    ]
    wires = wire_segments_of(editor)
    asked: list[Proposal] = []
    for name_a, ax, ay in pins:
        for name_b, bx, by in pins:
            if name_a == name_b:
                continue
            asked.append((name_a, name_b, []))
            if ax != bx and ay != by:
                asked.append((name_a, name_b, [GridPoint(x=ax, y=by)]))
                asked.append((name_a, name_b, [GridPoint(x=bx, y=ay)]))
            else:
                # Out to one side and back: a route with corners where the two
                # pins are in line, and one that ends on a waypoint at its end.
                asked.append(
                    (name_a, name_b, [GridPoint(x=ax + 64, y=ay), GridPoint(x=bx + 64, y=by)])
                )
                asked.append((name_a, name_b, [GridPoint(x=bx, y=by)]))
        for x1, y1, x2, y2 in wires:
            middle = GridPoint(x=(x1 + x2) // 2 // 16 * 16, y=(y1 + y2) // 2 // 16 * 16)
            asked.append((name_a, middle, []))
        for part in parts:
            # Level with the middle of each part, across to the far side of it.
            across = part["y"] + part["height"] // 2 // 16 * 16
            beyond = part["x"] + part["width"] + 32
            asked.append((name_a, GridPoint(x=beyond, y=across), [GridPoint(x=ax, y=across)]))
    if len(asked) > _MOST:
        asked = asked[:: -(-len(asked) // _MOST)]
    # An end given by the name of its net, for each name the sheet labels: to
    # each pin straight and round a corner, and from each pin round the other.
    by_name: list[Proposal] = []
    names: set[str] = set()
    for label in editor.labels:
        if label.text in names:
            continue
        names.add(label.text)
        lx, ly = int(label.coord.X), int(label.coord.Y)
        for pin, px, py in pins:
            by_name.append((f"net:{label.text}", pin, []))
            by_name.append((f"net:{label.text}", pin, [GridPoint(x=lx, y=py)]))
            by_name.append((pin, f"net:{label.text}", [GridPoint(x=px, y=ly)]))
    if len(by_name) > _MOST_BY_NAME:
        by_name = by_name[:: -(-len(by_name) // _MOST_BY_NAME)]
    return asked + by_name


def _answer(editor: AscEditor, proposal: Proposal) -> dict[str, Any]:
    from_pin, to_pin, waypoints = proposal
    try:
        plan = _plan_connect_route(editor, from_pin, to_pin, waypoints)
    except NetlistError as refused:
        return {"refused": str(refused)}
    return {
        "segments": [list(segment) for segment in plan.segments],
        "warnings": list(plan.warnings),
        "junctions": list(plan.junctions),
    }


def _shape(answer: dict[str, Any]) -> str:
    """An answer with what varies from pin to pin taken out of it."""
    if "refused" in answer:
        text = "refused: " + answer["refused"]
    else:
        text = "routed: " + " | ".join(answer["warnings"]) if answer["warnings"] else "routed"
    text = re.sub(r"net:[^\s:,]+", "net:NAME", text)
    text = re.sub(r"'[^']*'", "'…'", text)
    text = re.sub(r"\b[\w+\-]+\.[\w+\-]+", "PIN", text)
    return re.sub(r"-?\d+", "#", text)


def _stage(sheet: Path, sandbox: Path) -> Path:
    for symbol in sheet.parent.rglob("*.asy"):
        target = sandbox / symbol.relative_to(sheet.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(symbol, target)
    return Path(shutil.copyfile(sheet, sandbox / sheet.name))


def answers_of(
    sheet: Path, sandbox: Path
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]] | None:
    """What the record holds of ``sheet`` and an example of each shape, or
    ``None`` for a sheet the editor cannot open."""
    AscEditor.symbol_cache = {}
    symbol_geometry._symbol_cache.clear()  # pyright: ignore[reportPrivateUsage]
    try:
        editor = make_editor(_stage(sheet, sandbox))
    except Exception:
        return None
    assert isinstance(editor, AscEditor)
    proposals = _proposals(editor)
    answers = [_answer(editor, proposal) for proposal in proposals]
    shapes: dict[str, int] = {}
    examples: dict[str, dict[str, Any]] = {}
    for (from_pin, to_pin, waypoints), answer in zip(proposals, answers, strict=True):
        shape = _shape(answer)
        shapes[shape] = shapes.get(shape, 0) + 1
        examples.setdefault(
            shape,
            {
                "from": from_pin if isinstance(from_pin, str) else from_pin.model_dump(),
                "to": to_pin if isinstance(to_pin, str) else to_pin.model_dump(),
                "waypoints": [point.model_dump() for point in waypoints],
                **answer,
            },
        )
    digest = hashlib.sha256(json.dumps(answers, sort_keys=True).encode("utf-8")).hexdigest()
    entry = {"proposals": len(proposals), "sha256": digest, "shapes": dict(sorted(shapes.items()))}
    return entry, examples


@pytest.fixture(autouse=True)
def _only_the_suites_own_symbols(asc_symbols: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No LTspice library the machine has is searched, and spicelib's symbol
    cache, which ``answers_of`` empties, is put back afterwards."""
    monkeypatch.setattr(AscEditor, "simulator_lib_paths", [])
    monkeypatch.setattr(AscEditor, "symbol_cache", dict(AscEditor.symbol_cache))


def _recorded() -> dict[str, Any]:
    return json.loads(_RECORD.read_text(encoding="utf-8"))


@pytest.mark.skipif(not _RECORDING, reason="set LTSPICE_MCP_RECORD_ROUTE_PLANNER=1 to record")
def test_record_the_planners_answers(tmp_path: Path):
    sheets: dict[str, Any] = {}
    examples: dict[str, dict[str, Any]] = {}
    for number, sheet in enumerate(SUITE_SHEETS):
        sandbox = tmp_path / f"sheet{number}"
        sandbox.mkdir()
        answered = answers_of(sheet, sandbox)
        if answered is None:
            continue
        entry, seen = answered
        sheets[suite_name(sheet)] = entry
        for shape, example in seen.items():
            examples.setdefault(shape, {"sheet": suite_name(sheet), **example})
    record = {"sheets": sheets, "examples": dict(sorted(examples.items()))}
    _RECORD.write_text(
        json.dumps(record, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_the_record_asks_enough_to_hold_every_check():
    """Each refusal and each advisory the planner has is in it many times over."""
    record = _recorded()
    shapes: dict[str, int] = {}
    for entry in record["sheets"].values():
        for shape, count in entry["shapes"].items():
            shapes[shape] = shapes.get(shape, 0) + count
    assert sum(entry["proposals"] for entry in record["sheets"].values()) >= 5000
    for said in (
        "resolve to the same coordinate",
        "zero length after deduplicating",
        "Connecting them would short",
        "net:NAME is on net",
        "Multiple '…' labels found",
        "would merge named nets",
        "same-instance wire",
        "Diagonal wire",
        "will create unintended connection",
        "will create unintended junction",
        "runs along the wire",
        "Route touches",
        "already wired to this net",
        "crosses the wire",
        "The route touches",
        "Long wire run",
        "bounding box",
    ):
        seen = sum(count for shape, count in shapes.items() if said in shape)
        assert seen >= 3, f"{said!r} is in {seen} recorded answers"
    assert set(record["examples"]) == set(shapes)


@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=suite_name)
def test_the_planner_answers_as_the_record_says(sheet: Path, tmp_path: Path):
    answered = answers_of(sheet, tmp_path)
    recorded = _recorded()["sheets"].get(suite_name(sheet))
    if answered is None:
        assert recorded is None, "the editor opened this sheet when the record was made"
        pytest.skip("the editor cannot open this sheet")
    entry, _examples = answered
    assert recorded is not None, (
        "a sheet with no record; run this file with LTSPICE_MCP_RECORD_ROUTE_PLANNER=1"
    )
    # The shapes first: they say what kind of answer changed, where the digest
    # only says that one did.
    assert entry["shapes"] == recorded["shapes"]
    assert entry == recorded
