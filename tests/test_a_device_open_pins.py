"""The pins of an A-device that nothing is on.

LTspice writes such a pin as node 0, exactly as it writes a pin on ground
(``export/adevice_open_pin`` and the ``a-device-open-pin-forms`` recordings),
while the sheet's wiring leaves it on a net of its own. ``Nets.open_node``
says which node a netlister writes for an open pin, and every reader that
builds or compares a netlist, or tells a caller where a pin is, reads it there.
"""

from __future__ import annotations

import asyncio
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib.connectivity import (
    NUMBERED_OPEN_PIN_LETTERS,
    open_node_by_ltspice,
    signature,
)
from ltspice_mcp.lib.export_nets import exported_pin_nodes
from ltspice_mcp.lib.netlist_graph import canon_ref
from ltspice_mcp.lib.nets import OpenNode, net_change
from ltspice_mcp.lib.schematic_ops import (
    collect_component_geometry,
    element_class,
    export_parts,
    sheet_nets,
)
from ltspice_mcp.lib.schematic_scene import build_scene, sheet_view
from ltspice_mcp.lib.sheet_findings import findings
from ltspice_mcp.lib.simulator import symbol_library_for
from ltspice_mcp.lib.symbol_library import SymbolResolver
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests._asc_ops import apply_ops, inspect_one, load_editor, sheet_findings
from tests._schematic_fixtures import open_editor
from tests.conftest import SyncApi
from tests.ltspice_recorder import INPUTS
from tests.test_recorded_ltspice_schematics import _editor, exported_groups, netlisted_groups
from tests.test_verify_circuit import _run as run_verify_circuit

A8 = "SYMBOL a8 96 64 R0\nSYMATTR InstName A1\n"
#: ``export/adevice_open_pin``: pins 1, 2, 7 labelled a, b, y; pin 5 on ground.
OPEN_PIN = "FLAG 64 64 a\nFLAG 64 80 b\nFLAG 128 80 y\nFLAG 64 128 0\n" + A8
#: Where a8 at (96,64) places its pins, by name.
A8_PINS = {
    "1": (64, 64),
    "2": (64, 80),
    "3": (64, 96),
    "4": (64, 112),
    "5": (64, 128),
    "6": (128, 64),
    "7": (128, 80),
    "8": (128, 96),
}
DETAIL = (
    "A1 has 4 pins with nothing on them (3, 4, 6, 8 at (64,96), (64,112), (128,64), "
    "(128,96)); LTspice writes each as node 0, exactly as it writes a pin on ground, so "
    "they simulate as pins on ground. Wire or label a pin to give it another node."
)


def _sheet(directory: Path, body: str, name: str = "open.asc") -> Path:
    """A sheet of ``body`` beside a8 and the stock symbols the recordings name."""
    rec.stage_sheet("ltspice26", "export/adevice_prefix_by_attribute", directory)
    shutil.copyfile(INPUTS / "export/a8.asy", directory / "a8.asy")
    sheet = directory / name
    sheet.write_text(f"Version 4\nSHEET 1 880 680\n{body}", encoding="utf-8", newline="\n")
    return sheet


def _open_pins(found: list[Any]) -> dict[str, tuple[str, ...]]:
    return {one.refs[0]: one.facts["pins"] for one in found if one.rule == "a_device_open_pins"}


def _both_tools(sheet: Path) -> list[list[Any]]:
    """What ``edit_schematic`` and ``verify_circuit`` find of ``sheet``."""
    scene = build_scene(sheet, SymbolResolver(local_dir=sheet.parent))
    return [sheet_findings(_editor(sheet)), findings(sheet_view(scene))]


def _groups(sheet: Path) -> set[frozenset[str]]:
    return netlisted_groups(sheet)


# --------------------------------------------------------------------------
# The decider
# --------------------------------------------------------------------------


def test_an_a_device_pin_on_nothing_is_written_as_node_0_and_others_as_their_own():
    assert open_node_by_ltspice("A") == open_node_by_ltspice("a") == OpenNode(fixed="0")
    assert open_node_by_ltspice("R") == OpenNode(numbered="NC_")
    unknown = open_node_by_ltspice("N").unknown
    assert unknown is not None and "N" in unknown


@pytest.mark.parametrize("case_id", ["export/open_pin_letters", "export/open_pin_controlled"])
@pytest.mark.parametrize("build", rec.BUILDS)
def test_every_other_recorded_letter_numbers_its_open_pins(build: str, case_id: str):
    """D, Q, M, J, V, I, L, B, S and E, F, G, H: each open pin is a node of its
    own, which the decider calls numbered; Q1 is written with a fourth node,
    its substrate on node 0, which its symbol has no pin for."""
    letters = set()
    for name, card in rec.export_instances(build, case_id).items():
        letters.add(name[0].upper())
        drawn = card.nodes[:3] if name == "q1" else card.nodes
        assert all(re.fullmatch(r"NC_\d{2}", node) for node in drawn), (name, card.nodes)
        assert open_node_by_ltspice(name[0]).numbered == "NC_"
    expected = set("DQMJVILBS") if case_id.endswith("letters") else set("EFGH")
    assert letters == expected
    assert letters <= NUMBERED_OPEN_PIN_LETTERS
    if case_id.endswith("letters"):
        assert rec.export_instances(build, case_id)["q1"].nodes[3] == "0"


def test_a_bipolar_transistor_is_compared_over_the_three_pins_its_symbol_draws():
    pins = [((0, 0), ("q1", "C"), "Q"), ((0, 32), ("q1", "B"), "Q"), ((0, 64), ("q1", "E"), "Q")]
    drawn = signature(pins, [], [])
    assert {on_net for on_net, _names in drawn.groups} == {
        frozenset({("q1", "C")}),
        frozenset({("q1", "B")}),
        frozenset({("q1", "E")}),
    }


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_stock_and_gate_groups_as_ltspice_exported_it(build: str):
    """The build's own AND gate, whose symbol is not committed: eight pins of
    an A-device, pin 1 labelled a and pin 7 labelled y, the rest open."""
    points = [(64, 96), (64, 112), (64, 128), (64, 144), (64, 160), (128, 96), (128, 112), (128, 128)]
    pins = [(point, ("a1", str(order)), "A") for order, point in enumerate(points, start=1)]
    drawn = signature(pins, [((64, 96), "a"), ((128, 112), "y")], [])
    groups = {frozenset(f"{ref}.{name}" for ref, name in on_net) for on_net, _ in drawn.groups}
    assert groups == {
        frozenset({"a1.1"}),
        frozenset({"a1.2", "a1.3", "a1.4", "a1.5", "a1.6", "a1.8"}),
        frozenset({"a1.7"}),
    }
    assert groups == exported_groups(build, "export/adevice_open_pin_stock_and")


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_stock_and_gate_against_the_build(build: str, tmp_path: Path):
    """The same sheet, its AND gate read from the installed build's library."""
    installed = rec.installed_counterpart(build)
    if isinstance(installed, str):
        pytest.skip(installed)
    from spicelib.simulators.ltspice_simulator import LTspice

    launches = type(LTspice.__name__, (LTspice,), {"spice_exe": [str(installed.exe)]})
    library = symbol_library_for([], launches)
    case = "export/adevice_open_pin_stock_and"
    sheet = tmp_path / "adevice_open_pin_stock_and.asc"
    shutil.copyfile(INPUTS / f"{case}.asc", sheet)
    editor = open_editor(sheet, library.for_sheet(sheet))
    pins = [
        ((pin["x"], pin["y"]), (canon_ref(row["ref"]), str(pin["order"])), "A")
        for row in collect_component_geometry(editor)
        for pin in row["pins"]
    ]
    labels = [((int(lbl.coord.X), int(lbl.coord.Y)), lbl.text) for lbl in editor.labels]
    drawn = signature(pins, labels, [])
    groups = {frozenset(f"{ref}.{name}" for ref, name in on_net) for on_net, _ in drawn.groups}
    assert groups == exported_groups(build, case)


def test_open_nodes_by_pin(tmp_path: Path):
    nets = sheet_nets(_editor(_sheet(tmp_path, OPEN_PIN))).nets
    by_name = {pin.name: pin for pin in nets.pins}
    assert nets.unconnected(by_name["3"])
    assert {name: nets.open_node(pin) for name, pin in by_name.items()} == {
        "1": None,
        "2": None,
        "3": OpenNode(fixed="0"),
        "4": OpenNode(fixed="0"),
        "5": None,
        "6": OpenNode(fixed="0"),
        "7": None,
        "8": OpenNode(fixed="0"),
    }


def test_a_prefix_no_recording_establishes_leaves_the_pins_uncompared(tmp_path: Path):
    """cell4 given Prefix N: no recording says what LTspice writes for its
    open pins, so each is floating with the reason, and the signature leaves
    all four out of its groups."""
    shutil.copyfile(INPUTS / "export/cell4.asy", tmp_path / "cell4.asy")
    sheet = _sheet(
        tmp_path, "SYMBOL cell4 320 64 R0\nSYMATTR InstName N1\nSYMATTR Prefix N\n"
    )
    nets = sheet_nets(_editor(sheet)).nets
    assert all(nets.open_node(pin).unknown for pin in nets.pins)  # type: ignore[union-attr]
    for found in _both_tools(sheet):
        floating = [one for one in found if one.rule == "floating_pin"]
        assert len(floating) == 4
        assert all("; " in one.detail for one in floating)
        assert _open_pins(found) == {}
    editor = _editor(sheet)
    pins = [
        ((pin["x"], pin["y"]), (row["ref"], pin["name"]), element_class(editor, row["ref"]))
        for row in collect_component_geometry(editor)
        for pin in row["pins"]
    ]
    drawn = signature(pins, [], [])
    assert drawn.groups == frozenset()
    assert {(ref, name) for ref, name, _why in drawn.uncompared} == {
        ("N1", name) for name in ("a", "b", "c", "d")
    }


# --------------------------------------------------------------------------
# Sheets
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("extra", "open_pins"),
    [
        ("", ("3", "4", "6", "8")),
        ("FLAG 64 96 p3\nFLAG 64 112 p4\nFLAG 128 64 p6\nFLAG 128 96 p8\n", None),
        ("FLAG 64 96 c\n", ("4", "6", "8")),
        ("FLAG 64 96 0\n", ("4", "6", "8")),
    ],
    ids=["open", "all-wired", "labelled", "labelled-ground"],
)
def test_one_finding_names_the_open_pins(tmp_path: Path, extra: str, open_pins: Any):
    sheet = _sheet(tmp_path, OPEN_PIN + extra)
    for found in _both_tools(sheet):
        assert [one for one in found if one.rule == "floating_pin"] == []
        assert _open_pins(found) == ({"A1": open_pins} if open_pins else {})
    if not extra:
        assert _groups(sheet) == {
            frozenset({"a1.1"}),
            frozenset({"a1.2"}),
            frozenset({"a1.3", "a1.4", "a1.5", "a1.6", "a1.8"}),
            frozenset({"a1.7"}),
        }


def test_a_loose_wire_gives_the_pin_a_node_of_its_own(tmp_path: Path):
    sheet = _sheet(tmp_path, "WIRE 0 96 64 96\nFLAG 64 64 a\nFLAG 128 80 y\n" + A8)
    nets = sheet_nets(_editor(sheet)).nets
    assert nets.open_node(next(pin for pin in nets.pins if pin.name == "3")) is None
    for found in _both_tools(sheet):
        assert [one.points for one in found if one.rule == "dangling_wire_end"] == [((0, 96),)]
        assert _open_pins(found) == {"A1": ("2", "4", "5", "6", "8")}
    assert _groups(sheet) == {
        frozenset({"a1.1"}),
        frozenset({"a1.3"}),
        frozenset({"a1.2", "a1.4", "a1.5", "a1.6", "a1.8"}),
        frozenset({"a1.7"}),
    }


def test_two_a_devices_open_pins_are_one_node_but_not_one_wiring_net(tmp_path: Path):
    body = (
        "FLAG 64 64 a1\nFLAG 64 80 b1\nFLAG 128 80 y1\n"
        "FLAG 64 256 a2\nFLAG 64 272 b2\nFLAG 128 272 y2\n"
        + A8
        + "SYMBOL a8 96 256 R0\nSYMATTR InstName A2\n"
    )
    sheet = _sheet(tmp_path, body)
    for found in _both_tools(sheet):
        assert _open_pins(found) == {"A1": tuple("34568"), "A2": tuple("34568")}
    nets = sheet_nets(_editor(sheet)).nets
    third = [pin for pin in nets.pins if pin.name == "3"]
    assert nets.of_pin(third[0]) != nets.of_pin(third[1])
    assert net_change(nets, nets) == net_change(nets, nets, folded=False)
    assert not net_change(nets, nets).split and not net_change(nets, nets).joined
    open_group = frozenset(f"a{part}.{pin}" for part in (1, 2) for pin in "34568")
    assert open_group in _groups(sheet)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_open_pins_of_an_a_device_inside_a_block_are_the_subcircuits(
    build: str, tmp_path: Path
):
    top = rec.stage_sheet(build, "export/adevice_block_top", tmp_path)
    block = top.parent / "adevice_block.asc"
    assert _open_pins(sheet_findings(_editor(block))) == {"A1": tuple("234568")}
    assert _open_pins(sheet_findings(_editor(top))) == {}
    lines = rec.export_text(build, "export/adevice_block_top").splitlines()
    card = lines[lines.index(".subckt adevice_block IN OUT") + 1]
    nodes = card.split()[1:9]
    by_node: dict[str, set[str]] = {}
    for order, node in enumerate(nodes, start=1):
        by_node.setdefault(node, set()).add(f"a1.{order}")
    assert _groups(block) == {frozenset(pins) for pins in by_node.values()}


def test_stacked_pins_and_a_dropped_wire_leave_every_pin_open(tmp_path: Path):
    shutil.copyfile(INPUTS / "export/a8_stacked.asy", tmp_path / "a8_stacked.asy")
    body = (
        "WIRE 320 96 320 112\nSYMBOL a8_stacked 96 64 R0\nSYMATTR InstName A1\n"
        "SYMBOL a8 352 64 R0\nSYMATTR InstName A2\n"
    )
    sheet = _sheet(tmp_path, body)
    for found in _both_tools(sheet):
        assert _open_pins(found) == {"A1": tuple("12345678"), "A2": tuple("12345678")}
        assert [one.refs for one in found if one.rule == "dropped_wire"] == [("A2",)]
    assert _groups(sheet) == {frozenset(f"a{part}.{pin}" for part in (1, 2) for pin in "12345678")}


# --------------------------------------------------------------------------
# Edits
# --------------------------------------------------------------------------


async def test_wiring_an_open_pin_to_another_net_neither_shorts_nor_grounds_it(
    state_no_sim: SessionState, work_dir: Path
):
    body = OPEN_PIN + "SYMBOL res -96 80 R0\nSYMATTR InstName R1\n"
    sheet = _sheet(work_dir, body)
    label = await apply_ops(
        state_no_sim, sheet.name, [{"op": "add_net_label", "net": "in", "pin": "R1.A"}]
    )
    assert label["outcome"] == "complete", label
    before = sheet_nets(load_editor(state_no_sim, sheet)).nets
    data = await apply_ops(
        state_no_sim, sheet.name, [{"op": "wire_pins", "from_pin": "A1.3", "to_pin": "R1.A"}]
    )
    assert data["outcome"] == "complete", data
    assert [w for w in data.get("warnings", []) if w.startswith("A1 has 3 pins")], data
    assert not any("short" in w or "ground" in w for w in data.get("warnings", [])
                   if not w.startswith("A1 has")), data["warnings"]
    found = sheet_findings(load_editor(state_no_sim, sheet))
    assert _open_pins(found) == {"A1": ("4", "6", "8")}
    nets = sheet_nets(load_editor(state_no_sim, sheet))
    third = next(pin for pin in nets.pins if pin.reference == "A1" and pin.name == "3")
    assert nets.texts(nets.of_pin(third)) == frozenset({"in"})
    assert not net_change(before, nets.nets).split


async def test_labelling_the_other_open_pins_leaves_no_finding(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = _sheet(work_dir, OPEN_PIN + "FLAG 64 96 c\n")
    ops = [{"op": "add_net_label", "net": f"p{pin}", "pin": f"A1.{pin}"} for pin in "468"]
    data = await apply_ops(state_no_sim, sheet.name, ops)
    assert data["outcome"] == "complete", data
    assert data["rules_run"]["a_device_open_pins"] == 0
    verified = await run_verify_circuit(state_no_sim, path=str(sheet), checks=["layout"])
    assert [f for f in verified.get("findings", []) if f["rule_id"] == "a_device_open_pins"] == []
    assert verified["rules_run"]["a_device_open_pins"] == 0


async def test_removing_the_wire_of_a_pin_splits_its_net_and_says_nothing_of_ground(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = _sheet(work_dir, "WIRE 0 96 64 96\nFLAG 0 96 c\n" + OPEN_PIN)
    before = sheet_nets(load_editor(state_no_sim, sheet)).nets
    data = await apply_ops(
        state_no_sim, sheet.name, [{"op": "remove_wire", "x1": 0, "y1": 96, "x2": 64, "y2": 96}]
    )
    assert data["outcome"] == "complete", data
    after = sheet_nets(load_editor(state_no_sim, sheet)).nets
    assert net_change(before, after).split
    assert [w for w in data.get("warnings", []) if w.startswith("A1 has 4 pins")] == [DETAIL]
    words = str({key: value for key, value in data.items() if key != "warnings"}).lower()
    assert "ground" not in words and "node 0" not in words


async def test_labelling_an_open_pin_ground_puts_it_on_ground(
    state_no_sim: SessionState, work_dir: Path
):
    sheet = _sheet(work_dir, OPEN_PIN)
    data = await apply_ops(
        state_no_sim, sheet.name, [{"op": "add_net_label", "net": "0", "pin": "A1.3"}]
    )
    assert data["outcome"] == "complete", data
    nets = sheet_nets(load_editor(state_no_sim, sheet))
    third = next(pin for pin in nets.pins if pin.name == "3")
    assert "0" in nets.texts(nets.of_pin(third))
    assert _open_pins(sheet_findings(load_editor(state_no_sim, sheet))) == {"A1": ("4", "6", "8")}


async def test_replacing_a_resistor_with_an_a_device_bridges_nothing(
    state_no_sim: SessionState, work_dir: Path
):
    """The open pins are one node in the netlist but not on the sheet, so the
    replacement wires each moved pin back to its own net, not to ground."""
    sheet = _sheet(
        work_dir,
        "FLAG 96 0 n1\nFLAG 96 112 n2\nSYMBOL res 80 0 R0\nSYMATTR InstName R1\n",
    )
    before = sheet_nets(load_editor(state_no_sim, sheet)).nets
    data = await apply_ops(
        state_no_sim, sheet.name, [{"op": "replace_component", "reference": "R1", "symbol": "a8", "pins": {"A": "1", "B": "2"}}]
    )
    assert data["outcome"] == "complete", data
    after = sheet_nets(load_editor(state_no_sim, sheet)).nets
    assert not net_change(before, after).joined


# --------------------------------------------------------------------------
# Every reader names the same pins
# --------------------------------------------------------------------------

NAMED = ["A1.3", "A1.4", "A1.6", "A1.8"]


def _named_by(coords: list[list[int]]) -> list[str]:
    at = {point: name for name, point in A8_PINS.items()}
    return [f"A1.{at[x, y]}" for x, y in coords]


async def _verify_rows(state: SessionState, sheet: Path) -> list[str]:
    data = await run_verify_circuit(state, path=str(sheet), checks=["layout"])
    rows = [f for f in data["findings"] if f["rule_id"] == "a_device_open_pins"]
    assert len(rows) == 1, rows
    assert rows[0]["severity"] == "observation"
    assert (rows[0]["at"]["x"], rows[0]["at"]["y"]) == (64, 96)
    assert rows[0]["subject"] == "A1"
    assert rows[0]["evidence"]["detail"] == DETAIL
    assert rows[0]["evidence"]["refs"] == ["A1"]
    assert rows[0]["evidence"]["coords"] == [[64, 96], [64, 112], [128, 64], [128, 96]]
    assert data["rules_run"]["a_device_open_pins"] == 1
    return _named_by(rows[0]["evidence"]["coords"])


async def _edit_rows(state: SessionState, sheet: Path) -> list[str]:
    data = await apply_ops(state, sheet.name, [], dry_run=True, return_views=["pin_legend", "preexisting"])
    legend = data["views"]["pin_legend"]["items"][0]["pins"]
    named = [f"A1.{pin['name']}" for pin in legend if pin.get("open_node") == "0"]
    rows = [r for r in data["views"]["preexisting"]["items"] if r.get("kind") == "a_device_open_pins"]
    assert [row["message"] for row in rows] == [DETAIL], rows
    return named


async def _inspect_pins(state: SessionState, sheet: Path) -> list[str]:
    named = []
    for name in A8_PINS:
        data = await inspect_one(state, {"kind": "net", "path": str(sheet), "at": f"A1.{name}"})
        rows = [row for row in data["pins"] if row.get("open_node") == "0"]
        if rows:
            assert data["hint"] == (
                f"A1.{name} has nothing on it on the sheet; LTspice writes it as node 0, "
                "exactly as it writes a pin on ground, so it simulates as a pin on ground."
            )
            named.append(f"A1.{name}")
    return named


async def _inspect_ground(state: SessionState, sheet: Path) -> list[str]:
    data = await inspect_one(state, {"kind": "net", "path": str(sheet), "at": "0"})
    assert data["hint"] == (
        "LTspice also writes node 0 for 4 A-device pins with nothing on them "
        "(A1.3, A1.4, A1.6, A1.8), exactly as it writes a pin on ground; the sheet "
        "draws no connection to them."
    )
    return re.findall(r"A1\.\d", data["hint"])


async def _python_api(state: SessionState, sheet: Path) -> list[str]:
    data = await asyncio.to_thread(
        lambda: SyncApi(state).verify_circuit(path=str(sheet), checks=["layout"])
    )
    rows = [f for f in data["findings"] if f["rule_id"] == "a_device_open_pins"]
    assert [row["evidence"]["detail"] for row in rows] == [DETAIL]
    return _named_by(rows[0]["evidence"]["coords"])


@pytest.mark.parametrize(
    "reader", [_verify_rows, _edit_rows, _inspect_pins, _inspect_ground, _python_api]
)
async def test_every_reader_names_the_four_open_pins(
    reader: Any, state_no_sim: SessionState, work_dir: Path
):
    sheet = _sheet(work_dir, OPEN_PIN)
    assert await reader(state_no_sim, sheet) == NAMED


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_the_legend_and_the_export_agree_on_node_0(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """The legend's ``open_node`` says without an export what ``net_export``
    reads from one (``_name_nets_as_exported``), here from LTspice's own."""
    sheet = rec.stage_sheet(build, "export/adevice_open_pin", work_dir)
    data = await apply_ops(state_no_sim, sheet.name, [], dry_run=True, return_views=["pin_legend"])
    editor = _editor(sheet)
    nodes = exported_pin_nodes(
        rec.export_text(build, "export/adevice_open_pin"),
        export_parts(editor, collect_component_geometry(editor)),
    )
    said = {
        pin["name"]: pin["open_node"]
        for pin in data["views"]["pin_legend"]["items"][0]["pins"]
        if "open_node" in pin
    }
    assert said == {name: nodes["A1", name] for name in ("3", "4", "6", "8")} == dict.fromkeys(
        ("3", "4", "6", "8"), "0"
    )


# --------------------------------------------------------------------------
# What LTspice simulates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("build", rec.BUILDS)
def test_an_open_input_simulates_as_a_grounded_one(build: str, tmp_path: Path):
    """The two runs' decks differ only in their title line, and the gate's
    output is the same high level on both."""
    decks = [
        rec.export_text(build, f"export/adevice_and_{form}_input").splitlines()[1:]
        for form in ("open", "grounded")
    ]
    assert decks[0] == decks[1]
    for form in ("open", "grounded"):
        point = rec.operating_point(build, f"export/adevice_and_{form}_input", tmp_path / form)
        assert point["v(y)"] == pytest.approx(0.999001, abs=1e-6)
