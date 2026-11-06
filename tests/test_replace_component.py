"""edit_schematic's replace_component: a part's symbol swapped where it stands.

The part keeps its name, place, turn and attributes; each of its pins is paired
with a pin of the new symbol, by name or as the caller pairs them, and a pin
that moved is wired back to where its old wiring ends. The sheet's nets are
read through the same connectivity code every reader of nets uses
(``sheet_nets``), so "joined as before" is LTspice's own rule, not a re-count.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.schematic_ops import (
    OpReplaceComponent,
    apply_op_inplace,
    blank_sheet,
    sheet_nets,
)
from tests._schematic_fixtures import open_editor


def _write_sheet(path: Path, *lines: str) -> Path:
    """A blank sheet plus ``lines``, written byte-exact on every platform."""
    text = blank_sheet() + "".join(f"{line}\n" for line in lines)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _replace(sheet: Path, reference: str, symbol: str, **more: Any) -> dict[str, Any]:
    """Apply the op to the sheet and save it; a refusal raises before the save."""
    editor = open_editor(sheet)
    op = OpReplaceComponent(op="replace_component", reference=reference, symbol=symbol, **more)
    result = apply_op_inplace(editor, op, sheet)
    editor.save_netlist(sheet)
    return result


def _joined(sheet: Path, reference: str) -> dict[str, set[str]]:
    """Each pin of ``reference`` with the labels its net carries, read afresh
    from the saved sheet."""
    nets = sheet_nets(open_editor(sheet))
    return {
        pin.name: set(nets.texts(nets.of_pin(pin)))
        for pin in nets.pins
        if pin.reference == reference
    }


# A resistor (suite symbol: pin 1 at y-48, pin 2 at y+48) between net 'in' and
# ground, its value text placed by a WINDOW of its own.
_RESISTOR = (
    "WIRE 0 -96 0 -48",
    "WIRE 0 48 0 96",
    "FLAG 0 -96 in",
    "FLAG 0 96 0",
    "SYMBOL res 0 0 R0",
    "WINDOW 3 24 64 Left 2",
    "SYMATTR InstName R1",
    "SYMATTR Value 1k",
)


def test_a_pin_that_moved_is_wired_back_to_its_net(work_dir: Path, asc_symbols: Path):
    """The suite's cap has its pins 16 units nearer its middle than res: each
    is wired back to where the resistor's wiring ends, and the part keeps its
    name, place and value."""
    sheet = _write_sheet(work_dir / "rc.asc", *_RESISTOR)
    result = _replace(sheet, "R1", "cap")
    assert result["was"] == "res"
    assert sorted(
        (w["from"]["x"], w["from"]["y"], w["to"]["x"], w["to"]["y"]) for w in result["wires"]
    ) == [(0, -32, 0, -48), (0, 32, 0, 48)]
    assert _joined(sheet, "R1") == {"1": {"in"}, "2": {"0"}}
    text = sheet.read_text(encoding="utf-8")
    assert "SYMBOL cap 0 0 R0" in text
    assert "SYMATTR Value 1k" in text
    # The old symbol's text placement goes, so the new one places its own.
    assert "WINDOW 3 24 64" not in text


def test_a_symbol_whose_pins_land_where_the_old_ones_were_needs_no_wire(
    work_dir: Path, asc_symbols: Path
):
    sheet = _write_sheet(work_dir / "rl.asc", *_RESISTOR)
    result = _replace(sheet, "R1", "ind", value="10u")
    assert "wires" not in result and "left_at" not in result and "unpaired_pins" not in result
    assert _joined(sheet, "R1") == {"1": {"in"}, "2": {"0"}}
    assert "SYMATTR Value 10u" in sheet.read_text(encoding="utf-8")


def test_pins_of_other_names_are_paired_by_the_caller(work_dir: Path, asc_symbols: Path):
    """The diode's pins are A and K where the resistor's are 1 and 2. Unpaired,
    A would land on the wiring of pin 1 without being its replacement, so the
    op is refused and the sheet left as it was; paired, the swap goes through."""
    sheet = _write_sheet(work_dir / "rd.asc", *_RESISTOR)
    original = sheet.read_bytes()
    editor = open_editor(sheet)
    with pytest.raises(NetlistError) as refused:
        apply_op_inplace(
            editor,
            OpReplaceComponent(op="replace_component", reference="R1", symbol="diode"),
            sheet,
        )
    message = str(refused.value)
    assert "Pair the pins with pins (old pins: 1, 2; new pins: A, K)" in message
    assert "label 'in'" in message
    # Put back in the editor as it was, so a later op of the batch sees the resistor.
    assert editor.components["R1"].symbol == "res"
    assert "_WINDOW 3" in editor.components["R1"].attributes
    editor.save_netlist(sheet)
    assert sheet.read_text(encoding="utf-8").splitlines() == original.decode().splitlines()

    _replace(sheet, "R1", "diode", pins={"1": "A", "2": "K"})
    assert _joined(sheet, "R1") == {"A": {"in"}, "K": {"0"}}


def test_a_refusal_after_a_wire_was_drawn_takes_the_wire_back(work_dir: Path, asc_symbols: Path):
    """A label 'x' lies where the cap's pin 2 lands. Pin 1 is wired back first;
    then pin 2 would join 'x' as well as ground, so the op is refused, and the
    wire it drew for pin 1 goes with the rest of the swap."""
    sheet = _write_sheet(work_dir / "x.asc", *_RESISTOR, "FLAG 0 32 x")
    editor = open_editor(sheet)
    wires = [(w.V1.X, w.V1.Y, w.V2.X, w.V2.Y) for w in editor.wires]
    with pytest.raises(NetlistError) as refused:
        apply_op_inplace(
            editor,
            OpReplaceComponent(op="replace_component", reference="R1", symbol="cap"),
            sheet,
        )
    assert "label 'x'" in str(refused.value)
    assert [(w.V1.X, w.V1.Y, w.V2.X, w.V2.Y) for w in editor.wires] == wires
    assert editor.components["R1"].symbol == "res"


# An nmos4 (pins D 48,0; G 0,80; S 48,96; B 48,48) with each pin on a label.
_NMOS4 = (
    "WIRE 48 -32 48 0",
    "WIRE -32 80 0 80",
    "WIRE 48 96 48 128",
    "WIRE 48 48 80 48",
    "FLAG 48 -32 d",
    "FLAG -32 80 g",
    "FLAG 48 128 0",
    "FLAG 80 48 b",
    "SYMBOL nmos4 0 0 R0",
    "SYMATTR InstName M1",
    "SYMATTR Value NMOS",
)


def test_a_three_pin_symbol_takes_over_a_four_pin_one(work_dir: Path, asc_symbols: Path):
    """The suite's nmos puts D, G and S elsewhere, so each is wired back, round
    a corner where it must; it has no bulk, so B's wiring is left where it was,
    and the reply says where."""
    sheet = _write_sheet(work_dir / "m.asc", *_NMOS4)
    result = _replace(sheet, "M1", "nmos")
    assert result["left_at"] == [{"pin": "B", "x": 48, "y": 48}]
    assert "unpaired_pins" not in result
    assert _joined(sheet, "M1") == {"D": {"d"}, "G": {"g"}, "S": {"0"}}
    # The bulk's label keeps its own wire, joined to nothing of the part.
    nets = sheet_nets(open_editor(sheet))
    bulk = nets.net_named("b")
    assert bulk is not None
    assert not any(pin.reference == "M1" for pin in nets.nets.pins_on(bulk))


def test_a_new_pin_with_no_old_one_is_named(work_dir: Path, asc_symbols: Path):
    """The other way round, the bulk has no old pin: it is left unconnected,
    and the reply says so."""
    sheet = _write_sheet(
        work_dir / "m3.asc",
        "WIRE 0 -128 0 -96",
        "FLAG 0 -128 d",
        "SYMBOL nmos 0 0 R0",
        "SYMATTR InstName M1",
        "SYMATTR Value NMOS",
    )
    result = _replace(sheet, "M1", "nmos4")
    assert result["unpaired_pins"] == ["B"]
    assert _joined(sheet, "M1")["D"] == {"d"}


@pytest.mark.parametrize(
    ("change", "refused"),
    [
        ({"symbol": "nosuch"}, "Symbol 'nosuch' not found"),
        ({"reference": "R9"}, "R9"),
        ({"pins": {"7": "A"}}, "Pin '7' not found on R1 (res)"),
        ({"pins": {"1": "A", "2": "A"}}, "pairs each pin once"),
    ],
)
def test_what_cannot_be_paired_is_refused_before_anything_changes(
    work_dir: Path, asc_symbols: Path, change: dict[str, Any], refused: str
):
    sheet = _write_sheet(work_dir / "bad.asc", *_RESISTOR)
    editor = open_editor(sheet)
    arguments = {"reference": "R1", "symbol": "diode", **change}
    with pytest.raises(NetlistError, match=re.escape(refused)):
        apply_op_inplace(editor, OpReplaceComponent(op="replace_component", **arguments), sheet)
    assert editor.components["R1"].symbol == "res"
