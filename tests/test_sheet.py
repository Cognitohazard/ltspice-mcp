"""A sheet as a value: held to the scene and to the schematic editor on every sheet of the suite."""

from __future__ import annotations

import re
import typing
from collections import Counter
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.schematic_ops import make_editor, net_partition, placed_geometry
from ltspice_mcp.lib.schematic_scene import build_scene
from ltspice_mcp.lib.sheet import Sheet, editor_symbol_resolver, read_sheet
from ltspice_mcp.lib.symbol_library import SymbolResolver
from tests._proposal_record import stage
from tests._schematic_fixtures import BOX2_ASY, SUITE_SHEETS, suite_name, write_file

pytestmark = pytest.mark.usefixtures("suite_symbols_only")


def _staged(sheet: Path, tmp_path: Path) -> tuple[Path, Sheet]:
    staged = stage(sheet, tmp_path)
    return staged, read_sheet(staged.read_bytes(), editor_symbol_resolver(staged))


@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=suite_name)
def test_it_places_what_the_scene_places(sheet: Path, tmp_path: Path) -> None:
    staged, read = _staged(sheet, tmp_path)
    scene = build_scene(staged, editor_symbol_resolver(staged))
    assert len(read.parts) == len(scene.symbols)
    for part, drawn in zip(read.parts, scene.symbols, strict=True):
        assert (part.reference, part.symbol_name, part.at, part.rotation) == (
            drawn.reference,
            drawn.symbol,
            (drawn.x, drawn.y),
            drawn.rotation,
        )
        assert part.usable == (not drawn.missing)
        if part.usable:
            assert [(p.x, p.y, p.name) for p in part.pins] == [
                (p.x, p.y, p.name) for p in drawn.pins
            ]
            assert (part.box, part.body) == (drawn.box, drawn.body)
    assert read.wires == tuple((w.x1, w.y1, w.x2, w.y2) for w in scene.wires)
    assert read.labels == tuple((f.x, f.y, f.text) for f in scene.flags)


@pytest.mark.parametrize("sheet", SUITE_SHEETS, ids=suite_name)
def test_it_places_what_the_editor_places(sheet: Path, tmp_path: Path) -> None:
    staged, read = _staged(sheet, tmp_path)
    try:
        editor = make_editor(staged)
    except Exception:
        pytest.skip("the editor cannot open this sheet")
    assert isinstance(editor, AscEditor)
    named = Counter(part.reference for part in read.parts)
    once = [part for part in read.parts if part.reference and named[part.reference] == 1]
    for part in once:
        geometry = placed_geometry(editor, part.reference)
        assert (geometry is not None) == part.usable, part.reference
        if geometry is None or part.box is None:
            continue
        assert [
            {"name": p.name, "order": p.order, "x": p.x, "y": p.y, "dir": p.direction}
            for p in part.pins
        ] == geometry["pins"]
        assert part.box.to_origin_size_dict() == geometry["bounding_box"]
    if len(once) == len(read.parts):
        # With no reference repeated or missing, the nets are the editor's too.
        mine, theirs = read.partition(), net_partition(editor)
        assert {frozenset(m) for m in mine.members.values()} == {
            frozenset(m) for m in theirs.members.values()
        }
        assert {c: sorted(o) for c, o in mine.pin_owners.items()} == {
            c: sorted(o) for c, o in theirs.pin_owners.items()
        }
        assert mine.label_texts == theirs.label_texts


_SHEET = """Version 4
SHEET 1 880 680
WIRE 0 0 96 0
SYMBOL box2 96 0 R0
SYMATTR InstName U1
SYMATTR Value one
SYMATTR Value two
FLAG 0 0 in
SYMBOL nowhere 300 300 R0
SYMATTR InstName U2
SYMBOL broken 500 500 R0
SYMATTR InstName U3
"""

_BROKEN_ASY = """Version 4
SymbolType CELL
RECTANGLE Normal 0 0 32 16
PIN 0 8 NONE 0
PINATTR PinName A
PIN 32
"""


class TestAPart:
    def _read(self, tmp_path: Path) -> Sheet:
        write_file(tmp_path / "box2.asy", BOX2_ASY)
        write_file(tmp_path / "broken.asy", _BROKEN_ASY)
        return read_sheet(_SHEET.encode("ascii"), SymbolResolver(local_dir=tmp_path))

    def test_it_is_placed_with_its_pins_in_order_and_where_they_point(
        self, tmp_path: Path
    ) -> None:
        (u1,) = self._read(tmp_path).parts_named("U1")
        assert [(p.name, p.order, p.x, p.y, p.direction) for p in u1.pins] == [
            ("A", 1, 96, 8, "left"),
            ("B", 2, 128, 8, "right"),
        ]

    def test_a_repeated_attribute_reads_as_its_last(self, tmp_path: Path) -> None:
        (u1,) = self._read(tmp_path).parts_named("U1")
        assert u1.attribute("Value") == "two"
        assert u1.attribute("SpiceLine") is None

    def test_one_whose_symbol_is_not_found_has_no_pins(self, tmp_path: Path) -> None:
        (u2,) = self._read(tmp_path).parts_named("U2")
        assert (u2.symbol, u2.usable, u2.pins, u2.box) == (None, False, (), None)

    def test_one_whose_symbol_has_a_pin_line_that_does_not_read_is_unusable(
        self, tmp_path: Path
    ) -> None:
        """As it is to the schematic editor: a pin left out is a terminal no wire
        could be checked against."""
        (u3,) = self._read(tmp_path).parts_named("U3")
        assert u3.symbol is not None
        assert u3.symbol.problem == "has a pin line that does not read ('PIN 32')"
        assert (u3.usable, u3.pins, u3.box) == (False, (), None)

    def test_the_sheet_keeps_its_wires_and_labels(self, tmp_path: Path) -> None:
        read = self._read(tmp_path)
        assert read.wires == ((0, 0, 96, 0),)
        assert read.labels == ((0, 0, "in"),)
        assert read.partition().label_texts == {(0, 0): {"in"}}


class TestAPinByName:
    def _read(self, tmp_path: Path, body: str) -> Sheet:
        write_file(tmp_path / "box2.asy", BOX2_ASY)
        text = "Version 4\nSHEET 1 880 680\n" + body
        return read_sheet(text.encode("ascii"), SymbolResolver(local_dir=tmp_path))

    def test_by_name_then_by_spice_order(self, tmp_path: Path) -> None:
        read = self._read(tmp_path, "SYMBOL box2 96 0 R0\nSYMATTR InstName U1\n")
        assert read.pin_at("U1.a") == (96, 8)
        assert read.pin_at("U1.2") == (128, 8)

    def test_of_a_reference_two_parts_share_is_refused_naming_both(self, tmp_path: Path) -> None:
        twice = (
            "SYMBOL box2 96 0 R0\nSYMATTR InstName U1\nSYMBOL box2 300 0 R0\nSYMATTR InstName U1\n"
        )
        read = self._read(tmp_path, twice)
        assert len(read.parts_named("U1")) == 2
        with pytest.raises(
            NetlistError, match=r"'U1' names 2 parts, placed at \(96,0\), \(300,0\)"
        ):
            read.pin_at("U1.A")

    @pytest.mark.parametrize(
        ("spec", "said"),
        [
            ("U9.A", "Component 'U9' not found. Available: U1"),
            ("U1.C", "Pin 'C' not found on U1 (box2). Available: A (1), B (2)"),
            ("U1", "Invalid pin reference 'U1'"),
        ],
    )
    def test_that_is_not_there_is_refused(self, tmp_path: Path, spec: str, said: str) -> None:
        read = self._read(tmp_path, "SYMBOL box2 96 0 R0\nSYMATTR InstName U1\n")
        with pytest.raises(NetlistError, match=re.escape(said)):
            read.pin_at(spec)


class TestASymbolFileThatCannotBeRead:
    def test_is_found_and_unusable_and_says_why(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Found, so not reported as missing from every library."""
        from ltspice_mcp.lib import symbol_library

        write_file(tmp_path / "box2.asy", BOX2_ASY)

        def fails(_text: str) -> typing.NoReturn:
            raise ValueError("no such shape")

        monkeypatch.setattr(symbol_library, "read_symbol", fails)
        found = SymbolResolver(local_dir=tmp_path).load("box2")
        assert found is not None
        assert found.problem == "could not be read (no such shape)"
        read = read_sheet(
            b"Version 4\nSYMBOL box2 96 0 R0\nSYMATTR InstName U1\n",
            SymbolResolver(local_dir=tmp_path),
        )
        (u1,) = read.parts
        assert (u1.usable, u1.pins) == (False, ())
        with pytest.raises(
            NetlistError, match=r"symbol 'box2' could not be read \(no such shape\)"
        ):
            read.pin_at("U1.A")
