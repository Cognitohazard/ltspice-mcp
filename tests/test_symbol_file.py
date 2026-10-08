"""The symbol file reader: one reading of a ``.asy`` behind both the schematic
editor's pin geometry and the renderer's symbol body.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.lib.asc_document import ROTATIONS, Window
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_ops import make_editor
from ltspice_mcp.lib.schematic_ops import sheet_view as editor_view
from ltspice_mcp.lib.schematic_scene import SymbolResolver, build_scene, parse_symbol
from ltspice_mcp.lib.schematic_scene import sheet_view as scene_view
from ltspice_mcp.lib.symbol_file import PinInfo, SymbolArc, read_symbol
from ltspice_mcp.lib.symbol_geometry import parse_asy_file
from tests._schematic_fixtures import SUITE_SYMBOLS, suite_name

NMOS = (
    "Version 4\n"
    "SymbolType CELL\n"
    "LINE Normal 48 80 48 96\n"
    "LINE Normal 16 80 48 80\n"
    "RECTANGLE Normal 8 8 16 88\n"
    "CIRCLE Normal 0 40 8 48\n"
    "ARC Normal 16 16 48 48 48 32 16 32\n"
    "WINDOW 0 56 32 Left 2\n"
    "WINDOW 3 56 72 VTop 1\n"
    "SYMATTR Value NMOS\n"
    "SYMATTR Prefix MN\n"
    "SYMATTR Description N-Channel MOSFET transistor\n"
    "PIN 48 96 NONE 0\n"
    "PINATTR PinName S\n"
    "PINATTR SpiceOrder 3\n"
    "PIN 48 0 NONE 0\n"
    "PINATTR PinName D\n"
    "PINATTR SpiceOrder 1\n"
    "PIN 0 80 NONE 0\n"
    "PINATTR PinName G\n"
    "PINATTR SpiceOrder 2\n"
)


class TestWhatIsRead:
    def test_pins_come_in_spice_order(self) -> None:
        assert read_symbol(NMOS).pins == (
            PinInfo("D", 1, 48, 0),
            PinInfo("G", 2, 0, 80),
            PinInfo("S", 3, 48, 96),
        )

    def test_the_body_reads_shape_by_shape(self) -> None:
        symbol = read_symbol(NMOS)
        assert symbol.lines == ((48, 80, 48, 96), (16, 80, 48, 80))
        assert symbol.rects == ((8, 8, 16, 88),)
        assert symbol.circles == ((0, 40, 8, 48),)
        assert symbol.arcs == (SymbolArc(16, 16, 48, 48, 48, 32, 16, 32),)

    def test_windows_and_attributes(self) -> None:
        symbol = read_symbol(NMOS)
        assert symbol.windows == (Window(0, 56, 32, "Left", 2), Window(3, 56, 72, "VTop", 1))
        assert symbol.prefix == "MN"
        assert symbol.description == "N-Channel MOSFET transistor"
        assert symbol.attr("Value") == "NMOS"
        assert symbol.attr("SpiceModel") == ""

    def test_a_window_that_stops_short_is_left_aligned_at_normal_size(self) -> None:
        assert read_symbol("WINDOW 0 8 16\nWINDOW 3 8 32 Right big\n").windows == (
            Window(0, 8, 16, "Left", 2),
            Window(3, 8, 32, "Right", 2),
        )

    def test_the_last_of_a_repeated_attribute_is_the_one(self) -> None:
        assert read_symbol("SYMATTR Prefix R\nSYMATTR Prefix X\n").prefix == "X"

    def test_an_attribute_with_no_value_is_empty(self) -> None:
        assert read_symbol("SYMATTR Description\n").attrs == (("Description", ""),)

    def test_an_empty_file_is_an_empty_symbol(self) -> None:
        symbol = read_symbol("")
        assert symbol.pins == ()
        assert symbol.bbox == BBox(0, 0, 0, 0)


class TestTheBox:
    def test_it_bounds_the_body(self) -> None:
        symbol = read_symbol("LINE Normal 0 0 10 5\nLINE Normal -5 -10 0 0\n")
        assert symbol.bbox == BBox(-5, -10, 10, 5)

    def test_it_reaches_out_to_a_pin(self) -> None:
        symbol = read_symbol("LINE Normal 0 0 10 10\nPIN -5 20 NONE 0\n")
        assert symbol.bbox == BBox(-5, 0, 10, 20)

    def test_an_arc_counts_as_what_is_drawn_of_it(self) -> None:
        # Half a circle, from the lower right round the top to the upper left.
        # The two points only give its directions, so ones written far outside
        # the ellipse do not widen the box; the half not drawn does not either.
        symbol = read_symbol("ARC Normal 0 0 10 10 99 99 -99 -99\n")
        assert symbol.bbox == BBox(1, 0, 10, 9)
        assert symbol.body == BBox(1, 0, 10, 9)

    def test_the_body_leaves_the_pins_out(self) -> None:
        symbol = read_symbol("LINE Normal 0 0 10 10\nPIN -5 20 NONE 0\n")
        assert symbol.body == BBox(0, 0, 10, 10)
        assert read_symbol("PIN -5 20 NONE 0\n").body is None


class TestAnArc:
    """What is drawn of an ellipse: from the start point's direction to the end
    point's, counter-clockwise as displayed, with y pointing down."""

    def test_a_quarter(self) -> None:
        # From the bottom of the circle to its right: the lower right quarter.
        assert SymbolArc(0, 0, 40, 40, 20, 40, 40, 20).extent() == (20, 20, 40, 40)

    def test_the_other_three_quarters(self) -> None:
        # The same two points the other way round: all but that quarter, which
        # touches the box on every side.
        assert SymbolArc(0, 0, 40, 40, 40, 20, 20, 40).extent() == (0, 0, 40, 40)

    def test_a_shallow_arc_of_a_large_circle(self) -> None:
        # The curved plate of a polarized capacitor: sixty degrees across the
        # top of a circle of radius 32, four units deep.
        arc = SymbolArc(-16, 36, 48, 100, 32, 40, 0, 40)
        assert arc.extent() == (0, 36, 32, 41)

    def test_a_loop_of_a_coil(self) -> None:
        # Three quarters of a circle, open towards the left: it stops short of
        # the left side of its box.
        assert SymbolArc(0, 40, 32, 72, 4, 68, 4, 44).extent() == (4, 40, 32, 72)

    def test_one_direction_for_both_points_is_the_whole_ellipse(self) -> None:
        assert SymbolArc(0, 0, 40, 20, 40, 10, 40, 10).extent() == (0, 0, 40, 20)

    def test_an_arc_that_ends_on_a_side_of_its_box_reaches_it(self) -> None:
        # From the right of the circle to its top, a quarter turn exactly.
        assert SymbolArc(0, 0, 40, 40, 40, 20, 20, 0).extent() == (20, 0, 40, 20)

    def test_an_ellipse_with_no_area_draws_nothing(self) -> None:
        assert SymbolArc(0, 0, 0, 40, 0, 0, 0, 40).extent() is None
        assert read_symbol("ARC Normal 0 0 0 40 0 0 0 40\n").body is None

    def test_the_points_the_renderer_draws_are_inside_it(self) -> None:
        arc = SymbolArc(-16, 36, 48, 100, 32, 40, 0, 40)
        start, turn = arc.sweep() or (0.0, 0.0)
        x1, y1, x2, y2 = arc.extent() or (0, 0, 0, 0)
        for step in range(25):
            x, y = arc.at(start + turn * step / 24)
            assert x1 - 1e-6 <= x <= x2 + 1e-6
            assert y1 - 1e-6 <= y <= y2 + 1e-6


class TestLinesThatDoNotRead:
    @pytest.mark.parametrize(
        "line",
        [
            "LINE Normal 0 0 10 10",
            "RECTANGLE Normal -5 -10 5 10",
            "CIRCLE Normal 0 0 20 20",
            "ARC Normal 0 0 10 10 0 5 5 10",
        ],
    )
    def test_a_whole_shape_is_part_of_the_body(self, line: str) -> None:
        symbol = read_symbol(line + "\n")
        assert len(symbol.lines + symbol.rects + symbol.circles) + len(symbol.arcs) == 1

    @pytest.mark.parametrize(
        "line",
        [
            "Version 4",
            "",
            "LINE Normal 0",
            "LINE Normal a b c d",
            "RECTANGLE Normal 0 0 16",
            "ARC Normal 0 0 10 10",
            "ARC Normal 0 0 10 10 0 5 5 x",
            "WINDOW 0 a 16 Left 2",
            "SYMATTR",
        ],
    )
    def test_a_line_that_does_not_read_is_left_out(self, line: str) -> None:
        symbol = read_symbol(line + "\n")
        assert symbol == read_symbol("")

    def test_a_pin_that_does_not_read_is_left_out_and_listed(self) -> None:
        symbol = read_symbol(
            "PIN 5\nPINATTR PinName lost\nPIN 1 2 NONE 0\nPINATTR PinName Z\nPINATTR SpiceOrder 1\n"
        )
        assert symbol.pins == (PinInfo("Z", 1, 1, 2),)
        assert symbol.unread_pins == ("PIN 5",)

    def test_a_pin_whose_order_does_not_read_is_kept_and_listed(self) -> None:
        symbol = read_symbol("PIN 1 2 NONE 0\nPINATTR PinName Z\nPINATTR SpiceOrder first\n")
        assert symbol.pins == (PinInfo("Z", 0, 1, 2),)
        assert symbol.unread_pins == ("PINATTR SpiceOrder first",)

    def test_a_pin_line_outside_a_pin_is_not_read(self) -> None:
        assert read_symbol("PINATTR PinName stray\nPINATTR SpiceOrder x\n") == read_symbol("")


class TestAPinNameWithASpace:
    """LTspice's own library names pins ``OUT A`` and ``INV B``."""

    SYMBOL = (
        "Version 4\nSymbolType CELL\nRECTANGLE Normal 0 0 32 32\n"
        "PIN 0 16 NONE 0\nPINATTR PinName OUT A\nPINATTR SpiceOrder 1\n"
    )

    def test_the_name_is_everything_after_the_attribute(self) -> None:
        assert read_symbol(self.SYMBOL).pins == (PinInfo("OUT A", 1, 0, 16),)

    def test_the_editor_cannot_open_a_sheet_that_places_it(self, tmp_path: Path) -> None:
        """spicelib's symbol reader unpacks the line into three words
        (``docs/spicelib_bugs.md``, Bug 25), and it reads every symbol a sheet
        places while it loads the sheet."""
        (tmp_path / "dual.asy").write_text(self.SYMBOL, encoding="utf-8")
        sheet = tmp_path / "with_dual.asc"
        sheet.write_text(
            "Version 4\nSHEET 1 880 680\nSYMBOL dual 0 0 R0\nSYMATTR InstName U1\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="too many values to unpack"):
            make_editor(sheet)


class TestBothReadersUseIt:
    """The editor's pin geometry and the renderer's body are one reading."""

    def test_the_suite_holds_symbols_to_read(self) -> None:
        assert len(SUITE_SYMBOLS) >= 12

    @pytest.mark.parametrize("path", SUITE_SYMBOLS, ids=suite_name)
    def test_they_agree_on_pins_and_box(self, path: Path) -> None:
        for_the_editor = parse_asy_file(path)
        for_the_renderer = parse_symbol(path, path.stem)
        assert for_the_editor.pins == for_the_renderer.pins
        assert for_the_editor.bbox == for_the_renderer.bbox
        assert for_the_editor.pins, "a fixture symbol with no pins tests nothing here"

    def test_a_symbol_with_an_unreadable_pin(self, tmp_path: Path) -> None:
        asy = tmp_path / "broken.asy"
        asy.write_text("LINE Normal 0 0 16 16\nPIN 5\nPIN 0 0 NONE 0\nPINATTR PinName A\n")
        # Drawn with the pin that reads; refused where a missing pin would be a
        # connection nobody sees.
        assert [pin.name for pin in parse_symbol(asy, "broken").pins] == ["A"]
        with pytest.raises(ValueError, match="unreadable pin line 'PIN 5'"):
            parse_asy_file(asy)

    @pytest.mark.parametrize("rotation", ROTATIONS)
    def test_a_part_drawn_with_arcs_has_one_box(self, rotation: str, tmp_path: Path) -> None:
        """The box the editor reports for a part and the box the checker judges
        it by are the same box, an arc counting as what is drawn of it."""
        (tmp_path / "coil.asy").write_text(
            "Version 4\nSymbolType CELL\n"
            "ARC Normal 0 32 32 64 6 60 6 36\n"
            "ARC Normal -32 0 64 96 48 8 -16 8\n"
            "PIN 16 0 NONE 0\nPINATTR PinName A\nPINATTR SpiceOrder 1\n"
            "PIN 16 64 NONE 0\nPINATTR PinName B\nPINATTR SpiceOrder 2\n",
            encoding="utf-8",
        )
        sheet = tmp_path / "coiled.asc"
        sheet.write_text(
            f"Version 4\nSHEET 1 880 680\nSYMBOL coil 160 160 {rotation}\nSYMATTR InstName L1\n",
            encoding="utf-8",
        )
        editor = make_editor(sheet)
        assert isinstance(editor, AscEditor)
        (for_the_editor,) = editor_view(editor).parts
        scene = build_scene(sheet, SymbolResolver(local_dir=tmp_path))
        (for_the_checker,) = scene_view(scene).parts
        assert for_the_editor.box == for_the_checker.box
        assert for_the_editor.pins == for_the_checker.pins
        assert [name for name, _x, _y in for_the_checker.pins] == ["A", "B"]
        assert for_the_checker.box is not None and for_the_checker.body is not None
        # The shallow arc is a sliver of its circle's box, which is 96 units
        # each way: the part is nowhere near that size.
        assert max(for_the_checker.box.width, for_the_checker.box.height) <= 64

    def test_a_half_written_arc_is_in_neither_box(self, tmp_path: Path) -> None:
        asy = tmp_path / "arc.asy"
        asy.write_text("LINE Normal 0 0 16 16\nARC Normal -64 -64 64 64\n")
        assert parse_asy_file(asy).bbox == BBox(0, 0, 16, 16)
        assert parse_symbol(asy, "arc").bbox == BBox(0, 0, 16, 16)
