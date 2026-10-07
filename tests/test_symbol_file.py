"""The symbol file reader: one reading of a ``.asy`` behind both the schematic
editor's pin geometry and the renderer's symbol body.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.asc_document import Window
from ltspice_mcp.lib.geometry import BBox
from ltspice_mcp.lib.schematic_ops import make_editor
from ltspice_mcp.lib.schematic_scene import parse_symbol
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

    def test_an_arc_counts_as_the_box_of_its_ellipse(self) -> None:
        # The start and end points are on the ellipse; even ones written
        # outside its box do not widen the symbol's.
        symbol = read_symbol("ARC Normal 0 0 10 10 99 99 -99 -99\n")
        assert symbol.bbox == BBox(0, 0, 10, 10)


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

    def test_a_half_written_arc_is_in_neither_box(self, tmp_path: Path) -> None:
        asy = tmp_path / "arc.asy"
        asy.write_text("LINE Normal 0 0 16 16\nARC Normal -64 -64 64 64\n")
        assert parse_asy_file(asy).bbox == BBox(0, 0, 16, 16)
        assert parse_symbol(asy, "arc").bbox == BBox(0, 0, 16, 16)
