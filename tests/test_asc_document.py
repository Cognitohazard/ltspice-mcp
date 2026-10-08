"""The lossless ``.asc`` document: a sheet reads into records and writes back as
it was, an edit changes only the records it touched, and a formatted record is
the line the present schematic engine writes.
"""

from __future__ import annotations

import dataclasses
import io
from pathlib import Path

import pytest
from spicelib import AscEditor

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.asc_document import (
    AscDocument,
    Flag,
    Shape,
    SheetHeader,
    Symbol,
    Text,
    Window,
    Wire,
    parse_asc,
)
from ltspice_mcp.lib.schematic_ops import blank_sheet
from ltspice_mcp.state import SessionState

from ._asc_ops import build_sheet
from ._schematic_fixtures import SUITE_SHEETS, TESTS, suite_name
from ._schematic_fixtures import formatted as _formatted

#: Kinds a sheet's connectivity and parts are made of. One of these kept as
#: text means the reader did not understand the sheet.
_ELECTRICAL = {"SYMBOL", "WINDOW", "SYMATTR", "WIRE", "FLAG", "IOPIN"}

SHEET = (
    "Version 4\n"
    "SHEET 1 880 680\n"
    "WIRE 32 128 -48 128\n"
    "WIRE 240 128 112 128\n"
    "FLAG 240 256 0\n"
    "FLAG 208 128 out\n"
    "SYMBOL res 128 112 R90\n"
    "WINDOW 0 0 56 VBottom 2\n"
    "WINDOW 3 32 56 VTop 2\n"
    "SYMATTR InstName R1\n"
    "SYMATTR Value 1k\n"
    "SYMBOL voltage -48 128 R0\n"
    "SYMATTR InstName V1\n"
    "SYMATTR Value PULSE(0 5 0 1m 1m 0 1)\n"
    "TEXT -80 320 Left 2 !.tran 10m\n"
    "TEXT -80 352 Left 2 ;a note\n"
)


def _doc(text: str) -> AscDocument:
    return parse_asc(text.encode("utf-8"))


class TestEverySheetInTheSuite:
    """Every schematic the suite holds, in whatever encoding it is saved in."""

    def test_the_suite_holds_sheets_to_read(self) -> None:
        assert len(SUITE_SHEETS) >= 40

    @pytest.mark.parametrize("path", SUITE_SHEETS, ids=suite_name)
    def test_reads_and_writes_back_to_its_own_bytes(self, path: Path) -> None:
        data = path.read_bytes()
        assert parse_asc(data).to_bytes() == data

    @pytest.mark.parametrize("path", SUITE_SHEETS, ids=suite_name)
    def test_every_part_wire_and_label_reads_as_a_record(self, path: Path) -> None:
        kept_as_text = {record.keyword for record in parse_asc(path.read_bytes()).opaque}
        assert not kept_as_text & _ELECTRICAL


class TestRecords:
    def test_each_kind_reads_into_its_type(self) -> None:
        doc = _doc(SHEET)
        assert doc.version == "4"
        assert doc.sheet == SheetHeader(1, 880, 680)
        assert doc.wires == (Wire(32, 128, -48, 128), Wire(240, 128, 112, 128))
        assert doc.flags == (Flag(240, 256, "0"), Flag(208, 128, "out"))
        assert doc.texts == (
            Text(-80, 320, "Left", 2, "!", ".tran 10m"),
            Text(-80, 352, "Left", 2, ";", "a note"),
        )
        assert doc.opaque == ()

    def test_a_symbol_holds_the_lines_under_it(self) -> None:
        r1 = _doc(SHEET).symbol("R1")
        assert r1 is not None
        assert r1 == Symbol(
            "res",
            128,
            112,
            "R90",
            windows=(Window(0, 0, 56, "VBottom", 2), Window(3, 32, 56, "VTop", 2)),
            attrs=(("InstName", "R1"), ("Value", "1k")),
        )
        assert r1.reference == "R1"
        assert r1.attr("Value") == "1k"
        assert r1.attr("SpiceLine") is None

    def test_a_value_keeps_its_spaces(self) -> None:
        v1 = _doc(SHEET).symbol("V1")
        assert v1 is not None
        assert v1.attr("Value") == "PULSE(0 5 0 1m 1m 0 1)"

    def test_a_symbol_in_a_library_folder_keeps_its_whole_name(self) -> None:
        doc = _doc("SYMBOL Opamps\\\\opamp2 64 32 M180\nSYMATTR InstName U1\n")
        assert doc.symbols[0].symbol == "Opamps\\\\opamp2"
        assert doc.symbols[0].rotation == "M180"

    def test_an_attribute_with_no_value_reads_as_empty(self) -> None:
        text = "SYMBOL res 0 0 R0\nSYMATTR InstName R1\nSYMATTR Value\n"
        doc = _doc(text)
        assert doc.symbols[0].attr("Value") == ""
        assert doc.to_text() == text
        assert _formatted(doc).to_text() == text

    def test_a_port_is_the_flag_and_the_line_after_it(self) -> None:
        text = "FLAG 96 64 in\nIOPIN 96 64 In\nFLAG 96 128 0\n"
        doc = _doc(text)
        assert doc.flags == (Flag(96, 64, "in", port="In"), Flag(96, 128, "0"))
        assert doc.opaque == ()
        assert _formatted(doc).to_text() == text

    def test_a_directive_body_is_kept_as_stored(self) -> None:
        doc = _doc("TEXT 16 16 VRight 3 !.param a=1\\n.param b=2  \n")
        (text,) = doc.texts
        assert (text.align, text.size, text.is_directive) == ("VRight", 3, True)
        assert text.body == ".param a=1\\n.param b=2  "

    def test_drawn_shapes_read_with_their_style(self) -> None:
        text = (
            "LINE Normal 0 0 64 0\n"
            "RECTANGLE Normal 0 0 64 64 2\n"
            "CIRCLE Normal 0 0 32 32\n"
            "ARC Normal 0 0 32 32 32 16 0 16 1\n"
        )
        doc = _doc(text)
        assert doc.shapes == (
            Shape("LINE", "Normal", (0, 0, 64, 0)),
            Shape("RECTANGLE", "Normal", (0, 0, 64, 64), style=2),
            Shape("CIRCLE", "Normal", (0, 0, 32, 32)),
            Shape("ARC", "Normal", (0, 0, 32, 32, 32, 16, 0, 16), style=1),
        )
        assert _formatted(doc).to_text() == text


class TestLinesKeptAsText:
    """Nothing is refused and nothing is dropped."""

    @pytest.mark.parametrize(
        "line",
        [
            'DATAFLAG 128 64 "$"',
            "BUSTAP 96 64 112 80",
            "SOMETHING a later LTspice writes",
            "",
            "   ",
        ],
    )
    def test_a_line_with_no_record_type_is_kept(self, line: str) -> None:
        text = f"Version 4\nSHEET 1 880 680\n{line}\nWIRE 0 0 16 0\n"
        doc = _doc(text)
        assert [record.text for record in doc.opaque] == [line]
        assert doc.wires == (Wire(0, 0, 16, 0),)
        assert doc.to_text() == text

    @pytest.mark.parametrize(
        "line",
        [
            "WIRE 0 0 16",
            "WIRE 0 0 16 x",
            "WIRE 0 0 16 0 32",
            "FLAG 16 16",
            "FLAG a b net",
            "SHEET 1 880",
            "Version",
            "TEXT 16 16 Left 2 no kind mark",
            "TEXT 16 16 Sideways 2 !.op",
            "LINE Normal 0 0 64",
            "IOPIN 0 0 In",
            "WINDOW 0 0 56 Left 2",
            "SYMATTR Value 1k",
        ],
    )
    def test_a_known_kind_that_does_not_read_is_kept(self, line: str) -> None:
        text = f"Version 4\n{line}\nWIRE 0 0 16 0\n"
        doc = _doc(text)
        assert [record.text for record in doc.opaque] == [line]
        assert doc.to_text() == text

    @pytest.mark.parametrize(
        "block",
        [
            "SYMBOL res 0 0 R45\nSYMATTR InstName R1\n",
            "SYMBOL res 0 x R0\nSYMATTR InstName R1\n",
            "SYMBOL res 0 0\nSYMATTR InstName R1\n",
        ],
    )
    def test_a_symbol_whose_own_line_does_not_read_is_kept_as_text(self, block: str) -> None:
        # With no place or orientation there is no part to speak of.
        text = f"Version 4\n{block}WIRE 0 0 16 0\n"
        doc = _doc(text)
        assert doc.symbols == ()
        assert "".join(record.text + "\n" for record in doc.opaque) == block
        assert doc.to_text() == text

    @pytest.mark.parametrize(
        "line",
        ["WINDOW 0 a 56 Left 2", "WINDOW 3 0 56 Sideways 2", "WINDOW 0 0 56", "SYMATTR"],
    )
    def test_a_part_with_a_line_under_it_that_does_not_read_is_still_a_part(
        self, line: str
    ) -> None:
        text = (
            "Version 4\n"
            "SYMBOL res 0 0 R0\n"
            "WINDOW 3 32 56 VTop 2\n"
            f"{line}\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
            "WIRE 0 0 16 0\n"
        )
        doc = _doc(text)
        r1 = doc.symbol("R1")
        assert r1 is not None
        assert (r1.x, r1.y, r1.rotation, r1.attr("Value")) == (0, 0, "R0", "1k")
        assert r1.windows == (Window(3, 32, 56, "VTop", 2),)
        assert r1.unread == (line,)
        assert doc.opaque == ()
        assert doc.to_text() == text
        # An edit to the part keeps the line, after the ones that read.
        assert doc.replaced(r1, r1.changed(x=64)).to_text() == (
            "Version 4\n"
            "SYMBOL res 64 0 R0\n"
            "WINDOW 3 32 56 VTop 2\n"
            "SYMATTR InstName R1\n"
            "SYMATTR Value 1k\n"
            f"{line}\n"
            "WIRE 0 0 16 0\n"
        )

    def test_a_port_line_at_another_point_is_not_the_flags(self) -> None:
        text = "FLAG 96 64 in\nIOPIN 0 0 In\n"
        doc = _doc(text)
        assert doc.flags == (Flag(96, 64, "in"),)
        assert [record.keyword for record in doc.opaque] == ["IOPIN"]
        assert doc.to_text() == text

    def test_a_line_kept_as_text_keeps_its_own_ending(self) -> None:
        # CRLF lines in a sheet that is otherwise LF: neither is rewritten in
        # the sheet's usual ending.
        data = (
            b'Version 4\nSHEET 1 880 680\nDATAFLAG 0 0 ""\r\nSYMBOL res 0 0 R45\r\nWIRE 0 0 16 0\n'
        )
        doc = parse_asc(data)
        assert doc.newline == "\n"
        assert [record.keyword for record in doc.opaque] == ["DATAFLAG", "SYMBOL"]
        assert doc.to_bytes() == data


class TestEncodingsAndLineEndings:
    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
    def test_a_sheet_keeps_its_line_ending(self, newline: str) -> None:
        data = SHEET.replace("\n", newline).encode("utf-8")
        doc = parse_asc(data)
        assert doc.newline == newline
        assert doc.to_bytes() == data
        assert len(doc.symbols) == 2

    def test_a_new_record_takes_the_sheets_line_ending(self) -> None:
        data = SHEET.replace("\n", "\r\n").encode("utf-8")
        edited = parse_asc(data).added(Wire(0, 0, 16, 0)).to_bytes()
        assert b"WIRE 0 0 16 0\r\n" in edited
        assert b"\n" not in edited.replace(b"\r\n", b"")

    def test_mixed_line_endings_are_kept_line_by_line(self) -> None:
        data = b"Version 4\r\nSHEET 1 880 680\nWIRE 0 0 16 0\r\nWIRE 16 0 32 0\r\n"
        doc = parse_asc(data)
        assert doc.newline == "\r\n"
        assert doc.to_bytes() == data

    def test_a_last_line_with_no_ending_stays_that_way(self) -> None:
        data = b"Version 4\nSHEET 1 880 680\nWIRE 0 0 16 0"
        assert parse_asc(data).to_bytes() == data

    def test_a_record_added_after_an_unended_line_ends_it(self) -> None:
        doc = parse_asc(b"Version 4\nSHEET 1 880 680\nWIRE 0 0 16 0")
        assert doc.added(Wire(16, 0, 32, 0)).to_bytes() == (
            b"Version 4\nSHEET 1 880 680\nWIRE 0 0 16 0\nWIRE 16 0 32 0\n"
        )

    @pytest.mark.parametrize(
        ("codec", "bom"),
        [
            ("utf-16-le", b"\xff\xfe"),
            ("utf-16-le", b""),
            ("utf-16-be", b"\xfe\xff"),
            ("utf-8", b"\xef\xbb\xbf"),
            ("utf-8", b""),
        ],
    )
    def test_a_sheet_keeps_its_encoding_and_mark(self, codec: str, bom: bytes) -> None:
        data = bom + SHEET.replace("1k", "1µ").encode(codec)
        doc = parse_asc(data)
        assert doc.bom == bom
        r1 = doc.symbol("R1")
        assert r1 is not None and r1.attr("Value") == "1µ"
        assert doc.to_bytes() == data
        assert doc.replaced(r1, r1.with_attr("Value", "2µ")).to_bytes() == (
            bom + SHEET.replace("1k", "2µ").encode(codec)
        )

    def test_an_eight_bit_sheet_keeps_every_byte(self) -> None:
        # 0xB5 is the micro sign in cp1252; 0x81 has no character there and is
        # kept all the same.
        data = SHEET.encode("ascii").replace(b"1k", b"1\xb5").replace(b"a note", b"a \x81 note")
        doc = parse_asc(data)
        assert doc.encoding == "cp1252"
        r1 = doc.symbol("R1")
        assert r1 is not None and r1.attr("Value") == "1µ"
        assert doc.to_bytes() == data
        assert _formatted(doc).to_bytes() == data

    def test_a_character_the_encoding_cannot_spell_is_named(self) -> None:
        doc = parse_asc(SHEET.encode("ascii").replace(b"1k", b"1\xb5"))
        edited = doc.added(Text(0, 0, "Left", 2, ";", "10 kΩ"))
        with pytest.raises(NetlistError, match="Ω"):
            edited.to_bytes()
        assert "10 kΩ".encode() in edited.with_encoding("utf-8").to_bytes()

    def test_bytes_that_are_not_text_are_refused(self) -> None:
        with pytest.raises(NetlistError, match="not readable text"):
            parse_asc(b"\xff\xfeV\x00e\x00r\x00\x00\xd8")


class TestEdits:
    def test_a_blank_sheet_is_the_template_the_tool_starts_from(self) -> None:
        assert AscDocument.blank().to_bytes() == blank_sheet().encode("utf-8")

    def test_a_change_rewrites_only_the_record_it_touched(self) -> None:
        # Spacing no formatter would write, so a rewritten line shows.
        text = SHEET.replace("WIRE 32 128 -48 128", "WIRE  32 128   -48 128").replace(
            "SYMATTR Value 1k", "SYMATTR Value   1k  "
        )
        doc = _doc(text)
        second = doc.wires[1]
        edited = doc.replaced(second, second.changed(x2=96))
        assert edited.to_text() == text.replace("WIRE 240 128 112 128", "WIRE 240 128 96 128")

    def test_a_copy_made_without_changed_is_still_written_as_it_now_is(self) -> None:
        doc = _doc(SHEET.replace("WIRE 32 128 -48 128", "WIRE  32 128   -48 128"))
        first = doc.wires[0]
        stale = dataclasses.replace(first, x1=48)
        assert stale.source == first.source
        assert "WIRE 48 128 -48 128\n" in doc.replaced(first, stale).to_text()

    def test_a_changed_symbol_is_written_in_the_formatted_order(self) -> None:
        text = "SYMBOL res 0 0 R0\nSYMATTR Value 1k\nSYMATTR InstName R1\n"
        doc = _doc(text)
        assert doc.to_text() == text
        r1 = doc.symbols[0]
        assert doc.replaced(r1, r1.with_attr("SpiceLine", "tol=1")).to_text() == (
            "SYMBOL res 0 0 R0\nSYMATTR InstName R1\nSYMATTR Value 1k\nSYMATTR SpiceLine tol=1\n"
        )
        assert doc.replaced(r1, r1.without_attr("Value")).to_text() == (
            "SYMBOL res 0 0 R0\nSYMATTR InstName R1\n"
        )

    def test_a_record_is_addressed_as_the_one_handed_out(self) -> None:
        doc = _doc("Version 4\nWIRE 0 0 16 0\nWIRE 0 0 16 0\nWIRE 16 0 32 0\n")
        first, second, _ = doc.wires
        assert first == second
        assert doc.removed(second).wires[0] is first
        assert len(doc.removed(first, second).wires) == 1
        with pytest.raises(ValueError, match="not a record of this document"):
            doc.removed(Wire(0, 0, 16, 0))

    def test_a_new_record_goes_after_the_last_of_its_kind(self) -> None:
        doc = _doc(SHEET).added(
            Text(0, 0, "Left", 2, "!", ".op"),
            Wire(0, 0, 16, 0),
            Flag(16, 0, "n1"),
            Symbol("cap", 0, 0, "R0", attrs=(("InstName", "C1"),)),
            Shape("RECTANGLE", "Normal", (0, 0, 64, 64)),
            Shape("LINE", "Normal", (0, 0, 64, 0)),
        )
        kinds = [type(record).__name__ for record in doc.records]
        assert kinds == (
            ["Version", "SheetHeader"]
            + ["Wire"] * 3
            + ["Flag"] * 3
            + ["Symbol"] * 3
            + ["Text"] * 3
            + ["Shape"] * 2
        )
        assert doc.wires[-1] == Wire(0, 0, 16, 0)
        assert doc.symbols[-1].reference == "C1"
        assert [shape.kind for shape in doc.shapes] == ["LINE", "RECTANGLE"]

    def test_a_kind_the_sheet_lacks_goes_where_a_formatted_sheet_has_it(self) -> None:
        doc = AscDocument.blank().added(
            Symbol("res", 0, 0, "R0", attrs=(("InstName", "R1"),)),
            Text(16, 16, "Left", 2, "!", ".op"),
            Flag(0, 0, "0"),
            Wire(0, 0, 16, 0),
        )
        assert doc.to_text() == (
            "Version 4\n"
            "SHEET 1 880 680\n"
            "WIRE 0 0 16 0\n"
            "FLAG 0 0 0\n"
            "SYMBOL res 0 0 R0\n"
            "SYMATTR InstName R1\n"
            "TEXT 16 16 Left 2 !.op\n"
        )

    def test_a_change_leaves_the_document_it_was_made_from(self) -> None:
        doc = _doc(SHEET)
        doc.added(Wire(0, 0, 16, 0)).removed(doc.flags[0])
        assert doc.to_text() == SHEET


class TestTheFormatThePresentEngineWrites:
    """A formatted record is the line spicelib's writer produces, so moving the
    engine onto this document changes no byte of a sheet built from blank."""

    @pytest.mark.parametrize("name", ["Draft1.asc", "nmos4_mirrored_quarter_turns.asc"])
    def test_spicelibs_own_output_formats_back_to_itself(
        self, asc_symbols: Path, name: str
    ) -> None:
        buffer = io.StringIO()
        AscEditor(str(TESTS / "fixtures" / name)).save_netlist(buffer)
        written = buffer.getvalue()
        assert _formatted(parse_asc(written.encode("utf-8"))).to_text() == written

    async def test_a_sheet_built_through_the_tool_is_rebuilt_byte_for_byte(
        self, asc_state: SessionState, work_dir: Path
    ) -> None:
        envelope = await build_sheet(
            asc_state,
            "rebuilt",
            [
                {
                    "op": "add_component",
                    "reference": "V1",
                    "symbol": "voltage",
                    "x": 96,
                    "y": 96,
                    "value": "PULSE(0 5 0 1m 1m 0 1)",
                },
                {
                    "op": "add_component",
                    "reference": "R1",
                    "symbol": "res",
                    "x": 304,
                    "y": 64,
                    "rotation": "R90",
                    "value": "1k",
                    "attributes": {"SpiceLine": "tol=1"},
                },
                {
                    "op": "add_component",
                    "reference": "M1",
                    "symbol": "nmos",
                    "x": 400,
                    "y": 160,
                    "rotation": "M0",
                    "value": "NMOS_A",
                },
                {"op": "add_net_label", "net": "0", "pin": "V1.-"},
                {"op": "add_net_label", "net": "out", "pin": "M1.D"},
                {"op": "add_directive", "instruction": ".tran 10m"},
                {
                    "op": "add_directive",
                    "instruction": "a note",
                    "kind": "comment",
                    "x": 96,
                    "y": 320,
                },
                {"op": "wire_pins", "from_pin": "V1.+", "to_pin": "R1.2"},
            ],
        )
        assert envelope["outcome"] == "complete", envelope
        written = (work_dir / "rebuilt.asc").read_bytes()
        built = parse_asc(written)
        assert built.to_bytes() == written
        assert not built.opaque

        # The same records, made in memory in the order the ops ran.
        rebuilt = AscDocument.blank().added(
            *built.symbols, *built.flags, *built.texts, *built.wires
        )
        assert _formatted(rebuilt).to_bytes() == written
        assert [symbol.reference for symbol in built.symbols] == ["V1", "R1", "M1"]
        assert built.symbol("R1") == Symbol(
            "res",
            304,
            64,
            "R90",
            attrs=(("InstName", "R1"), ("Value", "1k"), ("SpiceLine", "tol=1")),
        )
        assert built.wires == (Wire(96, 64, 256, 64),)
        assert Text(96, 320, "Left", 2, ";", "a note") in built.texts
