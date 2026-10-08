"""The sheet LTspice writes when it saves one, held against the server's reading of it.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: sheets
that LTspice 26 and LTspice XVII each opened in their window and saved, with
nothing on them changed. They are the only sheets in the suite a build wrote
itself, so they are what the lossless document's claims are worth: that it
reads what LTspice writes, writes it back byte for byte, and formats a record
the way LTspice spells it.
"""

from __future__ import annotations

import os

import pytest

from ltspice_mcp.lib.asc_document import Shape, Symbol, parse_asc
from tests import _ltspice_recorded as rec
from tests._schematic_fixtures import formatted
from tests.ltspice_recorder import INPUTS, discover_builds

SAVES = rec.cases_of("schematic-save-encoding")
#: The sheets a build would not open, each for its byte order mark.
MARKED = ["save/micro_utf8_bom", "save/micro_utf16le_bom"]
SAVED = [case_id for case_id in SAVES if case_id not in MARKED]


def saved(build: str, case_id: str) -> bytes:
    return rec.recorded(build, f"{case_id}.asc").read_bytes()


def given(case_id: str) -> bytes:
    return (INPUTS / rec.CASES.case(case_id).source).read_bytes()


def test_the_recording_holds_a_save_of_each_kind_of_sheet():
    assert len(SAVED) >= 6
    assert set(MARKED) < set(SAVES)


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(SAVED)))
class TestWhatABuildWrites:
    def test_the_document_reads_every_line_and_writes_the_sheet_back(
        self, build: str, case_id: str
    ):
        data = saved(build, case_id)
        doc = parse_asc(data)
        assert doc.opaque == ()
        assert doc.to_bytes() == data

    def test_a_formatted_record_is_spelled_as_the_build_spells_it(self, build: str, case_id: str):
        """Line for line. The order of a symbol's attribute lines is the one
        thing that can differ, and has a test of its own below."""
        data = saved(build, case_id)
        written = formatted(parse_asc(data)).to_bytes()
        assert sorted(written.splitlines()) == sorted(data.splitlines())

    def test_line_endings_are_lf_whatever_the_sheet_had(self, build: str, case_id: str):
        data = saved(build, case_id)
        assert b"\r" not in data
        assert data.endswith(b"\n")
        assert parse_asc(data).newline == "\n"

    def test_kinds_come_in_the_order_a_formatted_sheet_has_them(self, build: str, case_id: str):
        """Wires, flags, symbols, text, drawn lines, then the other shapes: the
        order the document puts a new record in."""
        records = parse_asc(saved(build, case_id)).records
        ranks = [record.rank for record in records if record.rank is not None]
        assert len(ranks) == len(records)
        assert ranks == sorted(ranks)

    def test_the_text_is_eight_bit(self, build: str, case_id: str):
        """Whatever encoding the sheet came in, UTF-16 included."""
        data = saved(build, case_id)
        assert b"\0" not in data
        assert parse_asc(data).bom == b""


@pytest.mark.parametrize("build", rec.BUILDS)
class TestWhatASaveChanges:
    def test_the_version_line_is_the_builds_own(self, build: str):
        """LTspice 26 writes 4.1 over a sheet that said 4; XVII leaves 4."""
        assert parse_asc(given("save/built")).version == "4"
        version = parse_asc(saved(build, "save/built")).version
        assert version == ("4" if rec.generation(build) == "xvii" else "4.1")

    def test_a_crlf_sheet_comes_back_as_the_lf_one(self, build: str):
        assert b"\r\n" in given("save/crlf")
        assert given("save/crlf").replace(b"\r\n", b"\n") == given("save/built")
        assert saved(build, "save/crlf") == saved(build, "save/built")

    def test_a_sheet_as_the_server_writes_it_changes_only_in_its_wire_order(self, build: str):
        """Apart from the version line. The server writes wires in the order
        they were drawn; a build writes them in an order of its own."""
        ours, theirs = parse_asc(given("save/built")), parse_asc(saved(build, "save/built"))
        assert sorted(ours.wires, key=repr) == sorted(theirs.wires, key=repr)
        assert ours.wires != theirs.wires
        assert (ours.flags, ours.symbols, ours.texts) == (
            theirs.flags,
            theirs.symbols,
            theirs.texts,
        )

    def test_records_out_of_order_are_put_in_order_and_none_is_lost(self, build: str):
        ours, theirs = parse_asc(given("save/shuffled")), parse_asc(saved(build, "save/shuffled"))
        assert sorted(ours.wires, key=repr) == sorted(theirs.wires, key=repr)
        # Flags, symbols and text keep the order they were in; the port line
        # stays with its flag.
        assert ours.flags == theirs.flags
        assert [flag.port for flag in theirs.flags] == ["Out", None]
        assert [s.reference for s in theirs.symbols] == [s.reference for s in ours.symbols]
        assert ours.texts == theirs.texts
        assert sorted(ours.shapes, key=repr) == sorted(theirs.shapes, key=repr)
        assert [shape.kind for shape in theirs.shapes] == ["LINE", "RECTANGLE", "CIRCLE", "ARC"]

    def test_a_build_writes_the_attribute_a_window_line_names_first(self, build: str):
        """Where the server and a build differ, side by side: R1 has a window
        for its value, and the build writes the value before the name."""
        r1 = parse_asc(saved(build, "save/shuffled")).symbol("R1")
        assert isinstance(r1, Symbol)
        assert [window.number for window in r1.windows] == [3]
        assert [name for name, _value in r1.attrs] == ["Value", "InstName", "SpiceLine"]
        # The server, formatting the same part, leads with the name.
        assert r1.changed().format()[2:] == (
            "SYMATTR InstName R1",
            "SYMATTR Value 1k",
            "SYMATTR SpiceLine tol=1",
        )
        # With no window line, the order the sheet had is the order written.
        r2 = parse_asc(saved(build, "save/shuffled")).symbol("R2")
        assert isinstance(r2, Symbol)
        assert [name for name, _value in r2.attrs] == ["Value", "InstName"]

    def test_eight_bit_text_keeps_its_bytes(self, build: str):
        """A micro sign as one cp1252 byte stays one byte, and as two UTF-8
        bytes stays two: neither build reads a sheet as UTF-8."""
        assert b"1\xb5\n" in saved(build, "save/micro_cp1252")
        assert b"1\xc2\xb5\n" in saved(build, "save/micro_utf8")

    def test_a_utf16_sheet_is_saved_as_eight_bit_text(self, build: str):
        assert given("save/micro_utf16le").count(b"\0") > 50
        assert saved(build, "save/micro_utf16le") == saved(build, "save/micro_cp1252")

    @pytest.mark.parametrize("case_id", MARKED)
    def test_a_sheet_that_starts_with_a_byte_order_mark_is_not_opened(
        self, build: str, case_id: str
    ):
        """Either build's window stops on a box, and nothing is saved."""
        entry = rec.entry(build, case_id)
        assert "Unknown schematic syntax" in entry["dialog"]
        assert entry["outputs"] == {} and entry["written"] == []
        assert not rec.has(build, f"{case_id}.asc")


def test_the_shapes_in_the_recording_are_the_ones_the_document_types():
    shapes = parse_asc(given("save/shuffled")).shapes
    assert {shape.kind for shape in shapes} == {"LINE", "RECTANGLE", "CIRCLE", "ARC"}
    assert all(isinstance(shape, Shape) for shape in shapes)


#: Kinds of line the document keeps as text that are known to draw something
#: and connect nothing. A data flag shows a value on the sheet after a run.
COSMETIC = {"DATAFLAG"}


def test_every_sheet_an_installed_build_ships_is_read_and_written_back():
    """The example library of each build installed here: thousands of sheets
    LTspice wrote over many years, in every line ending and encoding it has
    used, which the repository cannot carry. Reads files; starts nothing.

    A kind of line kept as text that is not listed as cosmetic fails here by
    name, since it may be a connection the document does not see.
    """
    if os.environ.get("LTSPICE_MCP_RUN_LTSPICE_INTEGRATION") != "1":
        pytest.skip(
            "LTspice integration tests are opt-in; set LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1"
        )
    libraries = [
        build.library_root.parent / "examples"
        for build in discover_builds()
        if build.library_root is not None
    ]
    sheets = sorted(sheet for library in libraries for sheet in library.rglob("*.asc"))
    if not sheets:
        pytest.skip("no LTspice example library is installed here")
    wrong: list[str] = []
    for sheet in sheets:
        data = sheet.read_bytes()
        doc = parse_asc(data)
        if doc.to_bytes() != data:
            wrong.append(f"{sheet.name}: not written back byte for byte")
        kept = {record.keyword for record in doc.opaque if record.text.strip()} - COSMETIC
        if kept:
            wrong.append(f"{sheet.name}: kept as text {sorted(kept)}")
        if any(symbol.unread for symbol in doc.symbols):
            wrong.append(f"{sheet.name}: a line under a symbol did not read")
    assert not wrong, f"{len(wrong)} of {len(sheets)} sheets:\n  " + "\n  ".join(wrong[:20])
