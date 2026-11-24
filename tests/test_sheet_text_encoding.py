"""Which text the bytes of a sheet hold, read the same by ``inspect`` and by
``edit_schematic``, and written back by an edit in the bytes they came in.

``lib/encoding.py`` decides the reading for every reader of a sheet. An edit
of one part leaves every other line of the sheet as it was, in its encoding,
its byte order mark (where LTspice reads the sheet behind one) and its line
ends.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from ltspice_mcp.errors import NetlistError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import schematic_edit as se
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.schematic_edit import EditSchematicInput, handle_edit_schematic
from tests.conftest import LIVENESS_S
from tests.test_parallel_sessions import _hold_lock_then_write


def _sheet(label: str, value: str, comment: str = "drawn by hand") -> str:
    """Two resistors, a label on R1's top pin and a comment, in the order the
    editor writes a sheet's records, so an edit renders every line it did not
    change as it was."""
    return (
        "Version 4\n"
        "SHEET 1 880 680\n"
        "WIRE 80 -48 0 -48\n"
        f"FLAG 0 -48 {label}\n"
        "SYMBOL res 0 0 R0\n"
        "SYMATTR InstName R1\n"
        f"SYMATTR Value {value}\n"
        "SYMBOL res 160 0 R0\n"
        "SYMATTR InstName R2\n"
        "SYMATTR Value 1k\n"
        f"TEXT 0 120 Left 2 ;{comment}\n"
    )


def _cp1252(text: str) -> bytes:
    """``text`` as an 8-bit sheet holds it, the five bytes cp1252 leaves
    undefined written as the control character of the same number."""
    return text.encode("latin-1")


async def _inspect(state: SessionState, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sheet's parts and the net at R1's top pin, as ``inspect`` reads them."""
    results = (
        await handle_inspect(
            InspectInput.model_validate(
                {
                    "queries": [
                        {"kind": "components", "path": str(path), "detail": "full"},
                        {"kind": "net", "path": str(path), "at": [0, -48]},
                    ]
                }
            ),
            state,
        )
    ).structured_content["results"]
    components, net = results
    assert components["ok"], components
    assert net["ok"], net
    return components["data"], net["data"]


async def _edit(state: SessionState, path: Path, ops: list[dict], **kw: Any) -> dict[str, Any]:
    request = EditSchematicInput.model_validate(
        {
            "path": str(path),
            "expected_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),  # noqa: ASYNC240
            "ops": ops,
        }
        | kw
    )
    data = (await handle_edit_schematic(request, state)).structured_content
    assert data is not None
    jsonschema.Draft202012Validator(se._OUTPUT_SCHEMA).validate(data)
    return data


async def _editor_reading(state: SessionState, path: Path) -> tuple[str, str]:
    """R1's value and the label at its top pin, as the editor reads them.

    A dry run that sets R1's value reports the value it found, and one that
    puts a second name on the label's net names both labels it found there.
    Neither writes the sheet.
    """
    data = await _edit(
        state,
        path,
        [
            {"op": "set_component_value", "reference": "R1", "value": "5k"},
            {"op": "add_net_label", "net": "MID", "x": 80, "y": -48},
        ],
        dry_run=True,
    )
    assert data["outcome"] == "complete", data
    value = data["results"][0]["changed"]["Value"]["was"]
    named_twice = [one for one in data["warnings"] if "MID" in one]
    assert len(named_twice) == 1, data["warnings"]
    return value, named_twice[0]


async def _assert_both_read(
    state: SessionState, path: Path, value: str, label: str
) -> dict[str, Any]:
    """``inspect`` and the editor both read R1's value as ``value`` and the
    label as ``label``; returns the dry run's reading for further checks."""
    components, net = await _inspect(state, path)
    r1 = next(one for one in components["components"] if one["reference"] == "R1")
    assert r1["value"] == value
    assert net["labels"] == [label]
    edited_value, named_twice = await _editor_reading(state, path)
    assert edited_value == value
    assert repr(label) in named_twice, named_twice
    return {"value": edited_value, "warning": named_twice}


async def _edit_r2_keeps_every_other_byte(
    state: SessionState, path: Path, newline: bytes = b"\n"
) -> dict[str, Any]:
    """Set R2's value and check that only its value line changed."""
    before = path.read_bytes()  # noqa: ASYNC240
    data = await _edit(
        state, path, [{"op": "set_component_value", "reference": "R2", "value": "2k"}]
    )
    assert data["outcome"] == "complete", data
    assert data["commit_state"] == "committed", data
    old, new = b"SYMATTR Value 1k" + newline, b"SYMATTR Value 2k" + newline
    assert before.count(old) == 1
    assert path.read_bytes() == before.replace(old, new)  # noqa: ASYNC240
    assert data["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: ASYNC240
    return data


class TestEightBitSheet:
    """A sheet with no mark that is not UTF-8 is 8-bit text, read byte by byte
    as Windows-1252 reads it, the five bytes it leaves undefined as the control
    character of the same number (``lib/encoding.py``)."""

    async def test_micro_sign_and_accented_label_read_alike_and_keep_their_bytes(
        self, asc_state, work_dir
    ):
        path = work_dir / "accent.asc"
        path.write_bytes(_cp1252(_sheet("à", "1µ")))
        await _assert_both_read(asc_state, path, "1µ", "à")
        await _edit_r2_keeps_every_other_byte(asc_state, path)

    @pytest.mark.parametrize("undefined", ["\x8d", "\x81"], ids=["0x8D", "0x81"])
    async def test_a_byte_cp1252_leaves_undefined_is_read_alike_and_not_refused(
        self, asc_state, work_dir, undefined
    ):
        # The comment ends the undefined byte with a line end, which no double
        # byte code page takes as a second byte either.
        path = work_dir / "undefined.asc"
        path.write_bytes(_cp1252(_sheet("à" + undefined, "1µ", comment="x" + undefined)))
        await _assert_both_read(asc_state, path, "1µ", "à" + undefined)
        await _edit_r2_keeps_every_other_byte(asc_state, path)

    async def test_a_sheet_with_more_than_ascii_says_its_reading_is_uncertain(
        self, asc_state, work_dir
    ):
        path = work_dir / "accent.asc"
        path.write_bytes(_cp1252(_sheet("à", "1µ")))
        data = await _edit_r2_keeps_every_other_byte(asc_state, path)
        notes = [one for one in data.get("observations", []) if "Windows-1252" in one]
        assert len(notes) == 1, data.get("observations")
        assert "code page" in notes[0]

    async def test_an_ascii_sheet_has_no_reading_to_doubt(self, asc_state, work_dir):
        path = work_dir / "plain.asc"
        path.write_bytes(_sheet("IN", "10").encode("ascii"))
        data = await _edit_r2_keeps_every_other_byte(asc_state, path)
        assert not [one for one in data.get("observations", []) if "Windows-1252" in one]

    async def test_a_cp936_comment_keeps_its_bytes(self, asc_state, work_dir):
        # 丂 is 0x81 0x40 in cp936: a lead byte cp1252 leaves undefined.
        comment = "注释丂".encode("cp936")
        assert b"\x81" in comment
        path = work_dir / "cp936.asc"
        path.write_bytes(_sheet("IN", "10", comment="@@").encode("ascii").replace(b"@@", comment))
        await _edit_r2_keeps_every_other_byte(asc_state, path)
        assert comment in path.read_bytes()

    async def test_a_character_the_sheets_encoding_lacks_is_refused_naming_it(
        self, asc_state, work_dir
    ):
        path = work_dir / "omega.asc"
        original = _cp1252(_sheet("à", "1µ"))
        path.write_bytes(original)
        data = await _edit(
            asc_state, path, [{"op": "set_component_value", "reference": "R2", "value": "1Ω"}]
        )
        assert data["commit_state"] == "not_committed", data
        assert "Windows-1252, which has no 'Ω'" in data["error"]["message"]
        assert path.read_bytes() == original

    async def test_a_symbol_read_in_the_codec_the_sheet_is_read_in(self, asc_state, work_dir):
        # A symbol holding 0x8D and 0x81, each before a line end, which
        # spicelib's detection refused, no longer stops an edit of a sheet placing it.
        (work_dir / "part.asy").write_bytes(
            b"Version 4\nSymbolType CELL\nRECTANGLE Normal 0 0 32 64\n"
            b"SYMATTR Description x\x8d\n"
            b"SYMATTR SpiceLine y\x81\n"
            b"PIN 16 0 NONE 0\nPINATTR PinName A\nPINATTR SpiceOrder 1\n"
            b"PIN 16 64 NONE 0\nPINATTR PinName B\nPINATTR SpiceOrder 2\n"
        )
        path = work_dir / "placed.asc"
        path.write_bytes(
            _sheet("IN", "10").replace("SYMBOL res 160 0", "SYMBOL part 160 0").encode("ascii")
        )
        data = await _edit(asc_state, path, [{"op": "add_net_label", "net": "P", "pin": "R2.2"}])
        assert data["commit_state"] == "committed", data
        assert b"FLAG 176 64 P" in path.read_bytes()

    async def test_crlf_line_ends_stay_crlf(self, asc_state, work_dir):
        path = work_dir / "crlf.asc"
        path.write_bytes(_cp1252(_sheet("à", "1µ")).replace(b"\n", b"\r\n"))
        await _assert_both_read(asc_state, path, "1µ", "à")
        await _edit_r2_keeps_every_other_byte(asc_state, path, newline=b"\r\n")
        assert b"\n" not in path.read_bytes().replace(b"\r\n", b"")


class TestUnicodeSheet:
    async def test_utf8_without_a_mark_is_read_alike_and_written_back_as_it_was(
        self, asc_state, work_dir
    ):
        path = work_dir / "utf8.asc"
        path.write_bytes(_sheet("à", "1µ").encode("utf-8"))
        await _assert_both_read(asc_state, path, "1µ", "à")
        await _edit_r2_keeps_every_other_byte(asc_state, path)

    async def test_utf16_le_with_a_mark_is_read_alike_and_written_without_it(
        self, asc_state, work_dir
    ):
        # Neither LTspice build reads a sheet behind the mark, and both read
        # UTF-16 LE without it (export/micro_utf16le_bom, export/micro_utf16le),
        # so an edit keeps the encoding and every character and drops the mark.
        text = _sheet("à", "1µ")
        path = work_dir / "utf16.asc"
        path.write_bytes(b"\xff\xfe" + text.encode("utf-16-le"))
        await _assert_both_read(asc_state, path, "1µ", "à")
        data = await _edit(
            asc_state, path, [{"op": "set_component_value", "reference": "R2", "value": "2k"}]
        )
        assert data["commit_state"] == "committed", data
        expected = text.replace("SYMATTR Value 1k", "SYMATTR Value 2k").encode("utf-16-le")
        assert path.read_bytes() == expected

    async def test_utf16_le_without_a_mark_is_read_alike_and_written_back_as_it_was(
        self, asc_state, work_dir
    ):
        text = _sheet("à", "1µ")
        path = work_dir / "utf16.asc"
        path.write_bytes(text.encode("utf-16-le"))
        await _assert_both_read(asc_state, path, "1µ", "à")
        data = await _edit(
            asc_state, path, [{"op": "set_component_value", "reference": "R2", "value": "2k"}]
        )
        assert data["commit_state"] == "committed", data
        expected = text.replace("SYMATTR Value 1k", "SYMATTR Value 2k").encode("utf-16-le")
        assert path.read_bytes() == expected

    async def test_utf8_with_a_mark_is_read_by_inspect_and_refused_by_an_edit_naming_it(
        self, asc_state, work_dir
    ):
        # Neither LTspice build reads a sheet behind a UTF-8 mark
        # (export/micro_utf8_bom), so an edit refuses it and says why rather
        # than write a sheet LTspice will not open.
        path = work_dir / "utf8bom.asc"
        original = b"\xef\xbb\xbf" + _sheet("à", "1µ").encode("utf-8")
        path.write_bytes(original)
        components, net = await _inspect(asc_state, path)
        assert components["components"][0]["value"] == "1µ"
        assert net["labels"] == ["à"]
        with pytest.raises(NetlistError, match="UTF-8 byte order mark"):
            await _edit(
                asc_state, path, [{"op": "set_component_value", "reference": "R2", "value": "2k"}]
            )
        assert path.read_bytes() == original


class TestReadUnderTheLock:
    async def test_an_edit_reads_the_encoding_and_line_ends_a_peer_wrote_under_the_lock(
        self, asc_state, work_dir, monkeypatch
    ):
        # The session reads the sheet as UTF-8 with LF ends; a peer holding the
        # file's lock rewrites it as 8-bit text with CRLF ends. The edit takes
        # the lock before it reads the sheet, so it writes the peer's bytes back.
        path = work_dir / "peer.asc"
        path.write_bytes(_sheet("à", "1µ").encode("utf-8"))
        await _assert_both_read(asc_state, path, "1µ", "à")
        peer = _cp1252(_sheet("à", "1µ")).replace(b"\n", b"\r\n")
        writer = _hold_lock_then_write(path, peer, monkeypatch)
        data = await _edit(
            asc_state,
            path,
            [{"op": "set_component_value", "reference": "R2", "value": "2k"}],
            expected_sha256=hashlib.sha256(peer).hexdigest(),
        )
        writer.join(LIVENESS_S)
        assert data["commit_state"] == "committed", data
        old, new = b"SYMATTR Value 1k\r\n", b"SYMATTR Value 2k\r\n"
        assert path.read_bytes() == peer.replace(old, new)
