"""The record of what every ``edit_schematic`` call answers (``tests/sheet_edit_record.py``).

Its comparison of sheets must not tell apart the ways two engines write one
sheet (record order across kinds, line endings, an attribute line's place, a
directive's trailing spaces), and must tell apart any change to a record. A
call on two spellings of one sheet must have one key, and a call answering
differently from the record must be found.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.schematic_edit import EditSchematicInput
from tests import sheet_edit_record
from tests._asc_ops import build_sheet
from tests.sheet_edit_record import key_of, sheet_form

_SHEET = [
    "Version 4",
    "SHEET 1 880 680",
    "WIRE 16 96 112 96",
    "FLAG 16 96 in",
    "SYMBOL res 112 48 R0",
    "SYMATTR InstName R1",
    "SYMATTR Value 1k",
    "TEXT 16 200 Left 2 !.op",
]


def _bytes(lines: list[str], newline: str = "\n", encoding: str = "ascii") -> bytes:
    return "".join(line + newline for line in lines).encode(encoding)


class TestTheSheetComparison:
    def test_reads_two_writings_of_one_sheet_as_one(self):
        regrouped = [_SHEET[0], _SHEET[1], *_SHEET[4:7], _SHEET[2], _SHEET[3], _SHEET[7]]
        attributes_swapped = [*_SHEET[:5], _SHEET[6], _SHEET[5], _SHEET[7]]
        trailing_space = [*_SHEET[:7], _SHEET[7] + "  "]
        first = sheet_form(_bytes(_SHEET))
        assert sheet_form(_bytes(_SHEET, "\r\n")) == first
        assert sheet_form(_bytes(regrouped)) == first
        assert sheet_form(_bytes(attributes_swapped)) == first
        assert sheet_form(_bytes(trailing_space)) == first

    @pytest.mark.parametrize(
        ("index", "line"),
        [
            (2, "WIRE 16 96 128 96"),
            (3, "FLAG 16 96 out"),
            (4, "SYMBOL res 112 48 R90"),
            (6, "SYMATTR Value 2k"),
            (7, "TEXT 16 200 Left 2 !.tran 1m"),
        ],
    )
    def test_tells_any_changed_record_apart(self, index: int, line: str):
        changed = list(_SHEET)
        changed[index] = line
        assert sheet_form(_bytes(changed)) != sheet_form(_bytes(_SHEET))

    def test_tells_records_in_another_order_within_a_kind_apart(self):
        two_wires = [*_SHEET[:3], "WIRE 0 0 16 0", *_SHEET[3:]]
        swapped = [*_SHEET[:2], "WIRE 0 0 16 0", _SHEET[2], *_SHEET[3:]]
        assert sheet_form(_bytes(two_wires)) != sheet_form(_bytes(swapped))

    def test_tells_the_codec_apart(self):
        micro = [*_SHEET[:6], "SYMATTR Value 1µ"]
        assert sheet_form(_bytes(micro, encoding="cp1252")) != sheet_form(
            _bytes(micro, encoding="utf-8")
        )


def test_a_call_on_two_writings_of_one_sheet_has_one_key(tmp_path: Path):
    """Each handed the digest of its own sheet, as a caller does."""
    keys = []
    for name, newline in (("lf", "\n"), ("crlf", "\r\n")):
        folder = tmp_path / name
        folder.mkdir()
        target = folder / "sheet.asc"
        target.write_bytes(_bytes(_SHEET, newline))
        args = EditSchematicInput.model_validate(
            {
                "target": str(target),
                "expected_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "ops": [{"op": "add_directive", "instruction": ".tran 1m"}],
            }
        )
        keys.append(key_of(target, target.read_bytes(), args, (folder,)))
    assert keys[0] == keys[1]


class _AnswersOtherwise(dict[str, Any]):
    """A record that holds every key, each answered otherwise."""

    def get(self, key: str, default: Any = None) -> Any:
        return {"sheet": "recorded", "reply": "recorded"}


async def test_a_call_answering_differently_from_the_record_is_found(
    asc_state: SessionState, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(sheet_edit_record, "_recording", False)
    monkeypatch.setattr(sheet_edit_record, "_record", _AnswersOtherwise())
    await build_sheet(asc_state, "found", [{"op": "add_directive", "instruction": ".op"}])
    found = sheet_edit_record.take_mismatches()
    assert len(found) == 1
    assert found[0].startswith("call ")
