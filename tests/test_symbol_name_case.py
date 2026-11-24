"""A file a sheet names in another letter case than the file's own.

A sheet written on Windows names ``amp`` for ``Amp.asy`` and ``digital\\and``
for ``Digital/and.asy``, and Windows finds the file. ``symbol_library`` finds
it on every platform, matching names without regard to case, and every reader
of a sheet asks it: ``inspect`` for a part's pins, the schematic editor for a
part's symbol and a block's own sheet, and the staging of a sheet's folder for
its export.

Where one folder holds two files whose names differ only in case, the one
spelled as the sheet spells it is taken; where neither is, the name is
ambiguous and the reply names both files. Only a file system that keeps such
files apart can hold two of them, so those tests skip on one that does not,
and say so in their names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import SymbolResolutionError
from ltspice_mcp.lib.deck_prep import stage_sheet
from ltspice_mcp.state import SessionState
from tests._asc_ops import apply_ops, components_of, load_editor


def write_file(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _symbol(pin_y: int, kind: str = "CELL") -> str:
    """A two-pin symbol whose second pin is at ``pin_y``, which tells files apart."""
    return (
        f"Version 4\nSymbolType {kind}\nRECTANGLE Normal 0 0 32 {pin_y}\n"
        "PIN 16 0 NONE 0\nPINATTR PinName A\nPINATTR SpiceOrder 1\n"
        f"PIN 16 {pin_y} NONE 0\nPINATTR PinName B\nPINATTR SpiceOrder 2\n"
    )


def _sheet_placing(symbol: str, reference: str = "U1") -> str:
    return (
        "Version 4\nSHEET 1 880 680\n"
        f"SYMBOL {symbol} 0 0 R0\nSYMATTR InstName {reference}\nSYMATTR Value 1\n"
    )


#: A block's own sheet: two ports, as its symbol's pins name them.
_BLOCK_SHEET = (
    "Version 4\nSHEET 1 880 680\n"
    "FLAG 0 0 A\nIOPIN 0 0 In\nFLAG 0 96 B\nIOPIN 0 96 Out\n"
    "WIRE 0 0 0 96\n"
)


def _case_sensitive(folder: Path) -> bool:
    probe = folder / "case-probe"
    probe.write_text("", encoding="utf-8", newline="\n")
    try:
        return not (folder / "CASE-PROBE").exists()
    finally:
        probe.unlink()


@pytest.fixture
def case_sensitive(work_dir: Path) -> None:
    """Skip a test whose two files differ only in case where they cannot both be kept."""
    if not _case_sensitive(work_dir):
        pytest.skip("this file system keeps no two names that differ only in case")


def _pin_ys(row: dict[str, Any]) -> list[int]:
    return sorted(pin["y"] for pin in row["pins"])


async def _row(state: SessionState, sheet: Path, reference: str = "U1") -> dict[str, Any]:
    data = await components_of(state, sheet)
    return next(one for one in data["components"] if one["reference"] == reference)


async def _set_value(state: SessionState, sheet: Path) -> dict[str, Any]:
    result = await apply_ops(
        state,
        sheet.name,
        [{"op": "set_component_value", "reference": "U1", "value": "2"}],
    )
    assert result["outcome"] == "complete", result
    return result


class TestAnyFileSystem:
    """What a file system that ignores case finds too: one file, named in another case."""

    async def test_a_symbol_beside_the_sheet_named_in_another_case(self, asc_state, work_dir):
        write_file(work_dir / "Amp.asy", _symbol(64))
        sheet = write_file(work_dir / "amp.asc", _sheet_placing("amp"))
        row = await _row(asc_state, sheet)
        assert _pin_ys(row) == [0, 64]
        assert "symbol_problem" not in row
        await _set_value(asc_state, sheet)

    async def test_a_library_symbol_named_in_another_case(self, asc_state, work_dir):
        # The suite's library (``config``) holds ``res.asy``.
        sheet = write_file(work_dir / "upper.asc", _sheet_placing("RES", "R1"))
        row = await _row(asc_state, sheet, "R1")
        assert len(row["pins"]) == 2
        assert "symbol_problem" not in row

    @pytest.mark.parametrize("named", ["digital\\\\AND", "digital/AND"], ids=["backslash", "slash"])
    async def test_a_symbol_in_a_folder_named_in_another_case(self, asc_state, work_dir, named):
        (work_dir / "Digital").mkdir()
        write_file(work_dir / "Digital" / "and.asy", _symbol(80))
        sheet = write_file(work_dir / "gate.asc", _sheet_placing(named))
        row = await _row(asc_state, sheet)
        assert _pin_ys(row) == [0, 80]
        await _set_value(asc_state, sheet)

    async def test_a_block_sheet_named_in_another_case_opens_with_the_block(
        self, asc_state, work_dir
    ):
        write_file(work_dir / "sub.asy", _symbol(96, "BLOCK"))
        write_file(work_dir / "Sub.asc", _BLOCK_SHEET)
        top = write_file(work_dir / "top.asc", _sheet_placing("sub", "X1"))
        block = load_editor(asc_state, top).get_subcircuit("X1")
        assert block is not None
        assert Path(block.asc_file_path).name.casefold() == "sub.asc"

    async def test_a_part_whose_symbol_is_not_found_says_so(self, asc_state, work_dir):
        sheet = write_file(work_dir / "missing.asc", _sheet_placing("nowhere"))
        row = await _row(asc_state, sheet)
        assert "pins" not in row
        assert row["symbol_problem"] == "not found"


@pytest.mark.usefixtures("case_sensitive")
class TestCaseSensitiveFileSystem:
    """Two files in one folder whose names differ only in case."""

    async def test_case_sensitive_fs_the_symbol_spelled_as_the_sheet_spells_it_wins(
        self, asc_state, work_dir
    ):
        write_file(work_dir / "Amp.asy", _symbol(64))
        write_file(work_dir / "amp.asy", _symbol(32))
        sheet = write_file(work_dir / "two.asc", _sheet_placing("amp"))
        row = await _row(asc_state, sheet)
        assert _pin_ys(row) == [0, 32]
        await _set_value(asc_state, sheet)

    async def test_case_sensitive_fs_a_symbol_neither_file_spells_is_ambiguous(
        self, asc_state, work_dir
    ):
        write_file(work_dir / "Amp.asy", _symbol(64))
        write_file(work_dir / "AMP.asy", _symbol(32))
        sheet = write_file(work_dir / "two.asc", _sheet_placing("amp"))
        original = sheet.read_bytes()
        row = await _row(asc_state, sheet)
        assert "pins" not in row
        assert "AMP.asy" in row["symbol_problem"]
        assert "Amp.asy" in row["symbol_problem"]
        with pytest.raises(SymbolResolutionError, match=r"AMP\.asy, Amp\.asy"):
            await _set_value(asc_state, sheet)
        assert sheet.read_bytes() == original

    async def test_case_sensitive_fs_a_symbol_in_a_folder_two_folders_could_be(
        self, asc_state, work_dir
    ):
        for folder, pin_y in (("Digital", 80), ("DIGITAL", 48)):
            (work_dir / folder).mkdir()
            write_file(work_dir / folder / "and.asy", _symbol(pin_y))
        sheet = write_file(work_dir / "gate.asc", _sheet_placing("digital\\\\and"))
        row = await _row(asc_state, sheet)
        assert "DIGITAL" in row["symbol_problem"]
        assert "Digital" in row["symbol_problem"]
        exact = write_file(work_dir / "exact.asc", _sheet_placing("Digital\\\\and"))
        assert _pin_ys(await _row(asc_state, exact)) == [0, 80]

    async def test_case_sensitive_fs_a_block_sheet_neither_file_spells_is_ambiguous(
        self, asc_state, work_dir
    ):
        write_file(work_dir / "sub.asy", _symbol(96, "BLOCK"))
        write_file(work_dir / "Sub.asc", _BLOCK_SHEET)
        write_file(work_dir / "SUB.asc", _BLOCK_SHEET)
        top = write_file(work_dir / "top.asc", _sheet_placing("sub", "X1"))
        with pytest.raises(SymbolResolutionError, match=r"SUB\.asc, Sub\.asc"):
            load_editor(asc_state, top)

    def test_case_sensitive_fs_staging_takes_the_block_sheet_the_sheet_spells(self, work_dir):
        # The block's symbol is in a library, so its sheet is read beside the
        # sheet exported: staging copies what the lookup would take there.
        origin = work_dir / "origin"
        origin.mkdir()
        write_file(origin / "sub.asc", _BLOCK_SHEET)
        write_file(origin / "Sub.asc", _BLOCK_SHEET)
        top = write_file(origin / "top.asc", _sheet_placing("sub", "X1"))
        staged, _note = stage_sheet(top, work_dir / "staged")
        assert sorted(path.name for path in staged.parent.iterdir()) == ["sub.asc", "top.asc"]

    def test_case_sensitive_fs_staging_takes_every_block_sheet_a_name_could_mean(
        self, work_dir
    ):
        origin = work_dir / "origin"
        origin.mkdir()
        write_file(origin / "Sub.asc", _BLOCK_SHEET)
        write_file(origin / "SUB.asc", _BLOCK_SHEET)
        top = write_file(origin / "top.asc", _sheet_placing("sub", "X1"))
        staged, _note = stage_sheet(top, work_dir / "staged")
        assert sorted(path.name for path in staged.parent.iterdir()) == [
            "SUB.asc",
            "Sub.asc",
            "top.asc",
        ]
