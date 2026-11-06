"""Which file a name on a sheet means, on folders made for the purpose: a
part's symbol and a block's own sheet; a model library its symbol names is
looked for by no tool.

What LTspice does with the same arrangements is in
``test_recorded_ltspice_schematics.py``; this holds the rule itself, and that
every tool reading a sheet asks it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.symbol_library import (
    SymbolLibrary,
    SymbolResolver,
    find_beside,
    find_in_library,
    find_symbol,
    spellings,
)
from ltspice_mcp.state import SessionState
from tests._asc_ops import apply_ops, inspect_one, load_editor
from tests._schematic_fixtures import write_file
from tests.test_verify_circuit import _run as run_verify_circuit


def keep(folder: Path, *names: str) -> None:
    for name in names:
        path = folder / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("Version 4\n", encoding="utf-8")


class TestSpellings:
    @pytest.mark.parametrize(
        ("symbol", "expected"),
        [
            ("res", ("res", "res")),
            ("Opamps\\\\LT1001", ("Opamps/LT1001", "LT1001")),
            ("Opamps\\LT1001", ("Opamps/LT1001", "LT1001")),
            ("Opamps/LT1001", ("Opamps/LT1001", "LT1001")),
            ("a\\\\b\\\\c", ("a/b/c", "c")),
            ("", ("", "")),
        ],
    )
    def test_a_name_as_a_path_and_as_its_bare_name(
        self, symbol: str, expected: tuple[str, str]
    ) -> None:
        assert spellings(symbol) == expected


class TestBesideASheet:
    def test_a_bare_name_is_found_right_there(self, tmp_path: Path) -> None:
        keep(tmp_path, "part.asy")
        assert find_beside(tmp_path, "part") == tmp_path / "part.asy"

    def test_a_bare_name_is_not_looked_for_in_a_folder(self, tmp_path: Path) -> None:
        keep(tmp_path, "lib/part.asy")
        assert find_beside(tmp_path, "part") is None

    def test_a_name_that_says_a_folder_is_found_in_it(self, tmp_path: Path) -> None:
        keep(tmp_path, "lib/part.asy")
        assert find_beside(tmp_path, "lib\\\\part") == tmp_path / "lib" / "part.asy"

    def test_or_right_beside_the_sheet_under_its_bare_name(self, tmp_path: Path) -> None:
        keep(tmp_path, "part.asy")
        assert find_beside(tmp_path, "lib\\\\part") == tmp_path / "part.asy"

    def test_the_folder_it_says_comes_first(self, tmp_path: Path) -> None:
        keep(tmp_path, "lib/part.asy", "part.asy")
        assert find_beside(tmp_path, "lib\\\\part") == tmp_path / "lib" / "part.asy"

    def test_it_is_not_looked_for_in_another_folder(self, tmp_path: Path) -> None:
        keep(tmp_path, "other/part.asy")
        assert find_beside(tmp_path, "lib\\\\part") is None

    def test_a_name_that_is_empty_names_nothing(self, tmp_path: Path) -> None:
        keep(tmp_path, ".asy")
        assert find_beside(tmp_path, "") is None


class TestInALibrary:
    def test_a_bare_name_is_found_in_any_folder(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/battery.asy")
        assert find_in_library(tmp_path, "battery") == tmp_path / "Misc" / "battery.asy"

    def test_a_name_that_says_the_folder(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/battery.asy")
        assert find_in_library(tmp_path, "Misc\\\\battery") == tmp_path / "Misc" / "battery.asy"

    def test_a_name_that_says_another_folder(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/battery.asy")
        assert find_in_library(tmp_path, "Wrong\\\\battery") == tmp_path / "Misc" / "battery.asy"

    def test_the_top_of_the_library_comes_before_its_folders(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/res.asy", "res.asy")
        assert find_in_library(tmp_path, "res") == tmp_path / "res.asy"

    def test_a_name_with_brackets_is_a_name_and_not_a_pattern(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/a.asy", "Misc/[ab].asy")
        assert find_in_library(tmp_path, "[ab]") == tmp_path / "Misc" / "[ab].asy"

    def test_a_symbol_the_library_does_not_have(self, tmp_path: Path) -> None:
        keep(tmp_path, "Misc/battery.asy")
        assert find_in_library(tmp_path, "cell") is None
        assert find_in_library(tmp_path, "") is None


class TestSheetThenLibraries:
    def test_a_symbol_beside_the_sheet_wins(self, tmp_path: Path) -> None:
        sheet, library = tmp_path / "sheet", tmp_path / "library"
        keep(sheet, "res.asy")
        keep(library, "res.asy")
        assert find_symbol("res", sheet, [library]) == sheet / "res.asy"

    def test_libraries_are_tried_in_the_order_given(self, tmp_path: Path) -> None:
        first, second = tmp_path / "first", tmp_path / "second"
        keep(first, "Misc/res.asy")
        keep(second, "res.asy")
        assert find_symbol("res", None, [first, second]) == first / "Misc" / "res.asy"
        assert find_symbol("res", None, [second, first]) == second / "res.asy"

    def test_the_sheets_folder_is_never_searched_as_a_library(self, tmp_path: Path) -> None:
        """Named as a library too, it would be walked for a bare name."""
        keep(tmp_path, "lib/part.asy")
        assert find_symbol("part", tmp_path, [tmp_path]) is None

    def test_a_folder_that_is_not_there_is_passed_over(self, tmp_path: Path) -> None:
        library = tmp_path / "library"
        keep(library, "res.asy")
        missing = tmp_path / "missing"
        assert find_symbol("res", missing, [missing, library]) == library / "res.asy"

    def test_nowhere(self, tmp_path: Path) -> None:
        assert find_symbol("res", tmp_path, [tmp_path / "library"]) is None
        assert find_symbol("res", None, []) is None


# ---------------------------------------------------------------------------
# A block's own sheet, and the names a listing offers
# ---------------------------------------------------------------------------


def _resolver(sheet_folder: Path, *libraries: Path) -> SymbolResolver:
    """The session's resolver for a sheet in ``sheet_folder``, with ``libraries``
    configured."""
    return SymbolLibrary(libraries).for_sheet(sheet_folder / "top.asc")


class TestABlocksSheet:
    """Where a hierarchical block's own sheet is found. Both builds read it
    beside the block's symbol (``export-hierarchical-block``). Anywhere else is
    not recorded, so it is then found as a symbol of the block's name is, with
    ``.asc`` for ``.asy``: beside the sheet that places it, then in each
    library. Never in the server's working directory."""

    def test_beside_its_symbol(self, tmp_path: Path) -> None:
        keep(tmp_path / "library", "blocks/blk.asy", "blocks/blk.asc")
        (tmp_path / "sheet").mkdir()
        found = _resolver(tmp_path / "sheet", tmp_path / "library").block_sheet("blk")
        assert found == tmp_path / "library" / "blocks" / "blk.asc"

    def test_beside_the_sheet_that_places_it(self, tmp_path: Path) -> None:
        keep(tmp_path / "library", "blk.asy")
        keep(tmp_path / "sheet", "blk.asc")
        found = _resolver(tmp_path / "sheet", tmp_path / "library").block_sheet("blk")
        assert found == tmp_path / "sheet" / "blk.asc"

    def test_not_in_a_folder_below_the_sheets(self, tmp_path: Path) -> None:
        keep(tmp_path / "sheet", "blk.asy", "blocks/blk.asc")
        assert _resolver(tmp_path / "sheet").block_sheet("blk") is None

    def test_a_block_with_no_symbol_has_no_sheet(self, tmp_path: Path) -> None:
        keep(tmp_path / "sheet", "blk.asc")
        assert _resolver(tmp_path / "sheet").block_sheet("blk") is None

    def test_never_in_the_working_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        keep(tmp_path / "sheet", "blk.asy")
        keep(tmp_path / "elsewhere", "blk.asc", "blocks/blk.asc")
        monkeypatch.chdir(tmp_path / "elsewhere")
        assert _resolver(tmp_path / "sheet").block_sheet("blk") is None


class TestTheListing:
    """``SymbolResolver.listing`` is what ``inspect(kind="symbols")`` offers:
    every name it lists resolves, and a bare name it does not list does not."""

    @staticmethod
    def listed(resolver: SymbolResolver) -> set[str]:
        return {path.stem for _folder, _tier, files in resolver.listing() for path in files}

    def test_the_sheets_folder_offers_what_is_right_there(self, tmp_path: Path) -> None:
        keep(tmp_path / "sheet", "here.asy", "lib/below.asy")
        resolver = _resolver(tmp_path / "sheet")
        assert self.listed(resolver) == {"here"}
        assert resolver.resolve("below") is None

    def test_a_library_offers_everything_under_it(self, tmp_path: Path) -> None:
        keep(tmp_path / "library", "top.asy", "Misc/battery.asy")
        (tmp_path / "sheet").mkdir()
        resolver = _resolver(tmp_path / "sheet", tmp_path / "library")
        assert self.listed(resolver) == {"top", "battery"}
        assert resolver.resolve("battery") == tmp_path / "library" / "Misc" / "battery.asy"

    def test_the_sheets_folder_named_as_a_library_another_way_is_still_the_sheets(
        self, tmp_path: Path
    ) -> None:
        """Named as a library too, the sheet's folder would be walked for a bare
        name, which neither build does (``search/in_subfolder``), however the
        folder is spelled."""
        keep(tmp_path / "sheet", "lib/part.asy")
        (tmp_path / "other").mkdir()
        resolver = _resolver(tmp_path / "sheet", tmp_path / "other" / ".." / "sheet")
        assert resolver.resolve("part") is None
        assert self.listed(resolver) == set()
        assert [tier for _folder, tier, _files in resolver.listing()] == ["schematic-local"]


# ---------------------------------------------------------------------------
# Every reader of a sheet finds its files alike, wherever the server runs
# ---------------------------------------------------------------------------

#: A subcircuit part whose symbol names its model library, as a vendor's does.
_AMP_ASY = (
    "Version 4\nSymbolType CELL\nRECTANGLE Normal -32 -32 32 32\nSYMATTR Prefix X\n"
    "SYMATTR Value amp\nSYMATTR ModelFile amp.lib\n"
    "PIN -32 0 NONE 0\nPINATTR PinName a\nPINATTR SpiceOrder 1\n"
    "PIN 32 0 NONE 0\nPINATTR PinName b\nPINATTR SpiceOrder 2\n"
)
_AMP_SHEET = "Version 4\nSHEET 1 880 680\nSYMBOL amp 96 96 R0\nSYMATTR InstName X1\n"


def _amp_library(resistance: str) -> str:
    return f"* amp\n.subckt amp a b\nR1 a b {resistance}\n.ends amp\n"


@pytest.mark.parametrize("library", ["beside the sheet", "nowhere"])
@pytest.mark.parametrize("cwd", ["empty", "holding a library of that name"])
async def test_a_part_naming_a_model_library_opens_alike_whatever_the_working_directory(
    library: str,
    cwd: str,
    config: ServerConfig,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``edit_schematic`` opens and edits the sheet wherever the part's library
    is, as ``inspect`` and ``verify_circuit`` read it. A library is read when
    the deck runs, from the ``.lib`` line the export writes for the part, so no
    reader of the sheet looks for one. spicelib's editor looked for it beside
    the sheet, in every folder below it and in the folder the server runs in:
    it refused the sheet where it found none, and where it found one, saving
    the sheet failed on the subcircuit it had read (``docs/spicelib_bugs.md``,
    Bugs 37 and 39). So whether the sheet could be edited turned on where the
    server was started."""
    sheet = write_file(work_dir / "sheet" / "top.asc", _AMP_SHEET)
    write_file(sheet.parent / "amp.asy", _AMP_ASY)
    if library == "beside the sheet":
        write_file(sheet.parent / "amp.lib", _amp_library("1k"))
    elsewhere = work_dir / "elsewhere"
    elsewhere.mkdir()
    if cwd != "empty":
        write_file(elsewhere / "amp.lib", _amp_library("2k"))
    monkeypatch.chdir(elsewhere)
    state = SessionState.create(config, available={})

    edit = [{"op": "add_directive", "instruction": ".op", "x": 96, "y": 400}]
    data = await apply_ops(state, "sheet/top.asc", edit)
    assert data["commit_state"] == "committed", data.get("error") or data.get("failures")
    assert "!.op" in sheet.read_text(encoding="utf-8")
    assert load_editor(state, sheet).get_component("X1").attributes["_SUBCKT"] is None

    symbol = await inspect_one(state, {"kind": "symbol", "name": "amp", "path": str(sheet)})
    assert Path(symbol["source_path"]).parts[-2:] == ("sheet", "amp.asy")
    assert symbol["attributes"]["ModelFile"] == "amp.lib"
    listed = await inspect_one(state, {"kind": "symbols", "path": str(sheet), "filter": "amp"})
    assert listed["symbols"] == ["amp"]
    checked = await run_verify_circuit(state, path=str(sheet), checks=["symbols"])
    assert checked["rules_run"]["unresolved_symbol"] == 0, checked
