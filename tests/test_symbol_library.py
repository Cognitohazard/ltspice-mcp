"""Which file a symbol's name means, on folders made for the purpose.

What LTspice does with the same arrangements is in
``test_recorded_ltspice_schematics.py``; this holds the rule itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.lib.symbol_library import (
    find_beside,
    find_in_library,
    find_symbol,
    spellings,
)


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
