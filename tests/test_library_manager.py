"""Unit tests for LibraryManager and the model ranking and row every model
lookup shares — no simulator.

A detected simulator is stood in for by a class reporting its library
directories; what the install reports is the environment, and which files the
server reads because of it is what these tests pin. Ranking and rows are
exercised over real parses of real files.
"""

from pathlib import Path

import pytest

from ltspice_mcp.lib.library_manager import LibraryManager, model_row, rank_models
from ltspice_mcp.lib.library_parser import parse_library_file
from tests.conftest import installed_simulator


@pytest.fixture
def empty_manager() -> LibraryManager:
    return LibraryManager(available_simulators={})


def _install(root: Path, files: dict[str, str]) -> LibraryManager:
    """Write ``files`` under ``root`` and return a manager whose detected
    simulator ships ``root`` as its library."""
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return LibraryManager(available_simulators={"sim": installed_simulator(root)})


def _search(manager: LibraryManager, query: str) -> list[dict]:
    rows, _revisions = manager.search(query)
    return rows


def _rank(tmp_path: Path, text: str, query: str, cutoff: float = 0.6) -> list[dict]:
    lib = tmp_path / "parts.lib"
    lib.write_text(text)
    return rank_models([parse_library_file(lib)], query, cutoff=cutoff)


_FUZZY = (
    ".MODEL 2N3904 NPN(BF=200)\n"
    ".MODEL 2N3906 PNP(BF=200)\n"
    ".MODEL 2N2222 NPN(BF=300)\n"
    ".MODEL D1N4148 D(IS=2.52e-9)\n"
    ".SUBCKT LM741 in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
)


class TestSearch:
    """``LibraryManager.search``: every library file the simulator ships."""

    def test_finds_a_near_miss_in_the_simulators_library(self, tmp_path: Path):
        manager = _install(tmp_path / "lib", {"cmp/standard.bjt": _FUZZY})
        names = [r["name"] for r in _search(manager, "2N3905")]
        assert names[:2] == ["2N3904", "2N3906"]

    def test_without_a_simulator_library_finds_nothing(self, empty_manager: LibraryManager):
        assert empty_manager.search("2N3904") == ([], [])

    def test_sub_subcircuit_libraries_are_searched(self, tmp_path: Path):
        # ``.sub`` files hold the bulk of LTspice's bundled vendor subcircuit
        # models, so a SUBCKT defined in one is found like one in a ``.lib``.
        manager = _install(
            tmp_path / "lib", {"sub/vendor.sub": ".SUBCKT MYPART in out\nR1 in out 1k\n.ENDS\n"}
        )
        (row,) = _search(manager, "MYPART")
        assert row["type"] == ".SUBCKT"

    def test_stock_component_decks_are_searched(self, tmp_path: Path):
        # LTspice ships its stock component models as ``.bjt`` / ``.mos`` /
        # ``.dio`` / ``.jft`` decks under lib/cmp, none of them named ``.lib``.
        manager = _install(
            tmp_path / "lib",
            {
                "cmp/standard.bjt": ".MODEL QSTD NPN(BF=100)\n",
                "cmp/standard.mos": ".MODEL MSTD NMOS(KP=2e-5)\n",
            },
        )
        assert _search(manager, "QSTD")[0]["source_path"].endswith("standard.bjt")
        assert _search(manager, "MSTD")[0]["source_path"].endswith("standard.mos")

    def test_revisions_name_every_file_searched(self, tmp_path: Path):
        """A paged caller binds its cursor to these, so they must cover every
        file the search read, matched or not, with the stamp it was read at."""
        manager = _install(
            tmp_path / "lib",
            {"cmp/standard.bjt": _FUZZY, "sub/LT1001.sub": ".SUBCKT LT1001 a b\n.ENDS\n"},
        )
        _rows, revisions = manager.search("2N3904")
        files = manager.builtin_library_files()
        assert [path for path, _stamp in revisions] == [str(path) for path in files]
        for (_path, stamp), path in zip(revisions, files, strict=True):
            stat = path.stat()
            assert stamp == (stat.st_mtime_ns, stat.st_size)

    def test_an_encrypted_vendor_part_is_found_by_name(self, tmp_path: Path):
        # An encrypted library carries no plaintext card, so the part is
        # reported by name rather than vanishing from the search.
        manager = _install(
            tmp_path / "lib",
            {"sub/PART_A_enc.lib": "* LTspice Encrypted File\n* Begin:\n 05 AC A3 C2\n"},
        )
        (row,) = _search(manager, "PART_A_enc")
        assert row["type"] == ".ENCRYPTED"


class TestRankModels:
    def test_typo_finds_the_siblings(self, tmp_path: Path):
        rows = _rank(tmp_path, _FUZZY, "2N3905")
        assert {r["name"] for r in rows[:2]} == {"2N3904", "2N3906"}
        assert all(0.0 <= r["score"] <= 1.0 for r in rows)

    def test_ranked_by_score(self, tmp_path: Path):
        scores = [r["score"] for r in _rank(tmp_path, _FUZZY, "2N3905", cutoff=0.0)]
        assert scores == sorted(scores, reverse=True)

    def test_case_insensitive(self, tmp_path: Path):
        assert any(r["name"] == "LM741" for r in _rank(tmp_path, _FUZZY, "lm741"))

    def test_cutoff_filters(self, tmp_path: Path):
        assert _rank(tmp_path, _FUZZY, "XYZZY", cutoff=0.9) == []
        assert _rank(tmp_path, _FUZZY, "XYZZY", cutoff=0.0)

    def test_short_substring_names_not_inflated(self, tmp_path: Path):
        # 'ni'/'mp' are substrings of the query 'universalopamp'. A partial-
        # ratio score put such short substrings near 0.90, flooding the results
        # and burying the real match; the length-aware ratio scores them low.
        text = (
            ".MODEL NI NMOS(VTO=1)\n"
            ".MODEL MP PMOS(VTO=-1)\n"
            ".SUBCKT LTC3406 a b c\nR1 a b 1\n.ENDS\n"
        )
        scored = {r["name"]: r["score"] for r in _rank(tmp_path, text, "universalopamp", 0.0)}
        assert scored["NI"] < 0.6
        assert scored["MP"] < 0.6

    def test_a_name_in_several_files_is_reported_once_from_the_first(self, tmp_path: Path):
        # Vendor libraries repeat helper subcircuits; one name must not fill a
        # page, and which file it is reported from follows the order given.
        a = tmp_path / "a.lib"
        a.write_text(".MODEL DUP NPN(BF=100)\n")
        b = tmp_path / "b.lib"
        b.write_text(".MODEL DUP NPN(BF=200)\n")
        rows = rank_models([parse_library_file(b), parse_library_file(a)], "DUP", cutoff=0.0)
        assert [r["name"] for r in rows] == ["DUP"]
        assert rows[0]["source_path"] == str(b)

    def test_part_family_prefers_siblings_over_cross_family(self, tmp_path: Path):
        """2N3905 (typo) should rank 2N3904/2N3906 above BC547 or LM741."""
        text = (
            ".MODEL 2N3904 NPN(BF=200)\n"
            ".MODEL 2N3906 PNP(BF=200)\n"
            ".MODEL BC547 NPN(BF=300)\n"
            ".SUBCKT LM741 in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
        )
        rows = _rank(tmp_path, text, "2N3905", cutoff=0.0)
        assert {r["name"] for r in rows[:2]} == {"2N3904", "2N3906"}, [r["name"] for r in rows]

    def test_part_suffix_variant_ranks_high(self, tmp_path: Path):
        """LTC3406 should find LTC3406A/B ahead of a neighbouring part number."""
        text = (
            ".SUBCKT LTC3406A in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LTC3406B in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LTC3405 in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LM7812 in out\nR1 in out 1k\n.ENDS\n"
        )
        names = [r["name"] for r in _rank(tmp_path, text, "LTC3406", cutoff=0.0)]
        assert set(names[:2]) == {"LTC3406A", "LTC3406B"}, names


class TestModelRow:
    def _entry(self, tmp_path: Path, text: str, name: str):
        lib = tmp_path / "parts.lib"
        lib.write_text(text)
        return next(e for e in parse_library_file(lib).models if e.name == name)

    def test_include_directive_quotes_the_native_path(self, tmp_path: Path):
        # Simulator libraries commonly sit under a directory with a space in
        # it, and an unquoted .include stops at the first one.
        folder = tmp_path / "Program Files" / "lib"
        folder.mkdir(parents=True)
        lib = folder / "standard.bjt"
        lib.write_text(".MODEL 2N2222 NPN(BF=200)\n")
        (entry,) = parse_library_file(lib).models
        row = model_row(entry)
        assert row["source_path"] == str(lib)
        assert row["include_directive"] == f'.include "{lib}"'

    def test_npn_model_reports_device_type_and_usage(self, tmp_path: Path):
        # A .MODEL conveys parameters but not node order; the device token
        # (NPN here) dictates connection order.
        row = model_row(self._entry(tmp_path, ".MODEL 2N2222 NPN(BF=200 IS=1e-14)\n", "2N2222"))
        assert row["device_type"] == "NPN"
        assert row["usage"] == "Qxxx C B E 2N2222"

    def test_subckt_has_no_device_type(self, tmp_path: Path):
        row = model_row(
            self._entry(tmp_path, ".SUBCKT MYPART in out\nR1 in out 1k\n.ENDS\n", "MYPART")
        )
        assert row["ports"] == ["in", "out"]
        assert "device_type" not in row
        assert "usage" not in row

    def test_row_does_not_alias_the_cached_parse(self, tmp_path: Path):
        """Parses are cached and shared across lookups; a caller editing its
        rows must not edit the next caller's."""
        entry = self._entry(tmp_path, ".SUBCKT MYPART in out\nR1 in out 1k\n.ENDS\n", "MYPART")
        model_row(entry)["ports"].append("extra")
        assert model_row(entry)["ports"] == ["in", "out"]


class TestBuiltinLibraryFiles:
    """The built-in set is every library file under the detected simulators'
    own library directories: the directories staging and the model query's
    'libs' admit, so a built-in row always names a file both will read."""

    def test_caches_result(self, empty_manager: LibraryManager):
        first = empty_manager.builtin_library_files()
        second = empty_manager.builtin_library_files()
        assert first is second  # cached identical list object

    def test_files_under_every_detected_simulators_library(self, tmp_path: Path):
        first = tmp_path / "first" / "lib"
        (first / "cmp").mkdir(parents=True)
        (first / "cmp" / "standard.bjt").write_text(".model Q1 NPN\n")
        (first / "sub").mkdir()
        (first / "sub" / "LT1001.sub").write_text(".subckt LT1001 a b\n.ends\n")
        (first / "sym").mkdir()
        (first / "sym" / "res.asy").write_text("Version 4\n")
        second = tmp_path / "second"
        second.mkdir()
        (second / "models.lib").write_text(".model D1 D\n")
        (second / "notes.txt").write_text(".model D2 D\n")

        manager = LibraryManager(
            available_simulators={
                "a": installed_simulator(first),
                "b": installed_simulator(second),
            }
        )

        assert manager.library_roots() == [first.resolve(), second.resolve()]
        assert manager.builtin_library_files() == (
            first.resolve() / "cmp" / "standard.bjt",
            first.resolve() / "sub" / "LT1001.sub",
            second.resolve() / "models.lib",
        )

    def test_a_library_that_appears_later_is_found(self, tmp_path: Path):
        """LTspice extracts its library on first launch, which can be after the
        server started; the walk follows the roots instead of freezing them."""
        lib = tmp_path / "lib"
        manager = LibraryManager(available_simulators={"sim": installed_simulator(lib)})
        assert manager.builtin_library_files() == ()

        lib.mkdir()
        (lib / "standard.dio").write_text(".model D1 D\n")

        assert manager.builtin_library_files() == (lib.resolve() / "standard.dio",)

    def test_simulator_without_a_library_contributes_nothing(self):
        manager = LibraryManager(available_simulators={"ngspice": object})
        assert manager.library_roots() == []
        assert manager.builtin_library_files() == ()
