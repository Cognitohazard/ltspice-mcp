"""Unit tests for LibraryManager — load/unload/search/lookup, no simulator."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import LibraryError
from ltspice_mcp.lib.library_manager import LibraryManager


@pytest.fixture
def empty_manager() -> LibraryManager:
    return LibraryManager(available_simulators={})


@pytest.fixture
def lib_file(tmp_path: Path) -> Path:
    p = tmp_path / "models.lib"
    p.write_text(
        ".MODEL 2N2222 NPN(BF=200 IS=1e-14)\n"
        ".MODEL D1N4148 D(IS=2.52e-9)\n"
        ".SUBCKT opamp in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
    )
    return p


@pytest.fixture
def lib_dir(tmp_path: Path) -> Path:
    d = tmp_path / "libs"
    d.mkdir()
    (d / "a.lib").write_text(".MODEL Q1 NPN(BF=100)\n")
    (d / "b.lib").write_text(".MODEL Q2 PNP(BF=50)\n")
    return d


class TestLoadLibrary:
    def test_load_single_file(self, empty_manager: LibraryManager, lib_file: Path):
        summary = empty_manager.load_library(lib_file)
        assert summary["files_loaded"] == 1
        assert summary["models"] == 2
        assert summary["subcircuits"] == 1
        assert len(empty_manager) == 1

    def test_load_directory(self, empty_manager: LibraryManager, lib_dir: Path):
        summary = empty_manager.load_library(lib_dir)
        assert summary["files_loaded"] == 2
        assert summary["models"] == 2
        assert len(empty_manager) == 2

    def test_load_missing_path(self, empty_manager: LibraryManager, tmp_path: Path):
        with pytest.raises(LibraryError, match="does not exist"):
            empty_manager.load_library(tmp_path / "missing")

    def test_load_empty_dir(self, empty_manager: LibraryManager, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(LibraryError, match="No library files"):
            empty_manager.load_library(empty)

    def test_load_garbage_file(self, empty_manager: LibraryManager, tmp_path: Path):
        garbage = tmp_path / "junk.lib"
        garbage.write_text("not a spice file at all\nrandom text\n")
        with pytest.raises(LibraryError, match="No valid models"):
            empty_manager.load_library(garbage)


class TestUnloadLibrary:
    def test_unload_single(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        result = empty_manager.unload_library(lib_file)
        assert result["removed"] is True
        assert len(empty_manager) == 0

    def test_unload_directory(self, empty_manager: LibraryManager, lib_dir: Path):
        empty_manager.load_library(lib_dir)
        result = empty_manager.unload_library(lib_dir)
        assert result["removed"] is True
        assert len(empty_manager) == 0

    def test_unload_not_loaded(self, empty_manager: LibraryManager, lib_file: Path):
        result = empty_manager.unload_library(lib_file)
        assert result["removed"] is False
        assert "not loaded" in (result["warning"] or "").lower()


class TestSearchUserLibraries:
    def test_search_finds_results(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        result = empty_manager.search_user_libraries("2222")
        assert result["total"] == 1
        assert result["results"][0]["name"] == "2N2222"
        assert result["results"][0]["type"] == ".MODEL"

    def test_search_empty_query_returns_all(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        result = empty_manager.search_user_libraries("")
        assert result["total"] == 3

    def test_search_pagination(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        result = empty_manager.search_user_libraries("", offset=1, limit=1)
        assert result["total"] == 3
        assert len(result["results"]) == 1
        assert result["offset"] == 1

    def test_search_no_results(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        result = empty_manager.search_user_libraries("NONEXISTENT")
        assert result["total"] == 0
        assert result["results"] == []


class TestGetModelInfo:
    def test_found_in_user_lib(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        info = empty_manager.get_model_info("2N2222")
        assert info is not None
        assert info["name"] == "2N2222"
        assert "include_directive" in info
        assert ".include" in info["include_directive"]
        assert "raw_text" not in info

    def test_full_includes_raw_text(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        info = empty_manager.get_model_info("2N2222", full=True)
        assert info is not None
        assert "raw_text" in info

    def test_not_found(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        assert empty_manager.get_model_info("NOPE") is None

    def test_falls_back_to_builtin(self, empty_manager: LibraryManager, lib_file: Path):
        # Empty user libs, but builtin path resolves to our lib
        empty_manager._builtin_paths = [lib_file]
        info = empty_manager.get_model_info("D1N4148")
        assert info is not None
        assert info["name"] == "D1N4148"

    def test_include_builtin_false_skips_builtin(
        self, empty_manager: LibraryManager, lib_file: Path
    ):
        empty_manager._builtin_paths = [lib_file]
        assert empty_manager.get_model_info("D1N4148", include_builtin=False) is None

    def test_case_insensitive(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        info = empty_manager.get_model_info("2n2222")
        assert info is not None


class TestFindSimilarModels:
    @pytest.fixture
    def fuzzy_lib(self, tmp_path: Path) -> Path:
        p = tmp_path / "fuzzy.lib"
        p.write_text(
            ".MODEL 2N3904 NPN(BF=200)\n"
            ".MODEL 2N3906 PNP(BF=200)\n"
            ".MODEL 2N2222 NPN(BF=300)\n"
            ".MODEL D1N4148 D(IS=2.52e-9)\n"
            ".SUBCKT LM741 in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
        )
        return p

    def test_typo_finds_correct(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        results = empty_manager.find_similar_models("2N3905")
        assert len(results) > 0
        names = [r["name"] for r in results]
        assert "2N3904" in names or "2N3906" in names
        assert all("score" in r and 0.0 <= r["score"] <= 1.0 for r in results)

    def test_ranked_by_score(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        results = empty_manager.find_similar_models("2N3905", limit=5, cutoff=0.0)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_case_insensitive(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        results = empty_manager.find_similar_models("lm741")
        assert any(r["name"] == "LM741" for r in results)

    def test_cutoff_filters(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        strict = empty_manager.find_similar_models("XYZZY", cutoff=0.9)
        assert strict == []
        loose = empty_manager.find_similar_models("XYZZY", cutoff=0.0)
        assert len(loose) > 0

    def test_limit_caps_results(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        results = empty_manager.find_similar_models("2N", limit=2, cutoff=0.0)
        assert len(results) == 2

    def test_empty_when_no_libs_loaded(self, empty_manager: LibraryManager):
        assert empty_manager.find_similar_models("anything") == []

    def test_include_builtin_walks_builtin(
        self, empty_manager: LibraryManager, fuzzy_lib: Path
    ):
        empty_manager._builtin_paths = [fuzzy_lib]
        no_builtin = empty_manager.find_similar_models("2N3905", include_builtin=False)
        with_builtin = empty_manager.find_similar_models("2N3905", include_builtin=True)
        assert no_builtin == []
        assert len(with_builtin) > 0

    def test_include_directive_present(self, empty_manager: LibraryManager, fuzzy_lib: Path):
        empty_manager.load_library(fuzzy_lib)
        results = empty_manager.find_similar_models("2N3905")
        assert results
        assert all(".include" in r["include_directive"] for r in results)

    def test_part_family_prefers_siblings_over_cross_family(
        self, empty_manager: LibraryManager, tmp_path: Path
    ):
        """2N3905 (typo) should rank 2N3904/2N3906 above BC547 or LM741."""
        lib = tmp_path / "mixed.lib"
        lib.write_text(
            ".MODEL 2N3904 NPN(BF=200)\n"
            ".MODEL 2N3906 PNP(BF=200)\n"
            ".MODEL BC547 NPN(BF=300)\n"
            ".SUBCKT LM741 in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
        )
        empty_manager.load_library(lib)
        results = empty_manager.find_similar_models("2N3905", limit=5, cutoff=0.0)
        top_two_names = {r["name"] for r in results[:2]}
        assert top_two_names == {"2N3904", "2N3906"}, f"got ranking: {[r['name'] for r in results]}"

    def test_part_suffix_variant_ranks_high(
        self, empty_manager: LibraryManager, tmp_path: Path
    ):
        """LTC3406 should find LTC3406A/B near the top (substring bias)."""
        lib = tmp_path / "ltc.lib"
        lib.write_text(
            ".SUBCKT LTC3406A in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LTC3406B in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LTC3405 in out\nR1 in out 1k\n.ENDS\n"
            ".SUBCKT LM7812 in out\nR1 in out 1k\n.ENDS\n"
        )
        empty_manager.load_library(lib)
        results = empty_manager.find_similar_models("LTC3406", limit=4, cutoff=0.0)
        names = [r["name"] for r in results]
        # LTC3406A and LTC3406B should both be in the top 2 (order may vary).
        assert set(names[:2]) == {"LTC3406A", "LTC3406B"}, f"got: {names}"


class TestListLibraries:
    def test_empty(self, empty_manager: LibraryManager):
        assert empty_manager.list_libraries() == []

    def test_populated(self, empty_manager: LibraryManager, lib_file: Path):
        empty_manager.load_library(lib_file)
        libs = empty_manager.list_libraries()
        assert len(libs) == 1
        assert str(lib_file) in libs[0]


class TestDetectBuiltinPaths:
    def test_caches_result(self, empty_manager: LibraryManager):
        first = empty_manager._detect_builtin_paths()
        second = empty_manager._detect_builtin_paths()
        assert first is second  # cached identical list object

    def test_with_ltspice_calls_detector(self, monkeypatch, tmp_path: Path):
        mgr = LibraryManager(available_simulators={"ltspice": object})
        sentinel = [tmp_path / "x.lib"]
        monkeypatch.setattr(mgr, "_detect_ltspice_paths", lambda: sentinel)
        result = mgr._detect_builtin_paths()
        assert result == sentinel

    def test_with_ngspice_calls_detector(self, monkeypatch, tmp_path: Path):
        mgr = LibraryManager(available_simulators={"ngspice": object})
        sentinel = [tmp_path / "x.lib"]
        monkeypatch.setattr(mgr, "_detect_ngspice_paths", lambda: sentinel)
        result = mgr._detect_builtin_paths()
        assert result == sentinel


class TestDetectLtspicePaths:
    def test_no_paths_on_empty_system(self, monkeypatch):
        mgr = LibraryManager(available_simulators={})
        # Force is_wsl=False, sys.platform=linux
        import sys

        monkeypatch.setattr("ltspice_mcp.lib.library_manager.is_wsl", lambda: False)
        monkeypatch.setattr(sys, "platform", "linux")
        result = mgr._detect_ltspice_paths()
        # No Wine prefix on test system
        assert isinstance(result, list)

    def test_wsl_path_branch(self, monkeypatch, tmp_path: Path):
        mgr = LibraryManager(available_simulators={})
        monkeypatch.setattr("ltspice_mcp.lib.library_manager.is_wsl", lambda: True)
        # The /mnt/c/Users dir won't exist on test system → returns empty
        result = mgr._detect_ltspice_paths()
        assert isinstance(result, list)


class TestDetectNgspicePaths:
    def test_with_env_var(self, monkeypatch, tmp_path: Path):
        env_lib = tmp_path / "ngspice_lib"
        env_lib.mkdir()
        monkeypatch.setenv("SPICE_LIB_DIR", str(env_lib))
        mgr = LibraryManager(available_simulators={})
        result = mgr._detect_ngspice_paths()
        assert isinstance(result, list)

    def test_default_paths(self, monkeypatch):
        monkeypatch.delenv("SPICE_LIB_DIR", raising=False)
        mgr = LibraryManager(available_simulators={})
        result = mgr._detect_ngspice_paths()
        assert isinstance(result, list)
