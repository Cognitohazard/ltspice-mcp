"""Integration tests for library tool handlers."""

from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import LibraryError, PathSecurityError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.library import (
    FindModelInput,
    ListLibrariesInput,
    LoadLibraryInput,
    UnloadLibraryInput,
    handle_find_model,
    handle_list_libraries,
    handle_load_library,
    handle_unload_library,
)


@pytest.fixture
def lib_file(work_dir: Path) -> Path:
    p = work_dir / "models.lib"
    p.write_text(
        ".MODEL 2N2222 NPN(BF=200 IS=1e-14)\n"
        ".MODEL D1N4148 D(IS=2.52e-9)\n"
        ".SUBCKT opamp in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
    )
    return p


@pytest.fixture
def lib_dir(work_dir: Path) -> Path:
    d = work_dir / "libs"
    d.mkdir()
    (d / "a.lib").write_text(".MODEL Q1 NPN(BF=100)\n")
    (d / "b.lib").write_text(".MODEL Q2 PNP(BF=50)\n")
    return d


@pytest.mark.asyncio
class TestLoadLibrary:
    async def test_load_file(self, state_no_sim: SessionState, lib_file: Path):
        result = await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        text = result.content[0].text
        assert "Loaded" in text
        assert "models" in text
        assert len(state_no_sim.libraries) == 1

    async def test_load_dir(self, state_no_sim: SessionState, lib_dir: Path):
        result = await handle_load_library(LoadLibraryInput(path=lib_dir.name), state_no_sim)
        assert "2 file" in result.content[0].text

    async def test_load_dir_picks_up_ltspice_component_files(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # load_library(dir) must discover LTspice stock component decks
        # (.bjt/.mos/.dio/.jft) — as in LTspice's lib/cmp — not just .lib/.mod.
        # Regression: the dir scan globbed only *.lib/*.mod, so a cmp/ directory
        # of standard.bjt/.mos found 0 files and raised "No library files found".
        d = work_dir / "cmp"
        d.mkdir()
        (d / "standard.bjt").write_text(".MODEL QSTD NPN(BF=100)\n")
        (d / "standard.mos").write_text(".MODEL MSTD NMOS(KP=2e-5)\n")
        result = await handle_load_library(LoadLibraryInput(path=d.name), state_no_sim)
        assert "2 file" in result.content[0].text

    async def test_path_escape(self, state_no_sim: SessionState):
        with pytest.raises(PathSecurityError):
            await handle_load_library(LoadLibraryInput(path="/etc/passwd"), state_no_sim)

    async def test_not_found(self, state_no_sim: SessionState):
        with pytest.raises(LibraryError):
            await handle_load_library(LoadLibraryInput(path="missing.lib"), state_no_sim)


@pytest.mark.asyncio
class TestUnloadLibrary:
    async def test_unload_loaded(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_unload_library(UnloadLibraryInput(path=lib_file.name), state_no_sim)
        assert "Unloaded" in result.content[0].text
        assert len(state_no_sim.libraries) == 0

    async def test_unload_not_loaded(self, state_no_sim: SessionState, lib_file: Path):
        with pytest.raises(LibraryError, match="not loaded"):
            await handle_unload_library(UnloadLibraryInput(path=lib_file.name), state_no_sim)


@pytest.mark.asyncio
class TestFindModelFull:
    """`find_model` absorbed the old ``model_info`` tool: ``full=true`` returns
    the SPICE definition body alongside the candidate metadata."""

    async def test_full_emits_raw_text(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_find_model(
            FindModelInput(name="2N2222", exact=True, full=True), state_no_sim
        )
        data = result.structuredContent
        assert data is not None
        assert data["results"][0]["name"] == "2N2222"
        assert "raw_text" in data["results"][0]
        assert ".MODEL 2N2222" in data["results"][0]["raw_text"]

    async def test_default_omits_raw_text(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N2222", exact=True), state_no_sim)
        data = result.structuredContent
        assert data is not None
        assert "raw_text" not in data["results"][0]


@pytest.mark.asyncio
class TestFindModel:
    @pytest.fixture
    def fuzzy_lib(self, work_dir: Path) -> Path:
        p = work_dir / "fuzzy.lib"
        p.write_text(
            ".MODEL 2N3904 NPN(BF=200)\n"
            ".MODEL 2N3906 PNP(BF=200)\n"
            ".MODEL 2N2222 NPN(BF=300)\n"
            ".SUBCKT LM741 in+ in- out\nR1 in+ in- 1Meg\n.ENDS\n"
        )
        return p

    @pytest.fixture
    def tiebreak_lib(self, work_dir: Path) -> Path:
        # '2N3905' (typo) is LCS-equidistant from both names, so fuzz scores
        # tie; the tiebreak must prefer the longer shared prefix ('2n390' with
        # 2N3904 vs only '2n3' with 2N3055), not the alphabetical accident.
        p = work_dir / "tiebreak.lib"
        p.write_text(".MODEL 2N3055 NPN(BF=50)\n.MODEL 2N3904 NPN(BF=200)\n")
        return p

    async def test_fuzzy_tiebreak_prefers_shared_prefix(
        self, state_no_sim: SessionState, tiebreak_lib: Path
    ):
        await handle_load_library(LoadLibraryInput(path=tiebreak_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N3905"), state_no_sim)
        names = [r["name"] for r in result.structuredContent["results"]]
        assert "2N3904" in names and "2N3055" in names
        # Equal fuzzy score → shared-prefix tiebreak puts the real neighbour
        # first, not the alphabetically-earlier 2N3055.
        assert names.index("2N3904") < names.index("2N3055")

    async def test_typo_finds_candidates(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N3905"), state_no_sim)
        text = result.content[0].text
        assert "2N3904" in text or "2N3906" in text
        data = result.structuredContent
        assert data["query"] == "2N3905"
        assert len(data["results"]) > 0
        assert all(0.0 <= r["score"] <= 1.0 for r in data["results"])

    async def test_empty_returns_hint(self, state_no_sim: SessionState):
        result = await handle_find_model(
            FindModelInput(name="XYZZY", include_builtin=True), state_no_sim
        )
        assert "No fuzzy matches" in result.content[0].text
        assert "load_library" in result.content[0].text  # full profile names the tool
        assert result.structuredContent["results"] == []

    async def test_empty_hint_agentic_omits_hidden_tool(self, config: ServerConfig):
        # load_library is full-only; an agentic agent must not be told to call it.
        config.tool_profile = "agentic"
        state = SessionState.create(config, available={})
        result = await handle_find_model(FindModelInput(name="XYZZY", include_builtin=True), state)
        text = result.content[0].text
        assert "load_library" not in text
        assert ".lib/.include" in text

    async def test_empty_hint_points_to_symbol_when_name_is_a_symbol(
        self, state_no_sim: SessionState, monkeypatch
    ):
        # A name that is a schematic SYMBOL (not a library model) legitimately has
        # no .SUBCKT/.MODEL; find_model redirects to symbol_info rather than
        # leaving the caller to conclude the part doesn't exist. Patch the name in
        # library's namespace (it imports get_symbol_info at module top).
        import ltspice_mcp.tools.library as lib_mod

        monkeypatch.setattr(lib_mod, "get_symbol_info", lambda _name: {"pins": []})
        result = await handle_find_model(
            FindModelInput(name="UniversalOpamp2", include_builtin=True), state_no_sim
        )
        text = result.content[0].text
        assert "symbol_info" in text
        assert "UniversalOpamp2" in text

    async def test_exact_match_found(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N3904", exact=True), state_no_sim)
        data = result.structuredContent
        assert data["exact"] is True
        assert len(data["results"]) == 1
        assert data["results"][0]["name"] == "2N3904"
        assert data["results"][0]["score"] == 1.0
        assert "Exact match" in result.content[0].text

    async def test_exact_match_case_insensitive(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2n3904", exact=True), state_no_sim)
        assert len(result.structuredContent["results"]) == 1

    async def test_exact_no_match(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N3905", exact=True), state_no_sim)
        assert result.structuredContent["results"] == []
        assert "No exact match" in result.content[0].text
        assert "find_model" in result.content[0].text
        assert "exact=false" in result.content[0].text

    async def test_cutoff_filters(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="XYZZY", cutoff=0.95), state_no_sim)
        assert result.structuredContent["results"] == []

    async def test_json_format(self, state_no_sim: SessionState, fuzzy_lib: Path):
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(
            FindModelInput(name="2N3905", format="json"), state_no_sim
        )
        assert result.structuredContent is not None
        assert "results" in result.structuredContent

    async def test_model_device_type_and_usage_surfaced(
        self, state_no_sim: SessionState, fuzzy_lib: Path
    ):
        # A .MODEL NPN entry must report its device token and a connection-order
        # usage string (node order isn't conveyed by the .MODEL itself).
        await handle_load_library(LoadLibraryInput(path=fuzzy_lib.name), state_no_sim)
        result = await handle_find_model(FindModelInput(name="2N3904", exact=True), state_no_sim)
        entry = result.structuredContent["results"][0]
        assert entry["device_type"] == "NPN"
        assert "Qxxx C B E" in entry["usage"]
        assert "Qxxx C B E" in result.content[0].text


@pytest.mark.asyncio
class TestListLibraries:
    async def test_empty(self, state_no_sim: SessionState):
        result = await handle_list_libraries(ListLibrariesInput(), state_no_sim)
        text = result.content[0].text
        assert "No libraries" in text
        # Points the model at the built-in libraries it can still reach.
        assert "include_builtin=true" in text

    async def test_populated_simple(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_list_libraries(ListLibrariesInput(), state_no_sim)
        text = result.content[0].text
        assert "models.lib" in text

    async def test_populated_detail(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_list_libraries(ListLibrariesInput(detail=True), state_no_sim)
        text = result.content[0].text
        assert ".SUBCKT opamp" in text

    async def test_path_filter_no_match(
        self, state_no_sim: SessionState, lib_file: Path, work_dir: Path
    ):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        # Create a different file to filter on
        other = work_dir / "other.lib"
        other.write_text(".MODEL X NPN()\n")
        result = await handle_list_libraries(ListLibrariesInput(path=other.name), state_no_sim)
        assert "No libraries matching" in result.content[0].text

    async def test_detail_surfaces_encrypted_library(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # An encrypted vendor library must read as "exists but encrypted" in
        # detail mode, not vanish: it carries no plaintext .MODEL/.SUBCKT cards
        # but is surfaced by name with a clear encrypted marker.
        enc = work_dir / "2SA2206_enc.lib"
        enc.write_text("* LTspice Encrypted File\nencrypted body, not parseable\n")
        await handle_load_library(LoadLibraryInput(path=enc.name), state_no_sim)
        result = await handle_list_libraries(ListLibrariesInput(detail=True), state_no_sim)
        text = result.content[0].text
        assert "2SA2206_enc" in text
        assert "encrypted" in text.lower()
        libs = result.structuredContent["libraries"]
        assert any(lib.get("encrypted") for lib in libs)
