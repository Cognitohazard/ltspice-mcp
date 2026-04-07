"""Integration tests for library tool handlers."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import LibraryError, PathSecurityError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.library import (
    GetModelInfoInput,
    ListLibrariesInput,
    LoadLibraryInput,
    SearchLibraryInput,
    UnloadLibraryInput,
    handle_get_model_info,
    handle_list_libraries,
    handle_load_library,
    handle_search_library,
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
        result = await handle_load_library(
            LoadLibraryInput(path=lib_file.name), state_no_sim
        )
        text = result.content[0].text
        assert "Loaded" in text
        assert "models" in text
        assert len(state_no_sim.libraries) == 1

    async def test_load_dir(self, state_no_sim: SessionState, lib_dir: Path):
        result = await handle_load_library(
            LoadLibraryInput(path=lib_dir.name), state_no_sim
        )
        assert "2 file" in result.content[0].text

    async def test_path_escape(self, state_no_sim: SessionState):
        with pytest.raises(PathSecurityError):
            await handle_load_library(
                LoadLibraryInput(path="/etc/passwd"), state_no_sim
            )

    async def test_not_found(self, state_no_sim: SessionState):
        with pytest.raises(LibraryError):
            await handle_load_library(
                LoadLibraryInput(path="missing.lib"), state_no_sim
            )


@pytest.mark.asyncio
class TestUnloadLibrary:
    async def test_unload_loaded(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_unload_library(
            UnloadLibraryInput(path=lib_file.name), state_no_sim
        )
        assert "Unloaded" in result.content[0].text
        assert len(state_no_sim.libraries) == 0

    async def test_unload_not_loaded(self, state_no_sim: SessionState, lib_file: Path):
        with pytest.raises(LibraryError, match="not loaded"):
            await handle_unload_library(
                UnloadLibraryInput(path=lib_file.name), state_no_sim
            )


@pytest.mark.asyncio
class TestSearchLibrary:
    async def test_search_user_finds(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_search_library(
            SearchLibraryInput(query="2222", source="user"), state_no_sim
        )
        text = result.content[0].text
        assert "2N2222" in text

    async def test_search_user_no_results(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_search_library(
            SearchLibraryInput(query="ZZZNOPE"), state_no_sim
        )
        assert "No models" in result.content[0].text

    async def test_search_builtin_empty(self, state_no_sim: SessionState):
        result = await handle_search_library(
            SearchLibraryInput(query="anything", source="builtin"), state_no_sim
        )
        assert "No models" in result.content[0].text

    async def test_search_json_format(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_search_library(
            SearchLibraryInput(query="", format="json"), state_no_sim
        )
        assert result.structuredContent is not None
        assert "results" in result.structuredContent

    async def test_search_pagination(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_search_library(
            SearchLibraryInput(query="", offset=1, limit=1), state_no_sim
        )
        assert result.structuredContent["pagination"]["total"] == 3
        assert len(result.structuredContent["results"]) == 1


@pytest.mark.asyncio
class TestGetModelInfo:
    async def test_found(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_get_model_info(
            GetModelInfoInput(name="2N2222"), state_no_sim
        )
        text = result.content[0].text
        assert "2N2222" in text
        assert ".include" in text

    async def test_full(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_get_model_info(
            GetModelInfoInput(name="2N2222", full=True), state_no_sim
        )
        text = result.content[0].text
        assert "Full SPICE definition" in text

    async def test_not_found(self, state_no_sim: SessionState):
        with pytest.raises(LibraryError, match="not found"):
            await handle_get_model_info(
                GetModelInfoInput(name="NOPE"), state_no_sim
            )


@pytest.mark.asyncio
class TestListLibraries:
    async def test_empty(self, state_no_sim: SessionState):
        result = await handle_list_libraries(ListLibrariesInput(), state_no_sim)
        assert "No libraries" in result.content[0].text

    async def test_populated_simple(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_list_libraries(ListLibrariesInput(), state_no_sim)
        text = result.content[0].text
        assert "models.lib" in text

    async def test_populated_detail(self, state_no_sim: SessionState, lib_file: Path):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        result = await handle_list_libraries(
            ListLibrariesInput(detail=True), state_no_sim
        )
        text = result.content[0].text
        assert ".SUBCKT opamp" in text

    async def test_path_filter_no_match(
        self, state_no_sim: SessionState, lib_file: Path, work_dir: Path
    ):
        await handle_load_library(LoadLibraryInput(path=lib_file.name), state_no_sim)
        # Create a different file to filter on
        other = work_dir / "other.lib"
        other.write_text(".MODEL X NPN()\n")
        result = await handle_list_libraries(
            ListLibrariesInput(path=other.name), state_no_sim
        )
        assert "No libraries matching" in result.content[0].text
