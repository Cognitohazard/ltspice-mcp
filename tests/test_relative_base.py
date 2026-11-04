"""Where a relative path is taken from, said the same way it is used.

With a sandbox whose first entry is not the working directory, the MCP side
reads a relative name from that first entry and the Python API reads it from
its working directory. Each door's listings and refusal notes must name the
folder that door actually reads from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.types import TextResourceContents

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.pathutil import relative_paths_from
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from tests.conftest import SyncApi, stage_recorded_fixture

_DECK = "V1 in 0 1\nR1 in 0 1k\n.op\n.end\n"


@pytest.fixture
def split_state(tmp_path: Path) -> SessionState:
    """A sandbox of ``[first, work]`` with ``work`` as the working directory."""
    work = tmp_path / "work"
    first = tmp_path / "first"
    work.mkdir()
    first.mkdir()
    config = ServerConfig(
        working_dir=work,
        allowed_paths=[first, work],
        log_level="DEBUG",
        config_path=tmp_path / "unused.ltspice-mcp.toml",
        write_config=False,
    )
    return SessionState.create(config, available={})


def _first(state: SessionState) -> Path:
    paths = state.config.allowed_paths
    assert paths
    return paths[0]


def _listed_uris(state: SessionState) -> list[str]:
    listing = handle_read_resource("spice://netlists/", state).contents[0]
    assert isinstance(listing, TextResourceContents)
    return [entry["uri"] for entry in json.loads(listing.text)["netlists"]]


class TestNetlistResource:
    def test_every_listed_netlist_reads_back(self, split_state: SessionState) -> None:
        first = _first(split_state)
        (first / "in_first.cir").write_text(_DECK, encoding="utf-8")
        (split_state.working_dir / "in_work.cir").write_text(_DECK, encoding="utf-8")

        uris = _listed_uris(split_state)
        assert uris, "the listing names nothing to read back"
        for uri in uris:
            text = handle_read_resource(uri, split_state).contents[0]
            assert getattr(text, "text", "").startswith("V1 in 0 1")

    def test_the_listing_is_the_folder_a_relative_name_resolves_in(
        self, split_state: SessionState
    ) -> None:
        first = _first(split_state)
        (first / "deck.cir").write_text(_DECK, encoding="utf-8")
        assert _listed_uris(split_state) == ["spice://netlists/deck.cir"]
        assert split_state.resolve_path("deck.cir").parent == split_state.relative_base()


class TestRelativeBase:
    def test_it_is_the_folder_resolve_path_anchors_on(self, split_state: SessionState) -> None:
        first = _first(split_state)
        assert split_state.relative_base() == first.resolve()
        assert split_state.resolve_path("x.cir") == first.resolve() / "x.cir"

    def test_a_declared_base_moves_it(self, split_state: SessionState) -> None:
        work = split_state.working_dir
        with relative_paths_from(work):
            assert split_state.relative_base() == work.resolve()
            assert split_state.resolve_path("x.cir") == work.resolve() / "x.cir"

    def test_without_a_sandbox_it_is_the_working_directory(self, tmp_path: Path) -> None:
        config = ServerConfig(
            working_dir=tmp_path,
            allowed_paths=None,
            config_path=tmp_path / "unused.ltspice-mcp.toml",
            write_config=False,
        )
        state = SessionState.create(config, available={})
        assert state.relative_base() == tmp_path.resolve()


class TestRefusalNotes:
    def test_the_mcp_guidance_names_the_folder_mcp_reads_from(
        self, split_state: SessionState
    ) -> None:
        first = _first(split_state)
        guidance = split_state.sandbox_guidance()
        assert f"A relative path is taken from {first.resolve()}," in guidance

    def test_the_api_note_names_the_folder_the_api_reads_from(
        self, split_state: SessionState, tmp_path: Path
    ) -> None:
        work = split_state.working_dir
        stage_recorded_fixture(work, "ltspice_tran_rc")
        api = SyncApi(split_state)
        # The API reads a relative name from its working directory...
        assert (
            api.load_raw("ltspice_tran_rc.raw").source == (work / "ltspice_tran_rc.raw").resolve()
        )
        # ...so the note on a refusal says that is where relatives go.
        outside = tmp_path / "outside.raw"
        with pytest.raises(PathSecurityError) as caught:
            api.load_raw(str(outside))
        (note,) = caught.value.__notes__
        assert f"A relative path is taken from {work.resolve()}," in note
