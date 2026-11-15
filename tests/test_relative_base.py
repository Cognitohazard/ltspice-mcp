"""Where a relative path is taken from, said the same way it is used.

A relative name is read from the working directory, or from the first allowed
directory when the sandbox leaves the working directory out, whichever door
it comes through. Listings and refusal notes must name the folder a read of
that name opens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcp.types import TextResourceContents

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import PathSecurityError
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
        (split_state.working_dir / "deck.cir").write_text(_DECK, encoding="utf-8")
        assert _listed_uris(split_state) == ["spice://netlists/deck.cir"]
        assert split_state.resolve_path("deck.cir").parent == split_state.relative_base()


class TestRelativeBase:
    def test_it_is_the_working_directory_inside_the_sandbox(
        self, split_state: SessionState
    ) -> None:
        work = split_state.working_dir
        assert split_state.relative_base() == work.resolve()
        assert split_state.resolve_path("x.cir") == work.resolve() / "x.cir"

    def test_a_sandbox_without_the_working_directory_moves_it_to_the_first_entry(
        self, tmp_path: Path
    ) -> None:
        work, first = tmp_path / "work", tmp_path / "first"
        work.mkdir()
        first.mkdir()
        config = ServerConfig(
            working_dir=work,
            allowed_paths=[first],
            config_path=tmp_path / "unused.ltspice-mcp.toml",
            write_config=False,
        )
        state = SessionState.create(config, available={})
        assert state.relative_base() == first.resolve()
        assert state.resolve_path("x.cir") == first.resolve() / "x.cir"
        (first / "deck.cir").write_text(_DECK, encoding="utf-8")
        assert _listed_uris(state) == ["spice://netlists/deck.cir"]
        # The Python API on the same session reads the same file.
        assert SyncApi(state).verify_circuit(path="deck.cir")["path"] == str(
            first.resolve() / "deck.cir"
        )

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
        guidance = split_state.sandbox_guidance()
        assert f"A relative path is taken from {split_state.working_dir.resolve()}," in guidance

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
