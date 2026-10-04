"""Temporary parser files stay inside their admitted Store directory."""

from pathlib import Path

import pytest

from ltspice_mcp.lib.store import Store, StoreError, parser_file_in
from tests.conftest import symlink_or_skip


def test_parser_paths_follow_relocated_store(tmp_path, monkeypatch):
    working = tmp_path / "circuit"
    working.mkdir()
    monkeypatch.setenv("LTSPICE_MCP_STORE_DIR", str(tmp_path / "storage"))
    store = Store(working)
    directory = store.parser_dir("parse_1")
    directory.mkdir(parents=True)

    artifact = store.parser_file("parse_1", "p0_t0.bin")
    artifact.write_bytes(b"captured bytes")

    assert artifact == parser_file_in(directory, "p0_t0.bin")
    assert artifact.is_relative_to(store.root)
    assert list(working.iterdir()) == []


@pytest.mark.parametrize("name", ["", ".", "..", "../elsewhere", "sub/file", "sub\\file"])
def test_parser_names_cannot_escape(tmp_path: Path, name: str):
    store = Store(tmp_path)
    with pytest.raises(StoreError, match="Invalid parser id"):
        store.parser_dir(name)
    with pytest.raises(StoreError, match="Invalid parser artifact name"):
        store.parser_file("parse_1", name)


@pytest.mark.parametrize("redirect", ["directory", "artifact"])
def test_parser_paths_refuse_redirects(tmp_path, redirect):
    store = Store(tmp_path)
    directory = store.parser_dir("parse_1")
    peer = store.parser_dir("parse_2")
    peer.mkdir(parents=True)
    if redirect == "directory":
        symlink_or_skip(directory, peer, target_is_directory=True)
    else:
        directory.mkdir()
        target = peer / "source.raw"
        target.write_bytes(b"other parser's input")
        symlink_or_skip(directory / "source.raw", target)

    with pytest.raises(StoreError):
        store.parser_file("parse_1", "source.raw")
