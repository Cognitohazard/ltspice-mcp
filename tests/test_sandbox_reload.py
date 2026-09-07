"""The sandbox follows ``[security] allowed_paths`` in the config file while the
server runs, so the refusal an agent gets can name a line the agent edits itself."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import safe_path


def _write(toml: Path, roots: list[str], when: float) -> None:
    toml.write_text(
        "[security]\nallowed_paths = [" + ", ".join(json.dumps(r) for r in roots) + "]\n"
    )
    os.utime(toml, (when, when))  # a same-second rewrite must still register


def test_editing_allowed_paths_takes_effect_on_the_next_call(tmp_path: Path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    deck = elsewhere / "deck.cir"
    deck.write_text(".end\n")
    toml = work / "ltspice-mcp.toml"
    now = toml.parent.stat().st_mtime
    _write(toml, ["."], now)
    monkeypatch.chdir(work)
    state = SessionState.create(ServerConfig.load(toml), available={})

    with pytest.raises(PathSecurityError):
        safe_path(str(deck), state)

    _write(toml, [".", str(elsewhere)], now + 2)
    assert safe_path(str(deck), state) == deck.resolve()

    # Narrowing is honoured the same way: the file is the authority, not the first read.
    _write(toml, ["."], now + 4)
    with pytest.raises(PathSecurityError):
        safe_path(str(deck), state)
