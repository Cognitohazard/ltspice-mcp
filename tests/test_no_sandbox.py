"""With ``[security] allowed_paths`` unset there is no sandbox.

A path is resolved wherever it lands, a relative one from the working
directory, and a run stages every file its deck reaches. Setting the list in
the config file confines the server from the next call, and removing it lifts
the sandbox the same way. The reports and the refusal guidance say which of
the two a session is in.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.deck_staging import stage_deck
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import safe_path
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def unconfined(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SessionState:
    """A session whose config file sets no sandbox, run from a process
    directory that is not its working directory."""
    monkeypatch.delenv("LTSPICE_MCP_ALLOWED_PATHS", raising=False)
    work = tmp_path / "work"
    _write(work / "ltspice-mcp.toml", "[simulation]\ntimeout = 30\n")
    elsewhere = tmp_path / "process-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    return SessionState.create(ServerConfig.load(work / "ltspice-mcp.toml"), available={})


def test_any_path_is_admitted_and_a_relative_one_is_taken_from_the_working_directory(
    unconfined: SessionState, tmp_path: Path
):
    work = unconfined.config.working_dir
    assert unconfined.allowed_paths() is None
    assert safe_path("deck.cir", unconfined) == (work / "deck.cir").resolve()
    outside = _write(tmp_path / "anywhere" / "deck.cir", "* deck\n.end\n")
    assert safe_path(str(outside), unconfined) == outside.resolve()


def test_a_root_added_for_one_read_never_creates_a_sandbox(
    unconfined: SessionState, tmp_path: Path
):
    """The simulator's own library widens a sandbox for some reads. With no
    sandbox, naming it must not confine the read to that library."""
    library = tmp_path / "simulator-lib"
    library.mkdir()
    outside = _write(tmp_path / "anywhere" / "models.lib", ".model D D\n")
    assert unconfined.resolve_path(str(outside), library) == outside.resolve()


def test_an_unresolvable_path_is_still_refused_and_says_there_is_no_sandbox(
    unconfined: SessionState,
):
    with pytest.raises(PathSecurityError, match="null byte"):
        safe_path("deck\x00.cir", unconfined)
    guidance = unconfined.sandbox_guidance()
    assert "There is no sandbox" in guidance
    assert "Allowed paths" not in guidance


async def test_the_reports_say_there_is_no_sandbox(unconfined: SessionState):
    result = await handle_inspect(
        InspectInput.model_validate({"queries": [{"kind": "capabilities"}]}), unconfined
    )
    assert result.structured_content is not None
    (item,) = result.structured_content["results"]
    assert item["data"]["allowed_paths"] is None

    config = handle_read_resource("spice://config", unconfined)
    assert json.loads(config.contents[0].text)["allowed_paths"] is None  # type: ignore[union-attr]


def test_setting_and_removing_the_sandbox_in_the_file_applies_on_the_next_call(
    unconfined: SessionState, tmp_path: Path
):
    toml = unconfined.config.config_path
    deck = _write(tmp_path / "anywhere" / "deck.cir", "* deck\n.end\n")
    when = toml.stat().st_mtime
    assert safe_path(str(deck), unconfined) == deck.resolve()

    toml.write_text('[security]\nallowed_paths = ["."]\n', encoding="utf-8")
    os.utime(toml, (when + 2, when + 2))  # a same-second rewrite must still register
    with pytest.raises(PathSecurityError, match="outside allowed directories"):
        safe_path(str(deck), unconfined)

    toml.write_text("[simulation]\ntimeout = 30\n", encoding="utf-8")
    os.utime(toml, (when + 4, when + 4))
    assert safe_path(str(deck), unconfined) == deck.resolve()


def test_a_run_stages_every_file_its_deck_reaches(tmp_path: Path):
    """No sandbox: a library beside the deck's folder, one reached through it,
    and one named by an absolute path are all copied and hashed, and the
    staged deck reads the copies."""
    models = tmp_path / "models"
    top = _write(models / "top.lib", ".include sub.inc\n")
    sub = _write(models / "sub.inc", ".model D D\n")
    other = _write(tmp_path / "vendor" / "other.inc", ".param x=1\n")
    deck = _write(
        tmp_path / "bench" / "deck.cir",
        f'V1 a 0 1\n.include ../models/top.lib\n.include "{other}"\n.op\n.end\n',
    )

    staged = stage_deck(deck, tmp_path / "stage", None, origin=deck)

    by_source = {entry.path: entry for entry in staged.manifest}
    for source in (deck, top, sub, other):
        entry = by_source[source.resolve()]
        assert entry.staged and not entry.live, source
        assert entry.sha256 == hashlib.sha256(source.read_bytes()).hexdigest(), source
        assert entry.staged_path is not None and entry.staged_path.is_file(), source
    # The deck's copy is rewritten to read the copies; the libraries are not.
    for source in (top, sub, other):
        staged_path = by_source[source.resolve()].staged_path
        assert staged_path is not None and staged_path.read_bytes() == source.read_bytes()
    assert not staged.observations
    text = staged.staged_deck.read_text(encoding="utf-8")
    assert str(models) not in text and str(other) not in text
