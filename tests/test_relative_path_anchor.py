"""Which folder a relative path, or a relative sandbox entry, stands for.

The answer has to be the same in a tool call, in the in-process Python API
(which a detached job's owner runs), and in a ``run_code`` snippet, whatever
folder the server process was started in. Here the process starts in
``proc/``, the working directory is given relative to it, and the sandbox is a
relative entry: each door must read the circuit in the working directory, and
the server's own store must lie inside its own sandbox.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from mcp.types import TextResourceContents

from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.run_code import RunCodeInput, handle_run_code, worker_for
from tests.conftest import SyncApi, dispatched, make_experiment_job, stage_recorded_fixture

_DECK = "* {where}\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n"


class _Layout:
    def __init__(self, root: Path, working_dir: str) -> None:
        self.proc = root / "proc"
        self.proc.mkdir()
        self.work = (self.proc / working_dir).resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        for folder, where in ((self.proc, "process"), (self.work, "working")):
            (folder / "x.cir").write_text(_DECK.format(where=where), encoding="utf-8")


# The working directory as an operator writes it: a sibling of the folder the
# server was started in, and a folder inside it.
_WORKING_DIRS = ["../work", "work"]


@pytest.fixture(params=["toml", "env"])
def sandbox_source(request: pytest.FixtureRequest) -> str:
    return request.param


def _configure(
    monkeypatch: pytest.MonkeyPatch, layout: _Layout, working_dir: str, sandbox_source: str
) -> None:
    monkeypatch.chdir(layout.proc)
    monkeypatch.setenv("LTSPICE_MCP_DISABLE_SIMULATOR_DETECTION", "1")
    monkeypatch.setenv("LTSPICE_MCP_WRITE_CONFIG", "false")
    monkeypatch.setenv("LTSPICE_MCP_WORKING_DIR", working_dir)
    monkeypatch.delenv("LTSPICE_MCP_CONFIG", raising=False)
    monkeypatch.delenv("LTSPICE_MCP_ALLOWED_PATHS", raising=False)
    if sandbox_source == "toml":
        (layout.proc / "ltspice-mcp.toml").write_text(
            '[security]\nallowed_paths = ["."]\n', encoding="utf-8"
        )
    else:
        monkeypatch.setenv("LTSPICE_MCP_ALLOWED_PATHS", ".")


@pytest.fixture(params=_WORKING_DIRS)
def server(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sandbox_source: str,
) -> tuple[SessionState, _Layout]:
    working_dir = request.param
    layout = _Layout(tmp_path, working_dir)
    _configure(monkeypatch, layout, working_dir, sandbox_source)
    # The server loads its configuration as main does: from the environment
    # and the file in the folder it was started in.
    return SessionState.create(ServerConfig.load(), available={}), layout


def _read_from(reply: dict) -> Path:
    assert "path" in reply, reply
    return Path(reply["path"])


class TestEveryDoorReadsTheWorkingDirectory:
    async def test_a_tool_call(self, server: tuple[SessionState, _Layout]) -> None:
        state, layout = server
        reply = await dispatched(state, "verify_circuit", path="x.cir")
        assert _read_from(reply) == layout.work / "x.cir"
        assert not any(f.get("rule_id") == "path_denied" for f in reply.get("findings", []))

    def test_the_netlist_resource(self, server: tuple[SessionState, _Layout]) -> None:
        state, _ = server
        listing = handle_read_resource("spice://netlists/", state).contents[0]
        assert isinstance(listing, TextResourceContents)
        assert [e["name"] for e in json.loads(listing.text)["netlists"]] == ["x.cir"]
        text = handle_read_resource("spice://netlists/x.cir", state).contents[0]
        assert isinstance(text, TextResourceContents)
        assert text.text.startswith("* working")

    def test_the_python_api_on_the_servers_session(
        self, server: tuple[SessionState, _Layout]
    ) -> None:
        state, layout = server
        reply = SyncApi(state).verify_circuit(path="x.cir")
        assert _read_from(reply) == layout.work / "x.cir"

    def test_an_api_opened_as_a_detached_owner_opens_it(
        self, server: tuple[SessionState, _Layout]
    ) -> None:
        # A detached job's owner starts in the caller's folder and opens
        # ``Api(working_dir=None)``: the configuration names the folder.
        _, layout = server
        with Api() as api:
            assert api.verify_circuit(path="x.cir")["path"] == str(layout.work / "x.cir")

    def test_the_sandbox_holds_the_store(self, server: tuple[SessionState, _Layout]) -> None:
        state, layout = server
        assert state.working_dir == layout.work
        assert state.relative_base() == layout.work
        sandbox = state.allowed_paths()
        assert sandbox is not None and all(p.is_absolute() for p in sandbox)
        assert state.resolve_path(str(state.store.root / "x.raw")).is_relative_to(layout.work)


class TestStoredArtifacts:
    """The server reads back what it stored, by job and by path."""

    async def test_analyze_and_inspect_read_the_store(
        self, server: tuple[SessionState, _Layout]
    ) -> None:
        state, _ = server
        kept = state.store.root / "runs" / "kept"
        kept.mkdir(parents=True)
        raw = stage_recorded_fixture(kept, "ltspice_tran_rc")
        make_experiment_job(state, job_id="kept", raw=raw)
        recipe = [{"key": "v", "metric": "value", "expr": "V(out)", "at": "900u"}]
        for source in ({"job_id": "kept"}, {"path": str(raw)}):
            reply = await dispatched(
                state, "analyze_results", sources=[{**source, "label": "s"}], recipes=recipe
            )
            assert reply["results"]["v"].get("values"), (source, reply)
        for query in ({"kind": "results", "job_id": "kept"}, {"kind": "results", "path": str(raw)}):
            reply = await dispatched(state, "inspect", queries=[query])
            assert "error" not in json.dumps(reply["results"][0]).lower()[:200], reply


# -- run_code: a real worker process --------------------------------------------


@pytest.fixture
async def code_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[SessionState, _Layout]]:
    layout = _Layout(tmp_path, "work")
    _configure(monkeypatch, layout, "work", "toml")
    state = SessionState.create(ServerConfig.load(), available={})
    yield state, layout
    await state.shutdown()


async def test_a_run_code_snippet_reads_the_working_directory(
    code_server: tuple[SessionState, _Layout],
) -> None:
    state, layout = code_server
    assert worker_for(state).working_dir == layout.work
    code = (
        "import os, json\n"
        "r = api.verify_circuit(path='x.cir')\n"
        "print(json.dumps({'cwd': os.getcwd(), 'path': r.get('path')}))\n"
    )
    reply = (await handle_run_code(RunCodeInput(code=code), state)).structured_content
    assert reply is not None
    seen = json.loads(reply["stdout"].strip().splitlines()[-1])
    assert Path(seen["path"]) == layout.work / "x.cir", reply
    assert Path(seen["cwd"]) == layout.work
