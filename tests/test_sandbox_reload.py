"""The sandbox follows ``[security] allowed_paths`` in the config file while the
server runs, so the refusal an agent gets can name a line the agent edits itself.

Two halves, both pinned here. Every reader of the sandbox goes through the
reload, so a report or a resolution made right after an edit sees the edit even
when no earlier call happened to reload it. And every surface that reports a
refusal carries the same guidance (the config file, the key, and that the file
is re-read on the next call): a tool's structured ``hint``, and a note on the
exception the Python API raises.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from ltspice_mcp import engine
from ltspice_mcp.api import Api
from ltspice_mcp.config import ServerConfig, generate_default_config
from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib.deck_prep import resolve_netlist_path
from ltspice_mcp.lib.services import resolve_analysis_source
from ltspice_mcp.resources import handle_read_resource
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import safe_path
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit
from tests.conftest import FakeSim

_DECK = "* deck\nR1 in 0 1k\n.end\n"


def _write(toml: Path, roots: list[str], when: float) -> None:
    toml.write_text(
        "[security]\nallowed_paths = [" + ", ".join(json.dumps(r) for r in roots) + "]\n"
    )
    os.utime(toml, (when, when))  # a same-second rewrite must still register


class _Sandbox:
    """A working directory whose config allows only itself, a directory outside
    it holding a deck, and the means to widen the sandbox to that directory."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("LTSPICE_MCP_ALLOWED_PATHS", raising=False)
        self.work = tmp_path / "work"
        self.work.mkdir()
        self.elsewhere = tmp_path / "elsewhere"
        self.elsewhere.mkdir()
        self.deck = self.elsewhere / "deck.cir"
        # LF on every platform: one reader compares the text it serves.
        self.deck.write_text(_DECK, newline="\n")
        self.toml = self.work / "ltspice-mcp.toml"
        self.now = self.toml.parent.stat().st_mtime
        _write(self.toml, ["."], self.now)
        monkeypatch.chdir(self.work)

    def state(self, available: dict[str, type] | None = None) -> SessionState:
        return SessionState.create(ServerConfig.load(self.toml), available=available or {})

    def widen(self) -> None:
        _write(self.toml, [".", str(self.elsewhere)], self.now + 2)


def test_editing_allowed_paths_takes_effect_on_the_next_call(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    with pytest.raises(PathSecurityError):
        safe_path(str(box.deck), state)

    box.widen()
    assert safe_path(str(box.deck), state) == box.deck.resolve()

    # Narrowing is honoured the same way: the file is the authority, not the first read.
    _write(box.toml, ["."], box.now + 4)
    with pytest.raises(PathSecurityError):
        safe_path(str(box.deck), state)


# ---------------------------------------------------------------------------
# Every reader follows the file, not only the ones that call safe_path
# ---------------------------------------------------------------------------


async def test_capabilities_alone_reports_an_edited_sandbox(tmp_path: Path, monkeypatch):
    """The capabilities report is where an agent checks that its edit took, so
    it must not depend on some other call in the batch having reloaded first."""
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    box.widen()
    result = await handle_inspect(
        InspectInput.model_validate({"queries": [{"kind": "capabilities"}]}), state
    )

    assert result.structured_content is not None
    (item,) = result.structured_content["results"]
    assert str(box.elsewhere) in item["data"]["allowed_paths"]


def test_config_resource_reports_an_edited_sandbox(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    box.widen()
    result = handle_read_resource("spice://config", state)

    text = result.contents[0].text  # type: ignore[union-attr]
    assert str(box.elsewhere) in json.loads(text)["allowed_paths"]


def _netlist_path(state: SessionState, deck: Path) -> None:
    assert resolve_netlist_path(str(deck), state) == deck.resolve()


def _raw_source(state: SessionState, deck: Path) -> None:
    raw = deck.with_suffix(".raw")
    assert resolve_analysis_source(state, raw_file=str(raw)).raw == raw.resolve()


def _log_source(state: SessionState, deck: Path) -> None:
    log = deck.with_suffix(".log")
    assert resolve_analysis_source(state, log_file=str(log)).log == log.resolve()


def _netlist_resource(state: SessionState, deck: Path) -> None:
    result = handle_read_resource(f"spice://netlists/{quote(str(deck), safe='')}", state)
    assert result.contents[0].text == _DECK  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "read",
    [_netlist_path, _raw_source, _log_source, _netlist_resource],
    ids=["netlist-path", "raw-source", "log-source", "netlist-resource"],
)
def test_every_path_reader_admits_a_newly_allowed_directory(tmp_path: Path, monkeypatch, read):
    """Each reader is the first call after the edit, on a session that has not
    resolved a path since, so none of them can lean on another's reload."""
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    box.widen()
    read(state, box.deck)


async def test_hierarchy_query_admits_a_newly_allowed_directory(tmp_path: Path, monkeypatch):
    """The hierarchy loader checks every file it opens against the list it is
    handed, so it has to be handed the reloaded one."""
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    box.widen()
    result = await handle_inspect(
        InspectInput.model_validate(
            {"queries": [{"kind": "hierarchy", "path": str(box.deck), "simulator": "ngspice"}]}
        ),
        state,
    )

    assert result.structured_content is not None
    (item,) = result.structured_content["results"]
    assert item["ok"], item["error"]


#: The modules allowed to read ``config.allowed_paths``: the one that defines it
#: and the one that owns the reload.
_SANDBOX_OWNERS = {"config.py", "state.py"}


def test_nothing_else_reads_the_boot_sandbox():
    """``config.allowed_paths`` is the list the session opened with. A module
    that reads it instead of ``state.allowed_paths()`` misses every later edit,
    which is how a capabilities report once listed stale roots."""
    src = Path(__file__).resolve().parent.parent / "src" / "ltspice_mcp"
    offenders = []
    for module in sorted(src.rglob("*.py")):
        if module.name in _SANDBOX_OWNERS and module.parent == src:
            continue
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == "allowed_paths"
                and (
                    (isinstance(node.value, ast.Attribute) and node.value.attr == "config")
                    or (isinstance(node.value, ast.Name) and node.value.id in {"config", "cfg"})
                )
            ):
                offenders.append(f"{module.relative_to(src)}:{node.lineno}")
    assert not offenders, f"read state.allowed_paths() instead: {offenders}"


# ---------------------------------------------------------------------------
# Every refusal carries the same guidance in its structured hint
# ---------------------------------------------------------------------------


def _assert_guidance(hint: object, config_path: Path) -> None:
    """The guidance names the file, the key, and that no restart is needed."""
    assert isinstance(hint, str), hint
    assert "[security] allowed_paths" in hint
    assert str(config_path) in hint
    assert "next call" in hint


async def test_inspect_item_refusal_carries_the_guidance(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    result = await handle_inspect(
        InspectInput.model_validate(
            {
                "queries": [
                    {"kind": "components", "path": str(box.deck)},
                    {"kind": "hierarchy", "path": str(box.deck), "simulator": "ngspice"},
                ]
            }
        ),
        state,
    )

    assert result.structured_content is not None
    for item in result.structured_content["results"]:
        # The hierarchy query refuses through the same sandbox, so it reports
        # the same code, not a generic error.
        assert item["error"]["code"] == "path_denied", item
        _assert_guidance(item["error"]["hint"], box.toml)


async def test_verify_refusal_carries_the_guidance(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    result = await handle_verify_circuit(VerifyCircuitInput(path=str(box.deck)), state)

    data = result.structured_content
    assert result.is_error and data is not None
    (finding,) = data["findings"]
    assert finding["rule_id"] == "path_denied"
    _assert_guidance(finding["evidence"]["hint"], box.toml)
    _assert_guidance(data["hint"], box.toml)


async def test_verify_denied_include_carries_the_guidance(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()
    lib = box.elsewhere / "part.lib"
    lib.write_text(".subckt PART 1 2\nR9 1 2 1\n.ends\n")
    cand = box.work / "cand.cir"
    cand.write_text(f"* c\nX1 in out PART\nR1 in out 1k\n.include {lib}\n.end\n")
    ref = box.work / "ref.cir"
    ref.write_text("* r\nR1 in out 1k\n.end\n")

    result = await handle_verify_circuit(
        VerifyCircuitInput.model_validate(
            {"path": str(cand), "checks": ["compare"], "compare": {"reference": str(ref)}}
        ),
        state,
    )

    assert result.structured_content is not None
    denied = [f for f in result.structured_content["findings"] if f["rule_id"] == "path_denied"]
    assert denied
    _assert_guidance(denied[0]["evidence"]["hint"], box.toml)


async def test_run_experiments_case_refusal_carries_the_guidance(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state({"fake": FakeSim})

    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "outside-sandbox",
                "circuits": [{"path": str(box.deck)}],
                "execution": {"wait_s": 1},
            }
        ),
        state,
    )

    assert result.structured_content is not None
    (failure,) = result.structured_content["failures"]
    assert failure["code"] == "path_denied"
    _assert_guidance(failure["hint"], box.toml)


async def test_jobs_refusal_carries_the_guidance(tmp_path: Path, monkeypatch):
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    result = await handle_jobs(
        JobsInput.model_validate({"action": "list", "circuit": str(box.deck)}), state
    )

    data = result.structured_content
    assert result.is_error and data is not None
    assert data["error"]["code"] == "path_denied"
    _assert_guidance(data["hint"], box.toml)


async def test_analyze_source_refusal_is_path_denied_with_the_guidance(
    tmp_path: Path, monkeypatch
):
    """A raw path the sandbox refused is not an unavailable source: the file may
    be perfectly readable, and the remedy is the config line, not the run."""
    box = _Sandbox(tmp_path, monkeypatch)
    state = box.state()

    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(box.deck.with_suffix(".raw")), "label": "out"}],
                "recipes": [{"key": "s", "metric": "summary"}],
            }
        ),
        state,
    )

    data = result.structured_content
    assert data is not None
    (missing,) = data["coverage"]["missing_cases"]["items"]
    assert missing["code"] == "path_denied"
    _assert_guidance(missing["hint"], box.toml)
    # The call-level hint points at the row, so a caller reading only the top
    # of the envelope still finds the remedy.
    assert "missing_cases" in data["hint"]


def test_guidance_says_when_the_environment_overrides_the_file(tmp_path: Path, monkeypatch):
    """With LTSPICE_MCP_ALLOWED_PATHS set, the variable replaces the file's list,
    so pointing the agent at the file would send it to edit a line that has no
    effect."""
    box = _Sandbox(tmp_path, monkeypatch)
    monkeypatch.setenv("LTSPICE_MCP_ALLOWED_PATHS", str(box.work))
    state = box.state()

    guidance = state.sandbox_guidance()

    assert "LTSPICE_MCP_ALLOWED_PATHS" in guidance
    assert "restart" in guidance
    assert "next call" not in guidance


# ---------------------------------------------------------------------------
# An explicit Api sandbox outranks the file for the whole session
# ---------------------------------------------------------------------------


def test_an_explicit_api_sandbox_outranks_a_config_file_written_later(tmp_path: Path, monkeypatch):
    """``Api(allowed_paths=...)`` beats the file at boot, and has to keep beating
    it: a server session in the same directory writes its default config on its
    first tool call, and the API session's next path check must not trade the
    caller's list for that file's."""
    box = _Sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "detect_simulators", lambda config, diagnostics: {})
    box.toml.unlink()
    api = Api(
        working_dir=box.work,
        allowed_paths=[box.work, box.elsewhere],
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        generate_default_config(box.toml)

        collected = api.inspect(
            queries=[{"kind": "capabilities"}, {"kind": "components", "path": str(box.deck)}]
        )

        capabilities, components = collected["results"]
        assert str(box.elsewhere) in capabilities["data"]["allowed_paths"]
        assert components["ok"], components["error"]
    finally:
        api.close()


def _refusal_note(api: Api, raw: Path) -> str:
    with pytest.raises(PathSecurityError) as refused:
        api.load_raw(raw_path=raw)
    return "\n".join(getattr(refused.value, "__notes__", []))


def test_an_api_refusal_carries_the_guidance(tmp_path: Path, monkeypatch):
    """A refusal raised through the Python API reaches a script, or a run_code
    snippet, as a traceback: the guidance has to be in it."""
    box = _Sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "detect_simulators", lambda config, diagnostics: {})
    api = Api(working_dir=box.work, persist_jobs=False, preload_recent_count=0)
    try:
        note = _refusal_note(api, box.deck.with_suffix(".raw"))
    finally:
        api.close()

    _assert_guidance(note, box.toml)


def test_an_explicit_api_sandbox_refusal_names_the_argument(tmp_path: Path, monkeypatch):
    """Editing the file does nothing for a session whose sandbox was given
    explicitly, so the refusal names the argument that set it."""
    box = _Sandbox(tmp_path, monkeypatch)
    monkeypatch.setattr(engine, "detect_simulators", lambda config, diagnostics: {})
    api = Api(
        working_dir=box.work,
        allowed_paths=[box.work],
        persist_jobs=False,
        preload_recent_count=0,
    )
    try:
        note = _refusal_note(api, box.deck.with_suffix(".raw"))
    finally:
        api.close()

    assert "Api(allowed_paths=...)" in note
    assert "next call" not in note
