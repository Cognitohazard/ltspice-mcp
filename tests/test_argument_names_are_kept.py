"""A renamed argument is told to the caller that still sends the old name.

Design principle "Repeatable requests" (``docs/DESIGN.md``): a client keeps its
own copy of the tool definitions, so a renamed argument breaks it until it
reloads them. Renames are made together, in one release, and the refusal of an
old name says so.

``tests/fixtures/tool_arguments.json`` records every name a caller can write in
each tool's arguments, at any depth (``path``, ``circuits[].id`` as ``id``, an
op's ``dx``), and every name the tool retired, with the name that replaced it.
When a recorded name is no longer taken anywhere in its tool:

* the tool's ``renamed`` map (``reference_index.RENAMED_ARGUMENTS``, behind
  ``errors.compact_validation_error``) names the argument that replaced it,
  and a call that sends the old name, through the server's own dispatch, is
  refused with the new one;
* ``CHANGELOG.md``'s Unreleased section names the old name.

A retired name stays held to the first rule for good, and to being named
somewhere in the changelog. Re-record after adding or renaming an argument:

    LTSPICE_MCP_RECORD_TOOL_ARGUMENTS=1 uv run pytest tests/test_argument_names_are_kept.py

Recording refuses a name that disappeared with no ``renamed`` entry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import ltspice_mcp.tools  # noqa: F401  (registers every tool)
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import registry
from ltspice_mcp.tools.reference_index import RENAMED_ARGUMENTS
from tests.conftest import call_tool_params, fake_request_context

RECORD = Path(__file__).parent / "fixtures" / "tool_arguments.json"
CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"
RECORDING = os.environ.get("LTSPICE_MCP_RECORD_TOOL_ARGUMENTS") == "1"


def _names_in(schema: dict[str, Any]) -> set[str]:
    """Every property name anywhere in an input schema, through its definitions."""
    names: set[str] = set()
    seen: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/") and ref not in seen:
            seen.add(ref)
            walk(schema["$defs"][ref.split("/")[-1]])
        properties = node.get("properties")
        if isinstance(properties, dict):
            names.update(properties)
            walk(list(properties.values()))
        for key in ("items", "additionalProperties", "anyOf", "oneOf", "allOf"):
            walk(node.get(key))

    walk(schema)
    return names


def _current() -> dict[str, list[str]]:
    return {
        rt.definition.name: sorted(_names_in(rt.definition.input_schema))
        for rt in registry._registered
    }


def _recorded() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _unreleased() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    start = text.index("## [Unreleased]")
    end = text.find("\n## [", start + 1)
    return text[start : end if end != -1 else len(text)]


def _disappeared(recorded: dict[str, Any]) -> dict[str, list[str]]:
    current = _current()
    return {
        tool: sorted(set(names) - set(current.get(tool, ())))
        for tool, names in recorded["arguments"].items()
        if set(names) - set(current.get(tool, ()))
    }


def _record() -> None:
    previous = _recorded() if RECORD.exists() else {"arguments": {}, "retired": {}}
    current = _current()
    retired: dict[str, dict[str, str]] = {
        tool: dict(names) for tool, names in previous.get("retired", {}).items()
    }
    for tool, names in _disappeared(previous).items():
        for name in names:
            new = RENAMED_ARGUMENTS.get(tool, {}).get(name)
            assert new is not None, (
                f"{tool} no longer takes {name!r} and RENAMED_ARGUMENTS names nothing "
                "that replaced it: add the rename before recording"
            )
            retired.setdefault(tool, {})[name] = new
    # A rename declared in the map is a retirement too, recorded or not.
    for tool, renames in RENAMED_ARGUMENTS.items():
        for name, new in renames.items():
            if name not in current.get(tool, ()):
                retired.setdefault(tool, {})[name] = new
    for tool, names in retired.items():
        for name in [name for name in names if name in current.get(tool, ())]:
            del names[name]  # taken again: no longer retired
    record = {"arguments": current, "retired": {t: n for t, n in retired.items() if n}}
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    RECORD.write_text(text, encoding="utf-8", newline="\n")


if RECORDING:
    _record()


def _retirements() -> list[Any]:
    """Every recorded name the tool no longer takes, and every retired one,
    with whether it is new since the record (and so still Unreleased)."""
    recorded = _recorded()
    cases = []
    for tool, names in _disappeared(recorded).items():
        for name in names:
            cases.append(pytest.param(tool, name, True, id=f"{tool}-{name}-unrecorded"))
    for tool, names in recorded["retired"].items():
        for name in names:
            cases.append(pytest.param(tool, name, False, id=f"{tool}-{name}"))
    return cases


def test_every_argument_name_is_recorded():
    recorded = _recorded()["arguments"]
    current = _current()
    added = {
        tool: sorted(set(names) - set(recorded.get(tool, ())))
        for tool, names in current.items()
        if set(names) - set(recorded.get(tool, ()))
    }
    assert not added, (
        f"New argument names {added}: record them with "
        "LTSPICE_MCP_RECORD_TOOL_ARGUMENTS=1, so a later rename of one is caught."
    )


def test_the_record_holds_the_renames_already_made():
    # A record that lost its retired names would pass every check below.
    retired = _recorded()["retired"]
    assert retired["edit_schematic"]["target"] == "path"
    assert retired["plot_waveform"]["raw_file"] == "path"


@pytest.fixture
def state(config) -> SessionState:
    return SessionState.create(config, available={})


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "old", "unrecorded"), _retirements())
async def test_an_old_name_is_refused_with_the_new_one_and_announced(
    state: SessionState, tool: str, old: str, unrecorded: bool
):
    new = RENAMED_ARGUMENTS.get(tool, {}).get(old)
    assert new is not None, (
        f"{tool} no longer takes {old!r}: add it to RENAMED_ARGUMENTS with the name that "
        "replaced it, so a client with an old copy of the tool is told"
    )
    assert new in _current()[tool], (
        f"{tool}: {old!r} is renamed to {new!r}, which it does not take"
    )

    result = await call_tool(fake_request_context(state), call_tool_params(tool, {old: 1}))

    assert result.is_error
    assert result.structured_content is not None
    message = result.structured_content["error"]
    assert f"{old!r} is now {new!r}" in message, message
    changelog = _unreleased() if unrecorded else CHANGELOG.read_text(encoding="utf-8")
    assert f"`{old}`" in changelog or f"`{old}=" in changelog, (
        f"CHANGELOG.md{' (Unreleased)' if unrecorded else ''} does not name {tool}'s retired "
        f"argument `{old}`"
    )
