"""A renamed argument is told to the caller that still sends the old name.

Design principle "Repeatable requests" (``docs/DESIGN.md``): a client keeps its
own copy of the tool definitions, so a renamed argument breaks it until it
reloads them. Renames are made together, in one release, and the refusal of an
old name says so.

``tests/fixtures/tool_arguments.json`` records each tool's arguments by the
place a caller writes each name (``path``, ``circuits[].id``, ``ops[].dx``: a
union's branch name is no part of a place, since a caller never writes one),
and every place where the tool retired a name, with the name that replaced
it (null for a name removed with nothing in its place). When a recorded place
no longer takes its name, though the name may live on elsewhere in the tool:

* the tool's ``renamed`` map (``reference_index.RENAMED_ARGUMENTS``, behind
  ``errors.compact_validation_error``) has that place, with the argument that
  replaced it, or its ``removed`` map (``REMOVED_ARGUMENTS``) has it, with why
  it went; and a call that sends the old name there, through the server's own
  dispatch, is refused with the new one or the reason;
* a changelog fragment under ``changelog.d/`` names the old name.

A retired place stays held to the first rule for good, and its name to being
named somewhere in the changelog. A map's keys (``assign``'s parts) are the
caller's data, written ``*``. Re-record after adding or renaming an argument:

    LTSPICE_MCP_RECORD_TOOL_ARGUMENTS=1 uv run pytest tests/test_argument_names_are_kept.py

Recording refuses a place that disappeared with no ``renamed`` entry.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

import ltspice_mcp.tools  # noqa: F401  (registers every tool)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.reference_index import REMOVED_ARGUMENTS, RENAMED_ARGUMENTS
from tests.conftest import advertised_input_schemas, dispatch

RECORD = Path(__file__).parent / "fixtures" / "tool_arguments.json"
CHANGELOG = Path(__file__).parent.parent / "CHANGELOG.md"
FRAGMENTS = Path(__file__).parent.parent / "changelog.d"
RECORDING = os.environ.get("LTSPICE_MCP_RECORD_TOOL_ARGUMENTS") == "1"


def _places(schema: dict[str, Any]) -> set[str]:
    """Every place a caller writes an argument name in an input schema."""
    places: set[str] = set()

    def walk(node: Any, at: str, seen: frozenset[str]) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, at, seen)
            return
        if not isinstance(node, dict):
            return
        ref = node.get("$ref", "")
        if ref.startswith("#/$defs/") and ref not in seen:
            walk(schema["$defs"][ref.split("/")[-1]], at, seen | {ref})
        for name, child in (node.get("properties") or {}).items():
            place = f"{at}.{name}" if at else name
            places.add(place)
            walk(child, place, seen)
        walk(node.get("items"), f"{at}[]", seen)
        if isinstance(node.get("additionalProperties"), dict):
            walk(node["additionalProperties"], f"{at}.*", seen)
        for key in ("anyOf", "oneOf", "allOf"):
            walk(node.get(key), at, seen)

    walk(schema, "", frozenset())
    return places


def _current() -> dict[str, list[str]]:
    return {tool: sorted(_places(schema)) for tool, schema in advertised_input_schemas().items()}


def _name(place: str) -> str:
    return place.rpartition(".")[2]


def _sending(place: str) -> dict[str, Any]:
    """The smallest call that writes a name at ``place``: ``sources[].raw_path``
    as ``{"sources": [{"raw_path": 1}]}``."""
    value: Any = 1
    for part in reversed(place.split(".")):
        name = part.removesuffix("[]")
        value = {name: [value] if part.endswith("[]") else value}
    return value


def _recorded() -> dict[str, Any]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _unreleased() -> str:
    """The changelog entries not yet released: the fragments under changelog.d/."""
    return "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FRAGMENTS.glob("+*.md"))
    )


def _beyond(names: dict[str, list[str]], others: dict[str, list[str]]) -> dict[str, list[str]]:
    """Per tool, the names in ``names`` that ``others`` lacks, where there are any."""
    return {
        tool: sorted(extra)
        for tool, listed in names.items()
        if (extra := set(listed) - set(others.get(tool, ())))
    }


def _record() -> None:
    previous = _recorded() if RECORD.exists() else {"arguments": {}, "retired": {}}
    current = _current()
    for tool, places in _beyond(previous["arguments"], current).items():
        unnamed = [
            place
            for place in places
            if place not in RENAMED_ARGUMENTS.get(tool, {})
            and place not in REMOVED_ARGUMENTS.get(tool, {})
        ]
        assert not unnamed, (
            f"{tool} no longer takes {unnamed} and neither RENAMED_ARGUMENTS nor "
            "REMOVED_ARGUMENTS names them: add them before recording"
        )
    # Every rename or removal the maps declare, and every one recorded before,
    # whose old place the tool does not take again.
    removals = {tool: dict.fromkeys(places) for tool, places in REMOVED_ARGUMENTS.items()}
    retired: dict[str, dict[str, str | None]] = {}
    for tool, renames in [
        *previous["retired"].items(),
        *RENAMED_ARGUMENTS.items(),
        *removals.items(),
    ]:
        for name, new in renames.items():
            if name not in current.get(tool, ()):
                retired.setdefault(tool, {})[name] = new
    record = {"arguments": current, "retired": retired}
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    RECORD.write_text(text, encoding="utf-8", newline="\n")


if RECORDING:
    _record()


def _retirements() -> list[Any]:
    """Every recorded place the tool no longer takes, and every retired one,
    with whether it is new since the record (and so not yet released)."""
    recorded = _recorded()
    cases = []
    for tool, names in _beyond(recorded["arguments"], _current()).items():
        for name in names:
            cases.append(pytest.param(tool, name, True, id=f"{tool}-{name}-unrecorded"))
    for tool, names in recorded["retired"].items():
        for name in names:
            cases.append(pytest.param(tool, name, False, id=f"{tool}-{name}"))
    return cases


def test_every_argument_name_is_recorded():
    added = _beyond(_current(), _recorded()["arguments"])
    assert not added, (
        f"New argument places {added}: record them with "
        "LTSPICE_MCP_RECORD_TOOL_ARGUMENTS=1, so a later rename of one is caught."
    )


def test_the_record_holds_the_renames_already_made():
    # A record that lost its retired names would pass every check below.
    retired = _recorded()["retired"]
    assert retired["edit_schematic"]["target"] == "path"
    assert retired["plot_waveform"]["raw_file"] == "path"
    assert retired["analyze_results"]["sources[].raw_path"] == "sources[].path"
    assert retired["plot_waveform"]["format"] is None


def test_a_place_is_spelled_as_a_caller_writes_it():
    # A union's branch is no part of a place, and a list item is "[]".
    places = set(_current()["edit_schematic"])
    assert {"ops[].dx", "ops[].labels[].x", "compare.reference"} <= places
    assert not [place for place in places if place.startswith("ops[].move")]
    assert {"circuits[].path", "variations[].assign"} <= set(_current()["run_experiments"])


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "old", "unrecorded"), _retirements())
async def test_a_retired_name_is_refused_with_its_referral_and_announced(
    state_no_sim: SessionState, tool: str, old: str, unrecorded: bool
):
    new = RENAMED_ARGUMENTS.get(tool, {}).get(old)
    why = REMOVED_ARGUMENTS.get(tool, {}).get(old)
    assert new is not None or why is not None, (
        f"{tool} no longer takes {old!r}: add it to RENAMED_ARGUMENTS with the name that "
        "replaced it, or to REMOVED_ARGUMENTS with why it went, so a client with an old "
        "copy of the tool is told"
    )
    assert new is None or new in _current()[tool], (
        f"{tool}: {old!r} is renamed to {new!r}, which it does not take"
    )

    result = await dispatch(state_no_sim, tool, _sending(old))

    assert result.is_error
    assert result.structured_content is not None
    message = result.structured_content["error"]
    told = f"is now {_name(new)!r}" if new is not None else f"was removed: {why}"
    assert f"{_name(old)!r} {told}" in message, message
    changelog = _unreleased() if unrecorded else (
        _unreleased() + CHANGELOG.read_text(encoding="utf-8")
    )
    assert f"`{_name(old)}`" in changelog or f"`{_name(old)}=" in changelog, (
        f"no fragment under changelog.d/{'' if unrecorded else ' and no section of CHANGELOG.md'}"
        f" names {tool}'s retired argument `{_name(old)}`"
    )
