"""Every argument refusal names the offending field and the way round.

Design principle "Validate before write" (``docs/DESIGN.md``): a refusal names
the field, the place and the way round, such as the fields an argument accepts.

The cases are generated from every registered tool's advertised input schema
and from the reference table behind ``inspect(kind="reference")``, and each is
sent through the server's own dispatch (``server.call_tool``), so a tool or an
argument added later is covered with no edit here. Three mistakes are sent:

* a misspelled top-level field: the refusal names the misspelling and the field
  that was meant;
* a call missing a required field: the refusal names the field where it is
  missing, and for a field that chooses a branch (``jobs``' ``action``) the
  values it takes;
* for a tool whose arguments are actions, a field sent with an action that does
  not take it: the refusal names the field and every field that action takes.

``tests/test_validation_errors.py`` pins the wording of particular refusals;
this module holds every tool to the rule.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

import ltspice_mcp.tools  # noqa: F401  (registers every tool)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.reference_index import build_index
from tests.conftest import advertised_input_schemas, dispatch


def _branch_fields() -> dict[str, dict[str, dict[str, tuple[str, ...]]]]:
    """For each tool whose own top-level field chooses a branch, that field
    and each branch's top-level fields: ``{"jobs": {"action": {"list":
    ("circuit", ...)}}}``.

    A field chooses a branch when its advertised values are exactly the names
    of one family of the tool's branches in the reference table.
    """
    families: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for entry in build_index():
        if entry.family != "argument":
            families.setdefault(entry.tool, {}).setdefault(entry.family, {})[entry.name] = tuple(
                field.name for field in entry.fields if "." not in field.name
            )
    chosen: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {}
    for tool, schema in advertised_input_schemas().items():
        for field, spec in schema.get("properties", {}).items():
            values = set(spec.get("enum", ()))
            if not values:
                continue
            for branches in families.get(tool, {}).values():
                if values == set(branches):
                    chosen.setdefault(tool, {})[field] = branches
    return chosen


_SCHEMAS = advertised_input_schemas()
_BRANCHES = _branch_fields()


def _misspelled(field: str) -> str:
    """A typo of ``field``: its last letter doubled."""
    return field + field[-1]


def _misspelling_cases() -> list[Any]:
    cases = []
    for tool, schema in _SCHEMAS.items():
        choosers = _BRANCHES.get(tool, {})
        for field in schema.get("properties", {}):
            arguments: dict[str, Any] = {_misspelled(field): 1}
            if choosers and field not in choosers:
                # The misspelling is checked inside a branch that takes the field.
                chooser, branches = next(iter(choosers.items()))
                taking = [name for name, fields in branches.items() if field in fields]
                assert taking, f"{tool}: no {chooser} takes {field!r}"
                arguments[chooser] = taking[0]
            cases.append(pytest.param(tool, field, arguments, id=f"{tool}-{field}"))
    return cases


def _missing_cases() -> list[Any]:
    """Every required field, at the top level and one level down (as the
    reference table lists them), with the call nested just deep enough to
    reach it."""
    cases = []
    for tool, schema in _SCHEMAS.items():
        for field in schema.get("required", ()):
            cases.append(pytest.param(tool, field, {}, field, id=f"{tool}-{field}"))
    for entry in build_index():
        if entry.family != "argument":
            continue
        for field in entry.fields:
            if not field.required or "." not in field.name:
                continue
            outer, _, inner = field.name.rpartition(".")
            if "." in outer:
                continue
            listed = outer.endswith("[]")
            name = outer.removesuffix("[]")
            arguments: dict[str, Any] = {name: [{}] if listed else {}}
            where = f"{name}.0.{inner}" if listed else f"{name}.{inner}"
            cases.append(
                pytest.param(entry.tool, inner, arguments, where, id=f"{entry.tool}-{field.name}")
            )
    return cases


def _mismatch_cases() -> list[Any]:
    cases = []
    for tool, choosers in _BRANCHES.items():
        properties = _SCHEMAS[tool].get("properties", {})
        for chooser, branches in choosers.items():
            for branch, fields in branches.items():
                for field in properties:
                    if field == chooser or field in fields:
                        continue
                    cases.append(
                        pytest.param(
                            tool,
                            {chooser: branch, field: 1},
                            field,
                            fields,
                            id=f"{tool}-{branch}-{field}",
                        )
                    )
    return cases


async def _refusal(state: SessionState, tool: str, arguments: dict[str, Any]) -> str:
    """The message of the refusal, as a structured-aware client reads it."""
    result = await dispatch(state, tool, arguments)
    assert result.is_error, f"{tool} accepted {arguments}"
    data = result.structured_content
    assert data is not None
    assert data["code"] == "invalid_arguments", data
    return data["error"]


def _names(message: str, name: str) -> bool:
    return re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", message) is not None


def test_the_cases_cover_every_tool_and_the_action_tools():
    # A generator that found nothing would pass every check below.
    covered = {case.values[0] for case in _misspelling_cases()}
    assert covered == set(_SCHEMAS)
    assert "jobs" in _BRANCHES and "action" in _BRANCHES["jobs"]
    assert _mismatch_cases()


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "field", "arguments"), _misspelling_cases())
async def test_a_misspelled_field_is_refused_with_the_field_it_meant(
    state_no_sim: SessionState, tool: str, field: str, arguments: dict[str, Any]
):
    message = await _refusal(state_no_sim, tool, arguments)

    typo = _misspelled(field)
    assert _names(message, typo), f"{tool}: the refusal does not name {typo!r}: {message}"
    assert _names(message, field), (
        f"{tool}: the refusal of {typo!r} does not name {field!r}, the field it meant: {message}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "field", "arguments", "where"), _missing_cases())
async def test_a_missing_required_field_is_named_where_it_is_missing(
    state_no_sim: SessionState, tool: str, field: str, arguments: dict[str, Any], where: str
):
    message = await _refusal(state_no_sim, tool, arguments)

    assert re.search(rf"(?<![\w.]){re.escape(where)}: Field required", message), (
        f"{tool}: the refusal does not say {where!r} is required: {message}"
    )
    branches = _BRANCHES.get(tool, {}).get(field)
    for branch in branches or ():
        assert _names(message, branch), (
            f"{tool}: the refusal of a missing {field!r} does not offer {branch!r}: {message}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "arguments", "field", "accepted"), _mismatch_cases())
async def test_a_field_sent_with_an_action_that_does_not_take_it_names_what_it_takes(
    state_no_sim: SessionState,
    tool: str,
    arguments: dict[str, Any],
    field: str,
    accepted: tuple[str, ...],
):
    message = await _refusal(state_no_sim, tool, arguments)

    assert _names(message, field), f"{tool}: the refusal does not name {field!r}: {message}"
    missing = [name for name in accepted if not _names(message, name)]
    assert not missing, (
        f"{tool} {arguments}: the refusal does not name {missing}, which the action takes: "
        f"{message}"
    )
