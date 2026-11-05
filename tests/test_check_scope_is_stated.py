"""A check that did not run is reported as not run, with the reason.

Design principle "Safe edits" (``docs/DESIGN.md``): a check says what it did
not look at, and a check that did not run is reported as not run, because a
verdict that reads as complete while it covered only part of the circuit is
worse than no verdict.

Generated from every registered tool that takes a list of checks (an array
argument named ``checks`` whose items are an enumeration): each check is asked
for on its own, and all of them by default, on a sheet and on a netlist, in a
session with no LTspice and with no comparison reference, which is where an
export and a comparison cannot run. Each call goes through the server's own
dispatch. Every check the tool offers must come back as run, as skipped with
a reason, or as a failure that names it; none may be silently absent. A check
asked for on its own that did not run leaves nothing the caller asked for, so
the call's outcome is ``failed``, never ``complete``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import ltspice_mcp.tools  # noqa: F401  (registers every tool)
from ltspice_mcp.state import SessionState
from tests.conftest import advertised_input_schemas, dispatched


def _tools_with_checks() -> dict[str, list[str]]:
    """Each tool with a ``checks`` argument that is a list of named checks,
    and the checks it names."""
    tools: dict[str, list[str]] = {}
    for tool, schema in advertised_input_schemas().items():
        spec = schema.get("properties", {}).get("checks", {})
        kinds = spec.get("type")
        if "array" in (kinds if isinstance(kinds, list) else [kinds]):
            tools[tool] = list(spec.get("items", {}).get("enum", ()))
    return tools


_TOOLS = _tools_with_checks()


def _cases() -> list[Any]:
    cases = []
    for tool, checks in _TOOLS.items():
        for kind in ("asc", "netlist"):
            cases.append(pytest.param(tool, kind, None, id=f"{tool}-{kind}-default"))
            for check in checks:
                cases.append(pytest.param(tool, kind, [check], id=f"{tool}-{kind}-{check}"))
    return cases


def test_the_tools_that_take_checks_are_found():
    # A generator that found no tool would pass every case below.
    assert _TOOLS.get("verify_circuit"), _TOOLS
    assert "export" in _TOOLS["verify_circuit"]
    assert "compare" in _TOOLS["verify_circuit"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool", "kind", "checks"), _cases())
async def test_every_check_is_run_or_said_not_to_have_run(
    asc_state: SessionState,
    asc_file: Path,
    sample_netlist: Path,
    tool: str,
    kind: str,
    checks: list[str] | None,
):
    assert "ltspice" not in asc_state.available_simulators
    path = asc_file if kind == "asc" else sample_netlist
    arguments: dict[str, Any] = {"path": str(path)}
    if checks is not None:
        arguments["checks"] = checks

    data = await dispatched(asc_state, tool, **arguments)
    ran = set(data.get("checks_run", ()))
    skipped = {row["check"]: row["reason"] for row in data.get("checks_skipped", ())}
    failed = {row["stage"] for row in data.get("failures", ())}
    for check in _TOOLS[tool]:
        assert check in ran or check in skipped or check in failed, (
            f"{tool} on a {kind} file with checks={checks}: {check!r} is neither run, "
            f"skipped nor failed: {data}"
        )
        if check in skipped:
            assert skipped[check].strip(), f"{check!r} was skipped with no reason: {data}"
    assert not ran & set(skipped), f"a check is both run and skipped: {data}"
    if checks is not None and not ran & set(checks):
        assert data["outcome"] == "failed", (
            f"{tool} on a {kind} file with checks={checks}: nothing asked for ran, "
            f"yet the outcome is {data['outcome']!r}: {data}"
        )
