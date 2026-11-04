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
a reason, or as a failure that names it; none may be silently absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import ltspice_mcp.tools  # noqa: F401  (registers every tool)
from ltspice_mcp.server import call_tool
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import registry
from tests.conftest import call_tool_params, fake_request_context

_NETLIST = "* divider\nV1 in 0 5\nR1 in out 1k\nR2 out 0 2k\n.op\n.end\n"


def _enum_in(node: Any) -> list[str]:
    """The first enumeration anywhere under a schema node."""
    if isinstance(node, dict):
        if "enum" in node:
            return list(node["enum"])
        for value in node.values():
            found = _enum_in(value)
            if found:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _enum_in(value)
            if found:
                return found
    return []


def _tools_with_checks() -> dict[str, list[str]]:
    tools: dict[str, list[str]] = {}
    for registered in registry._registered:
        spec = registered.definition.input_schema.get("properties", {}).get("checks")
        if spec is not None and "array" in str(spec):
            tools[registered.definition.name] = _enum_in(spec)
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
    work_dir: Path,
    tool: str,
    kind: str,
    checks: list[str] | None,
):
    assert "ltspice" not in asc_state.available_simulators
    if kind == "asc":
        path = work_dir / "Draft1.asc"
    else:
        path = work_dir / "divider.cir"
        path.write_text(_NETLIST, encoding="utf-8", newline="\n")
    arguments: dict[str, Any] = {"path": str(path)}
    if checks is not None:
        arguments["checks"] = checks

    result = await call_tool(fake_request_context(asc_state), call_tool_params(tool, arguments))

    data = result.structured_content
    assert data is not None
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
