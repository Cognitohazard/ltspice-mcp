"""Guards that the output-schema conformance hook actually protects, and that
the end-to-end scenario script only calls tools that still exist.

The session-scoped hook in ``conftest.py`` validates every ``structuredContent``
emission against its tool's declared ``output_schema``. It can quietly stop
protecting anything in two ways: the handler-frame walk finds no match and
skips validation, or the response-helper patch never takes effect and the
unpatched helpers run. The first test pins both at once: a rejection is only
possible when the patched helper runs AND the frame walk attributes the
emission, so forcing a real schema violation through a registered handler
proves the whole chain is armed.

The second test keeps ``scenario_active_filter.py`` honest: a renamed or
removed tool would leave the script calling a name the server no longer
answers to, and the script isn't exercised by the normal suite.
"""

import re
from pathlib import Path
from typing import cast

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools_for_profile
from ltspice_mcp.tools.status import ServerStatusInput, handle_server_status


async def test_hook_rejects_a_schema_violating_emission(state_no_sim: SessionState):
    """A registered handler emitting structuredContent that contradicts its
    output_schema is caught at emission time, not silently passed through."""
    # server_status types ``diagnostics`` as an array of strings; seeding it
    # with a non-string makes the real handler emit a genuinely non-conforming
    # payload through the (patched) format_response. The cast injects the bad
    # value deliberately — the whole point is that it is NOT a valid list[str].
    state_no_sim.diagnostics = cast("list[str]", [123])
    with pytest.raises(AssertionError, match="output_schema"):
        await handle_server_status(ServerStatusInput(), state_no_sim)


def _scenario_tool_names() -> set[str]:
    """Tool names passed to ``session.call_tool(...)`` in the scenario script."""
    source = (Path(__file__).parent / "scenario_active_filter.py").read_text()
    return set(re.findall(r"call_tool\(\s*[\"']([A-Za-z_]\w*)[\"']", source))


def test_scenario_calls_only_registered_tools():
    """Every tool the scenario script drives must exist in the live registry
    (aliases included), so a rename can't leave the script calling a dead name."""
    _, dispatch = get_tools_for_profile("full")
    called = _scenario_tool_names()
    assert called, "parsed no call_tool names from the scenario — the regex has rotted"
    unknown = sorted(called - set(dispatch))
    assert not unknown, f"scenario calls tools absent from the registry: {unknown}"
