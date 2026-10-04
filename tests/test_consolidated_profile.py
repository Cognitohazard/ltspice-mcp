"""The registered tool surface: its exact membership, the annotation table, and
error hints that never name a tool a caller cannot call.
"""

from __future__ import annotations

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.server import _ERROR_HINTS, _get_error_hint
from ltspice_mcp.tools import get_tools
from tests._text import names
from tests.conftest import (
    REGISTERED_TOOLS,
    registered_tool_names,
    removed_tools_named_in,
    resolve_local_ref,
)

# The one annotation table for the surface, each row
# (readOnlyHint, destructiveHint, idempotentHint, openWorldHint).
# docs/design/mcp_surface.md section 4 carries the design table for the six.
ANNOTATIONS_TABLE: dict[str, tuple[bool, bool, bool, bool]] = {
    # Without a caller request_id the same arguments start new work, so an
    # auto-retrying client must not treat it as idempotent. It launches a
    # simulator, so it leaves the process.
    "run_experiments": (False, False, False, True),
    "jobs": (False, True, True, False),
    "analyze_results": (False, False, True, False),
    # Each call advances the sheet's revision (and the batch can remove a
    # component or rewrite the file), so it is destructive and not idempotent.
    "edit_schematic": (False, True, False, False),
    "verify_circuit": (False, True, True, False),
    # The only tool that never writes.
    "inspect": (True, False, True, False),
    # Writes an HTML file and hands it to the local desktop: a fresh artifact
    # per call, and it leaves the process.
    "plot_waveform": (False, False, False, True),
    # Runs whatever the snippet does, in a process of its own; a retry would
    # run it again.
    "run_code": (False, True, False, True),
}


class TestExposureCounts:
    """Exact membership — a tool registered by accident trips this."""

    def test_the_registry_is_exactly_the_declared_surface(self):
        # The envelope six, the plot widget, and run_code (served only while
        # the operator leaves it on, but registered always).
        assert registered_tool_names() == set(REGISTERED_TOOLS)


class TestAnnotationsTable:
    def test_every_tool_advertises_its_table_row(self):
        defs, _ = get_tools()
        actual = {}
        for tool_def in defs:
            annotations = tool_def.annotations
            assert annotations is not None, f"{tool_def.name} advertises no annotations"
            actual[tool_def.name] = (
                annotations.read_only_hint,
                annotations.destructive_hint,
                annotations.idempotent_hint,
                annotations.open_world_hint,
            )
        assert actual == ANNOTATIONS_TABLE

    def test_edit_schematic_earns_its_destructive_hint(self):
        """The hint is what a client gates write-risk on, and edit_schematic
        earns it because its batch can run remove_component. The op union names
        every op it accepts, so the hint cannot outlive the op unnoticed."""
        tool = next(d for d in get_tools()[0] if d.name == "edit_schematic")
        schema = tool.input_schema
        ops = schema["properties"]["ops"]["items"]
        branches = [resolve_local_ref(schema, branch) for branch in ops["oneOf"]]
        assert "remove_component" in {branch["properties"]["op"]["const"] for branch in branches}


class TestErrorHints:
    """Hints are recovery instructions, so they must name callable tools."""

    def test_every_hint_resolves(self):
        for err_type in _ERROR_HINTS:
            hint = _get_error_hint(err_type)
            assert isinstance(hint, str) and hint

    def test_no_hint_names_a_removed_tool(self):
        for err_type, hint in _ERROR_HINTS.items():
            named = removed_tools_named_in(hint)
            assert not named, f"{err_type.__name__} hint names removed tools {named}: {hint!r}"

    def test_every_hint_names_a_tool_the_caller_can_actually_call(self):
        """A hint that names no tool is a dead end: the caller just failed, and
        the recovery step has to be something it can invoke — on every session,
        so a tool the operator can switch off does not count."""
        served = {d.name for d in get_tools(config=ServerConfig(run_code=False))[0]}
        for err_type, hint in _ERROR_HINTS.items():
            named = {tool for tool in served if names(hint, tool)}
            assert named, f"{err_type.__name__} hint names no callable tool: {hint!r}"
