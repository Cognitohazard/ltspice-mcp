"""The two ways the same tool surface can be served: full and compact.

``[tools] listing`` changes only how much of each definition a client is shown
at ``tools/list``. Neither mode adds or removes a capability, both are static
listings, and ``full`` — the default — must stay byte for byte what the
registry produces, because every other contract test in the suite reads it.
"""

from __future__ import annotations

from typing import Any

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import get_tools
from ltspice_mcp.tools._base import registry
from ltspice_mcp.tools._schema import strip_argument_descriptions
from tests.conftest import REGISTERED_TOOLS


def _descriptions(node: Any, path: str = "") -> dict[str, str]:
    """Every description in a schema, keyed by where it sits."""
    found: dict[str, str] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "description" and isinstance(value, str):
                found[path or "<root>"] = value
            else:
                found |= _descriptions(value, f"{path}.{key}" if path else key)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found |= _descriptions(item, f"{path}[{index}]")
    return found


class TestFullListingIsUnchanged:
    def test_default_argument_is_the_registry_output(self):
        """Every other test in the suite calls get_tools() with no argument; the
        listing parameter must not have moved what that returns."""
        registry_defs, registry_dispatch = registry.get_tools()
        defs, dispatch = get_tools()
        assert [d.model_dump_json(by_alias=True) for d in defs] == [
            d.model_dump_json(by_alias=True) for d in registry_defs
        ]
        assert dispatch == registry_dispatch

    def test_full_is_the_same_as_the_default(self):
        assert [d.model_dump_json(by_alias=True) for d in get_tools("full")[0]] == [
            d.model_dump_json(by_alias=True) for d in get_tools()[0]
        ]


class TestCompactListing:
    def test_same_tools_and_same_tool_level_descriptions(self):
        full = {d.name: d for d in get_tools("full")[0]}
        compact = {d.name: d for d in get_tools("compact")[0]}
        assert set(compact) == set(REGISTERED_TOOLS)
        assert list(compact) == list(full), "compact must not reorder the surface"
        for name, definition in compact.items():
            assert definition.description == full[name].description

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_every_argument_description_is_gone(self, name: str):
        full = {d.name: d for d in get_tools("full")[0]}[name]
        compact = {d.name: d for d in get_tools("compact")[0]}[name]
        assert _descriptions(full.inputSchema), f"{name} advertises no descriptions to strip"
        assert _descriptions(compact.inputSchema) == {}

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_nothing_but_the_descriptions_changes(self, name: str):
        """Schema equality with the descriptions removed: structure, enums,
        defaults, required and $defs must all survive."""
        full = {d.name: d for d in get_tools("full")[0]}[name]
        compact = {d.name: d for d in get_tools("compact")[0]}[name]
        assert compact.inputSchema == strip_argument_descriptions(full.inputSchema)
        assert set(compact.inputSchema.get("$defs", {})) == set(full.inputSchema.get("$defs", {}))

    def test_compact_is_materially_smaller_than_full(self):
        """What the mode is for. docs/design/mcp_surface.md claims roughly 40%
        off; a change that left the two listings the same size would mean the
        transform stopped being applied."""

        def total(listing: str) -> int:
            return sum(
                len(d.model_dump_json(by_alias=True, exclude_none=True))
                for d in get_tools(listing)[0]
            )

        full, compact = total("full"), total("compact")
        assert compact < full * 0.75, (
            f"compact listing is {compact} chars against full's {full} — the "
            "argument-description strip is no longer paying for itself"
        )

    def test_dispatch_and_validation_are_untouched(self):
        """The published copy is filtered, never the models — a compact-mode
        client's calls are validated exactly as a full-mode client's are."""
        assert get_tools("compact")[1] == registry.get_tools()[1]

    def test_the_registry_schema_is_not_mutated_by_compaction(self):
        get_tools("compact")
        assert _descriptions(get_tools("full")[0][0].inputSchema)

    def test_an_argument_literally_named_description_survives(self):
        """The filter descends structurally: inside ``properties`` the keys are
        argument names, so a field called ``description`` is data, not prose."""
        schema = {
            "type": "object",
            "description": "the tool's own prose",
            "properties": {
                "description": {"type": "string", "description": "what this field means"},
                "nested": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "gone"}},
                },
            },
            "$defs": {"description": {"type": "string", "description": "also gone"}},
        }
        stripped = strip_argument_descriptions(schema)
        assert "description" not in stripped
        assert set(stripped["properties"]) == {"description", "nested"}
        assert stripped["properties"]["description"] == {"type": "string"}
        assert stripped["properties"]["nested"]["properties"]["title"] == {"type": "string"}
        assert set(stripped["$defs"]) == {"description"}
        assert stripped["$defs"]["description"] == {"type": "string"}


def _state(work_dir, listing: str) -> SessionState:
    return SessionState.create(
        ServerConfig(working_dir=work_dir, allowed_paths=[work_dir], tool_listing=listing),  # type: ignore[arg-type]
        available={},
    )


class TestSessionStateHonoursTheListing:
    @pytest.mark.parametrize("listing", ["full", "compact"])
    def test_state_serves_the_configured_listing(self, work_dir, listing: str):
        state = _state(work_dir, listing)
        names = [d.name for d in state.tool_defs]
        assert names == [d.name for d in get_tools(listing)[0]]

    def test_compact_state_serves_no_argument_prose(self, work_dir):
        state = _state(work_dir, "compact")
        for definition in state.tool_defs:
            assert _descriptions(definition.inputSchema) == {}
