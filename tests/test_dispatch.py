"""Tests for server dispatch table and tool schema validation."""

import pytest

from ltspice_mcp.server import _DISPATCH
from ltspice_mcp.tools import ALL_MODULES


class TestDispatchTable:

    def test_all_tools_wired(self):
        """Every TOOL_DEFS entry should have a matching dispatch handler."""
        expected = set()
        for mod in ALL_MODULES:
            for tool_def in mod.TOOL_DEFS:
                expected.add(tool_def.name)

        dispatched = set(_DISPATCH.keys())
        missing = expected - dispatched
        assert not missing, f"Tools defined but not dispatched: {missing}"

    def test_no_extra_handlers(self):
        """No dispatch entries without a matching tool definition."""
        defined = set()
        for mod in ALL_MODULES:
            for tool_def in mod.TOOL_DEFS:
                defined.add(tool_def.name)

        extra = set(_DISPATCH.keys()) - defined
        assert not extra, f"Dispatched but no definition: {extra}"

    def test_all_handlers_callable(self):
        for name, handler in _DISPATCH.items():
            assert callable(handler), f"{name} handler is not callable"


class TestToolSchemas:

    def test_all_schemas_valid(self):
        for mod in ALL_MODULES:
            for tool_def in mod.TOOL_DEFS:
                schema = tool_def.inputSchema
                assert schema, f"{tool_def.name}: no inputSchema"
                assert schema.get("type") == "object", (
                    f"{tool_def.name}: schema type is not 'object'"
                )
                assert "properties" in schema, (
                    f"{tool_def.name}: no properties"
                )

    def test_required_fields_in_properties(self):
        for mod in ALL_MODULES:
            for tool_def in mod.TOOL_DEFS:
                schema = tool_def.inputSchema
                required = schema.get("required", [])
                props = schema.get("properties", {})
                for req in required:
                    assert req in props, (
                        f"{tool_def.name}: required '{req}' not in properties"
                    )
