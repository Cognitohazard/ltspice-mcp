"""Tests for tool dispatch, schema validation, and profile filtering."""

import asyncio

from ltspice_mcp.tools import AGENTIC_TOOLS, ALL_MODULES, get_tools_for_profile


class TestDispatchTable:
    def test_all_tools_wired(self):
        """Every TOOL_DEFS entry should have a matching dispatch handler."""
        defs, handlers = get_tools_for_profile("full")
        expected = {td.name for td in defs}
        dispatched = set(handlers.keys())
        missing = expected - dispatched
        assert not missing, f"Tools defined but not dispatched: {missing}"

    def test_no_extra_handlers(self):
        """No dispatch entries without a matching tool definition."""
        defs, handlers = get_tools_for_profile("full")
        defined = {td.name for td in defs}
        extra = set(handlers.keys()) - defined
        assert not extra, f"Dispatched but no definition: {extra}"

    def test_all_handlers_callable(self):
        _, handlers = get_tools_for_profile("full")
        for name, handler in handlers.items():
            assert callable(handler), f"{name} handler is not callable"

    def test_handlers_reject_missing_args(self):
        """Each handler raises on empty args — no silent success on bad input."""
        _, handlers = get_tools_for_profile("full")
        loop = asyncio.new_event_loop()
        try:
            for name, handler in handlers.items():
                raised = False
                try:
                    loop.run_until_complete(handler({}, None))
                except Exception:
                    raised = True
                assert raised, f"{name} handler accepted empty args without raising"
        finally:
            loop.close()


class TestToolSchemas:
    def test_all_schemas_valid(self):
        defs, _ = get_tools_for_profile("full")
        for tool_def in defs:
            schema = tool_def.inputSchema
            assert schema, f"{tool_def.name}: no inputSchema"
            assert schema.get("type") == "object", f"{tool_def.name}: schema type is not 'object'"
            assert "properties" in schema, f"{tool_def.name}: no properties"

    def test_required_fields_in_properties(self):
        defs, _ = get_tools_for_profile("full")
        for tool_def in defs:
            schema = tool_def.inputSchema
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for req in required:
                assert req in props, f"{tool_def.name}: required '{req}' not in properties"


class TestToolProfiles:
    def test_full_profile_returns_all_tools(self):
        """Full profile should return every tool from every module."""
        defs, handlers = get_tools_for_profile("full")
        all_names = set()
        for mod in ALL_MODULES:
            for td in mod.TOOL_DEFS:
                all_names.add(td.name)
        assert {td.name for td in defs} == all_names
        assert set(handlers.keys()) == all_names

    def test_agentic_profile_returns_subset(self):
        """Agentic profile should return exactly the AGENTIC_TOOLS set."""
        defs, handlers = get_tools_for_profile("agentic")
        assert {td.name for td in defs} == AGENTIC_TOOLS
        assert set(handlers.keys()) == AGENTIC_TOOLS

    def test_agentic_is_strict_subset_of_full(self):
        full_defs, _ = get_tools_for_profile("full")
        full_names = {td.name for td in full_defs}
        assert full_names > AGENTIC_TOOLS, "AGENTIC_TOOLS should be a strict subset of full"

    def test_agentic_tools_all_exist(self):
        """Every name in AGENTIC_TOOLS must correspond to a real tool."""
        full_defs, _ = get_tools_for_profile("full")
        full_names = {td.name for td in full_defs}
        missing = AGENTIC_TOOLS - full_names
        assert not missing, f"AGENTIC_TOOLS references non-existent tools: {missing}"

    def test_unknown_profile_treated_as_full(self):
        """Unrecognized profile name should behave like 'full'."""
        full_defs, _ = get_tools_for_profile("full")
        other_defs, _ = get_tools_for_profile("nonexistent")
        assert {td.name for td in full_defs} == {td.name for td in other_defs}

    def test_agentic_tool_count(self):
        """Agentic profile tool count should match AGENTIC_TOOLS."""
        defs, _ = get_tools_for_profile("agentic")
        assert len(defs) == len(AGENTIC_TOOLS)

    def test_filtered_tools_not_in_agentic(self):
        """Verify specific tools that should NOT be in agentic profile."""
        filtered_out = {
            "ltspice_create_netlist",
            "ltspice_read_circuit",
            "ltspice_set_component_value",
            "ltspice_parameter",
            "ltspice_edit_directive",
            "ltspice_configure_sweep",
            "ltspice_configure_montecarlo",
            "ltspice_remove_component",
            "ltspice_move_component",
            "ltspice_set_component_attribute",
            "ltspice_load_library",
            "ltspice_unload_library",
            "ltspice_list_libraries",
        }
        _, handlers = get_tools_for_profile("agentic")
        present = filtered_out & set(handlers.keys())
        assert not present, f"Tools that should be filtered out are present: {present}"
