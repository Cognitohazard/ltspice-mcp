"""Tests for tool dispatch, schema validation, and profile filtering."""

from pydantic import ValidationError

from ltspice_mcp.tools import get_tools_for_profile


class TestDispatchTable:
    def test_all_tools_wired(self):
        """Every registered tool definition should have a matching dispatch entry."""
        defs, handlers = get_tools_for_profile("full")
        expected = {tool_def.name for tool_def in defs}
        dispatched = set(handlers.keys())
        missing = expected - dispatched
        assert not missing, f"Tools defined but not dispatched: {missing}"

    def test_no_extra_handlers(self):
        """No dispatch entries without a matching tool definition."""
        defs, handlers = get_tools_for_profile("full")
        defined = {tool_def.name for tool_def in defs}
        extra = set(handlers.keys()) - defined
        assert not extra, f"Dispatched but no definition: {extra}"

    def test_all_handlers_callable(self):
        _, handlers = get_tools_for_profile("full")
        for name, registered in handlers.items():
            assert callable(registered.handler), f"{name} handler is not callable"

    def test_required_inputs_reject_empty_args(self):
        """Tools with required fields should reject an empty argument object."""
        _, handlers = get_tools_for_profile("full")
        for name, registered in handlers.items():
            if registered.input_model is None:
                continue
            required = registered.definition.inputSchema.get("required", [])
            if not required:
                continue
            try:
                registered.input_model.model_validate({})
            except ValidationError:
                continue
            raise AssertionError(
                f"{name} accepted empty args despite required fields {required}"
            )


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
    def test_full_profile_returns_all_dispatch_entries(self):
        defs, handlers = get_tools_for_profile("full")
        assert {tool_def.name for tool_def in defs} == set(handlers.keys())

    def test_agentic_profile_returns_subset(self):
        defs, handlers = get_tools_for_profile("agentic")
        agentic_names = {tool_def.name for tool_def in defs}
        assert agentic_names == set(handlers.keys())

    def test_agentic_is_strict_subset_of_full(self):
        full_defs, _ = get_tools_for_profile("full")
        agentic_defs, _ = get_tools_for_profile("agentic")
        full_names = {tool_def.name for tool_def in full_defs}
        agentic_names = {tool_def.name for tool_def in agentic_defs}
        assert full_names > agentic_names, "agentic tools should be a strict subset of full"

    def test_unknown_profile_treated_as_full(self):
        """Unrecognized profile name should behave like 'full'."""
        full_defs, _ = get_tools_for_profile("full")
        other_defs, _ = get_tools_for_profile("nonexistent")
        assert {tool_def.name for tool_def in full_defs} == {tool_def.name for tool_def in other_defs}

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
