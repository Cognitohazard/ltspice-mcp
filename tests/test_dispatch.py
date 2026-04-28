"""Tests for tool dispatch, schema validation, and profile filtering."""

import json

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
            raise AssertionError(f"{name} accepted empty args despite required fields {required}")


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
        assert {tool_def.name for tool_def in full_defs} == {
            tool_def.name for tool_def in other_defs
        }

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


def _assert_no_key_at_depth(node, key: str, tool_name: str, path: str) -> None:
    """Recursively assert a key does not exist at any depth."""
    if isinstance(node, dict):
        assert key not in node, f"{tool_name}: '{key}' found at {path}"
        for k, v in node.items():
            _assert_no_key_at_depth(v, key, tool_name, f"{path}.{k}")
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _assert_no_key_at_depth(item, key, tool_name, f"{path}[{i}]")


class TestSchemaPostProcessing:
    """Verify that Pydantic-generated schemas are cleaned for MCP compatibility."""

    def test_no_defs_in_any_schema(self):
        """No tool schema should contain $defs after inlining."""
        defs, _ = get_tools_for_profile("full")
        for tool_def in defs:
            assert "$defs" not in tool_def.inputSchema, (
                f"{tool_def.name}: schema still contains $defs"
            )

    def test_no_title_at_any_depth(self):
        """No 'title' key should exist at any depth in any tool schema."""
        defs, _ = get_tools_for_profile("full")
        for tool_def in defs:
            _assert_no_key_at_depth(tool_def.inputSchema, "title", tool_def.name, "root")

    def test_no_ref_at_any_depth(self):
        """No '$ref' key should exist after inlining."""
        defs, _ = get_tools_for_profile("full")
        for tool_def in defs:
            schema_str = json.dumps(tool_def.inputSchema)
            assert "$ref" not in schema_str, f"{tool_def.name}: schema contains un-inlined $ref"

    def test_nested_model_inlining(self):
        """Tools with nested models should have schemas fully inlined."""
        defs, _ = get_tools_for_profile("full")
        sweep_tools = [d for d in defs if d.name == "ltspice_configure_sweep"]
        assert sweep_tools, "ltspice_configure_sweep not found"
        schema = sweep_tools[0].inputSchema
        # parameters property should have inlined items schema
        params_prop = schema["properties"]["parameters"]
        assert "items" in params_prop, "parameters should have items schema"
        assert "properties" in params_prop["items"], "nested items should have inlined properties"
