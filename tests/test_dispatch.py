"""Tests for tool dispatch, schema validation, and profile filtering."""

import json
import typing

from pydantic import ValidationError

from ltspice_mcp.tools import get_tools_for_profile
from ltspice_mcp.tools.circuit import SchematicOp


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
            "create_netlist",
            "read_circuit",
            "set_component_value",
            "parameter",
            "edit_directive",
            "configure_sweep",
            "configure_montecarlo",
            "load_library",
            "unload_library",
            "list_libraries",
        }
        _, handlers = get_tools_for_profile("agentic")
        present = filtered_out & set(handlers.keys())
        assert not present, f"Tools that should be filtered out are present: {present}"

    def test_schematic_construction_writes_in_agentic(self):
        """The schematic-construction writes stay in agentic: geometry-aware
        .asc editing (orthogonal routing, pin-collision/junction checks) is
        something an agent can't replicate by hand-writing the file, so it must
        not be dropped from the agent-facing profile."""
        construction = {
            "create_schematic",
            "add_component",
            "apply_schematic_ops",
        }
        _, handlers = get_tools_for_profile("agentic")
        missing = construction - set(handlers.keys())
        assert not missing, f"Construction writes missing from agentic: {missing}"


class TestDestructiveAnnotations:
    """A tool's destructiveHint is what an MCP client gates write-risk on. A
    batch writer that can delete or overwrite must not advertise itself as
    non-destructive — especially now that the schematic writes are in the
    agent-facing profile."""

    def test_component_removing_tools_are_destructive(self):
        defs, _ = get_tools_for_profile("full")
        by_name = {d.name: d for d in defs}
        for name in ("create_schematic", "apply_schematic_ops"):
            tool = by_name[name]
            assert tool.annotations is not None
            assert tool.annotations.destructiveHint is True, f"{name} not marked destructive"
        # apply_schematic_ops earns the hint because its batch can run the
        # remove_component op (and persist a partial subset); keep the two tied
        # so the hint can't silently rot if that op is ever dropped.
        assert "remove_component" in (by_name["apply_schematic_ops"].description or "")


# A self-inverse op reverts itself: re-applying it with the prior arguments
# undoes the change (move it back, re-set the old value). Distinct from a paired
# inverse, which is a *different* op. The reversal holds when the prior state
# existed to restore — re-setting a value/position that was already there. It
# does not reconstruct a slot that was previously absent (setting an attribute
# that did not exist captures no prior value); reset_schematic is the recovery
# hatch for that residual. This table guards that an undo *capability* exists,
# not that every edit round-trips byte-for-byte.
_SELF_INVERSE = "<self>"

# Declared inverse for every schematic op. This table is a forcing function, not
# documentation: a new add_*/connect/create op with no entry fails
# test_every_op_has_a_declared_inverse, so shipping a one-way mutation becomes a
# reviewed decision instead of an accident. The schematic editor once shipped
# add_net_label without remove_net_label and connect without remove_wire — both
# invisible to happy-path stress tests because a missing capability has no code
# path to walk. This suite walks the op *surface* instead, where absence shows.
_DECLARED_INVERSES: dict[str, str] = {
    "add_component": "remove_component",
    "remove_component": "add_component",
    "add_net_label": "remove_net_label",
    "remove_net_label": "add_net_label",
    "connect": "remove_wire",
    "remove_wire": "connect",
    "add_directive": "remove_directive",
    "remove_directive": "add_directive",
    "set_component_value": _SELF_INVERSE,
    "set_component_attribute": _SELF_INVERSE,
    "move_component": _SELF_INVERSE,
}


def _schematic_op_literals() -> set[str]:
    """The ``op`` discriminator strings in the SchematicOp union, derived from
    the union itself so the test cannot silently miss a newly added op."""
    literals: set[str] = set()
    for member in typing.get_args(SchematicOp):
        (literal,) = typing.get_args(member.model_fields["op"].annotation)
        literals.add(literal)
    return literals


class TestOpInverseClosure:
    """The schematic-editing OP surface must be closed under inversion: for every
    op that mutates the .asc, an inverse op exists (or it is self-inverse). This
    is a SURFACE guard — it asserts an undo *capability* exists, not that state
    round-trips byte-for-byte (e.g. remove_component(cleanup_wires=true) drops
    wires add_component will not restore; reset_schematic is the recovery hatch
    for those). It is scoped to the apply_schematic_ops op union; the standalone
    mutating tools are guarded separately by TestMutatingToolsAreReversible.
    Absence-class gaps have no happy path to stress-test, so a missing inverse
    ships silently (as remove_wire/remove_net_label once did) — asserting the
    property over the op union catches it the moment the asymmetry lands."""

    def test_every_op_has_a_declared_inverse(self):
        """Each op in the union must classify its inverse in _DECLARED_INVERSES.
        A new op with no entry fails here, forcing the author to add an inverse
        op (or declare it self-inverse) rather than ship a one-way mutation."""
        undeclared = _schematic_op_literals() - _DECLARED_INVERSES.keys()
        assert not undeclared, (
            f"Schematic ops with no declared inverse: {sorted(undeclared)}. "
            "Add an inverse op to the SchematicOp union (mirroring "
            "remove_wire/remove_net_label) and register the pair in "
            "_DECLARED_INVERSES, or map it to _SELF_INVERSE if re-applying it "
            "with the prior arguments undoes it."
        )

    def test_no_stale_inverse_entries(self):
        """_DECLARED_INVERSES must not name ops that no longer exist — a stale
        entry would hide a real op that has lost its inverse."""
        stale = _DECLARED_INVERSES.keys() - _schematic_op_literals()
        assert not stale, f"_DECLARED_INVERSES names ops not in the union: {sorted(stale)}"

    def test_paired_inverses_exist_in_the_union(self):
        """Every named (non-self) inverse must be a real op in the union — the
        check that would have failed on add_net_label-without-remove_net_label
        and connect-without-remove_wire."""
        ops = _schematic_op_literals()
        for op, inverse in _DECLARED_INVERSES.items():
            if inverse == _SELF_INVERSE:
                continue
            assert inverse in ops, (
                f"Op {op!r} declares inverse {inverse!r}, but no such op exists "
                "in the SchematicOp union — the mutation is one-way."
            )

    def test_pairings_are_symmetric(self):
        """If A's inverse is B, B's inverse must be A — a one-directional pairing
        means one of the two directions is actually unhandled."""
        for op, inverse in _DECLARED_INVERSES.items():
            if inverse == _SELF_INVERSE:
                continue
            assert _DECLARED_INVERSES.get(inverse) == op, (
                f"Asymmetric pairing: {op!r} -> {inverse!r}, but {inverse!r} -> "
                f"{_DECLARED_INVERSES.get(inverse)!r}."
            )


# Every registered tool that can mutate state must have a reversal path, or be a
# deliberately-accepted one-way mutation. Like _DECLARED_INVERSES, this is a
# forcing function: a NEW mutating tool (a future delete_component, rename, ...)
# added with @registry.tool and no entry here fails the test, so a one-way tool
# becomes a reviewed decision instead of a silent absence-class gap. This is the
# coverage the op-union closure alone lacks — a standalone mutate tool lives
# outside the SchematicOp union. Each entry names how the mutation is undone, or
# why a one-way mutation is accepted (see docs/TESTING.md).
_TOOL_REVERSAL: dict[str, str] = {
    # Schematic op batch — per-op closure guarded by TestOpInverseClosure.
    "apply_schematic_ops": "per-op inverse (see TestOpInverseClosure)",
    # Schematic standalone writes whose inverse is an apply_schematic_ops op.
    "add_component": "remove_component op",
    "connect": "remove_wire op",
    # Self-inverse standalone edits (re-invoke with the prior value/state).
    "set_component_value": "re-set to prior value",
    "parameter": "re-set to prior value",
    "edit_directive": "action=add <-> action=remove",
    # Recovery hatch — reset_schematic IS the inverse mechanism for .asc edits.
    "reset_schematic": "reverts to the pre-edit snapshot (it is the undo)",
    # Accepted one-way mutations (documented in docs/TESTING.md).
    "create_netlist": "creates a file; deletion is a native filesystem op",
    "create_schematic": "creates a file; deletion is a native filesystem op",
    "configure_sweep": "overwrite-in-place config; a stale config is inert",
    "configure_montecarlo": "overwrite-in-place config; a stale config is inert",
    # Job lifecycle — not a file mutation; cancel / re-launch via the registry.
    "run_simulation": "cancel_job; re-launch",
    "run_sweep": "cancel_job; re-launch",
    "run_montecarlo": "cancel_job; re-launch",
    "cancel_job": "re-launch the run",
    # Library session — paired load/unload.
    "load_library": "unload_library",
    "unload_library": "load_library",
    # Export / render — emit a derived artifact (netlist, CSV, plot) from
    # existing data; the source circuit/raw is untouched, so no edit-inverse
    # applies. Not read-only because writing the artifact is an environment
    # side effect, but the output is regenerable and deletable natively.
    "export_netlist": "derived export; source .asc untouched, output regenerable",
    "export_waveform": "derived export; source raw untouched, output regenerable",
    "plot_waveform": "derived render; source raw untouched, output regenerable",
}


class TestMutatingToolsAreReversible:
    """Companion to TestOpInverseClosure at the TOOL level. Every registered tool
    that is not read-only must declare a reversal path in _TOOL_REVERSAL, so a new
    one-way mutating tool can't ship without a reviewed decision — the guard the
    op-union closure alone does not provide, since a standalone mutate tool added
    directly via @registry.tool lives outside the SchematicOp union."""

    def test_every_mutating_tool_declares_a_reversal(self):
        defs, _ = get_tools_for_profile("full")
        mutating = {d.name for d in defs if not (d.annotations and d.annotations.readOnlyHint)}
        undeclared = mutating - _TOOL_REVERSAL.keys()
        assert not undeclared, (
            f"Mutating tools with no declared reversal: {sorted(undeclared)}. "
            "Add each to _TOOL_REVERSAL naming how the mutation is undone, or — if "
            "it is a deliberately-accepted one-way mutation — note why (see the "
            "accepted-one-way entries and docs/TESTING.md)."
        )

    def test_no_stale_reversal_entries(self):
        defs, _ = get_tools_for_profile("full")
        names = {d.name for d in defs}
        stale = _TOOL_REVERSAL.keys() - names
        assert not stale, f"_TOOL_REVERSAL names tools not in the registry: {sorted(stale)}"


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
        sweep_tools = [d for d in defs if d.name == "configure_sweep"]
        assert sweep_tools, "configure_sweep not found"
        schema = sweep_tools[0].inputSchema
        # parameters property should have inlined items schema
        params_prop = schema["properties"]["parameters"]
        assert "items" in params_prop, "parameters should have items schema"
        assert "properties" in params_prop["items"], "nested items should have inlined properties"
