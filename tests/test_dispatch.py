"""Tests for tool dispatch, schema validation, and profile filtering."""

import json
import typing

import pytest
from mcp import types
from pydantic import ValidationError

from ltspice_mcp.config import VALID_PROFILES
from ltspice_mcp.tools import _base, get_tools_for_profile
from ltspice_mcp.tools.circuit import SchematicOp
from tests.conftest import resolve_local_ref


def _all_profile_defs() -> list[types.Tool]:
    """Union of tool definitions across every valid profile, deduped by name.

    The reversal/reversibility guards must see tools that only a non-"full"
    profile exposes (the six consolidated tools), so a one-way mutating tool
    there can't ship without a reviewed _TOOL_REVERSAL entry (R1-F19)."""
    seen: dict[str, types.Tool] = {}
    for profile in VALID_PROFILES:
        defs, _ = get_tools_for_profile(profile)
        for tool_def in defs:
            seen[tool_def.name] = tool_def
    return list(seen.values())


def _all_profile_declared_defs() -> list[types.Tool]:
    """Union of DISPATCH-side definitions across every profile, deduped.

    The wire tool list drops outputSchema (followups item 30); the declared
    output contract lives on the dispatch definitions, which is what the
    conformance hook validates emissions against. Contract pins on output
    shapes must read this side, not the advertised list."""
    seen: dict[str, types.Tool] = {}
    for profile in VALID_PROFILES:
        _, dispatch = get_tools_for_profile(profile)
        for registered in dispatch.values():
            seen[registered.definition.name] = registered.definition
    return list(seen.values())


class TestDispatchTable:
    def test_all_tools_wired(self):
        """Every registered tool definition should have a matching dispatch entry."""
        defs, handlers = get_tools_for_profile("full")
        expected = {tool_def.name for tool_def in defs}
        dispatched = set(handlers.keys())
        missing = expected - dispatched
        assert not missing, f"Tools defined but not dispatched: {missing}"

    def test_no_extra_handlers(self):
        """Every dispatch entry either matches a tool definition, or is a
        deprecated alias declared on the tool it dispatches to (see
        RegisteredTool.aliases — e.g. 'connect' dispatching to 'wire_pins').
        Anything else is a stray handler with no definition or alias."""
        defs, handlers = get_tools_for_profile("full")
        defined = {tool_def.name for tool_def in defs}
        extra = set(handlers.keys()) - defined
        unexplained = {name for name in extra if name not in handlers[name].aliases}
        assert not unexplained, f"Dispatched but no definition or alias: {unexplained}"

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


class TestToolAliases:
    """Deprecated former tool names (e.g. 'connect' -> 'wire_pins') stay
    callable but are not advertised — they dispatch through tool_dispatch
    without a matching entry in tool_defs (RegisteredTool.aliases)."""

    def test_connect_dispatches_to_wire_pins_handler(self):
        defs, handlers = get_tools_for_profile("full")
        def_names = {tool_def.name for tool_def in defs}
        assert "wire_pins" in def_names
        assert "connect" not in def_names, "alias must not be advertised in tool_defs"
        assert "connect" in handlers, "alias must still resolve via tool_dispatch"
        assert handlers["connect"] is handlers["wire_pins"], (
            "the 'connect' alias must dispatch to the exact same registration "
            "(same handler) as 'wire_pins'"
        )

    def test_connect_alias_present_in_agentic_profile_too(self):
        _, handlers = get_tools_for_profile("agentic")
        assert "connect" in handlers
        assert handlers["connect"] is handlers["wire_pins"]


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


class TestConsolidatedInputDocumentation:
    """The published input schema is the ONLY documentation a model gets about
    how to call a tool — it has no README and no source. A top-level argument
    with no ``description`` is therefore an argument the caller has to guess,
    and the guess is silent: it validates or it does not, with no way to learn
    what the field meant. The six consolidated tools carry the whole surface,
    so every one of their top-level fields must say what it is for."""

    def test_every_consolidated_top_level_field_is_documented(self):
        defs, _ = get_tools_for_profile("consolidated")
        assert defs, "consolidated profile exposes no tools"
        undocumented: list[str] = []
        for tool_def in defs:
            for field, prop in (tool_def.inputSchema.get("properties") or {}).items():
                if not (prop.get("description") or "").strip():
                    undocumented.append(f"{tool_def.name}.{field}")
        assert not undocumented, (
            "Consolidated tools with undocumented top-level input fields "
            f"({len(undocumented)}): {sorted(undocumented)}. Give each a "
            "Field(description=...) saying what the caller should put there."
        )


class TestToolProfiles:
    def test_full_profile_returns_all_dispatch_entries(self):
        """Every tool definition has a dispatch entry; the dispatch map may
        also carry deprecated aliases that are intentionally absent from the
        definition list (see RegisteredTool.aliases)."""
        defs, handlers = get_tools_for_profile("full")
        def_names = {tool_def.name for tool_def in defs}
        assert def_names <= set(handlers.keys())
        alias_only = set(handlers.keys()) - def_names
        assert all(name in handlers[name].aliases for name in alias_only)

    def test_agentic_profile_returns_subset(self):
        defs, handlers = get_tools_for_profile("agentic")
        agentic_names = {tool_def.name for tool_def in defs}
        assert agentic_names <= set(handlers.keys())
        alias_only = set(handlers.keys()) - agentic_names
        assert all(name in handlers[name].aliases for name in alias_only)

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
            "load_library",
            "unload_library",
            "list_libraries",
        }
        _, handlers = get_tools_for_profile("agentic")
        present = filtered_out & set(handlers.keys())
        assert not present, f"Tools that should be filtered out are present: {present}"

    def test_sweep_montecarlo_reachable_in_agentic(self):
        """Sweep and Monte Carlo must be runnable end to end in the agentic
        profile. run_sweep/run_montecarlo consume a config_id that ONLY
        configure_sweep/configure_montecarlo produce, so the config builders
        have to ship in the same profile as the runners — otherwise every
        config_id is rejected and the runners are structurally dead. Monte
        Carlo perturbation + N-run aggregation and the batch-sweep route are
        not something an agent reproduces with native file edits (unlike a
        plain LTspice .step), so they belong in the agent-facing profile."""
        _, handlers = get_tools_for_profile("agentic")
        names = set(handlers.keys())
        required = {
            "configure_sweep",
            "run_sweep",
            "configure_montecarlo",
            "run_montecarlo",
            "batch_results",
        }
        missing = required - names
        assert not missing, f"Sweep/MC chain broken in agentic, missing: {missing}"

    def test_schematic_construction_writes_in_agentic(self):
        """The schematic-construction writes stay in agentic: geometry-aware
        .asc editing (orthogonal routing, pin-collision/junction checks) is
        something an agent can't replicate by hand-writing the file, so it must
        not be dropped from the agent-facing profile."""
        construction = {
            "create_schematic",
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
# documentation: a new add_*/wire_pins/create op with no entry fails
# test_every_op_has_a_declared_inverse, so shipping a one-way mutation becomes a
# reviewed decision instead of an accident. The schematic editor once shipped
# add_net_label without remove_net_label and wire_pins without remove_wire — both
# invisible to happy-path stress tests because a missing capability has no code
# path to walk. This suite walks the op *surface* instead, where absence shows.
_DECLARED_INVERSES: dict[str, str] = {
    "add_component": "remove_component",
    "remove_component": "add_component",
    "add_net_label": "remove_net_label",
    "remove_net_label": "add_net_label",
    "wire_pins": "remove_wire",
    "remove_wire": "wire_pins",
    "add_directive": "remove_directive",
    "remove_directive": "add_directive",
    "set_component_value": _SELF_INVERSE,
    "set_component_attribute": _SELF_INVERSE,
    "move_component": _SELF_INVERSE,
}

# Deprecated ``op`` discriminator aliases: a second literal value that
# deserializes to the SAME model as its primary name (``_OpWirePins.op`` is
# ``Literal["wire_pins", "connect"]``). An alias is the same mutation under
# an old spelling, so it shares its primary's declared inverse rather than
# getting its own _DECLARED_INVERSES entry; test_every_alias_resolves_to_a_
# declared_op still forces a linkage to exist so a stray/typo'd alias can't
# silently escape the closure guard.
_OP_ALIASES: dict[str, str] = {
    "connect": "wire_pins",
}


def _schematic_op_literals() -> set[str]:
    """The ``op`` discriminator strings in the SchematicOp union, derived from
    the union itself so the test cannot silently miss a newly added op. A
    member's ``op`` field may carry more than one literal (``_OpWirePins``'s
    is ``Literal["wire_pins", "connect"]`` — the deprecated alias shares the
    model), so this collects every literal per member rather than assuming one."""
    literals: set[str] = set()
    for member in typing.get_args(SchematicOp):
        literals.update(typing.get_args(member.model_fields["op"].annotation))
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
        """Each op in the union must classify its inverse in _DECLARED_INVERSES,
        or be a declared alias of one that does (_OP_ALIASES). A new op with no
        entry fails here, forcing the author to add an inverse op (or declare
        it self-inverse) rather than ship a one-way mutation."""
        undeclared = _schematic_op_literals() - _DECLARED_INVERSES.keys() - _OP_ALIASES.keys()
        assert not undeclared, (
            f"Schematic ops with no declared inverse: {sorted(undeclared)}. "
            "Add an inverse op to the SchematicOp union (mirroring "
            "remove_wire/remove_net_label) and register the pair in "
            "_DECLARED_INVERSES, or map it to _SELF_INVERSE if re-applying it "
            "with the prior arguments undoes it."
        )

    def test_every_alias_resolves_to_a_declared_op(self):
        """Every _OP_ALIASES entry must be a real literal in the union and
        must resolve to a primary op that IS in _DECLARED_INVERSES — an alias
        pointing at an unrecognized or undeclared primary name would silently
        escape the inverse-closure guard above."""
        literals = _schematic_op_literals()
        for alias, primary in _OP_ALIASES.items():
            assert alias in literals, (
                f"Alias {alias!r} is declared in _OP_ALIASES but is not a real "
                "op literal in the SchematicOp union."
            )
            assert primary in _DECLARED_INVERSES, (
                f"Alias {alias!r} resolves to {primary!r}, which has no "
                "declared inverse in _DECLARED_INVERSES."
            )

    def test_no_stale_inverse_entries(self):
        """_DECLARED_INVERSES must not name ops that no longer exist — a stale
        entry would hide a real op that has lost its inverse."""
        stale = _DECLARED_INVERSES.keys() - _schematic_op_literals()
        assert not stale, f"_DECLARED_INVERSES names ops not in the union: {sorted(stale)}"

    def test_paired_inverses_exist_in_the_union(self):
        """Every named (non-self) inverse must be a real op in the union — the
        check that would have failed on add_net_label-without-remove_net_label
        and wire_pins-without-remove_wire."""
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
    # Schematic standalone write whose inverse is an apply_schematic_ops op.
    "wire_pins": "remove_wire op",
    # Self-inverse standalone edits (re-invoke with the prior value/state).
    "set_component_value": "re-set to prior value",
    "parameter": "re-set to prior value, or delete=true to undo an added param",
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
    # Consolidated profile (EXPERIMENTAL) — the six-tool surface. inspect is
    # read-only and needs no entry; the other five are not read-only.
    "run_experiments": "cancel via jobs; re-launch (idempotent by request_id)",
    "jobs": "cancel action; re-launch the run to reverse a cancel",
    "analyze_results": "derived artifacts; sources untouched, output regenerable",
    "edit_schematic": "compensating op batch, or restore the file natively (file-access agent)",
    "verify_circuit": "export_to:sidecar overwrites the .net, regenerable from the source; source untouched",
}


class TestMutatingToolsAreReversible:
    """Companion to TestOpInverseClosure at the TOOL level. Every registered tool
    that is not read-only must declare a reversal path in _TOOL_REVERSAL, so a new
    one-way mutating tool can't ship without a reviewed decision — the guard the
    op-union closure alone does not provide, since a standalone mutate tool added
    directly via @registry.tool lives outside the SchematicOp union."""

    def test_every_mutating_tool_declares_a_reversal(self):
        defs = _all_profile_defs()
        mutating = {d.name for d in defs if not (d.annotations and d.annotations.readOnlyHint)}
        undeclared = mutating - _TOOL_REVERSAL.keys()
        assert not undeclared, (
            f"Mutating tools with no declared reversal: {sorted(undeclared)}. "
            "Add each to _TOOL_REVERSAL naming how the mutation is undone, or — if "
            "it is a deliberately-accepted one-way mutation — note why (see the "
            "accepted-one-way entries and docs/TESTING.md)."
        )

    def test_no_stale_reversal_entries(self):
        names = {d.name for d in _all_profile_defs()}
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


def _assert_no_title_annotation(node, tool_name: str, path: str, *, in_name_map=False) -> None:
    """Assert no 'title' SCHEMA KEYWORD survives, at any depth.

    A key named 'title' inside a properties/$defs map is an argument name and
    is left alone — that distinction is the whole point of the walk.
    """
    if isinstance(node, dict):
        if in_name_map:
            for name, value in node.items():
                _assert_no_title_annotation(value, tool_name, f"{path}.{name}")
            return
        assert "title" not in node, f"{tool_name}: title annotation at {path}"
        for key, value in node.items():
            _assert_no_title_annotation(
                value,
                tool_name,
                f"{path}.{key}",
                in_name_map=key in _base._SCHEMA_NAME_MAPS,
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _assert_no_title_annotation(item, tool_name, f"{path}[{index}]")


class TestSchemaPostProcessing:
    """Verify that Pydantic-generated schemas are cleaned for MCP compatibility."""

    def test_every_ref_resolves_within_its_own_schema(self):
        """Input schemas keep $defs (followups item 30) — every $ref must be
        internal and resolve against that same schema's $defs, in every
        profile. A dangling or external ref is a schema a strict client
        cannot resolve, and at least one schema must actually use $defs so a
        silent return to inlining fails here instead of quietly re-bloating."""
        any_defs = False
        for tool_def in _all_profile_defs():
            schema = tool_def.inputSchema
            defs = schema.get("$defs", {})
            any_defs = any_defs or bool(defs)

            def walk(node, path, tool=tool_def.name, defs=defs):
                if isinstance(node, dict):
                    ref = node.get("$ref")
                    if ref is not None:
                        assert isinstance(ref, str) and ref.startswith("#/$defs/"), (
                            f"{tool}: non-local $ref {ref!r} at {path}"
                        )
                        assert ref.split("/")[-1] in defs, (
                            f"{tool}: dangling $ref {ref!r} at {path}"
                        )
                    for key, value in node.items():
                        walk(value, f"{path}.{key}")
                elif isinstance(node, list):
                    for i, item in enumerate(node):
                        walk(item, f"{path}[{i}]")

            walk(schema, "root")
        assert any_defs, "no schema uses $defs — inlining silently returned"

    def test_no_defs_entry_is_unreferenced(self):
        """A $defs entry nobody points at is pure weight on the wire. The
        schema slimmer mints definitions of its own, so a rule that stopped
        earning its keep would otherwise leave an orphan behind silently."""
        for tool_def in _all_profile_defs():
            schema = tool_def.inputSchema
            text = json.dumps(schema)
            orphans = [name for name in schema.get("$defs", {}) if f'"#/$defs/{name}"' not in text]
            assert not orphans, f"{tool_def.name}: unreferenced $defs entries {orphans}"

    def test_nullable_unions_are_folded_to_type_arrays(self):
        """``X | None`` is advertised as ``{"type": [X, "null"]}``, not as a
        two-branch ``anyOf`` of bare types. Same acceptance, far fewer
        characters; an anyOf whose branches differ only by type means the fold
        stopped running."""
        for tool_def in _all_profile_defs():

            def walk(node, path, tool=tool_def.name):
                if isinstance(node, dict):
                    branches = node.get("anyOf")
                    if isinstance(branches, list) and len(branches) > 1:
                        assert not all(
                            isinstance(b, dict) and b.keys() == {"type"} for b in branches
                        ), f"{tool}: unfolded type-only anyOf at {path}"
                    for key, value in node.items():
                        walk(value, f"{path}.{key}")
                elif isinstance(node, list):
                    for i, item in enumerate(node):
                        walk(item, f"{path}[{i}]")

            walk(tool_def.inputSchema, "root")

    def test_const_carries_no_redundant_type(self):
        """A literal already pins its own type, so the ``type`` beside a
        ``const`` narrows nothing — and the tagged unions carry one per
        member."""
        for tool_def in _all_profile_defs():

            def walk(node, path, tool=tool_def.name):
                if isinstance(node, dict):
                    if "const" in node:
                        assert "type" not in node, (
                            f"{tool}: redundant 'type' beside 'const' at {path}"
                        )
                    for key, value in node.items():
                        walk(value, f"{path}.{key}")
                elif isinstance(node, list):
                    for i, item in enumerate(node):
                        walk(item, f"{path}[{i}]")

            walk(tool_def.inputSchema, "root")

    def test_no_title_annotation_survives_in_any_profile(self):
        """Pydantic's 'title' metadata is stripped wherever it is a keyword.

        Structural, not by key name: inside a properties/$defs map the keys are
        argument names, and one of them really is called 'title'."""
        for tool_def in _all_profile_defs():
            _assert_no_title_annotation(tool_def.inputSchema, tool_def.name, "root")

    def test_a_property_actually_named_title_is_advertised(self):
        """The plot recipe takes a 'title'; the handler reads it. Stripping the
        title keyword at every level deleted the property entry too, so an
        accepted argument was in no published schema and no client could find
        it."""
        from ltspice_mcp.lib.recipes import PlotRecipe

        assert "title" in PlotRecipe.model_fields
        for tool_def in _all_profile_defs():
            if tool_def.name != "analyze_results":
                continue
            advertised = json.dumps(tool_def.inputSchema)
            assert '"title"' in advertised, (
                "analyze_results advertises no 'title' property — the plot "
                "recipe's title argument is undiscoverable again"
            )
            break
        else:  # pragma: no cover - the consolidated profile always registers it
            pytest.fail("analyze_results is not registered in any profile")

    def test_wire_tool_list_omits_output_schema(self):
        """The advertised list carries no outputSchema (followups item 30 —
        it was 84% of `jobs`); the dispatch definition keeps the declared
        shape so the conformance hook still enforces it. Both directions
        pinned, so neither side can silently regress."""
        declared = {t.name: t for t in _all_profile_declared_defs()}
        stripped = 0
        for tool_def in _all_profile_defs():
            assert tool_def.outputSchema is None, (
                f"{tool_def.name}: wire definition still advertises outputSchema"
            )
            if declared[tool_def.name].outputSchema is not None:
                stripped += 1
        assert stripped > 0, "no tool declares an output schema — hook is vacuous"

    def test_output_schema_top_level_is_object(self):
        """MCP requires outputSchema to be an object schema at the top level.

        Claude Code's client validates this literally and rejects the ENTIRE
        tools/list response when any one tool violates it, disabling every
        tool on the server for that session. The wire no longer carries
        outputSchema, but the pin stays on the declared side against the day
        it is re-exposed."""
        for tool_def in _all_profile_declared_defs():
            schema = tool_def.outputSchema
            if schema is None:
                continue
            assert schema.get("type") == "object", (
                f"{tool_def.name}: outputSchema top-level type is "
                f"{schema.get('type')!r}; MCP requires 'object'"
            )

    def test_every_output_schema_admits_warnings(self):
        """sanitize_payload can add ``warnings`` to any payload, so every schema
        must accept it.

        A tool that closes itself with additionalProperties:false and omits the
        key rejects its own response exactly when a run diverged — and a strict
        client rejects the whole tools/list over it. Registration injects the
        key so no individual tool has to remember; this pins that it reached
        every one of them, in every profile."""
        for tool_def in _all_profile_declared_defs():
            schema = tool_def.outputSchema
            if schema is None:
                continue
            declared = (schema.get("properties") or {}).get("warnings")
            assert declared is not None, (
                f"{tool_def.name}: outputSchema does not declare 'warnings'"
            )
            # A tool that owns its own warnings channel may describe it, but the
            # shape has to be the list of strings sanitize_payload writes.
            assert declared.get("type") == "array", (
                f"{tool_def.name}: 'warnings' is declared as {declared.get('type')!r}, not an array"
            )
            assert declared.get("items") == {"type": "string"}, (
                f"{tool_def.name}: 'warnings' items are {declared.get('items')!r}, not strings"
            )

    def test_nested_models_resolve_through_defs(self):
        """Nested submodels are $refs into the schema's own $defs (followups
        item 30) — the composition contract is that they resolve to full
        object schemas a local-ref-following client can read."""
        defs, _ = get_tools_for_profile("full")
        sweep_tools = [d for d in defs if d.name == "configure_sweep"]
        assert sweep_tools, "configure_sweep not found"
        schema = sweep_tools[0].inputSchema
        params_prop = schema["properties"]["parameters"]
        assert "items" in params_prop, "parameters should have items schema"
        resolved = resolve_local_ref(schema, params_prop["items"])
        assert "properties" in resolved, "nested items must resolve to an object schema"
