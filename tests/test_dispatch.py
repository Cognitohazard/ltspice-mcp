"""Tests for tool dispatch, schema validation, and profile filtering."""

import json
import typing

import pytest
from mcp import types
from pydantic import ValidationError

from ltspice_mcp.lib.schematic_ops import SchematicOp
from ltspice_mcp.tools import _base, _schema, get_tools
from tests.conftest import resolve_local_ref


def _all_profile_defs() -> list[types.Tool]:
    """Every advertised tool definition.

    The reversal/reversibility guards must see every registered tool, so a
    one-way mutating tool can't ship without a reviewed _TOOL_REVERSAL entry."""
    defs, _ = get_tools()
    return list(defs)


def _all_profile_declared_defs() -> list[types.Tool]:
    """Every DISPATCH-side definition.

    The tool list sent over the wire drops outputSchema; the declared
    output contract lives on the dispatch definitions, which is what the
    conformance hook validates emissions against. Contract pins on output
    shapes must read this side, not the advertised list."""
    _, dispatch = get_tools()
    return [registered.definition for registered in dispatch.values()]


class TestDispatchTable:
    def test_all_tools_wired(self):
        """Every registered tool definition should have a matching dispatch entry."""
        defs, handlers = get_tools()
        expected = {tool_def.name for tool_def in defs}
        dispatched = set(handlers.keys())
        missing = expected - dispatched
        assert not missing, f"Tools defined but not dispatched: {missing}"

    def test_no_extra_handlers(self):
        """Every dispatch entry matches a tool definition — a stray handler
        with no advertised definition would be callable but undiscoverable."""
        defs, handlers = get_tools()
        defined = {tool_def.name for tool_def in defs}
        assert set(handlers.keys()) == defined

    def test_all_handlers_callable(self):
        _, handlers = get_tools()
        for name, registered in handlers.items():
            assert callable(registered.handler), f"{name} handler is not callable"

    def test_required_inputs_reject_empty_args(self):
        """Tools with required fields should reject an empty argument object."""
        _, handlers = get_tools()
        for name, registered in handlers.items():
            if registered.input_model is None:
                continue
            required = registered.definition.input_schema.get("required", [])
            if not required:
                continue
            try:
                registered.input_model.model_validate({})
            except ValidationError:
                continue
            raise AssertionError(f"{name} accepted empty args despite required fields {required}")


class TestToolSchemas:
    def test_all_schemas_valid(self):
        defs, _ = get_tools()
        for tool_def in defs:
            schema = tool_def.input_schema
            assert schema, f"{tool_def.name}: no inputSchema"
            assert schema.get("type") == "object", f"{tool_def.name}: schema type is not 'object'"
            assert "properties" in schema, f"{tool_def.name}: no properties"

    def test_required_fields_in_properties(self):
        defs, _ = get_tools()
        for tool_def in defs:
            schema = tool_def.input_schema
            required = schema.get("required", [])
            props = schema.get("properties", {})
            for req in required:
                assert req in props, f"{tool_def.name}: required '{req}' not in properties"


class TestConsolidatedInputDocumentation:
    """Every top-level argument must say what it is for. The description on
    the model is what a client is shown, what api.reference() prints, and what
    spice://guide renders — one text, three readers — so an argument with none
    is undocumented everywhere at once. Keep each one short; the depth belongs
    in docs/design/mcp_surface.md or the guide, with a pointer left behind."""

    def test_every_consolidated_top_level_field_is_documented(self):
        _, dispatch = get_tools()
        registered = [rt.definition for rt in dispatch.values()]
        assert registered, "no tools registered"
        undocumented: list[str] = []
        for tool_def in registered:
            for field, prop in (tool_def.input_schema.get("properties") or {}).items():
                if not (prop.get("description") or "").strip():
                    undocumented.append(f"{tool_def.name}.{field}")
        assert not undocumented, (
            "Tools with undocumented top-level input fields "
            f"({len(undocumented)}): {sorted(undocumented)}. Give each a "
            "Field(description=...) saying what the caller should put there."
        )


class TestRegisteredSurface:
    def test_profile_returns_all_dispatch_entries(self):
        """Every tool definition has a dispatch entry, and vice versa."""
        defs, handlers = get_tools()
        assert {tool_def.name for tool_def in defs} == set(handlers.keys())

    def test_an_empty_registry_is_a_hard_error(self):
        """An empty surface still completes the MCP handshake, so a client reads
        it as 'this server has no capabilities' rather than 'misconfigured'.
        Registration breakage must fail loudly instead of serving nothing."""
        empty = _base.ToolRegistry()
        with pytest.raises(RuntimeError, match="zero tools"):
            empty.get_tools()


class TestDestructiveAnnotations:
    """A tool's destructiveHint is what an MCP client gates write-risk on. A
    batch writer that can delete or overwrite must not advertise itself as
    non-destructive."""

    def test_the_schematic_writer_is_destructive(self):
        defs, _ = get_tools()
        by_name = {d.name: d for d in defs}
        tool = by_name["edit_schematic"]
        assert tool.annotations is not None
        assert tool.annotations.destructive_hint is True, "edit_schematic not marked destructive"
        # edit_schematic earns the hint because its batch can run the
        # remove_component op (and commit a whole-file rewrite); keep the two
        # tied so the hint can't silently rot if that op is ever dropped. The
        # tie is the op union itself, which names every op it accepts.
        ops = tool.input_schema["properties"]["ops"]["items"]
        defs = tool.input_schema["$defs"]
        branches = [defs[branch["$ref"].split("/")[-1]] for branch in ops["oneOf"]]
        assert "remove_component" in {branch["properties"]["op"]["const"] for branch in branches}


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


def _schematic_op_literals() -> set[str]:
    """The ``op`` discriminator strings in the SchematicOp union, derived from
    the union itself so the test cannot silently miss a newly added op. A
    member's ``op`` field may carry more than one literal, so this collects
    every literal per member rather than assuming one."""
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
    # Render — emits a derived artifact from existing data; the source raw is
    # untouched, so no edit-inverse applies. Not read-only because writing the
    # artifact is an environment side effect, but the output is regenerable
    # and deletable natively.
    "plot_waveform": "derived render; source raw untouched, output regenerable",
    # The consolidated surface. inspect is read-only and needs no entry; the
    # others are not read-only.
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
        mutating = {d.name for d in defs if not (d.annotations and d.annotations.read_only_hint)}
        undeclared = mutating - _TOOL_REVERSAL.keys()
        assert not undeclared, (
            f"Mutating tools with no declared reversal: {sorted(undeclared)}. "
            "Add each to _TOOL_REVERSAL naming how the mutation is undone, or — if "
            "it is a deliberately-accepted one-way mutation — note why (see "
            "docs/TESTING.md)."
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
                in_name_map=key in _schema._SCHEMA_NAME_MAPS,
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _assert_no_title_annotation(item, tool_name, f"{path}[{index}]")


def _assert_no_null_default(node, tool_name: str, path: str, *, in_name_map=False) -> None:
    """Assert no ``"default": null`` SCHEMA KEYWORD survives, at any depth.

    A key named 'default' inside a properties/$defs map is an argument name and
    is left alone, exactly as the title walk leaves a property called 'title'.
    """
    if isinstance(node, dict):
        if in_name_map:
            for name, value in node.items():
                _assert_no_null_default(value, tool_name, f"{path}.{name}")
            return
        assert node.get("default", "absent") is not None, (
            f"{tool_name}: null default annotation at {path}"
        )
        for key, value in node.items():
            _assert_no_null_default(
                value,
                tool_name,
                f"{path}.{key}",
                in_name_map=key in _schema._SCHEMA_NAME_MAPS,
            )
    elif isinstance(node, list):
        for index, item in enumerate(node):
            _assert_no_null_default(item, tool_name, f"{path}[{index}]")


class TestSchemaPostProcessing:
    """Verify that Pydantic-generated schemas are cleaned for MCP compatibility."""

    def test_every_ref_resolves_within_its_own_schema(self):
        """Input schemas keep $defs instead of inlining them — every $ref must
        be internal and resolve against that same schema's $defs, in every
        profile. A dangling or external ref is a schema a strict client
        cannot resolve, and at least one schema must actually use $defs so a
        silent return to inlining fails here instead of quietly re-bloating."""
        any_defs = False
        for tool_def in _all_profile_defs():
            schema = tool_def.input_schema
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
            schema = tool_def.input_schema
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

            walk(tool_def.input_schema, "root")

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

            walk(tool_def.input_schema, "root")

    def test_no_null_default_is_advertised(self):
        """``"default": null`` says nothing ``required`` has not said.

        Pydantic writes one for every optional field with a ``None`` default,
        and a JSON Schema ``default`` does not constrain anything, so the
        annotation is characters every client downloads for no information.
        Non-null defaults stay — those carry the value the server uses."""
        for tool_def in _all_profile_defs():
            _assert_no_null_default(tool_def.input_schema, tool_def.name, "root")

    def test_discriminators_advertise_no_branch_mapping(self):
        """``discriminator.mapping`` repeats each branch's own ``const``.

        The useful half is ``propertyName``, which names the field that picks
        the branch; the mapping restates, once per branch, the discriminant
        value the branch already declares plus the ``$ref`` the ``oneOf`` list
        already carries.

        So the branch ``const`` is the whole justification for dropping the
        table, and it is asserted here rather than left to two spot checks
        elsewhere: a branch that stopped declaring its own discriminant would
        leave a client with no way to pick one at all.
        """
        seen = 0
        branches = 0
        for tool_def in _all_profile_defs():
            schema = tool_def.input_schema

            def walk(node, path, tool=tool_def.name, schema=schema):
                nonlocal seen, branches
                if isinstance(node, dict):
                    block = node.get("discriminator")
                    if isinstance(block, dict):
                        seen += 1
                        assert "mapping" not in block, f"{tool}: discriminator.mapping at {path}"
                        assert "propertyName" in block, (
                            f"{tool}: discriminator without propertyName at {path}"
                        )
                        tag = block["propertyName"]
                        for index, member in enumerate(node.get("oneOf", [])):
                            branches += 1
                            resolved = resolve_local_ref(schema, member)
                            tagged = resolved.get("properties", {}).get(tag)
                            assert isinstance(tagged, dict) and "const" in tagged, (
                                f"{tool}: {path}.oneOf[{index}] declares no {tag!r} const, "
                                "so nothing tells a client which branch it is"
                            )
                    for key, value in node.items():
                        walk(value, f"{path}.{key}")
                elif isinstance(node, list):
                    for index, item in enumerate(node):
                        walk(item, f"{path}[{index}]")

            walk(schema, "root")
        assert seen, "no discriminator survives — the tagged unions stopped being advertised"
        assert branches > seen, "a tagged union with no branches is not a union"

    def test_a_property_actually_named_default_would_survive(self):
        """Both new passes descend structurally, like the title stripper.

        Filtering by key name at every level would delete an argument called
        ``default`` (or a ``discriminator`` object a caller sends), which is
        exactly the bug the title stripper already had once."""
        schema = {
            "properties": {
                "default": {"type": "string", "default": None},
                "discriminator": {"type": "object"},
            },
            "default": None,
        }
        cleaned = _schema._drop_discriminator_mappings(_schema._drop_null_defaults(schema))
        assert set(cleaned["properties"]) == {"default", "discriminator"}
        assert cleaned["properties"]["default"] == {"type": "string"}
        assert cleaned["properties"]["discriminator"] == {"type": "object"}
        assert "default" not in cleaned

    def test_no_title_annotation_survives_in_any_profile(self):
        """Pydantic's 'title' metadata is stripped wherever it is a keyword.

        Structural, not by key name: inside a properties/$defs map the keys are
        argument names, and one of them really is called 'title'."""
        for tool_def in _all_profile_defs():
            _assert_no_title_annotation(tool_def.input_schema, tool_def.name, "root")

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
            advertised = json.dumps(tool_def.input_schema)
            assert '"title"' in advertised, (
                "analyze_results advertises no 'title' property — the plot "
                "recipe's title argument is undiscoverable again"
            )
            break
        else:  # pragma: no cover - the consolidated profile always registers it
            pytest.fail("analyze_results is not registered in any profile")

    def test_wire_tool_list_omits_output_schema(self):
        """The advertised list carries no outputSchema (it was 84% of the
        `jobs` entry); the dispatch definition keeps the declared
        shape so the conformance hook still enforces it. Both directions
        pinned, so neither side can silently regress."""
        declared = {t.name: t for t in _all_profile_declared_defs()}
        stripped = 0
        for tool_def in _all_profile_defs():
            assert tool_def.output_schema is None, (
                f"{tool_def.name}: wire definition still advertises outputSchema"
            )
            if declared[tool_def.name].output_schema is not None:
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
            schema = tool_def.output_schema
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
            schema = tool_def.output_schema
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
        """Nested submodels are $refs into the schema's own $defs — the
        composition contract is that they resolve to full object schemas a
        local-ref-following client can read."""
        defs, _ = get_tools()
        experiment_tools = [d for d in defs if d.name == "run_experiments"]
        assert experiment_tools, "run_experiments not found"
        schema = experiment_tools[0].input_schema
        circuits_prop = schema["properties"]["circuits"]
        assert "items" in circuits_prop, "circuits should have items schema"
        resolved = resolve_local_ref(schema, circuits_prop["items"])
        assert "properties" in resolved, "nested items must resolve to an object schema"


class TestAdvertisedOrderIsStable:
    def test_tool_list_order_is_pinned(self):
        """Servers should return tools/list in a deterministic order — clients
        cache the list and LLM prompt caching keys on the exact bytes. Ours is
        registration order, fixed by the sorted module imports in
        tools/__init__; this pin turns an accidental reorder (a set, a dict
        rebuild, an import shuffle) into a failure instead of a silent
        cache-buster for every connected client."""
        names = [t.name for t in get_tools()[0]]
        assert names == [
            "plot_waveform",
            "analyze_results",
            "run_experiments",
            "jobs",
            "inspect",
            "edit_schematic",
            "verify_circuit",
        ]


class TestLayering:
    """``lib`` must not import ``tools``.

    The layering is not decoration: ``tools/_base`` imports half of ``lib``, so
    a ``lib`` module reaching back up closes a cycle. It resolves silently
    whenever a tool module happens to be imported first and explodes on the one
    entry point that imports the lib module first — which is how the schematic
    engine's move out of ``tools/`` shipped a latent ImportError that only one
    test file's import order revealed.
    """

    def test_no_lib_module_imports_the_tool_layer(self):
        import ast
        from pathlib import Path

        lib = Path(__file__).resolve().parent.parent / "src" / "ltspice_mcp" / "lib"
        offenders: list[str] = []
        for path in sorted(lib.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
                    "ltspice_mcp.tools"
                ):
                    offenders.append(f"{path.name}:{node.lineno} -> {node.module}")
                elif isinstance(node, ast.Import):
                    offenders.extend(
                        f"{path.name}:{node.lineno} -> {alias.name}"
                        for alias in node.names
                        if alias.name.startswith("ltspice_mcp.tools")
                    )
        assert not offenders, "lib modules importing the tool layer: " + "; ".join(offenders)
