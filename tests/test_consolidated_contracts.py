"""Cross-cutting contract battery over the six consolidated tools.

Forcing functions for the shared conventions in mcp_v1_design.md sections 2/3
that no single per-tool test guards: the page-object shape, the findings shape,
channel separation, the error-object envelope where declared, outcome presence,
callable pagination, isError vs per-item failure isolation, stable error codes,
and the bounded-parse routing of every untrusted raw/log parse. A seventh tool
or a regression in any of the six trips one of these.

Uniform envelope (asserted, not merely documented): all six tools carry a
call-level ``outcome`` drawn from the single ratified vocabulary
``complete|partial|failed|in_progress`` — no per-tool dialect survives. Every
outcome enum a tool declares must be a subset of that vocabulary, and every
top-level error object must carry the full ``{code, message, stage, retryable,
commit_state}`` envelope on the failure paths that declare one. inspect is a
per-item read batch: its call-level outcome is ``complete`` when every query
succeeds and ``partial`` when any isolates a failure.
"""

from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.errors import NetlistError, ResultError
from ltspice_mcp.lib import result_store
from ltspice_mcp.lib.pin_legend import PageCursorError, paginate_view
from ltspice_mcp.tools import get_tools_for_profile
from ltspice_mcp.tools.experiments import _decode_jobs_cursor
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.schematic_edit import _validate_view_cursors, _ViewCursors
from ltspice_mcp.tools.verify import VerifyCircuitInput, handle_verify_circuit

# CONSOLIDATED_TOOLS = the six envelope ops; REGISTERED_TOOLS adds the plot
# widget, which is registered by ruling but predates the shared response
# envelope — it joins the surface completeness and size pins (every client
# pays its schema), not the envelope contract matrix. Shared in conftest so
# every file naming the surface reads one constant.
from tests.conftest import CONSOLIDATED_TOOLS, REGISTERED_TOOLS

# The single ratified outcome vocabulary (design section 2). No per-tool dialect
# is allowed: every outcome enum any of the six declares must be a subset.
CONTRACT_OUTCOMES = frozenset({"complete", "partial", "failed", "in_progress"})


def _registered() -> dict[str, Any]:
    defs, _ = get_tools_for_profile("consolidated")
    return {tool_def.name: tool_def for tool_def in defs}


def _source_definitions() -> dict[str, Any]:
    """The dispatch-side definitions — full prose and outputSchema intact."""
    _, dispatch = get_tools_for_profile("consolidated")
    return {name: rt.definition for name, rt in dispatch.items()}


def _output_schemas() -> dict[str, dict[str, Any]]:
    # The wire defs drop outputSchema; the declared shapes live on the
    # dispatch-side definitions (keyed here by wire names so deprecated
    # aliases can't double-count). Non-None is asserted so every consumer
    # gets a plain dict; the named per-tool pin is TestOutputSchemaCoverage.
    _, dispatch = get_tools_for_profile("consolidated")
    schemas: dict[str, dict[str, Any]] = {}
    for name in _registered():
        schema = dispatch[name].definition.outputSchema
        assert schema is not None, f"{name}: consolidated tool must declare an outputSchema"
        schemas[name] = schema
    return schemas


def _input_schemas() -> dict[str, dict[str, Any]]:
    return {name: tool_def.inputSchema for name, tool_def in _registered().items()}


def _schema_variants(schema: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """The schema itself plus each top-level oneOf branch (jobs is a oneOf)."""
    yield schema
    for branch in schema.get("oneOf", []):
        if isinstance(branch, dict):
            yield branch


def _property_objects(node: Any) -> Iterator[dict[str, Any]]:
    """Every dict node that declares a JSON-Schema ``properties`` map, recursively."""
    if isinstance(node, dict):
        if isinstance(node.get("properties"), dict):
            yield node
        for value in node.values():
            yield from _property_objects(value)
    elif isinstance(node, list):
        for item in node:
            yield from _property_objects(item)


def _property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    for obj in _property_objects(schema):
        names.update(obj["properties"].keys())
    return names


def _as_type_set(node: dict[str, Any]) -> set[str]:
    declared = node.get("type")
    if isinstance(declared, str):
        return {declared}
    if isinstance(declared, list):
        return set(declared)
    return set()


def test_budget_owner_matrix_pins_the_complete_consolidated_surface():
    schemas = _input_schemas()
    assert set(schemas) == set(REGISTERED_TOOLS)
    owners = {name for name, schema in schemas.items() if "budget" in schema.get("properties", {})}
    assert owners == {"run_experiments", "jobs", "analyze_results", "inspect"}


class TestPageObjectShape:
    """Every page declares {items, total, returned, truncated} (design section 2)."""

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_pages_declare_the_full_page_shape(self, name: str):
        schema = _output_schemas()[name]
        assert schema is not None
        page_count = 0
        for obj in _property_objects(schema):
            props = obj["properties"]
            # A page is an object with both an ``items`` and a ``returned``
            # property (``returned`` disambiguates from unrelated ``items``).
            if "items" in props and "returned" in props:
                page_count += 1
                for field in ("items", "total", "returned", "truncated"):
                    assert field in props, f"{name}: page object missing {field!r}"
        # The tools that surface a canonical page object must declare it well.
        # inspect is excluded: it is a per-item results batch whose pagination
        # lives as a per-item ``page`` {total,returned,truncated} + next_cursor,
        # not a nested {items,...} page object (its callability is covered by
        # TestCallablePagination).
        if name in {"jobs", "analyze_results", "edit_schematic"}:
            assert page_count >= 1, f"{name}: expected at least one page object"


class TestFindingsShape:
    """Every finding carries at + subject so an agent can locate what to fix."""

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_findings_require_at_and_subject(self, name: str):
        schema = _output_schemas()[name]
        for obj in _property_objects(schema):
            if "rule_id" not in obj["properties"]:
                continue
            required = set(obj.get("required", []))
            assert {"at", "subject"} <= required, (
                f"{name}: a findings object omits at/subject from required "
                f"(has {sorted(required)})"
            )


class TestChannelSeparation:
    """observations / warnings / failures are distinct arrays, never merged."""

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_each_channel_is_a_separate_array(self, name: str):
        schema = _output_schemas()[name]
        for obj in _property_objects(schema):
            props = obj["properties"]
            for channel in ("observations", "warnings", "failures"):
                if channel not in props:
                    continue
                assert "array" in _as_type_set(props[channel]), (
                    f"{name}: channel {channel!r} is not an array — a merged or "
                    "retyped channel violates the three-channel contract"
                )


class TestOutcomeEnvelope:
    """All six tools carry a call-level outcome from the one ratified vocabulary."""

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_declares_a_non_empty_outcome_enum(self, name: str):
        schema = _output_schemas()[name]
        enums = [
            obj["properties"]["outcome"].get("enum")
            for obj in _property_objects(schema)
            if "outcome" in obj["properties"]
        ]
        assert enums, f"{name}: no outcome property declared"
        assert all(e for e in enums), f"{name}: an outcome property has an empty enum"

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_outcome_enum_is_a_subset_of_the_ratified_vocabulary(self, name: str):
        schema = _output_schemas()[name]
        declared_something = False
        for obj in _property_objects(schema):
            enum = obj["properties"].get("outcome", {}).get("enum")
            if not enum:
                continue
            declared_something = True
            stray = set(enum) - CONTRACT_OUTCOMES
            assert not stray, (
                f"{name}: outcome enum declares {sorted(stray)} outside the ratified "
                f"vocabulary {sorted(CONTRACT_OUTCOMES)}"
            )
        assert declared_something, f"{name}: no outcome enum declared"


class TestErrorEnvelope:
    """Error objects carry stable, machine-actionable fields."""

    def _top_error_objects(self, name: str) -> list[dict[str, Any]]:
        found = []
        for variant in _schema_variants(_output_schemas()[name]):
            props = variant.get("properties", {})
            err = props.get("error")
            if isinstance(err, dict) and isinstance(err.get("properties"), dict):
                found.append(err)
        return found

    _FULL_ENVELOPE = frozenset({"code", "message", "stage", "retryable", "commit_state"})

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_every_top_level_error_object_is_the_full_envelope(self, name: str):
        # Uniform failure-shape contract: wherever a tool declares a top-level
        # error object it carries the full {code, message, stage, retryable,
        # commit_state} envelope, so a caller can decide whether re-submission
        # is safe (design section 2). Tools that route failures through findings
        # or per-item channels declare no top-level error object and pass vacuously.
        for err in self._top_error_objects(name):
            required = set(err.get("required", []))
            assert required >= self._FULL_ENVELOPE, (
                f"{name}: error envelope missing fields (has {sorted(required)})"
            )

    @pytest.mark.parametrize("name", ["run_experiments", "jobs", "edit_schematic"])
    def test_write_and_execute_planes_declare_a_full_error_envelope(self, name: str):
        # The durable EXECUTE plane and the transactional AUTHOR write both MUST
        # surface a top-level error object, not just may (design section 2).
        errors = self._top_error_objects(name)
        assert errors, f"{name}: no top-level error object declared"
        for err in errors:
            required = set(err.get("required", []))
            assert required >= self._FULL_ENVELOPE, (
                f"{name}: error envelope missing fields (has {sorted(required)})"
            )


_CURSOR_INPUT_FIELDS = frozenset({"cursor", "continue", "continuation", "view_cursors"})


class TestCallablePagination:
    """A tool that returns a next_cursor accepts a matching cursor input (R2-N7)."""

    @pytest.mark.parametrize("name", ["jobs", "analyze_results", "inspect", "edit_schematic"])
    def test_paginated_tools_accept_a_cursor_input(self, name: str):
        out_props = _property_names(_output_schemas()[name])
        assert "next_cursor" in out_props, f"{name}: expected a paginated output"
        in_props = _property_names(_input_schemas()[name])
        assert _CURSOR_INPUT_FIELDS & in_props, (
            f"{name}: returns a next_cursor but declares no cursor input — "
            "pagination must be callable, not declarative"
        )


class TestOutputSchemaCoverage:
    """Every consolidated tool declares an outputSchema, so the session-wide
    conformance hook (conftest, now iterating the profile union) validates every
    structuredContent the six emit."""

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_declares_an_output_schema(self, name: str):
        assert _output_schemas()[name] is not None


# Serialized size, in characters, of each tool's advertised definition — the
# name, description and inputSchema a client loads before it can call anything.
# Every session pays it whether or not the tool is used, so each has an upper
# bound: a field, an option or a sentence that grows past it fails here, and
# the bound is then raised deliberately, in the same change that earns it. A
# shrink needs no re-pin.
#
# What the wire carries: each tool's own description verbatim (a client that
# does not show server instructions has nothing else to route on), field
# descriptions only when they carry a unit/convention/inversion/pointer marker
# (_WIRE_PROSE_KEEP in tools/_base.py), and no outputSchema. The full prose
# stays on the registered definition, api.reference(), and spice://guide.
_SURFACE_BUDGET_CHARS: dict[str, int] = {
    # Variations, attached analysis, and the receipt row shape.
    "run_experiments": 7516,
    # Five actions and the receipt/page shapes.
    "jobs": 1724,
    # Twenty-odd recipe branches; the largest schema on the surface.
    "analyze_results": 16897,
    # Five query kinds, each with its own argument shape.
    "inspect": 4991,
    # The typed op union plus render/compare views.
    "edit_schematic": 6760,
    # Checks, the render policy object, and the boolean render shorthand.
    "verify_circuit": 2535,
    # Job/case addressing, windowing, and delivery flags.
    "plot_waveform": 2174,
}

# Recipe branches no recorded workload has ever called (measured over 477
# campaign transcripts, both doors — .claude/plans/u3_recipe_census.md). Their
# advertised schema is a discriminant-plus-pointer stub; the branch itself
# stays fully callable. A metric may join this tuple only with a fresh census
# showing zero use; a metric the census showed used may never be stubbed.
DORMANT_WIRE_STUBS: tuple[str, ...] = ("noise_integral", "periodic", "return_loss")


def _recipe_def_name(metric: str) -> str:
    """The $defs key for one recipe metric, read off the live union."""
    from ltspice_mcp.lib.recipes import RECIPE_MODELS, _discriminant_of

    return next(m for m in RECIPE_MODELS if _discriminant_of(m) == metric).__name__


class TestDormantRecipeWireStubs:
    """The dormant-branch diet: the wire shrinks, the capability does not."""

    @staticmethod
    def _analyze_defs() -> dict[str, Any]:
        return _registered()["analyze_results"].inputSchema["$defs"]

    @pytest.mark.parametrize("metric", DORMANT_WIRE_STUBS)
    def test_stubbed_branch_advertises_only_discriminant_and_pointer(self, metric: str):
        body = self._analyze_defs()[_recipe_def_name(metric)]
        assert set(body["properties"]) == {"metric"}
        assert body["properties"]["metric"]["const"] == metric
        # Both discovery channels are named, and the stub must stay permissive
        # so a client pre-validating a full call against the wire still sends it.
        assert "api.reference('analyze_results')" in body["description"]
        assert "spice://guide" in body["description"]
        assert "additionalProperties" not in body

    def test_no_branch_the_census_showed_used_is_stubbed(self):
        from ltspice_mcp.lib.recipes import RECIPE_MODELS, _discriminant_of

        defs = self._analyze_defs()
        for model in RECIPE_MODELS:
            metric = _discriminant_of(model)
            if metric in DORMANT_WIRE_STUBS:
                continue
            body = defs[model.__name__]
            assert set(body.get("properties", {})) != {"metric"}, (
                f"{metric} is stubbed on the wire but is not in DORMANT_WIRE_STUBS — "
                "stubbing a used branch needs a fresh census, not just the model config"
            )

    def test_stubbed_branches_still_validate_their_full_field_tree(self):
        from ltspice_mcp.lib.recipes import RECIPE_ADAPTER

        full_calls = [
            {
                "key": "n",
                "metric": "noise_integral",
                "signal": "V(onoise)",
                "from_hz": 10.0,
                "to_hz": "1Meg",
            },
            {
                "key": "p",
                "metric": "periodic",
                "signal": "V(out)",
                "window": {"start": 1e-3, "end": 2e-3},
                "reduce": ["mean"],
                "reduce_field": "frequency",
            },
            {"key": "r", "metric": "return_loss", "signal": "V(in)/I(Rin)", "z0": 75.0},
        ]
        for call in full_calls:
            recipe = RECIPE_ADAPTER.validate_python(call)
            assert recipe.key == call["key"]

    @staticmethod
    def _member_segment(text: str, metric: str) -> str:
        """The catalogue lines belonging to one recipe-union member."""
        marker = f"metric='{metric}'"
        rest = text[text.index(marker) + len(marker) :]
        nxt = rest.find("metric='")
        return rest if nxt < 0 else rest[:nxt]

    def test_reference_still_renders_the_stubbed_field_trees(self):
        from ltspice_mcp.api import _reference
        from ltspice_mcp.lib.recipes import _DORMANT_POINTER

        text = _reference.reference("analyze_results")
        # The catalogue renders from the live models; the wire stub appearing
        # here would mean the renderer started reading the advertised schema —
        # exactly the regression that would also erase the field trees.
        assert _DORMANT_POINTER not in text
        for metric, fields in {
            "noise_integral": ("from_hz", "to_hz"),
            "periodic": ("signal", "window"),
            "return_loss": ("z0",),
        }.items():
            segment = self._member_segment(text, metric)
            for field in fields:
                assert field in segment, f"{metric} lost {field} in the catalogue"


class TestSemanticsOnlyWire:
    """The advertised wire serves only load-bearing prose; the depth stays at
    the source. Pins both halves of that split so neither can silently rot:
    a description reaching the wire without a marker means the strip stopped
    running; a source definition losing its prose means the depth channels
    (api.reference, spice://guide, the doc gates) went blind."""

    @staticmethod
    def _descriptions(node: Any) -> Iterator[str]:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    yield value
                else:
                    yield from TestSemanticsOnlyWire._descriptions(value)
        elif isinstance(node, list):
            for item in node:
                yield from TestSemanticsOnlyWire._descriptions(item)

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_every_advertised_description_carries_a_keep_marker(self, name: str):
        from ltspice_mcp.tools._base import _WIRE_PROSE_KEEP

        tool_def = _registered()[name]
        # The tool's own description ships verbatim: a client that does not
        # surface server instructions has nothing else to route on.
        assert tool_def.description == _source_definitions()[name].description
        for text in self._descriptions(tool_def.inputSchema):
            assert _WIRE_PROSE_KEEP.search(text), (
                f"{name}: advertised field description without a unit/convention/"
                f"pointer marker reached the wire: {text[:120]!r}"
            )

    def test_the_source_definition_keeps_prose_the_wire_dropped(self):
        source = _source_definitions()["edit_schematic"]
        advertised = _registered()["edit_schematic"]
        source_ops = source.inputSchema["properties"]["ops"].get("description") or ""
        advertised_ops = advertised.inputSchema["properties"]["ops"].get("description")
        assert "remove_component" in source_ops
        assert advertised_ops is None, (
            "the ops description carries no keep marker, so the wire copy "
            "should have dropped it — the strip is not running"
        )

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_structure_survives_the_strip(self, name: str):
        """Names, enums, and defaults are untouchable — only prose moves."""
        from ltspice_mcp.tools._base import _strip_wire_prose

        source = _source_definitions()[name].inputSchema
        advertised = _registered()[name].inputSchema

        def skeleton(node: Any) -> Any:
            if isinstance(node, dict):
                return {k: skeleton(v) for k, v in node.items() if k != "description"}
            if isinstance(node, list):
                return [skeleton(v) for v in node]
            return node

        assert skeleton(advertised) == skeleton(source)
        # And the advertised copy is exactly the strip of the source — no
        # second transformation hiding in the pipeline.
        assert advertised == _strip_wire_prose(source)


def _wire_sizes() -> dict[str, int]:
    """Serialized length of each advertised definition, as a client receives it."""
    return {
        name: len(tool_def.model_dump_json(by_alias=True, exclude_none=True))
        for name, tool_def in _registered().items()
    }


class TestAdvertisedSurfaceBudget:
    """Each advertised tool definition stays under its size bound."""

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_tool_stays_within_its_pin(self, name: str):
        actual = _wire_sizes()[name]
        budget = _SURFACE_BUDGET_CHARS[name]
        assert actual <= budget, (
            f"{name}: advertised definition grew to {actual} chars (pinned at "
            f"{budget}). Every client pays this before calling anything — either "
            "spend the growth somewhere else in the schema or raise the pin "
            "deliberately."
        )

    def test_whole_surface_is_pinned(self):
        assert set(_wire_sizes()) == set(_SURFACE_BUDGET_CHARS), (
            "the consolidated profile changed shape — every advertised tool needs a size pin"
        )


class TestStableErrorCodesAndIsError:
    """path_denied is the canonical code; isError is call-level only."""

    async def test_verify_path_denied_is_a_call_level_failure(self, state_no_sim):
        result = await handle_verify_circuit(
            VerifyCircuitInput(path="/etc/definitely_denied.cir"), state_no_sim
        )
        assert result.isError is True
        data = result.structuredContent
        assert data is not None
        codes = {finding["rule_id"] for finding in data["findings"]}
        assert "path_denied" in codes
        assert data["outcome"] == "failed"

    async def test_inspect_path_denied_isolates_to_the_item(self, state_no_sim):
        result = await handle_inspect(
            InspectInput.model_validate(
                {"queries": [{"kind": "components", "path": "/etc/denied.net"}]}
            ),
            state_no_sim,
        )
        # A per-item failure is NOT a call-level failure.
        assert not result.isError
        data = result.structuredContent
        assert data is not None
        # ...but it does move the call-level outcome to partial.
        assert data["outcome"] == "partial"
        (item,) = data["results"]
        assert item["ok"] is False
        assert item["error"]["code"] == "path_denied"


class TestCursorTamperOnEveryPaginatedInput:
    """Every paginated surface rejects a malformed/tampered/cross-bound cursor
    through its own decode path rather than crashing."""

    def test_jobs_cursor_rejects_garbage(self):
        from ltspice_mcp.tools.experiments import _JobsActionError

        with pytest.raises(_JobsActionError) as exc:
            _decode_jobs_cursor("not-a-cursor")
        assert exc.value.code == "invalid_cursor"

    def test_analyze_cursor_rejects_garbage(self):
        with pytest.raises(ResultError):
            result_store.cursor_result_set_id("not-a-cursor")

    def test_shared_view_paginator_rejects_garbage(self):
        # Backs both inspect (net/symbols/components/model) and edit_schematic
        # view pages through the shared codec.
        with pytest.raises(PageCursorError):
            paginate_view([1, 2, 3], "bind:x", cursor="tampered", limit=2)

    def test_shared_view_paginator_rejects_cross_binding(self):
        good = paginate_view([1, 2, 3, 4], "bind:a", cursor=None, limit=2)
        cursor = good["next_cursor"]
        assert cursor is not None
        with pytest.raises(PageCursorError):
            paginate_view([1, 2, 3, 4], "bind:b", cursor=cursor, limit=2)

    def test_edit_schematic_view_cursor_rejects_garbage(self):
        with pytest.raises(NetlistError):
            _validate_view_cursors(_ViewCursors(pin_legend="tampered"))


# ---------------------------------------------------------------------------
# Static backstop: untrusted raw/log parses route through services.bounded_parse
# ---------------------------------------------------------------------------

# Raw readers that must never be constructed directly in a tool module — every
# raw read goes through the bounded services.load_raw wrapper.
_FORBIDDEN_DIRECT = {"OffsetAwareRawRead", "RawRead", "load_raw_sync"}
# Log/step file parsers that read an untrusted artifact and must be invoked only
# inside a services.bounded_parse thunk (deadline + cooldown).
_MUST_BE_BOUNDED = {"parse_step_iterations", "extract_log_diagnostics"}
_BOUNDED_WRAPPERS = {"bounded_parse", "load_raw"}

_SIX_MODULES = (
    "experiments",
    "analyze",
    "schematic_edit",
    "verify",
    "inspect_tools",
)


def _module_source(mod: str) -> tuple[Path, str]:
    import importlib

    module = importlib.import_module(f"ltspice_mcp.tools.{mod}")
    assert module.__file__ is not None
    path = Path(module.__file__)
    return path, path.read_text(encoding="utf-8")


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


class TestBoundedParseBackstop:
    """A static forcing function (backstop; behavior proof lives in the U1/U4
    injected-slow-parser tests). Every raw/log parse in the six tool modules
    routes through services.bounded_parse (directly, via a thunk, or via the
    bounded services.load_raw wrapper)."""

    @pytest.mark.parametrize("mod", _SIX_MODULES)
    def test_no_direct_raw_reader_construction(self, mod: str):
        path, source = _module_source(mod)
        tree = ast.parse(source, filename=str(path))
        offenders: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name is not None and name in _FORBIDDEN_DIRECT:
                offenders.add(name)
        assert not offenders, (
            f"{mod}: constructs a raw reader directly {sorted(offenders)} — "
            "route raw reads through services.load_raw (bounded)."
        )

    @pytest.mark.parametrize("mod", _SIX_MODULES)
    def test_log_parsers_are_bounded(self, mod: str):
        path, source = _module_source(mod)
        tree = ast.parse(source, filename=str(path))
        parents: dict[int, ast.AST] = {}
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                parents[id(child)] = node

        def _is_bounded(call: ast.Call) -> bool:
            cur: ast.AST | None = call
            while cur is not None:
                if (
                    isinstance(cur, ast.Call)
                    and cur is not call
                    and _call_name(cur) in _BOUNDED_WRAPPERS
                ):
                    return True
                cur = parents.get(id(cur))
            return False

        unbounded: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name is not None and name in _MUST_BE_BOUNDED and not _is_bounded(node):
                unbounded.add(name)
        assert not unbounded, (
            f"{mod}: calls a log parser {sorted(unbounded)} outside a "
            "services.bounded_parse thunk — untrusted parses need a deadline."
        )
