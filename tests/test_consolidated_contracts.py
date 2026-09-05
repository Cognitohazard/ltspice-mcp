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

import jsonschema
import pytest

from ltspice_mcp.errors import NetlistError, ResultError, SimulationError
from ltspice_mcp.lib import result_store
from ltspice_mcp.lib.pin_legend import PageCursorError, paginate_view
from ltspice_mcp.lib.recipes import DISCRIMINANTS
from ltspice_mcp.tools import _base, get_tools
from ltspice_mcp.tools.experiments import (
    AttachedAnalysis,
    RunExperimentsInput,
    _validate_attached_analysis,
)
from ltspice_mcp.tools.inspect_tools import InspectInput, handle_inspect
from ltspice_mcp.tools.jobs import _decode_jobs_cursor
from ltspice_mcp.tools.schematic_edit import EditViewCursors, _validate_view_cursors
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
    defs, _ = get_tools()
    return {tool_def.name: tool_def for tool_def in defs}


def _source_definitions() -> dict[str, Any]:
    """The dispatch-side definitions — full prose and outputSchema intact."""
    _, dispatch = get_tools()
    return {name: rt.definition for name, rt in dispatch.items()}


def _output_schemas() -> dict[str, dict[str, Any]]:
    # The wire defs drop outputSchema; the declared shapes live on the
    # dispatch-side definitions (keyed here by wire names so deprecated
    # aliases can't double-count). Non-None is asserted so every consumer
    # gets a plain dict; the named per-tool pin is TestOutputSchemaCoverage.
    _, dispatch = get_tools()
    schemas: dict[str, dict[str, Any]] = {}
    for name in _registered():
        schema = dispatch[name].definition.output_schema
        assert schema is not None, f"{name}: consolidated tool must declare an outputSchema"
        schemas[name] = schema
    return schemas


def _input_schemas() -> dict[str, dict[str, Any]]:
    return {name: tool_def.input_schema for name, tool_def in _registered().items()}


class TestAttachedRecipeGrammar:
    """``run_experiments.analyze.recipes`` IS an ``analyze_results`` request.

    Advertised as a bare object, the grammar was discoverable only from the
    other tool, and a typo in it surfaced at the analysis stage — after the
    whole simulation had run. Advertised as a second copy of the recipe union,
    it cost 13 KB on every session to restate what that tool already publishes
    on the same wire. What ships is what a caller cannot derive: the metric
    names, the key every recipe carries, and a pointer to the field trees. The
    model behind it is the union either way.
    """

    @staticmethod
    def _attached_recipe_items(schema: dict[str, Any]) -> dict[str, Any]:
        """The advertised shape of one attached recipe."""
        # 'analyze' is optional, so the block sits in a nullable branch.
        ref = next(
            item["$ref"] for item in schema["properties"]["analyze"]["anyOf"] if "$ref" in item
        )
        attached = schema["$defs"][ref.split("/")[-1]]
        return attached["properties"]["recipes"]["items"]

    def test_the_advertised_metrics_are_exactly_the_recipe_union(self):
        """Derived from the union on both sides, so a new recipe metric that
        never reaches this enum fails here instead of being unmentionable in
        an attached block."""
        items = self._attached_recipe_items(_registered()["run_experiments"].input_schema)
        assert set(items["properties"]["metric"]["enum"]) == set(DISCRIMINANTS)

        analyze = _registered()["analyze_results"].input_schema
        standalone = analyze["properties"]["recipes"]["items"]
        # The branches themselves, not a discriminator mapping: the advertised
        # schema drops that table because each branch's own 'metric' const
        # already carries the value it maps.
        branches = [
            analyze["$defs"][branch["$ref"].split("/")[-1]] for branch in standalone["oneOf"]
        ]
        assert {branch["properties"]["metric"]["const"] for branch in branches} == set(
            DISCRIMINANTS
        )

    def test_the_stub_points_at_the_channels_that_carry_the_fields(self):
        items = self._attached_recipe_items(_registered()["run_experiments"].input_schema)
        description = items.get("description") or ""
        assert "analyze_results.recipes" in description
        assert "api.reference('analyze_results')" in description
        assert "spice://guide" in description
        # Permissive on purpose, like the dormant recipe stubs: a client
        # pre-validating a full recipe against this shape must still send it.
        assert "additionalProperties" not in items
        assert set(items["required"]) == {"key", "metric"}

    def test_a_full_recipe_is_still_what_the_tool_takes(self):
        """The stub narrows the advertisement, never the accepted call.

        All three gates a full recipe passes through: the published schema
        (which the server SDK checks before dispatch), the input model, and the
        submission gate.
        """
        payload = {
            "request_id": "attached-full-recipe",
            "circuits": [{"path": "dut.cir"}],
            "analyze": {
                "recipes": [
                    {
                        "key": "pm",
                        "metric": "stability",
                        "signal": "V(out)",
                        "reduce": ["min"],
                        "field": "phase_margin_deg",
                        "spec": {"min": 45.0},
                    }
                ]
            },
        }
        jsonschema.validate(
            instance=payload,
            schema=_registered()["run_experiments"].input_schema,
        )
        args = RunExperimentsInput.model_validate(payload)
        assert args.analyze is not None
        _validate_attached_analysis(args.analyze)

    @pytest.mark.parametrize(
        "recipe",
        [
            {"key": "x", "metric": "no_such_metric"},
            {"key": "x", "metric": "stability", "signal": "V(out)", "not_a_field": 1},
            {"metric": "summary"},
        ],
        ids=["unknown-metric", "unknown-field", "no-key"],
    )
    def test_a_malformed_recipe_is_refused_before_anything_is_staged(self, recipe: dict):
        """The submission gate, which runs before a deck is staged or a case
        submitted (the no-simulation half is pinned in test_run_experiments.py).
        """
        block = AttachedAnalysis.model_validate({"recipes": [recipe]})
        with pytest.raises(SimulationError, match="attached analyze block"):
            _validate_attached_analysis(block)


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
    """observations / warnings / failures are distinct arrays, never merged.

    The channel list comes from ``_base.ENVELOPE_CHANNELS`` rather than being
    restated here, so the type that declares the envelope is what this checks.
    """

    @pytest.mark.parametrize("name", CONSOLIDATED_TOOLS)
    def test_each_channel_is_a_separate_array(self, name: str):
        schema = _output_schemas()[name]
        for obj in _property_objects(schema):
            props = obj["properties"]
            for channel in _base.ENVELOPE_CHANNELS:
                if channel not in props:
                    continue
                assert "array" in _as_type_set(props[channel]), (
                    f"{name}: channel {channel!r} is not an array — a merged or "
                    "retyped channel violates the three-channel contract"
                )


def _module_source(mod: str) -> tuple[Path, str]:
    import importlib

    module = importlib.import_module(f"ltspice_mcp.tools.{mod}")
    assert module.__file__ is not None
    path = Path(module.__file__)
    return path, path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# The outcome scan: which modules, and what counts as naming an outcome
# ---------------------------------------------------------------------------

# ``_base`` owns the rule, so it is the one module allowed to spell the
# vocabulary out; everything else has to call it.
_OUTCOME_RULE_MODULE = "_base"


def _tools_module_closure(seeds: Iterator[str] | list[str]) -> tuple[str, ...]:
    """Every ``ltspice_mcp.tools`` module reachable from the seed modules.

    Derived rather than listed: ``jobs`` and ``run_experiments`` build their
    envelopes in ``receipts``, so scanning only the modules that carry
    ``@registry.tool`` would miss where their outcome is actually decided.
    """
    seen: set[str] = set()
    queue = list(seeds)
    while queue:
        name = queue.pop()
        if name in seen or name == _OUTCOME_RULE_MODULE:
            continue
        seen.add(name)
        _, source = _module_source(name)
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module == "ltspice_mcp.tools":
                queue += [alias.name for alias in node.names]
            elif node.module and node.module.startswith("ltspice_mcp.tools."):
                queue.append(node.module.split(".")[-1])
    return tuple(sorted(seen))


def _handler_modules() -> list[str]:
    """The module each registered consolidated tool's handler lives in."""
    _, dispatch = get_tools()
    return [dispatch[name].handler.__module__.split(".")[-1] for name in CONSOLIDATED_TOOLS]


_OUTCOME_MODULES = _tools_module_closure(_handler_modules())


def _outcome_value_nodes(tree: ast.AST) -> Iterator[ast.expr]:
    """Every expression that becomes a payload's ``outcome``.

    A dict entry, a keyword argument, an assignment to ``outcome`` or to
    ``…["outcome"]``, and what an ``…_outcome`` helper returns. A bare
    ``"failed"`` elsewhere is a job status that happens to share a spelling, and
    an ``enum`` list is the declared vocabulary — neither is a decision.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "outcome":
                    yield value
        elif isinstance(node, ast.keyword) and node.arg == "outcome":
            yield node.value
        elif isinstance(node, ast.Assign) and node.value is not None:
            for target in node.targets:
                if (isinstance(target, ast.Name) and target.id == "outcome") or (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "outcome"
                ):
                    yield node.value
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.endswith(
            "_outcome"
        ):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Return) and inner.value is not None:
                    yield inner.value


def _binds_an_outcome(tree: ast.AST) -> bool:
    return any(True for _ in _outcome_value_nodes(tree))


def _spelled_out(value: ast.expr) -> Iterator[ast.Constant]:
    """The outcome names an expression states directly, ladder branches included.

    Only the shapes a hand-rolled verdict takes: the bare literal and the
    conditional/boolean trees it hides in. A call is NOT descended into —
    ``outcome_schema("complete", "partial")`` declares the vocabulary and
    ``outcome_of(...)`` applies the rule; neither is a tool deciding for itself.
    """
    if isinstance(value, ast.Constant) and value.value in CONTRACT_OUTCOMES:
        yield value
    elif isinstance(value, ast.IfExp):
        yield from _spelled_out(value.body)
        yield from _spelled_out(value.orelse)
    elif isinstance(value, ast.BoolOp):
        for operand in value.values:
            yield from _spelled_out(operand)


def _outcome_literals(source: str) -> list[tuple[int, str]]:
    """Outcome names spelled out where the shared rule should have decided them."""
    return [
        (node.lineno, str(node.value))
        for value in _outcome_value_nodes(ast.parse(source))
        for node in _spelled_out(value)
    ]


class TestSharedOutcomeRule:
    """One rule decides every call-level outcome, and every tool routes through it.

    The vocabulary test below pins what a tool may SAY; this pins how it
    decides. Six hand-written ladders agreeing today is not the same as one
    rule they all call, and the ladders are where a dialect creeps in.
    """

    def test_base_owns_the_vocabulary_the_contract_pins(self):
        assert set(_base.CONTRACT_OUTCOMES) == CONTRACT_OUTCOMES

    def test_the_envelope_type_declares_the_five_shared_keys(self):
        assert set(_base.ENVELOPE_KEYS) == {
            "outcome",
            "failures",
            "observations",
            "warnings",
            "hint",
        }
        assert set(_base.ENVELOPE_CHANNELS) < set(_base.ENVELOPE_KEYS)

    def test_outcome_schema_refuses_a_value_outside_the_vocabulary(self):
        with pytest.raises(ValueError, match="ratified vocabulary"):
            _base.outcome_schema("complete", "mostly_fine")

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({}, "complete"),
            ({"partial": True}, "partial"),
            ({"failures": [1]}, "partial"),
            ({"failures": [1], "delivered": False}, "failed"),
            ({"failures": [1], "in_progress": True}, "in_progress"),
            ({"partial": True, "in_progress": True}, "in_progress"),
        ],
    )
    def test_the_rule_ranks_in_progress_over_failure_over_shortfall(
        self, kwargs: dict[str, Any], expected: str
    ):
        failures = kwargs.pop("failures", [])
        assert _base.outcome_of(failures, **kwargs) == expected

    @pytest.mark.parametrize("module", _OUTCOME_MODULES)
    def test_no_handler_module_writes_an_outcome_literal(self, module: str):
        """Source-level, because the point is that no tool re-derives the
        vocabulary: a re-added ``"outcome": "complete"`` fails here even while
        it happens to agree with the rule it replaced.

        The module list is DERIVED from the registered tools (see
        ``_OUTCOME_MODULES``), so a tool that never routed through the rule
        cannot be missing from a hand-written list the way edit_schematic was:
        it is scanned because it is registered.
        """
        _, source = _module_source(module)
        offenders = sorted(f"line {line}: {value!r}" for line, value in _outcome_literals(source))
        assert not offenders, (
            f"{module} names an outcome with a literal ({'; '.join(offenders)}) — "
            "_base.outcome_of is what decides it"
        )

    @pytest.mark.parametrize("module", _OUTCOME_MODULES)
    def test_every_module_that_names_an_outcome_calls_the_shared_rule(self, module: str):
        _, source = _module_source(module)
        tree = ast.parse(source)
        if not _binds_an_outcome(tree):
            pytest.skip(f"{module} does not build an outcome")
        called = {_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        assert "outcome_of" in called, f"{module} builds an outcome without calling outcome_of"


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
# The rule these bounds enforce: source equals wire. Every description a model
# declares ships verbatim — nothing is filtered on the way out — so a
# description is kept short at the source, and the depth it cannot hold moves
# to docs/design/mcp_surface.md or spice://guide with a pointer left behind.
# The only thing the advertised copy drops is outputSchema.
#
# The bounds below were re-pinned when the prose filter was removed. About half
# of what a client now loads is structure the discriminated unions cannot say
# in fewer characters; the rest is the 210 descriptions inside those schemas
# plus the seven tool descriptions. Each bound also covers the tool's display
# title, which the 2026-07-28 revision puts on the wire: 24 to 26 characters
# apiece, added to every number below when titles were introduced.
_SURFACE_BUDGET_CHARS: dict[str, int] = {
    # Variations, attached analysis, and the receipt row shape. The attached
    # recipes advertise their metric names and a pointer, not a second copy of
    # the recipe branches: a client cannot resolve a $ref into another tool's
    # document, so carrying the grammar twice measured 13 KB more on every
    # session, to restate what analyze_results publishes on the same wire.
    # The .step selection hoisted off the recipes lands here too, so an
    # attached analysis and a standalone one read the same steps: two more
    # arguments and one shared StepSelector definition.
    # Raised deliberately by about 120 characters to give 'combine' its own
    # description and to say what a random variation's 'rules' list holds:
    # 'combine' had none at all, so 'zip' was an enum member no channel
    # defined, and 'rules' claimed a one-entry-per-kind rule that does not
    # exist.
    "run_experiments": 13580,
    # Five actions, each advertised as its own branch: one flat property list
    # could not say which action takes which field, so it said nothing and the
    # server decided after the fact. Stating it costs roughly 2.3 KB more.
    "jobs": 5030,
    # Twenty-odd recipe branches; the largest schema on the surface. The
    # description carries the recipe roster with plain synonyms, because a host
    # that matches a request against tool descriptions cannot otherwise route
    # "phase margin" or "distortion" to this tool at all. Each of the three
    # dormant branches spends about 75 characters more than the rest: its
    # pointer names the MCP reference lookup with the query that finds it, and
    # it carries the marker that keeps that one sentence on the compact
    # listing, where the branch has nothing else at all.
    "analyze_results": 18700,
    # Seven query kinds, each with its own argument shape — including the
    # reference lookup, which is what a session on the compact listing uses to
    # learn a branch's fields at all.
    "inspect": 7150,
    # The typed op union — eleven ops, each its own branch — plus the compare
    # object, in its one spelling. Rendering lives on verify_circuit, whose
    # policy is the more capable one, so no render argument is advertised here.
    "edit_schematic": 11400,
    # Checks, the render policy and the compare spec (each with the
    # verify-only fields on a subclass), each in one spelling. The checks are
    # named in the tool's own description because a caller cannot ask for what
    # the description does not say it looks at.
    "verify_circuit": 4850,
    # Job/case addressing, windowing, and delivery flags.
    "plot_waveform": 3200,
}

# Recipe branches no recorded workload has ever called (measured over 477
# campaign transcripts, MCP and the Python API — .claude/plans/u3_recipe_census.md). Their
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
        return _registered()["analyze_results"].input_schema["$defs"]

    @pytest.mark.parametrize("metric", DORMANT_WIRE_STUBS)
    def test_stubbed_branch_advertises_only_discriminant_and_pointer(self, metric: str):
        body = self._analyze_defs()[_recipe_def_name(metric)]
        assert set(body["properties"]) == {"metric"}
        assert body["properties"]["metric"]["const"] == metric
        # All three discovery channels are named, the MCP lookup first because
        # it is the one every client on this surface can call; and the stub
        # must stay permissive so a client pre-validating a full call against
        # the wire still sends it.
        assert f"inspect(kind='reference', query='{metric}')" in body["description"]
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
                "field": "frequency",
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


class TestAdvertisedProseIsTheSource:
    """A client is shown exactly the prose the source declares.

    Nothing filters descriptions between the registered definition and the
    advertised one, so a reader of the models knows what ships. The surface
    stays small because each description is written short — the size pins
    below are what holds that — not because some of them are hidden. A
    transform that started dropping or rewriting prose on the way out would
    fail here.
    """

    @staticmethod
    def _descriptions(node: Any, path: str = "") -> dict[str, str]:
        """Every description in a schema, keyed by where it sits."""
        found: dict[str, str] = {}
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    found[path or "<root>"] = value
                else:
                    found |= TestAdvertisedProseIsTheSource._descriptions(
                        value, f"{path}.{key}" if path else key
                    )
        elif isinstance(node, list):
            for index, item in enumerate(node):
                found |= TestAdvertisedProseIsTheSource._descriptions(item, f"{path}[{index}]")
        return found

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_advertised_descriptions_equal_the_source(self, name: str):
        source = _source_definitions()[name]
        advertised = _registered()[name]
        assert advertised.description == source.description
        source_fields = self._descriptions(source.input_schema)
        advertised_fields = self._descriptions(advertised.input_schema)
        assert set(advertised_fields) == set(source_fields), (
            f"{name}: the advertised schema documents different places than the "
            "source does — the two definitions must carry the same descriptions"
        )
        for where, text in source_fields.items():
            assert advertised_fields[where] == text, f"{name}: {where} differs from the source"


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
        assert result.is_error is True
        data = result.structured_content
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
        assert not result.is_error
        data = result.structured_content
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
        from ltspice_mcp.tools.jobs import _JobsActionError

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
            _validate_view_cursors(EditViewCursors(pin_legend="tampered"))


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


# Claude Code silently truncates both a tool description and the server
# instructions at this many characters. Truncation is invisible: the client
# shows nothing, the server logs nothing, and what is lost is the tail — which
# on this surface is where the caveats and the recovery routes are written.
CLIENT_TEXT_TRUNCATION_CHARS = 2048


class TestTextFitsTheClientTruncation:
    """No description or instruction block reaches the client half-read."""

    @pytest.mark.parametrize("name", REGISTERED_TOOLS)
    def test_tool_description_fits(self, name: str):
        description = _registered()[name].description or ""
        assert len(description) <= CLIENT_TEXT_TRUNCATION_CHARS, (
            f"{name}: description is {len(description)} chars; a client truncating "
            f"at {CLIENT_TEXT_TRUNCATION_CHARS} would silently drop the tail"
        )

    def test_the_instructions_constant_fits(self):
        """The runtime simulator prefix is measured separately, in
        tests/test_server.py, across every prefix shape. This pins the constant
        itself so a text edit is caught where the text lives."""
        from ltspice_mcp.server import CONSOLIDATED_INSTRUCTIONS

        assert len(CONSOLIDATED_INSTRUCTIONS) <= CLIENT_TEXT_TRUNCATION_CHARS


class TestAnalyzeDescriptionNamesEveryRecipe:
    """A host that routes on tool descriptions can only find a metric the
    description names.

    ``analyze_results`` answers twenty-one different questions behind one name,
    and the compact tool listing strips the per-branch schema prose, so this
    text is the only place the metric names appear. A recipe added to the union
    without joining the roster is a capability nothing can route to — which is
    why this is derived from the live discriminants rather than from a list.
    """

    def test_every_discriminant_appears_in_the_description(self):
        description = _registered()["analyze_results"].description or ""
        missing = [metric for metric in DISCRIMINANTS if metric not in description]
        assert not missing, (
            f"analyze_results' description does not name {', '.join(missing)}; add "
            "each with the plain words a caller would search for"
        )

    def test_the_plain_synonyms_that_route_to_this_tool_are_present(self):
        """A sample of the words a caller types instead of a discriminant. They
        are what makes description-matching reach the right tool at all."""
        description = (_registered()["analyze_results"].description or "").lower()
        for phrase in (
            "phase margin",
            "gain margin",
            "distortion",
            "bias point",
            "rise/fall time",
            "propagation delay",
            "overshoot",
            "duty cycle",
            "vswr",
            "bandwidth",
        ):
            assert phrase in description, f"the roster no longer names {phrase!r}"

    def test_it_points_at_the_reference_lookup(self):
        description = _registered()["analyze_results"].description or ""
        assert "inspect(kind='reference'" in description
