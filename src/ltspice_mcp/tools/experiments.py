"""Consolidated durable experiment submission tool."""

from __future__ import annotations

import asyncio
import contextlib
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from mcp import types
from pydantic import (
    BeforeValidator,
    Field,
    SkipValidation,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
)

from ltspice_mcp.errors import (
    PathSecurityError,
    ResultError,
    SimulationError,
    compact_validation_error,
    raise_site_code,
)
from ltspice_mcp.lib import experiment_store, response_budget
from ltspice_mcp.lib.deck_prep import resolve_runnable_netlist
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    stage_deck,
    verify_staged_manifest,
)
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    AnalysisCallback,
    ExperimentReceipt,
    ExperimentRunRequest,
    IdempotencyConflictError,
    canonical_fingerprint,
    verify_replay_sources,
)
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.lint_rules import RULES_BY_ID, lint_deck, linter_version
from ltspice_mcp.lib.recipes import DISCRIMINANTS, Recipe, validate_recipe
from ltspice_mcp.lib.simulator import simulator_dialect, simulator_library_roots
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    DeckFile,
    ExpandedCase,
    RandomVariation,
    Variation,
    VariationError,
    check_case_cap,
    expand_variations,
    format_case_id,
    materialize_variants,
    normalize_circuit_decks,
    projected_case_count,
    validate_variation_circuit_ids,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from ltspice_mcp.tools._base import (
    ResponseBudget,
    StrictModel,
    ToolInput,
    format_response,
    outcome_of,
    registry,
    resolve_response_budget,
    resolve_run_simulator,
    safe_path,
)
from ltspice_mcp.tools._schema import prune_unreferenced_defs
from ltspice_mcp.tools.analyze import (
    MAX_PAGE_SIZE,
    coerce_per_run_default,
    include_flag_coercer,
)
from ltspice_mcp.tools.receipts import (
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    TERMINAL_EXPERIMENT_STATUSES,
    ReceiptBuilt,
    finalize_receipt,
    render_receipt_snapshot,
    render_run_receipt,
    runs_page,
    snapshot_receipt,
)

SUBMISSION_DWELL_CAP_S = 120.0
# The jobs tool's own wait cap. It lives here rather than in tools/jobs so the
# run_experiments field description below can name it without importing that
# module — which would register the jobs tool ahead of this one and reorder the
# advertised tool list.
JOBS_WAIT_CAP_S = 300.0

# The source label an attached analysis analyzes its own experiment under.
_ATTACHED_ANALYSIS_LABEL = "experiment"


@dataclass
class _CircuitPreparation:
    circuit_id: str
    cases: list[ExperimentCase]
    source: SourceRecord | None
    lint_findings: list[dict[str, Any]]


class ExperimentCircuit(StrictModel):
    path: str = Field(
        description=(
            "Deck to run: .cir/.net/.sp (export an .asc through LTspice first). It "
            "is staged content-addressed at submission, so later edits to the file "
            "cannot change what this job ran."
        ),
    )
    id: str | None = Field(
        default=None,
        description=(
            "Short name for this circuit, used to scope a variation's 'applies_to' "
            "and to label its rows. Defaults to the file stem."
        ),
    )


class ExperimentExecution(StrictModel):
    """How long this call waits, how hard the job runs, and on which simulator."""

    wait_s: float = Field(
        default=60.0,
        ge=0.0,
        le=SUBMISSION_DWELL_CAP_S,
        description=(
            "Dwell 0-120s before returning; the durable job keeps running. "
            "Continue with jobs(action='wait'); 0 returns immediately."
        ),
    )

    @field_validator("wait_s", mode="wrap")
    @classmethod
    def _wait_s_names_the_continuation_route(
        cls,
        value: Any,
        handler: ValidatorFunctionWrapHandler,
    ) -> float:
        try:
            return handler(value)
        except ValidationError as exc:
            if any(error["type"] == "less_than_equal" for error in exc.errors()):
                raise ValueError(
                    f"execution.wait_s cannot exceed {SUBMISSION_DWELL_CAP_S:g}s; "
                    "submit within that dwell, then continue with "
                    f'jobs(action="wait", ..., timeout_s<={JOBS_WAIT_CAP_S:g})'
                ) from exc
            raise

    run_timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Kill any single case whose simulator exceeds this and mark it failed; "
            "the other cases continue."
        ),
    )
    job_deadline_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Wall-clock budget for the whole job, measured from submission. On "
            "expiry unfinished cases are cancelled and already-produced results are "
            "kept."
        ),
    )
    max_parallel: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Cases in flight at once. Defaults to the server's concurrency cap; "
            "lower it only to leave room for other work."
        ),
    )
    simulator: Literal["ltspice", "ngspice"] | None = Field(
        default=None,
        description=(
            "Engine for every case; it also decides the dialect the results are "
            "parsed with. Defaults to the server's default simulator."
        ),
    )


class AnalysisPerRun(StrictModel):
    # Bound taken from analyze_results itself, never a copy of its number: the
    # attached block is handed straight to that engine, so a limit this schema
    # advertised but the engine rejected would be a lever that cannot work.
    limit: int = Field(default=50, ge=1, le=MAX_PAGE_SIZE)
    cursor: str | None = Field(
        default=None,
        description=(
            "Nothing has been paged at submission time; leave it unset and page the "
            "finished analysis with analyze_results."
        ),
    )


class AnalysisInclude(StrictModel):
    per_run: Annotated[
        AnalysisPerRun | None,
        BeforeValidator(
            coerce_per_run_default,
            json_schema_input_type=AnalysisPerRun | bool | None,
        ),
    ] = Field(
        default=None,
        description=(
            "Return the individual attributed rows, paginated; true takes the "
            "default page. Omitted, a recipe with 'reduce' returns only its "
            "reductions."
        ),
    )
    outliers: bool = Field(
        default=False,
        description="Add the spec-failing records to each recipe's spec block.",
    )
    signals_available: bool = Field(
        default=False,
        description=(
            "List the trace names each run's .raw carries. Costs one raw load per "
            "run; use it to find signal names, then turn it off."
        ),
    )
    fields: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description=(
            "Dotted row paths to keep on per_run/values rows, using "
            "analyze_results include.fields grammar. Presentation only."
        ),
    )


coerce_attached_include_flags = include_flag_coercer(AnalysisInclude)


class AttachedAnalysis(StrictModel):
    # The same typed union analyze_results advertises, not a free-form object:
    # this block IS an analyze_results request, and a schema that said
    # "any object" left a caller to discover the recipe grammar by having a
    # whole simulation run and then fail at the analysis stage. SkipValidation
    # keeps the strict union in the published schema while leaving the items as
    # the caller sent them, which is what _validate_attached_analysis then
    # checks recipe by recipe, before anything is staged.
    recipes: list[SkipValidation[Recipe]] = Field(
        description=(
            "analyze_results recipes, in that tool's exact shape, run over this "
            "job's own runs once they finish."
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description=(
            "Split reductions along these dimensions: a variation assignment "
            "parameter name, 'circuit', or a .step axis name."
        ),
    )
    include: Annotated[
        AnalysisInclude | None,
        BeforeValidator(
            coerce_attached_include_flags,
            json_schema_input_type=AnalysisInclude | list[str] | None,
        ),
    ] = Field(
        default=None,
        description=(
            "Optional per-run, outlier, signal-listing, and row-view controls; "
            "a bare list of flag names switches them on."
        ),
    )

    @field_serializer("recipes")
    def _serialize_recipes(self, recipes: list[Any]) -> list[Any]:
        """Serialize each recipe exactly as it arrived.

        Skipped validation leaves a wire recipe as the dict the caller sent,
        while the annotation promises a model. Without this the default
        serializer warns on every dump — including the one that computes the
        durable fingerprint, which must keep hashing the caller's own bytes.
        """
        return [
            recipe if isinstance(recipe, dict) else recipe.model_dump(mode="json")
            for recipe in recipes
        ]


#: What the wire says one attached recipe is: the metric names, the key every
#: recipe carries, and where the per-metric field trees live.
#:
#: The grammar is ``analyze_results``', and that tool publishes all twenty-odd
#: branches in full on the same wire. A second copy here measured 13 KB, paid
#: by every client in every session whether or not it ever attaches an
#: analysis, to say something already said one tool away. What a caller cannot
#: derive is which metrics exist, so that is what the stub keeps — the same
#: bargain the dormant recipe branches strike, including their two-channel
#: pointer and their deliberate permissiveness: no ``additionalProperties``,
#: so a client pre-validating a full recipe against this shape still sends it.
#: The model behind it is the real union, and every attached recipe is
#: validated at submission, before a deck is staged.
_ATTACHED_RECIPE_WIRE_STUB: dict[str, Any] = {
    "type": "object",
    "description": (
        "One entry of analyze_results.recipes: same grammar, same metrics, "
        "validated at submission. Fields per metric: "
        "api.reference('analyze_results') or spice://guide."
    ),
    "properties": {
        "key": {"type": "string", "minLength": 1},
        "metric": {"type": "string", "enum": list(DISCRIMINANTS)},
    },
    "required": ["key", "metric"],
}


class RunExperimentsInput(ToolInput):
    # Fields that choose how the receipt is rendered rather than what runs.
    # canonical_fingerprint excludes them, so re-asking for the same experiment
    # at a different verbosity replays instead of conflicting. execution.wait_s
    # is excluded the same way: it bounds only this response's dwell (the job
    # is durable either way), so a different dwell is the same experiment and
    # a wait_s=0 submission hands back a receipt a later call can replay. A version
    # bump is required only when a previously valid request's canonical bytes
    # change; a presentation field excluded from its first valid day changes no
    # old bytes and does not bump the canonicalizer.
    PRESENTATION_FIELDS: ClassVar[dict[str, Any]] = {
        "provenance": True,
        "run_fields": True,
        "budget": True,
        "execution": {"wait_s"},
        "analyze": {"include": {"fields"}},
    }

    def strip_presentation(self) -> dict[str, Any]:
        """Return the execution-shaped request with render controls removed."""
        payload = self.model_dump(
            mode="json",
            exclude_unset=False,
            exclude=self.PRESENTATION_FIELDS,
        )

        def collapse_optional_models(
            model: StrictModel,
            rendered: dict[str, Any],
            exclusions: dict[str, Any],
        ) -> None:
            for name, nested in exclusions.items():
                child = getattr(model, name, None)
                if not isinstance(child, StrictModel) or name not in rendered:
                    continue
                excluded_names = set(nested) if isinstance(nested, (set, dict)) else set()
                field = type(model).model_fields.get(name)
                if (
                    field is not None
                    and field.default is None
                    and child.model_fields_set
                    and child.model_fields_set <= excluded_names
                ):
                    rendered[name] = None
                elif isinstance(nested, dict) and isinstance(rendered[name], dict):
                    collapse_optional_models(child, rendered[name], nested)

        collapse_optional_models(self, payload, self.PRESENTATION_FIELDS)
        return payload

    def canonical_fingerprint_payload(self) -> dict[str, Any]:
        """Exclude receipt fields without changing old include=None bytes."""
        return self.strip_presentation()

    @classmethod
    def wire_input_schema(cls) -> dict[str, Any]:
        """Advertise an attached recipe as its stub, not a second recipe union.

        The stub replaces the union only in what is published; the model still
        validates against the union, so this changes nothing a call is allowed
        to send. Definitions the union kept alive are then unreachable, and an
        unreferenced definition is weight no client can use.
        """
        schema = copy.deepcopy(super().wire_input_schema())
        recipes = schema["$defs"][AttachedAnalysis.__name__]["properties"]["recipes"]
        recipes["items"] = copy.deepcopy(_ATTACHED_RECIPE_WIRE_STUB)
        return prune_unreferenced_defs(schema)

    request_id: str = Field(
        default_factory=lambda: generate_id("req"),
        min_length=1,
        description=(
            "Idempotency key, optional. Reusing one replays that receipt when the "
            "arguments and decks are unchanged, and is a conflict when either "
            "changed. Omitted, a fresh id is generated and echoed back."
        ),
    )
    circuits: list[ExperimentCircuit] = Field(
        min_length=1,
        description=(
            "Decks to run. Several in one call share one variation grid and one "
            "job, which compares designs under identical conditions; scope a "
            "variation to one of them with 'applies_to'."
        ),
    )
    variations: list[Variation] = Field(
        default_factory=list,
        description=(
            "The sweep. 'assign' entries build the case grid (cartesian across "
            "entries, lock-step within one via combine:'zip'); at most one "
            "'random' entry adds Monte Carlo runs. Cases run in parallel — ask "
            "for the whole grid in one call. Empty runs each circuit as "
            "authored."
        ),
    )
    execution: ExperimentExecution = Field(
        default_factory=ExperimentExecution,
        description=(
            "How the job runs and how long this call dwells. wait_s bounds only "
            "this response — the job is durable either way. Defaults suit a quick "
            "check."
        ),
    )
    analyze: AttachedAnalysis | None = Field(
        default=None,
        description=(
            "Measure the runs as a stage of this job, so the terminal response "
            "carries the numbers. A failed analysis does not fail the runs, which "
            "stay analyzable with analyze_results."
        ),
    )
    lint: Literal["block", "warn", "off"] = Field(
        default="block",
        description=(
            "What to do with lint findings on the staged deck: 'block' refuses to "
            "submit a circuit with a blocking finding (its cases report "
            "'skipped'), 'warn' runs anyway, 'off' skips linting. Prefer "
            "'suppress' over lowering this."
        ),
    )
    suppress: list[str] = Field(
        default_factory=list,
        description=(
            "Lint rule_ids to drop, taken from the rule_id of a finding already "
            "reported. Narrower than lint:'warn': everything else still blocks."
        ),
    )
    allow_live_includes: bool = Field(
        default=False,
        description=(
            "Let a .include/.lib that cannot be staged (outside the allowed "
            "roots, or past the recursion depth) be read live at run time "
            "instead of failing that circuit. The job cannot then prove what "
            "those files held, so its request_id re-runs rather than replaying."
        ),
    )
    provenance: bool = Field(
        default=False,
        description=(
            "Emit the full audit trail on each source: content digests, the "
            "staged deck path, every staged file, and the linter version. "
            "Actionable entries are reported either way."
        ),
    )
    run_fields: list[str] | None = Field(
        default=None,
        description=(
            "Keep only these keys on each row of 'runs.items', dotted for "
            "nesting (e.g. 'assignments.RDEG'); escape a dot inside a key name "
            "as '\\.'."
        ),
    )
    budget: int | None = Field(
        default=None,
        ge=response_budget.BUDGET_MIN_TOKENS,
        description=response_budget.BUDGET_DESCRIPTION,
    )


@registry.tool(
    name="run_experiments",
    description=(
        "Run SPICE and get the numbers back in one call: point it at your deck(s), "
        "attach an 'analyze' block, and the measured values return with the "
        "results — from a one-off spot check to a full sweep or Monte Carlo grid. "
        "Express the whole sweep as one 'variations' grid rather than a call per "
        "point: cases run in parallel up to the server's cap, so a grid costs one "
        "call and one receipt, not one of each per case. When unsure about a "
        "behavior, assumption, or sizing, run a small experiment and read the "
        "numbers rather than reasoning it out. Quick runs return results inline; "
        "longer ones return a receipt to follow with 'jobs'. Every job is recorded "
        "on disk; pass your own request_id to make a retry replay the same job "
        "instead of running a new one."
    ),
    input_model=RunExperimentsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    output_schema=RUN_EXPERIMENTS_OUTPUT_SCHEMA,
)
async def handle_run_experiments(
    args: RunExperimentsInput,
    state: SessionState,
) -> types.CallToolResult:
    """Stage, lint, expand, durably submit, and dwell on an experiment."""
    analysis_fields = (
        args.analyze.include.fields
        if args.analyze is not None and args.analyze.include is not None
        else None
    )
    fingerprint = canonical_fingerprint(args)
    budget = resolve_response_budget(args.budget, state)
    try:
        replay = await _load_matching_replay(args, state, fingerprint)
        if replay is not None:
            return await _dwell_and_respond(
                replay,
                args.execution.wait_s,
                state,
                provenance=args.provenance,
                run_fields=args.run_fields,
                analysis_fields=analysis_fields,
                budget=budget,
            )

        simulator = resolve_run_simulator(args.execution.simulator, state)
        circuit_inputs = _circuit_decks_for_validation(args.circuits)
        normalize_circuit_decks(circuit_inputs)
        validate_variation_circuit_ids(circuit_inputs, args.variations)
        if sum(isinstance(item, RandomVariation) for item in args.variations) > 1:
            raise VariationError(
                "multiple_random_variations",
                "At most one random variation entry is allowed per run_experiments call",
            )
        projected = sum(
            projected_case_count(circuit.circuit_id, args.variations) for circuit in circuit_inputs
        )
        check_case_cap(projected, state.config.max_experiment_cases)
        if args.analyze is not None:
            _validate_attached_analysis(args.analyze)

        # The first circuit's id (the deck's file stem unless the caller named
        # it) rides in the job id so the handle says what it ran.
        job_id = generate_id("exp", circuit_inputs[0].circuit_id)
        try:
            route = await asyncio.to_thread(
                resolve_experiment_paths,
                state.working_dir,
                job_id,
                circuit_inputs[0].circuit_id,
                simulator,
            )
        except DeckStagingError as exc:
            return await _routing_failure_response(
                args, circuit_inputs, exc, projected, budget=budget
            )
        await asyncio.to_thread(route.output_folder.mkdir, parents=True, exist_ok=True)

        cases: list[ExperimentCase] = []
        sources: list[SourceRecord] = []
        lint_by_circuit: dict[str, list[dict[str, Any]]] = {
            circuit.circuit_id: [] for circuit in circuit_inputs
        }
        preparations = await asyncio.gather(
            *(
                _prepare_circuit(
                    circuit_input,
                    circuit_arg,
                    args,
                    state,
                    simulator,
                    job_id,
                )
                for circuit_input, circuit_arg in zip(
                    circuit_inputs,
                    args.circuits,
                    strict=True,
                )
            )
        )
        for preparation in preparations:
            lint_by_circuit[preparation.circuit_id] = preparation.lint_findings
            if preparation.source is not None:
                sources.append(preparation.source)
            for case in preparation.cases:
                case.run_index = len(cases)
                cases.append(case)

        if len(cases) != projected:
            raise VariationError(
                "completeness_mismatch",
                f"Prepared {len(cases)} cases but variation expansion declared {projected}",
            )

        analysis_request = args.strip_presentation()["analyze"]
        runner = state.runners.get_experiment_runner(
            loop=asyncio.get_running_loop(),
            simulator_class=simulator,
            output_folder=route.output_folder,
            # The runner's cap is the SERVER's, shared by every experiment it
            # runs; a request's own max_parallel divides that share below.
            max_parallel=state.config.max_parallel_sims,
        )
        request = ExperimentRunRequest(
            state=state,
            request_id=args.request_id,
            fingerprint=fingerprint,
            cases=cases,
            sources=sources,
            simulator=simulator.__name__,
            job_id=job_id,
            declared=len(args.circuits),
            max_parallel=args.execution.max_parallel,
            run_timeout_s=args.execution.run_timeout_s,
            job_deadline_s=args.execution.job_deadline_s,
            analysis_request=analysis_request,
            analysis_callback=(
                _attached_analysis_callback(state) if analysis_request is not None else None
            ),
        )
        receipt = await asyncio.shield(runner.submit(request))
        # Holding the receipt means the cases are live and durable: submission
        # is irreversible from here, so no escape below may reach the
        # not_started handlers underneath. The nested guard is what makes that
        # structural rather than a rule about which exception types to list.
        try:
            return await _dwell_and_respond(
                receipt,
                args.execution.wait_s,
                state,
                lint_by_circuit=lint_by_circuit,
                provenance=args.provenance,
                run_fields=args.run_fields,
                analysis_fields=analysis_fields,
                budget=budget,
            )
        except Exception as exc:
            return await _post_submit_error_response(
                receipt,
                exc,
                lint_by_circuit,
                budget=budget,
            )
    except IdempotencyConflictError as exc:
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="submission",
            retryable=False,
            commit_state="not_started",
            budget=budget,
        )
    except VariationError as exc:
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="variation",
            retryable=False,
            commit_state="not_started",
            budget=budget,
        )
    except PathSecurityError as exc:
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="resolution",
            retryable=False,
            commit_state="not_started",
            budget=budget,
        )
    except (SimulationError, ResultError, DeckStagingError, OSError, ValueError) as exc:
        return await _error_response(
            args.request_id,
            code=raise_site_code(exc) or "submission_failed",
            message=str(exc),
            stage="submission",
            retryable=True,
            commit_state="not_started",
            budget=budget,
        )


async def _render_static_run_receipt(
    data: dict[str, Any],
    text: str,
    budget: ResponseBudget,
    *,
    is_error: bool = False,
) -> types.CallToolResult:
    return await render_run_receipt(
        budget,
        lambda _limit, _rung: (copy.deepcopy(data), text),
        is_error=is_error,
    )


async def _prepare_circuit(
    circuit_input: CircuitDeck,
    circuit_arg: ExperimentCircuit,
    args: RunExperimentsInput,
    state: SessionState,
    simulator: type,
    job_id: str,
) -> _CircuitPreparation:
    """Stage, lint, and materialize one circuit with isolated failures."""
    circuit_id = circuit_input.circuit_id
    expected_count = projected_case_count(circuit_id, args.variations)
    source_path: Path | None = None
    runnable: Path | None = None
    expanded: list[ExpandedCase] | None = None
    source: SourceRecord | None = None
    findings: list[dict[str, Any]] = []
    cases: list[ExperimentCase] = []
    try:
        source_path = safe_path(circuit_arg.path, state)
        if source_path.suffix.casefold() not in {".cir", ".net", ".sp", ".asc"}:
            raise VariationError(
                "unsupported_variant",
                f"Circuit {circuit_id!r} uses unsupported extension "
                f"{source_path.suffix or '<none>'!r}; supported extensions are "
                ".cir, .net, .sp, and .asc",
            )
        if not await asyncio.to_thread(source_path.is_file):
            raise FileNotFoundError(f"Circuit file not found: {source_path}")
        await state.note_recent_circuit(source_path)
        runnable = await resolve_runnable_netlist(
            circuit_arg.path,
            state,
            simulator=simulator,
        )
        paths = await asyncio.to_thread(
            resolve_experiment_paths,
            state.working_dir,
            job_id,
            circuit_id,
            simulator,
        )
        staged = await asyncio.to_thread(
            stage_deck,
            runnable,
            paths.staging_root,
            state.config.allowed_paths,
            # A schematic is simulated through an exported netlist, so without
            # this the manifest describes only that export — and re-exporting is
            # exactly what a replay skips, leaving an edited .asc invisible to
            # every check made over this record.
            origin=source_path,
            allow_live_includes=args.allow_live_includes,
            windows_paths=paths.windows_native,
            # LTspice's own .asc netlister appends a .lib pointing into the
            # install's model library on every schematic with a MOSFET on it,
            # so without this no transistor sheet stages under a default
            # sandbox. Resolved per run from the simulator this job uses.
            simulator_roots=await asyncio.to_thread(simulator_library_roots, simulator),
        )
        dialect = simulator_dialect(simulator)
        findings = (
            []
            if args.lint == "off"
            else await asyncio.to_thread(
                lint_deck,
                staged.text,
                staged.staged_deck,
                dialect,
                simulator,
                suppress=args.suppress,
                # The staged closure's own snapshots: on a Windows-native
                # staging route the deck's rewritten references cannot be
                # re-read from the Linux side, and a model defined in an
                # include must not lint as missing.
                includes=[(included.staged_path, included.text) for included in staged.includes],
            )
        )
        source = SourceRecord(
            circuit=circuit_id,
            path=source_path,
            # The digest of the path this record names. For a schematic that is
            # the .asc, not the netlist exported from it — the staged export's
            # own digest stays reachable through its manifest entry.
            sha256=staged.origin_sha256,
            staged_deck=staged.staged_deck,
            manifest=staged.manifest,
            linter_version=linter_version,
            simulator=simulator.__name__,
            dialect=dialect,
            lint_findings=findings,
        )
        observations = [
            *staged.observations,
            *await asyncio.to_thread(verify_staged_manifest, staged.manifest),
        ]
        deck = CircuitDeck(
            circuit_id=circuit_id,
            path=staged.staged_deck,
            text=staged.text,
            includes=tuple(
                DeckFile(path=included.staged_path, text=included.text)
                for included in staged.includes
            ),
        )
        expanded = await asyncio.to_thread(
            expand_variations,
            [deck],
            args.variations,
            max_cases=state.config.max_experiment_cases,
            validate_applies_to=False,
        )
        blocking = [
            finding
            for finding in findings
            if RULES_BY_ID[finding["rule_id"]].disposition == "blocking"
        ]
        if args.lint == "block" and blocking:
            _append_terminal_cases(
                cases,
                circuit_id=circuit_id,
                circuit_path=source_path,
                staged_deck=staged.staged_deck,
                count=len(expanded),
                status="skipped",
                code="lint_blocked",
                message=(
                    f"{len(blocking)} blocking lint finding(s) prevented simulator submission"
                ),
                observations=observations,
                expanded_cases=expanded,
            )
        else:
            materialized = await asyncio.to_thread(
                materialize_variants,
                deck,
                expanded,
                staged.staged_deck.parent,
            )
            cases.extend(
                ExperimentCase(
                    case_id=variant.case_id,
                    run_index=offset,
                    circuit=circuit_id,
                    circuit_path=source_path,
                    staged_deck=variant.path,
                    deck_sha256=variant.sha256,
                    assignments=variant.assignments,
                    observations=list(observations),
                )
                for offset, variant in enumerate(materialized)
            )
    except (
        PathSecurityError,
        VariationError,
        DeckStagingError,
        SimulationError,
        OSError,
        ValueError,
    ) as exc:
        code, message = _circuit_error(exc, source_path)
        _append_terminal_cases(
            cases,
            circuit_id=circuit_id,
            circuit_path=source_path or Path(circuit_arg.path),
            staged_deck=runnable or Path(circuit_arg.path),
            count=expected_count,
            status="failed",
            code=code,
            message=message,
            expanded_cases=expanded,
        )
    return _CircuitPreparation(
        circuit_id=circuit_id,
        cases=cases,
        source=source,
        lint_findings=findings,
    )


def _attached_analysis_payload(job_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """The analyze_results request an attached block expands to.

    One builder serves submission pre-flight and post-run execution. Pre-flight
    also validates presentation-only fields; the persisted execution request
    has already removed them before the post-run call reaches this builder.
    """
    payload: dict[str, Any] = {
        "sources": [
            {
                "job_id": job_id,
                "runs": "all",
                "label": _ATTACHED_ANALYSIS_LABEL,
            }
        ],
        "recipes": request.get("recipes") or [],
        "group_by": request.get("group_by") or [],
    }
    include = request.get("include")
    if include is not None:
        payload["include"] = include
    return payload


def _validate_attached_analysis(analyze_block: AttachedAnalysis) -> None:
    """Refuse a malformed attached analyze block BEFORE anything is staged.

    Validated only at the analysis stage, a typo'd recipe burns the whole
    simulation cycle — and the corrected block then changes the canonical
    fingerprint, so the retry re-runs every case. The placeholder job id used
    for this shape check never resolves, because validation does not touch the
    registry.
    """
    request = analyze_block.model_dump(mode="json", exclude_unset=False)
    try:
        analyze.AnalyzeResultsInput.model_validate(
            _attached_analysis_payload("preflight", request)
        )
        # The input model deliberately skips per-recipe validation
        # (SkipValidation[Recipe] — items are validated lazily at execution so
        # one bad recipe fails one item). At SUBMISSION that laziness is the
        # bug: every recipe must parse before a simulator is asked to run.
        for raw_recipe in request.get("recipes") or []:
            validate_recipe(raw_recipe)
    except (ValidationError, ValueError) as exc:
        raise SimulationError(
            "The attached analyze block is not a valid analyze_results request: "
            f"{compact_validation_error(exc)}"
        ) from exc


def _attached_analysis_callback(state: SessionState) -> AnalysisCallback:
    """Bind the session onto the coordinator's job-only analysis hook.

    The coordinator hands the callback nothing but the job, so the session it
    must analyze against is closed over here.
    """

    async def run_attached_analysis(job: ExperimentJob) -> dict[str, Any]:
        payload = _attached_analysis_payload(job.job_id, job.analysis.request or {})
        try:
            args = analyze.AnalyzeResultsInput.model_validate(payload)
        except ValidationError as exc:
            raise SimulationError(
                "The attached analyze block is not a valid analyze_results request: "
                f"{compact_validation_error(exc)}"
            ) from exc
        # Resolved on the module, not bound at import: the analysis stage is
        # patched through ``tools.analyze`` in tests, and the attribute lookup
        # is what keeps that seam where the engine actually lives.
        return await analyze.capture_attached_analysis(args, state)

    return run_attached_analysis


def _circuit_decks_for_validation(circuits: list[ExperimentCircuit]) -> list[CircuitDeck]:
    return [
        CircuitDeck(
            circuit_id=circuit.id or Path(circuit.path).stem,
            path=Path(circuit.path),
            text="",
            id_from_file_stem=not circuit.id,
        )
        for circuit in circuits
    ]


async def _load_matching_replay(
    args: RunExperimentsInput,
    state: SessionState,
    fingerprint: str,
) -> ExperimentReceipt | None:
    index = await asyncio.to_thread(
        experiment_store.load_request_index,
        args.request_id,
        state.working_dir,
    )
    if index is None:
        return None
    if (
        index.get("canonicalizer_version") != CANONICALIZER_VERSION
        or index.get("fingerprint") != fingerprint
    ):
        raise IdempotencyConflictError(
            f"request_id {args.request_id!r} was already used for a different "
            "request payload or canonicalizer version"
        )
    job_id = str(index.get("job_id", ""))
    job = state.all_jobs.get(job_id)
    if not isinstance(job, ExperimentJob):
        job = await asyncio.to_thread(
            experiment_store.load_job,
            job_id,
            state.working_dir,
            own_is_alive=True,
        )
        if job is None:
            return None
        state.add_experiment_job(job, already_persisted=True)
    if (
        job.request_id != args.request_id
        or job.fingerprint != fingerprint
        or job.canonicalizer_version != CANONICALIZER_VERSION
    ):
        raise IdempotencyConflictError(
            f"request_id {args.request_id!r} points to an inconsistent coordinator record"
        )
    await asyncio.to_thread(verify_replay_sources, job, args.request_id)
    if not any(item.get("code") == "idempotent_replay" for item in job.observations):
        job.observations.append(
            {
                "code": "idempotent_replay",
                "kind": "submission",
                "detail": (
                    "The request_id and canonical payload matched an existing "
                    "durable experiment; its receipt was returned without "
                    "resubmitting cases."
                ),
            }
        )
        state.persist_job(job)
    return ExperimentReceipt(job=job, replayed=True, control_token=job.control_token)


async def _dwell_and_respond(
    receipt: ExperimentReceipt,
    wait_s: float,
    state: SessionState,
    *,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
    provenance: bool = False,
    run_fields: list[str] | None = None,
    analysis_fields: list[str] | None = None,
    budget: ResponseBudget,
) -> types.CallToolResult:
    job = receipt.job
    if job.status not in TERMINAL_EXPERIMENT_STATUSES and wait_s > 0:
        runner = state.runners.get_experiment_runner_for(job)
        if runner is not None:
            await runner.wait(job, wait_s, wait_for="all")
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done_event.wait(), wait_s)
    snapshot = snapshot_receipt(
        job,
        None,
        control_token=receipt.control_token,
        lint_by_circuit=lint_by_circuit,
    )
    text = (
        f"Experiment {snapshot.job_id}: {snapshot.status} "
        f"({snapshot.completeness.terminal}/{snapshot.completeness.expanded} terminal cases)"
    )

    def build(limit: int, rung: response_budget.Rung | None) -> ReceiptBuilt:
        data = render_receipt_snapshot(
            snapshot,
            control_token=receipt.control_token,
            provenance=provenance,
            run_fields=run_fields,
            runs_cap=limit,
            analysis_fields=analysis_fields,
            analysis_answer_channel=rung is not None and rung.answer_channel,
            analysis_rows_cap=limit if rung is not None and rung.shrink else None,
        )
        return finalize_receipt(data), text

    return await render_run_receipt(budget, build)


def _append_terminal_cases(
    cases: list[ExperimentCase],
    *,
    circuit_id: str,
    circuit_path: Path | None,
    staged_deck: Path | None,
    count: int,
    status: Literal["failed", "skipped"],
    code: str,
    message: str,
    observations: list[dict[str, Any]] | None = None,
    expanded_cases: list[ExpandedCase] | None = None,
) -> None:
    descriptors: list[ExpandedCase | None] = (
        list(expanded_cases) if expanded_cases is not None else [None] * count
    )
    if len(descriptors) != count:
        raise VariationError(
            "completeness_mismatch",
            f"Prepared {len(descriptors)} terminal case descriptors, expected {count}",
        )
    for local_index, expanded in enumerate(descriptors):
        assignments = dict(expanded.assignments) if expanded is not None else {}
        if expanded is not None and expanded.random_index is not None:
            assignments["_random_run"] = expanded.random_index
            if expanded.random is not None and expanded.random.id is not None:
                assignments["_random_id"] = expanded.random.id
        cases.append(
            ExperimentCase(
                case_id=(
                    expanded.case_id
                    if expanded is not None
                    else format_case_id(circuit_id, local_index)
                ),
                run_index=len(cases),
                circuit=circuit_id,
                circuit_path=circuit_path or Path(circuit_id),
                staged_deck=staged_deck or Path(circuit_id),
                deck_sha256="",
                assignments=assignments,
                status=status,
                error=message,
                failure_code=code,
                observations=list(observations or []),
            )
        )


def _circuit_error(exc: Exception, source_path: Path | None) -> tuple[str, str]:
    if isinstance(exc, PathSecurityError):
        return exc.code, str(exc)
    if isinstance(exc, VariationError):
        return exc.code, str(exc)
    if isinstance(exc, DeckStagingError):
        return exc.code, str(exc)
    if (
        source_path is not None
        and source_path.suffix.casefold() == ".asc"
        and isinstance(exc, SimulationError)
    ):
        return "asc_export_unavailable", str(exc)
    if isinstance(exc, FileNotFoundError):
        return "source_not_found", str(exc)
    return "submission_failed", str(exc)


async def _routing_failure_response(
    args: RunExperimentsInput,
    circuits: list[CircuitDeck],
    exc: DeckStagingError,
    projected: int,
    *,
    budget: ResponseBudget,
) -> types.CallToolResult:
    cases: list[ExperimentCase] = []
    for circuit in circuits:
        count = projected_case_count(circuit.circuit_id, args.variations)
        _append_terminal_cases(
            cases,
            circuit_id=circuit.circuit_id,
            circuit_path=circuit.path,
            staged_deck=circuit.path,
            count=count,
            status="failed",
            code=exc.code,
            message=str(exc),
        )
    failures = [
        {
            "case_id": case.case_id,
            "code": exc.code,
            "message": str(exc),
        }
        for case in cases
    ]
    completeness = Completeness(
        declared=len(circuits),
        expanded=projected,
        failed=projected,
    )
    data = _empty_payload(args.request_id)
    data.update(
        {
            "status": "failed",
            # Every declared case failed to stage, but the lint pass and the
            # completeness reconciliation did report.
            "outcome": outcome_of(failures),
            "completeness": completeness,
            "lint": [{"circuit": circuit.circuit_id, "findings": []} for circuit in circuits],
            "failures": failures,
            "hint": (
                "Configure an available Windows-native temp directory before "
                "retrying WSL LTspice experiments."
            ),
        }
    )
    finalize_receipt(data)

    def build(limit: int, _rung: response_budget.Rung | None) -> ReceiptBuilt:
        rendered = copy.deepcopy(data)
        rendered["runs"] = runs_page(cases, args.run_fields, cap=limit)
        return rendered, str(exc)

    return await render_run_receipt(budget, build)


async def _error_response(
    request_id: str,
    *,
    code: str,
    message: str,
    stage: str,
    retryable: bool,
    commit_state: Literal["not_started", "committed", "unknown"],
    budget: ResponseBudget,
) -> types.CallToolResult:
    data = _empty_payload(request_id)
    data.update(
        {
            "hint": message,
            "error": {
                "code": code,
                "message": message,
                "stage": stage,
                "retryable": retryable,
                "commit_state": commit_state,
            },
        }
    )
    finalize_receipt(data)
    return await _render_static_run_receipt(
        data,
        message,
        budget,
        is_error=True,
    )


async def _post_submit_error_response(
    receipt: ExperimentReceipt,
    exc: Exception,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None,
    *,
    budget: ResponseBudget,
) -> types.CallToolResult:
    """Envelope for a failure that escaped AFTER the cases were submitted.

    Submission is the irreversible step: once the receipt exists the simulator
    runs are under way, and the job_id plus its control_token are the only
    handles that reach them. Reporting ``not_started`` here — or letting the
    exception out, which returns no structuredContent at all — strands running
    cases with no way to poll or cancel them. Those orphaned runs are precisely
    what commit_state exists to prevent, so a post-submit escape is always
    reported as committed.
    """
    job = receipt.job

    def error_message(*errors: Exception | None) -> str:
        messages: list[str] = []
        for error in errors:
            if error is not None and str(error) not in messages:
                messages.append(str(error))
        return "; ".join(messages)

    build_error: Exception | None = None
    try:
        snapshot = snapshot_receipt(
            job,
            None,
            control_token=receipt.control_token,
            lint_by_circuit=lint_by_circuit,
        )
        handles = {
            "job_id": snapshot.job_id,
            "status": snapshot.status,
            "outcome": outcome_of([], in_progress=True),
            "control_token": receipt.control_token,
            "completeness": copy.deepcopy(snapshot.completeness),
        }
        data = render_receipt_snapshot(snapshot, control_token=receipt.control_token)
    except Exception as payload_exc:
        build_error = payload_exc
        # The snapshot itself can be what failed. These minimum handles are the
        # last-resort recovery surface and avoid copying any other mutable
        # receipt field independently.
        handles = {
            "job_id": job.job_id,
            "status": job.status,
            "outcome": outcome_of([], in_progress=True),
            "control_token": receipt.control_token,
            "completeness": copy.deepcopy(job.completeness),
        }
        # Even the receipt builder failed. Fall back to the minimum that keeps
        # the running job reachable rather than losing the handles with it.
        data = _empty_payload(job.request_id)
        data.update(handles)
    route = (
        f"The experiment was submitted and is running. Use jobs(status) with job_id "
        f"{job.job_id} to follow it, or jobs(cancel) with that job_id and its "
        f"control_token to stop it."
    )
    data["hint"] = route
    data["error"] = {
        "code": raise_site_code(exc) or "receipt_failed",
        "message": error_message(exc, build_error),
        "stage": "receipt",
        "retryable": True,
        "commit_state": "committed",
    }
    finalize_receipt(data)
    text = (
        f"Experiment {job.job_id} was submitted, but building its receipt failed: {exc}. "
        f"The cases ARE running."
    )
    try:
        return await _render_static_run_receipt(
            data,
            text,
            budget,
            is_error=True,
        )
    except Exception as render_exc:
        # Non-recursive rescue boundary: no renderer or full payload builder is
        # called again after it fails. The minimum durable handles survive.
        minimal = _empty_payload(job.request_id)
        minimal.update(
            {
                **handles,
                "hint": route,
                "error": {
                    **data["error"],
                    "message": error_message(exc, build_error, render_exc),
                },
            }
        )
        finalize_receipt(minimal)
        result = format_response(text, minimal)
        result.isError = True
        return result


def _empty_payload(request_id: str) -> dict[str, Any]:
    """The skeleton for a call that produced no job.

    Its outcome says so — something went wrong and nothing came back with it.
    A caller that goes on to fill in a job overwrites it.
    """
    return {
        "job_id": None,
        "request_id": request_id,
        "status": "failed",
        "outcome": outcome_of(True, delivered=False),
        "source": [],
        "completeness": Completeness(),
        "lint": [],
        "runs": runs_page([]),
        "failures": [],
        "observations": [],
        "warnings": [],
        "artifacts": [],
        "hint": "",
    }
