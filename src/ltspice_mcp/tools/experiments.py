"""Consolidated durable experiment submission tool."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

from mcp import types
from pydantic import (
    BeforeValidator,
    Field,
    SkipValidation,
    ValidationError,
    field_serializer,
    model_validator,
)

from ltspice_mcp.errors import (
    NetlistError,
    PathSecurityError,
    ResultError,
    SimulationError,
    raise_site_code,
)
from ltspice_mcp.lib import (
    CIRCUIT_EXTENSIONS,
    NETLIST_SUFFIX_TEXT,
    experiment_store,
    now,
    response_budget,
)
from ltspice_mcp.lib.deck_prep import resolve_runnable_netlist
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    stage_deck,
    verify_staged_manifest,
)
from ltspice_mcp.lib.experiment_inputs import capture_case_inputs
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    AnalysisCallback,
    ExperimentReceipt,
    ExperimentRunRequest,
    IdempotencyConflictError,
    RequestGateBusy,
    StagedDecks,
    SubmissionCommitted,
    canonical_fingerprint,
    verify_replay,
)
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.hierarchy import SemanticProfile
from ltspice_mcp.lib.lint_rules import RULES_BY_ID, lint_deck, linter_version
from ltspice_mcp.lib.native_inputs import NativeCaseValidator
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pdk_native import (
    NGBEHAVIOR,
    NativeCaseError,
    NativeRequest,
    NativeRequestError,
    profile_pins,
    validate_family_ownership,
    validate_request,
)
from ltspice_mcp.lib.recipes import (
    DISCRIMINANTS,
    Recipe,
    StepSelectionFields,
    validate_recipe,
)
from ltspice_mcp.lib.recovery_records import CaseAttempt, CaseRecovery, RecoveryError
from ltspice_mcp.lib.services import cp1252_ltspice
from ltspice_mcp.lib.simulator import (
    SIMULATOR_SELECTOR_PATTERN,
    current_ngbehavior,
    simulator_dialect,
    simulator_library_roots,
)
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import Store
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.lib.variations import (
    CircuitDeck,
    DeckFile,
    ExpandedCase,
    PdkNativeVariation,
    RandomVariation,
    Variation,
    VariationError,
    check_case_cap,
    check_random_families,
    derive_circuit_ids,
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
    NEW_WORK_ANNOTATIONS,
    NotedModel,
    OptionalRawSelectionFields,
    ResponseBudget,
    StrictModel,
    ToolInput,
    format_response,
    held_to_cap,
    outcome_of,
    path_denied_text,
    registry,
    resolve_response_budget,
    resolve_run_simulator,
    safe_path,
)
from ltspice_mcp.tools._schema import prune_unreferenced_defs
from ltspice_mcp.tools.analyze import (
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
    snapshot_receipt_live,
)
from ltspice_mcp.tools.reference_index import validation_error_detail

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
            f"Deck to run: {NETLIST_SUFFIX_TEXT}, or an .asc, which LTspice exports. "
            "It is staged content-addressed at submission, so later edits to the "
            "file cannot change what this job ran."
        ),
    )
    id: str | None = Field(
        default=None,
        description=(
            "Names this circuit in 'applies_to' and in its rows. "
            "Default: the file stem, made valid and unique."
        ),
    )


class ExperimentExecution(StrictModel):
    """How long this call waits, how hard the job runs, and on which simulator."""

    recoverable: bool = Field(
        default=False,
        strict=True,
        description=(
            "Freeze validated inputs for jobs(action='resume'); unsupported inputs "
            "refuse before submission. Defaults to false; see spice://guide."
        ),
    )

    simulator_seed: int | None = Field(
        default=None,
        strict=True,
        ge=1,
        le=2147483646,
        description=(
            "Reseed ngspice before loading every case, including retries. Requires "
            "recoverable ngspice with one .op, .ac, .dc or .tran; cannot mix with native statistics."
        ),
    )

    @model_validator(mode="after")
    def _seed_requires_recovery(self) -> ExperimentExecution:
        if self.simulator_seed is not None and (
            not self.recoverable or self.simulator == "ltspice"
        ):
            raise ValueError("execution.simulator_seed requires recoverable ngspice")
        return self

    wait_s: float = Field(
        default=60.0,
        ge=0.0,
        description=(
            "Dwell before returning, held to 120s; the job keeps running. "
            "Continue with jobs(action='wait'); 0 returns immediately."
        ),
    )

    run_timeout_s: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Kill any single case whose simulator exceeds this and mark it failed; "
            "the other cases continue. Unset: none, or [simulation] run_timeout."
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
    simulator: str | None = Field(
        default=None,
        pattern=SIMULATOR_SELECTOR_PATTERN,
        description=(
            "Engine for every case; its family decides the dialect the results are "
            "parsed with. 'family:name' (e.g. 'ltspice:xvii') runs a named executable. "
            "inspect capabilities lists those and which simulators a run can select. "
            "Defaults to the server's default simulator."
        ),
    )


class AnalysisPerRun(analyze.CappedPerRunLimit):
    # The limit and its cap are analyze_results' own, never a copy of its
    # number: the attached block is handed straight to that engine, so a bound
    # this schema held to and the engine's could not drift apart.
    cursor: str | None = Field(
        default=None,
        description=(
            "Nothing has been paged at submission time; leave it unset and page the "
            "finished analysis with analyze_results."
        ),
    )


class AnalysisInclude(NotedModel):
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

    @model_validator(mode="after")
    def _fields_read_once(self) -> AnalysisInclude:
        if self.fields is not None:
            self.keep_first("fields")
        return self


coerce_attached_include_flags = include_flag_coercer(AnalysisInclude)


class AttachedAnalysis(OptionalRawSelectionFields, StepSelectionFields, NotedModel):
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

    @model_validator(mode="after")
    def _group_by_read_once(self) -> AttachedAnalysis:
        # Normalized here, not only when the analysis runs: group_by is part
        # of the request fingerprint, so a repeat must not make a resend of
        # the same request look like a different one.
        self.keep_first("group_by")
        return self

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
        "api.reference('analyze_results') or guide section 'tools'."
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
        # Preserve canonical bytes for ordinary requests recorded before the
        # opt-in existed. Only requesting recovery changes execution identity.
        if not self.execution.recoverable:
            payload["execution"].pop("recoverable", None)
        if self.execution.simulator_seed is None:
            payload["execution"].pop("simulator_seed", None)

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
    title="Run Simulations",
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
    annotations=NEW_WORK_ANNOTATIONS,
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
    wait_s, wait_note = held_to_cap(
        "execution.wait_s", args.execution.wait_s, SUBMISSION_DWELL_CAP_S, "s"
    )
    cap_warnings = _argument_warnings(args, wait_note)
    try:
        replay = await _load_matching_replay(args, state, fingerprint)
        if replay is not None:
            try:
                return await _dwell_and_respond(
                    replay,
                    wait_s,
                    state,
                    provenance=args.provenance,
                    run_fields=args.run_fields,
                    analysis_fields=analysis_fields,
                    budget=budget,
                    warnings=cap_warnings,
                )
            except Exception as exc:
                return await _post_submit_error_response(
                    replay, exc, None, budget=budget, state=state
                )

        simulator = resolve_run_simulator(args.execution.simulator, state)
        if args.execution.simulator_seed is not None and (
            simulator_dialect(simulator) != "ngspice"
            or any(isinstance(item, PdkNativeVariation) for item in args.variations)
        ):
            raise RecoveryError(
                "recovery_seed_unsupported",
                "Explicit seed requires recoverable ngspice without native statistics",
            )
        circuit_inputs, id_notes = _circuit_decks_for_validation(args.circuits)
        normalize_circuit_decks(circuit_inputs)
        validate_variation_circuit_ids(circuit_inputs, args.variations)
        native_ids = [
            item.id.casefold() for item in args.variations if isinstance(item, PdkNativeVariation)
        ]
        if len(set(native_ids)) != len(native_ids):
            raise NativeRequestError("native family ids must be unique")
        for circuit in circuit_inputs:
            native = _native_family(circuit.circuit_id, args.variations)
            if native is not None:
                validate_request(
                    _native_request(native, circuit.circuit_id, native.sample_start),
                    backend=simulator_dialect(simulator) or "",
                )
        check_random_families([circuit.circuit_id for circuit in circuit_inputs], args.variations)
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

        # Staging runs inside the coordinator's request gate, not here: a second
        # call carrying this request_id waits on that gate and replays the job it
        # finds, so only one submission ever copies a deck set. The findings this
        # pass produces are read back through the closure below, and stay empty
        # when the gate answered with a replay — the recorded job's own findings
        # are what a replay receipt reports.
        lint_by_circuit: dict[str, list[dict[str, Any]]] = {}

        async def stage_decks() -> StagedDecks:
            cases: list[ExperimentCase] = []
            sources: list[SourceRecord] = []
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
                ),
                # Admission can refuse a frozen-input capture. Wait for every
                # staging worker before the coordinator cleans an unclaimed tree.
                return_exceptions=True,
            )
            for preparation in preparations:
                if isinstance(preparation, BaseException):
                    raise preparation
                lint_by_circuit[preparation.circuit_id] = preparation.lint_findings
                if preparation.source is not None:
                    sources.append(preparation.source)
                id_note = id_notes.get(preparation.circuit_id)
                for case in preparation.cases:
                    case.run_index = len(cases)
                    if id_note is not None:
                        case.observations.append(copy.deepcopy(id_note))
                    cases.append(case)

            if len(cases) != projected:
                raise VariationError(
                    "completeness_mismatch",
                    f"Prepared {len(cases)} cases but variation expansion declared {projected}",
                )
            return StagedDecks(cases=cases, sources=sources)

        analysis_request = args.strip_presentation()["analyze"]
        # Identified once, before the request gate and after every cheap check
        # (the first identification in a process digests the executable): the
        # gate's replay check compares it with what a recorded job ran on, and
        # a new job records it.
        executable = await asyncio.to_thread(executable_identity, simulator)
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
            stage=stage_decks,
            simulator=simulator.__name__,
            simulator_executable=executable,
            job_id=job_id,
            declared=len(args.circuits),
            max_parallel=args.execution.max_parallel,
            run_timeout_s=args.execution.run_timeout_s,
            job_deadline_s=args.execution.job_deadline_s,
            recoverable=args.execution.recoverable,
            simulator_seed=args.execution.simulator_seed,
            analysis_request=analysis_request,
            analysis_callback=(
                attached_analysis_callback(state) if analysis_request is not None else None
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
                wait_s,
                state,
                lint_by_circuit=lint_by_circuit or None,
                provenance=args.provenance,
                run_fields=args.run_fields,
                analysis_fields=analysis_fields,
                budget=budget,
                warnings=cap_warnings,
            )
        except Exception as exc:
            return await _post_submit_error_response(
                receipt,
                exc,
                lint_by_circuit or None,
                budget=budget,
                state=state,
            )
    except SubmissionCommitted as exc:
        if exc.receipt is not None:
            return await _post_submit_error_response(
                exc.receipt, exc, None, budget=budget, state=state
            )
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="submission",
            retryable=True,
            # The submission is durable; only the rest of the call fell over.
            commit_state="committed",
            budget=budget,
        )
    except RequestGateBusy as exc:
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="submission",
            retryable=True,
            # The holder is the submission that claims this id, and it was
            # still working when the wait ran out. Saying not_started here
            # would license a resubmission under a fresh id, which is how one
            # experiment ends up running twice.
            commit_state="unknown",
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
    except NativeRequestError as exc:
        return await _error_response(
            args.request_id,
            code="pdk_native_request",
            message=str(exc),
            stage="variation",
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
            hint=path_denied_text(exc, state),
        )
    except RecoveryError as exc:
        return await _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="recovery",
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
    native = _native_family(circuit_id, args.variations)
    expected_count = projected_case_count(circuit_id, args.variations)
    source_path: Path | None = None
    runnable: Path | None = None
    expanded: list[ExpandedCase] | None = None
    source: SourceRecord | None = None
    findings: list[dict[str, Any]] = []
    cases: list[ExperimentCase] = []
    try:
        source_path = safe_path(circuit_arg.path, state)
        if source_path.suffix.casefold() not in CIRCUIT_EXTENSIONS:
            raise VariationError(
                "unsupported_variant",
                f"Circuit {circuit_id!r} uses unsupported extension "
                f"{source_path.suffix or '<none>'!r}; supported extensions are "
                f"{NETLIST_SUFFIX_TEXT} and .asc",
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
        dialect = simulator_dialect(simulator)
        compact_digests: frozenset[str] = frozenset()
        if native is not None and sys.platform == "win32" and dialect == "ngspice":
            # ngspice's Windows file reader cannot open the long paths produced
            # by the full PDK layout. Compact only the audited pinned models.
            compact_digests = frozenset((await asyncio.to_thread(profile_pins)).values())
        staged = await asyncio.to_thread(
            stage_deck,
            runnable,
            paths.staging_root,
            state.allowed_paths(),
            # A schematic is simulated through an exported netlist, so without
            # this the manifest describes only that export — and re-exporting is
            # exactly what a replay skips, leaving an edited .asc invisible to
            # every check made over this record.
            origin=source_path,
            allow_live_includes=args.allow_live_includes,
            windows_paths=paths.windows_native,
            compact_digests=compact_digests,
            # LTspice's own .asc netlister appends a .lib pointing into the
            # install's model library on every schematic with a MOSFET on it,
            # so without this no transistor sheet stages under a default
            # sandbox. Resolved per run from the simulator this job uses.
            simulator_roots=await asyncio.to_thread(simulator_library_roots, simulator),
            # A micro sign spelled 'u' changes what a value means only to an
            # LTspice that decodes decks as cp1252, so only then is it reported.
            # The identity is cached per executable, so this is a stat here.
            cp1252_reader=cp1252_ltspice(
                state, await asyncio.to_thread(executable_identity, simulator)
            ),
        )
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
                ngbehavior=NGBEHAVIOR if native is not None else None,
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
            codec=staged.codec,
            includes=tuple(
                DeckFile(
                    path=included.staged_path,
                    text=included.text,
                    sha256=included.sha256,
                    codec=included.codec,
                )
                for included in staged.includes
            ),
            semantic_profile=(
                SemanticProfile(
                    "ngspice" if dialect == "ngspice" else "ltspice",
                    (NGBEHAVIOR if native is not None else current_ngbehavior() or "")
                    if dialect == "ngspice"
                    else None,
                )
                if dialect in {"ltspice", "ngspice"}
                else None
            ),
            record_source_lineage=native is not None or args.execution.recoverable,
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
            native_validator = NativeCaseValidator(staged, paths.staging_root)
            lineage_root = (
                await asyncio.to_thread(Store(state.working_dir).run_dir, job_id, simulator)
                if args.execution.recoverable
                else None
            )
            for offset, (variant, descriptor) in enumerate(
                zip(materialized, expanded, strict=True)
            ):
                case = ExperimentCase(
                    case_id=variant.case_id,
                    run_index=offset,
                    circuit=circuit_id,
                    circuit_path=source_path,
                    staged_deck=variant.path,
                    deck_sha256=variant.sha256,
                    assignments=variant.assignments,
                    observations=list(observations),
                )
                if native is not None:
                    request = _native_request(native, circuit_id, descriptor.native_index)
                    case.native_statistics = NativeCaseRecord(request)
                    try:
                        case.native_statistics = await asyncio.to_thread(
                            native_validator.validate,
                            request,
                            variant,
                            descriptor,
                        )
                    except NativeCaseError as exc:
                        case.status = "failed"
                        case.failure_code = "pdk_native_validation"
                        case.failure_evidence = {"reason": exc.code}
                        case.error = str(exc)
                        case.completed_at = now()
                        case.native_statistics.unavailable_reason = str(exc)
                if args.execution.recoverable and case.status == "queued":
                    assert lineage_root is not None
                    inputs = await asyncio.to_thread(
                        capture_case_inputs,
                        case,
                        source,
                        lineage_root=lineage_root,
                        materialized=variant,
                        seeded=args.execution.simulator_seed is not None,
                    )
                    case.recovery = CaseRecovery(
                        inputs,
                        CaseAttempt(job_id, 0, f"{job_id}-{offset:04d}"),
                    )
                cases.append(case)
    except RecoveryError:
        # Opt-in is an admission contract: a capture refusal must reach the
        # caller before the coordinator can claim any of this inventory.
        raise
    except NativeRequestError:
        raise
    except (
        PathSecurityError,
        NetlistError,
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
    if native is not None:
        for index, case in enumerate(cases):
            if case.native_statistics is None:
                case.native_statistics = NativeCaseRecord(
                    _native_request(native, circuit_id, native.sample_start + index % native.runs),
                    unavailable_reason=case.error or case.status,
                )
    return _CircuitPreparation(
        circuit_id=circuit_id,
        cases=cases,
        source=source,
        lint_findings=findings,
    )


def _native_family(circuit_id: str, variations: list[Variation]) -> PdkNativeVariation | None:
    applicable = [
        item
        for item in variations
        if item.applies_to is None
        or circuit_id.casefold() in {name.casefold() for name in item.applies_to}
    ]
    native = [item for item in applicable if isinstance(item, PdkNativeVariation)]
    validate_family_ownership(
        [item.id for item in native],
        caller_random=any(isinstance(item, RandomVariation) for item in applicable),
    )
    return native[0] if native else None


def _native_request(
    family: PdkNativeVariation, circuit_id: str, index: int | None
) -> NativeRequest:
    return NativeRequest(circuit_id, family.id, family.profile, family.mode, family.seed, index)


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
                "plot_index": request.get("plot_index"),
                "dialect": request.get("dialect"),
            }
        ],
        "recipes": request.get("recipes") or [],
        "group_by": request.get("group_by") or [],
        # The one serializer for a step selection, so the two keys are spelled
        # here exactly as analyze_results stores them.
        **analyze.StepSelection.from_inputs(request).as_inputs(),
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
            f"{validation_error_detail('analyze_results', exc)}"
        ) from exc


def attached_analysis_callback(state: SessionState) -> AnalysisCallback:
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
                f"{validation_error_detail('analyze_results', exc)}"
            ) from exc
        # Resolved on the module, not bound at import: the analysis stage is
        # patched through ``tools.analyze`` in tests, and the attribute lookup
        # is what keeps that seam where the engine actually lives.
        return await analyze.capture_attached_analysis(args, state)

    return run_attached_analysis


def _circuit_decks_for_validation(
    circuits: list[ExperimentCircuit],
) -> tuple[list[CircuitDeck], dict[str, dict[str, Any]]]:
    """The circuits as decks to validate, and an observation per derived id.

    A circuit with no ``id`` takes its file stem, made valid and unique rather
    than refused (see ``derive_circuit_ids``); the observation, keyed by the id
    it ran under, says so on every case of that circuit.
    """
    derived = derive_circuit_ids(
        [circuit.path for circuit in circuits], [circuit.id for circuit in circuits]
    )
    decks: list[CircuitDeck] = []
    notes: dict[str, dict[str, Any]] = {}
    for circuit, (circuit_id, note) in zip(circuits, derived, strict=True):
        decks.append(CircuitDeck(circuit_id=circuit_id, path=Path(circuit.path), text=""))
        if note is not None:
            notes[circuit_id] = {
                "code": "circuit_id_derived",
                "kind": "provenance",
                "detail": note,
                "evidence": {"path": circuit.path, "circuit_id": circuit_id},
            }
    return decks, notes


async def _load_matching_replay(
    args: RunExperimentsInput,
    state: SessionState,
    fingerprint: str,
) -> ExperimentReceipt | None:
    """Resolve a recorded submission before entering new staging.

    Only for a request_id the caller passed. An id this server minted a moment
    ago cannot name a recorded submission, so looking one up is a thread hop
    and a failed open on the majority of calls. Nothing is missed: the request
    gate does the authoritative lookup either way.
    """
    if "request_id" not in args.model_fields_set:
        return None
    from ltspice_mcp.lib.experiment_resume import lookup_root_recovery

    simulator = resolve_run_simulator(args.execution.simulator, state)
    recovery = await lookup_root_recovery(
        state,
        request_id=args.request_id,
        fingerprint=fingerprint,
        simulator=simulator,
        recoverable=args.execution.recoverable,
        analysis_callback=attached_analysis_callback(state) if args.analyze is not None else None,
    )
    if recovery is not None:
        return recovery
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
    if job is None:
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
    # What this request would run on now, resolved as a fresh submission
    # resolves it: a default simulator that changed since is a different build.
    await asyncio.to_thread(
        lambda: verify_replay(job, args.request_id, executable_identity(simulator))
    )
    # The record is left as it is: the receipt's ``replayed`` is the fact about
    # this call, and only the owner writes a job's record.
    return ExperimentReceipt(job=job, replayed=True, control_token=job.control_token)


def _argument_warnings(args: RunExperimentsInput, wait_note: str | None) -> list[str]:
    """What this call asked for that is served differently: a value held to its
    cap, and a repeat in the attached analysis read once."""
    notes: list[str] = []
    if wait_note is not None:
        notes.append(
            f"{wait_note} The job keeps running; continue with "
            f"jobs(action='wait', timeout_s<={JOBS_WAIT_CAP_S:g})."
        )
    if args.analyze is not None:
        notes.extend(f"analyze.{note}" for note in args.analyze.argument_notes())
        per_run = args.analyze.include.per_run if args.analyze.include else None
        held = per_run.limit_note() if per_run is not None else None
        if held is not None:
            notes.append(f"analyze.include.per_run.{held}")
    return notes


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
    warnings: list[str] | None = None,
) -> types.CallToolResult:
    job = receipt.job
    if job.status not in TERMINAL_EXPERIMENT_STATUSES and wait_s > 0:
        runner = state.runners.get_experiment_runner_for(job)
        if runner is not None:
            await runner.wait(job, wait_s, wait_for="all")
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done_event.wait(), wait_s)
    snapshot = await snapshot_receipt_live(
        job,
        state,
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
        data["replayed"] = receipt.replayed
        if warnings:
            data["warnings"] = [*warnings, *data.get("warnings", [])]
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


def submission_error_payload(
    request_id: str,
    *,
    code: str,
    message: str,
    stage: str,
    retryable: bool,
    commit_state: Literal["not_started", "committed", "unknown"],
    hint: str | None = None,
) -> dict[str, Any]:
    """The receipt-shaped payload for a call that produced no job.

    Public because a submission can now fail outside this handler: the Python
    API's detached mode spawns a process to submit, and a failure there has to
    reach the caller in the same shape as a failure here. ``hint`` defaults to
    the message; a failure with a known remedy passes the message with it.
    """
    data = _empty_payload(request_id)
    data.update(
        {
            "hint": hint if hint is not None else message,
            "error": {
                "code": code,
                "message": message,
                "stage": stage,
                "retryable": retryable,
                "commit_state": commit_state,
            },
        }
    )
    return finalize_receipt(data)


async def _error_response(
    request_id: str,
    *,
    code: str,
    message: str,
    stage: str,
    retryable: bool,
    commit_state: Literal["not_started", "committed", "unknown"],
    budget: ResponseBudget,
    hint: str | None = None,
) -> types.CallToolResult:
    data = submission_error_payload(
        request_id,
        code=code,
        message=message,
        stage=stage,
        retryable=retryable,
        commit_state=commit_state,
        hint=hint,
    )
    return await _render_static_run_receipt(
        data,
        data["hint"],
        budget,
        is_error=True,
    )


async def _post_submit_error_response(
    receipt: ExperimentReceipt,
    exc: Exception,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None,
    *,
    budget: ResponseBudget,
    state: SessionState,
) -> types.CallToolResult:
    """Envelope for a failure after the experiment committed.

    Commitment is irreversible even when launch is still pending. The job_id
    and control_token must survive a response failure so the caller can follow
    or cancel the committed work instead of submitting it again under a new id.
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
            state,
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
    data["replayed"] = receipt.replayed
    route = (
        f"The experiment is committed. Use jobs(status) with job_id "
        f"{job.job_id} to follow it, or jobs(cancel) with that job_id and its "
        f"control_token to stop it."
    )
    data["hint"] = route
    data["error"] = {
        "code": (
            exc.code
            if isinstance(exc, SubmissionCommitted)
            else raise_site_code(exc) or "receipt_failed"
        ),
        "message": error_message(exc, build_error),
        "stage": "submission" if isinstance(exc, SubmissionCommitted) else "receipt",
        "retryable": True,
        "commit_state": "committed",
    }
    finalize_receipt(data)
    text = f"Experiment {job.job_id} was committed, but completing the call failed: {exc}."
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
        result.is_error = True
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
        # Nothing came back, so nothing was answered from an existing job.
        "replayed": False,
    }
