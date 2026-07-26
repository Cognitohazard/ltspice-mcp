"""Consolidated durable experiment submission tool."""

from __future__ import annotations

import asyncio
import contextlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self

from mcp import types
from pydantic import Field, ValidationError, model_validator

from ltspice_mcp.errors import (
    JobNotFoundError,
    LTSpiceMCPError,
    PathSecurityError,
    SimulationError,
)
from ltspice_mcp.lib import experiment_store, job_store, recent, services
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    stage_deck,
    verify_staged_manifest,
)
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    AnalysisCallback,
    ExperimentCancellationError,
    ExperimentReceipt,
    ExperimentRunRequest,
    IdempotencyConflictError,
    canonical_fingerprint,
    verify_replay_sources,
)
from ltspice_mcp.lib.experiment_types import (
    TERMINAL_CASE_STATUSES,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.job_lifecycle import runs_terminal
from ltspice_mcp.lib.job_types import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    SimulationJob,
)
from ltspice_mcp.lib.lint_rules import RULES_BY_ID, lint_deck, linter_version
from ltspice_mcp.lib.recipes import validate_recipe
from ltspice_mcp.lib.simulator import simulator_dialect
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
    StrictModel,
    ToolInput,
    format_response,
    keep_plan,
    paginate,
    pagination_metadata,
    project_row,
    registry,
    resolve_run_simulator,
    resolve_runnable_netlist,
    result_text,
    safe_path,
)
from ltspice_mcp.tools.analyze import MAX_PAGE_SIZE

_RUN_PAGE_LIMIT = 50
# The source label an attached analysis analyzes its own experiment under.
_ATTACHED_ANALYSIS_LABEL = "experiment"
_TERMINAL_EXPERIMENT_STATUSES = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
        "interrupted",
    }
)


@dataclass
class _CircuitPreparation:
    circuit_id: str
    cases: list[ExperimentCase]
    source: SourceRecord | None
    lint_findings: list[dict[str, Any]]


class ExperimentCircuit(StrictModel):
    path: str = Field(
        description=(
            "Deck to run: .cir/.net/.sp, or an .asc exported through LTspice first. "
            "It is staged content-addressed at submission, so later edits to the "
            "file cannot change what this job ran."
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
        le=120.0,
        description=(
            "How long this call waits for the job before returning a receipt, "
            "0-120s. It bounds the RESPONSE only: the job is durable and keeps "
            "running past it, and 0 returns the receipt immediately."
        ),
    )
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
    per_run: AnalysisPerRun | None = Field(
        default=None,
        description=(
            "Return the individual attributed rows, paginated. Omitted, a recipe "
            "with 'reduce' returns only its reductions."
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
            "run — a discovery aid, not something to leave on."
        ),
    )


class AttachedAnalysis(StrictModel):
    recipes: list[dict[str, Any]] = Field(
        description=(
            "analyze_results recipes, in that tool's exact shape, run over this "
            "job's own runs once they finish. Saves a round trip when the "
            "measurements are known up front."
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description=(
            "Split reductions along these dimensions: a variation assignment "
            "parameter name, 'circuit', or a .step axis name."
        ),
    )
    include: AnalysisInclude | None = Field(
        default=None,
        description=(
            "Optional analysis blocks. Field projection (analyze_results "
            "include.fields) is not available on the attached stage — for that, "
            "analyze the finished job with analyze_results."
        ),
    )


class RunExperimentsInput(ToolInput):
    # Fields that choose how the receipt is rendered rather than what runs.
    # canonical_fingerprint excludes them, so re-asking for the same experiment
    # at a different verbosity replays instead of conflicting.
    PRESENTATION_FIELDS: ClassVar[frozenset[str]] = frozenset({"provenance", "run_fields"})

    request_id: str = Field(
        default_factory=lambda: generate_id("req"),
        min_length=1,
        description=(
            "Idempotency key, optional. Omit it for a one-off run — a fresh id is "
            "generated and echoed on the receipt. Pass your own to make submission "
            "durable: the same id with the same arguments AND unchanged source "
            "decks replays the existing receipt instead of running anything again; "
            "the same id after either changed is a conflict, not a replay."
        ),
    )
    circuits: list[ExperimentCircuit] = Field(
        min_length=1,
        description=(
            "Decks to run. Several circuits in one call share the variation grid "
            "and one job, which is how designs are compared under identical "
            "conditions; scope a variation to one of them with its 'applies_to'."
        ),
    )
    variations: list[Variation] = Field(
        default_factory=list,
        description=(
            "The sweep. 'assign' entries build the case grid (cartesian across "
            "entries, or lock-step within one entry via combine:'zip'); at most one "
            "'random' entry adds Monte Carlo runs. Cases run in PARALLEL up to the "
            "concurrency cap, so a whole grid costs little more wall-clock than a "
            "single case — express the sweep here rather than as repeated one-case "
            "calls. Empty runs each circuit once as authored."
        ),
    )
    execution: ExperimentExecution = Field(
        default_factory=ExperimentExecution,
        description=(
            "How the job runs and how long this call dwells: wait_s bounds only "
            "this response (the job is durable either way), plus per-case "
            "timeout_s, simulator choice, and the parallelism cap. Defaults suit "
            "a quick check."
        ),
    )
    analyze: AttachedAnalysis | None = Field(
        default=None,
        description=(
            "Measure the runs as a stage of this job, so terminal responses carry "
            "the numbers already. Its failure does not fail the runs; it sets its "
            "own analysis status and the runs stay analyzable with analyze_results."
        ),
    )
    lint: Literal["block", "warn", "off"] = Field(
        default="block",
        description=(
            "What to do with SPICE lint findings on the staged deck. 'block' refuses "
            "to submit a circuit with a blocking finding (its cases are reported "
            "'skipped'); 'warn' runs anyway and reports them; 'off' skips linting. "
            "Prefer 'suppress' over lowering this — the rules catch decks that "
            "simulate to silently wrong answers."
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
            "Let a .include/.lib that cannot be staged (outside the allowed roots, "
            "or past the recursion depth) be read live at run time instead of "
            "failing that circuit's submission. The job then cannot prove what "
            "those files held when it ran: the manifest marks the reference "
            "live:true, an observation says so, and reusing its request_id runs "
            "the experiment again rather than replaying a receipt whose inputs "
            "cannot be checked."
        ),
    )
    provenance: bool = Field(
        default=False,
        description=(
            "Emit the full audit trail on each source: content digests, the staged "
            "deck path, every staged file, and the linter version. Off by default "
            "because it is a third of a receipt's bytes and names files you do not "
            "open — the analysis tools address runs by job_id. Entries that say "
            "something actionable (a live, unprovable include) are reported either "
            "way."
        ),
    )
    run_fields: list[str] | None = Field(
        default=None,
        description=(
            "Keep only these keys on each row of 'runs.items', dotted for nesting "
            "(e.g. 'case_id', 'assignments.RDEG'). Nesting is preserved, so a row "
            "reads the same way with fewer keys. Use it on a wide sweep: the run "
            "list is per-case and grows with the grid. Escape a dot inside a key "
            "name as '\\.'."
        ),
    )


_FINDING_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_id": {"type": "string"},
        "severity": {"type": "string"},
        "ok": {"type": "boolean"},
        "evidence": {},
        "at": {
            "type": "object",
            "properties": {
                "file": {"type": "string"},
                "line": {"type": "integer"},
            },
            "required": ["file"],
        },
        "subject": {"type": "string"},
    },
    "required": ["rule_id", "severity", "ok", "evidence", "at", "subject"],
}

_MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "sha256": {"type": "string"},
        "staged": {"type": "boolean"},
        "live": {"type": "boolean"},
        "staged_path": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "section": {"type": ["string", "null"]},
    },
    "required": [
        "path",
        "sha256",
        "staged",
        "live",
        "staged_path",
        "reason",
        "section",
    ],
}

_OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "kind": {"type": "string"},
        "detail": {"type": "string"},
        "evidence": {},
    },
    "required": ["code", "kind", "detail"],
}

_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "code": {"type": "string"},
        "message": {"type": "string"},
    },
    "required": ["case_id", "code", "message"],
}

_ARTIFACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content_type": {"type": "string"},
        "sha256": {"type": "string"},
        "bytes": {"type": "integer"},
    },
    "required": ["path", "content_type", "sha256", "bytes"],
}

RUN_EXPERIMENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": ["string", "null"]},
        "request_id": {"type": "string"},
        "control_token": {"type": "string"},
        "status": {"type": "string"},
        "outcome": {
            "type": "string",
            "enum": ["complete", "partial", "failed", "in_progress"],
        },
        "source": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "path": {"type": "string"},
                    "sha256": {"type": "string"},
                    "staged_deck": {"type": "string"},
                    "manifest": {"type": "array", "items": _MANIFEST_SCHEMA},
                    "linter_version": {"type": "string"},
                    "simulator": {"type": "string"},
                    "dialect": {"type": ["string", "null"]},
                    "staged_files": {"type": "integer"},
                },
                # Only the fields every receipt carries are required. The
                # digests, staging paths and full manifest are provenance: they
                # are emitted when 'provenance' is set, and otherwise omitted,
                # because they are a third of a receipt's bytes and name files
                # the caller does not open.
                "required": [
                    "circuit",
                    "path",
                    "simulator",
                    "dialect",
                ],
            },
        },
        "completeness": {
            "type": "object",
            "properties": {
                "declared": {"type": "integer"},
                "expanded": {"type": "integer"},
                "submitted": {"type": "integer"},
                "produced": {"type": "integer"},
                "failed": {"type": "integer"},
                "cancelled": {"type": "integer"},
                "skipped": {"type": "integer"},
            },
            "required": [
                "declared",
                "expanded",
                "submitted",
                "produced",
                "failed",
                "cancelled",
                "skipped",
            ],
        },
        "lint": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "findings": {"type": "array", "items": _FINDING_SCHEMA},
                },
                "required": ["circuit", "findings"],
            },
        },
        "runs": {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "case_id": {"type": "string"},
                            "run_index": {"type": "integer"},
                            "circuit": {"type": "string"},
                            "assignments": {"type": "object"},
                            "status": {"type": "string"},
                            "raw": {"type": ["string", "null"]},
                            "log": {"type": ["string", "null"]},
                        },
                        # Nothing is required: 'run_fields' lets the caller keep
                        # only the keys it wants, so any key here is one a
                        # projection may legitimately have dropped. Declaring
                        # them required would make a projected receipt violate
                        # this tool's own schema.
                    },
                },
                "total": {"type": "integer"},
                "returned": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "next_cursor": {"type": ["string", "null"]},
            },
            "required": ["items", "total", "returned", "truncated", "next_cursor"],
        },
        "analysis": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "request": {"type": ["object", "null"]},
                "result": {"type": ["object", "null"]},
                "error": {"type": ["string", "null"]},
                "observations": {
                    "type": "array",
                    "items": _OBSERVATION_SCHEMA,
                },
            },
            # "request" is the caller's own input replayed back — emitted
            # under provenance only, so it cannot be required.
            "required": [
                "status",
                "result",
                "error",
                "observations",
            ],
        },
        "failures": {"type": "array", "items": _FAILURE_SCHEMA},
        "observations": {"type": "array", "items": _OBSERVATION_SCHEMA},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
        "hint": {"type": "string"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "stage": {"type": "string"},
                "retryable": {"type": "boolean"},
                "commit_state": {
                    "type": "string",
                    "enum": ["not_started", "committed", "unknown"],
                },
                "item_id": {"type": "string"},
            },
            "required": ["code", "message", "stage", "retryable", "commit_state"],
        },
    },
    "required": [
        "job_id",
        "request_id",
        "status",
        "outcome",
        "source",
        "completeness",
        "lint",
        "runs",
        "failures",
        "observations",
        "warnings",
        "artifacts",
        "hint",
    ],
}


@registry.tool(
    name="run_experiments",
    description=(
        "Run SPICE and get the numbers back in one call: point it at your deck(s), "
        "attach an 'analyze' block, and the measured values return with the "
        "results — from a one-off spot check to a full sweep or Monte Carlo grid. "
        "Cases run in parallel, so express the whole sweep as one 'variations' "
        "grid rather than a call per point; a large grid costs about what one "
        "case costs. When unsure about a behavior, assumption, or sizing, run a "
        "small experiment and read the numbers rather than reasoning it out. "
        "Quick runs return results inline; longer ones return a receipt to follow "
        "with 'jobs', and passing a request_id makes the submission durable and "
        "idempotent across retries."
    ),
    input_model=RunExperimentsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
    profiles=("consolidated",),
    output_schema=RUN_EXPERIMENTS_OUTPUT_SCHEMA,
)
async def handle_run_experiments(
    args: RunExperimentsInput,
    state: SessionState,
) -> types.CallToolResult:
    """Stage, lint, expand, durably submit, and dwell on an experiment."""
    fingerprint = canonical_fingerprint(args)
    try:
        replay = await _load_matching_replay(args, state, fingerprint)
        if replay is not None:
            return await _dwell_and_respond(
                replay,
                args.execution.wait_s,
                state,
                provenance=args.provenance,
                run_fields=args.run_fields,
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

        job_id = generate_id("exp")
        try:
            route = await asyncio.to_thread(
                resolve_experiment_paths,
                state.working_dir,
                job_id,
                circuit_inputs[0].circuit_id,
                simulator,
            )
        except DeckStagingError as exc:
            return _routing_failure_response(args, circuit_inputs, exc, projected)
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

        analysis_request = (
            args.analyze.model_dump(mode="json", exclude_unset=False)
            if args.analyze is not None
            else None
        )
        runner = state.runners.get_experiment_runner(
            loop=asyncio.get_running_loop(),
            simulator_class=simulator,
            output_folder=route.output_folder,
            max_parallel=args.execution.max_parallel or state.config.max_parallel_sims,
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
            )
        except Exception as exc:
            return _post_submit_error_response(receipt, exc, lint_by_circuit)
    except IdempotencyConflictError as exc:
        return _error_response(
            args.request_id,
            code="idempotency_conflict",
            message=str(exc),
            stage="submission",
            retryable=False,
            commit_state="not_started",
        )
    except VariationError as exc:
        return _error_response(
            args.request_id,
            code=exc.code,
            message=str(exc),
            stage="variation",
            retryable=False,
            commit_state="not_started",
        )
    except PathSecurityError as exc:
        return _error_response(
            args.request_id,
            code="path_denied",
            message=str(exc),
            stage="resolution",
            retryable=False,
            commit_state="not_started",
        )
    except (SimulationError, DeckStagingError, OSError, ValueError) as exc:
        return _error_response(
            args.request_id,
            code=getattr(exc, "code", "submission_failed"),
            message=str(exc),
            stage="submission",
            retryable=True,
            commit_state="not_started",
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

    One builder for the submission pre-flight AND the post-run analysis
    stage, so what the pre-flight validates is byte-for-byte what will run.
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
    fingerprint, so the retry re-runs every case. The probe job id never
    resolves because shape validation does not touch the registry.
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
            f"The attached analyze block is not a valid analyze_results request: {exc}"
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
                f"The attached analyze block is not a valid analyze_results request: {exc}"
            ) from exc
        # Resolved on the module, not bound at import: the analysis stage is
        # patched through ``tools.analyze`` in tests, and the attribute lookup
        # is what keeps that seam where the engine actually lives.
        result = await analyze.handle_analyze_results(args, state)
        data = result.structuredContent
        if result.isError or data is None:
            raise SimulationError(
                result_text(result, joined=True)
                or "Attached analysis returned no structured result"
            )
        return data

    return run_attached_analysis


def _circuit_decks_for_validation(circuits: list[ExperimentCircuit]) -> list[CircuitDeck]:
    return [
        CircuitDeck(
            circuit_id=circuit.id or Path(circuit.path).stem,
            path=Path(circuit.path),
            text="",
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
) -> types.CallToolResult:
    job = receipt.job
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES and wait_s > 0:
        runner = state.runners.get_experiment_runner_for(job)
        if runner is not None:
            await runner.wait(job, wait_s, wait_for="all")
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done_event.wait(), wait_s)
    data = _job_payload(
        job,
        receipt.control_token,
        lint_by_circuit=lint_by_circuit,
        provenance=provenance,
        run_fields=run_fields,
    )
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES:
        data["hint"] = (
            f"Experiment {job.job_id} is still running; use jobs(wait) with this "
            "job_id to continue waiting."
        )
    text = (
        f"Experiment {job.job_id}: {job.status} "
        f"({job.completeness.terminal}/{job.completeness.expanded} terminal cases)"
    )
    return format_response(text, data)


def _job_payload(
    job: ExperimentJob,
    control_token: str | None,
    *,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
    provenance: bool = False,
    run_fields: list[str] | None = None,
) -> dict[str, Any]:
    if lint_by_circuit is None:
        lint_map = {}
        for case in job.cases:
            lint_map.setdefault(case.circuit, [])
        for source in job.sources:
            lint_map[source.circuit] = source.lint_findings
    else:
        lint_map = lint_by_circuit
    observations = list(job.observations)
    seen_observations = {(item.get("code"), item.get("detail")) for item in observations}
    for case in job.cases:
        for observation in case.observations:
            key = (observation.get("code"), observation.get("detail"))
            if key not in seen_observations:
                observations.append(observation)
                seen_observations.add(key)
    outcome = _terminal_outcome(job)
    runs = _runs_page(job.cases, run_fields)
    hint = _terminal_hint(job, runs["truncated"])
    data: dict[str, Any] = {
        "job_id": job.job_id,
        "request_id": job.request_id,
        "status": job.status,
        "outcome": outcome,
        "source": [_source_payload(source, provenance=provenance) for source in job.sources],
        "completeness": asdict(job.completeness),
        # Findings always emit; a circuit absent from the list is clean —
        # the empty-findings entry was per-circuit ceremony.
        "lint": [
            {"circuit": circuit, "findings": findings}
            for circuit, findings in lint_map.items()
            if findings
        ],
        "runs": runs,
        "failures": list(job.failures),
        "observations": observations,
        "warnings": [],
        "artifacts": list(job.artifacts),
        "hint": hint,
    }
    if control_token is not None:
        data["control_token"] = control_token
    if job.analysis.status != "not_requested":
        data["analysis"] = {
            "status": job.analysis.status,
            "result": job.analysis.result,
            "error": job.analysis.error,
            "observations": job.analysis.observations,
        }
        if provenance:
            # The caller's own attached-analysis input, replayed back —
            # proof of what ran, not something to re-read every turn.
            data["analysis"]["request"] = job.analysis.request
    return data


def _source_payload(source: SourceRecord, *, provenance: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "circuit": source.circuit,
        "path": str(source.path),
        "simulator": source.simulator,
        "dialect": source.dialect,
    }
    if provenance:
        payload["sha256"] = source.sha256
        payload["staged_deck"] = str(source.staged_deck)
        payload["manifest"] = [_manifest_payload(entry) for entry in source.manifest]
        payload["linter_version"] = source.linter_version
        return payload
    # Keep the entries that say something the caller must act on; dropping those
    # with the bulk would turn a disclosure into a silent omission. The predicate
    # is "not an ordinary staged reference" rather than a list of known-bad
    # states, so it fails CLOSED — the store rebuilds `staged` with a False
    # default, and an entry that is neither staged, live, nor explained is
    # exactly the anomaly that must not be hidden.
    notable = [
        entry for entry in source.manifest if entry.live or entry.reason or not entry.staged
    ]
    if notable:
        payload["manifest"] = [_manifest_payload(entry) for entry in notable]
    payload["staged_files"] = len(source.manifest)
    return payload


def _manifest_payload(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "path": str(entry.path),
        "sha256": entry.sha256,
        "staged": entry.staged,
        "live": entry.live,
        "staged_path": str(entry.staged_path) if entry.staged_path else None,
        "reason": entry.reason,
        "section": entry.section,
    }


def _run_item(case: ExperimentCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "circuit": case.circuit,
        "assignments": case.assignments,
        "status": case.status,
        "raw": str(case.raw_file) if case.raw_file else None,
        "log": str(case.log_file) if case.log_file else None,
    }


def _runs_page(cases: list[ExperimentCase], run_fields: list[str] | None = None) -> dict[str, Any]:
    page, total, offset, limit = paginate(cases, None, cap=_RUN_PAGE_LIMIT)
    pagination = pagination_metadata(total, offset, limit)
    rows = [_run_item(case) for case in page]
    if run_fields:
        plan = keep_plan(run_fields)
        rows = [project_row(row, plan) for row in rows]
    else:
        # Lean default: a produced row's artifact paths are provenance the
        # analysis tools resolve by id (fetch them via jobs(runs) or
        # run_fields). Every other status keeps them — failures entries
        # carry only {case_id, code, message}, so the failed row's log path
        # is its diagnostic.
        for row in rows:
            if row["status"] == "produced":
                del row["raw"], row["log"]
    # Same "o:<offset>" grammar jobs(action="runs") decodes; the shared
    # receipt assembly reads this key to build the continuation hint.
    # Nullable-key-always-present is the ruled cursor convention: readers may
    # do an unconditional ``page["next_cursor"]`` — the omit form is what
    # produced the KeyError fixed in 2fa1bc6.
    data: dict[str, Any] = {
        "items": rows,
        "total": pagination["total"],
        "returned": len(page),
        "truncated": pagination["has_more"],
        "next_cursor": f"o:{offset + len(page)}" if pagination["has_more"] else None,
    }
    return data


def _terminal_outcome(job: ExperimentJob) -> str:
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES:
        return "in_progress"
    if job.status == "failed":
        return "failed"
    if job.status == "cancelled":
        # A cancelled experiment never delivered what it promised, whatever the
        # counters say: a cancel landing after every run but before the
        # analysis leaves them fully reconciled, and one landing before
        # expansion leaves them all at zero.
        return "partial"
    return "partial" if job.completeness.fell_short else "complete"


def _terminal_hint(job: ExperimentJob, truncated: bool) -> str:
    if truncated:
        return (
            f"The inline run page is truncated; use jobs(runs) with job_id "
            f"{job.job_id} for the remaining cases."
        )
    if job.failures:
        return "Inspect failures and lint findings before retrying omitted cases."
    if job.analysis.status in {"failed", "cancelled"}:
        return (
            f"All declared experiment cases reached terminality, but the attached "
            f"analysis {job.analysis.status}; read analysis.error and re-run it with "
            f"analyze_results over job_id {job.job_id}."
        )
    return "All declared experiment cases reached terminality."


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
        return "path_denied", str(exc)
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


def _routing_failure_response(
    args: RunExperimentsInput,
    circuits: list[CircuitDeck],
    exc: DeckStagingError,
    projected: int,
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
            "outcome": "partial",
            "completeness": asdict(completeness),
            "lint": [{"circuit": circuit.circuit_id, "findings": []} for circuit in circuits],
            "runs": _runs_page(cases),
            "failures": failures,
            "hint": (
                "Configure an available Windows-native temp directory before "
                "retrying WSL LTspice experiments."
            ),
        }
    )
    return format_response(str(exc), data)


def _error_response(
    request_id: str,
    *,
    code: str,
    message: str,
    stage: str,
    retryable: bool,
    commit_state: Literal["not_started", "committed", "unknown"],
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
    result = format_response(message, data)
    result.isError = True
    return result


def _post_submit_error_response(
    receipt: ExperimentReceipt,
    exc: Exception,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None,
) -> types.CallToolResult:
    """Envelope for a failure that escaped AFTER the cases were submitted.

    Submission is the irreversible step: once the receipt exists the simulator
    fleet is running, and the job_id plus its control_token are the only handles
    that reach it. Reporting ``not_started`` here — or letting the exception out,
    which returns no structuredContent at all — strands running cases with no way
    to poll or cancel them. That orphaned fleet is precisely what commit_state
    exists to prevent, so a post-submit escape is always reported as committed.
    """
    job = receipt.job
    try:
        data = _job_payload(job, receipt.control_token, lint_by_circuit=lint_by_circuit)
    except Exception:
        # Even the receipt builder failed. Fall back to the minimum that keeps
        # the running job reachable rather than losing the handles with it.
        data = _empty_payload(job.request_id)
        data["job_id"] = job.job_id
        data["status"] = job.status
        data["outcome"] = "in_progress"
        data["control_token"] = receipt.control_token
    data["hint"] = (
        f"The experiment was submitted and is running. Use jobs(status) with job_id "
        f"{job.job_id} to follow it, or jobs(cancel) with that job_id and its "
        f"control_token to stop it."
    )
    data["error"] = {
        "code": getattr(exc, "code", "receipt_failed"),
        "message": str(exc),
        "stage": "receipt",
        "retryable": True,
        "commit_state": "committed",
    }
    result = format_response(
        f"Experiment {job.job_id} was submitted, but building its receipt failed: {exc}. "
        f"The cases ARE running.",
        data,
    )
    result.isError = True
    return result


def _empty_payload(request_id: str) -> dict[str, Any]:
    return {
        "job_id": None,
        "request_id": request_id,
        "status": "failed",
        "outcome": "failed",
        "source": [],
        "completeness": asdict(Completeness()),
        "lint": [],
        "runs": _runs_page([]),
        "failures": [],
        "observations": [],
        "warnings": [],
        "artifacts": [],
        "hint": "",
    }


# ---------------------------------------------------------------------------
# Consolidated jobs control plane
# ---------------------------------------------------------------------------


_JOBS_PAGE_LIMIT = 50
_FOREIGN_WAIT_POLL_S = 2.0
_ADDRESSED_JOB_ACTIONS = frozenset({"status", "wait", "cancel", "runs"})

Job = SimulationJob | BatchJob | ExperimentJob


class JobsInput(ToolInput):
    """Action-specific inputs for the consolidated jobs control plane."""

    action: Literal["status", "wait", "cancel", "list", "runs"] = Field(
        description=(
            "'status' snapshots a job now; 'wait' blocks until it finishes or "
            "timeout_s elapses; 'cancel' stops it; 'list' pages recent circuits and "
            "their job counts; 'runs' pages one job's per-run records. Each action "
            "accepts only its own fields and rejects the rest, so send exactly what "
            "the action takes."
        ),
    )
    job_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Address the job directly. Required by status/wait/cancel/runs unless "
            "request_id is given instead; never both, and never with 'list'."
        ),
    )
    request_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Address the job by the idempotency key it was submitted under — the "
            "way back to a job whose id was lost. Alternative to job_id."
        ),
    )
    timeout_s: float = Field(
        default=60.0,
        ge=0.0,
        le=300.0,
        description=(
            "'wait' only: how long to block, 0-300s. Timing out is not a failure — "
            "the response comes back with timed_out set and the job keeps running, "
            "so wait again. Polling with 'status' in a loop costs calls this avoids."
        ),
    )
    wait_for: Literal["all", "runs"] = Field(
        default="all",
        description=(
            "'wait' only: 'all' waits for the runs AND any attached analysis stage; "
            "'runs' returns as soon as the last run is terminal, before the "
            "analysis it would then have to wait for separately."
        ),
    )
    control_token: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "'cancel' only: the token from the original run_experiments receipt. "
            "Needed only when this process did not submit the job — the owning "
            "process may always cancel its own. Status and list never disclose it."
        ),
    )
    circuit: str | None = Field(
        default=None,
        description=(
            "'list' only: restrict to jobs of this circuit file. Omitted, 'list' is "
            "the recently-touched-circuits view — the way to find work from an "
            "earlier session."
        ),
    )
    limit: int = Field(
        default=_JOBS_PAGE_LIMIT,
        ge=1,
        le=_JOBS_PAGE_LIMIT,
        description="'list' only: circuit groups per page.",
    )
    cursor: str | None = Field(
        default=None,
        description=(
            "'list'/'runs': next_cursor from the previous page. Absent means the first page."
        ),
    )

    @model_validator(mode="after")
    def validate_action_fields(self) -> Self:
        """Require one selector and reject fields that do not belong to an action."""
        selected = int(self.job_id is not None) + int(self.request_id is not None)
        if self.action in _ADDRESSED_JOB_ACTIONS and selected != 1:
            raise ValueError(
                f"jobs action {self.action!r} requires exactly one of job_id or request_id"
            )
        if self.action == "list" and selected:
            raise ValueError("jobs action 'list' does not accept job_id or request_id")

        allowed_fields = {
            "status": {"action", "job_id", "request_id"},
            "wait": {
                "action",
                "job_id",
                "request_id",
                "timeout_s",
                "wait_for",
            },
            "cancel": {
                "action",
                "job_id",
                "request_id",
                "control_token",
            },
            "list": {"action", "circuit", "limit", "cursor"},
            "runs": {"action", "job_id", "request_id", "cursor"},
        }[self.action]
        unexpected = self.model_fields_set - allowed_fields
        if unexpected:
            raise ValueError(
                f"jobs action {self.action!r} does not accept: {', '.join(sorted(unexpected))}"
            )
        return self


_JOBS_ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "message": {"type": "string"},
        "stage": {"type": "string"},
        "retryable": {"type": "boolean"},
        "commit_state": {
            "type": "string",
            "enum": ["not_started", "committed", "unknown"],
        },
        "item_id": {"type": "string"},
    },
    "required": ["code", "message", "stage", "retryable", "commit_state"],
}

_JOBS_COMMON_PROPERTIES: dict[str, Any] = {
    "outcome": {
        "type": "string",
        "enum": ["complete", "partial", "failed", "in_progress"],
    },
    "observations": {"type": "array", "items": _OBSERVATION_SCHEMA},
    "warnings": {"type": "array", "items": {"type": "string"}},
    "failures": {"type": "array", "items": _FAILURE_SCHEMA},
    "hint": {"type": "string"},
    "error": _JOBS_ERROR_SCHEMA,
}

_JOBS_COMMON_REQUIRED = [
    "action",
    "outcome",
    "observations",
    "warnings",
    "failures",
    "hint",
]

_JOBS_PAGE_PROPERTIES: dict[str, Any] = {
    "items": {"type": "array"},
    "total": {"type": "integer"},
    "returned": {"type": "integer"},
    "truncated": {"type": "boolean"},
    "next_cursor": {"type": ["string", "null"]},
}

_JOBS_RECEIPT_PROPERTIES: dict[str, Any] = {
    "job_id": {"type": ["string", "null"]},
    "request_id": {"type": ["string", "null"]},
    "job_type": {"type": "string"},
    "status": {"type": "string"},
    "analysis_status": {"type": "string"},
    "dialect": {"type": ["string", "null"]},
    "source": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["source"],
    "completeness": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["completeness"],
    "lint": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["lint"],
    "runs": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["runs"],
    "analysis": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["analysis"],
    "artifacts": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["artifacts"],
}

_JOBS_RECEIPT_REQUIRED = [
    "job_id",
    "request_id",
    "job_type",
    "status",
    "analysis_status",
    "dialect",
    "source",
    "completeness",
    "lint",
    "runs",
    "artifacts",
]

_KILL_RECEIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "run_index": {"type": "integer"},
        "prior_status": {"type": "string"},
        "status": {"type": "string"},
    },
    "required": ["case_id", "run_index", "prior_status", "status"],
}

_CIRCUIT_GROUP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "exists": {"type": "boolean"},
        "last_activity": {"type": ["string", "null"]},
        "status_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer"},
        },
        "interrupted_job_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "path",
        "exists",
        "last_activity",
        "status_counts",
        "interrupted_job_ids",
    ],
}

_RUN_RECORD_SCHEMA = RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["runs"]["properties"]["items"][
    "items"
]


def _jobs_receipt_schema(action: Literal["status", "wait"]) -> dict[str, Any]:
    properties = {
        "action": {"const": action},
        **_JOBS_COMMON_PROPERTIES,
        **_JOBS_RECEIPT_PROPERTIES,
    }
    required = [*_JOBS_COMMON_REQUIRED, *_JOBS_RECEIPT_REQUIRED]
    if action == "wait":
        properties["timed_out"] = {"type": "boolean"}
        required.append("timed_out")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _jobs_page_schema(
    action: Literal["cancel", "list", "runs"],
    item_schema: dict[str, Any],
    *,
    addressed: bool,
) -> dict[str, Any]:
    properties = {
        "action": {"const": action},
        **_JOBS_COMMON_PROPERTIES,
        **_JOBS_PAGE_PROPERTIES,
    }
    properties["items"] = {"type": "array", "items": item_schema}
    required = [*_JOBS_COMMON_REQUIRED, "items", "total", "returned", "truncated", "next_cursor"]
    if addressed:
        properties.update(
            {
                "job_id": {"type": ["string", "null"]},
                "request_id": {"type": ["string", "null"]},
                "status": {"type": "string"},
            }
        )
        required.extend(["job_id", "request_id", "status"])
    if action == "runs":
        properties["dialect"] = {"type": ["string", "null"]}
        required.append("dialect")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


JOBS_OUTPUT_SCHEMA: dict[str, Any] = {
    # MCP requires outputSchema to be an object schema at the top level;
    # Claude Code rejects the whole tools/list response when it is not.
    "type": "object",
    # Every branch below declares these and requires most of them, so hoisting
    # them constrains nothing new. What it buys is the introspecting client that
    # reads `properties` and never looks at `oneOf`: it sees the shape all five
    # actions share instead of the lone `warnings` key the registry injects into
    # a schema that declares no properties of its own.
    "properties": {"action": {"type": "string"}, **_JOBS_COMMON_PROPERTIES},
    "discriminator": {"propertyName": "action"},
    "oneOf": [
        _jobs_receipt_schema("status"),
        _jobs_receipt_schema("wait"),
        _jobs_page_schema("cancel", _KILL_RECEIPT_SCHEMA, addressed=True),
        _jobs_page_schema("list", _CIRCUIT_GROUP_SCHEMA, addressed=False),
        _jobs_page_schema("runs", _RUN_RECORD_SCHEMA, addressed=True),
    ],
}


@dataclass(frozen=True)
class _CircuitGroupsRead:
    groups: list[dict[str, Any]]
    observations: list[dict[str, Any]]


class _JobsActionError(Exception):
    """Call-level jobs error with a stable machine-readable code."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        stage: str,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.retryable = retryable


def _decode_jobs_cursor(cursor: str | None) -> int:
    if cursor is None:
        return 0
    prefix, separator, raw_offset = cursor.partition(":")
    if separator != ":" or prefix != "o" or not raw_offset.isdecimal():
        raise _JobsActionError(
            "invalid_cursor",
            "Invalid jobs cursor; use the opaque next_cursor returned by the prior page",
            stage="pagination",
        )
    return int(raw_offset)


def _jobs_page(
    items: list[dict[str, Any]],
    *,
    cursor: str | None,
    limit: int,
) -> dict[str, Any]:
    offset = min(_decode_jobs_cursor(cursor), len(items))
    page = items[offset : offset + limit]
    truncated = offset + len(page) < len(items)
    # Nullable-key-always-present, the ruled cursor convention: readers may do
    # an unconditional ``page["next_cursor"]`` — the omit form is what produced
    # the KeyError fixed in 2fa1bc6.
    data: dict[str, Any] = {
        "items": page,
        "total": len(items),
        "returned": len(page),
        "truncated": truncated,
        "next_cursor": f"o:{offset + len(page)}" if truncated else None,
    }
    return data


def _jobs_unpaged(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "items": items,
        "total": len(items),
        "returned": len(items),
        "truncated": False,
        "next_cursor": None,
    }


def _without_control_tokens(value: Any) -> Any:
    """Recursively remove cancel authority from every jobs response channel."""
    if isinstance(value, dict):
        return {
            key: _without_control_tokens(item)
            for key, item in value.items()
            if key != "control_token"
        }
    if isinstance(value, list):
        return [_without_control_tokens(item) for item in value]
    return value


async def _resolve_jobs_target(args: JobsInput, state: SessionState) -> Job:
    job_id = args.job_id
    if job_id is None:
        assert args.request_id is not None
        index = await asyncio.to_thread(
            experiment_store.load_request_index,
            args.request_id,
            state.working_dir,
        )
        if index is None:
            raise JobNotFoundError(
                f"No experiment job is indexed for request_id {args.request_id!r}"
            )
        raw_job_id = index.get("job_id")
        if not isinstance(raw_job_id, str):
            raise JobNotFoundError(
                f"The request index for {args.request_id!r} does not name a valid job"
            )
        job_id = raw_job_id
    return await services.resolve_job_async(job_id, state)


def _job_type_name(job: Job) -> str:
    if isinstance(job, ExperimentJob):
        return "experiment"
    if isinstance(job, BatchJob):
        return job.job_type
    return "single"


def _analysis_status(job: Job) -> str:
    return job.analysis.status if isinstance(job, ExperimentJob) else "not_requested"


def _jobs_outcome(job: Job) -> Literal["complete", "partial", "failed", "in_progress"]:
    if job.status in NON_TERMINAL_LIVE_STATUSES:
        return "in_progress"
    if job.status in {"failed", "timeout", "interrupted"}:
        return "failed"
    if job.status == "completed_with_failures":
        # The coordinator marks an experiment completed_with_failures when its
        # attached analysis fails even though every run landed, so read an
        # experiment's outcome off its own run counters rather than its status.
        # Only this status: a cancelled experiment is partial no matter how the
        # counters read, including the cancel that lands before expansion and
        # leaves them all at zero.
        if isinstance(job, ExperimentJob):
            return "partial" if job.completeness.fell_short else "complete"
        return "partial"
    if job.status == "cancelled":
        return "partial"
    return "complete"


def _legacy_run_status(job: SimulationJob, raw_file: Path | None) -> str:
    if job.status == "completed":
        return "produced"
    if raw_file is not None and job.status in {"failed", "timeout", "interrupted"}:
        return "produced"
    return job.status


def _run_records(
    job: Job,
    state: SessionState,
    *,
    dialect: str | None,
) -> list[dict[str, Any]]:
    if isinstance(job, ExperimentJob):
        return [_run_item(case) for case in sorted(job.cases, key=lambda item: item.run_index)]

    records: list[dict[str, Any]] = []
    for run in services.runs_of(job):
        if run.raw_file is not None:
            state.raw_dialect_hints[run.raw_file] = dialect
        if isinstance(job, BatchJob):
            status = "produced" if run.raw_file is not None else "failed"
        else:
            status = _legacy_run_status(job, run.raw_file)
        records.append(
            {
                "case_id": f"{job.job_id}-case-{run.index:04d}",
                "run_index": run.index,
                "circuit": str(job.netlist),
                "assignments": dict(run.params),
                "status": status,
                "raw": str(run.raw_file) if run.raw_file is not None else None,
                "log": str(run.log_file) if run.log_file is not None else None,
            }
        )
    return records


def _legacy_completeness(
    job: SimulationJob | BatchJob,
    records: list[dict[str, Any]],
) -> dict[str, int]:
    expanded = job.total_runs if isinstance(job, BatchJob) else 1
    produced = sum(item["status"] == "produced" for item in records)
    if isinstance(job, BatchJob):
        failed = min(job.failed_runs, expanded - produced)
        submitted = min(job.completed_runs, expanded)
    else:
        failed = int(job.status in {"failed", "timeout", "interrupted"} and not produced)
        submitted = int(job.status != "queued")
    remaining = max(0, expanded - produced - failed)
    cancelled = remaining if job.status == "cancelled" else 0
    if job.status == "interrupted":
        failed += remaining
    return {
        "declared": expanded,
        "expanded": expanded,
        "submitted": submitted,
        "produced": produced,
        "failed": failed,
        "cancelled": cancelled,
        "skipped": 0,
    }


def _legacy_failures(
    job: SimulationJob | BatchJob,
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    failures = [
        {
            "case_id": item["case_id"],
            "code": "run_failed",
            "message": "The legacy run did not produce a raw result",
        }
        for item in records
        if item["status"] == "failed"
    ]
    if job.error and not failures:
        failures.append(
            {
                "case_id": f"{job.job_id}-case-0000",
                "code": job.status,
                "message": job.error,
            }
        )
    return failures


def _legacy_source(
    job: SimulationJob | BatchJob,
    *,
    dialect: str | None,
) -> dict[str, Any]:
    # A legacy job never staged anything, so it has no digest, no staged deck
    # and no manifest — it used to say so with empty strings and an empty list.
    # Omitting them says the same thing without spending the bytes, and matches
    # the experiment receipt, where those keys mean "provenance was requested".
    return {
        "circuit": job.netlist.stem,
        "path": str(job.netlist),
        "simulator": job.simulator,
        "dialect": dialect,
    }


def _receipt_snapshot(
    action: Literal["status", "wait"],
    job: Job,
    state: SessionState,
    *,
    timed_out: bool | None = None,
) -> dict[str, Any]:
    if isinstance(job, ExperimentJob):
        data = _job_payload(job, None)
        data["job_type"] = "experiment"
        data["dialect"] = services.dialect_for_job(job, state)
    else:
        dialect = services.dialect_for_job(job, state)
        records = _run_records(job, state, dialect=dialect)
        observations = list(job.observations or []) if isinstance(job, SimulationJob) else []
        data = {
            "job_id": job.job_id,
            "request_id": None,
            "job_type": _job_type_name(job),
            "status": job.status,
            "outcome": _jobs_outcome(job),
            "source": [_legacy_source(job, dialect=dialect)],
            "completeness": _legacy_completeness(job, records),
            "lint": [],
            "runs": _jobs_page(records, cursor=None, limit=_JOBS_PAGE_LIMIT),
            "failures": _legacy_failures(job, records),
            "observations": observations,
            "warnings": [],
            "artifacts": [],
            "hint": "",
            "dialect": dialect,
        }
    data["action"] = action
    data["analysis_status"] = _analysis_status(job)
    if timed_out is not None:
        data["timed_out"] = timed_out
    if job.status in NON_TERMINAL_LIVE_STATUSES:
        data["hint"] = (
            f"Job {job.job_id} is still {job.status}; continue with "
            f"jobs(action='wait', job_id='{job.job_id}')."
        )
    elif data["runs"]["truncated"]:
        data["hint"] = (
            f"Run records are paged; continue with jobs(action='runs', "
            f"job_id='{job.job_id}', cursor={data['runs']['next_cursor']!r})."
        )
    elif not data.get("hint"):
        data["hint"] = f"Job {job.job_id} is {job.status}."
    return data


def _runs_finished(job: Job, wait_for: Literal["all", "runs"]) -> bool:
    if isinstance(job, ExperimentJob) and wait_for == "runs":
        # Three ways to know, in cost order: the event this session set, the
        # status the lifecycle guarantees it for, then the cases themselves —
        # a job loaded from a peer's sidecar has no event of ours to read.
        return (
            job.runs_done_event.is_set()
            or runs_terminal(job.status)
            or all(case.status in TERMINAL_CASE_STATUSES for case in job.cases)
        )
    return job.done_event.is_set() or job.status in TERMINAL_STATUSES


async def _wait_for_jobs_target(
    job: Job,
    state: SessionState,
    *,
    timeout_s: float,
    wait_for: Literal["all", "runs"],
) -> tuple[Job, bool]:
    if _runs_finished(job, wait_for):
        return job, False

    if job.owner_pid == os.getpid():
        if isinstance(job, ExperimentJob):
            runner = state.runners.get_experiment_runner_for(job)
            if runner is None:
                return job, True
            await runner.wait(job, timeout_s, wait_for=wait_for)
            current = state.all_jobs.get(job.job_id, job)
            return current, not _runs_finished(current, wait_for)
        try:
            await asyncio.wait_for(job.done_event.wait(), timeout_s)
        except TimeoutError:
            return job, True
        current = state.all_jobs.get(job.job_id, job)
        return current, not _runs_finished(current, wait_for)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    current = job
    while True:
        current = await state.job_registry.refresh_foreign_job_async(current)
        if _runs_finished(current, wait_for):
            return current, False
        remaining = deadline - loop.time()
        if remaining <= 0:
            return current, True
        await asyncio.sleep(min(_FOREIGN_WAIT_POLL_S, remaining))


def _activity_timestamp(job: ExperimentJob) -> str:
    activity = job.completed_at or job.started_at
    return activity.isoformat()


def _collect_circuit_groups(
    state: SessionState,
    circuit: Path | None,
    own_experiments: dict[str, ExperimentJob],
) -> _CircuitGroupsRead:
    """Blocking recent-index, legacy-summary, and experiment-pointer join.

    ``own_experiments`` is a registry snapshot taken on the event loop —
    this process's live jobs are counted from it, not from possibly-lagging
    disk records.
    """
    recent_entries = recent.load(prune_missing=False)
    recent_by_path: dict[str, str | None] = {}
    for entry in recent_entries:
        raw_path = entry.get("path")
        if not isinstance(raw_path, str):
            continue
        try:
            resolved = str(Path(raw_path).resolve())
        except OSError:
            resolved = raw_path
        recent_by_path[resolved] = (
            entry.get("last_touched") if isinstance(entry.get("last_touched"), str) else None
        )

    if circuit is not None:
        candidates = [(circuit, recent_by_path.get(str(circuit)))]
    else:
        candidates = []
        seen: set[str] = set()
        for entry in recent_entries:
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            candidate = Path(raw_path)
            try:
                key = str(candidate.resolve())
            except OSError:
                key = raw_path
            if key in seen:
                continue
            seen.add(key)
            candidates.append((candidate, recent_by_path.get(key)))

    groups: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for circuit_path, last_touched in candidates:
        legacy = job_store.summarize_circuit(circuit_path)
        experiment_jobs, pointer_observations = experiment_store.load_pointer_jobs(
            circuit_path,
            state.working_dir,
            prefer=own_experiments,
        )
        try:
            resolved_circuit = circuit_path.resolve()
        except OSError:
            resolved_circuit = circuit_path
        experiment_jobs = [
            experiment
            for experiment in experiment_jobs
            if any(
                (source.path.resolve() if source.path.exists() else source.path)
                == resolved_circuit
                for source in experiment.sources
            )
        ]
        observations.extend(pointer_observations)
        counts = dict(legacy.get("status_counts") or {})
        interrupted = list(legacy.get("interrupted_job_ids") or [])
        activities = [last_touched] if last_touched is not None else []
        for experiment in experiment_jobs:
            counts[experiment.status] = counts.get(experiment.status, 0) + 1
            if experiment.status == "interrupted":
                interrupted.append(experiment.job_id)
            activities.append(_activity_timestamp(experiment))
        groups.append(
            {
                "path": str(circuit_path),
                "exists": bool(legacy.get("exists")),
                "last_activity": max(activities) if activities else None,
                "status_counts": counts,
                "interrupted_job_ids": sorted(set(interrupted)),
            }
        )
    return _CircuitGroupsRead(groups=groups, observations=observations)


def _merge_registry_observations(
    state: SessionState,
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    seen = {
        (item.get("code"), item.get("detail"))
        for item in state.job_registry.observations
        if isinstance(item, dict)
    }
    for observation in observations:
        key = (observation.get("code"), observation.get("detail"))
        if key not in seen:
            state.job_registry.observations.append(observation)
            seen.add(key)
    return [dict(item) for item in state.job_registry.observations if isinstance(item, dict)]


def _cancel_receipts_for_legacy(
    job: SimulationJob | BatchJob,
    prior_status: str,
    prior_run_indices: set[int],
) -> list[dict[str, Any]]:
    if isinstance(job, SimulationJob):
        return [
            {
                "case_id": f"{job.job_id}-case-0000",
                "run_index": 0,
                "prior_status": prior_status,
                "status": job.status,
            }
        ]
    return [
        {
            "case_id": f"{job.job_id}-case-{run_index:04d}",
            "run_index": run_index,
            "prior_status": "queued",
            "status": job.status,
        }
        for run_index in range(job.total_runs)
        if run_index not in prior_run_indices
    ]


_FOREIGN_CANCEL_ACK_WAIT_S = 10.0


async def _await_foreign_experiment_cancellation(
    job: ExperimentJob,
    state: SessionState,
) -> ExperimentJob:
    """Poll briefly for the owner to act on the durable cancellation barrier.

    The barrier itself is the contract's acknowledgement (no further case
    enters submission); this bounded wait only improves the receipt detail.
    On expiry the latest snapshot is returned — its non-terminal statuses
    are the honest answer.
    """
    current = job
    deadline = asyncio.get_running_loop().time() + _FOREIGN_CANCEL_ACK_WAIT_S
    while current.status not in TERMINAL_STATUSES:
        if asyncio.get_running_loop().time() >= deadline:
            break
        await asyncio.sleep(0.5)
        refreshed = await state.job_registry.refresh_foreign_job_async(current)
        if not isinstance(refreshed, ExperimentJob):
            raise _JobsActionError(
                "cancel_failed",
                f"Experiment job {job.job_id} changed kind while cancellation was pending",
                stage="cancellation",
            )
        current = refreshed
    return current


async def _cancel_jobs_target(
    job: Job,
    args: JobsInput,
    state: SessionState,
) -> list[dict[str, Any]]:
    if job.status in TERMINAL_STATUSES:
        return []

    if isinstance(job, ExperimentJob):
        if not experiment_store.cancel_authorized(job, args.control_token):
            raise _JobsActionError(
                "cancel_not_authorized",
                (
                    f"Cancellation is not authorized for experiment job {job.job_id}; "
                    "use the control token returned by its original submission or replay"
                ),
                stage="authorization",
            )
        runner = state.runners.get_experiment_runner_for(job)
        if runner is None:
            if job.owner_pid == os.getpid():
                raise _JobsActionError(
                    "cancel_unavailable",
                    (
                        f"Experiment job {job.job_id} belongs to this process, but its "
                        "coordinator is no longer live; cancellation was not acknowledged"
                    ),
                    stage="cancellation",
                    retryable=True,
                )
            prior = {
                case.case_id: (case.run_index, case.status)
                for case in job.cases
                if case.status not in TERMINAL_CASE_STATUSES
            }
            token = args.control_token
            if token is None:
                raise _JobsActionError(
                    "cancel_not_authorized",
                    f"Job {job.job_id} is owned by another live process; cancelling it "
                    "requires the control_token from its submission receipt",
                    stage="cancellation",
                )
            persisted = await asyncio.to_thread(
                experiment_store.request_cancellation,
                job.job_id,
                state.working_dir,
                token,
            )
            if persisted is None:
                raise JobNotFoundError(f"Job not found: {job.job_id}")
            finished = await _await_foreign_experiment_cancellation(job, state)
            final_by_case = {case.case_id: case.status for case in finished.cases}
            return [
                {
                    "case_id": case_id,
                    "run_index": run_index,
                    "prior_status": prior_status,
                    "status": final_by_case.get(case_id, "cancelled"),
                }
                for case_id, (run_index, prior_status) in prior.items()
            ]
        receipts = await runner.cancel(job, control_token=args.control_token)
        run_indices = {case.case_id: case.run_index for case in job.cases}
        return [
            {
                **receipt,
                "run_index": run_indices[receipt["case_id"]],
            }
            for receipt in receipts
        ]

    if job.owner_pid != os.getpid():
        raise _JobsActionError(
            "cancel_not_authorized",
            (
                f"Legacy job {job.job_id} is owned by another process and legacy jobs "
                "have no transferable control token; cancellation was not attempted"
            ),
            stage="authorization",
        )

    prior_status = job.status
    prior_run_indices = (
        {run.index for run in services.runs_of(job)} if isinstance(job, BatchJob) else set()
    )
    from ltspice_mcp.tools.simulation import CancelJobInput, handle_cancel_job

    await handle_cancel_job(CancelJobInput(job_id=job.job_id), state)
    return _cancel_receipts_for_legacy(job, prior_status, prior_run_indices)


def _jobs_error_payload(
    args: JobsInput,
    *,
    code: str,
    message: str,
    stage: str,
    retryable: bool,
) -> dict[str, Any]:
    common: dict[str, Any] = {
        "action": args.action,
        "outcome": "failed",
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": message,
        "error": {
            "code": code,
            "message": message,
            "stage": stage,
            "retryable": retryable,
            "commit_state": "not_started",
        },
    }
    if args.action in {"status", "wait"}:
        common.update(
            {
                "job_id": args.job_id,
                "request_id": args.request_id,
                "job_type": "unknown",
                "status": "unknown",
                "analysis_status": "not_requested",
                "dialect": None,
                "source": [],
                "completeness": asdict(Completeness()),
                "lint": [],
                "runs": _jobs_page([], cursor=None, limit=_JOBS_PAGE_LIMIT),
                "artifacts": [],
            }
        )
        if args.action == "wait":
            common["timed_out"] = False
        return common
    common.update(_jobs_unpaged([]))
    if args.action in {"cancel", "runs"}:
        common.update(
            {
                "job_id": args.job_id,
                "request_id": args.request_id,
                "status": "unknown",
            }
        )
    if args.action == "runs":
        common["dialect"] = None
    return common


def _jobs_error_details(exc: Exception) -> tuple[str, str, bool]:
    if isinstance(exc, _JobsActionError):
        return exc.code, exc.stage, exc.retryable
    if isinstance(exc, JobNotFoundError):
        return "job_not_found", "resolution", False
    if isinstance(exc, PathSecurityError):
        return "path_denied", "resolution", False
    if isinstance(exc, ExperimentCancellationError):
        code = "cancel_not_authorized" if "not authorized" in str(exc).lower() else "cancel_failed"
        return code, "cancellation", False
    if isinstance(exc, PermissionError):
        return "cancel_not_authorized", "authorization", False
    if isinstance(exc, LTSpiceMCPError):
        return "jobs_failed", "execution", False
    if isinstance(exc, (OSError, ValueError)):
        return "jobs_failed", "execution", True
    return "jobs_failed", "execution", False


@registry.tool(
    name="jobs",
    description=(
        "Check on, wait for, or stop a run you started, by job_id or the "
        "request_id it was submitted under. 'wait' blocks server-side until the "
        "job finishes — prefer it to polling 'status' in a loop; 'status' snapshots "
        "it now; 'cancel' stops it (owner process, or the receipt's control_token); "
        "'runs' pages the full per-run records, artifact paths included; 'list' "
        "with no circuit is the recently-touched-circuits view for picking up "
        "work from an earlier session."
    ),
    input_model=JobsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=True,
        openWorldHint=False,
    ),
    profiles=("consolidated",),
    output_schema=JOBS_OUTPUT_SCHEMA,
)
async def handle_jobs(args: JobsInput, state: SessionState) -> types.CallToolResult:
    """Execute one jobs control-plane action with an action-discriminated response."""
    is_error = False
    try:
        if args.action == "list":
            circuit = (
                await asyncio.to_thread(safe_path, args.circuit, state)
                if args.circuit is not None
                else None
            )
            own_experiments = {
                job_id: job
                for job_id, job in state.all_jobs.items()
                if isinstance(job, ExperimentJob)
            }
            loaded = await asyncio.to_thread(
                _collect_circuit_groups, state, circuit, own_experiments
            )
            observations = _merge_registry_observations(state, loaded.observations)
            data = {
                "action": "list",
                "outcome": "complete",
                **_jobs_page(loaded.groups, cursor=args.cursor, limit=args.limit),
                "observations": observations,
                "warnings": [],
                "failures": [],
                "hint": (
                    "Recent circuit groups are ordered by the recent-circuits index."
                    if circuit is None
                    else f"Persisted job summary for {circuit}."
                ),
            }
            text = f"Listed {data['returned']} of {data['total']} circuit group(s)"
        else:
            job = await _resolve_jobs_target(args, state)
            request_id = job.request_id if isinstance(job, ExperimentJob) else None
            if args.action == "status":
                data = _receipt_snapshot("status", job, state)
                text = f"Job {job.job_id}: {job.status}"
            elif args.action == "wait":
                job, timed_out = await _wait_for_jobs_target(
                    job,
                    state,
                    timeout_s=args.timeout_s,
                    wait_for=args.wait_for,
                )
                data = _receipt_snapshot(
                    "wait",
                    job,
                    state,
                    timed_out=timed_out,
                )
                text = (
                    f"Wait for job {job.job_id} timed out at status {job.status}"
                    if timed_out
                    else f"Job {job.job_id} reached {args.wait_for} terminality"
                )
            elif args.action == "runs":
                dialect = services.dialect_for_job(job, state)
                records = _run_records(job, state, dialect=dialect)
                page = _jobs_page(
                    records,
                    cursor=args.cursor,
                    limit=_JOBS_PAGE_LIMIT,
                )
                data = {
                    "action": "runs",
                    "outcome": _jobs_outcome(job),
                    "job_id": job.job_id,
                    "request_id": request_id,
                    "status": job.status,
                    "dialect": dialect,
                    **page,
                    "observations": [],
                    "warnings": [],
                    "failures": [],
                    "hint": (
                        "Use next_cursor to continue the run page."
                        if page["truncated"]
                        else f"Returned all recorded runs for job {job.job_id}."
                    ),
                }
                text = f"Returned {data['returned']} of {data['total']} run record(s)"
            else:
                receipts = await _cancel_jobs_target(job, args, state)
                job = state.all_jobs.get(job.job_id, job)
                data = {
                    "action": "cancel",
                    "outcome": "complete",
                    "job_id": job.job_id,
                    "request_id": request_id,
                    "status": job.status,
                    **_jobs_unpaged(receipts),
                    "observations": [],
                    "warnings": [],
                    "failures": [],
                    "hint": (
                        f"Job {job.job_id} was already terminal; no cancellation was needed."
                        if not receipts
                        else (
                            f"Cancellation of job {job.job_id} is acknowledged; no further "
                            "case can enter submission."
                        )
                    ),
                }
                text = f"Cancellation acknowledged for job {job.job_id}"
    except Exception as exc:
        code, stage, retryable = _jobs_error_details(exc)
        data = _jobs_error_payload(
            args,
            code=code,
            message=str(exc),
            stage=stage,
            retryable=retryable,
        )
        text = str(exc)
        is_error = True

    data = _without_control_tokens(data)
    result = format_response(text, data)
    result.isError = is_error
    return result
