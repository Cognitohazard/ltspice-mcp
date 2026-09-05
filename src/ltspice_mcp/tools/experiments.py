"""Consolidated durable experiment submission tool."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias, get_args

from mcp import types
from pydantic import (
    BeforeValidator,
    Field,
    SkipValidation,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_validator,
)

from ltspice_mcp.errors import (
    JobNotFoundError,
    LTSpiceMCPError,
    PathSecurityError,
    ResultError,
    SimulationError,
    compact_validation_error,
)
from ltspice_mcp.lib import experiment_store, job_store, recent, response_budget, services
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
from ltspice_mcp.lib.log_parser import diagnostic_collapse_key
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
    keep_plan,
    paginate,
    project_row,
    prune_unreferenced_defs,
    registry,
    resolve_response_budget,
    resolve_run_simulator,
    resolve_runnable_netlist,
    safe_path,
)
from ltspice_mcp.tools.analyze import (
    MAX_PAGE_SIZE,
    coerce_per_run_default,
    include_flag_coercer,
)

_RUN_PAGE_LIMIT = 50
SUBMISSION_DWELL_CAP_S = 120.0
JOBS_WAIT_CAP_S = 300.0
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
            "run — a discovery aid, not something to leave on."
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
        "One analyze_results recipe. This block mirrors analyze_results.recipes "
        "exactly — same grammar, same metrics, validated the same way at "
        "submission. Fields per metric: api.reference('analyze_results') or "
        "spice://guide."
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
    budget: int | None = Field(
        default=None,
        ge=response_budget.BUDGET_MIN_TOKENS,
        description=response_budget.BUDGET_DESCRIPTION,
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
        "case_ids": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer"},
        "code": {"type": "string"},
        "message": {"type": "string"},
        "evidence": {"type": "object"},
        "hint": {"type": "string"},
    },
    "required": ["case_id", "code", "message"],
}

# Recovery guidance per classified failure code, worded for the six tools this
# profile exposes. These are the surviving arms of three exception hints that
# were keyed by exception classes nothing ever raised, so no caller saw them;
# the case-level failure code is the signal that actually reaches a caller.
# They live beside the receipt that carries them because the failure channel
# exists only here — another profile growing one would need its own wording,
# not a share of this table.
_FAILURE_CODE_HINTS: dict[str, str] = {
    "convergence_failed": (
        "Add a .OPTIONS directive to the deck (e.g. .OPTIONS reltol=0.003 or "
        ".OPTIONS method=gear), or check component values for very large/small ratios."
    ),
    "singular_matrix": (
        "This usually means a floating node or short circuit. Use inspect with a "
        "net query to trace connectivity, or read the netlist directly."
    ),
    "missing_model": (
        'Use inspect with a model query (mode:"search") to fuzzy-match against '
        "loaded libraries, or add a .lib/.include for it to the deck."
    ),
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

_RUN_RECORD_SCHEMA: dict[str, Any] = {
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
    # run_fields may project away any key, so the shared object/columnar row
    # fragment deliberately requires none of them.
}

_RUNS_PAGE_SCHEMA: dict[str, Any] = response_budget.row_page_schema(
    {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": _RUN_RECORD_SCHEMA},
            "total": {"type": "integer"},
            "returned": {"type": "integer"},
            "truncated": {"type": "boolean"},
            "next_cursor": {"type": ["string", "null"]},
        },
        "required": ["items", "total", "returned", "truncated", "next_cursor"],
    },
    item_schema=_RUN_RECORD_SCHEMA,
)

_COMPLETENESS_SCHEMA: dict[str, Any] = {
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
}

# The derived view of ``completeness``, and only that: the seven raw counters
# live in ``completeness`` and are not restated here.
_PROGRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expanded": {"type": "integer"},
        "terminal": {"type": "integer"},
        "remaining": {"type": "integer"},
    },
    "required": ["expanded", "terminal", "remaining"],
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
        "completeness": _COMPLETENESS_SCHEMA,
        "progress": _PROGRESS_SCHEMA,
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
        "runs": _RUNS_PAGE_SCHEMA,
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
        "progress",
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
    profiles=("consolidated",),
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
            code="idempotency_conflict",
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
            code="path_denied",
            message=str(exc),
            stage="resolution",
            retryable=False,
            commit_state="not_started",
            budget=budget,
        )
    except (SimulationError, ResultError, DeckStagingError, OSError, ValueError) as exc:
        return await _error_response(
            args.request_id,
            code=getattr(exc, "code", "submission_failed"),
            message=str(exc),
            stage="submission",
            retryable=True,
            commit_state="not_started",
            budget=budget,
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


_ReceiptBuilt = tuple[dict[str, Any], str]
_ReceiptBuild = Callable[[int, response_budget.Rung | None], _ReceiptBuilt]
_ReceiptRows = Callable[[dict[str, Any]], list[Any]]

# Rung 0's allowlist, shared by run_experiments and jobs status/wait because
# both render the same receipt envelope.
_TRIM_REMOVE_RECEIPT: tuple[str, ...] = ("analysis",)
# `source` is deliberately absent, and no other rung empties it either. It is
# not an identity echo the way the analysis envelope's `source_hashes` is: with
# `provenance` off it still carries the staging disclosures — the live,
# unstaged or unexplained manifest entries `_source_payload` keeps precisely so
# a caller is told about them — and with `provenance` on it carries an opt-in
# the caller asked for, which the trim rung's charter forbids revoking. Facts
# under one flag and an opt-in under the other leaves no rung a claim on it.

_RUN_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced; no run, failure, or analysis fact was dropped.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or continue "
        "through the returned cursor/jobs route."
    ),
)


def _receipt_row_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every run page carried at the top level or under a receipt."""
    pages = [data] if isinstance(data.get("items"), list) else []
    runs = data.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("items"), list):
        pages.append(runs)
    return pages


def _jobs_rows(data: dict[str, Any]) -> list[Any]:
    return [row for page in _receipt_row_pages(data) for row in page["items"]]


def _run_receipt_rows(data: dict[str, Any]) -> list[Any]:
    rows = _jobs_rows(data)
    analysis_block = data.get("analysis")
    if isinstance(analysis_block, dict):
        result = analysis_block.get("result")
        if isinstance(result, dict):
            rows.extend(analyze.analysis_rows(result))
    return rows


def _degrade_receipt(data: dict[str, Any], rung: response_budget.Rung) -> None:
    """Apply presentation rungs to either public receipt envelope."""
    if rung.trim:
        response_budget.apply_trim(data, remove=_TRIM_REMOVE_RECEIPT)
    if rung.answer_channel:
        for page in _receipt_row_pages(data):
            for row in page["items"]:
                if isinstance(row, dict) and row.get("status") == "produced":
                    row.pop("raw", None)
                    row.pop("log", None)
    if rung.columnar:
        for page in _receipt_row_pages(data):
            response_budget.columnarize(page, "items")
        analysis_block = data.get("analysis")
        if isinstance(analysis_block, dict) and isinstance(analysis_block.get("result"), dict):
            analyze.columnarize_analysis_view(analysis_block["result"])


async def _negotiate_receipt(
    budget: ResponseBudget,
    build: _ReceiptBuild,
    page_limit: int,
    *,
    rows: _ReceiptRows,
    notes: response_budget.Notes,
) -> _ReceiptBuilt:
    """Render a receipt at the mildest shared budget rung that fits."""
    text = ""
    rendered: dict[str, Any] = {}
    built_for: tuple[int, bool, bool] | None = None

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        nonlocal text, rendered, built_for
        limit = page_limit
        if rung.shrink:
            limit = response_budget.RowMeasure.of(rows(rendered)).fit_limit(page_limit, rung)
        candidate = (limit, rung.answer_channel, rung.shrink)
        if built_for != candidate:
            rendered, text = build(limit, rung)
            built_for = candidate
        _degrade_receipt(rendered, rung)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(
        budget.tokens, render, notes, max_rung=budget.max_rung
    )
    response_budget.attach_notes(result, notes)
    return result.data, text


async def _render_run_receipt(
    budget: ResponseBudget,
    build: _ReceiptBuild,
    *,
    is_error: bool = False,
) -> types.CallToolResult:
    if budget.tokens is None:
        data, text = build(_RUN_PAGE_LIMIT, None)
    else:
        data, text = await _negotiate_receipt(
            budget,
            build,
            _RUN_PAGE_LIMIT,
            rows=_run_receipt_rows,
            notes=_RUN_BUDGET_NOTES,
        )
    result = format_response(text, data)
    result.isError = is_error
    return result


async def _render_static_run_receipt(
    data: dict[str, Any],
    text: str,
    budget: ResponseBudget,
    *,
    is_error: bool = False,
) -> types.CallToolResult:
    return await _render_run_receipt(
        budget,
        lambda _limit, _rung: (copy.deepcopy(data), text),
        is_error=is_error,
    )


def progress_from_completeness(completeness: Completeness) -> dict[str, int]:
    """Project durable accounting into the shared progress fact.

    The three DERIVED numbers only. ``completeness`` keeps all seven counters,
    unchanged and always — this block used to restate every one of them beside
    its own projection, so the same accounting arrived twice in one receipt and
    a third time in the hint. Dropping the copy removes no fact: each counter is
    one key away, in the ``completeness`` block itself.
    """
    return {
        "expanded": completeness.expanded,
        "terminal": completeness.terminal,
        "remaining": completeness.expanded - completeness.terminal,
    }


def finalize_receipt(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize completeness and attach its two public progress projections."""
    raw = data["completeness"]
    completeness = raw if isinstance(raw, Completeness) else Completeness(**raw)
    data["completeness"] = asdict(completeness)
    progress = progress_from_completeness(completeness)
    data["progress"] = progress
    response_budget.append_hint(
        data,
        f"Progress: {progress['terminal']}/{progress['expanded']} terminal; "
        f"{progress['remaining']} remaining.",
    )
    return data


@dataclass(frozen=True)
class ReceiptSnapshot:
    """One loop-atomic copy of every job-derived receipt fact.

    The coordinator mutates jobs only on the event loop.  Constructing this
    value is deliberately synchronous, and every mutable leaf is detached from
    the job before control can return to the loop.  Presentation can therefore
    page, project, and budget-negotiate repeatedly without observing a later
    job transition.
    """

    job_id: str
    request_id: str | None
    job_type: str
    status: str
    dialect: str | None
    control_token: str | None
    sources: tuple[SourceRecord | dict[str, Any], ...]
    lint: tuple[dict[str, Any], ...]
    runs_by_key: dict[tuple[str, int], dict[str, Any]]
    completeness: Completeness
    failures: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    analysis_status: str
    analysis_result: dict[str, Any] | None
    analysis_error: str | None
    analysis_observations: tuple[dict[str, Any], ...]
    analysis_request: dict[str, Any] | None

    @property
    def outcome(self) -> Literal["complete", "partial", "failed", "in_progress"]:
        """Receipt outcome derived only from copied status and completeness."""
        if self.job_type == "experiment":
            return _terminal_outcome(self)
        return _jobs_outcome(self)


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
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES and wait_s > 0:
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

    def build(limit: int, rung: response_budget.Rung | None) -> _ReceiptBuilt:
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

    return await _render_run_receipt(budget, build)


def render_receipt_snapshot(
    snapshot: ReceiptSnapshot,
    *,
    control_token: str | None = None,
    provenance: bool = False,
    run_fields: list[str] | None = None,
    runs_cap: int = _RUN_PAGE_LIMIT,
    analysis_fields: list[str] | None = None,
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
) -> dict[str, Any]:
    """Render the existing receipt envelope from detached neutral facts.

    The snapshot's leaves are already detached from the job, and rendering
    never writes through them — the only row edits build or own their dicts —
    so the envelope shares them rather than copying them per render.
    """
    runs = project_receipt_runs(
        snapshot,
        run_fields,
        lean_default=True,
        limit=runs_cap,
    )
    data: dict[str, Any] = {
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "outcome": snapshot.outcome,
        "source": [
            _source_payload(source, provenance=provenance)
            if isinstance(source, SourceRecord)
            else dict(source)
            for source in snapshot.sources
        ],
        "completeness": snapshot.completeness,
        "lint": list(snapshot.lint),
        "runs": runs,
        "failures": _render_failures(snapshot.failures),
        "observations": list(snapshot.observations),
        "warnings": [],
        "artifacts": list(snapshot.artifacts),
        "hint": (
            f"Experiment {snapshot.job_id} is still running; use jobs(wait) with this "
            "job_id to continue waiting."
            if snapshot.job_type == "experiment"
            and snapshot.status not in _TERMINAL_EXPERIMENT_STATUSES
            else _terminal_hint(snapshot, runs["truncated"])
        ),
    }
    # Cancel authority, only where a cancel can still do anything. A terminal
    # job has nothing left to stop, so the token there is bytes on every receipt
    # buying an action the lifecycle already refuses.
    emitted_control_token = control_token if control_token is not None else snapshot.control_token
    if emitted_control_token is not None and snapshot.status not in _TERMINAL_EXPERIMENT_STATUSES:
        data["control_token"] = emitted_control_token
    if snapshot.analysis_status != "not_requested":
        rendered_result: dict[str, Any] | None = None
        legacy_result = False
        if snapshot.analysis_result is not None:
            rendered_result, legacy_result = analyze.render_attached_analysis(
                snapshot.analysis_result,
                fields=analysis_fields,
                answer_channel=analysis_answer_channel,
                row_limit=analysis_rows_cap,
            )
        analysis_observations = list(snapshot.analysis_observations)
        if legacy_result:
            analysis_observations.append(
                {
                    "code": "legacy_analysis_result",
                    "kind": "provenance",
                    "detail": (
                        "This job predates neutral attached-analysis snapshots; its "
                        "stored public result was served without reinterpreting it."
                    ),
                }
            )
        data["analysis"] = {
            "status": snapshot.analysis_status,
            "result": rendered_result,
            "error": snapshot.analysis_error,
            "observations": analysis_observations,
        }
        if provenance:
            # The caller's own attached-analysis input, replayed back —
            # proof of what ran, not something to re-read every turn.
            data["analysis"]["request"] = copy.deepcopy(snapshot.analysis_request)
    return data


_FAILURE_CASE_ID_CAP = 10
"""How many case ids a collapsed failure row names before deferring to ``count``."""


def _render_failures(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Collapse repeated failures into one counted row and attach recovery hints.

    A case failure carries a ~20-line log excerpt in its message, and a sweep
    or Monte Carlo that fails for one reason fails that way in every case — a
    hundred cases is a hundred copies of the same kilobyte in a channel the
    budget ladder is forbidden to trim. Rows sharing a ``(code, message)`` key
    therefore become one row naming its cases, exactly as the log reader
    already collapses a repeated diagnostic within one log.

    The key runs through :func:`diagnostic_collapse_key` because the excerpt
    ends in the case's own numeric state, so cases that failed for one reason
    are byte-identical only when they are also numerically identical — which in
    a Monte Carlo they never are. The emitted row is one member verbatim.

    Which member is decided by case id, not by which case happened to finish
    first: cases run in parallel, so completion order varies between runs, and
    a receipt whose excerpt and named cases change when nothing else did is not
    one two runs can be compared with.

    No fact is dropped: ``count`` is the true number of cases, so a capped
    ``case_ids`` list reports its own shortfall rather than rounding it away.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("code", "")), diagnostic_collapse_key(str(row.get("message", ""))))
        grouped.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for (code, _message), members in grouped.items():
        group = sorted(members, key=lambda item: str(item.get("case_id", "")))
        rendered = dict(group[0])
        if len(group) > 1:
            rendered["case_ids"] = [
                str(item.get("case_id", "")) for item in group[:_FAILURE_CASE_ID_CAP]
            ]
            rendered["count"] = len(group)
        hint = _FAILURE_CODE_HINTS.get(code)
        if hint is not None:
            rendered["hint"] = hint
        collapsed.append(rendered)
    return collapsed


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


def _project_run_rows(
    rows: list[dict[str, Any]],
    run_fields: list[str] | None,
    *,
    lean_default: bool,
) -> list[dict[str, Any]]:
    """Apply one request's run-row projection policy to rows the caller owns."""
    if run_fields:
        plan = keep_plan(run_fields)
        return [project_row(row, plan) for row in rows]
    if lean_default:
        # Lean default: a produced row's artifact paths are provenance the
        # analysis tools resolve by id (fetch them via jobs(runs) or
        # run_fields). Every other status keeps them — failures entries
        # carry only {case_id, code, message}, so the failed row's log path
        # is its diagnostic.
        for row in rows:
            if row["status"] == "produced":
                del row["raw"], row["log"]
    return rows


def _runs_page(
    cases: list[ExperimentCase],
    run_fields: list[str] | None = None,
    *,
    cap: int = _RUN_PAGE_LIMIT,
) -> dict[str, Any]:
    page, total, offset, _limit = paginate(cases, None, cap=cap)
    rows = _project_run_rows(
        [_run_item(case) for case in page],
        run_fields,
        lean_default=True,
    )
    # Same "o:<offset>" grammar jobs(action="runs") decodes; the shared
    # receipt assembly reads this key to build the continuation hint.
    # The cursor key is always present and nullable, so readers may do an
    # unconditional ``page["next_cursor"]``. Omitting the key on the last page
    # instead makes those readers raise KeyError.
    data: dict[str, Any] = {
        "items": [],
        "total": total,
        "returned": 0,
        "truncated": False,
        "next_cursor": None,
    }
    next_offset = analyze.retotal_page(data, rows, offset)
    if data["truncated"]:
        data["next_cursor"] = f"o:{next_offset}"
    return data


def project_receipt_runs(
    snapshot: ReceiptSnapshot,
    run_fields: list[str] | None,
    *,
    lean_default: bool,
    cursor: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Purely apply one invocation's run projection to copied canonical rows.

    ``run_fields`` and ``lean_default`` together are the original request's
    projection policy.  Receipt requests use the lean default when no explicit
    fields were supplied; ``jobs(runs)`` requests use the full-row policy.
    ``limit=None`` renders the complete page shape for an in-process consumer.
    """
    rows = _project_run_rows(
        [dict(row) for row in snapshot.runs_by_key.values()],
        run_fields,
        lean_default=lean_default,
    )
    page_limit = max(1, len(rows)) if limit is None else limit
    return _jobs_page(rows, cursor=cursor, limit=page_limit)


def _terminal_outcome(
    snapshot: ReceiptSnapshot,
) -> Literal["complete", "partial", "failed", "in_progress"]:
    if snapshot.status not in _TERMINAL_EXPERIMENT_STATUSES:
        return "in_progress"
    if snapshot.status == "failed":
        return "failed"
    if snapshot.status == "cancelled":
        # A cancelled experiment never delivered what it promised, whatever the
        # counters say: a cancel landing after every run but before the
        # analysis leaves them fully reconciled, and one landing before
        # expansion leaves them all at zero.
        return "partial"
    return "partial" if snapshot.completeness.fell_short else "complete"


# Case count at which a terminal receipt starts pointing at the in-process
# door. Ten is past any spot-check and squarely in sweep/corner territory —
# the workload class where the per-call cost of going through the tool surface
# is large enough to be worth avoiding.
_API_POINTER_MIN_CASES = 10


def _terminal_hint(snapshot: ReceiptSnapshot, truncated: bool) -> str:
    """Every recovery route this receipt has, not the first one that matched.

    A server restart mid-run sets BOTH conditions: the abandoned cases become
    failures AND the attached analysis is marked failed. Under an exclusive
    ladder the failures branch won and the caller was never told that the runs
    that DID produce data are still analyzable by job_id — so the obvious move
    was to re-run an experiment whose results were sitting on disk.
    """
    if snapshot.job_type != "experiment":
        return ""
    if truncated:
        routes = [
            f"The inline run page is truncated; use jobs(runs) with job_id "
            f"{snapshot.job_id} for the remaining cases."
        ]
        return " ".join(routes + _api_pointer_route(snapshot))
    routes: list[str] = []
    if snapshot.analysis_status in {"failed", "cancelled"}:
        routes.append(
            f"The attached analysis {snapshot.analysis_status}; read analysis.error and "
            f"re-run it with analyze_results over job_id {snapshot.job_id} — the runs "
            "that produced data need no re-run."
        )
    if snapshot.failures:
        routes.append("Inspect failures and lint findings before retrying omitted cases.")
    if not routes:
        routes.append("All declared experiment cases reached terminality.")
    return " ".join(routes + _api_pointer_route(snapshot))


def _api_pointer_route(snapshot: ReceiptSnapshot) -> list[str]:
    """The second discovery surface for the in-process door (the first is the
    initialize instructions): it lands exactly on the caller who is iterating —
    a many-case receipt is the loop shape where per-call wire overhead
    compounds and the Python door pays for itself. Appended on EVERY terminal
    experiment route, the truncated one included: a receipt big enough to
    truncate is the biggest loop of all."""
    if snapshot.completeness.expanded < _API_POINTER_MIN_CASES:
        return []
    return [
        "Loop-shaped follow-ups run cheaper in-process: "
        "from ltspice_mcp.api import Api (same ops; api.reference() documents them)."
    ]


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
            "outcome": "partial",
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

    def build(limit: int, _rung: response_budget.Rung | None) -> _ReceiptBuilt:
        rendered = copy.deepcopy(data)
        rendered["runs"] = _runs_page(cases, args.run_fields, cap=limit)
        return rendered, str(exc)

    return await _render_run_receipt(budget, build)


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
            "outcome": "in_progress",
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
            "outcome": "in_progress",
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
        "code": getattr(exc, "code", "receipt_failed"),
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
    return {
        "job_id": None,
        "request_id": request_id,
        "status": "failed",
        "outcome": "failed",
        "source": [],
        "completeness": Completeness(),
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

Job = SimulationJob | BatchJob | ExperimentJob


class JobsInput(ToolInput):
    """The shared half of every jobs call, and the door the five actions enter by.

    ``JobsInput.model_validate({"action": ...})`` routes on ``action`` and
    returns that action's own model, so a caller may keep addressing this one
    name while each action validates against exactly its own fields. Construct
    an action model directly when the action is known statically.
    """

    action: str = Field(
        description=(
            "'status' snapshots a job now; 'wait' blocks until it finishes or "
            "timeout_s elapses; 'cancel' stops it; 'list' pages recent circuits and "
            "their job counts; 'runs' pages one job's per-run records. Each action "
            "takes only its own fields, listed under that action below."
        ),
    )
    budget: int | None = Field(
        default=None,
        ge=response_budget.BUDGET_MIN_TOKENS,
        description=response_budget.BUDGET_DESCRIPTION,
    )

    #: The five action models, in advertised order. Set below, once they exist.
    VARIANTS: ClassVar[tuple[type[JobsInput], ...]] = ()

    def __init__(self, /, **data: Any) -> None:
        # Validation routes through the union; construction cannot, because a
        # model validator that returned another action's model from __init__
        # would hand back a silently empty instance. Naming the actions here
        # is the difference between a clear error and that empty object.
        if type(self) is JobsInput:
            raise TypeError(
                "JobsInput is the union of the five jobs actions; validate with "
                "JobsInput.model_validate({'action': ...}) or construct the action "
                "model itself"
            )
        super().__init__(**data)

    @model_validator(mode="wrap")
    @classmethod
    def _route_on_action(
        cls,
        data: Any,
        handler: ValidatorFunctionWrapHandler,
    ) -> Any:
        """Validate the base name as whichever action the payload names."""
        if cls is JobsInput and isinstance(data, dict):
            return _JOBS_ADAPTER.validate_python(data)
        return handler(data)

    @classmethod
    def wire_input_schema(cls) -> dict[str, Any]:
        """Advertise the five actions as one discriminated union."""
        if cls is not JobsInput:
            return super().wire_input_schema()
        return jobs_input_schema()


class _AddressedJobsInput(JobsInput):
    """The actions that name one job: exactly one of job_id or request_id."""

    job_id: str | None = Field(
        default=None,
        min_length=1,
        description="Address the job directly. Give this or request_id, never both.",
    )
    request_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Address the job by the idempotency key it was submitted under — the "
            "way back to a job whose id was lost. Alternative to job_id."
        ),
    )

    @model_validator(mode="after")
    def _one_selector(self) -> Self:
        if int(self.job_id is not None) + int(self.request_id is not None) != 1:
            raise ValueError(
                f"jobs action {self.action!r} requires exactly one of job_id or request_id"
            )
        return self


class JobsStatusInput(_AddressedJobsInput):
    """Snapshot one job's receipt as it stands now, without blocking."""

    action: Literal["status"]  # pyright: ignore[reportIncompatibleVariableOverride]


class JobsWaitInput(_AddressedJobsInput):
    """Block server-side until one job finishes, then return its receipt."""

    action: Literal["wait"]  # pyright: ignore[reportIncompatibleVariableOverride]
    timeout_s: float = Field(
        default=60.0,
        ge=0.0,
        le=JOBS_WAIT_CAP_S,
        description=(
            "How long to block, 0-300s. Timing out is not a failure — the response "
            "comes back with timed_out set and the job keeps running, so wait again. "
            "Polling with 'status' in a loop costs calls this avoids."
        ),
    )
    wait_for: Literal["all", "runs"] = Field(
        default="all",
        description=(
            "'all' waits for the runs AND any attached analysis stage; 'runs' "
            "returns as soon as the last run is terminal, before the analysis it "
            "would then have to wait for separately."
        ),
    )


class JobsCancelInput(_AddressedJobsInput):
    """Stop one job: no further case enters submission."""

    action: Literal["cancel"]  # pyright: ignore[reportIncompatibleVariableOverride]
    control_token: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "The token from the original run_experiments receipt. Needed only when "
            "this process did not submit the job — the owning process may always "
            "cancel its own. Status and list never disclose it."
        ),
    )


class JobsListInput(JobsInput):
    """Page the recently-touched circuits and the jobs recorded against them."""

    action: Literal["list"]  # pyright: ignore[reportIncompatibleVariableOverride]
    circuit: str | None = Field(
        default=None,
        description=(
            "Restrict to jobs of this circuit file. Omitted, 'list' is the "
            "recently-touched-circuits view — the way to find work from an "
            "earlier session."
        ),
    )
    limit: int = Field(
        default=_JOBS_PAGE_LIMIT,
        ge=1,
        le=_JOBS_PAGE_LIMIT,
        description="Circuit groups per page.",
    )
    cursor: str | None = Field(
        default=None,
        description="next_cursor from the previous page. Absent means the first page.",
    )


class JobsRunsInput(_AddressedJobsInput):
    """Page one job's per-run records, artifact paths included."""

    action: Literal["runs"]  # pyright: ignore[reportIncompatibleVariableOverride]
    cursor: str | None = Field(
        default=None,
        description="next_cursor from the previous page. Absent means the first page.",
    )


#: One jobs call: the action models, told apart by ``action``. Discriminated
#: rather than a plain union so an unknown action is one error naming the five
#: legal ones, and a known action with a bad field reports against that action
#: alone instead of five sets of complaints.
JobsAction: TypeAlias = Annotated[
    JobsStatusInput | JobsWaitInput | JobsCancelInput | JobsListInput | JobsRunsInput,
    Field(discriminator="action"),
]

_JOBS_ADAPTER: TypeAdapter[JobsAction] = TypeAdapter(JobsAction)
JobsInput.VARIANTS = get_args(get_args(JobsAction)[0])
JOBS_ACTIONS: tuple[str, ...] = tuple(
    get_args(model.model_fields["action"].annotation)[0] for model in JobsInput.VARIANTS
)


def jobs_input_schema() -> dict[str, Any]:
    """The five actions as one object schema, each action's shape its own branch.

    Three requirements shape this, and only one spelling meets all three.

    MCP requires an object schema at the top level — a bare ``oneOf`` makes a
    strict client reject the whole tool list — and a client that reads
    ``properties`` and stops there must still see what every action shares, so
    the two shared arguments are hoisted beside the branches exactly as
    ``JOBS_OUTPUT_SCHEMA`` hoists the shared response keys. Every branch
    declares them itself, so hoisting constrains nothing new.

    The branches are then applied through ``if``/``then`` on the discriminant
    rather than pydantic's ``oneOf``, because the server SDK validates
    arguments against this schema before the tool is dispatched and reports the
    single best error. Under ``oneOf`` that error is always the root
    "is not valid under any of the given schemas", which names neither the
    legal actions nor the offending field; under ``if``/``then`` only the
    matching action's constraints fail, so the caller is told that 'frobnicate'
    is not one of the five, or exactly which field this action does not take.
    The ``discriminator`` mapping is kept beside them: it is what says the
    branches are alternatives chosen by ``action``, and it is the reference a
    reader (or an OpenAPI-shaped client) follows into ``$defs``.
    """
    union = _JOBS_ADAPTER.json_schema(ref_template="#/$defs/{model}")
    shared = JobsInput.model_json_schema()["properties"]
    mapping: dict[str, str] = union["discriminator"]["mapping"]
    return {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(JOBS_ACTIONS),
                "description": shared["action"]["description"],
            },
            "budget": shared["budget"],
        },
        "required": ["action"],
        "discriminator": union["discriminator"],
        "allOf": [
            {"if": {"properties": {"action": {"const": action}}}, "then": {"$ref": ref}}
            for action, ref in mapping.items()
        ],
        "$defs": union["$defs"],
    }


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
    "items_columns": response_budget.COLUMNAR_ROWS_SCHEMA,
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
    "progress": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["progress"],
    "lint": RUN_EXPERIMENTS_OUTPUT_SCHEMA["properties"]["lint"],
    "runs": _RUNS_PAGE_SCHEMA,
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
    "progress",
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

# A circuit group is a discovery row, not a job listing: it names enough recent
# jobs to get back to one whose id was lost, and reports the true count so a
# caller can tell a short list from a complete one.
_RECENT_JOBS_CAP = 5

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
        "recent_jobs": {
            "type": "array",
            "description": (
                "The circuit's experiment jobs, newest activity first, capped at "
                f"{_RECENT_JOBS_CAP}. These job_ids are what jobs(status)/"
                "analyze_results address. Legacy .cir simulation runs are counted "
                "in status_counts but carry no id here."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "job_id": {"type": "string"},
                    "status": {"type": "string"},
                    "request_id": {"type": "string"},
                    "finished_at": {"type": ["string", "null"]},
                },
                "required": ["job_id", "status", "request_id", "finished_at"],
            },
        },
        "recent_jobs_total": {
            "type": "integer",
            "description": (
                "How many experiment jobs the circuit has, before the recent_jobs "
                "cap. Greater than len(recent_jobs) means older jobs exist that "
                "this page does not name."
            ),
        },
    },
    "required": [
        "path",
        "exists",
        "last_activity",
        "status_counts",
        "interrupted_job_ids",
        "recent_jobs",
        "recent_jobs_total",
    ],
}


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
    properties["items"] = response_budget.row_items_schema(item_schema)
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
    # The cursor key is always present and nullable, so readers may do an
    # unconditional ``page["next_cursor"]``. Omitting the key on the last page
    # instead makes those readers raise KeyError.
    data: dict[str, Any] = {
        "items": page,
        "total": len(items),
        "returned": len(page),
        "truncated": truncated,
        "next_cursor": f"o:{offset + len(page)}" if truncated else None,
    }
    return data


def unpaged_jobs_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the established jobs page shape with every item included."""
    return {
        "items": items,
        "total": len(items),
        "returned": len(items),
        "truncated": False,
        "next_cursor": None,
    }


# One jobs response, rendered at some page limit: the payload and its text line.
_JobsBuilt = _ReceiptBuilt
_JobsBuild = _ReceiptBuild


#: This tool's budget epilogue. No hint mirror: a jobs envelope's ``hint`` is the
#: control-plane's next step, and the ladder's own note reaches the caller on
#: ``observations`` without displacing it.
_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced, no run or receipt was dropped.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or page on with next_cursor."
    ),
)


async def _negotiate_jobs(
    budget: ResponseBudget,
    build: _JobsBuild,
    page_limit: int,
) -> _JobsBuilt:
    """Render this jobs response at the mildest ladder rung that fits ``budget``."""
    return await _negotiate_receipt(
        budget,
        build,
        page_limit,
        rows=_jobs_rows,
        notes=_BUDGET_NOTES,
    )


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


async def _resolve_jobs_target(args: _AddressedJobsInput, state: SessionState) -> Job:
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


def _jobs_outcome(
    snapshot: ReceiptSnapshot,
) -> Literal["complete", "partial", "failed", "in_progress"]:
    if snapshot.status in NON_TERMINAL_LIVE_STATUSES:
        return "in_progress"
    if snapshot.status in {"failed", "timeout", "interrupted"}:
        return "failed"
    if snapshot.status == "completed_with_failures":
        # The coordinator marks an experiment completed_with_failures when its
        # attached analysis fails even though every run landed, so read an
        # experiment's outcome off its own run counters rather than its status.
        # Only this status: a cancelled experiment is partial no matter how the
        # counters read, including the cancel that lands before expansion and
        # leaves them all at zero.
        if snapshot.job_type == "experiment":
            return "partial" if snapshot.completeness.fell_short else "complete"
        return "partial"
    if snapshot.status == "cancelled":
        return "partial"
    return "complete"


def _legacy_run_status(job: SimulationJob, raw_file: Path | None) -> str:
    if job.status == "completed":
        return "produced"
    if raw_file is not None and job.status in {"failed", "timeout", "interrupted"}:
        return "produced"
    return job.status


def _run_records(
    job: SimulationJob | BatchJob,
    state: SessionState,
    *,
    dialect: str | None,
) -> list[dict[str, Any]]:
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
) -> Completeness:
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
    if job.status in {"failed", "interrupted"}:
        failed += remaining
    return Completeness(
        declared=expanded,
        expanded=expanded,
        submitted=submitted,
        produced=produced,
        failed=failed,
        cancelled=cancelled,
    )


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


def snapshot_receipt(
    job: Job,
    state: SessionState | None,
    *,
    control_token: str | None = None,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
) -> ReceiptSnapshot:
    """Copy a job's complete receipt state without suspending the event loop.

    The first job read through the last mutable copy occur in this synchronous
    call.  Outcome and guidance are intentionally absent from that live-read
    interval; renderers derive them only from the returned detached value.
    """
    if isinstance(job, ExperimentJob):
        lint_map: dict[str, list[dict[str, Any]]]
        if lint_by_circuit is None:
            lint_map = {}
            for case in job.cases:
                lint_map.setdefault(case.circuit, [])
            for source in job.sources:
                lint_map[source.circuit] = source.lint_findings
        else:
            lint_map = lint_by_circuit

        observations = copy.deepcopy(job.observations)
        seen_observations = {(item.get("code"), item.get("detail")) for item in observations}
        runs_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for case in job.cases:
            row = copy.deepcopy(_run_item(case))
            runs_by_key[(case.case_id, case.run_index)] = row
            for observation in case.observations:
                copied = copy.deepcopy(observation)
                key = (copied.get("code"), copied.get("detail"))
                if key not in seen_observations:
                    observations.append(copied)
                    seen_observations.add(key)

        analysis = job.analysis
        return ReceiptSnapshot(
            job_id=job.job_id,
            request_id=job.request_id,
            job_type="experiment",
            status=job.status,
            dialect=services.dialect_for_job(job, state) if state is not None else None,
            control_token=control_token,
            sources=tuple(copy.deepcopy(job.sources)),
            lint=tuple(
                copy.deepcopy(
                    [
                        {"circuit": circuit, "findings": findings}
                        for circuit, findings in lint_map.items()
                        if findings
                    ]
                )
            ),
            runs_by_key=runs_by_key,
            completeness=copy.deepcopy(job.completeness),
            failures=tuple(copy.deepcopy(job.failures)),
            observations=tuple(observations),
            artifacts=tuple(copy.deepcopy(job.artifacts)),
            analysis_status=analysis.status,
            analysis_result=copy.deepcopy(analysis.result),
            analysis_error=analysis.error,
            analysis_observations=tuple(copy.deepcopy(analysis.observations)),
            analysis_request=copy.deepcopy(analysis.request),
        )

    if state is None:
        raise ValueError("A session state is required to snapshot a legacy job")
    dialect = services.dialect_for_job(job, state)
    records = _run_records(job, state, dialect=dialect)
    completeness = _legacy_completeness(job, records)
    observations = copy.deepcopy(job.observations or []) if isinstance(job, SimulationJob) else []
    return ReceiptSnapshot(
        job_id=job.job_id,
        request_id=None,
        job_type=_job_type_name(job),
        status=job.status,
        dialect=dialect,
        control_token=control_token,
        sources=(copy.deepcopy(_legacy_source(job, dialect=dialect)),),
        lint=(),
        runs_by_key={
            (str(record["case_id"]), int(record["run_index"])): copy.deepcopy(record)
            for record in records
        },
        completeness=completeness,
        failures=tuple(copy.deepcopy(_legacy_failures(job, records))),
        observations=tuple(observations),
        artifacts=(),
        analysis_status="not_requested",
        analysis_result=None,
        analysis_error=None,
        analysis_observations=(),
        analysis_request=None,
    )


def render_jobs_receipt_snapshot(
    action: Literal["status", "wait"],
    snapshot: ReceiptSnapshot,
    *,
    timed_out: bool | None = None,
    runs_cap: int = _JOBS_PAGE_LIMIT,
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
) -> dict[str, Any]:
    """Render one complete jobs receipt from detached snapshot facts."""
    data = render_receipt_snapshot(
        snapshot,
        runs_cap=runs_cap,
        analysis_answer_channel=analysis_answer_channel,
        analysis_rows_cap=analysis_rows_cap,
    )
    if snapshot.job_type == "experiment":
        data["job_type"] = snapshot.job_type
        data["dialect"] = snapshot.dialect
    else:
        # Preserve the legacy receipt's established key order as well as its
        # values; structured payloads are serialized in insertion order.
        data = {
            "job_id": data["job_id"],
            "request_id": data["request_id"],
            "job_type": snapshot.job_type,
            "status": data["status"],
            "outcome": data["outcome"],
            "source": data["source"],
            "completeness": data["completeness"],
            "lint": data["lint"],
            "runs": data["runs"],
            "failures": data["failures"],
            "observations": data["observations"],
            "warnings": data["warnings"],
            "artifacts": data["artifacts"],
            "hint": data["hint"],
            "dialect": snapshot.dialect,
        }
    data["action"] = action
    data["analysis_status"] = snapshot.analysis_status
    if timed_out is not None:
        data["timed_out"] = timed_out
    if snapshot.status in NON_TERMINAL_LIVE_STATUSES:
        data["hint"] = (
            f"Job {snapshot.job_id} is still {snapshot.status}; continue with "
            f"jobs(action='wait', job_id='{snapshot.job_id}')."
        )
    elif data["runs"]["truncated"]:
        data["hint"] = (
            f"Run records are paged; continue with jobs(action='runs', "
            f"job_id='{snapshot.job_id}', cursor={data['runs']['next_cursor']!r})."
        )
    elif not data.get("hint"):
        data["hint"] = f"Job {snapshot.job_id} is {snapshot.status}."
    return finalize_receipt(data)


def render_runs_envelope(
    snapshot: ReceiptSnapshot,
    *,
    cursor: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Render one jobs(runs) envelope over a snapshot's full run records.

    ``limit=None`` returns every recorded run, which is also what makes the
    truncation hint below collapse to the complete-page wording.
    """
    page = project_receipt_runs(
        snapshot,
        None,
        lean_default=False,
        cursor=cursor,
        limit=limit,
    )
    return {
        "action": "runs",
        "outcome": _jobs_outcome(snapshot),
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "dialect": snapshot.dialect,
        **page,
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": (
            "Use next_cursor to continue the run page."
            if page["truncated"]
            else f"Returned all recorded runs for job {snapshot.job_id}."
        ),
    }


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
        newest_first = sorted(experiment_jobs, key=_activity_timestamp, reverse=True)
        groups.append(
            {
                "path": str(circuit_path),
                "exists": bool(legacy.get("exists")),
                "last_activity": max(activities) if activities else None,
                "status_counts": counts,
                "interrupted_job_ids": sorted(set(interrupted)),
                "recent_jobs": [
                    {
                        "job_id": experiment.job_id,
                        "status": experiment.status,
                        "request_id": experiment.request_id,
                        "finished_at": (
                            experiment.completed_at.isoformat()
                            if experiment.completed_at is not None
                            else None
                        ),
                    }
                    for experiment in newest_first[:_RECENT_JOBS_CAP]
                ],
                "recent_jobs_total": len(experiment_jobs),
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
    args: JobsCancelInput,
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


def _jobs_error_payload(evaluation: JobsEvaluation) -> dict[str, Any]:
    """The failed-call envelope for whichever action was asked for."""
    error = evaluation.error
    assert error is not None
    args = evaluation.args
    addressed = args if isinstance(args, _AddressedJobsInput) else None
    common: dict[str, Any] = {
        "action": args.action,
        "outcome": "failed",
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": error.message,
        "error": {
            "code": error.code,
            "message": error.message,
            "stage": error.stage,
            "retryable": error.retryable,
            "commit_state": "not_started",
        },
    }
    if isinstance(args, (JobsStatusInput, JobsWaitInput)):
        common.update(
            {
                "job_id": args.job_id,
                "request_id": args.request_id,
                "job_type": "unknown",
                "status": "unknown",
                "analysis_status": "not_requested",
                "dialect": None,
                "source": [],
                "completeness": Completeness(),
                "lint": [],
                "runs": _jobs_page([], cursor=None, limit=_JOBS_PAGE_LIMIT),
                "artifacts": [],
            }
        )
        if isinstance(args, JobsWaitInput):
            common["timed_out"] = False
        return finalize_receipt(common)
    common.update(unpaged_jobs_items([]))
    if addressed is not None:
        common.update(
            {
                "job_id": addressed.job_id,
                "request_id": addressed.request_id,
                "status": "unknown",
            }
        )
    if isinstance(args, JobsRunsInput):
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


@dataclass(frozen=True)
class _JobsError:
    """A jobs action that did not complete, as the envelope will report it."""

    code: str
    message: str
    stage: str
    retryable: bool


@dataclass(frozen=True)
class JobsEvaluation:
    """One jobs action, executed and detached from the registry.

    Every jobs response is rendered from one of these and nothing else. The MCP
    page and the Python API's complete dict are two presentations of the SAME
    read, which is what keeps the two doors from disagreeing: reading the job
    again to render a second time would let a transition land between the reads
    and report two different jobs in one answer.
    """

    args: JobsInput
    snapshot: ReceiptSnapshot | None = None
    timed_out: bool | None = None
    kill_receipts: tuple[dict[str, Any], ...] = ()
    job_id: str | None = None
    request_id: str | None = None
    status: str | None = None
    groups: tuple[dict[str, Any], ...] = ()
    observations: tuple[dict[str, Any], ...] = ()
    circuit: Path | None = None
    error: _JobsError | None = None

    @property
    def is_error(self) -> bool:
        return self.error is not None


async def evaluate_jobs(args: JobsInput, state: SessionState) -> JobsEvaluation:
    """Execute one jobs action, reading the job exactly once.

    Returns facts only: no page, no hint, no budget. A failure is returned as
    an evaluation carrying its error rather than raised, so both doors report
    it through the same envelope.
    """
    try:
        if isinstance(args, JobsListInput):
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
            return JobsEvaluation(
                args=args,
                groups=tuple(loaded.groups),
                observations=tuple(_merge_registry_observations(state, loaded.observations)),
                circuit=circuit,
            )

        # Every action but 'list' addresses one job, so the union has no other
        # member left here.
        assert isinstance(args, _AddressedJobsInput)
        job = await _resolve_jobs_target(args, state)
        request_id = job.request_id if isinstance(job, ExperimentJob) else None

        if isinstance(args, JobsCancelInput):
            receipts = await _cancel_jobs_target(job, args, state)
            # Re-read after the cancel: the registry entry is what carries the
            # status the receipt reports.
            cancelled = state.all_jobs.get(job.job_id, job)
            return JobsEvaluation(
                args=args,
                kill_receipts=tuple(receipts),
                job_id=cancelled.job_id,
                request_id=request_id,
                status=cancelled.status,
            )

        timed_out: bool | None = None
        if isinstance(args, JobsWaitInput):
            job, timed_out = await _wait_for_jobs_target(
                job,
                state,
                timeout_s=args.timeout_s,
                wait_for=args.wait_for,
            )
        snapshot = snapshot_receipt(job, state)
        return JobsEvaluation(
            args=args,
            snapshot=snapshot,
            timed_out=timed_out,
            job_id=snapshot.job_id,
            request_id=snapshot.request_id,
            status=snapshot.status,
        )
    except Exception as exc:
        return _failed_jobs_evaluation(args, exc)


def _failed_jobs_evaluation(args: JobsInput, exc: Exception) -> JobsEvaluation:
    """Carry one failure as an evaluation, classified for the error envelope."""
    code, stage, retryable = _jobs_error_details(exc)
    return JobsEvaluation(
        args=args,
        error=_JobsError(code=code, message=str(exc), stage=stage, retryable=retryable),
    )


def render_jobs_data(
    evaluation: JobsEvaluation,
    *,
    limit: int | None = None,
    rung: response_budget.Rung | None = None,
) -> _JobsBuilt:
    """Present one evaluation as a response payload and its text line.

    ``limit`` is the presentation argument the two doors differ by: an integer
    is one MCP page of that size (the budget ladder re-renders at smaller ones),
    and ``None`` is the complete result the in-process door returns. Cancel
    receipts are never paged either way — they are the acknowledgement itself,
    not a page over a larger set.
    """
    args = evaluation.args
    if evaluation.error is not None:
        return _without_control_tokens(_jobs_error_payload(evaluation)), evaluation.error.message

    if isinstance(args, JobsListInput):
        groups = list(evaluation.groups)
        page = (
            unpaged_jobs_items(groups)
            if limit is None
            else _jobs_page(groups, cursor=args.cursor, limit=limit)
        )
        data = {
            "action": "list",
            "outcome": "complete",
            **page,
            "observations": list(evaluation.observations),
            "warnings": [],
            "failures": [],
            "hint": (
                (
                    "Recent circuit groups are ordered by the recent-circuits index."
                    if evaluation.circuit is None
                    else f"Persisted job summary for {evaluation.circuit}."
                )
                + " Each group's recent_jobs names the job_ids to address with "
                "jobs(status) or analyze_results; recent_jobs_total says how "
                "many were left out."
            ),
        }
        text = f"Listed {data['returned']} of {data['total']} circuit group(s)"
        return _without_control_tokens(data), text

    if isinstance(args, JobsCancelInput):
        receipts = list(evaluation.kill_receipts)
        data = {
            "action": "cancel",
            "outcome": "complete",
            "job_id": evaluation.job_id,
            "request_id": evaluation.request_id,
            "status": evaluation.status,
            **unpaged_jobs_items(receipts),
            "observations": [],
            "warnings": [],
            "failures": [],
            "hint": (
                f"Job {evaluation.job_id} was already terminal; no cancellation was needed."
                if not receipts
                else (
                    f"Cancellation of job {evaluation.job_id} is acknowledged; no "
                    "further case can enter submission."
                )
            ),
        }
        return _without_control_tokens(data), (
            f"Cancellation acknowledged for job {evaluation.job_id}"
        )

    snapshot = evaluation.snapshot
    assert snapshot is not None  # every remaining action snapshots its job

    if isinstance(args, JobsRunsInput):
        data = render_runs_envelope(snapshot, cursor=args.cursor, limit=limit)
        text = f"Returned {data['returned']} of {data['total']} run record(s)"
        return _without_control_tokens(data), text

    action: Literal["status", "wait"] = "wait" if isinstance(args, JobsWaitInput) else "status"
    data = render_jobs_receipt_snapshot(
        action,
        snapshot,
        timed_out=evaluation.timed_out,
        runs_cap=limit if limit is not None else max(1, len(snapshot.runs_by_key)),
        analysis_answer_channel=rung is not None and rung.answer_channel,
        analysis_rows_cap=limit if rung is not None and rung.shrink else None,
    )
    if action == "status":
        text = f"Job {snapshot.job_id}: {snapshot.status}"
    elif evaluation.timed_out:
        text = f"Wait for job {snapshot.job_id} timed out at status {snapshot.status}"
    else:
        assert isinstance(args, JobsWaitInput)
        text = f"Job {snapshot.job_id} reached {args.wait_for} terminality"
    return _without_control_tokens(data), text


def complete_jobs_data(evaluation: JobsEvaluation) -> dict[str, Any]:
    """The whole result of one evaluation: every run, group and receipt."""
    return render_jobs_data(evaluation)[0]


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
    evaluation = await evaluate_jobs(args, state)
    built: _JobsBuilt | None = None
    if evaluation.error is None:
        page_limit = args.limit if isinstance(args, JobsListInput) else _JOBS_PAGE_LIMIT
        try:
            budget = resolve_response_budget(args.budget, state)
            if budget.tokens is None:
                built = render_jobs_data(evaluation, limit=page_limit)
            else:
                # The presentation re-runs at a smaller page for each rung the
                # ladder tries; the control-plane work above ran once.
                built = await _negotiate_jobs(
                    budget,
                    lambda limit, rung: render_jobs_data(evaluation, limit=limit, rung=rung),
                    page_limit,
                )
        except Exception as exc:
            # Presentation failed over an action that already happened — a
            # cancel among them. Report it through this tool's own envelope
            # rather than letting it out as a protocol error carrying no
            # structuredContent at all.
            evaluation = _failed_jobs_evaluation(args, exc)
    if built is None:
        # A failed call renders once, off the budget ladder: its envelope is
        # already the irreducible floor, and the ladder's trim rung would take
        # the empty fact channels the output schema requires.
        built = render_jobs_data(evaluation)

    data, text = built
    result = format_response(text, data)
    result.isError = evaluation.is_error
    return result
