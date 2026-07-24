"""Consolidated durable experiment submission tool."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from mcp import types
from pydantic import Field

from ltspice_mcp.errors import PathSecurityError, SimulationError
from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.deck_staging import (
    DeckStagingError,
    resolve_experiment_paths,
    stage_deck,
    verify_staged_manifest,
)
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    ExperimentReceipt,
    ExperimentRunRequest,
    IdempotencyConflictError,
    canonical_fingerprint,
)
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.lint_rules import RULES_BY_ID, lint_deck, linter_version
from ltspice_mcp.lib.simulator import simulator_dialect
from ltspice_mcp.lib.sweep_utils import generate_id
from ltspice_mcp.lib.variations import (
    CircuitDeck,
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
from ltspice_mcp.tools._base import (
    StrictModel,
    ToolInput,
    format_response,
    paginate,
    pagination_metadata,
    registry,
    resolve_run_simulator,
    resolve_runnable_netlist,
    safe_path,
)

_RUN_PAGE_LIMIT = 50
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
    path: str
    id: str | None = None


class ExperimentExecution(StrictModel):
    wait_s: float = Field(default=60.0, ge=0.0, le=120.0)
    run_timeout_s: float | None = Field(default=None, gt=0.0)
    job_deadline_s: float | None = Field(default=None, gt=0.0)
    max_parallel: int | None = Field(default=None, ge=1)
    simulator: Literal["ltspice", "ngspice"] | None = None


class AnalysisPerRun(StrictModel):
    limit: int = Field(default=50, ge=1)
    cursor: str | None = None


class AnalysisInclude(StrictModel):
    per_run: AnalysisPerRun | None = None
    outliers: bool = False
    signals_available: bool = False


class AttachedAnalysis(StrictModel):
    recipes: list[dict[str, Any]]
    group_by: list[str] = Field(default_factory=list)
    include: AnalysisInclude | None = None


class RunExperimentsInput(ToolInput):
    request_id: str = Field(min_length=1)
    circuits: list[ExperimentCircuit] = Field(min_length=1)
    variations: list[Variation] = Field(default_factory=list)
    execution: ExperimentExecution = Field(default_factory=ExperimentExecution)
    analyze: AttachedAnalysis | None = None
    lint: Literal["block", "warn", "off"] = "block"
    suppress: list[str] = Field(default_factory=list)
    allow_live_includes: bool = False


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
                },
                "required": [
                    "circuit",
                    "path",
                    "sha256",
                    "staged_deck",
                    "manifest",
                    "linter_version",
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
                        "required": [
                            "case_id",
                            "run_index",
                            "circuit",
                            "assignments",
                            "status",
                            "raw",
                            "log",
                        ],
                    },
                },
                "total": {"type": "integer"},
                "returned": {"type": "integer"},
                "truncated": {"type": "boolean"},
                "next_cursor": {"type": "string"},
            },
            "required": ["items", "total", "returned", "truncated"],
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
            "required": [
                "status",
                "request",
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
        "Run one or more staged SPICE decks across strict assignment and random "
        "variations. The required request_id makes submission durable and "
        "idempotent; quick jobs return inline and longer jobs return a receipt."
    ),
    input_model=RunExperimentsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
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
            return await _dwell_and_respond(replay, args.execution.wait_s, state)

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
            analysis_callback=_analysis_not_available if analysis_request is not None else None,
        )
        receipt = await asyncio.shield(runner.submit(request))
        return await _dwell_and_respond(
            receipt,
            args.execution.wait_s,
            state,
            lint_by_circuit=lint_by_circuit,
        )
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
            allow_live_includes=args.allow_live_includes,
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
            sha256=staged.sha256,
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


async def _analysis_not_available(_job: ExperimentJob) -> dict[str, Any]:
    raise SimulationError("Attached analysis is not yet available in this build")


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
) -> types.CallToolResult:
    job = receipt.job
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES and wait_s > 0:
        runner = state.runners.get_experiment_runner_for(job)
        if runner is not None:
            await runner.wait(job, wait_s, wait_for="all")
        else:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(job.done_event.wait(), wait_s)
    data = _job_payload(job, receipt.control_token, lint_by_circuit=lint_by_circuit)
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
    control_token: str,
    *,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
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
    runs = _runs_page(job.cases)
    hint = _terminal_hint(job, runs["truncated"])
    data: dict[str, Any] = {
        "job_id": job.job_id,
        "request_id": job.request_id,
        "control_token": control_token,
        "status": job.status,
        "outcome": outcome,
        "source": [_source_payload(source) for source in job.sources],
        "completeness": asdict(job.completeness),
        "lint": [
            {"circuit": circuit, "findings": findings} for circuit, findings in lint_map.items()
        ],
        "runs": runs,
        "failures": list(job.failures),
        "observations": observations,
        "warnings": [],
        "artifacts": list(job.artifacts),
        "hint": hint,
    }
    if job.analysis.status != "not_requested":
        data["analysis"] = {
            "status": job.analysis.status,
            "request": job.analysis.request,
            "result": job.analysis.result,
            "error": job.analysis.error,
            "observations": job.analysis.observations,
        }
    return data


def _source_payload(source: SourceRecord) -> dict[str, Any]:
    return {
        "circuit": source.circuit,
        "path": str(source.path),
        "sha256": source.sha256,
        "staged_deck": str(source.staged_deck),
        "manifest": [_manifest_payload(entry) for entry in source.manifest],
        "linter_version": source.linter_version,
        "simulator": source.simulator,
        "dialect": source.dialect,
    }


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


def _runs_page(cases: list[ExperimentCase]) -> dict[str, Any]:
    page, total, offset, limit = paginate(cases, None, cap=_RUN_PAGE_LIMIT)
    pagination = pagination_metadata(total, offset, limit)
    return {
        "items": [_run_item(case) for case in page],
        "total": pagination["total"],
        "returned": len(page),
        "truncated": pagination["has_more"],
    }


def _terminal_outcome(job: ExperimentJob) -> str:
    if job.status not in _TERMINAL_EXPERIMENT_STATUSES:
        return "in_progress"
    if job.status == "failed":
        return "failed"
    if (
        job.completeness.failed
        or job.completeness.cancelled
        or job.completeness.skipped
        or job.analysis.status in {"failed", "cancelled"}
    ):
        return "partial"
    return "complete"


def _terminal_hint(job: ExperimentJob, truncated: bool) -> str:
    if truncated:
        return (
            f"The inline run page is truncated; use jobs(runs) with job_id "
            f"{job.job_id} for the remaining cases."
        )
    if job.failures:
        return "Inspect failures and lint findings before retrying omitted cases."
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
