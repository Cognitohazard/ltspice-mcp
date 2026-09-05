"""The jobs control plane: status, wait, cancel, list, and run pages.

One tool over five actions, each with its own input model, so a caller
addresses ``jobs`` and only that action's fields are accepted. The evaluation
(``evaluate_jobs``) is separated from the presentation (``render_jobs_data``)
because MCP and the Python API differ only by that presentation argument: the wire
renders one page, the Python API renders the complete result. Neither can
report a job the other did not read.

The receipt this tool returns is the one ``run_experiments`` returns; its shape
and renderers live in ``tools/receipts``.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias, get_args

from mcp import types
from pydantic import (
    Field,
    TypeAdapter,
    ValidatorFunctionWrapHandler,
    model_validator,
)

from ltspice_mcp.errors import (
    JobNotFoundError,
    LTSpiceMCPError,
    PathSecurityError,
)
from ltspice_mcp.lib import experiment_store, job_store, recent, response_budget, services
from ltspice_mcp.lib.experiment_runner import ExperimentCancellationError
from ltspice_mcp.lib.experiment_types import (
    TERMINAL_CASE_STATUSES,
    Completeness,
    ExperimentJob,
)
from ltspice_mcp.lib.job_lifecycle import runs_terminal
from ltspice_mcp.lib.job_types import (
    TERMINAL_STATUSES,
    legacy_record_message,
)
from ltspice_mcp.lib.pagination import decode_offset, unpaged
from ltspice_mcp.lib.pagination import page as _page
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    HINT_SCHEMA,
    OUTCOME_SCHEMA,
    ResponseBudget,
    ToolInput,
    failures_schema,
    format_response,
    outcome_of,
    page_schema,
    registry,
    resolve_response_budget,
    safe_path,
)
from ltspice_mcp.tools.experiments import JOBS_WAIT_CAP_S
from ltspice_mcp.tools.receipts import (
    CASE_FAILURE_SCHEMA,
    JOBS_PAGE_LIMIT,
    OBSERVATION_SCHEMA,
    RUN_EXPERIMENTS_OUTPUT_SCHEMA,
    RUN_RECORD_SCHEMA,
    RUNS_PAGE_SCHEMA,
    Job,
    ReceiptBuild,
    ReceiptBuilt,
    ReceiptSnapshot,
    finalize_receipt,
    jobs_rows,
    negotiate_receipt,
    render_jobs_receipt_snapshot,
    render_runs_envelope,
    snapshot_receipt,
)

_FOREIGN_WAIT_POLL_S = 2.0


class JobsInput(ToolInput):
    """The shared half of every jobs call, and the entry point for the five actions.

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
            "Address the job by the idempotency key it was submitted under; use it "
            "when the job_id was lost. Alternative to job_id."
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
            "Polling with 'status' in a loop uses more calls."
        ),
    )
    wait_for: Literal["all", "runs"] = Field(
        default="all",
        description=(
            "'all' waits for the runs and any attached analysis stage; 'runs' "
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
        default=JOBS_PAGE_LIMIT,
        ge=1,
        le=JOBS_PAGE_LIMIT,
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
    "outcome": OUTCOME_SCHEMA,
    "observations": {"type": "array", "items": OBSERVATION_SCHEMA},
    "warnings": {"type": "array", "items": {"type": "string"}},
    "failures": failures_schema(CASE_FAILURE_SCHEMA),
    "hint": HINT_SCHEMA,
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
    "runs": RUNS_PAGE_SCHEMA,
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
    page = page_schema({"type": "array", "items": item_schema})
    properties = {
        "action": {"const": action},
        **_JOBS_COMMON_PROPERTIES,
        **page["properties"],
    }
    required = [*_JOBS_COMMON_REQUIRED, *page["required"]]
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
        _jobs_page_schema("runs", RUN_RECORD_SCHEMA, addressed=True),
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
    offset = decode_offset(cursor)
    if offset is None:
        raise _JobsActionError(
            "invalid_cursor",
            "Invalid jobs cursor; use the opaque next_cursor returned by the prior page",
            stage="pagination",
        )
    return offset


# One jobs response, rendered at some page limit: the payload and its text line.
_JobsBuilt = ReceiptBuilt
_JobsBuild = ReceiptBuild


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
    return await negotiate_receipt(
        budget,
        build,
        page_limit,
        rows=jobs_rows,
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


def _runs_finished(job: Job, wait_for: Literal["all", "runs"]) -> bool:
    if not isinstance(job, ExperimentJob):
        # A record an earlier release wrote is finished by definition: nothing
        # in this version could still be running it.
        return True
    if wait_for == "runs":
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

    if isinstance(job, ExperimentJob) and job.owner_pid == os.getpid():
        runner = state.runners.get_experiment_runner_for(job)
        if runner is None:
            return job, True
        await runner.wait(job, timeout_s, wait_for=wait_for)
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
        experiment_jobs, pointer_observations = experiment_store.load_jobs_for_circuit(
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

    # Unreachable in practice: a record an earlier release wrote always loads
    # terminal (job_store._effective_status), so the already-terminal branch
    # above answers it. Kept as a loud failure rather than a silent success in
    # case a future record shape reaches here still claiming to be live.
    raise _JobsActionError(
        "legacy_job_record",
        legacy_record_message(job.job_id),
        stage="cancellation",
    )


def _jobs_error_payload(evaluation: JobsEvaluation) -> dict[str, Any]:
    """The failed-call envelope for whichever action was asked for."""
    error = evaluation.error
    assert error is not None
    args = evaluation.args
    addressed = args if isinstance(args, _AddressedJobsInput) else None
    common: dict[str, Any] = {
        "action": args.action,
        # The action never ran, so nothing came back beside the error.
        "outcome": outcome_of(error, delivered=False),
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
                "runs": _page([], limit=JOBS_PAGE_LIMIT),
                "artifacts": [],
            }
        )
        if isinstance(args, JobsWaitInput):
            common["timed_out"] = False
        return finalize_receipt(common)
    common.update(unpaged([]))
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
        return exc.code, "resolution", False
    if isinstance(exc, PathSecurityError):
        return exc.code, "resolution", False
    if isinstance(exc, ExperimentCancellationError):
        return exc.code, "cancellation", False
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
    read, which is what keeps MCP and the Python API from disagreeing: reading the job
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
    an evaluation carrying its error rather than raised, so MCP and the Python API report
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

    ``limit`` is the presentation argument MCP and the Python API differ by: an integer
    is one MCP page of that size (the budget ladder re-renders at smaller ones),
    and ``None`` is the complete result the Python API returns. Cancel
    receipts are never paged either way — they are the acknowledgement itself,
    not a page over a larger set.
    """
    args = evaluation.args
    if evaluation.error is not None:
        return _without_control_tokens(_jobs_error_payload(evaluation)), evaluation.error.message

    if isinstance(args, JobsListInput):
        groups = list(evaluation.groups)
        page = (
            unpaged(groups)
            if limit is None
            else _page(groups, offset=_decode_jobs_cursor(args.cursor), limit=limit)
        )
        # A listing has no per-item failure channel to fill: every group it
        # found is reported, and a short page is a cursor, not a shortfall.
        list_failures: list[Any] = []
        data = {
            "action": "list",
            "outcome": outcome_of(list_failures),
            **page,
            "observations": list(evaluation.observations),
            "warnings": [],
            "failures": list_failures,
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
        # An acknowledged cancellation is the whole result; a job that was
        # already terminal needed none, which is not a shortfall either.
        cancel_failures: list[Any] = []
        data = {
            "action": "cancel",
            "outcome": outcome_of(cancel_failures),
            "job_id": evaluation.job_id,
            "request_id": evaluation.request_id,
            "status": evaluation.status,
            **unpaged(receipts),
            "observations": [],
            "warnings": [],
            "failures": cancel_failures,
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
        data = render_runs_envelope(snapshot, offset=_decode_jobs_cursor(args.cursor), limit=limit)
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
    output_schema=JOBS_OUTPUT_SCHEMA,
)
async def handle_jobs(args: JobsInput, state: SessionState) -> types.CallToolResult:
    """Execute one jobs control-plane action with an action-discriminated response."""
    evaluation = await evaluate_jobs(args, state)
    built: _JobsBuilt | None = None
    if evaluation.error is None:
        page_limit = args.limit if isinstance(args, JobsListInput) else JOBS_PAGE_LIMIT
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
