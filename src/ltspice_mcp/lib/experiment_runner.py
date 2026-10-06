"""Durable coordinator for expanded experiment cases."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import secrets
import shutil
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.background import BackgroundTasks
from ltspice_mcp.lib.controlled_ngspice import (
    controlled_ngspice,
    verify_execution_policy,
    verify_seeded_driver,
)
from ltspice_mcp.lib.deck_staging import sha256_file, verify_staged_manifest
from ltspice_mcp.lib.experiment_inputs import (
    capture_produced_artifacts,
    verify_case_inputs,
    verify_startup,
)
from ltspice_mcp.lib.experiment_types import (
    ACTIVE_CASE_STATUSES,
    TERMINAL_CASE_STATUSES,
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
    failure_row,
)
from ltspice_mcp.lib.filelock import async_file_lock, file_lock
from ltspice_mcp.lib.job_lifecycle import (
    VALID_EXPERIMENT_TRANSITIONS,
    InvalidTransitionError,
    LiveJob,
    finished,
)
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES
from ltspice_mcp.lib.native_execution import prepare_native_cases
from ltspice_mcp.lib.ngspice_driver import validate_seed
from ltspice_mcp.lib.pdk_native import (
    ArtifactDigest,
    NativeCaseError,
    NativeLaunchPolicy,
    verify_launch,
)
from ltspice_mcp.lib.raw_parser import read_partial_raw_progress
from ltspice_mcp.lib.recovery_records import LaunchIntent, RecoveryError
from ltspice_mcp.lib.runner_base import (
    DEFAULT_MAX_PARALLEL,
    NativeLaunchContext,
    NativePrelaunchRefused,
    RunnerBase,
    RunOutcome,
    discard_generated_netlist,
    inject_logopinfo,
    inject_ngspice_control_write,
)
from ltspice_mcp.lib.simulator import dialect_for_simulator_name, is_ngspice
from ltspice_mcp.lib.simulator_build import (
    SimulatorExecutable,
    describe_executable,
    same_executable,
)
from ltspice_mcp.lib.store import Store, run_dir_in, run_filename_in, validate_job_id
from ltspice_mcp.lib.sweep_utils import generate_id

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_log import DecodedLog
    from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

CANONICALIZER_VERSION = experiment_store.CANONICALIZER_VERSION
DEFAULT_KILL_GRACE_S = 10.0

#: How far past a case's own run timeout and kill grace spicelib's subprocess
#: bound is set. The coordinator's timer is the one that decides a timeout, and
#: it starts a little after the process does, so spicelib's must not fire
#: first: a run it ends reports exit code -2 as an ordinary failure rather
#: than as ``run_timeout``. What it is for is the case the coordinator could
#: not finish: after a token-scoped kill nobody confirmed (a WSL interop query
#: alone may take 45s), it ends the process spicelib holds a handle to, so a
#: retained permit comes back through the late-exit path.
SPICELIB_TIMEOUT_MARGIN_S = 60.0

#: Scoped kills made for one stopped case, and the pause between them, all
#: inside its kill grace. One scan can miss: spicelib returns from a launch
#: before its worker thread has spawned the simulator, so a stop landing in
#: that gap finds no process yet.
KILL_MAX_PASSES = 5
KILL_RESCAN_INTERVAL_S = 0.5

RunTimeoutSource = Literal["request", "server_default"]
StopReason = Literal["cancelled", "job_deadline", "run_timeout"]

AnalysisCallback = Callable[[ExperimentJob], Awaitable[dict[str, Any]]]

#: Stages one submission's decks, inside the request gate. It returns the cases
#: and sources the job is built from, so a submission that turns out to be a
#: replay never runs it at all.
StageDecks = Callable[[], Awaitable["StagedDecks"]]

# How long a submission waits for the request gate. Longer than the file-lock
# default because the gate is now held across staging: a duplicate carrying the
# same request_id waits for the first submission's deck copies, and a large
# matrix takes longer to stage than an index write.
REQUEST_GATE_TIMEOUT_S = 300.0

# How each staging-time drift observation reads when it blocks a replay
# instead of annotating a fresh stage.
_DRIFT_REASONS = {
    "source_modified_after_staging": "content changed",
    "source_unavailable_after_staging": "no longer readable",
}


def cancel_receipt_row(
    case: ExperimentCase,
    prior_status: str,
    final_status: str,
) -> dict[str, Any]:
    """One row of a cancel receipt: which case, what it was doing, where it ended.

    Both cancel routes report the same four keys — this coordinator when it
    owns the job, and ``jobs(cancel)`` when the owner is another live process
    and the durable marker is all this one can write. One shape is what the
    receipt schema in ``tools/jobs.py`` describes.
    """
    return {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "prior_status": prior_status,
        "status": final_status,
    }


class IdempotencyConflictError(SimulationError):
    """A request id was reused for a different canonical payload."""

    code = "idempotency_conflict"


class RequestGateBusy(SimulationError):
    """Another submission held this request id's gate for the whole wait.

    Distinct from a plain lock timeout because of what it says about the
    durable state: the holder is the submission that claims this id, so
    whether a job now exists under it is exactly what this process could not
    find out.
    """

    code = "request_gate_busy"


class SubmissionCommitted(SimulationError):
    """The submission is durable, and something after the claim failed.

    The request index and the coordinator record are both on disk under this
    request_id before anything this wraps can go wrong. A caller told nothing
    started resubmits, typically under a fresh id, and the same experiment
    runs twice.
    """

    code = "submission_committed"

    def __init__(
        self,
        request_id: str,
        cause: BaseException,
        *,
        receipt: ExperimentReceipt | None = None,
    ) -> None:
        self.receipt = receipt
        super().__init__(
            f"The submission for request_id {request_id!r} is recorded, but the "
            f"call could not be completed: {cause}. Ask again with the same "
            "request_id to read back what was committed."
        )


class ExperimentCancellationError(SimulationError):
    """An experiment could not be cancelled by this coordinator."""

    code = "cancel_failed"


@contextlib.asynccontextmanager
async def request_gate(gate: Path, request_id: str) -> AsyncIterator[None]:
    """Hold one request id's gate, naming a wait that ran out for what it means.

    Only the acquisition is translated. A timeout raised by the work inside
    the gate is that work's own, and answering it with "another submission
    holds this id" would be a guess.
    """
    stack = contextlib.AsyncExitStack()
    try:
        await stack.enter_async_context(
            async_file_lock(gate, acquire_timeout=REQUEST_GATE_TIMEOUT_S)
        )
    except TimeoutError as exc:
        raise RequestGateBusy(
            f"request_id {request_id!r} is held by another submission that did not "
            f"finish within {REQUEST_GATE_TIMEOUT_S:.0f}s. Whatever it committed is "
            "recorded under that id: ask again with the same request_id to replay it."
        ) from exc
    async with stack:
        yield


class CancelNotAuthorized(ExperimentCancellationError):
    """The caller holds neither ownership of the job nor its control token.

    A refusal to cancel is told from every other cancellation failure by this
    type. Reading it out of the message ("not authorized") tied a wire code to
    a sentence, so rewording the refusal — or any other cancellation failure
    borrowing those words — silently reclassified it.
    """

    code = "cancel_not_authorized"


def canonical_fingerprint(request_model: Any) -> str:
    """Hash the normalized, explicit-default JSON representation of a request.

    Fields the model declares as PRESENTATION_FIELDS are excluded: they choose
    how the receipt is rendered or how long the call dwells, not what runs.
    The declaration is a pydantic exclude object, so it can reach nested
    sub-fields (execution.wait_s — the response dwell). Hashing them would
    make asking for the same experiment at a different verbosity or dwell an
    idempotency CONFLICT — refusing to hand back a receipt precisely when the
    caller wants to read more of it.
    """
    if hasattr(request_model, "model_dump"):
        payload_builder = getattr(request_model, "canonical_fingerprint_payload", None)
        if callable(payload_builder):
            payload = payload_builder()
        else:
            excluded = getattr(type(request_model), "PRESENTATION_FIELDS", None)
            if isinstance(excluded, set | frozenset):
                excluded = set(excluded)
            payload = request_model.model_dump(mode="json", exclude_unset=False, exclude=excluded)
    else:
        payload = request_model
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def verify_replay_sources(job: ExperimentJob, request_id: str) -> None:
    """Refuse to replay a receipt whose source decks have changed since it ran.

    The canonical fingerprint hashes the request arguments only, so nothing in
    it moves when a circuit is edited underneath a reused request_id — the
    replay would hand back the earlier run's numbers as a confirmed success for
    a circuit that no longer exists.

    Staging already reports exactly this drift through
    ``verify_staged_manifest``; replay is simply the path that never stages, so
    the check is lifted onto it rather than restated. Comparing against the
    manifest the coordinator persisted is what keeps the replay cheap: it needs
    the stored record, not a second staging pass.

    A changed deck raises the conflict a changed payload already raises: both
    are one request_id reused for a different experiment.

    A live include is refused for the same reason a record with no digests is:
    its content was never hashed, so nothing here can show it still holds what
    it held. ``allow_live_includes`` already says in as many words that the job
    cannot prove what those files contained; a replay is that claim made a
    second time, on evidence that has aged. The cost is a re-run of a job that
    opted out of provenance, which is the direction every other case here
    fails.
    """
    for source in job.sources:
        if not any(entry.staged and not entry.live for entry in source.manifest):
            raise IdempotencyConflictError(
                f"request_id {request_id!r} points to experiment {job.job_id}, whose "
                f"record carries no source digest for circuit {source.circuit!r}; its "
                "results cannot be shown to describe the current deck. Submit under a "
                "new request_id to run it again."
            )
        # Ahead of the drift check below: this one reads the record in memory,
        # while that one re-hashes every staged byte. A replay this rejects is
        # rejected either way, so paying for the closure hash first would be
        # work spent on an answer already known.
        live = [str(entry.path) for entry in source.manifest if entry.live]
        if live:
            raise IdempotencyConflictError(
                f"request_id {request_id!r} already ran circuit {source.circuit!r} "
                f"against live include(s) {', '.join(live)}, whose content was never "
                "digested; this receipt cannot be shown to describe them now. Submit "
                "under a new request_id to run it again."
            )
        drift = [
            # Fail closed on a code the table does not know: an unnamed drift
            # kind still blocks the replay, worded by its code verbatim.
            f"{item['evidence']['path']} ({_DRIFT_REASONS.get(item['code'], item['code'])})"
            for item in verify_staged_manifest(source.manifest)
        ]
        if drift:
            raise IdempotencyConflictError(
                f"request_id {request_id!r} already ran circuit {source.circuit!r}, "
                f"whose sources have changed since: {', '.join(drift)}. Submit under a "
                "new request_id to run the updated circuit."
            )


def verify_replay(
    job: ExperimentJob,
    request_id: str,
    executable: SimulatorExecutable | None,
) -> None:
    """Refuse to replay a recorded job the request would not reproduce now.

    The fingerprint covers the request. This covers the server state a request
    does not name, and is the one check both replay routes run: the tool's
    lookup before the request gate, and the coordinator's under it.

    First the decks (``verify_replay_sources``). Then the simulator build:
    swap the executable for another build (or change which simulator is the
    default) and restart, and a reused request_id would hand back the earlier
    build's numbers as the answer for this one. So the program recorded at the
    job's submission is compared with ``executable``, the one this request
    would launch now, and a different build is the same conflict a changed deck
    is. It is checked beside the fingerprint rather than hashed into it: it is
    the server's state, not the caller's request, and folding it in would
    report a changed executable as "a different request payload". A record
    that names no executable fails closed, as a record with no source digests
    does.

    Blocking (the deck check re-hashes staged files): call off the loop.
    """
    verify_replay_sources(job, request_id)
    if same_executable(job.simulator_executable, executable):
        return
    raise IdempotencyConflictError(
        f"request_id {request_id!r} already ran experiment {job.job_id} on "
        f"{describe_executable(job.simulator_executable)}; this request would now run "
        f"on {describe_executable(executable)}. Its results cannot be shown to come from "
        "the current build. Submit under a new request_id to run it on the current "
        "executable; jobs(action='status') still reads the recorded job."
    )


async def already_staged() -> StagedDecks:
    """Stand-in for a staging pass that has run.

    An execution keeps its request for the life of the job, and a real staging
    closure captures the whole submission — the validated arguments, the
    resolved circuits, the lint findings. Swapping it out once the barrier has
    called it lets that scope go.
    """
    raise RuntimeError("This request's decks were staged already")


@dataclass(frozen=True)
class ExperimentRunRequest:
    """One submission's input to the experiment coordinator.

    ``stage`` produces the staged decks the job runs. It is a callable rather
    than the decks themselves because staging copies files and must happen
    inside the request gate, after the coordinator knows this submission is not
    a replay of one already recorded — a caller holding decks already passes
    ``lambda: StagedDecks(cases, sources)``.
    """

    state: SessionState
    request_id: str
    fingerprint: str
    simulator: str
    stage: StageDecks
    job_id: str | None = None
    #: The program this request's cases will launch, identified before the
    #: request gate. Recorded on a new job and compared against a recorded one.
    simulator_executable: SimulatorExecutable | None = None
    declared: int | None = None
    canonicalizer_version: int = CANONICALIZER_VERSION
    max_parallel: int | None = None
    run_timeout_s: float | None = None
    job_deadline_s: float | None = None
    kill_grace_s: float = DEFAULT_KILL_GRACE_S
    analysis_request: dict[str, Any] | None = None
    analysis_callback: AnalysisCallback | None = None
    recoverable: bool = False
    simulator_seed: int | None = None


def effective_run_timeout(
    request: ExperimentRunRequest,
) -> tuple[float | None, RunTimeoutSource | None]:
    """The per-case run timeout a request runs under, and where it came from.

    The request's own ``run_timeout_s`` when it sets one, else the server's
    ``[simulation] run_timeout``, else none: by default a case runs until it
    ends or is cancelled. A timeout destroys the partial result, and the agent
    watching the job sees each case's progress, so the bound is the caller's
    choice rather than a server guess. The source is part of the answer because
    the remedy differs: a request can raise its own value, while the server's
    is the operator's.
    """
    if request.run_timeout_s is not None:
        return request.run_timeout_s, "request"
    if request.state.config.run_timeout is not None:
        return request.state.config.run_timeout, "server_default"
    return None, None


@dataclass(frozen=True)
class ExperimentReceipt:
    """Durable submission receipt returned only by submit/replay."""

    job: ExperimentJob
    replayed: bool
    control_token: str


@dataclass(frozen=True)
class StagedDecks:
    """What a staging pass produced: the cases to run and the decks they ran."""

    cases: list[ExperimentCase]
    sources: list[SourceRecord]


@dataclass(frozen=True)
class AdmissionResult:
    job: ExperimentJob
    replayed: bool
    start: bool | None = None


@dataclass(frozen=True)
class _IndexLookup:
    """What the request index said: the job to replay, or that it named none."""

    existing: ExperimentJob | None
    dangling: bool


@dataclass
class _Execution:
    request: ExperimentRunRequest
    live: LiveJob
    semaphore: asyncio.Semaphore
    capacity: int
    # ``effective_run_timeout``, resolved once so the timer, the message and the
    # evidence of one job all report the same bound.
    run_timeout_s: float | None = None
    run_timeout_source: RunTimeoutSource | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    # Orders a worker thread's launch against a stop requested on the loop.
    # Held for two attribute writes at a time, never across a launch.
    launch_lock: threading.Lock = field(default_factory=threading.Lock)
    stop_reason: Literal["cancelled", "job_deadline"] | None = None
    slots_held: set[str] = field(default_factory=set)
    retained_slots: set[str] = field(default_factory=set)
    futures: dict[str, asyncio.Future[RunOutcome]] = field(default_factory=dict)
    case_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    deadline_task: asyncio.Task[None] | None = None
    external_cancel_task: asyncio.Task[None] | None = None
    case_event_count: int = 0
    # Which cases a cancel has already reported stopping, and what they were
    # doing when it claimed them. A launched case stays non-terminal until its
    # kill lands, so two cancels arriving in that window would each take their
    # own snapshot and each report the same transition.
    claimed_cancel_priors: dict[str, str] = field(default_factory=dict)
    persistence_error: Exception | None = None

    @property
    def job(self) -> ExperimentJob:
        """The record of the job this execution runs."""
        return self.live.job


class ExperimentRunner(RunnerBase):
    """Coordinates case submission, cancellation, completeness, and analysis."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        simulator_class: type,
        output_folder: Path,
        max_parallel: int = DEFAULT_MAX_PARALLEL,
    ):
        super().__init__(loop, simulator_class, output_folder, max_parallel)
        self._executions: dict[str, _Execution] = {}
        # Submission pipelines, and the follow-ups a job leaves behind once its
        # coordinator has finished (keyed by job id).
        self._background = BackgroundTasks()

    def has_active_work(self) -> bool:
        """Whether a submission pipeline or execution still owns live work."""
        return bool(self._background or self._executions)

    def background_pending(self) -> list[asyncio.Task[Any]]:
        """The tasks this runner started and has not finished."""
        return self._background.pending()

    async def settled(self, job: ExperimentJob) -> None:
        """Wait until the work this runner started for ``job`` has finished.

        That is the job's coordinator and the follow-ups it leaves behind,
        such as recording a simulator that exited after its kill grace. Nothing
        outside the process is waited on: a simulator that has not exited is
        not this runner's work until its exit is reported.
        """
        await self._background.settled(job.job_id)

    def owns_experiment_job(self, job_id: str) -> bool:
        """Whether this runner launched ``job_id`` in the current process."""
        return job_id in self._executions

    def submit(
        self,
        request: ExperimentRunRequest,
    ) -> asyncio.Future[ExperimentReceipt]:
        """Start the detached durable-submission pipeline and return its receipt future.

        Callers await the returned future through ``asyncio.shield``. Cancelling
        that dwell cannot cancel this detached pipeline or the durable job.
        """
        receipt_ready: asyncio.Future[ExperimentReceipt] = self.loop.create_future()
        self._background.spawn(self._submission_pipeline(request, receipt_ready), loop=self.loop)
        return receipt_ready

    def _validate_request(self, request: ExperimentRunRequest) -> None:
        """Check what can be checked before a single deck is copied."""
        if request.simulator_seed is not None:
            try:
                validate_seed(request.simulator_seed)
            except ValueError as exc:
                raise RecoveryError("recovery_seed_unsupported", str(exc)) from exc
            if not request.recoverable or not is_ngspice(self.simulator_class):
                raise RecoveryError(
                    "recovery_seed_unsupported", "Explicit seed requires recoverable ngspice"
                )
        if not request.request_id:
            raise SimulationError("request_id is required for durable experiment submission")
        if request.canonicalizer_version != CANONICALIZER_VERSION:
            raise IdempotencyConflictError(
                "Unsupported request canonicalizer version "
                f"{request.canonicalizer_version}; this server uses {CANONICALIZER_VERSION}"
            )
        if self.case_capacity(request) < 1:
            raise SimulationError("max_parallel must be at least 1")

    def materialize_job(self, request: ExperimentRunRequest, staged: StagedDecks) -> ExperimentJob:
        job_id = request.job_id or generate_id("exp")
        validate_job_id(job_id)
        control_token = secrets.token_urlsafe(32)
        store_path = Store(request.state.working_dir).job_record(job_id)
        case_ids = [case.case_id for case in staged.cases]
        run_indices = [case.run_index for case in staged.cases]
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise SimulationError("Experiment case_id values must be non-empty and unique")
        if any(run_index < 0 for run_index in run_indices) or len(set(run_indices)) != len(
            run_indices
        ):
            raise SimulationError("Experiment run_index values must be non-negative and unique")
        for case in staged.cases:
            case.run_token = f"{job_id}_case_{case.run_index}"
            if case.recovery is not None:
                case.recovery = replace(
                    case.recovery,
                    attempt=replace(
                        case.recovery.attempt,
                        execution_job_id=job_id,
                        run_token=case.run_token,
                    ),
                )
        completeness = Completeness(
            declared=request.declared if request.declared is not None else len(staged.cases),
            expanded=len(staged.cases),
        )
        completeness.recount(staged.cases)
        failures = [
            failure_row(case)
            for case in staged.cases
            if case.status in {"failed", "cancelled", "skipped"}
        ]
        analysis = AnalysisStage(
            status=(
                "pending"
                if request.analysis_callback is not None or request.analysis_request is not None
                else "not_requested"
            ),
            request=request.analysis_request,
        )
        return ExperimentJob(
            job_id=job_id,
            request_id=request.request_id,
            fingerprint=request.fingerprint,
            canonicalizer_version=request.canonicalizer_version,
            control_token=control_token,
            store_path=store_path,
            cases=staged.cases,
            sources=staged.sources,
            simulator=request.simulator,
            simulator_executable=request.simulator_executable,
            completeness=completeness,
            # The job's own directory inside the runner's stable output folder,
            # not the folder itself: everything this job wrote is under it, and
            # the reconciliation that re-finds a case's artifacts after a crash
            # reconstructs them from here.
            output_folder=run_dir_in(self.output_folder, job_id),
            failures=failures,
            analysis=analysis,
        )

    async def _submission_pipeline(
        self,
        request: ExperimentRunRequest,
        receipt_ready: asyncio.Future[ExperimentReceipt],
    ) -> None:
        try:
            if (
                not request.state.config.persist_jobs
                or not request.state.job_registry.persist_enabled
            ):
                raise SimulationError(
                    "run_experiments requires durable job persistence; set "
                    "[state] persist_jobs = true and restart the server"
                )
            self._validate_request(request)
            barrier = await self._durable_barrier(request)
            try:
                await self.start_committed(request, barrier, receipt_ready)
            except Exception as exc:
                raise SubmissionCommitted(
                    request.request_id,
                    exc,
                    receipt=ExperimentReceipt(
                        barrier.job, barrier.replayed, barrier.job.control_token
                    ),
                ) from exc
        except Exception as exc:
            if not receipt_ready.done():
                receipt_ready.set_exception(exc)

    async def start_committed(
        self,
        request: ExperimentRunRequest,
        barrier: AdmissionResult,
        receipt_ready: asyncio.Future[ExperimentReceipt],
    ) -> None:
        """Register the durable job and start it.

        Split out so the caller can wrap it whole: past the barrier the job
        exists on disk, and every failure from here has to say so.
        """
        # Registration and execution-task creation intentionally have no
        # await between them. A replay racing the original barrier can
        # therefore never observe a durable job that this process has not
        # either registered or recognized as already registered.
        registry = request.state.job_registry
        registered = request.state.all_jobs.get(barrier.job.job_id)
        running = registry.live_job(barrier.job.job_id)
        coordinating = running is not None and running.task is not None and not running.task.done()
        should_start = not barrier.replayed if barrier.start is None else barrier.start
        if should_start and registered is not barrier.job:
            if self.owns_experiment_job(barrier.job.job_id) or coordinating:
                raise RecoveryError(
                    "recovery_owner_active", "A live coordinator owns this attempt"
                )
            job = barrier.job
            request.state.add_experiment_job(job, already_persisted=True)
        elif registered is not None and registered is not barrier.job:
            # The live job, reserved by the admission that is starting it: a
            # replay answers from that, never from the copy it read.
            job = registered
        else:
            # Unregistered, or reserved by this admission: register it now.
            job = barrier.job
            request.state.add_experiment_job(job, already_persisted=True)
        execution = None
        if should_start and not self.owns_experiment_job(job.job_id) and not coordinating:
            execution = self._new_execution(request, registry.reserve(job))
            self._executions[job.job_id] = execution
        # A replay leaves the record as it is: the receipt's ``replayed`` says
        # this call was answered from it, and only the owner writes a record.
        receipt = ExperimentReceipt(
            job=job,
            replayed=barrier.replayed,
            control_token=job.control_token,
        )
        if not receipt_ready.done():
            receipt_ready.set_result(receipt)
        if execution is not None:
            execution.live.task = self._background.spawn(
                self._run_job(execution), key=job.job_id, loop=self.loop
            )

    async def _durable_barrier(self, request: ExperimentRunRequest) -> AdmissionResult:
        """Claim the request id first, then stage under it.

        The gate is one file lock per ``request_id``, so holding it across
        staging delays exactly one thing: another submission carrying the same
        id. That is the submission that should wait — it finds the index on the
        way in and replays the recorded job instead of copying a second deck
        set no record would ever claim. Every blocking step inside runs off the
        event loop; the staging pass is itself a coroutine that offloads its
        own file work.

        A crash while the gate is held releases the file lock with the process
        and leaves no index entry behind, so the next submission carrying that
        id stages afresh; whatever the dead process had copied under
        ``runs/{job_id}/`` is a crash residual that no record names.
        """
        if request.recoverable:
            from ltspice_mcp.lib.experiment_resume import admit_initial

            return await admit_initial(self, request)
        working_dir = request.state.working_dir
        gate = Store(working_dir).request_lock(request.request_id)
        async with request_gate(gate, request.request_id):
            lookup = await asyncio.to_thread(self.read_request_index, request)
            if lookup.existing is not None:
                return AdmissionResult(lookup.existing, replayed=True)
            try:
                staged = await request.stage()
                candidate = self.materialize_job(request, staged)
                if any(case.native_statistics is not None for case in candidate.cases):
                    await asyncio.to_thread(
                        prepare_native_cases, candidate, working_dir, self.simulator_class
                    )
                if lookup.dangling:
                    candidate.observations.append(
                        {
                            "code": "dangling_request_index_replaced",
                            "kind": "persistence",
                            "detail": (
                                "The request index named a missing coordinator record; "
                                "the durable submission was recreated."
                            ),
                        }
                    )
                # Reserved before the claim makes the id findable on disk, so a
                # replay that finds it gets this job rather than a copy of it.
                request.state.job_registry.reserve(candidate)
                try:
                    await asyncio.to_thread(self._claim_request_id, request, candidate)
                except BaseException:
                    request.state.job_registry.release(candidate)
                    raise
            except Exception:
                # Staged, then refused. The decks are already copied and the
                # claim never landed, so this is the one window in which a run
                # tree exists that no record will ever name.
                await asyncio.to_thread(self.discard_staged_run_dir, request, working_dir)
                raise
        # Outside the gate: the per-circuit index is discovery, not identity,
        # so a slow directory here holds up nothing but this submission.
        try:
            await asyncio.to_thread(self._register_circuits, candidate, working_dir)
        except Exception as exc:
            raise SubmissionCommitted(request.request_id, exc) from exc
        return AdmissionResult(candidate, replayed=False)

    def discard_staged_run_dir(self, request: ExperimentRunRequest, working_dir: Path) -> None:
        """Remove the run tree of a submission that staged and then failed.

        Called only from inside the request gate, only for a job id this
        submission minted, and only while no record claims it. Provenance comes
        from the record that claims an artifact, so a tree nothing claims in
        the box-wide runs root is what a later inventory of that folder
        mistakes for its own work.

        The record check is the refusal: a claimed tree belongs to its job,
        even when this submission failed under the same id. A failure here is
        logged, not raised — leaving a directory behind must not replace the
        error the caller came for.
        """
        job_id = request.job_id
        if not job_id or Store(working_dir).job_record(job_id).exists():
            return
        run_dir = run_dir_in(self.output_folder, job_id)
        try:
            shutil.rmtree(run_dir)
        except FileNotFoundError:
            return
        except OSError as exc:
            logger.warning("could not discard the unclaimed run directory %s: %s", run_dir, exc)

    @staticmethod
    def read_request_index(request: ExperimentRunRequest) -> _IndexLookup:
        """The gate's lookup half: a replay, a conflict, or nothing recorded yet."""
        working_dir = request.state.working_dir
        from ltspice_mcp.lib.experiment_resume import refuse_reserved_request

        refuse_reserved_request(request)
        index = experiment_store.load_request_index(request.request_id, working_dir)
        if index is None:
            return _IndexLookup(existing=None, dangling=False)
        indexed_version = index.get("canonicalizer_version")
        if indexed_version != request.canonicalizer_version:
            raise IdempotencyConflictError(
                f"request_id {request.request_id!r} was stored with canonicalizer "
                f"version {indexed_version!r}, not {request.canonicalizer_version}"
            )
        if index.get("fingerprint") != request.fingerprint:
            raise IdempotencyConflictError(
                f"request_id {request.request_id!r} was already used for a "
                "different request payload"
            )
        indexed_job_id = str(index.get("job_id", ""))
        try:
            existing = experiment_store.load_job(
                indexed_job_id,
                working_dir,
                own_is_alive=True,
            )
        except ValueError:
            existing = None
        if existing is None:
            # The index names a record that is gone; this submission recreates it.
            return _IndexLookup(existing=None, dangling=True)
        if (
            existing.request_id != request.request_id
            or existing.fingerprint != request.fingerprint
            or existing.canonicalizer_version != request.canonicalizer_version
        ):
            raise IdempotencyConflictError(
                f"request_id {request.request_id!r} points to an inconsistent coordinator record"
            )
        # The cheap pre-staging replay check the caller may have run cannot see
        # a record written after it looked, so the same drift is re-checked
        # here, under the gate.
        verify_replay(existing, request.request_id, request.simulator_executable)
        if existing.restart_reconciled:
            experiment_store.save_job(existing)
        return _IndexLookup(existing=existing, dangling=False)

    @staticmethod
    def _claim_request_id(
        request: ExperimentRunRequest,
        candidate: ExperimentJob,
    ) -> None:
        """The gate's write half: the request index, then the coordinator record."""
        # The request is the single source of the idempotency identity: stamping
        # it here means no caller can persist a coordinator record whose identity
        # disagrees with its own request index.
        candidate.request_id = request.request_id
        candidate.fingerprint = request.fingerprint
        candidate.canonicalizer_version = request.canonicalizer_version
        experiment_store.save_request_index(
            request_id=request.request_id,
            fingerprint=request.fingerprint,
            canonicalizer_version=request.canonicalizer_version,
            job_id=candidate.job_id,
            working_dir=request.state.working_dir,
        )
        experiment_store.save_job(candidate)

    @staticmethod
    def _register_circuits(candidate: ExperimentJob, working_dir: Path) -> None:
        """Index the job under every circuit it ran; a failure is durable-but-noted."""
        try:
            experiment_store.register_circuits(candidate, working_dir)
        except OSError as exc:
            candidate.observations.append(
                {
                    "code": "experiment_index_write_failed",
                    "kind": "persistence",
                    "detail": (
                        "The coordinator is durable, but one or more circuit "
                        f"index entries could not be written: {exc}"
                    ),
                }
            )
            try:
                experiment_store.save_job(candidate)
            except OSError:
                # The note about the index could not be persisted either. It
                # still reaches this submission's receipt, and the record the
                # claim wrote is the durable one — losing a note is no reason
                # to fail a job that exists and can run.
                logger.warning(
                    "could not record the circuit-index failure on %s", candidate.job_id
                )

    def case_capacity(self, request: ExperimentRunRequest) -> int:
        """One job's share of the runner's cap.

        A request's own ``max_parallel`` divides that share; it can never raise
        it, because the runner's permits are what the machine is protected by
        and every other job in this process is drawing on the same pool.
        """
        requested = request.max_parallel if request.max_parallel is not None else self.max_parallel
        return min(requested, self.max_parallel)

    def _new_execution(
        self,
        request: ExperimentRunRequest,
        live: LiveJob,
    ) -> _Execution:
        job = live.job
        capacity = self.case_capacity(request)
        run_timeout_s, run_timeout_source = effective_run_timeout(request)
        if job.recovery is not None:
            recorded = job.recovery.execution
            capacity = recorded.max_parallel
            run_timeout_s = recorded.run_timeout_s
            if recorded.timeout_source in {"request", "server_default"}:
                run_timeout_source = (
                    "request" if recorded.timeout_source == "request" else "server_default"
                )
            elif recorded.timeout_source == "unbounded":
                run_timeout_source = None
            else:
                raise RecoveryError("recovery_execution_invalid", "Unsupported timeout source")
            request = replace(
                request,
                kill_grace_s=recorded.kill_grace_s,
                job_deadline_s=recorded.job_deadline_s,
                simulator_seed=recorded.simulator_seed,
            )
        return _Execution(
            # Without the staging closure: this execution outlives the
            # submission call, and the closure holds that whole scope.
            request=replace(request, stage=already_staged),
            live=live,
            semaphore=asyncio.Semaphore(capacity),
            capacity=capacity,
            run_timeout_s=run_timeout_s,
            run_timeout_source=run_timeout_source,
        )

    async def wait(
        self,
        job: ExperimentJob,
        timeout_s: float | None = None,
        *,
        wait_for: Literal["all", "runs"] = "all",
    ) -> bool:
        """Wait for full terminality or run terminality without mutating the job.

        Waits on the live job this runner executes under that id, whichever
        record of it the caller holds. A job it does not execute (finished and
        released, or never its own) has nothing here to wait on, so its record
        answers at once.
        """
        execution = self._executions.get(job.job_id)
        if execution is None:
            return finished(job, wait_for)
        return await execution.live.wait(timeout_s, wait_for=wait_for)

    async def _run_job(self, execution: _Execution) -> None:
        job = execution.job
        request = execution.request
        try:
            execution.external_cancel_task = self.loop.create_task(
                self._external_cancel_watch(execution)
            )
            await self._transition_job(execution, "running", total_cases=len(job.cases))
            for case in job.cases:
                if case.status in TERMINAL_CASE_STATUSES:
                    continue
                task = self.loop.create_task(self._run_case(execution, case))
                execution.case_tasks[case.case_id] = task
            if request.job_deadline_s is not None:
                execution.deadline_task = self.loop.create_task(
                    self._deadline_watch(execution, request.job_deadline_s)
                )
            if execution.case_tasks:
                await asyncio.gather(*execution.case_tasks.values(), return_exceptions=True)
            self._reconcile_unfinished_cases(execution)
            job.completeness.validate_terminal()
            await self._persist_job(execution)
            execution.live.mark_runs_done()

            if execution.stop_reason == "cancelled":
                self._finish_unstarted_analysis(
                    job,
                    status="cancelled",
                    error="Experiment cancelled before attached analysis started",
                )
                if job.status not in {"cancelled", "failed"}:
                    await self._transition_job(execution, "cancelled")
                return
            if execution.stop_reason == "job_deadline":
                self._finish_unstarted_analysis(
                    job,
                    status="cancelled",
                    error="Experiment deadline elapsed before attached analysis started",
                )
                await self._transition_job(execution, "completed_with_failures")
                return

            analysis_failed = await self._run_analysis(execution)
            if execution.stop_reason == "cancelled":
                await self._transition_job(execution, "cancelled")
            elif (
                execution.stop_reason == "job_deadline"
                or analysis_failed
                or job.completeness.fell_short
            ):
                await self._transition_job(execution, "completed_with_failures")
            else:
                await self._transition_job(execution, "completed")
        except Exception as exc:
            logger.exception("Experiment job %s failed", job.job_id)
            job.error = str(exc)
            self._reconcile_unfinished_cases(execution)
            self._recount_completeness(job)
            execution.live.mark_runs_done()
            self._finish_unstarted_analysis(
                job,
                status="failed",
                error="Experiment coordination failed before attached analysis completed",
            )
            if job.status not in {
                "completed",
                "completed_with_failures",
                "failed",
                "cancelled",
                "interrupted",
            }:
                try:
                    await self._transition_job(execution, "failed", error=job.error)
                except Exception as checkpoint_error:
                    # The committed identity remains replayable even when the
                    # storage device will not accept its failure checkpoint.
                    job.error += f"; failure checkpoint: {checkpoint_error}"
                    execution.live.transition("failed", error=job.error)
        finally:
            if execution.deadline_task is not None:
                execution.deadline_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await execution.deadline_task
            if execution.external_cancel_task is not None:
                execution.external_cancel_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await execution.external_cancel_task
            if not execution.retained_slots:
                self._executions.pop(job.job_id, None)

    async def _external_cancel_watch(self, execution: _Execution) -> None:
        """Observe durable cancellation requests made by another server process."""
        while not execution.live.done:
            requested = await asyncio.to_thread(
                experiment_store.cancellation_requested,
                execution.job.job_id,
                execution.request.state.working_dir,
            )
            if requested:
                execution.job.observations.append(
                    {
                        "code": "external_cancellation_requested",
                        "kind": "execution",
                        "detail": (
                            "A server process holding the experiment control token "
                            "requested cancellation through the durable job store."
                        ),
                    }
                )
                self._request_stop(execution, "cancelled")
                await self._persist_job(execution)
                return
            await asyncio.sleep(experiment_store.FOREIGN_RECORD_POLL_S)

    async def _deadline_watch(self, execution: _Execution, deadline_s: float) -> None:
        if deadline_s <= 0:
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(deadline_s)
        if execution.live.done:
            return
        execution.job.observations.append(
            {
                "code": "job_deadline",
                "kind": "execution",
                "detail": f"The experiment deadline of {deadline_s:g}s elapsed.",
            }
        )
        self._request_stop(execution, "job_deadline")

    async def _persist_job(self, execution: _Execution, job: ExperimentJob | None = None) -> None:
        """Await recovery checkpoints in the registry's existing per-job order."""
        if execution.job.recovery is None:
            execution.request.state.persist_job(execution.job)
            return
        try:
            await execution.request.state.job_registry.persist_strict(job or execution.job)
        except Exception as exc:
            execution.persistence_error = exc
            raise

    async def _transition_job(self, execution: _Execution, status: str, **extra: Any) -> None:
        """Publish terminal recovery state only after its durable checkpoint."""
        job = execution.job
        if job.recovery is None:
            execution.live.transition(status, state=execution.request.state, **extra)
            return
        if status in TERMINAL_STATUSES:
            if status not in VALID_EXPERIMENT_TRANSITIONS.get(job.status, frozenset()):
                raise InvalidTransitionError(
                    f"Illegal recovery transition {job.status} to {status}"
                )
            snapshot = replace(job, status=status, completed_at=now())
            await self._persist_job(execution, snapshot)
            execution.live.transition(status, **extra)
            job.completed_at = snapshot.completed_at
        else:
            execution.live.transition(status, **extra)
            await self._persist_job(execution)

    def _verify_recovery_case(self, execution: _Execution, case: ExperimentCase) -> Path:
        """Check frozen policy and bytes at staging and again immediately before spawn."""
        if execution.persistence_error is not None:
            raise RecoveryError("recovery_persistence_failed", "A durable checkpoint failed")
        job = execution.job
        recovery = job.recovery
        if recovery is None or case.recovery is None or job.output_folder is None:
            raise RecoveryError("recovery_record_invalid", "Recovery launch facts are missing")
        recorded = recovery.execution
        root = Store(execution.request.state.working_dir).lineage_run_dir(
            recovery.root_job_id, job.output_folder, self.simulator_class
        )
        if case.recovery.inputs.lineage_root != root:
            raise RecoveryError("recovery_path_escape", "Case inputs name a different lineage")
        verify_execution_policy(recorded, self.simulator_class)
        verify_case_inputs(
            case.recovery.inputs,
            native=case.native_statistics is not None,
            seeded=recorded.simulator_seed is not None,
        )
        verify_seeded_driver(case, recorded, root)
        verify_startup(recorded.startup, root)
        return root

    def _record_launch_intent(
        self, execution: _Execution, case: ExperimentCase, copied: Path
    ) -> None:
        """Spicelib has copied the deck but has not constructed a RunTask yet."""
        root = self._verify_recovery_case(execution, case)
        assert case.recovery is not None
        attempt = case.recovery.attempt
        if (
            attempt.launch is not None
            or attempt.execution_job_id != execution.job.job_id
            or attempt.reused
        ):
            raise RecoveryError("recovery_attempt_invalid", "An attempt cannot be launched twice")
        copied = copied.resolve(strict=True)
        if not copied.is_relative_to(root):
            raise RecoveryError("recovery_path_escape", "Executed copy is outside its lineage")
        copied_digest = sha256_file(copied)
        adaptation: Literal["identity", "logopinfo", "native_driver", "seeded_driver"]
        if case.native_statistics is not None:
            adaptation = "native_driver"
        elif attempt.seeded_driver is not None:
            if copied_digest != attempt.seeded_driver.sha256:
                raise RecoveryError("recovery_input_drift", "Executed seeded driver bytes changed")
            adaptation = "seeded_driver"
        elif copied_digest == case.recovery.inputs.electrical.sha256:
            adaptation = "identity"
        else:
            if is_ngspice(self.simulator_class):
                raise RecoveryError(
                    "recovery_input_drift",
                    "Executed copy differs from the frozen electrical input",
                )
            adaptation = "logopinfo"
        intent = LaunchIntent(
            now(),
            case.recovery.inputs.electrical.sha256,
            ArtifactDigest(copied, copied_digest),
            adaptation,
        )
        case.recovery = replace(case.recovery, attempt=replace(attempt, launch=intent))
        # The loop remains free for cancellation and the ordered registry lock;
        # only this worker waits. Never hold launch_lock across this barrier.
        asyncio.run_coroutine_threadsafe(self._persist_job(execution), self.loop).result()
        self._verify_executed_copy(execution, case)
        if execution.cancel_event.is_set():
            raise RecoveryError("recovery_launch_cancelled", "Cancelled before simulator spawn")

    def _verify_executed_copy(self, execution: _Execution, case: ExperimentCase) -> None:
        root = self._verify_recovery_case(execution, case)
        assert case.recovery is not None
        intent = case.recovery.attempt.launch
        if intent is None or sha256_file(intent.executed.path) != intent.executed.sha256:
            raise RecoveryError("recovery_input_drift", "Executed copy changed before launch")
        assert execution.job.recovery is not None
        recorded = execution.job.recovery.execution
        if recorded.startup.ini_template is not None:
            from ltspice_mcp.lib.controlled_ltspice import verify_attempt_ini

            ini = Store(execution.request.state.working_dir).recovery_ini(
                execution.job.recovery.root_job_id, case.run_token, self.simulator_class
            )
            verify_attempt_ini(recorded.startup.ini_template, ini, root)
        if execution.job.recovery.execution.simulator_seed is not None and any(
            path.exists() or path.is_symlink()
            for path in (root / (case.run_token + ".raw"), root / (case.run_token + ".log"))
        ):
            raise RecoveryError(
                "recovery_input_drift", "Refusing to overwrite previous seeded outputs"
            )

    def _submit_case_under_cancel_gate(
        self,
        execution: _Execution,
        case: ExperimentCase,
        suffix: str,
    ) -> bool:
        """Submit only if no cancellation won the gate, recording the launch first.

        The submitted stamp goes on here, in the worker thread and before the
        launch, so "a simulator may now exist under this run token" and "the
        record says so" cannot be observed out of order. Stamping afterwards --
        or back on the event loop once this thread returns -- leaves a window
        where a cancel reads the case as never submitted: it then reports a run
        that did start as pre-submission, counts it out of ``submitted``, and
        skips the token-scoped kill, so the simulator runs to completion into an
        artifact no record claims. The lock is what makes the gate and the stamp
        one decision against a stop requested on the loop; it is released before
        the launch so the loop never waits on a simulator.
        """
        working_dir = execution.request.state.working_dir
        job_id = execution.job.job_id
        recovery = execution.job.recovery
        recoverable = recovery is not None
        lineage_id = recovery.root_job_id if recovery is not None else job_id
        run_dir = (
            self._verify_recovery_case(execution, case)
            if recoverable
            else run_dir_in(self.output_folder, job_id)
        )
        native = None
        prepared = case.native_statistics.prepared if case.native_statistics else None
        if case.native_statistics is not None:
            if prepared is None:
                raise NativeCaseError("preparation", "native setup was not prepared")
            if recovery is not None and prepared.policy != recovery.execution.native_policy:
                raise RecoveryError("recovery_execution_changed", "Native launch policy changed")
            verify_launch(prepared)
            native = NativeLaunchContext(
                input_deck=Path(prepared.paths.electrical_input),
                cwd=Path(prepared.paths.cwd),
                verify_execution=lambda: verify_launch(prepared, executed_copy=True),
                policy=prepared.policy,
            )
            suffix = ".cir"
        if prepared is not None:
            run_deck = Path(prepared.paths.prepared_driver)
        elif recovery is not None and recovery.execution.simulator_seed is not None:
            assert case.recovery is not None and case.recovery.attempt.seeded_driver is not None
            run_deck = case.recovery.attempt.seeded_driver.path
            native = NativeLaunchContext(
                input_deck=case.recovery.inputs.electrical.path,
                cwd=run_dir,
                policy=NativeLaunchPolicy(
                    ngbehavior=recovery.execution.ngbehavior or "", ng_nomodcheck=False
                ),
            )
            suffix = ".cir"
        else:
            # On LTspice .op cases, hand the simulator a sibling copy carrying
            # '.options logopinfo' — without it the log has no per-device
            # small-signal block and analysis reads back no gm/vth/vdsat. Injecting
            # here rather than at staging is what keeps the staged deck and the
            # deck_sha256 the record pins byte-identical: those are what a replay
            # and every provenance check compare against. No-op for ngspice and for
            # decks with no .op. The run_token stamp keeps concurrent cases sharing
            # one staged deck from clobbering each other's copy.
            run_deck = inject_logopinfo(case.staged_deck, self.simulator_class, case.run_token)
            # On ngspice, a `.control` script replaces the raw the simulator would
            # otherwise write, so a scripted deck that never calls write/wrdata
            # produces no rawfile at all and every recipe over the case reads
            # nothing. Give it one at the path this case's artifacts already use.
            # Mutually exclusive with the LTspice injection above by simulator, so
            # chaining on run_deck is safe. The injection is a fact about the run,
            # not about the deck the record pins: the staged deck and its digest
            # stay byte-identical either way.
            run_dir.mkdir(parents=True, exist_ok=True)
            scripted_deck = inject_ngspice_control_write(
                run_deck, self.simulator_class, case.run_token, run_dir
            )
            if scripted_deck != run_deck:
                run_deck = scripted_deck
                case.observations.append(
                    {
                        "code": "control_write_injected",
                        "kind": "execution",
                        "detail": (
                            "The deck's .control script wrote no rawfile of its own, so a "
                            "'write <this case's raw path>' was added before .endc for this "
                            "run. A script that runs several analyses, or writes per "
                            "iteration, still needs its own writes: 'write' captures the "
                            "current plot only."
                        ),
                    }
                )
        native_args: dict[str, Any] = {"native": native} if native is not None else {}
        if recoverable:
            assert execution.job.recovery is not None
            recorded = execution.job.recovery.execution

            def verify_copy() -> None:
                self._verify_executed_copy(execution, case)

            if recorded.startup.ini_template is not None:
                from ltspice_mcp.lib.controlled_ltspice import controlled_ltspice

                ini = Store(working_dir).recovery_ini(
                    lineage_id, case.run_token, self.simulator_class
                )
                adapter = controlled_ltspice(recorded, ini, verify_copy)
            else:
                adapter = controlled_ngspice(recorded, verify_copy)
            native_args.update(
                simulator_class=adapter,
                prelaunch_check=lambda copied: self._record_launch_intent(execution, case, copied),
            )
        run_timeout = execution.run_timeout_s

        def completion_logs(raw: Path | None, log: Path | None) -> DecodedLog:
            from ltspice_mcp.lib.services import AnalysisSource, load_logs_sync

            source = AnalysisSource(
                raw=raw,
                log=log,
                console=log.with_suffix(".exe.log") if log is not None else None,
                netlist=native.input_deck if native is not None else run_deck,
                dialect=dialect_for_simulator_name(self.simulator_class.__name__),
                identity=None,
                trusted_job_artifact=True,
            )
            return load_logs_sync(source, execution.request.state)

        try:
            with file_lock(Store(working_dir).cancellation_lock(job_id)):
                if experiment_store.cancellation_requested(job_id, working_dir):
                    return False
                with execution.launch_lock:
                    if execution.cancel_event.is_set():
                        return False
                    case.status = "submitted"
                    case.submitted_at = now()
                self.submit_netlist(
                    run_deck,
                    # A sub-path, not a bare name: the simulator layer joins it onto
                    # the runner's output folder, so this is what lands the run's
                    # deck copy, raw and log in the job's own directory without the
                    # runner (and its shared concurrency semaphore) ever moving.
                    run_filename_in(lineage_id, f"{case.run_token}{suffix}"),
                    lambda outcome: self._handle_case_completion(
                        execution.job.job_id,
                        case.case_id,
                        outcome,
                    ),
                    timeout_s=(
                        run_timeout + execution.request.kill_grace_s + SPICELIB_TIMEOUT_MARGIN_S
                        if run_timeout is not None
                        else None
                    ),
                    completion_logs=completion_logs,
                    **native_args,
                )
        except NativePrelaunchRefused:
            # The stamp guarded cancellation while launch was in progress, but
            # this refusal happened before spicelib could create a RunTask.
            if not recoverable:
                with execution.launch_lock:
                    case.submitted_at = None
            raise
        finally:
            # spicelib stages the deck synchronously inside run(), so the copy
            # has done its job by the time submit returns — and on a submit that
            # raised, nothing will ever read it. The marker guard inside the
            # helper makes this incapable of touching the staged deck itself.
            if native is None and not recoverable:
                discard_generated_netlist(run_deck)
        return True

    async def _run_case(self, execution: _Execution, case: ExperimentCase) -> None:
        acquired = False
        try:
            # Two layers, innermost last: the job's own share of the cap, then
            # the runner-wide permit every launch in this process competes for.
            await execution.semaphore.acquire()
            try:
                await self.acquire_launch_slot()
            except BaseException:
                execution.semaphore.release()
                raise
            acquired = True
            execution.slots_held.add(case.case_id)
            if case.status != "queued" or execution.cancel_event.is_set():
                if case.status == "queued":
                    self._terminalize_stopped_case(execution, case)
                self._release_slot(execution, case.case_id)
                return

            future: asyncio.Future[RunOutcome] = self.loop.create_future()
            execution.futures[case.case_id] = future
            suffix = case.staged_deck.suffix or ".net"
            try:
                if execution.job.recovery is not None:
                    from ltspice_mcp.lib.experiment_resume import mark_attempt_launched

                    try:
                        await mark_attempt_launched(execution.job, execution.request.state)
                    except Exception as exc:
                        execution.persistence_error = exc
                        raise
                submitted = await asyncio.to_thread(
                    self._submit_case_under_cancel_gate,
                    execution,
                    case,
                    suffix,
                )
                if not submitted:
                    self._request_stop(
                        execution,
                        "cancelled",
                        exclude_case_id=case.case_id,
                    )
                    self._release_slot(execution, case.case_id)
                    return
                # Stamped in the worker thread next to the launch; persist that
                # transition now that we are back on the loop.
                self._checkpoint_case_transition(execution)
                case.status = "running"
                self._checkpoint_case_transition(execution)
            except Exception as exc:
                self._mark_case(
                    execution,
                    case,
                    "failed",
                    code="submission_failed",
                    error=f"Submission failed: {exc}",
                )
                self._release_slot(execution, case.case_id)
                return

            outcome, stop_reason = await self._await_case(execution, future)
            if stop_reason is not None:
                await self._stop_active_case(execution, case, future, stop_reason)
            elif outcome is not None:
                self._apply_outcome(case, outcome)
                if outcome.error is None:
                    if execution.job.recovery is not None:
                        assert case.recovery is not None
                        if case.raw_file is None or case.log_file is None:
                            raise RecoveryError(
                                "recovery_artifact_missing", "Completed run lacks raw/log pair"
                            )
                        outputs = await asyncio.to_thread(
                            capture_produced_artifacts,
                            case.raw_file,
                            case.log_file,
                            lineage_root=case.recovery.inputs.lineage_root,
                        )
                        produced = replace(
                            case,
                            status="produced",
                            completed_at=now(),
                            recovery=replace(
                                case.recovery,
                                attempt=replace(case.recovery.attempt, outputs=outputs),
                            ),
                        )
                        checkpoint = replace(
                            execution.job,
                            cases=[
                                produced if item is case else item for item in execution.job.cases
                            ],
                            completeness=replace(execution.job.completeness),
                        )
                        self._recount_completeness(checkpoint)
                        await self._persist_job(execution, checkpoint)
                        case.recovery = produced.recovery
                        self._mark_case(execution, case, "produced")
                        case.completed_at = produced.completed_at
                    else:
                        self._mark_case(execution, case, "produced")
                else:
                    self._mark_case(
                        execution,
                        case,
                        "failed",
                        code=outcome.failure_code or "execution_failed",
                        error=outcome.error,
                        evidence=outcome.failure_evidence,
                    )
                self._release_slot(execution, case.case_id)
        except asyncio.CancelledError:
            if acquired and case.case_id in execution.slots_held and case.status == "queued":
                self._release_slot(execution, case.case_id)
            raise
        except Exception as exc:
            if execution.job.recovery is None:
                raise
            self._mark_case(
                execution,
                case,
                "failed",
                code="recovery_persistence_failed"
                if execution.persistence_error is not None
                else "recovery_artifact_invalid",
                error=str(exc),
            )
            if acquired and case.case_id not in execution.retained_slots:
                self._release_slot(execution, case.case_id)

    async def _await_case(
        self,
        execution: _Execution,
        future: asyncio.Future[RunOutcome],
    ) -> tuple[RunOutcome | None, StopReason | None]:
        stop_wait = self.loop.create_task(execution.cancel_event.wait())
        completion_wait = asyncio.shield(future)
        try:
            done, _ = await asyncio.wait(
                {completion_wait, stop_wait},
                timeout=execution.run_timeout_s,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if completion_wait in done:
                return future.result(), None
            if stop_wait in done:
                return None, execution.stop_reason or "cancelled"
            return None, "run_timeout"
        finally:
            stop_wait.cancel()
            completion_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_wait

    async def _stop_active_case(
        self,
        execution: _Execution,
        case: ExperimentCase,
        future: asyncio.Future[RunOutcome],
        reason: StopReason,
    ) -> None:
        cause, limits = _stop_bound(execution, reason)
        outcome = await self._kill_until_exit(case, future, execution.request.kill_grace_s)
        if outcome is None:
            terminal_status = "cancelled" if reason == "cancelled" else "failed"
            self._mark_case(
                execution,
                case,
                terminal_status,
                code="kill_unconfirmed",
                error=(
                    f"Case stopped because {cause}; simulator exit was not confirmed "
                    f"within {execution.request.kill_grace_s:g}s"
                ),
                evidence={
                    "stop_reason": reason,
                    **limits,
                    "kill_grace_s": execution.request.kill_grace_s,
                },
            )
            case.observations.append(
                {
                    "code": "kill_unconfirmed",
                    "kind": "execution",
                    "detail": (
                        "The simulator did not report exit within the bounded kill "
                        "grace period; its concurrency permit remains reserved."
                    ),
                }
            )
            execution.retained_slots.add(case.case_id)
            await self._persist_job(execution)
            self._fail_if_capacity_retained(execution)
            return

        if execution.job.recovery is not None:
            self._apply_outcome(case, outcome)
        else:
            self._apply_stopped_outcome(case, outcome)
        await self._record_progress_then_remove(execution.job, case)
        terminal_status = "cancelled" if reason == "cancelled" else "failed"
        self._mark_case(
            execution,
            case,
            terminal_status,
            code=reason,
            error=f"Case stopped because {cause}",
            evidence=_stopped_run_evidence(limits, outcome),
        )
        self._release_slot(execution, case.case_id)

    async def _kill_until_exit(
        self,
        case: ExperimentCase,
        future: asyncio.Future[RunOutcome],
        grace_s: float,
    ) -> RunOutcome | None:
        """Kill the case's simulator until it reports exit; None if it never does.

        The grace period starts once the first kill has returned (on WSL that
        kill is itself a Windows process query that can take many seconds), and
        up to ``KILL_MAX_PASSES`` kills are made inside it,
        ``KILL_RESCAN_INTERVAL_S`` apart. A failing kill is recorded once.
        """
        failure_noted = False

        async def kill() -> None:
            nonlocal failure_noted
            try:
                await self._kill_case(case.run_token)
            except Exception as exc:
                if not failure_noted:
                    failure_noted = True
                    case.observations.append(
                        {
                            "code": "kill_attempt_failed",
                            "kind": "execution",
                            "detail": f"Scoped simulator termination raised an error: {exc}",
                        }
                    )

        await kill()
        deadline = self.loop.time() + grace_s
        for kill_pass in range(1, KILL_MAX_PASSES + 1):
            remaining = max(0.0, deadline - self.loop.time())
            last = kill_pass == KILL_MAX_PASSES or remaining <= KILL_RESCAN_INTERVAL_S
            try:
                return await asyncio.wait_for(
                    asyncio.shield(future), remaining if last else KILL_RESCAN_INTERVAL_S
                )
            except TimeoutError:
                # The wait to the deadline ends the grace, not a clock read after
                # it: asyncio fires a timer up to one clock resolution early
                # (15.6 ms on Windows), so the clock can still read short.
                if last:
                    return None
            await kill()
        return None

    def _handle_case_completion(
        self,
        job_id: str,
        case_id: str,
        outcome: RunOutcome,
    ) -> None:
        execution = self._executions.get(job_id)
        if execution is None:
            return
        future = execution.futures.get(case_id)
        if case_id not in execution.retained_slots:
            if future is not None and not future.done():
                future.set_result(outcome)
            return
        if future is not None and not future.done():
            future.set_result(outcome)
        case = next((item for item in execution.job.cases if item.case_id == case_id), None)
        if case is None:
            return
        if execution.job.recovery is not None:
            self._apply_outcome(case, outcome)
        else:
            self._apply_stopped_outcome(case, outcome)
        case.observations.append(
            {
                "code": "late_simulator_exit",
                "kind": "execution",
                "detail": (
                    "The simulator reported exit after its kill grace period; "
                    "final artifact facts were recorded."
                ),
                "evidence": {
                    "reported_raw": outcome.raw_file or None,
                    "raw_size": outcome.raw_size,
                    "exit_error": outcome.error,
                },
            }
        )
        # The permit is free once the process is gone. Read progress off-loop,
        # preserve recovery artifacts, and save the final facts.
        self._background.spawn(self._retire_late_exit(execution, case), key=job_id, loop=self.loop)
        self._release_slot(execution, case_id)
        if execution.live.done and not execution.retained_slots:
            self._executions.pop(job_id, None)

    async def _retire_late_exit(self, execution: _Execution, case: ExperimentCase) -> None:
        """Record late-exit progress and persist final artifact facts."""
        try:
            await self._record_progress_then_remove(execution.job, case)
        finally:
            await self._persist_job(execution)

    async def _record_progress_then_remove(self, job: ExperimentJob, case: ExperimentCase) -> None:
        """Record how far a stopped case's simulator got, then remove its heavy artifacts.

        In that order, because the partial raw is among what is removed:
        reading it is the last moment anything can say how far the run got.
        The observation is recorded on the loop before the removal starts, so
        nothing that sees the artifacts gone can see the case without it.
        """
        try:
            progress = await asyncio.to_thread(self._read_progress, job, case)
            if progress is not None:
                case.observations.append(progress)
        finally:
            await asyncio.to_thread(self._remove_case_artifacts, job, case)

    def _read_progress(self, job: ExperimentJob, case: ExperimentCase) -> dict[str, Any] | None:
        """How far a stopped case's simulator got, read off its partial raw.

        Neither simulator writes progress anywhere else a stopped run keeps:
        LTspice's log carries only its preamble until the run ends, and ngspice
        prints its ``Reference value`` progress to stdout only when it has no
        ``-o`` log, which spicelib always passes — so the ``.log`` and
        ``.exe.log`` of a killed ngspice run hold none either (observed on
        ngspice 42 killed with SIGKILL, and on LTspice 26.1 killed the same way
        under Wine).
        """
        return _progress_observation(
            case.case_id,
            case.run_index,
            experiment_store.case_raw_path(job, case),
            dialect_for_simulator_name(job.simulator),
            code="partial_progress",
        )

    @staticmethod
    def _apply_outcome(case: ExperimentCase, outcome: RunOutcome) -> None:
        case.raw_file = Path(outcome.raw_file) if outcome.raw_file else None
        case.log_file = Path(outcome.log_file) if outcome.log_file else None
        case.simulator_version = outcome.simulator_version
        if outcome.observations:
            case.observations.extend(outcome.observations)

    @staticmethod
    def _apply_stopped_outcome(case: ExperimentCase, outcome: RunOutcome) -> None:
        """Keep post-kill diagnostics without advertising artifacts we remove."""
        ExperimentRunner._apply_outcome(case, outcome)
        case.raw_file = None

    def _mark_case(
        self,
        execution: _Execution,
        case: ExperimentCase,
        status: Literal["produced", "failed", "cancelled", "skipped"],
        *,
        code: str | None = None,
        error: str | None = None,
        evidence: dict[str, Any] | None = None,
    ) -> None:
        if case.status in TERMINAL_CASE_STATUSES:
            return
        case.status = status
        case.failure_code = code
        case.failure_evidence = evidence
        case.error = error
        case.completed_at = now()
        if status in {"failed", "cancelled", "skipped"}:
            execution.job.failures.append(failure_row(case))
        self._checkpoint_case_transition(execution)

    def _checkpoint_case_transition(self, execution: _Execution) -> None:
        """Recount case state and sparsely persist case-level progress."""
        self._recount_completeness(execution.job)
        execution.case_event_count += 1
        total = len(execution.job.cases)
        step = max(1, total // 20) if total else 1
        if execution.job.recovery is None and execution.case_event_count % step == 0:
            execution.request.state.persist_job(execution.job)

    def _release_slot(self, execution: _Execution, case_id: str) -> None:
        """Release a case's acquired capacity exactly once on the event loop."""
        if case_id not in execution.slots_held:
            return
        execution.slots_held.discard(case_id)
        execution.retained_slots.discard(case_id)
        execution.semaphore.release()
        self.release_launch_slot()

    def _fail_if_capacity_retained(self, execution: _Execution) -> None:
        if len(execution.retained_slots) < execution.capacity:
            return
        for case in execution.job.cases:
            if case.status != "queued":
                continue
            self._mark_case(
                execution,
                case,
                "failed",
                code="kill_unconfirmed_capacity",
                error=(
                    "All experiment capacity is reserved by simulator processes "
                    "whose exit could not be confirmed"
                ),
            )
            task = execution.case_tasks.get(case.case_id)
            if task is not None:
                task.cancel()
        execution.job.observations.append(
            {
                "code": "kill_unconfirmed_capacity",
                "kind": "execution",
                "detail": (
                    "All concurrency permits remain reserved by unconfirmed "
                    "simulator exits; queued cases were failed without submission."
                ),
            }
        )
        if execution.job.recovery is None:
            execution.request.state.persist_job(execution.job)

    def _request_stop(
        self,
        execution: _Execution,
        reason: Literal["cancelled", "job_deadline"],
        *,
        exclude_case_id: str | None = None,
    ) -> None:
        if execution.stop_reason is None:
            execution.stop_reason = reason
        # A case mid-launch either stamped itself submitted before this point --
        # and is stopped through the kill path, which can actually reach its
        # simulator -- or sees the flag and never launches. Reading the statuses
        # under the same lock is what leaves no third answer.
        with execution.launch_lock:
            execution.cancel_event.set()
            unstarted = [case for case in execution.job.cases if case.status == "queued"]
        for case in unstarted:
            self._terminalize_stopped_case(execution, case)
            task = execution.case_tasks.get(case.case_id)
            if task is not None and case.case_id != exclude_case_id:
                task.cancel()

    def _terminalize_stopped_case(
        self,
        execution: _Execution,
        case: ExperimentCase,
    ) -> None:
        # Reachable for a launched case only through end-of-job reconciliation,
        # which is guarded on "not terminal" alone. Say which one happened
        # rather than assume the case never started: an accounting message that
        # can be false is how a completed run gets read as one that never ran.
        launched = case.submitted_at is not None
        if execution.stop_reason == "job_deadline":
            self._mark_case(
                execution,
                case,
                "failed",
                code="job_deadline",
                error=(
                    "The experiment deadline elapsed while this case was running"
                    if launched
                    else "The experiment deadline elapsed before this case was submitted"
                ),
            )
        else:
            self._mark_case(
                execution,
                case,
                "cancelled",
                code="cancelled",
                error=(
                    "Cancelled after the simulator was launched"
                    if launched
                    else "Cancelled before submission"
                ),
            )

    def _reconcile_unfinished_cases(self, execution: _Execution) -> None:
        for case in execution.job.cases:
            if case.status in TERMINAL_CASE_STATUSES:
                continue
            if execution.stop_reason is not None:
                self._terminalize_stopped_case(execution, case)
            else:
                self._mark_case(
                    execution,
                    case,
                    "failed",
                    code="recovery_persistence_failed"
                    if execution.persistence_error is not None
                    else "missing_completion",
                    error=f"Durable checkpoint failed: {execution.persistence_error}"
                    if execution.persistence_error is not None
                    else "The case task ended without a terminal result",
                )

    @staticmethod
    def _recount_completeness(job: ExperimentJob) -> None:
        job.completeness.recount(job.cases, execution_job_id=job.job_id)

    @staticmethod
    def _finish_unstarted_analysis(
        job: ExperimentJob,
        *,
        status: Literal["failed", "cancelled"],
        error: str,
    ) -> None:
        if job.analysis.status not in {"pending", "running"}:
            return
        job.analysis.status = status
        job.analysis.error = error
        job.analysis.completed_at = now()

    async def _run_analysis(self, execution: _Execution) -> bool:
        job = execution.job
        request = execution.request
        if job.analysis.status == "not_requested":
            return False
        if request.analysis_callback is None:
            job.analysis.status = "failed"
            job.analysis.error = (
                "Attached analysis was requested but no analysis callback is wired"
            )
            job.analysis.completed_at = now()
            await self._persist_job(execution)
            return True

        await self._transition_job(execution, "analyzing")
        job.analysis.status = "running"
        job.analysis.started_at = now()
        await self._persist_job(execution)
        try:
            analysis_task = asyncio.ensure_future(request.analysis_callback(job))
            stop_wait = self.loop.create_task(execution.cancel_event.wait())
            done, _ = await asyncio.wait(
                {analysis_task, stop_wait},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_wait in done:
                analysis_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await analysis_task
                job.analysis.status = "cancelled"
                job.analysis.error = "Attached analysis cancelled"
                return True
            stop_wait.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stop_wait
            result = analysis_task.result()
            job.analysis.result = result
            job.analysis.status = "completed"
            return False
        except Exception as exc:
            job.analysis.status = "failed"
            job.analysis.error = str(exc)
            job.analysis.observations.append(
                {
                    "code": "analysis_failed",
                    "kind": "analysis",
                    "detail": str(exc),
                }
            )
            return True
        finally:
            job.analysis.completed_at = now()
            await self._persist_job(execution)

    async def cancel(
        self,
        job: ExperimentJob,
        *,
        control_token: str | None = None,
    ) -> list[dict[str, Any]]:
        """Stop further submissions and kill active cases when authorized."""
        if not experiment_store.cancel_authorized(job, control_token):
            raise CancelNotAuthorized(
                f"Cancellation is not authorized for experiment job {job.job_id}"
            )
        execution = self._executions.get(job.job_id)
        if execution is None:
            if finished(job):
                return []
            raise ExperimentCancellationError(
                f"Experiment job {job.job_id} is not owned by a live coordinator in this process"
            )
        # The record this execution runs, whichever copy the caller resolved.
        job = execution.job
        if execution.live.done:
            retained = [
                case.run_token for case in job.cases if case.case_id in execution.retained_slots
            ]
            if retained:
                await asyncio.gather(
                    *(self._kill_case(token) for token in retained),
                    return_exceptions=True,
                )
            return []
        # Claim the transitions before anything can suspend this coroutine.
        # Cancels of one job run on the one loop that owns its coordinator, so
        # the read and the claim below are indivisible with respect to every
        # other cancel of it — which is what makes "one transition, one
        # acknowledgement" hold for a case that is still running.
        before = {
            case.case_id: case.status
            for case in job.cases
            if case.status not in TERMINAL_CASE_STATUSES
            and case.case_id not in execution.claimed_cancel_priors
        }
        execution.claimed_cancel_priors.update(before)
        self._request_stop(execution, "cancelled")
        await execution.live.wait()
        return [
            cancel_receipt_row(case, before[case.case_id], case.status)
            for case in job.cases
            if case.case_id in before
        ]

    async def _kill_case(self, token: str) -> None:
        await asyncio.to_thread(self._kill_by_token, token, "experiment case")

    def _remove_case_artifacts(self, job: ExperimentJob, case: ExperimentCase) -> None:
        """Best-effort removal of a killed case's exact heavy-artifact paths."""
        if job.recovery is not None:
            return
        raw_extension = getattr(self.simulator_class, "raw_extension", ".raw")
        run_suffix = ".cir" if case.native_statistics else case.staged_deck.suffix or ".net"
        # The job's own run directory, which is also what the record persists —
        # the same reconstruction the crash reconciliation uses to FIND these.
        run_dir = job.output_folder or run_dir_in(self.output_folder, job.job_id)
        run_netlist = run_dir / f"{case.run_token}{run_suffix}"
        paths = {
            run_netlist,
            run_netlist.with_suffix(raw_extension),
        }
        if case.raw_file is not None:
            paths.add(case.raw_file)
        if case.log_file is not None:
            paths.add(case.log_file.with_suffix(raw_extension))
        for path in paths:
            with contextlib.suppress(OSError):
                path.unlink()


def _stop_bound(execution: _Execution, reason: StopReason) -> tuple[str, dict[str, Any]]:
    """Why the coordinator stopped a case, and the bound behind it as evidence."""
    if reason == "run_timeout":
        origin = (
            " (the server default)" if execution.run_timeout_source == "server_default" else ""
        )
        return f"the run timeout of {execution.run_timeout_s:g}s{origin} elapsed", {
            "run_timeout_s": execution.run_timeout_s,
            "run_timeout_source": execution.run_timeout_source,
        }
    if reason == "job_deadline":
        deadline = execution.request.job_deadline_s
        return f"the job deadline of {deadline:g}s elapsed", {"job_deadline_s": deadline}
    return "the job was cancelled", {}


def _stopped_run_evidence(limits: dict[str, Any], outcome: RunOutcome) -> dict[str, Any] | None:
    """What a stopped case keeps from the run the coordinator killed.

    The limit that applied, then whatever the outcome collector read off the
    killed run: its exit code, a cause the log names (``log_failure_code``,
    omitted when the collector fell back to ``execution_failed`` because the
    log named nothing), that cause's own evidence, and the log excerpt. The
    collector's ``error`` text is not kept: it describes a run that ended on
    its own ("no output generated"), which a killed run did not.
    """
    evidence: dict[str, Any] = {**limits, **(outcome.failure_evidence or {})}
    if outcome.failure_code and outcome.failure_code != "execution_failed":
        evidence["log_failure_code"] = outcome.failure_code
    if outcome.log_excerpt:
        evidence["log_excerpt"] = outcome.log_excerpt
    return evidence or None


def _raw_progress(raw: Path | None, dialect: str | None) -> tuple[dict[str, Any], str]:
    """How far a case's raw has got, as evidence fields and a phrase for a detail."""
    try:
        progress = read_partial_raw_progress(raw, dialect) if raw is not None else None
        present = raw is not None
    except FileNotFoundError:
        progress, present = None, False
    if progress is None:
        fields: dict[str, Any] = {
            "raw_present": present,
            "points": None if present else 0,
            "last_axis_value": None,
        }
        phrase = "a file at its raw path that is not a readable raw" if present else "no raw file"
        return fields, phrase
    fields = {
        "raw_present": True,
        "raw_bytes": progress.raw_bytes,
        "header_complete": progress.header_complete,
        "plot": progress.plot,
        "axis": progress.axis,
        "points": progress.points,
        "last_axis_value": progress.last_axis_value,
    }
    if progress.stepped:
        fields["stepped"] = True
    if progress.points is None:
        phrase = "a number of points that could not be counted"
    else:
        phrase = f"{progress.points} point{'s' if progress.points != 1 else ''}"
    if progress.axis is not None and progress.last_axis_value is not None:
        phrase += f", the last at {progress.axis} = {progress.last_axis_value:g}"
        if progress.stepped:
            phrase += " within the current .step"
    if progress.plot:
        phrase += f" of its {progress.plot}"
    return fields, phrase


def _progress_observation(
    case_id: str,
    run_index: int,
    raw: Path | None,
    dialect: str | None,
    *,
    code: Literal["partial_progress", "run_progress"],
    running_s: float | None = None,
) -> dict[str, Any] | None:
    """One case's progress fact, read from its raw.

    ``partial_progress`` for a case that was stopped, ``run_progress`` for one
    still running, which also carries ``running_s``. A case with no raw on
    disk says so rather than going quiet, and the detail names the case
    because a receipt merges every case's observations into one job-level
    list. A read that fails is logged and returns None: neither a case's
    outcome nor a status call may depend on it.
    """
    try:
        fields, reached = _raw_progress(raw, dialect)
    except Exception:
        logger.warning("could not read the raw of case %s", case_id, exc_info=True)
        return None
    evidence: dict[str, Any] = {"case_id": case_id, "run_index": run_index}
    if code == "partial_progress":
        detail = f"Case {case_id} was stopped after its simulator wrote {reached}."
    else:
        elapsed = (
            f"has been running for {running_s:.0f}s" if running_s is not None else "is running"
        )
        detail = f"Case {case_id} {elapsed}; its simulator has written {reached} so far."
        evidence["running_s"] = round(running_s, 1) if running_s is not None else None
    return {
        "code": code,
        "kind": "execution",
        "detail": detail,
        "evidence": {**evidence, **fields},
    }


async def live_run_progress(job: ExperimentJob) -> dict[str, dict[str, Any]]:
    """A ``run_progress`` observation for each case of ``job`` still running.

    Keyed by case id, for a receipt to attach to the cases that are still
    running when it is built. Read when someone asks for the job, never on a
    timer, so a job nobody watches costs nothing; each read is one raw's header
    and last record (the tail, for an ASCII raw), off the loop, for at most as
    many cases as the job has in flight. The raw's path comes from the record,
    so a job another process owns reads the same way.

    With no default run timeout this is what shows a stuck case: its point
    count stops moving from one read to the next while ``running_s`` grows.
    """
    moment = now()
    targets = [
        (
            case.case_id,
            case.run_index,
            experiment_store.case_raw_path(job, case),
            (moment - case.submitted_at).total_seconds() if case.submitted_at else None,
        )
        for case in job.cases
        if case.status in ACTIVE_CASE_STATUSES
    ]
    if not targets:
        return {}
    return await asyncio.to_thread(
        _read_live_progress, targets, dialect_for_simulator_name(job.simulator)
    )


def _read_live_progress(
    targets: list[tuple[str, int, Path | None, float | None]],
    dialect: str | None,
) -> dict[str, dict[str, Any]]:
    observations: dict[str, dict[str, Any]] = {}
    for case_id, run_index, raw, running_s in targets:
        observation = _progress_observation(
            case_id, run_index, raw, dialect, code="run_progress", running_s=running_s
        )
        if observation is not None:
            observations[case_id] = observation
    return observations
