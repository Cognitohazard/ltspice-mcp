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
from ltspice_mcp.lib.deck_staging import verify_staged_manifest
from ltspice_mcp.lib.experiment_types import (
    TERMINAL_CASE_STATUSES,
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
    failure_row,
)
from ltspice_mcp.lib.filelock import async_file_lock, file_lock
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.runner_base import (
    DEFAULT_MAX_PARALLEL,
    RunnerBase,
    RunOutcome,
    discard_generated_netlist,
    inject_logopinfo,
    inject_ngspice_control_write,
)
from ltspice_mcp.lib.store import Store, run_dir_in, run_filename_in, validate_job_id
from ltspice_mcp.lib.sweep_utils import generate_id

if TYPE_CHECKING:
    from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

CANONICALIZER_VERSION = experiment_store.CANONICALIZER_VERSION
DEFAULT_KILL_GRACE_S = 10.0

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

#: The durable replay note, written on the job record once and read by
#: everyone who looks at the job afterwards — the original submitter included,
#: which is why it states what happened to the record rather than making a
#: claim about the reader's own call.
REPLAY_RECORD_DETAIL = (
    "A later call carrying this request_id was answered from this record; "
    "no cases were resubmitted for it."
)

# How each staging-time drift observation reads when it blocks a replay
# instead of annotating a fresh stage.
_DRIFT_REASONS = {
    "source_modified_after_staging": "content changed",
    "source_unavailable_after_staging": "no longer readable",
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

    def __init__(self, request_id: str, cause: BaseException) -> None:
        super().__init__(
            f"The submission for request_id {request_id!r} is recorded, but the "
            f"call could not be completed: {cause}. Ask again with the same "
            "request_id to read back what was committed."
        )


class ExperimentCancellationError(SimulationError):
    """An experiment could not be cancelled by this coordinator."""

    code = "cancel_failed"


@contextlib.asynccontextmanager
async def _request_gate(gate: Path, request_id: str) -> AsyncIterator[None]:
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


def _already_staged() -> StagedDecks:
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
    declared: int | None = None
    canonicalizer_version: int = CANONICALIZER_VERSION
    max_parallel: int | None = None
    run_timeout_s: float | None = None
    job_deadline_s: float | None = None
    kill_grace_s: float = DEFAULT_KILL_GRACE_S
    analysis_request: dict[str, Any] | None = None
    analysis_callback: AnalysisCallback | None = None


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
class _BarrierResult:
    job: ExperimentJob
    replayed: bool


@dataclass(frozen=True)
class _IndexLookup:
    """What the request index said: the job to replay, or that it named none."""

    existing: ExperimentJob | None
    dangling: bool


@dataclass
class _Execution:
    request: ExperimentRunRequest
    job: ExperimentJob
    semaphore: asyncio.Semaphore
    capacity: int
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
        self._pipeline_tasks: set[asyncio.Task[None]] = set()

    def has_active_work(self) -> bool:
        """Whether a submission pipeline or execution still owns live work."""
        return bool(self._pipeline_tasks or self._executions)

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
        task = self.loop.create_task(self._submission_pipeline(request, receipt_ready))
        self._pipeline_tasks.add(task)
        task.add_done_callback(self._pipeline_tasks.discard)
        return receipt_ready

    def _validate_request(self, request: ExperimentRunRequest) -> None:
        """Check what can be checked before a single deck is copied."""
        if not request.request_id:
            raise SimulationError("request_id is required for durable experiment submission")
        if request.canonicalizer_version != CANONICALIZER_VERSION:
            raise IdempotencyConflictError(
                "Unsupported request canonicalizer version "
                f"{request.canonicalizer_version}; this server uses {CANONICALIZER_VERSION}"
            )
        if self._case_capacity(request) < 1:
            raise SimulationError("max_parallel must be at least 1")

    def _materialize_job(
        self, request: ExperimentRunRequest, staged: StagedDecks
    ) -> ExperimentJob:
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
                await self._start_committed(request, barrier, receipt_ready)
            except Exception as exc:
                raise SubmissionCommitted(request.request_id, exc) from exc
        except Exception as exc:
            if not receipt_ready.done():
                receipt_ready.set_exception(exc)

    async def _start_committed(
        self,
        request: ExperimentRunRequest,
        barrier: _BarrierResult,
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
        registered = request.state.all_jobs.get(barrier.job.job_id)
        if registered is not None:
            job = registered
        else:
            job = barrier.job
            request.state.add_experiment_job(job, already_persisted=True)
        execution = None
        if not barrier.replayed and job.task is None:
            execution = self._new_execution(request, job)
            self._executions[job.job_id] = execution
        if barrier.replayed:
            experiment_store.note_once(
                job.observations,
                {
                    "code": "idempotent_replay",
                    "kind": "submission",
                    "detail": REPLAY_RECORD_DETAIL,
                },
            )
        receipt = ExperimentReceipt(
            job=job,
            replayed=barrier.replayed,
            control_token=job.control_token,
        )
        if not receipt_ready.done():
            receipt_ready.set_result(receipt)
        if execution is not None:
            job.task = self.loop.create_task(self._run_job(execution))

    async def _durable_barrier(self, request: ExperimentRunRequest) -> _BarrierResult:
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
        working_dir = request.state.working_dir
        gate = Store(working_dir).request_lock(request.request_id)
        async with _request_gate(gate, request.request_id):
            lookup = await asyncio.to_thread(self._read_request_index, request)
            if lookup.existing is not None:
                return _BarrierResult(lookup.existing, replayed=True)
            try:
                staged = await request.stage()
                candidate = self._materialize_job(request, staged)
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
                await asyncio.to_thread(self._claim_request_id, request, candidate)
            except Exception:
                # Staged, then refused. The decks are already copied and the
                # claim never landed, so this is the one window in which a run
                # tree exists that no record will ever name.
                await asyncio.to_thread(self._discard_staged_run_dir, request, working_dir)
                raise
        # Outside the gate: the per-circuit index is discovery, not identity,
        # so a slow directory here holds up nothing but this submission.
        try:
            await asyncio.to_thread(self._register_circuits, candidate, working_dir)
        except Exception as exc:
            raise SubmissionCommitted(request.request_id, exc) from exc
        return _BarrierResult(candidate, replayed=False)

    def _discard_staged_run_dir(self, request: ExperimentRunRequest, working_dir: Path) -> None:
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
    def _read_request_index(request: ExperimentRunRequest) -> _IndexLookup:
        """The gate's lookup half: a replay, a conflict, or nothing recorded yet."""
        working_dir = request.state.working_dir
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
        verify_replay_sources(existing, request.request_id)
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

    def _case_capacity(self, request: ExperimentRunRequest) -> int:
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
        job: ExperimentJob,
    ) -> _Execution:
        capacity = self._case_capacity(request)
        return _Execution(
            # Without the staging closure: this execution outlives the
            # submission call, and the closure holds that whole scope.
            request=replace(request, stage=_already_staged),
            job=job,
            semaphore=asyncio.Semaphore(capacity),
            capacity=capacity,
        )

    async def wait(
        self,
        job: ExperimentJob,
        timeout_s: float | None = None,
        *,
        wait_for: Literal["all", "runs"] = "all",
    ) -> bool:
        """Wait for full terminality or run terminality without mutating the job."""
        event = job.done_event if wait_for == "all" else job.runs_done_event
        if timeout_s is None:
            await event.wait()
            return True
        try:
            await asyncio.wait_for(event.wait(), timeout_s)
        except TimeoutError:
            return False
        return True

    async def _run_job(self, execution: _Execution) -> None:
        job = execution.job
        request = execution.request
        try:
            execution.external_cancel_task = self.loop.create_task(
                self._external_cancel_watch(execution)
            )
            transition(job, "running", state=request.state, total_cases=len(job.cases))
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
            job.runs_done_event.set()
            request.state.persist_job(job)

            if execution.stop_reason == "cancelled":
                self._finish_unstarted_analysis(
                    job,
                    status="cancelled",
                    error="Experiment cancelled before attached analysis started",
                )
                if job.status not in {"cancelled", "failed"}:
                    transition(job, "cancelled", state=request.state)
                return
            if execution.stop_reason == "job_deadline":
                self._finish_unstarted_analysis(
                    job,
                    status="cancelled",
                    error="Experiment deadline elapsed before attached analysis started",
                )
                transition(job, "completed_with_failures", state=request.state)
                return

            analysis_failed = await self._run_analysis(execution)
            if execution.stop_reason == "cancelled":
                transition(job, "cancelled", state=request.state)
            elif (
                execution.stop_reason == "job_deadline"
                or analysis_failed
                or job.completeness.fell_short
            ):
                transition(job, "completed_with_failures", state=request.state)
            else:
                transition(job, "completed", state=request.state)
        except Exception as exc:
            logger.exception("Experiment job %s failed", job.job_id)
            job.error = str(exc)
            self._reconcile_unfinished_cases(execution)
            self._recount_completeness(job)
            job.runs_done_event.set()
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
                transition(job, "failed", state=request.state, error=job.error)
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
        while not execution.job.done_event.is_set():
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
                execution.request.state.persist_job(execution.job)
                return
            await asyncio.sleep(0.5)

    async def _deadline_watch(self, execution: _Execution, deadline_s: float) -> None:
        if deadline_s <= 0:
            await asyncio.sleep(0)
        else:
            await asyncio.sleep(deadline_s)
        if execution.job.done_event.is_set():
            return
        execution.job.observations.append(
            {
                "code": "job_deadline",
                "kind": "execution",
                "detail": f"The experiment deadline of {deadline_s:g}s elapsed.",
            }
        )
        self._request_stop(execution, "job_deadline")

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
        with file_lock(Store(working_dir).cancellation_lock(job_id)):
            if experiment_store.cancellation_requested(job_id, working_dir):
                return False
        with execution.launch_lock:
            if execution.cancel_event.is_set():
                return False
            case.status = "submitted"
            case.submitted_at = now()
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
        run_dir = run_dir_in(self.output_folder, job_id)
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
        try:
            self.submit_netlist(
                run_deck,
                # A sub-path, not a bare name: the simulator layer joins it onto
                # the runner's output folder, so this is what lands the run's
                # deck copy, raw and log in the job's own directory without the
                # runner (and its shared concurrency semaphore) ever moving.
                run_filename_in(job_id, f"{case.run_token}{suffix}"),
                lambda outcome: self._handle_case_completion(
                    execution.job.job_id,
                    case.case_id,
                    outcome,
                ),
            )
        finally:
            # spicelib stages the deck synchronously inside run(), so the copy
            # has done its job by the time submit returns — and on a submit that
            # raised, nothing will ever read it. The marker guard inside the
            # helper makes this incapable of touching the staged deck itself.
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

    async def _await_case(
        self,
        execution: _Execution,
        future: asyncio.Future[RunOutcome],
    ) -> tuple[RunOutcome | None, Literal["cancelled", "job_deadline", "run_timeout"] | None]:
        stop_wait = self.loop.create_task(execution.cancel_event.wait())
        completion_wait = asyncio.shield(future)
        try:
            done, _ = await asyncio.wait(
                {completion_wait, stop_wait},
                timeout=execution.request.run_timeout_s,
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
        reason: Literal["cancelled", "job_deadline", "run_timeout"],
    ) -> None:
        try:
            await self._kill_case(case.run_token)
        except Exception as exc:
            case.observations.append(
                {
                    "code": "kill_attempt_failed",
                    "kind": "execution",
                    "detail": f"Scoped simulator termination raised an error: {exc}",
                }
            )
        try:
            outcome = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=execution.request.kill_grace_s,
            )
        except TimeoutError:
            terminal_status = "cancelled" if reason == "cancelled" else "failed"
            self._mark_case(
                execution,
                case,
                terminal_status,
                code="kill_unconfirmed",
                error=(
                    f"{reason.replace('_', ' ')} elapsed and simulator exit was not "
                    f"confirmed within {execution.request.kill_grace_s:g}s"
                ),
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
            execution.request.state.persist_job(execution.job)
            self._fail_if_capacity_retained(execution)
            return

        self._apply_stopped_outcome(case, outcome)
        terminal_status = "cancelled" if reason == "cancelled" else "failed"
        self._mark_case(
            execution,
            case,
            terminal_status,
            code=reason,
            error=f"Case stopped because {reason.replace('_', ' ')} elapsed",
        )
        await asyncio.to_thread(self._remove_case_artifacts, execution.job, case)
        self._release_slot(execution, case.case_id)

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
        execution.request.state.persist_job(execution.job)
        self.loop.run_in_executor(None, self._remove_case_artifacts, execution.job, case)
        self._release_slot(execution, case_id)
        if execution.job.done_event.is_set() and not execution.retained_slots:
            self._executions.pop(job_id, None)

    @staticmethod
    def _apply_outcome(case: ExperimentCase, outcome: RunOutcome) -> None:
        case.raw_file = Path(outcome.raw_file) if outcome.raw_file else None
        case.log_file = Path(outcome.log_file) if outcome.log_file else None
        if outcome.observations:
            case.observations.extend(outcome.observations)

    @staticmethod
    def _apply_stopped_outcome(case: ExperimentCase, outcome: RunOutcome) -> None:
        """Keep post-kill diagnostics without advertising artifacts we remove."""
        case.raw_file = None
        case.log_file = Path(outcome.log_file) if outcome.log_file else None
        if outcome.observations:
            case.observations.extend(outcome.observations)

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
        if execution.case_event_count % step == 0:
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
                    code="missing_completion",
                    error="The case task ended without a terminal result",
                )

    @staticmethod
    def _recount_completeness(job: ExperimentJob) -> None:
        job.completeness.recount(job.cases)

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
            request.state.persist_job(job)
            return True

        transition(job, "analyzing", state=request.state)
        job.analysis.status = "running"
        job.analysis.started_at = now()
        request.state.persist_job(job)
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
            request.state.persist_job(job)

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
            if job.done_event.is_set():
                return []
            raise ExperimentCancellationError(
                f"Experiment job {job.job_id} is not owned by a live coordinator in this process"
            )
        if job.done_event.is_set():
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
        await job.done_event.wait()
        return [
            {
                "case_id": case.case_id,
                "prior_status": before[case.case_id],
                "status": case.status,
            }
            for case in job.cases
            if case.case_id in before
        ]

    async def _kill_case(self, token: str) -> None:
        await asyncio.to_thread(self._kill_by_token, token, "experiment case")

    def _remove_case_artifacts(self, job: ExperimentJob, case: ExperimentCase) -> None:
        """Best-effort removal of a killed case's exact heavy-artifact paths."""
        raw_extension = getattr(self.simulator_class, "raw_extension", ".raw")
        run_suffix = case.staged_deck.suffix or ".net"
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
