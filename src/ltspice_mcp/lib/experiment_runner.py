"""Durable coordinator for expanded experiment cases."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.experiment_types import (
    TERMINAL_CASE_STATUSES,
    AnalysisStage,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    SourceRecord,
)
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.job_lifecycle import transition
from ltspice_mcp.lib.runner_base import DEFAULT_MAX_PARALLEL, RunnerBase, RunOutcome
from ltspice_mcp.lib.sweep_utils import generate_id

if TYPE_CHECKING:
    from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)

CANONICALIZER_VERSION = experiment_store.CANONICALIZER_VERSION
DEFAULT_KILL_GRACE_S = 10.0

AnalysisCallback = Callable[[ExperimentJob], Awaitable[dict[str, Any]]]


class IdempotencyConflictError(SimulationError):
    """A request id was reused for a different canonical payload."""


class ExperimentCancellationError(SimulationError):
    """An experiment could not be cancelled by this coordinator."""


def canonical_fingerprint(request_model: Any) -> str:
    """Hash the normalized, explicit-default JSON representation of a request."""
    if hasattr(request_model, "model_dump"):
        payload = request_model.model_dump(mode="json", exclude_unset=False)
    else:
        payload = request_model
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


@dataclass
class ExperimentRunRequest:
    """Prepared, fully staged input to the experiment coordinator."""

    state: SessionState
    request_id: str
    fingerprint: str
    cases: list[ExperimentCase]
    sources: list[SourceRecord]
    simulator: str
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
class _BarrierResult:
    job: ExperimentJob
    replayed: bool


@dataclass
class _Execution:
    request: ExperimentRunRequest
    job: ExperimentJob
    semaphore: asyncio.Semaphore
    capacity: int
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    stop_reason: Literal["cancelled", "job_deadline"] | None = None
    slots_held: set[str] = field(default_factory=set)
    retained_slots: set[str] = field(default_factory=set)
    futures: dict[str, asyncio.Future[RunOutcome]] = field(default_factory=dict)
    case_tasks: dict[str, asyncio.Task[None]] = field(default_factory=dict)
    deadline_task: asyncio.Task[None] | None = None
    case_event_count: int = 0


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

    def _materialize_job(self, request: ExperimentRunRequest) -> ExperimentJob:
        if not request.request_id:
            raise SimulationError("request_id is required for durable experiment submission")
        if request.canonicalizer_version != CANONICALIZER_VERSION:
            raise IdempotencyConflictError(
                "Unsupported request canonicalizer version "
                f"{request.canonicalizer_version}; this server uses {CANONICALIZER_VERSION}"
            )
        capacity = request.max_parallel if request.max_parallel is not None else self.max_parallel
        if capacity < 1:
            raise SimulationError("max_parallel must be at least 1")
        job_id = request.job_id or generate_id("exp")
        experiment_store.validate_job_id(job_id)
        control_token = secrets.token_urlsafe(32)
        store_path = experiment_store.record_path(job_id, request.state.working_dir)
        case_ids = [case.case_id for case in request.cases]
        run_indices = [case.run_index for case in request.cases]
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise SimulationError("Experiment case_id values must be non-empty and unique")
        if any(run_index < 0 for run_index in run_indices) or len(set(run_indices)) != len(
            run_indices
        ):
            raise SimulationError("Experiment run_index values must be non-negative and unique")
        for case in request.cases:
            case.run_token = f"{job_id}_case_{case.run_index}"
        completeness = Completeness(
            declared=request.declared if request.declared is not None else len(request.cases),
            expanded=len(request.cases),
        )
        completeness.recount(request.cases)
        failures = [
            {
                "case_id": case.case_id,
                "code": case.failure_code or case.status,
                "message": case.error or case.status,
            }
            for case in request.cases
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
            cases=request.cases,
            sources=request.sources,
            simulator=request.simulator,
            completeness=completeness,
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
            candidate = self._materialize_job(request)
            barrier = await asyncio.to_thread(self._durable_barrier, request, candidate)

            # Registration and execution-task creation intentionally have no
            # await between them. A replay racing the original barrier can
            # therefore never observe a durable job that this process has not
            # either registered or recognized as already registered.
            registered = request.state.all_jobs.get(barrier.job.job_id)
            if isinstance(registered, ExperimentJob):
                job = registered
            else:
                job = barrier.job
                request.state.add_experiment_job(job, already_persisted=True)
            execution = None
            if not barrier.replayed and job.task is None:
                execution = self._new_execution(request, job)
                self._executions[job.job_id] = execution
            if barrier.replayed and not any(
                item.get("code") == "idempotent_replay" for item in job.observations
            ):
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
            receipt = ExperimentReceipt(
                job=job,
                replayed=barrier.replayed,
                control_token=job.control_token,
            )
            if not receipt_ready.done():
                receipt_ready.set_result(receipt)
            if execution is not None:
                job.task = self.loop.create_task(self._run_job(execution))
        except Exception as exc:
            if not receipt_ready.done():
                receipt_ready.set_exception(exc)

    @staticmethod
    def _durable_barrier(
        request: ExperimentRunRequest,
        candidate: ExperimentJob,
    ) -> _BarrierResult:
        """One blocking lock/lookup/write critical section for submission."""
        working_dir = request.state.working_dir
        with file_lock(experiment_store.request_lock_target(request.request_id, working_dir)):
            index = experiment_store.load_request_index(request.request_id, working_dir)
            dangling = False
            if index is not None:
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
                if existing is not None:
                    if (
                        existing.request_id != request.request_id
                        or existing.fingerprint != request.fingerprint
                        or existing.canonicalizer_version != request.canonicalizer_version
                    ):
                        raise IdempotencyConflictError(
                            f"request_id {request.request_id!r} points to an "
                            "inconsistent coordinator record"
                        )
                    if any(
                        item.get("code") == "server_restarted" for item in existing.observations
                    ):
                        experiment_store.save_job(existing)
                    return _BarrierResult(existing, replayed=True)
                dangling = True

            if dangling:
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
            # The request is the single source of the idempotency identity:
            # stamping it here means no caller can persist a coordinator
            # record whose identity disagrees with its own request index.
            candidate.request_id = request.request_id
            candidate.fingerprint = request.fingerprint
            candidate.canonicalizer_version = request.canonicalizer_version
            experiment_store.save_request_index(
                request_id=request.request_id,
                fingerprint=request.fingerprint,
                canonicalizer_version=request.canonicalizer_version,
                job_id=candidate.job_id,
                working_dir=working_dir,
            )
            experiment_store.save_job(candidate)

        try:
            experiment_store.save_pointers(candidate)
        except OSError as exc:
            candidate.observations.append(
                {
                    "code": "experiment_pointer_write_failed",
                    "kind": "persistence",
                    "detail": (
                        "The coordinator is durable, but one or more circuit "
                        f"pointers could not be written: {exc}"
                    ),
                }
            )
            experiment_store.save_job(candidate)
        return _BarrierResult(candidate, replayed=False)

    def _new_execution(
        self,
        request: ExperimentRunRequest,
        job: ExperimentJob,
    ) -> _Execution:
        capacity = request.max_parallel if request.max_parallel is not None else self.max_parallel
        return _Execution(
            request=request,
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
                or job.completeness.failed
                or job.completeness.cancelled
                or job.completeness.skipped
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
            if not execution.retained_slots:
                self._executions.pop(job.job_id, None)

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

    async def _run_case(self, execution: _Execution, case: ExperimentCase) -> None:
        acquired = False
        try:
            await execution.semaphore.acquire()
            acquired = True
            execution.slots_held.add(case.case_id)
            if case.status != "queued" or execution.cancel_event.is_set():
                if case.status == "queued":
                    self._terminalize_stopped_case(execution, case)
                self._release_slot(execution, case.case_id)
                return

            case.status = "submitted"
            case.submitted_at = now()
            self._checkpoint_case_transition(execution)
            future: asyncio.Future[RunOutcome] = self.loop.create_future()
            execution.futures[case.case_id] = future
            suffix = case.staged_deck.suffix or ".net"
            try:
                await asyncio.to_thread(
                    self.submit_netlist,
                    case.staged_deck,
                    f"{case.run_token}{suffix}",
                    lambda outcome: self._handle_case_completion(
                        execution.job.job_id,
                        case.case_id,
                        outcome,
                    ),
                )
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
                        code="execution_failed",
                        error=outcome.error,
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
        await asyncio.to_thread(self._remove_case_artifacts, case)
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
        self.loop.run_in_executor(None, self._remove_case_artifacts, case)
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
    ) -> None:
        if case.status in TERMINAL_CASE_STATUSES:
            return
        case.status = status
        case.failure_code = code
        case.error = error
        case.completed_at = now()
        if status in {"failed", "cancelled", "skipped"}:
            execution.job.failures.append(
                {
                    "case_id": case.case_id,
                    "code": code or status,
                    "message": error or status,
                }
            )
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
    ) -> None:
        if execution.stop_reason is None:
            execution.stop_reason = reason
        execution.cancel_event.set()
        for case in execution.job.cases:
            if case.status != "queued":
                continue
            self._terminalize_stopped_case(execution, case)
            task = execution.case_tasks.get(case.case_id)
            if task is not None:
                task.cancel()

    def _terminalize_stopped_case(
        self,
        execution: _Execution,
        case: ExperimentCase,
    ) -> None:
        if execution.stop_reason == "job_deadline":
            self._mark_case(
                execution,
                case,
                "failed",
                code="job_deadline",
                error="The experiment deadline elapsed before this case was submitted",
            )
        else:
            self._mark_case(
                execution,
                case,
                "cancelled",
                code="cancelled",
                error="Cancelled before submission",
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
        authorized = job.owner_pid == os.getpid() or (
            control_token is not None
            and bool(job.control_token)
            and secrets.compare_digest(control_token, job.control_token)
        )
        if not authorized:
            raise ExperimentCancellationError(
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
        before = {case.case_id: case.status for case in job.cases}
        self._request_stop(execution, "cancelled")
        await job.done_event.wait()
        return [
            {
                "case_id": case.case_id,
                "prior_status": before[case.case_id],
                "status": case.status,
            }
            for case in job.cases
            if before[case.case_id] not in TERMINAL_CASE_STATUSES
        ]

    async def _kill_case(self, token: str) -> None:
        await asyncio.to_thread(self._kill_by_token, token, "experiment case")

    def _remove_case_artifacts(self, case: ExperimentCase) -> None:
        """Best-effort removal of a killed case's exact heavy-artifact paths."""
        raw_extension = getattr(self.simulator_class, "raw_extension", ".raw")
        run_suffix = case.staged_deck.suffix or ".net"
        run_netlist = self.output_folder / f"{case.run_token}{run_suffix}"
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
