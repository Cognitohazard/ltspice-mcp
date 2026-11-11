"""Declarative state machine for simulation / batch-job lifecycle.

Every status write in production code now goes through ``transition()``
(or ``reconcile_experiment_restart()`` for a record whose owner died). The
transition tables below are the single source of truth for which status
changes are legal and which lifecycle event fires on each one.

Rationale: previously, 13 call sites directly mutated ``job.status`` and
separately called ``emit_job_event`` — two concerns, scattered. That
made it possible to change status without emitting, to emit twice, or
to transition into an invalid state (e.g. completed → running). The
chokepoint below closes all three gaps.

Registration events (``submitted``) and discovery events
(``interrupted_recovered`` when the status doesn't change on load) are
emitted directly by ``JobRegistry`` — they aren't transitions. The
state machine only covers actual status changes.

A job record is data. What a waiter waits on belongs to :class:`LiveJob`,
which exists only for a job this process admitted and runs: a record read
back from the store, this process's or another's, has nothing to wait on.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import (
    OWNER_STOPPED,
    TERMINAL_CASE_STATUSES,
    ExperimentJob,
    failure_row,
)
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES
from ltspice_mcp.lib.observability import JobEvent, emit_job_event

if TYPE_CHECKING:
    from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


VALID_EXPERIMENT_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset(
        {
            "running",
            "completed",
            "completed_with_failures",
            "failed",
            "cancelled",
            "interrupted",
        }
    ),
    "running": frozenset(
        {
            "analyzing",
            "completed",
            "completed_with_failures",
            "failed",
            "cancelled",
            "interrupted",
        }
    ),
    "analyzing": frozenset(
        {"completed", "completed_with_failures", "failed", "cancelled", "interrupted"}
    ),
    "completed": frozenset(),
    "completed_with_failures": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}


def runs_terminal(status: str) -> bool:
    """Have an experiment's runs all reached terminality at this status?

    True for every terminal status, and also for ``analyzing``: the coordinator
    validates completeness and marks the runs done BEFORE transitioning
    there, and the table above lets nothing but a terminal status follow it — so
    ``analyzing`` means every run is done and only the attached analysis is
    still in flight. That is what lets an experiment's own attached analysis
    read its own produced cases.

    Lives beside the transition table because the table is what makes it true.
    Every caller asking "are the runs done?" from a status reads this, so the
    equivalence is stated once rather than re-derived per module.
    """
    return status in TERMINAL_STATUSES or status == "analyzing"


WaitFor = Literal["all", "runs"]


def runs_done(job: ExperimentJob) -> bool:
    """Whether every run of ``job`` has finished, as its record reads now.

    True at a status that guarantees it (``runs_terminal``), and also once every
    case has a terminal status: the coordinator records the last case before it
    changes the job's status, and another process reads that record in between.
    A terminal case is never changed again, so nothing a reader takes from it
    can move under it.

    The one answer to "can this job's runs be read now?": ``jobs(wait,
    wait_for="runs")`` returns on it, and every reader of a job's runs
    (``services.runs_unready``) refuses until it holds, so a wait that returns
    is never followed by a refusal.
    """
    return runs_terminal(job.status) or all(
        case.status in TERMINAL_CASE_STATUSES for case in job.cases
    )


def finished(job: ExperimentJob, wait_for: WaitFor = "all") -> bool:
    """Whether a job's record says it is done, or with ``"runs"``, its runs are.

    What a record can answer with nothing live behind it: a job another process
    runs, or one no process runs any more. Runs are done when ``runs_done``
    says so.
    """
    if wait_for == "runs":
        return runs_done(job)
    return job.status in TERMINAL_STATUSES


# ---------------------------------------------------------------------------
# A job whose owning process stopped
# ---------------------------------------------------------------------------
#
# The process running a job can stop before the job is done in three ways: it
# is killed, and the next reader of the record finds it gone
# (``experiment_store``); the server shuts down; or a script closes its ``Api``,
# which shuts its session down the same way (``JobRegistry.cancel_running``,
# and the coordinator it asks to stop, ``ExperimentRunner``). The job, its
# cases and its attached analysis say the same thing whichever it was, so the
# words are kept here once and say "the process", which is true of all three.

#: The detail of the lifecycle observation a job, and its attached analysis,
#: records when the process running it stopped before it finished.
OWNER_STOPPED_DETAIL = "The process running the job stopped before the job finished."
#: The message of a case that process left unfinished.
OWNER_STOPPED_CASE_ERROR = (
    "The process running the job stopped before this case finished; keep that "
    "process running until the job finishes"
)
#: The error of an attached analysis that process left unfinished.
OWNER_STOPPED_ANALYSIS_ERROR = (
    "The process running the job stopped before the attached analysis finished"
)


def owner_stopped_observation() -> dict[str, str]:
    """The lifecycle observation of a job whose owning process stopped."""
    return {"code": OWNER_STOPPED, "kind": "lifecycle", "detail": OWNER_STOPPED_DETAIL}


def stop_analysis_for_owner(job: ExperimentJob) -> None:
    """Fail an attached analysis the stopped process left pending or running.

    Failed, not cancelled: nobody asked for it to stop, and an analysis is
    never re-run on its own, so the receipt routes the caller to
    ``analyze_results`` over the runs, as for any failed analysis.
    """
    if job.analysis.status not in {"pending", "running"}:
        return
    job.analysis = replace(
        job.analysis,
        status="failed",
        error=OWNER_STOPPED_ANALYSIS_ERROR,
        completed_at=now(),
        observations=[*job.analysis.observations, owner_stopped_observation()],
    )


def stop_for_owner(job: ExperimentJob, *, runs_were_finished: bool) -> str:
    """Settle ``job``, whose owning process stopped, and return its terminal status.

    The one answer to "what does a job whose process stopped before it
    finished say?", for every way the process stops. The job records the
    lifecycle observation; an attached analysis left pending or running fails
    (``stop_analysis_for_owner``); every case not yet terminal fails with
    ``OWNER_STOPPED``, the code a default resume retries. A case is changed in
    place, so a coordinator still holding it sees it terminal and leaves it.

    ``runs_were_finished`` is whether every run was done when the process
    stopped (``finished(job, "runs")`` read then). With runs left the job is
    ``interrupted``. With none, the runs earned a completed status and the
    stop took only the analysis: ``completed_with_failures`` when a run fell
    short or the analysis did not complete, else ``completed``.

    Settling a job already settled changes nothing on it, and with the same
    ``runs_were_finished`` gives the same status: a shutdown can reach a job
    both through its coordinator and through its own bookkeeping
    (``JobRegistry.cancel_running``), and the job is stopped once. A job this
    session runs is settled through ``LiveJob.stop_for_owner``, which gives
    both the same ``runs_were_finished``.
    """
    if owner_stopped_observation() not in job.observations:
        job.observations.append(owner_stopped_observation())
    stop_analysis_for_owner(job)
    stopped = [case for case in job.cases if case.status not in TERMINAL_CASE_STATUSES]
    stopped_at = now()
    for case in stopped:
        case.status = "failed"
        case.failure_code = OWNER_STOPPED
        case.error = OWNER_STOPPED_CASE_ERROR
        case.completed_at = stopped_at
    job.failures.extend(failure_row(case) for case in stopped)
    job.completeness.recount(job.cases, execution_job_id=job.job_id)
    if not runs_were_finished:
        return "interrupted"
    if job.completeness.fell_short or job.analysis.status in {"failed", "cancelled"}:
        return "completed_with_failures"
    return "completed"


# Which event name fires when a job enters a given status. Every entry must be
# a status some transition actually reaches — an unreachable one reads as a
# supported outcome nothing can produce. ('timeout' left with the job type that
# could reach it; a case that runs out of time is recorded as failed.)
STATUS_TO_EVENT: dict[str, JobEvent] = {
    "running": "started",
    "analyzing": "analyzing",
    "completed": "completed",
    "completed_with_failures": "completed_with_failures",
    "failed": "failed",
    "cancelled": "cancelled",
    "interrupted": "interrupted",
}


class InvalidTransitionError(ValueError):
    """Raised when code attempts a status change not in the transition table."""


def _apply(
    job: ExperimentJob,
    new_status: str,
    valid: dict[str, frozenset[str]],
) -> None:
    """Validate and apply a status change; set completed_at on terminal
    transitions.

    Same-status calls are rejected to surface double-emit bugs; callers
    that want idempotency should guard on ``job.status`` themselves.
    """
    old = job.status
    if old == new_status:
        raise InvalidTransitionError(
            f"no-op transition {old} → {new_status} for job {job.job_id}; "
            f"caller should guard if this is reachable"
        )
    allowed = valid.get(old, frozenset())
    if new_status not in allowed:
        raise InvalidTransitionError(
            f"illegal transition {old} → {new_status} for job {job.job_id}; "
            f"allowed from {old}: {sorted(allowed) or '[terminal]'}"
        )
    job.status = new_status  # type: ignore[assignment]
    if new_status in TERMINAL_STATUSES:
        job.completed_at = now()


def transition(
    job: ExperimentJob,
    new_status: str,
    *,
    state: SessionState | None = None,
    **event_extra: Any,
) -> None:
    """Transition ``job`` to ``new_status``, persist, and emit its event.

    The lifecycle event name is looked up from ``STATUS_TO_EVENT[new_status]``
    so every legal transition emits exactly one event with a consistent
    name. Additional keyword args flow through to the event payload.

    Raises ``InvalidTransitionError`` for same-status or out-of-table
    transitions.
    """
    valid = VALID_EXPERIMENT_TRANSITIONS
    event = STATUS_TO_EVENT.get(new_status)
    if event is None and new_status in valid.get(job.status, frozenset()):
        raise InvalidTransitionError(f"status {new_status!r} has no event mapping")
    _apply(job, new_status, valid)
    if state is not None:
        state.persist_job(job)
    if event is None:  # Defensive: every valid runtime target must have a mapping.
        raise InvalidTransitionError(
            f"no event mapping for status {new_status!r}; update STATUS_TO_EVENT"
        )
    emit_job_event(event, job, **event_extra)


def reconcile_experiment_restart(
    job: ExperimentJob,
    new_status: str,
) -> None:
    """Apply the terminal status inferred while loading an abandoned experiment.

    Restart reconciliation is persistence recovery, not a fresh runtime event.
    The registry emits the discovery event after it installs the loaded job.
    """
    if new_status not in {"interrupted", "completed", "completed_with_failures"}:
        raise InvalidTransitionError(f"invalid experiment restart outcome {new_status!r}")
    _apply(job, new_status, VALID_EXPERIMENT_TRANSITIONS)


@dataclass(eq=False)
class LiveJob:
    """A job this process admitted and runs: its record, and what waiters wait on.

    Only the registry makes one (``JobRegistry.reserve``), for a job this
    process admitted, and the coordinator running the job signals it. A record
    the store reads back has none, so nothing can wait on a copy that nothing
    will ever signal. Every status change of a live job goes through
    :meth:`transition`, which keeps the signals in step with the record.
    """

    job: ExperimentJob
    #: The coordinator running the job, once one has started.
    task: asyncio.Task[None] | None = None
    _runs_done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _done: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    #: Whether every run was done when the stop of this process reached them,
    #: once something has asked (``runs_done_at_owner_stop``).
    _runs_done_at_owner_stop: bool | None = field(default=None, repr=False)

    @property
    def done(self) -> bool:
        """Whether the job has reached a terminal status."""
        return self._done.is_set()

    @property
    def coordinator(self) -> asyncio.Task[None] | None:
        """The coordinator running the job now, if one is."""
        return self.task if self.task is not None and not self.task.done() else None

    @property
    def runs_done(self) -> bool:
        """Whether every run has finished; an attached analysis may still be running."""
        return self._runs_done.is_set()

    def mark_runs_done(self) -> None:
        """Record that every case is terminal and completeness is reconciled."""
        self._runs_done.set()

    def runs_done_at_owner_stop(self) -> bool:
        """Whether every run was done when the stop of this process reached them.

        Read from the record the first time it is asked and kept, so the job
        has one answer however the stop reaches it: the coordinator asks as
        the stop reaches it (``ExperimentRunner.cancel``), before it stops a
        run, and the shutdown's bookkeeping, which settles a job whose
        coordinator did not finish within the bound, then takes the same
        answer; if the stop never reached a coordinator, the bookkeeping asks
        as it stops the runs itself. A case that finishes in between is never
        read one way by one and the other way by the other.
        """
        if self._runs_done_at_owner_stop is None:
            self._runs_done_at_owner_stop = finished(self.job, "runs")
        return self._runs_done_at_owner_stop

    def stop_for_owner(self) -> str:
        """Settle the job because this process is stopping; return its status.

        ``stop_for_owner`` with the runs as the stop found them
        (``runs_done_at_owner_stop``). The coordinator and the shutdown's
        bookkeeping both settle through here, so the second to arrive changes
        nothing and gets the status the first got.
        """
        return stop_for_owner(self.job, runs_were_finished=self.runs_done_at_owner_stop())

    def transition(
        self, new_status: str, *, state: SessionState | None = None, **event_extra: Any
    ) -> None:
        """``transition()`` the record, then release whoever a terminal status frees."""
        transition(self.job, new_status, state=state, **event_extra)
        if new_status in TERMINAL_STATUSES:
            self._runs_done.set()
            self._done.set()

    async def wait(self, timeout_s: float | None = None, *, wait_for: WaitFor = "all") -> bool:
        """Wait until the job is done, or its runs are; False if ``timeout_s`` ran out."""
        event = self._done if wait_for == "all" else self._runs_done
        if event.is_set():
            return True
        if timeout_s is None:
            await event.wait()
            return True
        try:
            await asyncio.wait_for(event.wait(), timeout_s)
        except TimeoutError:
            return False
        return True
