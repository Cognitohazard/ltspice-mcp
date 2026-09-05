"""Declarative state machine for simulation / batch-job lifecycle.

Every status write in production code now goes through ``transition()``
(or ``recover()`` for the interrupted-job special case). The transition
tables below are the single source of truth for which status changes are
legal and which lifecycle event fires on each one.

Rationale: previously, 13 call sites directly mutated ``job.status`` and
separately called ``emit_job_event`` — two concerns, scattered. That
made it possible to change status without emitting, to emit twice, or
to transition into an invalid state (e.g. completed → running). The
chokepoint below closes all three gaps.

Registration events (``submitted``) and discovery events
(``interrupted_recovered`` when the status doesn't change on load) are
emitted directly by ``JobRegistry`` — they aren't transitions. The
state machine only covers actual status changes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import ExperimentJob
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
    validates completeness and sets ``runs_done_event`` BEFORE transitioning
    there, and the table above lets nothing but a terminal status follow it — so
    ``analyzing`` means every run is done and only the attached analysis is
    still in flight. That is what lets an experiment's own attached analysis
    read its own produced cases.

    Lives beside the transition table because the table is what makes it true.
    Every caller asking "are the runs done?" from a status reads this, so the
    equivalence is stated once rather than re-derived per module.
    """
    return status in TERMINAL_STATUSES or status == "analyzing"


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
}


class InvalidTransitionError(ValueError):
    """Raised when code attempts a status change not in the transition table."""


def _transitions_for(job: ExperimentJob) -> dict[str, frozenset[str]]:
    """Pick the transition table for a job's class.

    Only experiments transition, so anything else reaching here is a bug rather
    than an unmapped status.
    """
    if isinstance(job, ExperimentJob):
        return VALID_EXPERIMENT_TRANSITIONS
    raise TypeError(f"Unknown job type: {type(job).__name__}")


def _apply(
    job: ExperimentJob,
    new_status: str,
    valid: dict[str, frozenset[str]],
) -> None:
    """Validate and apply a status change; set completed_at + done_event
    on terminal transitions.

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
        job.done_event.set()


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
    valid = _transitions_for(job)
    event = STATUS_TO_EVENT.get(new_status)
    if event is None and new_status in valid.get(job.status, frozenset()):
        raise InvalidTransitionError(
            f"status {new_status!r} is restart-only and has no event mapping"
        )
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
