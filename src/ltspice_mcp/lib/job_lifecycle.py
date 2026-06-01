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
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES, BatchJob, SimulationJob
from ltspice_mcp.lib.observability import JobEvent, emit_job_event

if TYPE_CHECKING:
    from ltspice_mcp.state import SessionState

logger = logging.getLogger(__name__)


# Sim job: created "queued" in the tool layer (``tools/simulation.py``); the
# runner transitions queued → running once it acquires a max_parallel slot.
# A job can therefore sit "queued" for a while under load, so cancel/timeout
# must be able to terminate it directly from "queued" (not only "running").
VALID_SIM_TRANSITIONS: dict[str, frozenset[str]] = {
    "queued": frozenset({"running", "failed", "cancelled", "timeout"}),
    "running": frozenset({"completed", "failed", "cancelled", "timeout"}),
    "interrupted": frozenset({"completed"}),  # recovery promotion
    # Terminal: completed / failed / cancelled / timeout — no outgoing.
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "timeout": frozenset(),
}

# Batch job: dataclass default status is "running". No queued state.
# Interrupted stays interrupted — there's no mid-recovery promotion
# path for batches (no equivalent of a .raw file to validate).
VALID_BATCH_TRANSITIONS: dict[str, frozenset[str]] = {
    "running": frozenset({"completed", "failed", "cancelled"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
    "interrupted": frozenset(),
}

# Which event name fires when a job enters a given status.
# 'timeout' maps to 'failed' — it's a failure variant, not its own
# event type in the external log schema.
STATUS_TO_EVENT: dict[str, JobEvent] = {
    "running": "started",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "cancelled",
    "timeout": "failed",
}


class InvalidTransitionError(ValueError):
    """Raised when code attempts a status change not in the transition table."""


def _transitions_for(job: SimulationJob | BatchJob) -> dict[str, frozenset[str]]:
    """Pick the correct transition table for a job's class."""
    if isinstance(job, SimulationJob):
        return VALID_SIM_TRANSITIONS
    if isinstance(job, BatchJob):
        return VALID_BATCH_TRANSITIONS
    raise TypeError(f"Unknown job type: {type(job).__name__}")


def _apply(
    job: SimulationJob | BatchJob, new_status: str, valid: dict[str, frozenset[str]]
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
    job: SimulationJob | BatchJob,
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
    _apply(job, new_status, _transitions_for(job))
    if state is not None:
        state.persist_job(job)
    event = STATUS_TO_EVENT.get(new_status)
    if event is None:
        raise InvalidTransitionError(
            f"no event mapping for status {new_status!r}; update STATUS_TO_EVENT"
        )
    emit_job_event(event, job, **event_extra)


def recover(
    job: SimulationJob,
    new_status: str,
    *,
    state: SessionState | None = None,
    **event_extra: Any,
) -> None:
    """Promote an interrupted job to ``new_status`` after recovery.

    Emits ``interrupted_recovered`` (with ``recovered_as=new_status``)
    rather than the usual new-status event, because this transition
    semantically represents "we noticed a prior-session crash and
    reconciled" rather than a fresh run reaching ``new_status``.
    """
    if job.status != "interrupted":
        raise InvalidTransitionError(
            f"recover() requires current status 'interrupted', got {job.status!r}"
        )
    _apply(job, new_status, _transitions_for(job))
    if state is not None:
        state.persist_job(job)
    emit_job_event(
        "interrupted_recovered",
        job,
        recovered_as=new_status,
        **event_extra,
    )
