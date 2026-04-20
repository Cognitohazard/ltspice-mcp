"""Structured lifecycle events for simulation and batch jobs.

Emits machine-parseable log records on job state transitions — submit,
start, completion, failure, cancellation, interrupted recovery. Logs go
through a dedicated ``ltspice_mcp.events`` logger so operators can route
them to a different sink from the usual debug/info stream.

Each event carries:
    ts            ISO-8601 timestamp (UTC)
    event         lifecycle state: submitted | started | completed
                  | failed | cancelled | interrupted_recovered
    kind          'sim' | 'sweep' | 'montecarlo'
    job_id        job identifier
    netlist       circuit file path
    duration_s    wall-clock seconds from started_at to now (or None
                  when the event precedes ``started_at``)
    extra keys    anything passed via kwargs (e.g. error, run_index)

Payloads are attached to log records via the ``extra`` dict so a
structured-log shipper (python-json-logger etc.) can pick them up
without parsing the message string.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, Literal

from ltspice_mcp.lib import now
from ltspice_mcp.lib.job_types import BatchJob, SimulationJob

logger = logging.getLogger("ltspice_mcp.events")

JobEvent = Literal[
    "submitted",
    "started",
    "completed",
    "failed",
    "cancelled",
    "interrupted_recovered",
]

JobKind = Literal["sim", "sweep", "montecarlo"]


def _duration_seconds(started_at: datetime | None) -> float | None:
    """Wall-clock seconds since ``started_at`` in UTC, or None."""
    if started_at is None:
        return None
    # Both sides are tz-aware via ``lib.now()`` so subtraction yields a
    # correct timedelta regardless of the zone choice.
    current = now()
    # Guard against clock skew / reordered events.
    delta = (current - started_at).total_seconds()
    return max(delta, 0.0)


def emit_job_event(
    event: JobEvent,
    job: SimulationJob | BatchJob,
    *,
    kind: JobKind | None = None,
    **extra: Any,
) -> None:
    """Emit a structured lifecycle event for ``job``.

    ``kind`` is inferred from the job class when omitted: SimulationJob
    → 'sim'; BatchJob → 'sweep' or 'montecarlo' depending on job_type.
    Any additional keyword args are merged into the event payload.
    """
    inferred_kind = kind or _infer_kind(job)

    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        "kind": inferred_kind,
        "job_id": job.job_id,
        "netlist": str(job.netlist),
        "duration_s": _duration_seconds(getattr(job, "started_at", None)),
    }
    payload.update(extra)

    # Human-readable summary in the message, structured dict in extra.
    suffix = ""
    if payload.get("duration_s") is not None and event in (
        "completed",
        "failed",
        "cancelled",
    ):
        suffix = f" after {payload['duration_s']:.2f}s"
    logger.info(
        "%s.%s job=%s%s",
        inferred_kind,
        event,
        job.job_id,
        suffix,
        extra={"ltspice_event": payload},
    )


def _infer_kind(job: SimulationJob | BatchJob) -> JobKind:
    """Map a job instance to its lifecycle ``kind`` string."""
    if isinstance(job, SimulationJob):
        return "sim"
    if isinstance(job, BatchJob):
        if job.job_type == "sweep":
            return "sweep"
        if job.job_type == "montecarlo":
            return "montecarlo"
    raise TypeError(f"Cannot infer lifecycle kind for {type(job).__name__}")
