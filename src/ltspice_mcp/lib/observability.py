"""Structured events for experiment jobs and tool calls.

Emits machine-parseable log records on job state transitions — submit,
start, completion, failure, cancellation, interrupted recovery — and one
record per tool call (:func:`emit_tool_event`). Logs go through a dedicated
``ltspice_mcp.events`` logger so operators can route them to a different
sink from the usual debug/info stream.

Each event carries:
    ts            ISO-8601 timestamp (UTC)
    event         lifecycle state: submitted | started | analyzing
                  | completed | completed_with_failures | failed
                  | cancelled | interrupted_recovered
    kind          'experiment' — the one job kind that runs here
    job_id        job identifier
    sources       circuit paths the job runs over
    duration_s    wall-clock seconds from started_at to now (or None
                  when the event precedes ``started_at``)
    extra keys    anything passed via kwargs (e.g. error, total_cases,
                  recovered_as)

Payloads are attached to log records via the ``extra`` dict so a
structured-log shipper (python-json-logger etc.) can pick them up
without parsing the message string.

Stderr is the one channel these records travel on, so installing it lives
here too: :func:`configure_stderr_logging` is what the server's lifespan and
the detached owner both call, and it is the only place that decides the
format an operator reads.
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal

from ltspice_mcp.lib import now
from ltspice_mcp.lib.experiment_types import ExperimentJob

if TYPE_CHECKING:
    from mcp import types

logger = logging.getLogger("ltspice_mcp.events")

#: The stderr format every ltspice-mcp process writes.
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def configure_stderr_logging(level: str) -> None:
    """Install this process's stderr logging at ``level``.

    An unrecognized level falls back to INFO rather than raising: this runs at
    startup, before there is anywhere to report a bad value, and refusing to
    boot over a log level would be worse than logging more than asked.

    ``force=True``, so a process may call this again once it knows more —
    which is what the detached owner does after its config is loaded.
    """
    from ltspice_mcp.config import VALID_LOG_LEVELS

    named = level.upper()
    if named not in VALID_LOG_LEVELS:
        named = "INFO"
    logging.basicConfig(
        level=getattr(logging, named),
        format=_LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr)],
        force=True,
    )


JobEvent = Literal[
    "submitted",
    "started",
    "analyzing",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
    "interrupted_recovered",
]


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
    job: ExperimentJob,
    **extra: Any,
) -> None:
    """Emit a structured lifecycle event for ``job``.

    Any additional keyword args are merged into the event payload.
    """
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": event,
        # Every job this server runs is an experiment.
        "kind": "experiment",
        "job_id": job.job_id,
        "duration_s": _duration_seconds(getattr(job, "started_at", None)),
    }
    payload["sources"] = [str(source.path) for source in job.sources]
    payload.update(extra)

    # Human-readable summary in the message, structured dict in extra.
    suffix = ""
    if payload.get("duration_s") is not None and event in (
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
    ):
        suffix = f" after {payload['duration_s']:.2f}s"
    logger.info(
        "experiment.%s job=%s%s",
        event,
        job.job_id,
        suffix,
        extra={"ltspice_event": payload},
    )


def emit_tool_event(tool: str, duration_s: float, result: types.CallToolResult) -> None:
    """Emit one event for a finished call of ``tool``.

    It says how long the call took, whether it failed and with which ``code``,
    and how many characters the reply is as it is sent, so a reply too large
    for a client shows here first. The call's arguments are not logged.
    Measuring a reply serializes it again, so nothing is measured unless the
    record would be emitted.
    """
    if not logger.isEnabledFor(logging.INFO):
        return
    failed = bool(result.is_error)
    payload: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(),
        "event": "tool_call",
        "tool": tool,
        "duration_s": duration_s,
        "outcome": "error" if failed else "ok",
        "code": (result.structured_content or {}).get("code") if failed else None,
        "reply_chars": len(result.model_dump_json(by_alias=True, exclude_none=True)),
    }
    logger.info(
        "tool.%s %s after %.2fs, %d chars",
        tool,
        payload["outcome"],
        duration_s,
        payload["reply_chars"],
        extra={"ltspice_event": payload},
    )
