"""Domain dataclasses for job records.

This module is the leaf of the job-subsystem dependency graph: it defines the
types and nothing else. Extracting these from ``state.py`` broke a cluster of
import cycles where ``state`` imported its collaborators (``lib.job_registry``,
``lib.job_store``, ``lib.job_lifecycle``, ``lib.observability``) and those
collaborators needed the dataclasses back for their type signatures.

After this split, the graph is strictly layered:

    lib.job_types (this file)
        ↑
        ├── state (SessionState composes JobRegistry)
        ├── lib.job_registry
        ├── lib.job_lifecycle (transition chokepoint)
        ├── lib.observability (event emission)
        └── lib.job_store (disk persistence)

``state.py`` re-exports every name below, so downstream code that wrote
``from ltspice_mcp.state import LegacyJobRecord`` keeps working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Terminal statuses — eligible for eviction, represent finished work.
# 'interrupted' is terminal because the owning runner is gone; metadata
# and any partial outputs are preserved but the job cannot resume
# in-process (recovery promotes it via lib.job_lifecycle.recover).
TERMINAL_STATUSES: frozenset[str] = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "timeout",
        "cancelled",
        "interrupted",
    }
)

# Statuses that only make sense while a runner owns the job. Seeing one
# in a persisted record means the prior server died mid-run.
NON_TERMINAL_LIVE_STATUSES: frozenset[str] = frozenset({"queued", "running", "analyzing"})


@dataclass(frozen=True)
class LegacyJobRecord:
    """A job sidecar written by a release before 0.6.

    Those releases ran simulations through job types this version no longer
    has — the runners, the per-run result read model, and the tools that
    addressed them are gone. A directory of these records must still load
    without breaking the registry or the startup preload, so they are parsed
    into this inert shape and nothing more: every read of one reports
    :func:`legacy_record_observation` and refuses to produce results.

    Attributes:
        job_id: The record's own id, as the earlier release assigned it.
        netlist: The circuit the record was written beside.
        kind: What the sidecar called itself — ``"sim"`` or ``"batch"``.
        status: The status the earlier release last persisted.
    """

    job_id: str
    netlist: Path
    kind: str
    status: str


def legacy_record_observation(job_id: str) -> dict[str, str]:
    """The one thing this version can say about a pre-0.6 job record.

    Surfaced on read instead of a crash (which would take the whole listing
    down) and instead of a silent skip (which would leave a caller waiting on
    a job that will never report). One fact, one remedy.

    ``kind`` is ``coverage``: this is a read the version did NOT perform and
    why, not a verdict on the run itself. No ``severity`` — that field carries
    a simulator's own rating, and no simulator rated this.
    """
    return {
        "code": "legacy_job_record",
        "kind": "coverage",
        "detail": (
            f"Job {job_id} was written by an earlier release; its results are not "
            "readable through this version — re-run it with run_experiments."
        ),
    }


def legacy_record_message(job_id: str) -> str:
    """The same fact as a refusal message, for callers that raise instead."""
    return legacy_record_observation(job_id)["detail"]
