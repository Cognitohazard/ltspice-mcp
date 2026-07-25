"""Domain dataclasses for durable multi-circuit experiment jobs.

This module intentionally contains only data definitions. Experiment storage,
coordination, lifecycle, and MCP presentation live in separate layers.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from ltspice_mcp.lib import now

ExperimentStatus = Literal[
    "queued",
    "running",
    "analyzing",
    "completed",
    "completed_with_failures",
    "failed",
    "cancelled",
    "interrupted",
]

ExperimentCaseStatus = Literal[
    "queued",
    "submitted",
    "running",
    "produced",
    "failed",
    "cancelled",
    "skipped",
]

AnalysisStatus = Literal[
    "not_requested",
    "pending",
    "running",
    "completed",
    "failed",
    "cancelled",
]

TERMINAL_CASE_STATUSES: frozenset[str] = frozenset({"produced", "failed", "cancelled", "skipped"})


@dataclass
class ExperimentCase:
    """One expanded, staged simulator invocation owned by an experiment."""

    case_id: str
    run_index: int
    circuit: str
    circuit_path: Path
    staged_deck: Path
    deck_sha256: str
    assignments: dict[str, Any] = field(default_factory=dict)
    status: ExperimentCaseStatus = "queued"
    raw_file: Path | None = None
    log_file: Path | None = None
    error: str | None = None
    failure_code: str | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)
    submitted_at: datetime | None = None
    completed_at: datetime | None = None
    run_token: str = ""
    step_index: int | None = None
    step_values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Completeness:
    """Experiment accounting counters.

    At run terminality, ``produced + failed + cancelled + skipped`` must equal
    ``expanded``. ``declared`` records the pre-expansion case-family count and
    ``submitted`` records actual simulator submissions.
    """

    declared: int = 0
    expanded: int = 0
    submitted: int = 0
    produced: int = 0
    failed: int = 0
    cancelled: int = 0
    skipped: int = 0

    @property
    def terminal(self) -> int:
        return self.produced + self.failed + self.cancelled + self.skipped

    @property
    def fell_short(self) -> bool:
        """Did the runs deliver anything less than the expansion promised?

        Checked against ``expanded`` from both sides rather than by summing the
        shortfall counters. Under the terminal invariant all three agree; when
        they disagree the disagreement is the finding. ``terminal != expanded``
        catches a case that reached no counter at all, and ``produced !=
        expanded`` catches the same loss masked by a double-counted failure —
        either way an unreconciled run reads as a shortfall instead of being
        rounded down to success.

        Run-scoped on purpose. An attached analysis that failed or was
        cancelled is a separate fact with its own home in ``analysis.status``;
        folding it in here would send a caller hunting for dropped runs that do
        not exist.
        """
        return self.terminal != self.expanded or self.produced != self.expanded

    def recount(self, cases: list[ExperimentCase]) -> None:
        """Recompute all case-derived counters from the current case records."""
        submitted = produced = failed = cancelled = skipped = 0
        for case in cases:
            if case.submitted_at is not None or case.status in {
                "submitted",
                "running",
                "produced",
            }:
                submitted += 1
            if case.status == "produced":
                produced += 1
            elif case.status == "failed":
                failed += 1
            elif case.status == "cancelled":
                cancelled += 1
            elif case.status == "skipped":
                skipped += 1
        self.submitted = submitted
        self.produced = produced
        self.failed = failed
        self.cancelled = cancelled
        self.skipped = skipped

    def validate_terminal(self) -> None:
        """Raise when terminal accounting does not reconcile to expansion."""
        if self.terminal != self.expanded:
            raise ValueError(
                "Experiment completeness does not reconcile: "
                f"produced+failed+cancelled+skipped={self.terminal}, "
                f"expanded={self.expanded}"
            )


@dataclass
class ManifestEntry:
    """One primary deck or referenced include/library in a source snapshot."""

    path: Path
    sha256: str
    staged: bool
    live: bool
    staged_path: Path | None = None
    reason: str | None = None
    section: str | None = None


@dataclass
class SourceRecord:
    """Submission-time provenance for one source circuit."""

    circuit: str
    path: Path
    sha256: str
    staged_deck: Path
    manifest: list[ManifestEntry] = field(default_factory=list)
    linter_version: str = ""
    simulator: str = ""
    dialect: str | None = None
    lint_findings: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class AnalysisStage:
    """Persistent state for analysis attached to an experiment."""

    status: AnalysisStatus = "not_requested"
    request: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ExperimentJob:
    """Durable coordinator record spanning all cases and source circuits.

    An experiment deliberately has no ``netlist`` attribute. Each case names
    the staged deck it ran, and each source record names the authoring input.
    """

    job_id: str
    request_id: str
    fingerprint: str
    canonicalizer_version: int
    control_token: str
    store_path: Path
    cases: list[ExperimentCase]
    sources: list[SourceRecord]
    simulator: str
    completeness: Completeness
    status: ExperimentStatus = "queued"
    started_at: datetime = field(default_factory=now)
    completed_at: datetime | None = None
    error: str | None = None
    failures: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    analysis: AnalysisStage = field(default_factory=AnalysisStage)
    owner_pid: int = field(default_factory=os.getpid)
    runs_done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    done_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    task: asyncio.Task[None] | None = field(default=None, repr=False)
