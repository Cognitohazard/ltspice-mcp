"""The experiment receipt: its shape, its schema, and how it is rendered.

``run_experiments`` and ``jobs`` return the same envelope — job identity, source
provenance, completeness accounting, run rows, failures, observations, and the
attached analysis — so the shape lives here rather than in either tool.

The order of operations matters and is the reason ``ReceiptSnapshot`` exists.
The coordinator mutates jobs only on the event loop, so ``snapshot_receipt``
copies every job-derived fact synchronously; the renderers below then page,
project and budget-negotiate over that detached value as often as the ladder
needs, without ever observing a later job transition.
"""

from __future__ import annotations

import copy
import functools
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from mcp import types

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib import response_budget, services
from ltspice_mcp.lib.experiment_runner import live_run_progress
from ltspice_mcp.lib.experiment_types import (
    ACTIVE_CASE_STATUSES,
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES
from ltspice_mcp.lib.log_parser import diagnostic_collapse_key
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pagination import page_of
from ltspice_mcp.lib.projection import keep_plan, project_row
from ltspice_mcp.lib.simulator_build import SimulatorExecutable
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from ltspice_mcp.tools._base import (
    FINDING_SCHEMA,
    OUTCOME_SCHEMA,
    CallOutcome,
    ResponseBudget,
    failures_schema,
    format_response,
    outcome_of,
    page_schema,
)

_RUN_PAGE_LIMIT = 50

JOBS_PAGE_LIMIT = 50

TERMINAL_EXPERIMENT_STATUSES = frozenset(
    {
        "completed",
        "completed_with_failures",
        "failed",
        "cancelled",
        "interrupted",
    }
)

_MANIFEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "sha256": {"type": "string"},
        "staged": {"type": "boolean"},
        "live": {"type": "boolean"},
        "staged_path": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
        "section": {"type": ["string", "null"]},
    },
    "required": [
        "path",
        "sha256",
        "staged",
        "live",
        "staged_path",
        "reason",
        "section",
    ],
}

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "kind": {"type": "string"},
        "detail": {"type": "string"},
        "evidence": {},
    },
    "required": ["code", "kind", "detail"],
}

#: A receipt failure names the case that failed, not a stage (see Envelope).
CASE_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "case_ids": {"type": "array", "items": {"type": "string"}},
        "count": {"type": "integer"},
        "code": {"type": "string"},
        "message": {"type": "string"},
        "evidence": {"type": "object"},
        "hint": {"type": "string"},
    },
    "required": ["case_id", "code", "message"],
}

# Recovery guidance per classified failure code, worded for the six tools this
# profile exposes. These are the surviving arms of three exception hints that
# were keyed by exception classes nothing ever raised, so no caller saw them;
# the case-level failure code is the signal that actually reaches a caller.
# They live beside the receipt that carries them because the failure channel
# exists only here — another profile growing one would need its own wording,
# not a share of this table.
_FAILURE_CODE_HINTS: dict[str, str] = {
    "convergence_failed": (
        "Add a .OPTIONS directive to the deck (e.g. .OPTIONS reltol=0.003 or "
        ".OPTIONS method=gear), or check component values for very large/small ratios."
    ),
    "singular_matrix": (
        "This usually means a floating node or short circuit. Use inspect with a "
        "net query to trace connectivity, or read the netlist directly."
    ),
    "missing_model": (
        'Use inspect with a model query (mode:"search") to fuzzy-match against '
        "the simulator's own libraries, or add a .lib/.include for it to the deck."
    ),
    "missing_include": (
        "The deck names an .include or .lib file the simulator could not open. "
        "A relative path is resolved against the staged deck's directory, so "
        "give an absolute path or put the file beside the circuit."
    ),
    "ngspice_lib_section": (
        "ngspice is running in an LTspice/PSPICE-compatibility mode, which reads "
        "a sectioned '.lib <file> <section>' as two plain includes and drops the "
        "section — so the corner select came back as a missing file. Set "
        '[simulator] ngbehavior = "hsa" in ltspice-mcp.toml (or '
        "LTSPICE_MCP_NGBEHAVIOR=hsa) and restart the server, or add "
        "'set ngbehavior=hsa' to a .spiceinit in the run directory."
    ),
    "run_timeout": (
        "The simulator ran past the per-case run timeout and was stopped. evidence "
        "names the bound and whether it was the request's or the server default; "
        "the partial_progress observation says how far the run got. If it needs "
        "longer, resubmit with a larger execution.run_timeout_s (a server default comes "
        'from [simulation] run_timeout, limits.run_timeout_s in inspect(kind="capabilities")). '
        "If it should have been quick, read evidence.log_excerpt for a collapsing timestep."
    ),
    "job_deadline": (
        "The job's execution.job_deadline_s elapsed before this case finished; "
        "results already produced are kept. Resubmit the unfinished cases with a "
        "larger job_deadline_s, or split them across jobs."
    ),
    "kill_unconfirmed": (
        "The simulator was told to stop but did not report exit within the kill "
        "grace period, so it may still be running, and its concurrency slot stays "
        "reserved until it exits (the job then gains a late_simulator_exit "
        "observation). If it never does, end the simulator process whose command "
        "line carries this job_id, or restart the server."
    ),
}

_ARTIFACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content_type": {"type": "string"},
        "sha256": {"type": "string"},
        "bytes": {"type": "integer"},
    },
    "required": ["path", "content_type", "sha256", "bytes"],
}

RUN_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "run_index": {"type": "integer"},
        "circuit": {"type": "string"},
        "assignments": {"type": "object"},
        "native_statistics": {"type": "object"},
        "status": {"type": "string"},
        "raw": {"type": ["string", "null"]},
        "log": {"type": ["string", "null"]},
        "simulator_version": {"type": ["string", "null"]},
        "attempt": {
            "type": "object",
            "properties": {
                "execution_job_id": {"type": "string"},
                "attempt_index": {"type": "integer", "minimum": 0},
                "run_token": {"type": "string"},
                "reused": {"type": "boolean"},
            },
            "required": ["execution_job_id", "attempt_index", "run_token", "reused"],
            "additionalProperties": False,
        },
    },
    # run_fields may project away any key, so the shared row fragment
    # deliberately requires none of them.
}

RUNS_PAGE_SCHEMA: dict[str, Any] = page_schema({"type": "array", "items": RUN_RECORD_SCHEMA})

_COMPLETENESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "declared": {"type": "integer"},
        "expanded": {"type": "integer"},
        "submitted": {"type": "integer"},
        "produced": {"type": "integer"},
        "failed": {"type": "integer"},
        "cancelled": {"type": "integer"},
        "skipped": {"type": "integer"},
        "reused": {"type": "integer"},
    },
    "required": [
        "declared",
        "expanded",
        "submitted",
        "produced",
        "failed",
        "cancelled",
        "skipped",
        "reused",
    ],
}

# The derived view of ``completeness``, and only that: the raw counters
# live in ``completeness`` and are not restated here.
_PROGRESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "expanded": {"type": "integer"},
        "terminal": {"type": "integer"},
        "remaining": {"type": "integer"},
    },
    "required": ["expanded", "terminal", "remaining"],
}

LINEAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_job_id": {"type": "string"},
        "parent_job_id": {"type": ["string", "null"]},
        "attempt_index": {"type": "integer", "minimum": 0},
    },
    "required": ["root_job_id", "parent_job_id", "attempt_index"],
    "additionalProperties": False,
}


RUN_EXPERIMENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": ["string", "null"]},
        "request_id": {"type": "string"},
        "control_token": {"type": "string"},
        "lineage": LINEAGE_SCHEMA,
        # A fact about THIS call, not about the job: true when the request_id
        # and canonical payload matched an existing durable experiment, so its
        # receipt came back and no cases were submitted. The replay leaves the
        # job's record as it was.
        "replayed": {"type": "boolean"},
        "status": {"type": "string"},
        "outcome": OUTCOME_SCHEMA,
        "source": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "path": {"type": "string"},
                    "sha256": {"type": "string"},
                    "staged_deck": {"type": "string"},
                    "manifest": {"type": "array", "items": _MANIFEST_SCHEMA},
                    "linter_version": {"type": "string"},
                    "simulator": {"type": "string"},
                    "dialect": {"type": ["string", "null"]},
                    "staged_files": {"type": "integer"},
                },
                # Only the fields every receipt carries are required. The
                # digests, staging paths and full manifest are provenance: they
                # are emitted when 'provenance' is set, and otherwise omitted,
                # because they are a third of a receipt's bytes and name files
                # the caller does not open.
                "required": [
                    "circuit",
                    "path",
                    "simulator",
                    "dialect",
                ],
            },
        },
        # The program the job's cases launched, as identified at submission.
        # Provenance: emitted under 'provenance' only.
        "simulator_executable": {
            "type": ["object", "null"],
            "properties": {
                "path": {"type": "string"},
                "sha256": {"type": ["string", "null"]},
                "bytes": {"type": ["integer", "null"]},
                "modified": {"type": ["string", "null"]},
            },
            "required": ["path", "sha256", "bytes", "modified"],
        },
        "completeness": _COMPLETENESS_SCHEMA,
        "progress": _PROGRESS_SCHEMA,
        "lint": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "findings": {"type": "array", "items": FINDING_SCHEMA},
                },
                "required": ["circuit", "findings"],
            },
        },
        "runs": RUNS_PAGE_SCHEMA,
        "analysis": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "request": {"type": ["object", "null"]},
                "result": {"type": ["object", "null"]},
                "error": {"type": ["string", "null"]},
                "observations": {
                    "type": "array",
                    "items": OBSERVATION_SCHEMA,
                },
            },
            # "request" is the caller's own input replayed back — emitted
            # under provenance only, so it cannot be required.
            "required": [
                "status",
                "result",
                "error",
                "observations",
            ],
        },
        "failures": failures_schema(CASE_FAILURE_SCHEMA),
        "observations": {"type": "array", "items": OBSERVATION_SCHEMA},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
        "hint": {"type": "string"},
        "error": {
            "type": "object",
            "properties": {
                "code": {"type": "string"},
                "message": {"type": "string"},
                "stage": {"type": "string"},
                "retryable": {"type": "boolean"},
                "commit_state": {
                    "type": "string",
                    "enum": ["not_started", "committed", "unknown"],
                },
                "item_id": {"type": "string"},
            },
            "required": ["code", "message", "stage", "retryable", "commit_state"],
        },
    },
    "required": [
        "job_id",
        "request_id",
        "replayed",
        "status",
        "outcome",
        "source",
        "completeness",
        "progress",
        "lint",
        "runs",
        "failures",
        "observations",
        "warnings",
        "artifacts",
        "hint",
    ],
}

ReceiptBuilt = tuple[dict[str, Any], str]
ReceiptBuild = Callable[[int, response_budget.Rung | None], ReceiptBuilt]

# Rung 0's allowlist, shared by run_experiments and jobs status/wait because
# both render the same receipt envelope. The attached analysis is that tool's
# own envelope, trimmed by that tool's own allowlists (`analyze.trim_analysis`):
# its `source_hashes` is one identity row per run, so a receipt that kept it
# grew by every case it ran whatever the budget.
_TRIM_REMOVE_RECEIPT: tuple[str, ...] = ("analysis",)
# `source` is deliberately absent, and no other rung empties it either. It is
# not an identity echo the way the analysis envelope's `source_hashes` is: with
# `provenance` off it still carries the staging disclosures — the live,
# unstaged or unexplained manifest entries `_source_payload` keeps precisely so
# a caller is told about them — and with `provenance` on it carries an opt-in
# the caller asked for, which the trim rung's charter forbids revoking. Facts
# under one flag and an opt-in under the other leaves no rung a claim on it.

#: Where a receipt's trimmed echo still is, for a caller under the server's
#: default budget, who set none and so has none to raise.
RECEIPT_DEFAULT_ROUTE = (
    "Attached-analysis rows still name their run by case_id; analyze_results "
    "over this job_id with include.provenance returns source_hashes."
)

_RUN_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced; no run, failure, or analysis fact was dropped.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or continue "
        "through the returned cursor/jobs route."
    ),
    default_route=RECEIPT_DEFAULT_ROUTE,
)


def _receipt_row_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every run page carried at the top level or under a receipt."""
    pages = [data] if isinstance(data.get("items"), list) else []
    runs = data.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("items"), list):
        pages.append(runs)
    return pages


def _attached_result(data: dict[str, Any]) -> dict[str, Any] | None:
    """The attached analysis's rendered result, when the receipt carries one."""
    analysis_block = data.get("analysis")
    result = analysis_block.get("result") if isinstance(analysis_block, dict) else None
    return result if isinstance(result, dict) else None


def _receipt_surfaces(data: dict[str, Any]) -> list[list[Any]]:
    """Every row surface a receipt-shaped response shows, one list apiece: its
    run pages, and the attached analysis's surfaces when it carries one. The
    one measure every receipt is shrunk against, whichever tool returns it."""
    surfaces = [page["items"] for page in _receipt_row_pages(data)]
    result = _attached_result(data)
    if result is not None:
        surfaces.extend(analyze.analysis_surfaces(result))
    return surfaces


def _degrade_receipt(data: dict[str, Any], rung: response_budget.Rung) -> list[str]:
    """Apply presentation rungs to either public receipt envelope.

    Returns the blocks the trim emptied of content, for the rung's note.
    """
    emptied: list[str] = []
    if rung.trim:
        # Removes an empty block only, so it never empties anything a note
        # would have to name.
        response_budget.apply_trim(data, remove=_TRIM_REMOVE_RECEIPT)
        result = _attached_result(data)
        if result is not None:
            emptied = [f"analysis.result.{key}" for key in analyze.trim_analysis(result)]
    if rung.answer_channel:
        for page in _receipt_row_pages(data):
            for row in page["items"]:
                if isinstance(row, dict) and row.get("status") == "produced":
                    row.pop("raw", None)
                    row.pop("log", None)
    return emptied


async def negotiate_receipt(
    budget: ResponseBudget,
    build: ReceiptBuild,
    page_limit: int,
    *,
    notes: response_budget.Notes,
) -> ReceiptBuilt:
    """Render a receipt at the mildest shared budget rung that fits.

    The shrink rung measures the page its estimate priced and searches below it
    when that page is still over. The search goes down to a limit of zero: a
    receipt's rows preview surfaces other calls page (``jobs(runs)``,
    ``analyze_results``), so at the floor ``completeness``, ``runs.total`` and
    the cursor stand in for them and the floor is the same size however many
    cases the job ran. A page whose cursor continues itself floors its own
    limit at one row.
    """
    text = ""
    rendered: dict[str, Any] = {}

    def candidate(limit: int, rung: response_budget.Rung) -> response_budget.Rendered:
        data, line = build(limit, rung)
        return data, line, _degrade_receipt(data, rung)

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        nonlocal text, rendered
        if rung.shrink:
            rendered, text, cut = response_budget.shrink_to_fit(
                response_budget.RowMeasure.of(_receipt_surfaces(rendered), page=rung.measured),
                lambda limit: candidate(limit, rung),
                rung,
                floor=0,
                cap=page_limit,
            )
        elif rung.level == response_budget.RUNG_TRIM:
            # The undegraded rung built this same page; degrade it in place.
            cut = _degrade_receipt(rendered, rung)
        else:
            rendered, text, cut = candidate(page_limit, rung)
        rung.cut.extend(cut)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(budget.tokens, render, max_rung=budget.max_rung)
    response_budget.attach_notes(result, notes)
    return result.data, text


async def render_run_receipt(
    budget: ResponseBudget,
    build: ReceiptBuild,
    *,
    is_error: bool = False,
) -> types.CallToolResult:
    if budget.tokens is None:
        data, text = build(_RUN_PAGE_LIMIT, None)
    else:
        data, text = await negotiate_receipt(
            budget, build, _RUN_PAGE_LIMIT, notes=_RUN_BUDGET_NOTES
        )
    result = format_response(text, data)
    result.is_error = is_error
    return result


def progress_from_completeness(completeness: Completeness) -> dict[str, int]:
    """Project durable accounting into the shared progress fact.

    The three DERIVED numbers only. ``completeness`` keeps all raw counters,
    unchanged and always, so each counter is one key away and neither this
    block nor the hint restates them.
    """
    return {
        "expanded": completeness.expanded,
        "terminal": completeness.terminal,
        "remaining": completeness.expanded - completeness.terminal,
    }


def finalize_receipt(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize completeness and attach its ``progress`` projection."""
    raw = data["completeness"]
    completeness = raw if isinstance(raw, Completeness) else Completeness(**raw)
    data["completeness"] = asdict(completeness)
    data["progress"] = progress_from_completeness(completeness)
    return data


@dataclass(frozen=True)
class ReceiptSnapshot:
    """One loop-atomic copy of every job-derived receipt fact.

    The coordinator mutates jobs only on the event loop.  Constructing this
    value is deliberately synchronous, and every mutable leaf is detached from
    the job before control can return to the loop.  Presentation can therefore
    page, project, and budget-negotiate repeatedly without observing a later
    job transition.
    """

    job_id: str
    request_id: str | None
    status: str
    dialect: str | None
    control_token: str | None
    sources: tuple[SourceRecord | dict[str, Any], ...]
    lint: tuple[dict[str, Any], ...]
    runs_by_key: dict[tuple[str, int], dict[str, Any]]
    native_by_key: dict[tuple[str, int], NativeCaseRecord]
    completeness: Completeness
    failures: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    analysis_status: str
    analysis_result: dict[str, Any] | None
    analysis_error: str | None
    analysis_observations: tuple[dict[str, Any], ...]
    analysis_request: dict[str, Any] | None
    simulator_executable: SimulatorExecutable | None = None
    lineage: dict[str, Any] | None = None
    path_denied_hint: str | None = None
    """The session's sandbox guidance, which a ``path_denied`` failure row
    carries as its hint. Not a job fact: the remedy is the sandbox setting as
    it reads now, so it comes from the session rather than the record."""

    @property
    def outcome(self) -> CallOutcome:
        """Receipt outcome derived only from copied status and completeness."""
        return _terminal_outcome(self)

    @functools.cached_property
    def rendered_failures(self) -> tuple[dict[str, Any], ...]:
        """The failure rows, collapsed once: a budget re-renders the receipt
        once per limit it tries, and no limit changes these."""
        return tuple(_render_failures(self.failures, self.path_denied_hint))


def render_receipt_snapshot(
    snapshot: ReceiptSnapshot,
    *,
    control_token: str | None = None,
    provenance: bool = False,
    run_fields: list[str] | None = None,
    runs_cap: int = _RUN_PAGE_LIMIT,
    analysis_fields: list[str] | None = None,
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
) -> dict[str, Any]:
    """Render the existing receipt envelope from detached neutral facts.

    The snapshot's leaves are already detached from the job, and rendering
    never writes through them — the only row edits build or own their dicts —
    so the envelope shares them rather than copying them per render.
    """
    runs = project_receipt_runs(
        snapshot,
        run_fields,
        lean_default=True,
        limit=runs_cap,
    )
    data: dict[str, Any] = {
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "outcome": snapshot.outcome,
        "source": [
            _source_payload(source, provenance=provenance)
            if isinstance(source, SourceRecord)
            else dict(source)
            for source in snapshot.sources
        ],
        "completeness": snapshot.completeness,
        "lint": list(snapshot.lint),
        "runs": runs,
        "failures": list(snapshot.rendered_failures),
        "observations": list(snapshot.observations),
        "warnings": [],
        "artifacts": list(snapshot.artifacts),
        "hint": (
            f"Experiment {snapshot.job_id} is still running; use jobs(wait) with this "
            "job_id to continue waiting."
            if snapshot.status not in TERMINAL_EXPERIMENT_STATUSES
            else _terminal_hint(snapshot, runs)
        ),
    }
    emitted_control_token = control_token if control_token is not None else snapshot.control_token
    if emitted_control_token is not None and (
        snapshot.lineage is not None or snapshot.status not in TERMINAL_EXPERIMENT_STATUSES
    ):
        data["control_token"] = emitted_control_token
    if snapshot.lineage is not None:
        data["lineage"] = dict(snapshot.lineage)
    if snapshot.analysis_status != "not_requested":
        rendered_result: dict[str, Any] | None = None
        if snapshot.analysis_result is not None:
            rendered_result = analyze.render_attached_analysis(
                snapshot.analysis_result,
                fields=analysis_fields,
                answer_channel=analysis_answer_channel,
                row_limit=analysis_rows_cap,
            )
        analysis_observations = list(snapshot.analysis_observations)
        data["analysis"] = {
            "status": snapshot.analysis_status,
            "result": rendered_result,
            "error": snapshot.analysis_error,
            "observations": analysis_observations,
        }
        if provenance:
            # The caller's own attached-analysis input, replayed back —
            # proof of what ran, not something to re-read every turn.
            data["analysis"]["request"] = copy.deepcopy(snapshot.analysis_request)
    if provenance:
        executable = snapshot.simulator_executable
        data["simulator_executable"] = executable.to_record() if executable else None
    return data


_FAILURE_CASE_ID_CAP = 10
"""How many case ids a collapsed failure row names before deferring to ``count``."""


def _render_failures(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
    path_denied_hint: str | None = None,
) -> list[dict[str, Any]]:
    """Collapse repeated failures into one counted row and attach recovery hints.

    A case failure carries a ~20-line log excerpt in its message, and a sweep
    or Monte Carlo that fails for one reason fails that way in every case — a
    hundred cases is a hundred copies of the same kilobyte in a channel the
    budget ladder is forbidden to trim. Rows sharing a ``(code, message)`` key
    therefore become one row naming its cases, exactly as the log reader
    already collapses a repeated diagnostic within one log.

    The key runs through :func:`diagnostic_collapse_key` because the excerpt
    ends in the case's own numeric state, so cases that failed for one reason
    are byte-identical only when they are also numerically identical — which in
    a Monte Carlo they never are. The emitted row is one member verbatim.

    Which member is decided by case id, not by which case happened to finish
    first: cases run in parallel, so completion order varies between runs, and
    a receipt whose excerpt and named cases change when nothing else did is not
    one two runs can be compared with.

    No fact is dropped: ``count`` is the true number of cases, so a capped
    ``case_ids`` list reports its own shortfall rather than rounding it away.
    """
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row.get("code", "")), diagnostic_collapse_key(str(row.get("message", ""))))
        grouped.setdefault(key, []).append(row)

    collapsed: list[dict[str, Any]] = []
    for (code, _message), members in grouped.items():
        group = sorted(members, key=lambda item: str(item.get("case_id", "")))
        rendered = dict(group[0])
        if len(group) > 1:
            rendered["case_ids"] = [
                str(item.get("case_id", "")) for item in group[:_FAILURE_CASE_ID_CAP]
            ]
            rendered["count"] = len(group)
        hint = (
            path_denied_hint if code == PathSecurityError.code else _FAILURE_CODE_HINTS.get(code)
        )
        if hint is not None:
            rendered["hint"] = hint
        collapsed.append(rendered)
    return collapsed


def _source_payload(source: SourceRecord, *, provenance: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "circuit": source.circuit,
        "path": str(source.path),
        "simulator": source.simulator,
        "dialect": source.dialect,
    }
    if provenance:
        payload["sha256"] = source.sha256
        payload["staged_deck"] = str(source.staged_deck)
        payload["manifest"] = [_manifest_payload(entry) for entry in source.manifest]
        payload["linter_version"] = source.linter_version
        return payload
    # Keep the entries that say something the caller must act on; dropping those
    # with the bulk would turn a disclosure into a silent omission. The predicate
    # is "not an ordinary staged reference" rather than a list of known-bad
    # states, so it fails CLOSED — the store rebuilds `staged` with a False
    # default, and an entry that is neither staged, live, nor explained is
    # exactly the anomaly that must not be hidden.
    notable = [
        entry for entry in source.manifest if entry.live or entry.reason or not entry.staged
    ]
    if notable:
        payload["manifest"] = [_manifest_payload(entry) for entry in notable]
    payload["staged_files"] = len(source.manifest)
    return payload


def _manifest_payload(entry: ManifestEntry) -> dict[str, Any]:
    return {
        "path": str(entry.path),
        "sha256": entry.sha256,
        "staged": entry.staged,
        "live": entry.live,
        "staged_path": str(entry.staged_path) if entry.staged_path else None,
        "reason": entry.reason,
        "section": entry.section,
    }


#: Run-row keys a lean receipt leaves out of a produced row. Its artifacts can
#: be resolved by job/case identity, and the build that produced it is
#: confirming detail; jobs(runs) and run_fields return all three. A failed row
#: keeps them, its log path being how the caller inspects the failure.
_LEAN_PRODUCED_OMITS = ("raw", "log", "simulator_version")


def _run_item(case: ExperimentCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "circuit": case.circuit,
        "assignments": case.assignments,
        "status": case.status,
        "raw": str(case.raw_file) if case.raw_file else None,
        "log": str(case.log_file) if case.log_file else None,
        "simulator_version": case.simulator_version,
    }
    if case.recovery is not None:
        attempt = case.recovery.attempt
        row["attempt"] = {
            "execution_job_id": attempt.execution_job_id,
            "attempt_index": attempt.attempt_index,
            "run_token": attempt.run_token,
            "reused": attempt.reused,
        }
    return row


def _project_run_rows(
    rows: list[dict[str, Any]],
    run_fields: list[str] | None,
    *,
    native_by_key: dict[tuple[str, int], NativeCaseRecord],
    lean_default: bool,
) -> list[dict[str, Any]]:
    """Render native evidence and project only the rows this page carries."""
    plan = keep_plan(run_fields) if run_fields else None
    if plan is None or "native_statistics" in plan:
        for row in rows:
            native = native_by_key.get((row["case_id"], row["run_index"]))
            if native is not None:
                row["native_statistics"] = native.public(
                    detailed=plan is not None,
                    projection=plan.get("native_statistics") if plan is not None else None,
                )
    if plan is not None:
        return [project_row(row, plan) for row in rows]
    if lean_default:
        for row in rows:
            if row["status"] == "produced":
                for key in _LEAN_PRODUCED_OMITS:
                    del row[key]
    return rows


def runs_page(
    cases: list[ExperimentCase],
    run_fields: list[str] | None = None,
    *,
    cap: int = _RUN_PAGE_LIMIT,
) -> dict[str, Any]:
    # Paged first, projected second: the projection is one row in, one row out,
    # so it only has to run over the rows this page actually carries. The cap is
    # floored at one row for the same reason ``page`` floors its limit — a page
    # of none would report itself truncated with a cursor back at the same
    # offset, which is a pagination loop that never advances.
    selected = cases[: max(1, cap)]
    native_by_key = {}
    for case in selected:
        if case.native_statistics is not None:
            case.native_statistics.validate()
            native_by_key[(case.case_id, case.run_index)] = case.native_statistics
    rows = _project_run_rows(
        [_run_item(case) for case in selected],
        run_fields,
        native_by_key=native_by_key,
        lean_default=True,
    )
    return page_of(rows, offset=0, total=len(cases))


def project_receipt_runs(
    snapshot: ReceiptSnapshot,
    run_fields: list[str] | None,
    *,
    lean_default: bool,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Purely apply one invocation's run projection to copied canonical rows.

    ``run_fields`` and ``lean_default`` together are the original request's
    projection policy.  Receipt requests use the lean default when no explicit
    fields were supplied; ``jobs(runs)`` requests use the full-row policy.
    ``limit=None`` renders the complete page shape for an in-process consumer.
    ``offset`` is already decoded from the caller's cursor — a malformed one is
    that tool's error to raise, not this renderer's.

    ``limit=0`` is a receipt's budget floor, no row inline; ``jobs(runs)``
    floors its own limit at one row.
    """
    rows = list(snapshot.runs_by_key.values())
    start = min(max(offset, 0), len(rows))
    end = len(rows) if limit is None else start + limit
    page = page_of(rows[start:end], offset=start, total=len(rows))
    page["items"] = _project_run_rows(
        [dict(row) for row in page["items"]],
        run_fields,
        native_by_key=snapshot.native_by_key,
        lean_default=lean_default,
    )
    return page


def _terminal_outcome(snapshot: ReceiptSnapshot) -> CallOutcome:
    """An experiment receipt's outcome, read off its own status and counters.

    ``delivered=False``: an experiment that failed produced nothing to read, so
    a failure here IS the call's outcome rather than one item among many. A
    cancelled experiment is partial whatever the counters say — a cancel
    landing after every run but before the analysis leaves them fully
    reconciled, and one landing before expansion leaves them all at zero.
    """
    return outcome_of(
        snapshot.status == "failed",
        partial=snapshot.status == "cancelled" or snapshot.completeness.fell_short,
        in_progress=snapshot.status not in TERMINAL_EXPERIMENT_STATUSES,
        delivered=False,
    )


def _terminal_hint(snapshot: ReceiptSnapshot, runs: Mapping[str, Any]) -> str:
    """Every recovery route this receipt has, not the first one that matched.

    A server restart mid-run sets BOTH conditions: the abandoned cases become
    failures AND the attached analysis is marked failed. Under an exclusive
    ladder the failures branch won and the caller was never told that the runs
    that DID produce data are still analyzable by job_id — so the obvious move
    was to re-run an experiment whose results were sitting on disk. A paged
    run list is one more route, not a reason to drop the others: a budget's
    floor pages every receipt, failed ones included.
    """
    routes: list[str] = []
    if runs["truncated"]:
        routes.append(
            f"The inline run page is truncated; use jobs(runs) with job_id "
            f"{snapshot.job_id} for the remaining cases."
            if runs["returned"]
            else "No run row is inline at this budget; completeness and runs.total "
            f"count them, and jobs(runs) with job_id {snapshot.job_id} pages them."
        )
    routes.extend(_recovery_routes(snapshot))
    if not routes:
        routes.append(f"Experiment {snapshot.job_id} is {snapshot.status}.")
    return " ".join(routes)


def _recovery_routes(snapshot: ReceiptSnapshot) -> list[str]:
    """The routes a terminal receipt's failures leave, whatever its run page."""
    routes: list[str] = []
    if snapshot.analysis_status in {"failed", "cancelled"}:
        routes.append(
            f"The attached analysis {snapshot.analysis_status}; read analysis.error and "
            f"re-run it with analyze_results over job_id {snapshot.job_id} — the runs "
            "that produced data need no re-run."
        )
    if snapshot.failures:
        routes.append("Inspect failures and lint findings before retrying omitted cases.")
    return routes


#: Statuses on which a job delivered nothing at all, so the whole call failed.
_FAILED_JOB_STATUSES = frozenset({"failed", "timeout", "interrupted"})


def _jobs_outcome(snapshot: ReceiptSnapshot) -> CallOutcome:
    """A jobs receipt's outcome — the same rule, read off the job's status.

    What differs from ``_terminal_outcome`` is which statuses count. The
    ``jobs`` control plane reports a wider set as outright failure (a timeout
    and an interrupt deliver nothing either), and it takes terminality from the
    lifecycle's live-status set rather than the experiment status list.

    ``completed_with_failures`` is the one status that has to consult the
    counters: the coordinator sets it when an attached analysis fails even
    though every run landed, so an experiment's shortfall is read off its own
    run counters instead. Only this status — a cancelled experiment is partial
    no matter how the counters read, including the cancel that lands before
    expansion and leaves them all at zero.
    """
    shortfall = snapshot.status == "cancelled" or (
        snapshot.status == "completed_with_failures" and snapshot.completeness.fell_short
    )
    return outcome_of(
        snapshot.status in _FAILED_JOB_STATUSES,
        partial=shortfall,
        in_progress=snapshot.status in NON_TERMINAL_LIVE_STATUSES,
        delivered=False,
    )


def snapshot_receipt(
    job: ExperimentJob,
    state: SessionState | None,
    *,
    control_token: str | None = None,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
    live_progress: Mapping[str, dict[str, Any]] | None = None,
) -> ReceiptSnapshot:
    """Copy a job's complete receipt state without suspending the event loop.

    The first job read through the last mutable copy occur in this synchronous
    call.  Outcome and guidance are intentionally absent from that live-read
    interval; renderers derive them only from the returned detached value.
    Native record holders are copied, sharing only their frozen nested facts;
    large evidence lists are serialized after the renderer selects a page.

    ``live_progress`` is ``live_run_progress``'s read, taken off the loop just
    before this call (``snapshot_receipt_live`` pairs the two). An entry joins
    the observations only for a case still running here, so a case that
    finished in between is not reported running.

    The sandbox guidance is built only when a failure row needs it, so a
    receipt with no refused path never reads the config file.
    """
    lint_map: dict[str, list[dict[str, Any]]]
    if lint_by_circuit is None:
        lint_map = {}
        for case in job.cases:
            lint_map.setdefault(case.circuit, [])
        for source in job.sources:
            lint_map[source.circuit] = source.lint_findings
    else:
        lint_map = lint_by_circuit

    observations = copy.deepcopy(job.observations)
    seen_observations = {(item.get("code"), item.get("detail")) for item in observations}
    runs_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    native_by_key: dict[tuple[str, int], NativeCaseRecord] = {}
    for case in job.cases:
        run_key = (case.case_id, case.run_index)
        runs_by_key[run_key] = copy.deepcopy(_run_item(case))
        if case.native_statistics is not None:
            case.native_statistics.validate()
            native_by_key[run_key] = copy.copy(case.native_statistics)
        for observation in case.observations:
            copied = copy.deepcopy(observation)
            key = (copied.get("code"), copied.get("detail"))
            if key not in seen_observations:
                observations.append(copied)
                seen_observations.add(key)
        if live_progress and case.status in ACTIVE_CASE_STATUSES:
            live = live_progress.get(case.case_id)
            if live is not None:
                observations.append(copy.deepcopy(live))

    analysis = job.analysis
    return ReceiptSnapshot(
        job_id=job.job_id,
        request_id=job.request_id,
        status=job.status,
        dialect=services.dialect_for_job(job, state) if state is not None else None,
        control_token=control_token,
        sources=tuple(copy.deepcopy(job.sources)),
        lint=tuple(
            copy.deepcopy(
                [
                    {"circuit": circuit, "findings": findings}
                    for circuit, findings in lint_map.items()
                    if findings
                ]
            )
        ),
        runs_by_key=runs_by_key,
        native_by_key=native_by_key,
        completeness=copy.deepcopy(job.completeness),
        failures=tuple(copy.deepcopy(job.failures)),
        observations=tuple(observations),
        artifacts=tuple(copy.deepcopy(job.artifacts)),
        analysis_status=analysis.status,
        analysis_result=copy.deepcopy(analysis.result),
        analysis_error=analysis.error,
        analysis_observations=tuple(copy.deepcopy(analysis.observations)),
        analysis_request=copy.deepcopy(analysis.request),
        simulator_executable=job.simulator_executable,
        lineage=(
            {
                "root_job_id": job.recovery.root_job_id,
                "parent_job_id": job.recovery.parent_job_id,
                "attempt_index": job.recovery.attempt_index,
            }
            if job.recovery is not None
            else None
        ),
        path_denied_hint=(
            state.sandbox_guidance()
            if state is not None
            and any(row.get("code") == PathSecurityError.code for row in job.failures)
            else None
        ),
    )


async def snapshot_receipt_live(
    job: ExperimentJob,
    state: SessionState | None,
    *,
    control_token: str | None = None,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
) -> ReceiptSnapshot:
    """``snapshot_receipt`` with each running case's progress, read first.

    The read suspends (it is file I/O, off the loop); the snapshot after it
    does not. Every receipt that reports on a job in flight is taken here.
    """
    live_progress = await live_run_progress(job)
    return snapshot_receipt(
        job,
        state,
        control_token=control_token,
        lint_by_circuit=lint_by_circuit,
        live_progress=live_progress,
    )


def render_jobs_receipt_snapshot(
    action: Literal["status", "wait"],
    snapshot: ReceiptSnapshot,
    *,
    timed_out: bool | None = None,
    runs_cap: int = JOBS_PAGE_LIMIT,
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
) -> dict[str, Any]:
    """Render one complete jobs receipt from detached snapshot facts."""
    data = render_receipt_snapshot(
        snapshot,
        runs_cap=runs_cap,
        analysis_answer_channel=analysis_answer_channel,
        analysis_rows_cap=analysis_rows_cap,
    )
    data.update(
        {
            # Every job this server runs is an experiment. The key stays on the
            # wire because a client reads it to tell a receipt apart from the
            # error envelope, which says "unknown".
            "job_type": "experiment",
            "dialect": snapshot.dialect,
            "action": action,
            "analysis_status": snapshot.analysis_status,
            # Only 'wait' reports whether it ran out of time; 'status' never waited.
            **({"timed_out": timed_out} if timed_out is not None else {}),
        }
    )
    data["hint"] = _jobs_receipt_hint(snapshot, data)
    return finalize_receipt(data)


def _jobs_receipt_hint(snapshot: ReceiptSnapshot, data: dict[str, Any]) -> str:
    """The one next step this jobs receipt offers, in precedence order.

    A live job outranks a paged one: continuing the wait is what gets the rest
    of the runs in the first place. A paged one keeps the failure routes after
    its own, since a budget's floor pages every receipt. Whatever the receipt
    renderer already put on ``hint`` survives when neither applies.
    """
    if snapshot.status in NON_TERMINAL_LIVE_STATUSES:
        return (
            f"Job {snapshot.job_id} is still {snapshot.status}; continue with "
            f"jobs(action='wait', job_id='{snapshot.job_id}')."
        )
    if data["runs"]["truncated"]:
        return " ".join(
            [
                f"Run records are paged; continue with jobs(action='runs', "
                f"job_id='{snapshot.job_id}', cursor={data['runs']['next_cursor']!r}).",
                *_recovery_routes(snapshot),
            ]
        )
    return data.get("hint") or f"Job {snapshot.job_id} is {snapshot.status}."


def render_runs_envelope(
    snapshot: ReceiptSnapshot,
    *,
    offset: int = 0,
    limit: int | None = None,
    run_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Render one jobs(runs) envelope over a snapshot's full run records.

    ``limit=None`` returns every recorded run, which is also what makes the
    truncation hint below collapse to the complete-page wording. ``offset`` is
    the caller's decoded cursor position.
    """
    page = project_receipt_runs(
        snapshot,
        run_fields,
        lean_default=False,
        offset=offset,
        # This page's cursor continues this page, so it never renders empty.
        limit=None if limit is None else max(1, limit),
    )
    return {
        "action": "runs",
        "outcome": _jobs_outcome(snapshot),
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "dialect": snapshot.dialect,
        **page,
        "observations": [],
        "warnings": [],
        "failures": [],
        "hint": (
            "Use next_cursor to continue the run page."
            if page["truncated"]
            else f"Returned all recorded runs for job {snapshot.job_id}."
        ),
    }
