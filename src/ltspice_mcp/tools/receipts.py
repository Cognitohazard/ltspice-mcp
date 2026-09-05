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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any, Literal

from mcp import types

from ltspice_mcp.lib import response_budget, services
from ltspice_mcp.lib.experiment_types import (
    Completeness,
    ExperimentCase,
    ExperimentJob,
    ManifestEntry,
    SourceRecord,
)
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES
from ltspice_mcp.lib.log_parser import diagnostic_collapse_key
from ltspice_mcp.lib.pagination import page as _page
from ltspice_mcp.lib.pagination import page_of
from ltspice_mcp.lib.projection import keep_plan, project_row
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

Job = ExperimentJob

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
        "loaded libraries, or add a .lib/.include for it to the deck."
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
        "status": {"type": "string"},
        "raw": {"type": ["string", "null"]},
        "log": {"type": ["string", "null"]},
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
    },
    "required": [
        "declared",
        "expanded",
        "submitted",
        "produced",
        "failed",
        "cancelled",
        "skipped",
    ],
}

# The derived view of ``completeness``, and only that: the seven raw counters
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

RUN_EXPERIMENTS_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "job_id": {"type": ["string", "null"]},
        "request_id": {"type": "string"},
        "control_token": {"type": "string"},
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
_ReceiptRows = Callable[[dict[str, Any]], list[Any]]

# Rung 0's allowlist, shared by run_experiments and jobs status/wait because
# both render the same receipt envelope.
_TRIM_REMOVE_RECEIPT: tuple[str, ...] = ("analysis",)
# `source` is deliberately absent, and no other rung empties it either. It is
# not an identity echo the way the analysis envelope's `source_hashes` is: with
# `provenance` off it still carries the staging disclosures — the live,
# unstaged or unexplained manifest entries `_source_payload` keeps precisely so
# a caller is told about them — and with `provenance` on it carries an opt-in
# the caller asked for, which the trim rung's charter forbids revoking. Facts
# under one flag and an opt-in under the other leaves no rung a claim on it.

_RUN_BUDGET_NOTES = response_budget.Notes(
    cut="presentation was reduced; no run, failure, or analysis fact was dropped.",
    route=(
        "Ask again with a larger 'budget' for the full presentation, or continue "
        "through the returned cursor/jobs route."
    ),
)


def _receipt_row_pages(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every run page carried at the top level or under a receipt."""
    pages = [data] if isinstance(data.get("items"), list) else []
    runs = data.get("runs")
    if isinstance(runs, dict) and isinstance(runs.get("items"), list):
        pages.append(runs)
    return pages


def jobs_rows(data: dict[str, Any]) -> list[Any]:
    return [row for page in _receipt_row_pages(data) for row in page["items"]]


def _run_receipt_rows(data: dict[str, Any]) -> list[Any]:
    rows = jobs_rows(data)
    analysis_block = data.get("analysis")
    if isinstance(analysis_block, dict):
        result = analysis_block.get("result")
        if isinstance(result, dict):
            rows.extend(analyze.analysis_rows(result))
    return rows


def _degrade_receipt(data: dict[str, Any], rung: response_budget.Rung) -> None:
    """Apply presentation rungs to either public receipt envelope."""
    if rung.trim:
        response_budget.apply_trim(data, remove=_TRIM_REMOVE_RECEIPT)
    if rung.answer_channel:
        for page in _receipt_row_pages(data):
            for row in page["items"]:
                if isinstance(row, dict) and row.get("status") == "produced":
                    row.pop("raw", None)
                    row.pop("log", None)


async def negotiate_receipt(
    budget: ResponseBudget,
    build: ReceiptBuild,
    page_limit: int,
    *,
    rows: _ReceiptRows,
    notes: response_budget.Notes,
) -> ReceiptBuilt:
    """Render a receipt at the mildest shared budget rung that fits."""
    text = ""
    rendered: dict[str, Any] = {}
    built_for: tuple[int, bool, bool] | None = None

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        nonlocal text, rendered, built_for
        limit = page_limit
        if rung.shrink:
            limit = response_budget.RowMeasure.of(rows(rendered)).fit_limit(page_limit, rung)
        candidate = (limit, rung.answer_channel, rung.shrink)
        if built_for != candidate:
            rendered, text = build(limit, rung)
            built_for = candidate
        _degrade_receipt(rendered, rung)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(
        budget.tokens, render, notes, max_rung=budget.max_rung
    )
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
            budget,
            build,
            _RUN_PAGE_LIMIT,
            rows=_run_receipt_rows,
            notes=_RUN_BUDGET_NOTES,
        )
    result = format_response(text, data)
    result.is_error = is_error
    return result


def progress_from_completeness(completeness: Completeness) -> dict[str, int]:
    """Project durable accounting into the shared progress fact.

    The three DERIVED numbers only. ``completeness`` keeps all seven counters,
    unchanged and always — this block used to restate every one of them beside
    its own projection, so the same accounting arrived twice in one receipt and
    a third time in the hint. Dropping the copy removes no fact: each counter is
    one key away, in the ``completeness`` block itself.
    """
    return {
        "expanded": completeness.expanded,
        "terminal": completeness.terminal,
        "remaining": completeness.expanded - completeness.terminal,
    }


def finalize_receipt(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize completeness and attach its two public progress projections."""
    raw = data["completeness"]
    completeness = raw if isinstance(raw, Completeness) else Completeness(**raw)
    data["completeness"] = asdict(completeness)
    progress = progress_from_completeness(completeness)
    data["progress"] = progress
    response_budget.append_hint(
        data,
        f"Progress: {progress['terminal']}/{progress['expanded']} terminal; "
        f"{progress['remaining']} remaining.",
    )
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
    #: What kind of job the receipt describes. Every job is an experiment, so
    #: this is one value today; it stays on the wire because a client reads it
    #: to tell a receipt apart from the error envelope, which says "unknown".
    job_type: str
    status: str
    dialect: str | None
    control_token: str | None
    sources: tuple[SourceRecord | dict[str, Any], ...]
    lint: tuple[dict[str, Any], ...]
    runs_by_key: dict[tuple[str, int], dict[str, Any]]
    completeness: Completeness
    failures: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]
    artifacts: tuple[dict[str, Any], ...]
    analysis_status: str
    analysis_result: dict[str, Any] | None
    analysis_error: str | None
    analysis_observations: tuple[dict[str, Any], ...]
    analysis_request: dict[str, Any] | None

    @property
    def outcome(self) -> CallOutcome:
        """Receipt outcome derived only from copied status and completeness."""
        return _terminal_outcome(self)


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
        "failures": _render_failures(snapshot.failures),
        "observations": list(snapshot.observations),
        "warnings": [],
        "artifacts": list(snapshot.artifacts),
        "hint": (
            f"Experiment {snapshot.job_id} is still running; use jobs(wait) with this "
            "job_id to continue waiting."
            if snapshot.status not in TERMINAL_EXPERIMENT_STATUSES
            else _terminal_hint(snapshot, runs["truncated"])
        ),
    }
    # Cancel authority, only where a cancel can still do anything. A terminal
    # job has nothing left to stop, so the token there is bytes on every receipt
    # buying an action the lifecycle already refuses.
    emitted_control_token = control_token if control_token is not None else snapshot.control_token
    if emitted_control_token is not None and snapshot.status not in TERMINAL_EXPERIMENT_STATUSES:
        data["control_token"] = emitted_control_token
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
    return data


_FAILURE_CASE_ID_CAP = 10
"""How many case ids a collapsed failure row names before deferring to ``count``."""


def _render_failures(
    rows: tuple[dict[str, Any], ...] | list[dict[str, Any]],
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
        hint = _FAILURE_CODE_HINTS.get(code)
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


def _run_item(case: ExperimentCase) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "circuit": case.circuit,
        "assignments": case.assignments,
        "status": case.status,
        "raw": str(case.raw_file) if case.raw_file else None,
        "log": str(case.log_file) if case.log_file else None,
    }


def _project_run_rows(
    rows: list[dict[str, Any]],
    run_fields: list[str] | None,
    *,
    lean_default: bool,
) -> list[dict[str, Any]]:
    """Apply one request's run-row projection policy to rows the caller owns."""
    if run_fields:
        plan = keep_plan(run_fields)
        return [project_row(row, plan) for row in rows]
    if lean_default:
        # Lean default: a produced row's artifact paths are provenance the
        # analysis tools resolve by id (fetch them via jobs(runs) or
        # run_fields). Every other status keeps them — failures entries
        # carry only {case_id, code, message}, so the failed row's log path
        # is its diagnostic.
        for row in rows:
            if row["status"] == "produced":
                del row["raw"], row["log"]
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
    rows = _project_run_rows(
        [_run_item(case) for case in cases[: max(1, cap)]],
        run_fields,
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
    """
    rows = _project_run_rows(
        [dict(row) for row in snapshot.runs_by_key.values()],
        run_fields,
        lean_default=lean_default,
    )
    page_limit = max(1, len(rows)) if limit is None else limit
    return _page(rows, offset=offset, limit=page_limit)


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


# Case count at which a terminal receipt starts pointing at the in-process
# interface. Ten is past any spot-check and squarely in sweep/corner territory —
# the workload class where the per-call cost of going through the tool surface
# is large enough to be worth avoiding.
_API_POINTER_MIN_CASES = 10


def _terminal_hint(snapshot: ReceiptSnapshot, truncated: bool) -> str:
    """Every recovery route this receipt has, not the first one that matched.

    A server restart mid-run sets BOTH conditions: the abandoned cases become
    failures AND the attached analysis is marked failed. Under an exclusive
    ladder the failures branch won and the caller was never told that the runs
    that DID produce data are still analyzable by job_id — so the obvious move
    was to re-run an experiment whose results were sitting on disk.
    """
    if truncated:
        routes = [
            f"The inline run page is truncated; use jobs(runs) with job_id "
            f"{snapshot.job_id} for the remaining cases."
        ]
        return " ".join(routes + _api_pointer_route(snapshot))
    routes: list[str] = []
    if snapshot.analysis_status in {"failed", "cancelled"}:
        routes.append(
            f"The attached analysis {snapshot.analysis_status}; read analysis.error and "
            f"re-run it with analyze_results over job_id {snapshot.job_id} — the runs "
            "that produced data need no re-run."
        )
    if snapshot.failures:
        routes.append("Inspect failures and lint findings before retrying omitted cases.")
    if not routes:
        routes.append("All declared experiment cases reached terminality.")
    return " ".join(routes + _api_pointer_route(snapshot))


def _api_pointer_route(snapshot: ReceiptSnapshot) -> list[str]:
    """The second discovery surface for the Python API (the first is the
    initialize instructions): it lands exactly on the caller who is iterating —
    a many-case receipt is the loop shape where per-call wire overhead
    compounds and the Python API pays for itself. Appended on EVERY terminal
    experiment route, the truncated one included: a receipt big enough to
    truncate is the biggest loop of all."""
    if snapshot.completeness.expanded < _API_POINTER_MIN_CASES:
        return []
    return [
        "To run follow-up calls in a loop, use the in-process Python API: "
        "from ltspice_mcp.api import Api (same ops; api.reference() documents them)."
    ]


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
    job: Job,
    state: SessionState | None,
    *,
    control_token: str | None = None,
    lint_by_circuit: dict[str, list[dict[str, Any]]] | None = None,
) -> ReceiptSnapshot:
    """Copy a job's complete receipt state without suspending the event loop.

    The first job read through the last mutable copy occur in this synchronous
    call.  Outcome and guidance are intentionally absent from that live-read
    interval; renderers derive them only from the returned detached value.
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
    for case in job.cases:
        row = copy.deepcopy(_run_item(case))
        runs_by_key[(case.case_id, case.run_index)] = row
        for observation in case.observations:
            copied = copy.deepcopy(observation)
            key = (copied.get("code"), copied.get("detail"))
            if key not in seen_observations:
                observations.append(copied)
                seen_observations.add(key)

    analysis = job.analysis
    return ReceiptSnapshot(
        job_id=job.job_id,
        request_id=job.request_id,
        job_type="experiment",
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
        completeness=copy.deepcopy(job.completeness),
        failures=tuple(copy.deepcopy(job.failures)),
        observations=tuple(observations),
        artifacts=tuple(copy.deepcopy(job.artifacts)),
        analysis_status=analysis.status,
        analysis_result=copy.deepcopy(analysis.result),
        analysis_error=analysis.error,
        analysis_observations=tuple(copy.deepcopy(analysis.observations)),
        analysis_request=copy.deepcopy(analysis.request),
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
            "job_type": snapshot.job_type,
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
    of the runs in the first place. Whatever the receipt renderer already put
    on ``hint`` survives when neither applies.
    """
    if snapshot.status in NON_TERMINAL_LIVE_STATUSES:
        return (
            f"Job {snapshot.job_id} is still {snapshot.status}; continue with "
            f"jobs(action='wait', job_id='{snapshot.job_id}')."
        )
    if data["runs"]["truncated"]:
        return (
            f"Run records are paged; continue with jobs(action='runs', "
            f"job_id='{snapshot.job_id}', cursor={data['runs']['next_cursor']!r})."
        )
    return data.get("hint") or f"Job {snapshot.job_id} is {snapshot.status}."


def render_runs_envelope(
    snapshot: ReceiptSnapshot,
    *,
    offset: int = 0,
    limit: int | None = None,
) -> dict[str, Any]:
    """Render one jobs(runs) envelope over a snapshot's full run records.

    ``limit=None`` returns every recorded run, which is also what makes the
    truncation hint below collapse to the complete-page wording. ``offset`` is
    the caller's decoded cursor position.
    """
    page = project_receipt_runs(
        snapshot,
        None,
        lean_default=False,
        offset=offset,
        limit=limit,
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
