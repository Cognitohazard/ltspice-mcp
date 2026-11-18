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
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

from ltspice_mcp.errors import PathSecurityError
from ltspice_mcp.lib import response_budget, services
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.continuation import call_to_continue
from ltspice_mcp.lib.experiment_runner import live_run_progress
from ltspice_mcp.lib.experiment_types import (
    ACTIVE_CASE_STATUSES,
    TERMINAL_CASE_STATUSES,
    Completeness,
    ExperimentCase,
    ExperimentCaseStatus,
    ExperimentJob,
    ExperimentStatus,
    ManifestEntry,
    SourceRecord,
    sample_index,
)
from ltspice_mcp.lib.job_types import NON_TERMINAL_LIVE_STATUSES, TERMINAL_STATUSES
from ltspice_mcp.lib.log_conditions import reply_fields
from ltspice_mcp.lib.log_parser import diagnostic_collapse_key
from ltspice_mcp.lib.log_types import given_no_value
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pagination import page_of
from ltspice_mcp.lib.presenter import ListCut, Record
from ltspice_mcp.lib.projection import (
    absent_path_warnings,
    keep_plan,
    project_row,
    resolve_field_paths,
)
from ltspice_mcp.lib.raw_parser import other_raw_path
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS, NEVER, rows
from ltspice_mcp.lib.result_observations import measurements_failed_in_every_run
from ltspice_mcp.lib.sheet_results import run_by_ltspice, sheets_reported
from ltspice_mcp.lib.simulator_build import SimulatorExecutable
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analyze
from ltspice_mcp.tools._base import (
    CUT_SCHEMA,
    DIAGNOSTICS_COVERAGE_SCHEMA,
    ENVELOPE_ERROR_SCHEMA,
    FINDING_LIST_CUT,
    FINDING_SCHEMA,
    OUTCOME_SCHEMA,
    RUN_DIAGNOSTICS_SCHEMA,
    CallOutcome,
    JobAnswerParts,
    failures_schema,
    outcome_of,
    page_schema,
)

#: The most circuit groups one ``jobs(action="list")`` reply sends.
JOBS_PAGE_LIMIT = 50

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

#: How a standing note already sent in full this session is sent again: by its
#: code, with this count in place of its text (``lib/said_once.py``).
SAID_BEFORE_SCHEMA: dict[str, Any] = {
    "type": "integer",
    "minimum": 1,
    "description": (
        "Present in place of the note's text when this session was already sent it "
        "in full: how many replies carried it before. jobs(action='status') sends "
        "every note in full."
    ),
}

OBSERVATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "code": {"type": "string"},
        "kind": {"type": "string"},
        "detail": {"type": "string"},
        "evidence": {},
        "said_before": SAID_BEFORE_SCHEMA,
    },
    # A standing note sent before carries said_before in place of detail.
    "required": ["code", "kind"],
}

#: A receipt failure names the case that failed, not a stage (see Envelope).
CASE_FAILURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "case_ids": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
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
        "LTSPICE_MCP_NGBEHAVIOR=hsa) and restart the server. The server runs "
        "ngspice without any .spiceinit, so a 'set ngbehavior' there has no effect."
    ),
    "ngspice_mode_mismatch": (
        "ngspice read the deck in other compatibility modes than the ones the server "
        "passed and read the deck in; evidence names both. The server already skips any "
        ".spiceinit, so look for a 'set ngbehavior' in the installation's spinit (found "
        "through SPICE_SCRIPTS or ngspice's own scripts folder). Remove it, or set "
        "[simulator] ngbehavior to the same mode and restart the server."
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

#: Why a run row's measurement fields were not read (``measurements_coverage``).
MEASUREMENTS_COVERAGE = ("not_read", "log_unread", "simulator_unsupported")

#: A list of steps of one measurement: the run's step count bounds it.
_STEPS_SCHEMA: dict[str, Any] = {"type": "array", "items": {"type": "integer"}, CUT: NEVER}

RUN_RECORD_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "run_index": {"type": "integer"},
        "circuit": {"type": "string"},
        "assignments": {"type": "object"},
        "native_statistics": {"type": "object"},
        "status": {
            "type": "string",
            "enum": list(get_args(ExperimentCaseStatus)),
            "description": (
                "The case's own status, not the job's: produced once the simulator wrote "
                "its output; failed, cancelled or skipped when it ended without it; "
                "queued, submitted or running before."
            ),
        },
        "raw": {"type": ["string", "null"]},
        "log": {"type": ["string", "null"]},
        "other_raws": {"type": "object", "additionalProperties": {"type": "string"}},
        "simulator_version": {"type": ["string", "null"]},
        "diagnostics": RUN_DIAGNOSTICS_SCHEMA,
        "diagnostics_coverage": DIAGNOSTICS_COVERAGE_SCHEMA,
        "failed_measurements": {
            "type": "array",
            "items": {"type": "string"},
            # One name per .meas the run printed failed: the deck bounds it,
            # and leaving one out would read as taken.
            CUT: NEVER,
            "description": (
                "The .meas LTspice printed as failed in this run, by name in the log's "
                "order. The measurements recipe reads the values."
            ),
        },
        "failed_measurement_steps": {
            "type": "object",
            "additionalProperties": _STEPS_SCHEMA,
            "description": (
                "For a failed_measurements name that failed at some steps of a stepped "
                "run only: those steps, from 0. A name not here failed at every step."
            ),
        },
        "measurements_not_established": {
            "type": "object",
            "additionalProperties": _STEPS_SCHEMA,
            "description": (
                "Steps, from 0, at which a .meas's printed value is not established "
                "as its result (LTspice XVII prints 0 for a step whose measurement "
                "failed); the measurements recipe reads them as null."
            ),
        },
        "measurements_not_printed": {
            "type": "array",
            "items": {"type": "string"},
            # One name per .meas the deck requests: the deck bounds it.
            CUT: NEVER,
            "description": (
                "The .meas the deck requests that the run's log printed nothing for, "
                "not even a failure."
            ),
        },
        "measurements_coverage": {
            "type": "string",
            "enum": list(MEASUREMENTS_COVERAGE),
            "description": (
                "Present when the row's failed_measurements were not read: "
                "'not_read', a run recorded before the server kept them; "
                "'log_unread', the log's measurements could not be read; "
                "'simulator_unsupported', a simulator whose measure output the server "
                "does not read."
            ),
        },
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

#: How a list of run rows is cut: whole rows, each named by its case and run,
#: which ``run_fields`` may project out of the reply.
RUN_ROWS_CUT: dict[str, Any] = rows("case_id", "run_index", projected_by="run_fields")

RUNS_PAGE_SCHEMA: dict[str, Any] = page_schema(
    {"type": "array", "items": RUN_RECORD_SCHEMA, CUT: RUN_ROWS_CUT}
)

#: Every top-level key a run row can carry, read off its schema: the roots a
#: ``run_fields`` path may start at.
RUN_ROW_KEYS: tuple[str, ...] = tuple(RUN_RECORD_SCHEMA["properties"])


def resolve_run_fields(paths: list[str]) -> list[str]:
    """``run_fields`` as the run-row projection reads them, on
    ``run_experiments`` and ``jobs(runs)`` alike: rooted at a run row's own
    keys, each read once. ``ValueError`` for a path no run row can carry."""
    return list(dict.fromkeys(resolve_field_paths(paths, RUN_ROW_KEYS, argument="run_fields")))


#: A receipt's own ``status``, which is the job's: a separate vocabulary from
#: its run rows' case statuses.
JOB_STATUS_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        f"The job's status: one of {', '.join(get_args(ExperimentStatus))}; unknown in "
        "an error reply that read no job. A run row's status is its case's own: a case "
        "that ran is produced."
    ),
}

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

#: Marks a reply field that belongs to how the job's answer was delivered (a
#: wait's ``timed_out``, a run's ``replayed``, the jobs ``action``), not to the
#: answer: two deliveries of one finished job differ in these and in nothing else.
DELIVERY = "x-delivery"

#: Every job this server runs is an experiment. The key stays on the wire
#: because a client reads it to tell a receipt from the error envelope.
JOB_TYPE_SCHEMA: dict[str, Any] = {"type": "string"}

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
        "replayed": {"type": "boolean", DELIVERY: True},
        "status": JOB_STATUS_SCHEMA,
        "job_type": JOB_TYPE_SCHEMA,
        "dialect": {"type": ["string", "null"]},
        "analysis_status": {"type": "string"},
        "outcome": OUTCOME_SCHEMA,
        "source": {
            "type": "array",
            # One entry per circuit the request names: the request bounds it.
            CUT: NEVER,
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "path": {"type": "string"},
                    "sha256": {"type": "string"},
                    "staged_deck": {"type": "string"},
                    "manifest": {"type": "array", "items": _MANIFEST_SCHEMA, CUT: ITEMS},
                    "linter_version": {"type": "string"},
                    "simulator": {"type": "string"},
                    "dialect": {"type": ["string", "null"]},
                    "staged_files": {"type": "integer"},
                    "said_before": SAID_BEFORE_SCHEMA,
                },
                # Only the circuit is required: the digests, staging paths and
                # full manifest are provenance, emitted when 'provenance' is
                # set, and an entry this session was already sent in full is
                # sent as its circuit and said_before. Otherwise every entry
                # carries circuit, path, simulator and dialect.
                "required": ["circuit"],
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
            # One entry per circuit the request names: the request bounds it.
            CUT: NEVER,
            "items": {
                "type": "object",
                "properties": {
                    "circuit": {"type": "string"},
                    "findings": {"type": "array", "items": FINDING_SCHEMA, CUT: FINDING_LIST_CUT},
                    "said_before": SAID_BEFORE_SCHEMA,
                },
                # An entry already sent in full is its circuit and said_before.
                "required": ["circuit"],
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
                    CUT: NEVER,
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
        "observations": {"type": "array", "items": OBSERVATION_SCHEMA, CUT: NEVER},
        "warnings": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
        # What the job recorded beside its runs: the job bounds it.
        "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA, CUT: NEVER},
        "hint": {"type": "string"},
        "error": ENVELOPE_ERROR_SCHEMA,
        "cut": CUT_SCHEMA,
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

#: Where a receipt's run rows are, as the presenter names a list.
RUN_ROWS = "/runs/items"

#: The kind of record run rows are read back from: the job's own record,
#: which holds every row, so a cut of them stores nothing.
JOB_RUNS = "job_runs"


def run_rows_record(job_id: str) -> Record:
    """The record a job's run rows are read back from: ``jobs(action="runs")``."""
    return Record(JOB_RUNS, job_id)


# The attached analysis is that tool's own envelope, trimmed by that tool's own
# allowlist (`analyze.trim_analysis`) until analyze_results moves onto the
# presenter: its `source_hashes` is one identity row per run, so a receipt that
# kept it grew by every case it ran whatever the budget. The receipt's own
# lists are the presenter's: its run rows are cut as rows, read back from the
# job's record, and every other list is sent whole.


def _attached_result(data: dict[str, Any]) -> dict[str, Any] | None:
    """The attached analysis's rendered result, when the receipt carries one."""
    analysis_block = data.get("analysis")
    result = analysis_block.get("result") if isinstance(analysis_block, dict) else None
    return result if isinstance(result, dict) else None


#: Where a caller reads more of an attached waveform the receipt thinned.
_THINNED_ATTACHED_ROUTE = (
    "analyze_results over this job_id with a larger max_points returns more points, "
    "and format 'csv' writes every sample."
)


def _with_one_run_row(data: dict[str, Any]) -> dict[str, Any]:
    """``data`` as the presenter's smallest cut leaves it: one run row. What
    the attached analysis is fitted against, since the rows are cut after."""
    runs = data.get("runs")
    if not isinstance(runs, dict) or not isinstance(runs.get("items"), list):
        return data
    return {**data, "runs": {**runs, "items": runs["items"][:1]}}


def _degrade_receipt(data: dict[str, Any], rung: response_budget.Rung) -> list[str]:
    """Apply a ladder rung to a receipt's attached analysis, by that tool's
    own trim, in place.

    Returns the blocks the trim emptied of content. An attached analysis's
    inline waveforms are then thinned to fit at every rung, whatever
    ``max_points`` the request set, measured with the receipt cut to one run
    row as the presenter cuts it: the receipt's rows preview what
    ``analyze_results`` over the job returns, and that call sizes them.
    """
    emptied: list[str] = []
    if rung.trim:
        result = _attached_result(data)
        if result is not None:
            emptied = [f"analysis.result.{key}" for key in analyze.trim_analysis(result)]
            results = result.get("results")
            if isinstance(results, dict):
                analyze.fit_waveform_samples(
                    _with_one_run_row(data), results, rung, list(results), _THINNED_ATTACHED_ROUTE
                )
    return emptied


def trim_attached_analysis(
    data: dict[str, Any], rung: response_budget.Rung, rows: int | None, target: int
) -> None:
    """Fit a receipt's attached analysis to ``rung`` by ``analyze_results``'
    own trimming, in place, and say what it took. ``rows`` is the count the
    analysis was rendered with (None: all). Until that tool moves onto the
    presenter, this is how its envelope is shortened inside a receipt."""
    if _attached_result(data) is None:
        return
    emptied = _degrade_receipt(data, rung)
    parts: list[str] = []
    if emptied:
        parts.append("emptied " + ", ".join(emptied))
    if rung.answer_channel:
        parts.append("rendered on its answer channel, without the opt-ins the request set")
    if rows is not None:
        parts.append(f"its rows held to {rows} per list")
    if parts:
        data["observations"] = [
            *data.get("observations", []),
            response_budget.attached_trimmed_observation(target, parts, data.get("job_id")),
        ]


def _say_where_the_runs_are(
    run_fields: list[str] | None,
) -> Callable[[str], Callable[[dict[str, Any], Mapping[str, ListCut], Any, str | None], None]]:
    """The words a receipt adds when its run rows are cut: its ``runs`` page
    fields set from the cut, and the call that reads the rest, first in its
    hint."""

    def for_job(
        job_id: str,
    ) -> Callable[[dict[str, Any], Mapping[str, ListCut], Any, str | None], None]:
        def annotate(
            reply: dict[str, Any], cut: Mapping[str, ListCut], _stored: Any, _paged: str | None
        ) -> None:
            entry = cut.get(RUN_ROWS)
            if entry is None:
                return
            runs = reply["runs"]
            runs["returned"] = entry.returned
            runs["truncated"] = entry.next is not None
            runs["next_cursor"] = entry.next
            if entry.next is None:
                return
            sentence = _the_rest_of_the_rows(entry, job_id, run_fields)
            hint = reply.get("hint")
            reply["hint"] = f"{sentence} {hint}" if isinstance(hint, str) and hint else sentence

        return annotate

    return for_job


def _the_rest_of_the_rows(entry: ListCut, job_id: str, run_fields: list[str] | None) -> str:
    """The one sentence a reply cut short of a job's run rows says of the
    rest, on a receipt and a runs page alike."""
    read_on = call_to_continue("jobs", {"action": "runs", "job_id": job_id, "cursor": entry.next})
    same = " with the same run_fields" if run_fields else ""
    return (
        f"{entry.total - entry.offset - entry.returned} of {entry.total} run rows are not "
        f"in this reply; read them from the job's record with {read_on}{same}."
    )


def receipt_parts(run_fields: list[str] | None = None) -> JobAnswerParts:
    """What the reply layer needs to deliver a job's answer
    (``tools/_base.present_job_answer``): where its run rows are and the
    record they are read back from, the words a cut of them adds, the
    attached analysis's own trimming, and the measurements route, which is a
    standing note."""
    return JobAnswerParts(
        rows=RUN_ROWS,
        record=run_rows_record,
        annotate=_say_where_the_runs_are(run_fields),
        trim_attached=trim_attached_analysis,
        attached=lambda data: _attached_result(dict(data)) is not None,
        route=measurements_route,
        short_route=_SHORT_MEASUREMENTS_ROUTE,
    )


#: How many assign points an ``identical_samples`` observation names; the
#: rest are counted.
_SAMPLE_POINT_CAP = 10

#: Where a draw is recorded in a case's assignments, for a case recorded
#: before it kept its assign values apart (``ExperimentCase.assigned``).
_DRAWN_PREFIXES = ("random:", "_random_")
_DRAWN_INSTANCE_FIELDS = frozenset(
    {("mismatch", None), ("parameter", "delvto"), ("parameter", "mulu0")}
)


def _assign_point(case: ExperimentCase) -> dict[str, Any]:
    """The ``assign`` values ``case`` ran at: what it recorded, or, for a case
    recorded before it kept them, its assignments without the draws. A draw
    this misses keeps each sample at its own point, so it is compared with
    none rather than with another point's samples."""
    if case.assigned:
        return case.assigned
    point = {}
    for key, value in case.assignments.items():
        if key.startswith(_DRAWN_PREFIXES):
            continue
        if key.startswith("instance:"):
            try:
                *_path, attribute, parameter = json.loads(key[len("instance:") :])
            except (ValueError, TypeError):
                continue
            if attribute == "model_parameter" or (attribute, parameter) in _DRAWN_INSTANCE_FIELDS:
                continue
        point[key] = value
    return point


def _sample_observations(cases: Sequence[ExperimentCase]) -> list[dict[str, Any]]:
    """Where a job's samples all computed the same numbers.

    Each produced sample's raw carries a digest of what it computed
    (``raw_parser.result_digest``). Samples are compared one ``assign`` point
    at a time: at a point the draws cannot move (a supply of 0 V) every
    sample computes the same numbers while other points differ. A point is
    reported only where every sample compared there computed one result:
    two results among many still say some draw changed what was saved.
    "All samples" means every sample of every point ran and each point
    computed one result. A produced sample with no digest, at a point with
    another produced sample, is counted as not compared.
    """
    points: dict[tuple[str, str], tuple[dict[str, Any], list[ExperimentCase]]] = {}
    for case in cases:
        if sample_index(case) is None:
            continue
        point = _assign_point(case)
        key = (case.circuit, json.dumps(point, sort_keys=True, default=str))
        points.setdefault(key, (point, []))[1].append(case)
    running = any(case.status in ("queued", *ACTIVE_CASE_STATUSES) for case in cases)
    alike: dict[str, list[dict[str, Any]]] = {}
    compared_points: dict[str, int] = {}
    whole: dict[str, bool] = {}
    unread = 0
    # In the order the cases ran, so a capped list names the first points.
    for (circuit, _), (point, members) in points.items():
        produced = [case for case in members if case.status == "produced"]
        digests = [case.result_digest for case in produced if case.result_digest is not None]
        if len(produced) > 1:
            unread += len(produced) - len(digests)
        one_result = len(digests) > 1 and len(set(digests)) == 1
        whole[circuit] = whole.get(circuit, True) and one_result and len(digests) == len(members)
        if len(digests) > 1:
            compared_points[circuit] = compared_points.get(circuit, 0) + 1
        if one_result:
            alike.setdefault(circuit, []).append(
                {
                    "circuit": circuit,
                    **({"assigned": point} if point else {}),
                    "samples": len(digests),
                }
            )
    observations: list[dict[str, Any]] = []
    if alike:
        # Each circuit is one of three kinds, and the detail says which.
        every = [c for c in alike if whole[c]]
        everywhere = [
            c for c, rows in alike.items() if len(rows) == compared_points[c] and not whole[c]
        ]
        partly = [c for c, rows in alike.items() if len(rows) < compared_points[c]]
        groups: list[dict[str, Any]] = []
        named = 0
        listed = 0
        for circuit, rows in alike.items():
            if circuit not in partly and len(rows) > 1:
                groups.append({"circuit": circuit, "points": len(rows)})
                continue
            for row in rows:
                named += 1
                if listed < _SAMPLE_POINT_CAP:
                    groups.append(row)
                    listed += 1
        evidence: dict[str, Any] = {"groups": groups}
        if named > listed:
            evidence["points_truncated"] = named
        so_far = " so far" if running else ""
        sentences = []
        if every:
            sentences.append(
                f"All samples of {', '.join(every)} computed the same results{so_far}."
            )
        if everywhere:
            sentences.append(
                f"Every sample of {', '.join(everywhere)} that was compared computed the "
                f"same results{so_far}, though not every sample was compared."
            )
        if partly:
            sentences.append(
                f"Samples of {', '.join(partly)} computed the same results{so_far} at the "
                "assign points listed for them and differ at the others."
            )
        sentences.append(
            "Where samples computed the same results, their draws changed nothing the "
            "simulator saved"
            + (
                ": check that the rules reach a part the deck saves, or that the PDK's "
                "statistical models are switched on."
                if every or everywhere
                else "."
            )
        )
        observations.append(
            {
                "code": "identical_samples",
                "kind": "sampling",
                "detail": " ".join(sentences),
                "evidence": evidence,
            }
        )
    if unread:
        plural = unread != 1
        observations.append(
            {
                "code": "samples_not_compared",
                "kind": "sampling",
                "detail": (
                    f"{unread} produced sample{'s carry' if plural else ' carries'} no digest "
                    f"of what {'they' if plural else 'it'} computed (a job recorded before "
                    "samples were compared, or a raw the reader cannot walk to its end), so "
                    f"whether {'they differ' if plural else 'it differs'} from the other "
                    "samples at the same assign point was not checked."
                ),
                "evidence": {"count": unread},
            }
        )
    return observations


def _measurement_observations(cases: Sequence[ExperimentCase]) -> list[dict[str, Any]]:
    """Each circuit's ``.meas`` that gave no value in any run whose
    measurements were read (``result_observations.measurements_failed_in_every_run``),
    from what each case kept when its run ended (``log_types.given_no_value``)."""
    by_circuit: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        facts = case.measurement_facts
        if _ran(case) and facts is not None and "coverage" not in facts:
            by_circuit.setdefault(case.circuit, []).append(facts)
    observations: list[dict[str, Any]] = []
    for circuit, read in by_circuit.items():
        failed = [given_no_value(facts) for facts in read]
        requested = next((facts["requested"] for facts in read if "requested" in facts), None)
        found = measurements_failed_in_every_run(
            failed, requested, circuit=circuit if len(by_circuit) > 1 else None
        )
        if found is not None:
            observations.append(dict(found))
    return observations


def _results_in_the_job_folder(job_id: str, sheets: list[str]) -> dict[str, Any]:
    """Say that a sheet's run left its results in the job folder, not beside
    the sheet, where LTspice's own window opens a sheet's results."""
    named = ", ".join(sheets)
    return {
        "code": "results_not_beside_sheet",
        "kind": "provenance",
        "detail": (
            f"The results of {named} are in the job folder; nothing was written beside "
            "the sheet. LTspice's window opens a sheet's results from beside it, under "
            f'its name, so it does not show these. jobs(action="place_beside_sheet", '
            f'job_id="{job_id}", case_id=...) copies a case\'s there.'
        ),
        "evidence": {"sheets": sheets},
    }


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

    _projected: dict[tuple[str, ...] | None, tuple[dict[str, Any], tuple[str, ...]]] = field(
        default_factory=dict, compare=False, repr=False
    )
    """Each ``run_fields`` projection of the run rows, rendered once: a reply
    builds its answer once per presentation it tries, and no presentation
    changes the rows."""

    def projected_runs(
        self, run_fields: list[str] | None
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Every run row projected by ``run_fields``, as one page, with the
        warnings the projection gave; rendered once per projection. The page
        is shared: copy it before changing it."""
        key = None if run_fields is None else tuple(run_fields)
        if key not in self._projected:
            said: list[str] = []
            rows = list(self.runs_by_key.values())
            page = page_of(rows, offset=0, total=len(rows))
            page["items"] = _project_run_rows(
                [dict(row) for row in page["items"]],
                run_fields,
                native_by_key=self.native_by_key,
                warnings=said,
            )
            self._projected[key] = (page, tuple(said))
        return self._projected[key]

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
    analysis_fields: list[str] | None = None,
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
    analysis_served: Collection[str] | None = None,
) -> dict[str, Any]:
    """A job's complete answer at the moment of ``snapshot``, with every run
    row, as every delivery of it sends it: the run receipt, a wait, a status.
    It knows nothing of a wait; what a delivery adds (``timed_out``,
    ``replayed``, ``action``) is the delivery's to add.

    The snapshot's leaves are already detached from the job, and rendering
    never writes through them — the only row edits build or own their dicts —
    so the envelope shares them rather than copying them per render.
    """
    run_warnings: list[str] = []
    runs = project_receipt_runs(snapshot, run_fields, warnings=run_warnings)
    data: dict[str, Any] = {
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "job_type": "experiment",
        "status": snapshot.status,
        "analysis_status": snapshot.analysis_status,
        "dialect": snapshot.dialect,
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
        "warnings": run_warnings,
        "artifacts": list(snapshot.artifacts),
        "hint": (
            f"Experiment {snapshot.job_id} is still running; use jobs(wait) with this "
            "job_id to continue waiting."
            if snapshot.status not in TERMINAL_STATUSES
            else _terminal_hint(snapshot)
        ),
    }
    emitted_control_token = control_token if control_token is not None else snapshot.control_token
    if emitted_control_token is not None and (
        snapshot.lineage is not None or snapshot.status not in TERMINAL_STATUSES
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
                served=analysis_served,
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


def run_row_diagnostics(case: ExperimentCase) -> dict[str, Any]:
    """What the simulator said about ``case``'s run, as its row carries it:
    ``diagnostics`` and, when that is not the reading of the run's own simulator
    and build, ``diagnostics_coverage``. A case that never launched, or whose run
    has not ended, says nothing; one that ended before the server classified
    logs says its log was not classified."""
    if not _ran(case):
        return {}
    found = case.log_diagnostics
    if found is None:
        return {"diagnostics_coverage": "not_classified"}
    return reply_fields(found)


#: A case's measurement facts (``ExperimentCase.measurement_facts``) by the
#: run-row field each is sent as.
_MEASUREMENT_ROW_FIELDS = {
    "failed": "failed_measurements",
    "failed_steps": "failed_measurement_steps",
    "not_established": "measurements_not_established",
    "not_printed": "measurements_not_printed",
    "coverage": "measurements_coverage",
}


def run_row_measurements(case: ExperimentCase) -> dict[str, Any]:
    """What ``case``'s run did not give of its measurements, as its row
    carries it. A case that never launched, or whose run has not ended, says
    nothing; one that ended before the server kept its measurements says they
    were not read, so its silence never reads as "none failed"."""
    if not _ran(case):
        return {}
    facts = case.measurement_facts
    if facts is None:
        return {"measurements_coverage": "not_read"}
    return {
        _MEASUREMENT_ROW_FIELDS[key]: value
        for key, value in facts.items()
        if key in _MEASUREMENT_ROW_FIELDS
    }


def _ran(case: ExperimentCase) -> bool:
    """Whether ``case``'s simulator was launched and its run has ended."""
    return (
        case.submitted_at is not None
        and case.status in TERMINAL_CASE_STATUSES
        and case.failure_code != "submission_failed"
    )


def _run_item(case: ExperimentCase) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "run_index": case.run_index,
        "circuit": case.circuit,
        "assignments": case.assignments,
        "status": case.status,
        "raw": str(case.raw_file) if case.raw_file else None,
        "log": str(case.log_file) if case.log_file else None,
        "other_raws": (
            {kind: str(other_raw_path(case.raw_file, kind)) for kind in case.other_raws}
            if case.raw_file
            else {}
        ),
        "simulator_version": case.simulator_version,
        **run_row_diagnostics(case),
        **run_row_measurements(case),
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
    warnings: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Render native evidence and project only the rows this page carries.

    With no ``run_fields`` every row keeps every key, whatever its status, so a
    key read off a row is never missing because the run produced. A path no
    row on the page reaches is said in ``warnings``, naming the keys there."""
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
        if warnings is not None and run_fields:
            warnings.extend(
                absent_path_warnings(
                    rows, run_fields, argument="run_fields", rows_name="run row on this page"
                )
            )
        return [project_row(row, plan) for row in rows]
    return rows


def runs_page(
    cases: list[ExperimentCase],
    run_fields: list[str] | None = None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Every run row of ``cases``, projected by ``run_fields``, as one page.

    A ``run_fields`` path no row reaches is said in ``warnings``, as
    :func:`project_receipt_runs` says it for a receipt read from a job.
    """
    selected = cases
    native_by_key = {}
    for case in selected:
        if case.native_statistics is not None:
            case.native_statistics.validate()
            native_by_key[(case.case_id, case.run_index)] = case.native_statistics
    rows = _project_run_rows(
        [_run_item(case) for case in selected],
        run_fields,
        native_by_key=native_by_key,
        warnings=warnings,
    )
    return page_of(rows, offset=0, total=len(cases))


def project_receipt_runs(
    snapshot: ReceiptSnapshot,
    run_fields: list[str] | None,
    *,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    """Every run row of the snapshot, projected by one invocation's
    ``run_fields``, as one page: the job's whole table, which the presenter
    cuts as rows (:data:`RUN_ROWS`). Without ``run_fields`` every row is whole.
    """
    page, said = snapshot.projected_runs(run_fields)
    if warnings is not None:
        warnings.extend(said)
    return {**page, "items": list(page["items"])}


def _terminal_outcome(snapshot: ReceiptSnapshot) -> CallOutcome:
    """An experiment receipt's outcome, read off its own status and counters.

    The one outcome every job receipt reports, whichever action renders it:
    ``run_experiments``, ``jobs`` status and wait through
    ``ReceiptSnapshot.outcome``, and ``jobs`` runs.

    ``delivered=False``: an experiment that failed produced nothing to read, so
    a failure here IS the call's outcome rather than one item among many. A
    cancelled experiment is partial whatever the counters say — a cancel
    landing after every run but before the analysis leaves them fully
    reconciled, and one landing before expansion leaves them all at zero. An
    interrupted one is read off its counters: the cases whose runs wrote their
    results before the process running the job stopped count as produced, and
    are read by job.
    """
    return outcome_of(
        snapshot.status == "failed",
        partial=snapshot.status == "cancelled" or snapshot.completeness.fell_short,
        in_progress=snapshot.status not in TERMINAL_STATUSES,
        delivered=False,
    )


def _terminal_hint(snapshot: ReceiptSnapshot) -> str:
    """Every recovery route this receipt has, not the first one that matched.

    A process that stops mid-run sets BOTH conditions: the abandoned cases become
    failures AND the attached analysis is marked failed. Under an exclusive
    ladder the failures branch won and the caller was never told that the runs
    that DID produce data are still analyzable by job_id — so the obvious move
    was to re-run an experiment whose results were sitting on disk. Run rows a
    reply cut are one more route, which the presenter's annotation puts first.
    """
    routes: list[str] = list(_recovery_routes(snapshot))
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
    if any(
        key in row
        for row in snapshot.runs_by_key.values()
        for key in (
            "failed_measurements",
            "measurements_not_established",
            "measurements_not_printed",
        )
    ):
        routes.append(measurements_route(snapshot.job_id))
    return routes


def measurements_route(job_id: str) -> str:
    """The sentence of a receipt's hint that routes to the ``measurements``
    recipe: a standing note, said in full once per session
    (``lib/said_once.py``)."""
    return (
        "A run row names a .meas that gave no value (failed_measurements, "
        "measurements_not_established, measurements_not_printed); read every value "
        "with analyze_results("
        f'sources=[{{"job_id": "{job_id}"}}], recipes=[{{"metric": '
        '"measurements"}]), or api.analyze_results in Python.'
    )


#: The measurements route once this session was sent it in full.
_SHORT_MEASUREMENTS_ROUTE = (
    "A run row names a .meas that gave no value; the measurements recipe over this "
    "job reads every value (said before)."
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
    their evidence is rendered once per projection of the whole table
    (``ReceiptSnapshot.projected_runs``), off the loop.

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
    # A sheet the job's own placement reported on has its own observation.
    reported = sheets_reported(job.observations)
    sheets = sorted(
        {
            case.circuit_path.name
            for case in job.cases
            if case.status == "produced"
            and is_asc(case.circuit_path)
            and str(case.circuit_path) not in reported
        }
    )
    if sheets and run_by_ltspice(job.simulator):
        observations.append(_results_in_the_job_folder(job.job_id, sheets))
    observations.extend(_sample_observations(job.cases))
    observations.extend(_measurement_observations(job.cases))

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
    analysis_answer_channel: bool = False,
    analysis_rows_cap: int | None = None,
    analysis_served: Collection[str] | None = None,
) -> dict[str, Any]:
    """One jobs delivery of a job's complete answer: the answer
    (:func:`render_receipt_snapshot`) and the delivery's own fields."""
    data = render_receipt_snapshot(
        snapshot,
        analysis_answer_channel=analysis_answer_channel,
        analysis_rows_cap=analysis_rows_cap,
        analysis_served=analysis_served,
    )
    data.update(
        {
            "action": action,
            # Only 'wait' reports whether it ran out of time; 'status' never waited.
            "timed_out": timed_out,
        }
    )
    data["hint"] = _jobs_receipt_hint(snapshot, data)
    return finalize_receipt(data)


def _jobs_receipt_hint(snapshot: ReceiptSnapshot, data: dict[str, Any]) -> str:
    """The one next step this jobs receipt offers: continuing the wait for a
    live job, else whatever the receipt renderer put on ``hint``."""
    if snapshot.status in NON_TERMINAL_LIVE_STATUSES:
        return (
            f"Job {snapshot.job_id} is still {snapshot.status}; continue with "
            f"jobs(action='wait', job_id='{snapshot.job_id}')."
        )
    return data.get("hint") or f"Job {snapshot.job_id} is {snapshot.status}."


def render_runs_envelope(
    snapshot: ReceiptSnapshot,
    *,
    run_fields: list[str] | None = None,
) -> dict[str, Any]:
    """One job's ``jobs(action="runs")`` answer: every recorded run, read
    from the job as it stands, with its status and progress, which the
    presenter cuts as rows (:func:`say_which_rows`)."""
    run_warnings: list[str] = []
    page = project_receipt_runs(snapshot, run_fields, warnings=run_warnings)
    return {
        "action": "runs",
        "outcome": snapshot.outcome,
        "job_id": snapshot.job_id,
        "request_id": snapshot.request_id,
        "status": snapshot.status,
        "progress": progress_from_completeness(snapshot.completeness),
        "dialect": snapshot.dialect,
        **page,
        "observations": [],
        "warnings": run_warnings,
        "failures": [],
        "hint": f"Returned all recorded runs for job {snapshot.job_id}.",
    }


#: Where a ``jobs(action="runs")`` answer's rows are, as the presenter names a list.
RUNS_PAGE_ROWS = "/items"


def say_which_rows(
    job_id: str, run_fields: list[str] | None
) -> Callable[[dict[str, Any], Mapping[str, ListCut], Any, str | None], None]:
    """The words a ``jobs(action="runs")`` reply carries when its rows are
    cut: its page fields set from the cut, which rows it holds, that they were
    read from the job's record as it stands now, and the call that reads on."""

    def annotate(
        reply: dict[str, Any], cut: Mapping[str, ListCut], _stored: Any, _paged: str | None
    ) -> None:
        entry = cut.get(RUNS_PAGE_ROWS)
        if entry is None:
            return
        reply["returned"] = entry.returned
        reply["truncated"] = entry.next is not None
        reply["next_cursor"] = entry.next
        said = [
            f"Run rows {entry.offset + 1} to {entry.offset + entry.returned} of "
            f"{entry.total} of job {job_id}, read from its record now, while it is "
            f"{reply.get('status')}."
        ]
        if reply.get("status") not in TERMINAL_STATUSES:
            said.append("A row of a job still running may change between pages.")
        if entry.next is not None:
            said.append(_the_rest_of_the_rows(entry, job_id, run_fields))
        reply["hint"] = " ".join(said)

    return annotate
