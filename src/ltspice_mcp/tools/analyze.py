"""Consolidated, bounded analysis over immutable result-source manifests."""

from __future__ import annotations

import asyncio
import contextlib
import copy
import math
import os
import statistics
import time
from collections.abc import Callable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp import types
from pydantic import BeforeValidator, Field, SkipValidation, model_validator

from ltspice_mcp.errors import AnalysisDeadlineExceeded, LTSpiceMCPError, ResultError
from ltspice_mcp.lib import (
    analysis_snapshot,
    fsync_dir,
    fsync_fd,
    metrics,
    pagination,
    response_budget,
    result_store,
    services,
)
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.format import format_spice_value
from ltspice_mcp.lib.job_lifecycle import runs_terminal
from ltspice_mcp.lib.log_parser import (
    diagnostic_collapse_key,
    extract_log_diagnostics,
    parse_step_iterations,
)
from ltspice_mcp.lib.pagination import retotal_page
from ltspice_mcp.lib.raw_parser import get_step_count, safe_magnitude_db
from ltspice_mcp.lib.recipes import (
    OperatingPointRecipe,
    PlotRecipe,
    Recipe,
    WaveformRecipe,
    _KeyedRecipe,
    _MultiRecipe,
    _ScalarRecipe,
    recipe_error,
    validate_recipe,
)
from ltspice_mcp.lib.signal_analysis import downsample_minmax
from ltspice_mcp.state import SessionState, legacy_record_message
from ltspice_mcp.tools._base import (
    ABSENT,
    ResponseBudget,
    StrictModel,
    ToolInput,
    escape_field_segment,
    format_response,
    keep_plan,
    project_row,
    registry,
    resolve_response_budget,
    safe_path,
    sanitize_payload,
    split_field_path,
)

_ARTIFACT_SAFETY_FACTOR = 4.0
_MIN_ITEM_DEADLINE_S = 0.05
# Public: the largest per_run page this tool will return. Any surface that lets
# a caller ask for a page size — including run_experiments' attached-analysis
# request — bounds itself on THIS name, so a limit accepted at submission is a
# limit the analysis can actually serve.
MAX_PAGE_SIZE = 100
_FAIL_CASE_PAGE_CAP = 100
_FAILURE_CAP = 100

_GROUPS_OMITTED_WARNING = (
    "{omitted} of {total} group(s) omitted to fit the response budget; "
    "re-ask without 'budget' for every group."
)
_FAIL_CASES_OMITTED_WARNING = (
    "{omitted} failing case(s) omitted from spec.fail_cases, which is not pageable; "
    "request include.per_run for callable pagination over every attributed value."
)
_VALUES_OMITTED_WARNING = (
    "{omitted} value(s) omitted; request include.per_run for callable pagination."
)

# Per-call digest memo keyed by (path, mtime_ns, size); one hash per unchanged
# source across manifest build, precheck and postcheck.
_DigestCache = dict[tuple[str, int, int], str]


class CaseSelection(StrictModel):
    case_ids: list[str] = Field(
        min_length=1,
        description=(
            "Analyze only these cases, addressed by the stable case_id a "
            "run_experiments receipt reports. Experiment jobs only."
        ),
    )

    @model_validator(mode="after")
    def _unique_cases(self) -> CaseSelection:
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValueError("case_ids must be unique")
        return self


class AnalyzeSourceInput(StrictModel):
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze the results of a job this server ran — an experiment, "
            "simulation, sweep or Monte Carlo id. Exactly one of job_id or "
            "raw_path. Its runs must have finished; a job still running is "
            "reported under coverage.missing_cases instead of failing the call."
        ),
    )
    raw_path: str | None = Field(
        default=None,
        description=(
            "Analyze a .raw file directly, for results this server did not run. "
            "It has no job provenance, so its rows carry deck_sha256: null and an "
            "observation says so. Exactly one of job_id or raw_path."
        ),
    )
    runs: Literal["all"] | list[int] | CaseSelection = Field(
        default="all",
        description=(
            "Which runs of this source to read: 'all', a list of run indices, or "
            "{case_ids: [...]} for an experiment job. Narrowing here is the cheapest "
            "way to keep a large fan-out inside the call budget."
        ),
    )
    label: str = Field(
        min_length=1,
        description=(
            "Short unique name for this source; it tags every returned row and is "
            "what a recipe's own 'sources' list refers to. Name the condition "
            "('nominal', 'hot'), not the file."
        ),
    )

    @model_validator(mode="after")
    def _one_source(self) -> AnalyzeSourceInput:
        if bool(self.job_id) == bool(self.raw_path):
            raise ValueError("provide exactly one of job_id or raw_path")
        if isinstance(self.runs, list):
            if not self.runs:
                raise ValueError("runs must be 'all' or a non-empty list")
            if any(index < 0 for index in self.runs) or len(set(self.runs)) != len(self.runs):
                raise ValueError("run indices must be unique non-negative integers")
        if isinstance(self.runs, CaseSelection) and self.raw_path is not None:
            raise ValueError("case_ids selection is available only for experiment jobs")
        return self


@dataclass
class _PendingArtifact:
    pending: Path
    final: Path
    content_type: str
    manifest_id: str


@dataclass(frozen=True)
class RowIdentity:
    """What identifies one analyzed run/step, in emission order.

    Declared here rather than as a list of key strings so the row shape, the
    attribution block and the projector's alphabet all read off one definition.
    """

    case_id: str | None = None
    run_index: int | None = 0
    step_index: int | None = None
    step_values: dict[str, Any] = field(default_factory=dict)
    assignments: dict[str, Any] = field(default_factory=dict)
    circuit: str | None = None
    deck_sha256: str | None = None

    def wire(self, keys: tuple[str, ...]) -> dict[str, Any]:
        return {key: getattr(self, key) for key in keys}


@dataclass(frozen=True)
class Record:
    """One recipe's answer for one run at one step, before any rendering.

    ``manifest_id`` is the join back to the source manifest; it is evaluator
    bookkeeping and never reaches the wire, which is why it is a field here
    rather than a key smuggled into the row and stripped again later.
    """

    manifest_id: str
    source: str
    identity: RowIdentity
    value: dict[str, Any]

    def attribution(self) -> dict[str, Any]:
        """The identity block a reduced/spec row carries (no provenance pair)."""
        return self.identity.wire(_ATTRIBUTION_KEYS)

    def wire(self) -> dict[str, Any]:
        """This record as one per_run/values row."""
        return {
            "source": self.source,
            **self.identity.wire(_IDENTITY_KEYS),
            "value": self.value,
        }


@dataclass(frozen=True)
class Failure:
    """One item that did not produce a value, and what stage lost it."""

    code: str
    stage: str
    where: str
    message: str

    def wire(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "stage": self.stage,
            "where": self.where,
            "message": self.message,
        }


@dataclass(frozen=True)
class Observation:
    """A fact about the data or its provenance, for the reader to weigh."""

    code: str
    kind: str
    detail: str
    evidence: dict[str, Any] | None = None

    @classmethod
    def of(cls, data: Mapping[str, Any]) -> Observation:
        """One observation read back from a persisted result set."""
        return cls(
            code=str(data.get("code", "")),
            kind=str(data.get("kind", "")),
            detail=str(data.get("detail", "")),
            evidence=data.get("evidence"),
        )

    def wire(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": self.code, "kind": self.kind, "detail": self.detail}
        if self.evidence is not None:
            data["evidence"] = self.evidence
        return data


@dataclass(frozen=True)
class SourceFault:
    """Why one source failed verification: the wire code, plus any detail.

    A pair rather than a ``"code: detail"`` string, because both halves have
    consumers — the code decides how the failure is classified, the whole line
    is what the caller reads — and splitting the string back apart at each use
    is how the two drift.
    """

    code: str
    detail: str | None = None

    @property
    def message(self) -> str:
        return self.code if self.detail is None else f"{self.code}: {self.detail}"


@dataclass
class WorkUnit:
    """One recipe's evaluated result, plus where it sat in the work list."""

    key: str
    recipe: Recipe
    records: list[Record]
    failures: list[Failure]
    pending: list[_PendingArtifact]
    eligible_ids: set[str]
    #: Row offset this unit's per_run page starts at (a resumed unit starts mid-list).
    per_run_offset: int
    #: Index of this unit in the result set's work list.
    position: int
    observations: list[Observation]


# Reduced/spec attribution rows omit the trailing provenance pair (their output
# schema forbids the extra keys), so they take the first five — a slice of the
# one definition, never a parallel list.
_IDENTITY_KEYS: tuple[str, ...] = tuple(f.name for f in fields(RowIdentity))
_ATTRIBUTION_KEYS: tuple[str, ...] = _IDENTITY_KEYS[:5]
# Every key a per_run/values row carries, in emission order. ``include.fields``
# paths are rooted here, so this one list is both the projector's alphabet and
# the answer a caller gets when a path names something that does not exist.
_ROW_KEYS: tuple[str, ...] = ("source", *_IDENTITY_KEYS, "value")


def coerce_per_run_default(value: Any) -> Any:
    """Accept ``per_run=True`` for the default page, ``False`` for none.

    A list of flag names turns each one on with ``True``; ``per_run`` is the one
    include that is an object rather than a boolean, and a spelling that works
    for three of the four flags is worse than one that works for none.
    """
    if value is True:
        return {}
    if value is False:
        return None
    return value


class PerRunInclude(StrictModel):
    limit: int = Field(default=50, ge=1, le=MAX_PAGE_SIZE)
    cursor: str | None = Field(
        default=None,
        description=(
            "Resume from per_run.next_cursor. It is bound to the request and its "
            "include.fields view; an explicit conflicting view is rejected."
        ),
    )


class AnalyzeInclude(StrictModel):
    """Optional response blocks. The default response carries reductions, groups
    and spec verdicts; per-run rows, outlier records and signal listings are
    opt-in because each one grows the payload."""

    per_run: Annotated[
        PerRunInclude | None,
        BeforeValidator(
            coerce_per_run_default,
            json_schema_input_type=PerRunInclude | bool | None,
        ),
    ] = Field(
        default=None,
        description=(
            "Return the individual attributed rows, paginated; true takes the "
            "default page. Omitted, a recipe with 'reduce' returns only its "
            "reductions and a recipe without one inlines up to 100 unpaged rows. "
            "Pair with 'fields' on a wide sweep."
        ),
    )
    outliers: bool = Field(
        default=False,
        description=(
            "Add the spec-failing records to each recipe's spec block. No effect on "
            "a recipe that declares no 'spec'."
        ),
    )
    signals_available: bool = Field(
        default=False,
        description=(
            "List the trace names each source's .raw carries, keyed by source "
            "manifest id. Costs one raw load per run — a discovery aid for naming "
            "signals, not something to leave on."
        ),
    )
    provenance: bool = Field(
        default=False,
        description=(
            "Add the artifact paths and content digests to each entry of "
            "'source_hashes'. Off by default because runs are addressed by "
            "manifest_id and job_id, which are always present — the paths and "
            "hashes prove what was analyzed, they are not how you reach it."
        ),
    )
    fields: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description=(
            "Keep only these dotted row paths (e.g. 'value.phase_margin_worst_deg', "
            "'step_values') on per_run/values rows. Paths root at one of "
            f"{', '.join(_ROW_KEYS)}; an unknown root is rejected rather than "
            "silently returning empty rows. A dot inside a key's own name is "
            r"escaped as '\.' — a subcircuit node or device parameter is spelled "
            r"'value.voltages.v(x1\.out)', 'value.device_op_points.@m\.x1\.m1[gm]'. "
            "This is the payload lever for a wide sweep — on a 45-step case it cut "
            "the rows from ~39k to ~5k characters."
        ),
    )

    @model_validator(mode="after")
    def _fields_name_real_row_keys(self) -> AnalyzeInclude:
        # A projection that silently keeps nothing is worse than no projection
        # at all, so a path that cannot be rooted in a row is rejected here
        # rather than returning empty rows the caller has to explain.
        if self.fields is None:
            return self
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("include.fields paths must be unique")
        for path in self.fields:
            segments = split_field_path(path)
            if any(not segment for segment in segments):
                raise ValueError(f"include.fields path {path!r} has an empty segment")
            root = segments[0]
            if root not in _ROW_KEYS:
                raise ValueError(
                    f"include.fields path {path!r} starts at unknown row key {root!r}; "
                    f"choose one of: {', '.join(_ROW_KEYS)}"
                )
        return self


def include_flag_coercer(model: type[StrictModel]) -> Callable[[Any], Any]:
    """Build the ``include=['outliers', 'per_run']`` -> ``{name: True}`` reader.

    The list is what a caller writes first; the dict is the shape the engine
    wants, and rejecting the list bought nothing. The accepted names are read
    off the model, so a flag added later is spellable both ways without a
    second list to keep in step — minus ``fields``, which takes row paths
    rather than a boolean and cannot mean anything inside a list. An unknown
    name still fails, enumerating the flags, rather than being switched on
    under a typo or silently dropped.
    """
    flags = frozenset(model.model_fields) - {"fields"}
    enumerated = ", ".join(sorted(flags))

    def coerce(value: Any) -> Any:
        if isinstance(value, (str, bytes, Mapping)) or not isinstance(
            value, (Sequence, AbstractSet)
        ):
            return value
        names = [str(item) for item in value]
        unknown = sorted({name for name in names if name not in flags})
        if unknown:
            raise ValueError(
                f"unknown include flag(s) {', '.join(unknown)}; a list of flags takes "
                f"{enumerated}. 'fields' takes row paths, so pass it as an object: "
                "include={'fields': ['value.phase_margin_worst_deg']}"
            )
        return dict.fromkeys(names, True)

    return coerce


coerce_include_flags = include_flag_coercer(AnalyzeInclude)


class ContinueInput(StrictModel):
    result_set_id: str = Field(
        description="From the 'next' block of the partial response being resumed.",
    )
    cursor: str = Field(
        description=(
            "From that same 'next' block, or coverage.missing_cases.next_cursor to "
            "page missing cases. Resuming replays no completed work."
        ),
    )


class AnalyzeResultsInput(ToolInput):
    sources: list[AnalyzeSourceInput] | None = Field(
        default=None,
        max_length=64,
        description=(
            "What to read — up to 64 jobs and/or .raw files, each under a unique "
            "label. Every recipe runs against every source unless the recipe names "
            "a subset itself. Each source's .raw is parsed once and shared by all "
            "recipes in the call. Required unless 'continue' is given."
        ),
    )
    # SkipValidation preserves the strict A.2 union in JSON Schema while
    # allowing the handler to validate each item independently.
    recipes: list[SkipValidation[Recipe]] | None = Field(
        default=None,
        max_length=256,
        description=(
            "The measurements to take, up to 256, each a typed recipe returned "
            "under its own unique 'key'. Ask for every metric you want in one call "
            "rather than one call per metric; a recipe that fails fails alone into "
            "'failures'."
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description=(
            "Split each recipe's reductions into groups along these dimensions: a "
            "variation assignment parameter name, 'circuit', or a .step axis name. "
            "Empty gives one reduction over every row."
        ),
    )
    include: Annotated[
        AnalyzeInclude,
        BeforeValidator(
            coerce_include_flags,
            json_schema_input_type=AnalyzeInclude | list[str],
        ),
    ] = Field(
        default_factory=AnalyzeInclude,
        description=(
            "Named opt-in response blocks — per_run rows, outliers, "
            "signals_available, provenance, and the 'fields' row projection. "
            "A bare list of flag names switches them on. The default response "
            "carries reductions, groups and spec verdicts; each opt-in grows "
            "the payload, so ask only for what you will read."
        ),
    )
    budget: int | None = Field(
        default=None,
        ge=response_budget.BUDGET_MIN_TOKENS,
        description=response_budget.BUDGET_DESCRIPTION,
    )
    continuation: ContinueInput | None = Field(
        default=None,
        alias="continue",
        description=(
            "Resume a budget-truncated response using its stored execution request "
            "and cursor fields view. Other request fields are rejected."
        ),
    )

    @model_validator(mode="after")
    def _new_or_continue(self) -> AnalyzeResultsInput:
        if self.continuation is not None:
            # A continuation replays the execution request stored in the result
            # set and takes its presentation view from the cursor, never these
            # args. Reject them rather than accept and drop them: raising
            # include.per_run.limit on resume is the obvious thing to try, and
            # silently ignoring it hands back a page the caller did not ask for.
            supplied = {"sources", "recipes", "include", "group_by"} & self.model_fields_set
            if supplied:
                raise ValueError(
                    "'continue' is mutually exclusive with "
                    + "/".join(sorted(supplied))
                    + "; a continuation replays the stored execution request and "
                    "the cursor's fields view. "
                    "To change sources, recipes, grouping or include options, "
                    "start a new analysis."
                )
            return self
        if not self.sources or not self.recipes:
            raise ValueError("a new analysis requires non-empty sources and recipes")
        labels = [source.label for source in self.sources]
        if len(set(labels)) != len(labels):
            raise ValueError("source labels must be unique")
        keys = [
            str(item.get("key", "")) if isinstance(item, dict) else str(getattr(item, "key", ""))
            for item in self.recipes
        ]
        nonempty = [key for key in keys if key]
        if len(set(nonempty)) != len(nonempty):
            raise ValueError("recipe keys must be unique")
        if len(set(self.group_by)) != len(self.group_by):
            raise ValueError("group_by dimensions must be unique")
        return self


@dataclass(frozen=True)
class _ResolvedRun:
    manifest_id: str
    label: str
    source: services.AnalysisSource
    job_id: str | None


def _page(
    items: list[Any], offset: int = 0, limit: int = MAX_PAGE_SIZE
) -> tuple[dict[str, Any], int]:
    """One page of ``items`` at this tool's default cap.

    ``next_cursor`` starts null: every resumable view here (per_run,
    coverage.missing_cases) needs a cursor carrying the work position too, so it
    is minted during assembly from the offset this returns. A view that cannot
    be resumed at all — spec.fail_cases — says so in a warning instead.
    """
    return pagination.page(items, offset, limit)


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Project ``keys`` out of ``source`` (identity/attribution subsets)."""
    return {key: source[key] for key in keys}


def _at_segments(row: dict[str, Any], segments: list[str]) -> Any:
    """The value at ``segments``, or ``ABSENT`` when one of them is missing."""
    node: Any = row
    for segment in segments:
        if not isinstance(node, dict) or segment not in node:
            return ABSENT
        node = node[segment]
    return node


#: Nested value blocks the lean row keeps anyway. The rule is narrow on
#: purpose: a block belongs here only when the caller cannot get it back from
#: anything else in the response. An artifact handle names a file already
#: written to disk — drop it and the file is unreachable.
_LEAN_KEPT_BLOCKS: frozenset[str] = frozenset({"artifact"})


def _lean_row(row: dict[str, Any], *, keep_value_whole: bool = False) -> dict[str, Any]:
    """Default row rendering — the answer channel.

    Drops attribution keys that carry nothing (null step_index, empty
    step_values — the row schema declares no required keys, so absent and
    empty mean the same thing), drops the per-row deck digest (provenance,
    reachable via include.fields), and flattens ``value`` to its scalar
    leaves — the promoted headlines and the simple facts. The nested
    curve/list detail stays reachable by name: include.fields=["value"]
    returns the full block. If flattening would empty the value (an
    all-nested metric such as measurements), the full dict stays — lean
    never trades data for absence.

    An ``artifact`` handle survives the flattening. It is a dict, so the
    scalar-leaves rule dropped it — and it is the only thing a caller cannot
    recompute from the response, because it names a file this call has already
    written. A plot recipe rendered that way came back as a series count and
    nothing else, which is not a lean answer to "plot this", it is no answer.
    """
    out: dict[str, Any] = {}
    for key, item in row.items():
        if key == "value" or key == "deck_sha256":
            continue
        if item is None or item == {} or item == []:
            continue
        out[key] = item
    value = row.get("value")
    if isinstance(value, dict) and not keep_value_whole:
        flat = {
            key: item
            for key, item in value.items()
            if key in _LEAN_KEPT_BLOCKS or not isinstance(item, (dict, list))
        }
        out["value"] = flat if flat else value
    else:
        out["value"] = value
    return out


def _row_renderer(
    fields: list[str] | None,
    *,
    whole: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build one reusable renderer for every row on a result surface."""
    plan = keep_plan(fields) if fields else None

    def render(row: dict[str, Any]) -> dict[str, Any]:
        if plan is not None:
            return project_row(row, plan)
        return _lean_row(row, keep_value_whole=whole)

    return render


def _projection_warnings(records: list[dict[str, Any]], fields: list[str]) -> list[str]:
    """One warning per requested path that no row of this recipe carries.

    Root keys are validated at input, so only a path reaching INTO a value dict
    can miss — and it can miss legitimately, because ``include.fields`` is
    call-global while each metric has its own value shape. Naming the keys that
    are actually there turns a row missing the requested key from a silence into
    the next call's argument.
    """
    warnings: list[str] = []
    for path in fields:
        segments = split_field_path(path)
        if len(segments) == 1:
            continue
        if any(_at_segments(record, segments) is not ABSENT for record in records):
            continue
        parent_segments = segments[:-1]
        parent = ".".join(escape_field_segment(segment) for segment in parent_segments)
        present = sorted(
            {
                key
                for record in records
                for node in (_at_segments(record, parent_segments),)
                if isinstance(node, dict)
                for key in node
            }
        )
        # Naming a dotted key as it must be SPELLED, not just as it reads, is
        # the difference between a warning and a fix: the caller is one escape
        # away from the value, and no amount of staring at the key says so.
        addressable = ", ".join(escape_field_segment(key) for key in present)
        warnings.append(
            f"include.fields path {path!r} is absent from every row of this recipe; "
            + (
                f"keys present at {parent!r}: {addressable}"
                if present
                else f"no row reaches {parent!r} — a key whose own name contains a "
                r"'.' is one segment, so address it with '\.' (e.g. 'value.v(x1\.out)')"
            )
        )
    return warnings


def _projection_presence(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Bounded record-shaped union used for warnings after persistence."""
    root: dict[str, Any] = {}

    def add(node: dict[str, Any], value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, child_value in value.items():
            if str(key).startswith("_"):
                continue
            child = node.setdefault(str(key), {})
            add(child, child_value)

    for record in records:
        add(root, record)
    return root


def _identity(
    source: services.AnalysisSource, step: int | None, values: dict[str, Any]
) -> RowIdentity:
    """The identity of one step of one resolved run."""
    base = dict(source.identity or {})
    return RowIdentity(
        case_id=base.get("case_id"),
        run_index=base.get("run_index", 0),
        step_index=step,
        step_values=values,
        assignments=dict(base.get("assignments") or {}),
        circuit=base.get("circuit"),
        deck_sha256=base.get("deck_sha256"),
    )


def _serialize_run(run: _ResolvedRun) -> dict[str, Any]:
    return {
        "manifest_id": run.manifest_id,
        "label": run.label,
        "raw": str(run.source.raw),
        "log": str(run.source.log) if run.source.log else None,
        "netlist": str(run.source.netlist) if run.source.netlist else None,
        "dialect": run.source.dialect,
        "identity": run.source.identity,
        "trusted_job_artifact": run.source.trusted_job_artifact,
        "job_id": run.job_id,
    }


def _deserialize_runs(item: result_store.ResultSet, state: SessionState) -> list[_ResolvedRun]:
    runs: list[_ResolvedRun] = []
    for data in item.inputs.get("resolved_runs", []):
        raw = Path(data["raw"])
        dialect = data.get("dialect")
        state.raw_dialect_hints[raw] = dialect
        runs.append(
            _ResolvedRun(
                manifest_id=str(data["manifest_id"]),
                label=str(data["label"]),
                source=services.AnalysisSource(
                    raw=raw,
                    log=Path(data["log"]) if data.get("log") else None,
                    netlist=Path(data["netlist"]) if data.get("netlist") else None,
                    dialect=dialect,
                    identity=dict(data.get("identity") or {}),
                    trusted_job_artifact=bool(data.get("trusted_job_artifact")),
                ),
                job_id=data.get("job_id"),
            )
        )
    return runs


async def _resolve_sources(
    inputs: list[AnalyzeSourceInput], state: SessionState
) -> tuple[list[_ResolvedRun], list[dict[str, Any]], dict[str, str | None], list[Observation]]:
    runs: list[_ResolvedRun] = []
    missing: list[dict[str, Any]] = []
    source_jobs: dict[str, str | None] = {}
    observations: list[Observation] = []
    for source_input in inputs:
        if source_input.raw_path is not None:
            requested_indices = {0} if source_input.runs == "all" else set(source_input.runs)
            for index in sorted(requested_indices - {0}):
                missing.append(
                    {
                        "label": source_input.label,
                        "case_id": None,
                        "run_index": index,
                        "code": "run_not_found",
                        "detail": "A raw_path source has exactly one outer run (index 0).",
                    }
                )
            if 0 not in requested_indices:
                continue
            try:
                raw = safe_path(source_input.raw_path, state)
                resolved = services.source_for_raw_path(raw, state)
            except (LTSpiceMCPError, OSError) as exc:
                missing.append(
                    {
                        "label": source_input.label,
                        "case_id": None,
                        "run_index": 0,
                        "code": "source_unavailable",
                        "detail": str(exc),
                    }
                )
                continue
            # Preserve the expected sibling-log path even while it is absent:
            # an absent→present transition changes the composite manifest.
            source = replace(
                resolved,
                log=raw.with_suffix(".log"),
                identity={
                    "case_id": None,
                    "run_index": 0,
                    "assignments": {},
                    "circuit": raw.stem,
                    "deck_sha256": None,
                    "step_index": None,
                    "step_values": {},
                },
            )
            runs.append(
                _ResolvedRun(
                    f"{source_input.label}:0",
                    source_input.label,
                    source,
                    None,
                )
            )
            observations.append(
                Observation(
                    code="raw_path_without_deck_provenance",
                    kind="provenance",
                    detail=(
                        f"Source {source_input.label!r} is a caller-supplied raw path; "
                        "deck_sha256 is null because no producing job was supplied."
                    ),
                )
            )
            continue

        assert source_input.job_id is not None
        try:
            job = await services.resolve_job_async(source_input.job_id, state)
        except LTSpiceMCPError as exc:
            missing.append(
                {
                    "label": source_input.label,
                    "case_id": None,
                    "run_index": None,
                    "code": "source_unavailable",
                    "detail": str(exc),
                }
            )
            continue
        record = job.store_path if isinstance(job, ExperimentJob) else None
        source_jobs[job.job_id] = str(record) if record is not None and record.is_file() else None

        if isinstance(job, ExperimentJob):
            # Per-case readiness is gated below, not here.
            if not runs_terminal(job.status):
                missing.append(
                    {
                        "label": source_input.label,
                        "case_id": None,
                        "run_index": None,
                        "code": "job_not_terminal",
                        "detail": (
                            f"Experiment job {job.job_id!r} has no readable runs yet "
                            f"(status={job.status!r})"
                        ),
                    }
                )
                continue
            if isinstance(source_input.runs, CaseSelection):
                wanted = set(source_input.runs.case_ids)
                selected = [case for case in job.cases if case.case_id in wanted]
                found = {case.case_id for case in selected}
                for case_id in sorted(wanted - found):
                    missing.append(
                        {
                            "label": source_input.label,
                            "case_id": case_id,
                            "run_index": None,
                            "code": "case_not_found",
                        }
                    )
            elif isinstance(source_input.runs, list):
                wanted_indices = set(source_input.runs)
                selected = [case for case in job.cases if case.run_index in wanted_indices]
                found_indices = {case.run_index for case in selected}
                for index in sorted(wanted_indices - found_indices):
                    missing.append(
                        {
                            "label": source_input.label,
                            "case_id": None,
                            "run_index": index,
                            "code": "run_not_found",
                        }
                    )
            else:
                selected = list(job.cases)
            for case in selected:
                if case.status != "produced" or case.raw_file is None:
                    missing.append(
                        {
                            "label": source_input.label,
                            "case_id": case.case_id,
                            "run_index": case.run_index,
                            "code": "raw_not_produced",
                            "detail": f"case status is {case.status!r}",
                        }
                    )
                    continue
                ctx = services.experiment_run_context(job, state, case_id=case.case_id)
                resolved = services.source_for_run(ctx)
                runs.append(
                    _ResolvedRun(
                        f"{source_input.label}:{case.case_id}",
                        source_input.label,
                        resolved,
                        job.job_id,
                    )
                )
                await state.note_recent_circuit(case.circuit_path.resolve())
            continue

        # Anything that is not an experiment is a record an earlier release
        # wrote. Say so once, against the source the caller named: reporting it
        # as an empty run set would read as a job that simply produced nothing.
        missing.append(
            {
                "label": source_input.label,
                "case_id": None,
                "run_index": None,
                "code": "legacy_job_record",
                "detail": legacy_record_message(job.job_id),
            }
        )
    observations.extend(await _relay_solve_failures(runs))
    return runs, missing, source_jobs, observations


_SOLVE_FAILURE_RUN_CAP = 10
"""How many run labels a relayed solve failure names before deferring to ``runs``."""


async def _relay_solve_failures(runs: list[_ResolvedRun]) -> list[Observation]:
    """Relay each resolved run's simulator-declared solve failures.

    The one chokepoint for this rule on this surface: a run that produced a raw
    despite a failed solve is analyzed and reported like any other, so unless
    the simulator's own line is relayed here the caller reads a number with no
    way to know the solve behind it collapsed. The internal metric adapters
    enforce the same rule at ``analysis._finish_metric``; both classify through
    ``services.solve_failure_lines`` so they cannot drift.

    One observation per distinct cause rather than per run: a sweep that fails
    to converge fails the same way in every case, and the run labels are what
    distinguishes them. Distinct is measured by
    :func:`diagnostic_collapse_key`, because the simulator's line ends in the
    run's own numbers (``time = 4.4e-05, timestep = 1.2e-19``) and no two runs
    of a sweep abort at the same instant; the relayed line is the first run's,
    verbatim. A log the bounded parse could not read is reported as such — an
    unread log is a gap in this relay's coverage, not an absence of failures.
    """
    grouped: dict[str, tuple[str, list[str]]] = {}
    unread: list[str] = []
    for run in runs:
        log = run.source.log
        if log is None or not log.exists():
            continue
        try:
            diagnostics = await services.bounded_parse(
                log, lambda log=log: extract_log_diagnostics(log)
            )
        except ResultError:
            unread.append(run.label)
            continue
        for line in services.solve_failure_lines(diagnostics):
            grouped.setdefault(diagnostic_collapse_key(line), (line, []))[1].append(run.label)

    relayed: list[Observation] = [
        Observation(
            code="solve_failure",
            kind="relay",
            detail=(
                f"The simulator reported a failed solve on {len(labels)} run(s); "
                f"every value read from them is affected. Simulator line: {line}"
            ),
            evidence={
                "log": line,
                "runs": labels[:_SOLVE_FAILURE_RUN_CAP],
                "run_count": len(labels),
            },
        )
        for line, labels in grouped.values()
    ]
    if unread:
        relayed.append(
            Observation(
                code="log_unread",
                kind="coverage",
                detail=(
                    f"{len(unread)} run log(s) could not be parsed within the analysis "
                    "deadline, so a solve failure on them would not be reported here."
                ),
                evidence={
                    "runs": unread[:_SOLVE_FAILURE_RUN_CAP],
                    "run_count": len(unread),
                },
            )
        )
    return relayed


async def _digest(path: Path, deadline: float, cache: _DigestCache) -> str:
    """Digest ``path``, memoized by (path, mtime, size) for this call.

    A file whose mtime and size are unchanged since it was first digested is not
    re-read — manifest creation and the source-drift checks before and after
    evaluation share one hash per unchanged source. A drifting file gets a fresh
    (mtime, size) key, so a real change is always re-digested.
    """
    key: tuple[str, int, int] | None
    try:
        # A cheap metadata stat on the loop (like the sibling .is_file() checks
        # here); the expensive hash read is what bounded_parse offloads.
        stat = os.stat(path)
        key = (str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        key = None
    if key is not None and key in cache:
        return cache[key]
    digest = await services.bounded_parse(
        path,
        lambda: result_store.sha256_file(path),
        timeout_s=max(_MIN_ITEM_DEADLINE_S, deadline - asyncio.get_running_loop().time()),
    )
    if key is not None:
        cache[key] = digest
    return digest


async def _manifest_for(run: _ResolvedRun, deadline: float, cache: _DigestCache) -> dict[str, Any]:
    raw_sha = await _digest(run.source.raw, deadline, cache)
    log_present = run.source.log is not None and run.source.log.is_file()
    log_sha = (
        await _digest(run.source.log, deadline, cache) if log_present and run.source.log else None
    )
    composite = result_store.composite_digest(raw_sha, log_sha, log_present)
    return {
        "manifest_id": run.manifest_id,
        "label": run.label,
        "raw_path": str(run.source.raw),
        "raw_sha256": raw_sha,
        "log_path": str(run.source.log) if run.source.log else None,
        "log_present": log_present,
        "log_sha256": log_sha,
        "composite_sha256": composite,
        "trusted_job_artifact": run.source.trusted_job_artifact,
        "job_id": run.job_id,
    }


def _work_items(recipes: list[Any]) -> list[dict[str, Any]]:
    work: list[dict[str, Any]] = []
    for index, raw_recipe in enumerate(recipes):
        payload = (
            raw_recipe.model_dump(mode="json", by_alias=True)
            if hasattr(raw_recipe, "model_dump")
            else dict(raw_recipe)
            if isinstance(raw_recipe, dict)
            else {"metric": None, "key": f"recipe_{index}"}
        )
        work.append({"index": index, "recipe": payload})
    return work


def _request_hash(args: AnalyzeResultsInput, *, include_fields: bool = False) -> str:
    assert args.sources is not None and args.recipes is not None
    include = args.include.model_dump(mode="json")
    if isinstance(include.get("per_run"), dict):
        include["per_run"]["cursor"] = None
    if not include_fields:
        include.pop("fields", None)
    return result_store.canonical_hash(
        {
            "sources": [source.model_dump(mode="json") for source in args.sources],
            "work": _work_items(list(args.recipes)),
            "group_by": list(args.group_by),
            "include": include,
        }
    )


async def _create_result_set(
    args: AnalyzeResultsInput, state: SessionState, deadline: float, cache: _DigestCache
):
    assert args.sources is not None and args.recipes is not None
    runs, missing, source_jobs, observations = await _resolve_sources(args.sources, state)
    manifests: list[dict[str, Any]] = []
    for run in runs:
        try:
            manifests.append(await _manifest_for(run, deadline, cache))
        except (LTSpiceMCPError, OSError) as exc:
            manifests.append(
                {
                    "manifest_id": run.manifest_id,
                    "label": run.label,
                    "raw_path": str(run.source.raw),
                    "log_path": str(run.source.log) if run.source.log else None,
                    "trusted_job_artifact": run.source.trusted_job_artifact,
                    "job_id": run.job_id,
                    "digest_code": (
                        "analysis_deadline"
                        if isinstance(exc, AnalysisDeadlineExceeded)
                        else "source_unavailable"
                    ),
                    "digest_error": str(exc),
                }
            )
    work = _work_items(list(args.recipes))
    include = args.include.model_dump(mode="json")
    if isinstance(include.get("per_run"), dict):
        include["per_run"]["cursor"] = None
    inputs = {
        "working_dir": str(state.working_dir),
        "sources": [source.model_dump(mode="json") for source in args.sources],
        "group_by": list(args.group_by),
        "include": include,
        "request_hash": _request_hash(args),
        "resolved_runs": [_serialize_run(run) for run in runs],
        "missing": missing,
        "observations": [observation.wire() for observation in observations],
    }
    return await asyncio.to_thread(
        result_store.create,
        working_dir=state.working_dir,
        inputs=inputs,
        work=work,
        source_manifests=manifests,
        source_jobs=source_jobs,
        ttl_hours=state.config.result_set_ttl_hours,
    )


async def _verify_direct_sources(
    manifests: list[dict[str, Any]],
    selected_ids: set[str],
    deadline: float,
    cache: _DigestCache,
) -> dict[str, SourceFault]:
    failures: dict[str, SourceFault] = {}
    for manifest in manifests:
        manifest_id = str(manifest["manifest_id"])
        if (
            manifest_id not in selected_ids
            or manifest.get("trusted_job_artifact")
            or manifest.get("digest_error")
        ):
            continue
        try:
            raw_path = Path(manifest["raw_path"])
            raw_sha = await _digest(raw_path, deadline, cache)
            log_path = Path(manifest["log_path"]) if manifest.get("log_path") else None
            log_present = log_path is not None and log_path.is_file()
            log_sha = (
                await _digest(log_path, deadline, cache) if log_present and log_path else None
            )
            composite = result_store.composite_digest(raw_sha, log_sha, log_present)
            if composite != manifest["composite_sha256"]:
                failures[manifest_id] = SourceFault("source_drift")
        except (LTSpiceMCPError, OSError) as exc:
            failures[manifest_id] = SourceFault(
                "analysis_deadline"
                if isinstance(exc, AnalysisDeadlineExceeded)
                else "source_drift",
                str(exc),
            )
    return failures


def _spice(value: float | str | None) -> str | None:
    if value is None:
        return None
    return format_spice_value(value)


def _window_fields(window: Any) -> tuple[str | None, str | None]:
    if window is None:
        return None, None
    return _spice(window.start), _spice(window.end)


async def _step_plan(
    recipe: Recipe,
    source: services.AnalysisSource,
    state: SessionState,
    step_cache: dict[str, list[dict[str, Any]]],
) -> list[tuple[int, dict[str, Any]]]:
    raw = await services.load_raw(source.raw, state)
    count = get_step_count(raw)
    step_values: list[dict[str, Any]] = []
    if source.log is not None and count > 1:
        # One step-table parse per log path per call: many recipes over the same
        # stepped run share it instead of re-parsing the log each time.
        log_key = str(source.log)
        if log_key in step_cache:
            step_values = step_cache[log_key]
        else:
            step_values = await services.bounded_parse(
                source.log,
                lambda: parse_step_iterations(source.log),
            )
            step_cache[log_key] = step_values
    if recipe.all_steps:
        return [
            (index, step_values[index] if index < len(step_values) else {})
            for index in range(count)
        ]
    if recipe.step is None:
        return [(0, step_values[0] if step_values else {})]
    step_sel = recipe.step
    candidates = [
        (index, values) for index, values in enumerate(step_values) if step_sel.axis in values
    ]
    if not candidates:
        raise ResultError(f"step axis {step_sel.axis!r} is not present in the source log")
    try:
        target = float(format_spice_value(step_sel.value))
        index, values = min(
            candidates,
            key=lambda pair: abs(float(pair[1][step_sel.axis]) - target),
        )
    except (TypeError, ValueError):
        index, values = next(
            (pair for pair in candidates if str(pair[1][step_sel.axis]) == str(step_sel.value)),
            candidates[0],
        )
    return [(index, values)]


async def _adapter_value(
    recipe: Recipe,
    source: services.AnalysisSource,
    step: int,
    state: SessionState,
) -> dict[str, Any]:
    """One recipe's value, read by the metric function registered for its class.

    ``lib.metrics.METRICS`` is the whole dispatch: every recipe class that
    produces a value maps to the function that computes it, and a class with no
    entry fails at import rather than at the one call that needed it.

    Non-finite floats are nulled here rather than inside the metric, because
    that substitution is a property of what can cross the wire — a caller
    reading a metric in-process should see what the maths produced. Doing it at
    this edge also keeps the substitution note attached to the value it
    describes rather than to the whole response.
    """
    return sanitize_payload(await metrics.METRICS[type(recipe)](source, recipe, step, state))


def _absence_observations(recipe: Recipe, key: str, records: list[Record]) -> list[Observation]:
    """State what a successful recipe looked for and did not find.

    A keyed metric answers with a map, and an empty map is indistinguishable
    from a healthy one once the row is rendered — the caller who asked a
    transistor for its bias point gets 'complete' and no numbers, with no
    channel saying the params were never in the run. Gated the way the
    ``operating_point`` tool's own note is, on a terminal current proving a
    semiconductor is present, so a passive circuit's bias point stays
    note-free and the two channels cannot disagree about when to speak.
    """

    if not isinstance(recipe, OperatingPointRecipe):
        return []
    values = [record.value for record in records]
    if not values or any(value.get("device_op_points") for value in values):
        return []
    if not any(metrics.has_active_device(value.get("currents") or {}) for value in values):
        return []
    return [
        Observation(
            code="device_op_points_absent",
            kind="coverage",
            detail=(
                f"Recipe {key!r} read the bias point of a run carrying "
                f"semiconductor terminal currents, but no per-device @dev[param] "
                f"values: neither the raw's @-param traces nor the run's .log "
                f"'Semiconductor Device Operating Points:' block held any. "
                f"{metrics.NO_DEVICE_OP_POINTS_NOTE}"
            ),
            evidence={"recipe": key, "runs": len(values)},
        )
    ]


def _artifact_estimate(recipe: Recipe, runs: list[_ResolvedRun]) -> float:
    if not isinstance(recipe, WaveformRecipe) or recipe.format != "csv":
        return 0.0
    size = 0
    for run in runs:
        with contextlib.suppress(OSError):
            size += run.source.raw.stat().st_size
    return 0.05 + size / 25_000_000 * max(1, len(recipe.signals))


async def _waveform(
    recipe: WaveformRecipe,
    run: _ResolvedRun,
    step: int,
    step_values: dict[str, Any],
    state: SessionState,
    item: result_store.ResultSet,
    manifest: dict[str, Any],
    item_deadline: float,
    export_steps: list[int] | None = None,
) -> tuple[dict[str, Any], list[_PendingArtifact]]:
    from ltspice_mcp.tools import analysis as an

    raw = await services.load_raw(run.source.raw, state)
    start, end = _window_fields(recipe.window)
    if recipe.format == "inline":
        values: list[dict[str, Any]] = []
        total_max = 0
        point_limit = min(recipe.max_points, state.config.max_points_returned)
        for signal_input in recipe.signals:
            signal = services.validate_signal(raw, signal_input)
            axis = metrics.guarded_axis(raw, step, run.source.raw)
            wave = np.asarray(raw.get_wave(signal, step=step))
            if start is not None or end is not None:
                lo, hi = metrics.window_indices(
                    axis,
                    metrics.parse_time(start, "start"),
                    metrics.parse_time(end, "end"),
                )
                axis, wave = axis[lo:hi], wave[lo:hi]
            total_max = max(total_max, len(axis))
            if len(axis) > point_limit:
                if np.iscomplexobj(wave):
                    x, mag = downsample_minmax(
                        axis,
                        safe_magnitude_db(wave),
                        point_limit,
                    )
                    _, phase = downsample_minmax(
                        axis,
                        np.degrees(np.angle(wave)),
                        point_limit,
                    )
                    series: Any = {
                        "x": x.tolist(),
                        "magnitude_db": mag.tolist(),
                        "phase_deg": phase.tolist(),
                    }
                else:
                    x, y = downsample_minmax(axis, wave, point_limit)
                    series = {"x": x.tolist(), "y": y.tolist()}
            elif np.iscomplexobj(wave):
                series = {
                    "x": axis.tolist(),
                    "magnitude_db": safe_magnitude_db(wave).tolist(),
                    "phase_deg": np.degrees(np.angle(wave)).tolist(),
                }
            else:
                series = {"x": axis.tolist(), "y": wave.tolist()}
            values.append({"signal": signal, **series})
        return (
            {
                "format": "inline",
                "series": values,
                "points_returned": max(
                    (len(series["x"]) for series in values),
                    default=0,
                ),
                "points_total": total_max,
                **_identity(run.source, step, step_values).wire(_IDENTITY_KEYS),
            },
            [],
        )

    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]
    cols: list[str] = []
    for requested in recipe.signals:
        signal = services.validate_signal(raw, requested)
        if signal == axis_name:
            raise ResultError(f"{requested!r} is the sweep axis, not a signal")
        if signal not in cols:
            cols.append(signal)
    _, analysis_type, _, _ = metrics.classify_analysis(raw)
    recipe_hash = result_store.canonical_hash(recipe.model_dump(mode="json"))
    pending, final = result_store.artifact_paths(
        item,
        recipe_key=recipe.key,
        source_digest=manifest["composite_sha256"],
        recipe_hash=recipe_hash,
        suffix="csv",
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    facts = await asyncio.to_thread(
        an.build_waveform_csv,
        raw,
        run.source.raw,
        cols,
        get_step_count(raw),
        analysis_type,
        metrics.parse_time(start, "start"),
        metrics.parse_time(end, "end"),
        "mag_phase",
        pending,
        lambda: time.monotonic() >= item_deadline,
        export_steps,
        run.source.log,
    )
    return (
        {
            "format": "csv",
            "artifact": {
                "path": str(final),
                "content_type": "text/csv",
                "sha256": None,
                "bytes": None,
            },
            "row_count": facts["row_count"],
            **_identity(
                run.source,
                step if export_steps is None or len(export_steps) == 1 else None,
                step_values if export_steps is None or len(export_steps) == 1 else {},
            ).wire(_IDENTITY_KEYS),
        },
        [_PendingArtifact(pending, final, "text/csv", run.manifest_id)],
    )


async def _plot(
    recipe: PlotRecipe,
    run: _ResolvedRun,
    steps: list[tuple[int, dict[str, Any]]],
    state: SessionState,
    item: result_store.ResultSet,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], list[_PendingArtifact]]:
    from ltspice_mcp.tools import analysis as an

    raw = await services.load_raw(run.source.raw, state)
    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]
    cols = [services.validate_signal(raw, signal) for signal in recipe.signals]
    if axis_name in cols:
        raise ResultError("The sweep axis cannot be plotted as a signal")
    _, analysis_type, _, x_is_log = metrics.classify_analysis(raw)
    recipe_hash = result_store.canonical_hash(recipe.model_dump(mode="json"))
    pending, final = result_store.artifact_paths(
        item,
        recipe_key=recipe.key,
        source_digest=manifest["composite_sha256"],
        recipe_hash=recipe_hash,
        suffix="html",
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    span = recipe.span
    facts = await asyncio.to_thread(
        an.build_plot_file,
        raw,
        run.source.raw,
        cols,
        [step for step, _ in steps],
        [values for _, values in steps],
        analysis_type,
        x_is_log if recipe.log_x is None else recipe.log_x,
        metrics.parse_time(_spice(span.start) if span else None, "span.start"),
        metrics.parse_time(_spice(span.end) if span else None, "span.end"),
        min(100_000, an.PLOT_MAX_POINTS_CEILING),
        pending,
        recipe.title or f"{run.source.raw.stem} — {analysis_type}",
    )
    return (
        {
            "artifact": {
                "path": str(final),
                "content_type": "text/html",
                "sha256": None,
                "bytes": None,
            },
            "series_count": facts["series_count"],
            **_identity(
                run.source,
                steps[0][0] if len(steps) == 1 else None,
                steps[0][1] if len(steps) == 1 else {},
            ).wire(_IDENTITY_KEYS),
        },
        [_PendingArtifact(pending, final, "text/html", run.manifest_id)],
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


_MULTI_FIELDS: dict[str, dict[str, str]] = {
    "signal_stats": {"stddev": "std"},
    "edges": {
        "rise_time": "transition_time",
        "fall_time": "transition_time",
        "edges_found": "num_edges_in_window",
    },
    "timing": {"from_time": "t_a", "to_time": "t_b"},
    "periodic": {"duty_cycle": "duty_cycle_pct"},
    "transient_response": {
        "final_value": "steady_state_value",
        "deviation": "max_droop",
        "undershoot": "max_droop",
        "overshoot": "max_overshoot",
    },
    "stability": {
        "phase_margin_deg": "phase_margin_worst_deg",
        "gain_margin_db": "gain_margin_worst_db",
    },
    "return_loss": {"reflection_coefficient": "gamma_mag"},
}

_SCALAR_FIELDS = {
    "value": ("value", "magnitude_db"),
    "thd": ("thd_pct",),
    "bode_slope": ("slope_db_per_decade",),
    "noise_integral": ("total_rms",),
}


def _bode_point_sample(value: dict[str, Any]) -> tuple[str, Any]:
    """The first point's magnitude sits one level down under ``points``."""
    points = value.get("points") or []
    return "magnitude_db", (points[0].get("magnitude_db") if points else None)


def _crossing_sample(value: dict[str, Any]) -> tuple[str, Any]:
    """The first crossing's frequency, in sweep order — the name carries the
    rule. None when the level was never crossed, over a made-up number."""
    crossings = value.get("crossings") or []
    return "first_crossing_hz", (crossings[0].get("frequency_hz") if crossings else None)


def _unity_gain_sample(value: dict[str, Any]) -> tuple[str, Any]:
    """The unity-gain crossover frequency — the bandwidth half of "UGBW and PM".

    An engineer's prior for stability metrics is gain margin, phase margin, AND
    the crossover frequency; only the first two shipped flat, so the very first
    stability question a caller asks got half an answer and paid a second call
    for the rest. None when the loop never reaches unity, which is what
    ``stability`` already says in words."""
    crossovers = value.get("unity_gain_crossovers") or []
    return "unity_gain_hz", (crossovers[0].get("frequency_hz") if crossovers else None)


# Scalar metrics whose sample lives in a nested structure rather than a flat
# top-level field. Keyed alongside _SCALAR_FIELDS so no discriminant is special
# cased inside the loop.
_SCALAR_NESTED: dict[str, Callable[[dict[str, Any]], tuple[str, Any]]] = {
    "bode_point": _bode_point_sample,
}


# Metrics whose headline number lives only inside a list (points[]/crossings[])
# where dotted ``include.fields`` projection cannot reach — 13-22x the size of
# the equivalent shell output for a 12-case sweep table, because the caller
# could not name the one leaf it wanted. Promote that number to a flat ``value`` leaf
# at row-build time. bode_point's extractor is the reducer's own, so the
# projected leaf and a reduce over that recipe can never disagree;
# bode_crossing is a variable-length recipe whose category rejects ``reduce``
# at validation, so its rule lives only here. stability shipped its worst-case
# margins flat but left the crossover frequency in a list; this is that rule
# applied uniformly.
_HEADLINE_LEAVES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "bode_point": lambda value: dict([_bode_point_sample(value)]),
    # No crossing COUNT here: the adapter caps its list (max_results, default
    # 10) and reports no truncation, so a count would silently saturate — a
    # wrong number dressed as a fact. Null first_crossing_hz carries "never
    # crossed"; ambiguity is visible in the list itself.
    "bode_crossing": lambda value: dict([_crossing_sample(value)]),
    "stability": lambda value: dict([_unity_gain_sample(value)]),
}


def _promote_headlines(metric: str, value: dict[str, Any]) -> dict[str, Any]:
    """``value`` with the metric's headline leaves added; existing keys win."""
    promote = _HEADLINE_LEAVES.get(metric)
    if promote is None or not isinstance(value, dict):
        return value
    for name, leaf in promote(value).items():
        value.setdefault(name, leaf)
    return value


def _measurements_flat(value: dict[str, Any]) -> dict[str, Any]:
    return {
        name: entry.get("mean")
        for name, entry in value.get("stats", {}).items()
        if isinstance(entry, dict)
    }


def _operating_point_flat(value: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for bucket in ("voltages", "currents", "device_op_points"):
        if isinstance(value.get(bucket), dict):
            flat.update(value[bucket])
    return flat


# How each keyed metric flattens its value dict into a {name: number} map.
_KEYED_EXTRACTORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "measurements": _measurements_flat,
    "operating_point": _operating_point_flat,
}


# Metrics whose row value IS its nested structure, so the answer channel must
# keep it whole. A waveform's value is the curve the caller asked for. A keyed
# metric's value is a map addressed by name — and operating_point's only FLAT
# leaves are ``step``/``step_count``/``device``, so leaning it returns the row's
# bookkeeping and drops every number: a bias-point read that answers "complete"
# and carries nothing. Derived from _KEYED_EXTRACTORS rather than listed, so a
# new keyed metric cannot be added without this rule following it.
_WHOLE_VALUE_METRICS: frozenset[str] = frozenset({"waveform", *_KEYED_EXTRACTORS})


def _samples(recipe: Recipe, records: list[Record]) -> dict[str, list[tuple[Record, float]]]:
    # The reducer category is the base the recipe inherits (exactly one); a
    # variable-length recipe matches none and yields no samples.
    out: dict[str, list[tuple[Record, float]]] = {}
    for record in records:
        value = record.value
        if isinstance(recipe, _ScalarRecipe):
            nested = _SCALAR_NESTED.get(recipe.metric)
            if nested is not None:
                field, candidate = nested(value)
            else:
                names = _SCALAR_FIELDS.get(recipe.metric, ())
                field = next(
                    (name for name in names if name in value), names[0] if names else "value"
                )
                candidate = value.get(field)
            number = _number(candidate)
            if number is not None:
                out.setdefault(field, []).append((record, number))
        elif isinstance(recipe, _MultiRecipe):
            field = recipe.reduce_field
            if field is None and recipe.spec is not None:
                field = recipe.spec.field
            if field:
                actual = _MULTI_FIELDS.get(recipe.metric, {}).get(field, field)
                number = _number(value.get(actual))
                if number is not None:
                    out.setdefault(field, []).append((record, number))
        elif isinstance(recipe, _KeyedRecipe):
            for name, candidate in _KEYED_EXTRACTORS[recipe.metric](value).items():
                number = _number(candidate)
                if number is not None:
                    out.setdefault(name, []).append((record, number))
    return out


def _stat(name: str, values: list[float]) -> float | int | None:
    ordered = sorted(values)
    if name == "count":
        return len(values)
    if not values:
        return None
    if name == "min":
        return ordered[0]
    if name == "max":
        return ordered[-1]
    if name == "mean":
        return statistics.fmean(values)
    if name == "stddev":
        return statistics.stdev(values) if len(values) > 1 else 0.0
    fraction = 0.5 if name == "p50" else 0.9
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return (
        ordered[low]
        if low == high
        else ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    )


def _attribution(stat: str, samples: list[tuple[Record, float]]) -> dict[str, Any]:
    if stat not in {"min", "max"} or not samples:
        return RowIdentity(run_index=None).wire(_ATTRIBUTION_KEYS)
    chosen = (min if stat == "min" else max)(samples, key=lambda pair: pair[1])[0]
    return chosen.attribution()


def _reduce(recipe: Recipe, records: list[Record]) -> list[dict[str, Any]]:
    stats = list(getattr(recipe, "reduce", []))
    reduced: list[dict[str, Any]] = []
    for field_name, field_samples in _samples(recipe, records).items():
        values = [value for _, value in field_samples]
        for stat in stats:
            reduced.append(
                {
                    "field": field_name,
                    "stat": stat,
                    "value": _stat(stat, values),
                    **_attribution(stat, field_samples),
                }
            )
    return reduced


def _spec(
    recipe: Recipe,
    records: list[Record],
    *,
    incomplete: bool,
    include_outliers: bool,
    fail_case_limit: int,
) -> dict[str, Any] | None:
    limits = getattr(recipe, "spec", None)
    if limits is None:
        return None
    samples_by_field = _samples(recipe, records)
    field = limits.field or getattr(recipe, "reduce_field", None)
    if field is None and len(samples_by_field) == 1:
        field = next(iter(samples_by_field))
    samples = samples_by_field.get(field or "", [])
    failed: list[dict[str, Any]] = []
    pass_count = 0
    for record, value in samples:
        passed = (limits.min is None or value >= limits.min) and (
            limits.max is None or value <= limits.max
        )
        if passed:
            pass_count += 1
        else:
            failed.append({"value": value, **record.attribution()})
    if not samples or (incomplete and not limits.allow_incomplete):
        verdict = "indeterminate"
    else:
        verdict = "fail" if failed else "pass"
    result = {
        "field": field,
        "min": limits.min,
        "max": limits.max,
        "pass_count": pass_count,
        "fail_count": len(failed),
        # fail_cases is not resumable, so its next offset has no consumer.
        "fail_cases": _page(failed, limit=fail_case_limit)[0],
        "verdict": verdict,
        "allow_incomplete": limits.allow_incomplete,
    }
    if include_outliers:
        result["outliers"] = failed[:fail_case_limit]
    return result


def _group_values(
    recipe: Recipe, records: list[Record], dimensions: list[str]
) -> list[dict[str, Any]]:
    if not dimensions:
        return []
    grouped: dict[tuple[tuple[str, Any], ...], list[Record]] = {}
    for record in records:
        identity = record.identity
        values: list[tuple[str, Any]] = []
        for dimension in dimensions:
            if dimension == "circuit":
                value = identity.circuit
            else:
                value = identity.assignments.get(dimension, identity.step_values.get(dimension))
            values.append((dimension, value))
        grouped.setdefault(tuple(values), []).append(record)
    return [
        {
            "by": dict(group),
            "reduced": _reduce(recipe, group_records),
            "count": len(group_records),
        }
        for group, group_records in grouped.items()
    ]


def _artifact_record(run: _ResolvedRun, value: dict[str, Any]) -> Record:
    """A waveform/plot record, whose identity the value already carries.

    Those two recipes report the run they describe inside the value itself (a
    caller reading one series wants to know which case it came from), so the
    row identity is read back out of it rather than derived a second time.
    """
    return Record(
        manifest_id=run.manifest_id,
        source=run.label,
        identity=RowIdentity(**_pick(value, _IDENTITY_KEYS)),
        value=value,
    )


async def _evaluate_item(
    recipe: Recipe,
    runs: list[_ResolvedRun],
    manifests: dict[str, dict[str, Any]],
    state: SessionState,
    item: result_store.ResultSet,
    item_deadline: float,
    step_cache: dict[str, list[dict[str, Any]]],
) -> tuple[list[Record], list[Failure], list[_PendingArtifact]]:
    selected = set(recipe.sources or [run.label for run in runs])
    selected_runs = [run for run in runs if run.label in selected]
    records: list[Record] = []
    failures: list[Failure] = []
    pending: list[_PendingArtifact] = []
    for run in selected_runs:
        manifest = manifests[run.manifest_id]
        if manifest.get("digest_error"):
            failures.append(
                Failure(
                    code=manifest.get("digest_code", "source_unavailable"),
                    stage="source_digest",
                    where=run.manifest_id,
                    message=manifest["digest_error"],
                )
            )
            continue
        try:
            with services.analysis_deadline(item_deadline):
                step_plan = await _step_plan(recipe, run.source, state, step_cache)
                if isinstance(recipe, PlotRecipe):
                    value, artifacts = await _plot(
                        recipe,
                        run,
                        step_plan,
                        state,
                        item,
                        manifest,
                    )
                    records.append(_artifact_record(run, value))
                    pending.extend(artifacts)
                    continue
                if isinstance(recipe, WaveformRecipe) and recipe.format == "csv":
                    step, step_values = step_plan[0]
                    value, artifacts = await _waveform(
                        recipe,
                        run,
                        step,
                        step_values,
                        state,
                        item,
                        manifest,
                        item_deadline,
                        [index for index, _ in step_plan],
                    )
                    records.append(_artifact_record(run, value))
                    pending.extend(artifacts)
                    continue
                for step, step_values in step_plan:
                    if isinstance(recipe, WaveformRecipe):
                        value, artifacts = await _waveform(
                            recipe,
                            run,
                            step,
                            step_values,
                            state,
                            item,
                            manifest,
                            item_deadline,
                        )
                    else:
                        value = _promote_headlines(
                            recipe.metric,
                            await _adapter_value(recipe, run.source, step, state),
                        )
                        artifacts = []
                    records.append(
                        Record(
                            manifest_id=run.manifest_id,
                            source=run.label,
                            identity=_identity(run.source, step, step_values),
                            value=value,
                        )
                    )
                    pending.extend(artifacts)
        except (LTSpiceMCPError, ValueError, OSError) as exc:
            failures.append(
                Failure(
                    code=(
                        "analysis_deadline"
                        if isinstance(exc, AnalysisDeadlineExceeded)
                        else "recipe_failed"
                    ),
                    stage="analyze",
                    where=run.manifest_id,
                    message=str(exc),
                )
            )
    return records, failures, pending


def _rename_artifacts(pending: list[_PendingArtifact]) -> None:
    for artifact in pending:
        artifact.final.parent.mkdir(parents=True, exist_ok=True)
        # Match the repo's atomic-write durability: flush the fully-written temp
        # to stable storage, then the rename metadata after, so a crash between
        # digest and publish can't surface a truncated artifact under its final
        # name. The digest was already taken on this same byte content.
        fd = os.open(artifact.pending, os.O_RDONLY)
        try:
            fsync_fd(fd)
        finally:
            os.close(fd)
        os.replace(artifact.pending, artifact.final)
        fsync_dir(artifact.final.parent)


async def _publish(
    pending: list[_PendingArtifact],
    deadline: float,
) -> list[dict[str, Any]]:
    """Digest every complete temp under budget, then publish with rename."""
    prepared: list[dict[str, Any]] = []
    loop = asyncio.get_running_loop()
    for artifact in pending:
        digest = await services.bounded_parse(
            artifact.pending,
            lambda path=artifact.pending: result_store.sha256_file(path),
            timeout_s=max(_MIN_ITEM_DEADLINE_S, deadline - loop.time()),
        )
        size = await asyncio.to_thread(lambda path=artifact.pending: path.stat().st_size)
        prepared.append(
            {
                "path": str(artifact.final),
                "content_type": artifact.content_type,
                "sha256": digest,
                "bytes": size,
            }
        )
    await asyncio.to_thread(_rename_artifacts, pending)
    return prepared


def _remove_pending(pending: list[_PendingArtifact]) -> None:
    for artifact in pending:
        with contextlib.suppress(OSError):
            artifact.pending.unlink()


def _selected_manifest_ids(recipe: Recipe, runs: list[_ResolvedRun]) -> set[str]:
    labels = set(recipe.sources or [run.label for run in runs])
    return {run.manifest_id for run in runs if run.label in labels}


def _record_warnings(records: list[Record]) -> list[str]:
    """Every distinct record warning once, in first-appearance order, carrying
    how many records raised it.

    A warning repeated per record adds nothing after the first copy, but its
    REACH is itself a fact — one raised on 45 of 45 runs means something
    different from one raised on 3 — so identical texts collapse onto a single
    line that states the count and texts that differ stay separate. The count is
    left off a single-record result, where "1 of 1" says nothing. Counting is by
    record, not by occurrence, so the number always answers "how many runs".
    """
    counts: dict[str, int] = {}
    for record in records:
        seen: set[str] = set()
        for warning in record.value.get("warnings", []):
            if not isinstance(warning, str) or warning in seen:
                continue
            seen.add(warning)
            counts[warning] = counts.get(warning, 0) + 1
    if len(records) <= 1:
        return list(counts)
    return [f"{text} ({count} of {len(records)} records)" for text, count in counts.items()]


def _result_entry(
    recipe: Recipe,
    records: list[Record],
    failures: list[Failure],
    group_by: list[str],
    missing: list[dict[str, Any]],
    per_run_offset: int,
    per_run_limit: int | None,
    include_outliers: bool,
    fields: list[str] | None = None,
    *,
    values_limit: int,
    fail_case_limit: int,
    groups_limit: int | None,
) -> tuple[dict[str, Any], int]:
    """The one result entry for ``recipe``, plus the offset its ``per_run``
    page ends at — the caller turns that into the resume cursor."""
    incomplete = bool(failures or missing)
    entry: dict[str, Any] = {
        "metric": recipe.metric,
        "reduced": _reduce(recipe, records),
        "warnings": _record_warnings(records),
    }
    groups = _group_values(recipe, records, group_by)
    if groups:
        if groups_limit is not None and len(groups) > groups_limit:
            # Only a caller-set budget's shrink rung passes a limit here. A group
            # is an aggregate answer rather than a page of a longer list, so the
            # omission is stated rather than flagged — there is no cursor that
            # walks the rest.
            entry["warnings"].append(
                _GROUPS_OMITTED_WARNING.format(
                    omitted=len(groups) - groups_limit,
                    total=len(groups),
                )
            )
            groups = groups[:groups_limit]
        entry["groups"] = groups
    spec = _spec(
        recipe,
        records,
        incomplete=incomplete,
        include_outliers=include_outliers,
        fail_case_limit=fail_case_limit,
    )
    if spec is not None:
        entry["spec"] = spec
        if spec["fail_cases"]["truncated"]:
            # fail_cases is a projection of records computed in THIS call and
            # never persisted, so it has no cursor of its own. Name the route
            # that does page every attributed value instead of leaving the
            # caller with a truncation flag and nowhere to go.
            entry["warnings"].append(
                _FAIL_CASES_OMITTED_WARNING.format(
                    omitted=spec["fail_count"] - spec["fail_cases"]["returned"]
                )
            )
    # Everything outside _WHOLE_VALUE_METRICS defaults to the scalar leaves.
    # One plan serves both row surfaces, so projection never depends on an
    # unrelated pagination choice.
    render = _row_renderer(fields, whole=recipe.metric in _WHOLE_VALUE_METRICS)

    per_run_next = per_run_offset
    if per_run_limit is not None:
        page, per_run_next = _page(records, per_run_offset, per_run_limit)
        page["items"] = [render(record.wire()) for record in page["items"]]
        entry["per_run"] = page
    elif not getattr(recipe, "reduce", []):
        shown = records[:values_limit]
        entry["values"] = [render(record.wire()) for record in shown]
        if len(records) > values_limit:
            entry["warnings"].append(
                _VALUES_OMITTED_WARNING.format(omitted=len(records) - values_limit)
            )
    # Only report unresolved paths where rows were actually emitted: a
    # reduce-only recipe has no row surface by construction, so its fields did
    # not fail to resolve — they had nothing to apply to.
    if fields and records and ("per_run" in entry or "values" in entry):
        entry["warnings"].extend(
            _projection_warnings([record.wire() for record in records], fields)
        )
    return entry, per_run_next


# What a caller addresses a run by, versus the audit trail proving what it ran
# against. In one measured response the trail was a sixth of the whole receipt
# (1,174 of 7,835 characters), naming files the analysis tools already resolve
# by id.
_RUN_IDENTITY_KEYS = ("manifest_id", "label")
_RUN_PROVENANCE_KEYS = (
    "log_present",
    "job_id",
    "raw_path",
    "raw_sha256",
    "log_path",
    "log_sha256",
    "composite_sha256",
)


def _source_hashes(
    item: result_store.ResultSet, *, provenance: bool = False
) -> list[dict[str, Any]]:
    keys = _RUN_IDENTITY_KEYS + (_RUN_PROVENANCE_KEYS if provenance else ())
    return [{key: manifest.get(key) for key in keys} for manifest in item.source_manifests]


_PAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "items": response_budget.row_items_schema({"type": "object"}),
        "items_columns": response_budget.COLUMNAR_ROWS_SCHEMA,
        "total": {"type": "integer"},
        "returned": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "next_cursor": {"type": ["string", "null"]},
    },
    "required": ["items", "total", "returned", "truncated", "next_cursor"],
    "additionalProperties": False,
}

# One per_run/values row. No ``required`` list: include.fields projects a row
# down to the requested paths, so any subset of these keys is a valid row. The
# full set is what an unprojected call returns; ``additionalProperties: false``
# still holds because projection keeps the row's shape and only drops keys.
_ATTRIBUTED_VALUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "A case/run/step-attributed row. Carries every key below unless "
        "include.fields projected it down to the requested paths."
    ),
    "properties": {
        "source": {"type": "string"},
        "case_id": {"type": ["string", "null"]},
        "run_index": {"type": ["integer", "null"]},
        "step_index": {"type": ["integer", "null"]},
        "step_values": {"type": "object"},
        "assignments": {"type": "object"},
        "circuit": {"type": ["string", "null"]},
        "deck_sha256": {"type": ["string", "null"]},
        "value": {"type": "object"},
    },
    "additionalProperties": False,
}

_REDUCED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "stat": {"type": "string"},
        "value": {"type": ["number", "integer", "null"]},
        "case_id": {"type": ["string", "null"]},
        "run_index": {"type": ["integer", "null"]},
        "step_index": {"type": ["integer", "null"]},
        "step_values": {"type": "object"},
        "assignments": {"type": "object"},
    },
    "required": [
        "field",
        "stat",
        "value",
        "case_id",
        "run_index",
        "step_index",
        "step_values",
        "assignments",
    ],
    "additionalProperties": False,
}

_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "field": {"type": ["string", "null"]},
        "min": {"type": ["number", "null"]},
        "max": {"type": ["number", "null"]},
        "pass_count": {"type": "integer"},
        "fail_count": {"type": "integer"},
        "fail_cases": _PAGE_SCHEMA,
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "indeterminate"],
        },
        "allow_incomplete": {"type": "boolean"},
        "outliers": {"type": "array", "items": {"type": "object"}},
    },
    "required": [
        "field",
        "min",
        "max",
        "pass_count",
        "fail_count",
        "fail_cases",
        "verdict",
        "allow_incomplete",
    ],
    "additionalProperties": False,
}

_RESULT_ENTRY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "metric": {"type": "string"},
        "reduced": response_budget.row_items_schema(_REDUCED_SCHEMA),
        "reduced_columns": response_budget.COLUMNAR_ROWS_SCHEMA,
        "groups": {"type": "array", "items": {"type": "object"}},
        "steps": {"type": "array", "items": {"type": "object"}},
        "spec": _SPEC_SCHEMA,
        "per_run": response_budget.row_page_schema(
            _PAGE_SCHEMA, item_schema=_ATTRIBUTED_VALUE_SCHEMA
        ),
        "values": response_budget.row_items_schema(_ATTRIBUTED_VALUE_SCHEMA),
        "values_columns": response_budget.COLUMNAR_ROWS_SCHEMA,
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["metric", "reduced", "warnings"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "outcome": {"type": "string", "enum": ["complete", "partial", "failed"]},
        "coverage": {
            "type": "object",
            "properties": {
                "runs_requested": {"type": "integer"},
                "runs_analyzed": {"type": "integer"},
                "missing_cases": _PAGE_SCHEMA,
            },
            "required": ["runs_requested", "runs_analyzed", "missing_cases"],
            "additionalProperties": False,
        },
        "results": {
            "type": "object",
            "additionalProperties": _RESULT_ENTRY_SCHEMA,
        },
        "observations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "kind": {"type": "string"},
                    "detail": {"type": "string"},
                    "evidence": {"type": "object"},
                },
                "required": ["code", "kind", "detail"],
                "additionalProperties": True,
            },
        },
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "stage": {"type": "string"},
                    "where": {"type": "string"},
                    "message": {"type": "string"},
                },
                "required": ["code", "stage", "where", "message"],
                "additionalProperties": False,
            },
        },
        "signals_available": {
            "type": "object",
            "additionalProperties": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "source_hashes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "manifest_id": {"type": "string"},
                    "label": {"type": "string"},
                    "raw_path": {"type": "string"},
                    "raw_sha256": {"type": ["string", "null"]},
                    "log_path": {"type": ["string", "null"]},
                    "log_present": {"type": ["boolean", "null"]},
                    "log_sha256": {"type": ["string", "null"]},
                    "composite_sha256": {"type": ["string", "null"]},
                    "job_id": {"type": ["string", "null"]},
                },
                # Only the identity a caller addresses a run by is always
                # present; the paths and digests are the audit trail, emitted
                # under include.provenance. Requiring them would make the lean
                # payload violate this tool's own schema.
                "required": ["manifest_id", "label"],
                "additionalProperties": False,
            },
        },
        "result_set_id": {"type": "string"},
        "cursor": {"type": ["string", "null"]},
        "next": {
            "type": ["object", "null"],
            "properties": {
                "result_set_id": {"type": "string"},
                "cursor": {"type": "string"},
            },
            "required": ["result_set_id", "cursor"],
            "additionalProperties": False,
        },
        "hint": {"type": "string"},
    },
    "required": [
        "outcome",
        "coverage",
        "results",
        "observations",
        "failures",
        "source_hashes",
        "result_set_id",
        "cursor",
        "next",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _Limits:
    """The effective list limits one assembly renders at.

    Held apart from the request because the budget ladder shrinks them, and a
    shrunk page has to be decided BEFORE the entry and its cursor are built: a
    cursor minted against the caller's limit and then handed back with fewer
    rows would skip the difference on resume.
    """

    per_run: int | None
    #: The cap on the unpaged row surfaces that start at one number — the values
    #: list and the missing-cases page. One field because they are one number:
    #: both start at MAX_PAGE_SIZE and both shrink against the same measurement.
    rows: int
    #: The cap on ``spec.fail_cases``. Its own field only because it starts at
    #: its own constant; it shrinks with everything else.
    fail_cases: int
    #: The cap on a recipe's ``groups`` list. ``None`` is uncapped, which is what
    #: every unbudgeted call gets: a group_by answer is not a page, so nothing
    #: but a caller-set budget's shrink rung ever trims it.
    groups: int | None = None

    @classmethod
    def of(cls, include: AnalyzeInclude) -> _Limits:
        # Read here, not bound as field defaults: this is the one place the
        # module's caps enter a render, so a test that moves a cap moves it for
        # the whole path rather than for whichever call site happened to reread it.
        return cls(
            per_run=include.per_run.limit if include.per_run else None,
            rows=MAX_PAGE_SIZE,
            fail_cases=_FAIL_CASE_PAGE_CAP,
        )

    def scaled(self, measure: response_budget.RowMeasure, rung: response_budget.Rung) -> _Limits:
        """These limits, shrunk to what the previous rung's measurement affords.

        Every surface named here is also in :func:`analysis_rows`. That pairing
        is the cost model: a surface this shrinks but the measurement omits is
        charged to the fixed envelope, and one the measurement counts but this
        cannot shrink makes the fixed envelope look smaller than it is.
        """
        return _Limits(
            per_run=(None if self.per_run is None else measure.fit_limit(self.per_run, rung)),
            rows=measure.fit_limit(self.rows, rung),
            fail_cases=measure.fit_limit(self.fail_cases, rung),
            groups=(
                measure.fit_limit(self.groups, rung)
                if self.groups is not None
                else measure.affordable(rung)
            ),
        )


@dataclass(frozen=True)
class AnalysisContinuationPosition:
    """Evaluator position between bounded drives, before cursor encoding."""

    result_set_id: str
    work_index: int
    row_offset: int = 0
    missing_offset: int = 0
    view_fields: tuple[str, ...] | None = None
    has_explicit_view: bool = False


@dataclass(frozen=True)
class AnalysisEvaluation:
    """Everything a response is built from, once the work behind it is done.

    Assembly is a separate step from evaluation because the budget ladder
    re-renders: reaching a smaller page by re-running the recipes would pay for
    the budget in raw reads, and trimming an assembled page instead would
    desync the cursor it already minted.
    """

    item: result_store.ResultSet
    processed: list[WorkUnit]
    runs: list[_ResolvedRun]
    #: Runs a source named but could not deliver, already in wire shape: they
    #: are stored verbatim in the result set and paged straight into
    #: ``coverage.missing_cases``, so they are never anything else.
    missing: list[dict[str, Any]]
    missing_offset: int
    #: Rejections, each with the work position it was rejected at.
    skipped: list[tuple[int, Failure]]
    base_observations: list[Observation]
    include: AnalyzeInclude
    group_by: list[str]
    natural_position: int
    natural_intra: int
    deferred: bool
    signals: dict[str, list[str]] | None

    @property
    def failure_inventory(self) -> tuple[Failure, ...]:
        """Every failure produced by this drive, without the MCP failure cap."""
        failures = [failure for _position, failure in self.skipped]
        failures.extend(failure for unit in self.processed for failure in unit.failures)
        return tuple(failures)

    @property
    def continuation(self) -> AnalysisContinuationPosition | None:
        """Where a later neutral drive resumes, if compute work remains."""
        if self.natural_position >= len(self.item.work):
            return None
        return AnalysisContinuationPosition(
            result_set_id=self.item.result_set_id,
            work_index=self.natural_position,
            row_offset=self.natural_intra,
            missing_offset=self.missing_offset,
            view_fields=(tuple(self.include.fields) if self.include.fields is not None else None),
            has_explicit_view=self.include.fields is not None,
        )


def _assemble(
    a: AnalysisEvaluation,
    rung: response_budget.Rung | None,
    limits: _Limits,
) -> tuple[dict[str, Any], str]:
    """Build the response from finished work, at ``limits`` and ``rung``.

    ``rung`` is None when the caller set no budget, which is the only path that
    existed before budgets and is byte-for-byte what it always produced.
    """
    answer_channel = rung is not None and rung.answer_channel
    provenance = a.include.provenance and not answer_channel
    outliers = a.include.outliers and not answer_channel
    signals = None if answer_channel else a.signals
    item = a.item

    # Re-decide where this response stops, against the limit it is assembling
    # at. A smaller page stops at the same recipe or an earlier one, so every
    # unit it keeps was already evaluated; a unit it drops is re-evaluated on
    # resume and rewrites its artifacts under the same deterministic names.
    units = a.processed
    paginating = -1
    position, intra_item = a.natural_position, a.natural_intra
    if limits.per_run is not None:
        for index, unit in enumerate(a.processed):
            offset = unit.per_run_offset
            if (unit.records or not unit.failures) and offset + limits.per_run < len(unit.records):
                units = a.processed[: index + 1]
                paginating = index
                position = unit.position
                intra_item = offset + limits.per_run
                break

    # One resume point serves every cursor this call emits. The work list and
    # the coverage list are paged independently, so each cursor has to carry
    # BOTH offsets: one that dropped the work position would strand un-analyzed
    # work, and one that dropped the coverage offset would replay missing cases
    # already shown. Paged here, ahead of the cursors that have to quote it.
    missing_page, missing_next = _page(a.missing, a.missing_offset, limits.rows)

    # A rejection for work this response no longer reaches belongs to the call
    # that resumes it, not to this one.
    failures = [failure for at, failure in a.skipped if at < position]
    observations = [*a.base_observations, *(o for unit in units for o in unit.observations)]
    results: dict[str, Any] = {}
    analyzed_identities: set[tuple[Any, Any, Any]] = set()

    for index, unit in enumerate(units):
        failures.extend(unit.failures)
        relevant_missing = [
            case
            for case in a.missing
            if unit.recipe.sources is None or case.get("label") in set(unit.recipe.sources)
        ]
        entry, per_run_next = _result_entry(
            unit.recipe,
            unit.records,
            unit.failures,
            a.group_by,
            relevant_missing,
            unit.per_run_offset,
            limits.per_run,
            outliers,
            a.include.fields,
            values_limit=limits.rows,
            fail_case_limit=limits.fail_cases,
            groups_limit=limits.groups,
        )
        if unit.records or not unit.failures:
            results[unit.key] = entry
        for record in unit.records:
            analyzed_identities.add(
                (record.source, record.identity.case_id, record.identity.run_index)
            )
        if (
            index == paginating
            and unit.key in results
            and a.include.per_run is not None
            and entry["per_run"]["truncated"]
        ):
            entry["per_run"]["next_cursor"] = result_store.encode_cursor(
                item,
                unit.position,
                intra_item=per_run_next,
                missing_offset=missing_next,
                view_fields=a.include.fields,
            )

    next_value: dict[str, str] | None = None
    if position < len(item.work):
        next_value = {
            "result_set_id": item.result_set_id,
            "cursor": result_store.encode_cursor(
                item,
                position,
                intra_item=intra_item,
                missing_offset=missing_next,
                view_fields=a.include.fields,
            ),
        }
    failure_total = len(failures)
    if failure_total > _FAILURE_CAP:
        failures = failures[:_FAILURE_CAP]
        observations.append(
            Observation(
                code="failures_truncated",
                kind="coverage",
                detail=(
                    f"Returned {_FAILURE_CAP} of {failure_total} failure records; "
                    "coverage and recipe result presence still reflect the full call."
                ),
            )
        )
    runs_requested = len(a.runs) + len(a.missing)
    outcome = (
        "failed"
        if not results and failures and next_value is None
        else "partial"
        if failures or a.missing or next_value is not None
        else "complete"
    )
    if missing_page["truncated"]:
        # Carries the live work position, not the end of the work list: this
        # cursor advances the coverage view, and pointing it past the work would
        # discard whatever work the caller had left to resume.
        missing_page["next_cursor"] = result_store.encode_cursor(
            item,
            position,
            intra_item=intra_item,
            missing_offset=missing_next,
            view_fields=a.include.fields,
        )
    coverage = {
        "runs_requested": runs_requested,
        "runs_analyzed": len(analyzed_identities),
        "missing_cases": missing_page,
    }
    data: dict[str, Any] = {
        "outcome": outcome,
        "coverage": coverage,
        "results": results,
        "observations": [observation.wire() for observation in observations],
        "failures": [failure.wire() for failure in failures],
        "source_hashes": _source_hashes(item, provenance=provenance),
        "result_set_id": item.result_set_id,
        "cursor": next_value["cursor"] if next_value is not None else None,
        "next": next_value,
    }
    if signals is not None:
        data["signals_available"] = signals
    hints: list[str] = []
    if next_value is not None:
        reason = "an artifact item was deferred intact" if a.deferred else "the call budget ended"
        hints.append(
            f"Analysis is partial because {reason}; call analyze_results with "
            f"continue={{result_set_id, cursor}} from 'next'."
        )
    if missing_page["next_cursor"] is not None:
        hints.append(
            "coverage.missing_cases is truncated; call analyze_results with "
            "continue={result_set_id, cursor: coverage.missing_cases.next_cursor} "
            "for the next page of missing cases (no work is replayed)."
        )
    if hints:
        data["hint"] = " ".join(hints)
    text = (
        f"analyze_results: {outcome}; {len(results)} recipe result(s), "
        f"{failure_total} failure(s), "
        f"{coverage['runs_analyzed']}/{runs_requested} run(s) analyzed"
    )
    return data, text


def _snapshot_from_assembly(a: AnalysisEvaluation) -> dict[str, Any]:
    """Persist facts and bounded unprojected rows, never a rendered response."""
    wide_include = a.include.model_copy(update={"fields": list(_ROW_KEYS)})
    wide = replace(a, include=wide_include)
    requested, _ = _assemble(wide, None, _Limits.of(wide_include))
    top = requested
    requested_results = top.pop("results")

    def coverage_cursor_base(view: dict[str, Any]) -> str:
        missing_page = view["coverage"]["missing_cases"]
        cursor = missing_page.get("next_cursor")
        if not isinstance(cursor, str):
            next_value = view.get("next")
            cursor = next_value.get("cursor") if isinstance(next_value, dict) else None
        if isinstance(cursor, str):
            return cursor
        return result_store.encode_cursor(
            a.item,
            a.natural_position,
            intra_item=a.natural_intra,
            missing_offset=a.missing_offset,
        )

    requested_coverage_cursor = coverage_cursor_base(top)
    units = {unit.key: unit for unit in a.processed}
    missing_next = min(a.missing_offset + MAX_PAGE_SIZE, len(a.missing))
    natural_cursor_base = result_store.encode_cursor(
        a.item,
        a.natural_position,
        intra_item=a.natural_intra,
        missing_offset=missing_next,
    )
    results: dict[str, Any] = {}
    for key, requested_entry in requested_results.items():
        unit = units[key]
        entry = requested_entry
        per_run = entry.pop("per_run", None)
        had_values = "values" in entry
        entry.pop("values", None)
        rows = [record.wire() for record in unit.records]
        answer_rows = rows[:MAX_PAGE_SIZE] if not getattr(unit.recipe, "reduce", []) else []
        block: dict[str, Any] = {
            "facts": entry,
            "answer_rows": answer_rows,
            "answer_total": len(unit.records),
            "surface": ("per_run" if per_run is not None else "values" if had_values else "none"),
            "projection_presence": _projection_presence(rows),
        }
        if per_run is not None:
            block["per_run"] = per_run
            block["per_run_offset"] = unit.per_run_offset
            block["per_run_cursor_base"] = result_store.encode_cursor(
                a.item,
                unit.position,
                intra_item=unit.per_run_offset,
                missing_offset=missing_next,
            )
        results[key] = block
    return analysis_snapshot.envelope(
        {
            "top": top,
            "results": results,
            "missing_offset": a.missing_offset,
            "coverage_cursor_base": requested_coverage_cursor,
            "natural_cursor_base": natural_cursor_base,
            "natural_has_next": a.natural_position < len(a.item.work),
            "natural_deferred": a.deferred,
        }
    )


def _rebind_snapshot_cursors(data: dict[str, Any], fields: list[str] | None) -> None:
    cursor = data.get("cursor")
    if isinstance(cursor, str):
        data["cursor"] = result_store.reencode_cursor(cursor, view_fields=fields)
    next_value = data.get("next")
    if isinstance(next_value, dict) and isinstance(next_value.get("cursor"), str):
        next_value["cursor"] = result_store.reencode_cursor(
            next_value["cursor"], view_fields=fields
        )
        data["cursor"] = next_value["cursor"]
    coverage = data.get("coverage")
    if isinstance(coverage, dict):
        missing = coverage.get("missing_cases")
        if isinstance(missing, dict) and isinstance(missing.get("next_cursor"), str):
            missing["next_cursor"] = result_store.reencode_cursor(
                missing["next_cursor"], view_fields=fields
            )


def _shrink_snapshot_page(
    page: dict[str, Any],
    limit: int,
    *,
    offset: int = 0,
    cursor_base: str | None = None,
    fields: list[str] | None = None,
) -> None:
    items = page["items"][:limit]
    next_offset = retotal_page(page, items, offset)
    page["next_cursor"] = (
        result_store.reencode_cursor(
            cursor_base,
            view_fields=fields,
            missing_offset=next_offset,
        )
        if page["truncated"] and cursor_base is not None
        else None
    )


def _without_warnings(warnings: list[str], *substrings: str) -> list[str]:
    return [
        warning
        for warning in warnings
        if not any(substring in warning for substring in substrings)
    ]


def render_attached_analysis(
    stored: dict[str, Any],
    *,
    fields: list[str] | None,
    answer_channel: bool = False,
    row_limit: int | None = None,
) -> dict[str, Any]:
    """Render one public attached result from its stored neutral snapshot."""
    snapshot_kind = analysis_snapshot.classify(stored)
    if snapshot_kind != "snapshot":
        raise ResultError(
            "The attached analysis was stored by an earlier release in a shape this "
            "build cannot render — re-run analyze_results against the job."
        )

    data = copy.deepcopy(stored["top"])
    _rebind_snapshot_cursors(data, fields)
    natural_cursor_base = stored.get("natural_cursor_base")
    if answer_channel:
        data.pop("hint", None)
        if not isinstance(natural_cursor_base, str):
            raise ResultError("The attached analysis snapshot lacks its continuation handle")
        natural_cursor = result_store.reencode_cursor(
            natural_cursor_base,
            view_fields=fields,
        )
        if stored.get("natural_has_next"):
            data["cursor"] = natural_cursor
            data["next"] = {
                "result_set_id": data["result_set_id"],
                "cursor": natural_cursor,
            }
        else:
            data["cursor"] = None
            data["next"] = None
        coverage = data.get("coverage")
        if isinstance(coverage, dict):
            missing = coverage.get("missing_cases")
            if isinstance(missing, dict):
                missing["next_cursor"] = (
                    result_store.reencode_cursor(
                        natural_cursor_base,
                        view_fields=fields,
                        missing_offset=int(stored["missing_offset"]) + int(missing["returned"]),
                    )
                    if missing.get("truncated")
                    else None
                )
    rendered_results: dict[str, Any] = {}
    for key, block in stored["results"].items():
        entry = copy.deepcopy(block["facts"])
        metric = entry.get("metric")
        render_row = _row_renderer(fields, whole=metric in _WHOLE_VALUE_METRICS)
        rows_emitted = False
        value_rows: list[dict[str, Any]] | None = None
        if (answer_channel and block["answer_rows"]) or (
            not answer_channel and block["surface"] == "values"
        ):
            value_rows = block["answer_rows"]
        if value_rows is not None:
            selected = value_rows if row_limit is None else value_rows[:row_limit]
            rows = copy.deepcopy(selected)
            entry["values"] = [render_row(row) for row in rows]
            rows_emitted = bool(rows)
            if answer_channel:
                entry["warnings"] = _without_warnings(
                    entry.get("warnings", []),
                    " value(s) omitted; request include.per_run",
                    " value(s) omitted to fit the response budget",
                )
                omitted = int(block["answer_total"]) - len(rows)
                if omitted > 0:
                    entry["warnings"].append(_VALUES_OMITTED_WARNING.format(omitted=omitted))
        elif not answer_channel and block["surface"] == "per_run":
            page = dict(block["per_run"])
            stored_rows = page["items"]
            selected = stored_rows if row_limit is None else stored_rows[:row_limit]
            rows = copy.deepcopy(selected)
            rendered_rows = [render_row(row) for row in rows]
            offset = int(block["per_run_offset"])
            next_offset = retotal_page(page, rendered_rows, offset)
            page["next_cursor"] = (
                result_store.reencode_cursor(
                    block["per_run_cursor_base"],
                    view_fields=fields,
                    intra_item=next_offset,
                )
                if page["truncated"]
                else None
            )
            entry["per_run"] = page
            rows_emitted = bool(rows)
            if page["next_cursor"] is not None:
                data["cursor"] = page["next_cursor"]
                data["next"] = {
                    "result_set_id": data["result_set_id"],
                    "cursor": page["next_cursor"],
                }
                data["outcome"] = "partial"
                response_budget.append_hint(
                    data,
                    "Analysis is partial because the response budget reduced the "
                    "per_run page; call analyze_results with "
                    "continue={result_set_id, cursor} from 'next'.",
                )
        if row_limit is not None:
            groups = entry.get("groups")
            if isinstance(groups, list) and len(groups) > row_limit:
                omitted = len(groups) - row_limit
                entry["groups"] = groups[:row_limit]
                entry.setdefault("warnings", []).append(
                    _GROUPS_OMITTED_WARNING.format(
                        omitted=omitted,
                        total=len(groups),
                    )
                )
            spec = entry.get("spec")
            if isinstance(spec, dict) and isinstance(spec.get("fail_cases"), dict):
                fail_cases = spec["fail_cases"]
                _shrink_snapshot_page(fail_cases, row_limit)
                entry["warnings"] = _without_warnings(
                    entry.get("warnings", []),
                    " failing case(s) omitted from spec.fail_cases",
                )
                if fail_cases["truncated"]:
                    entry["warnings"].append(
                        _FAIL_CASES_OMITTED_WARNING.format(
                            omitted=spec["fail_count"] - fail_cases["returned"]
                        )
                    )
        if fields and rows_emitted:
            entry.setdefault("warnings", []).extend(
                _projection_warnings([block["projection_presence"]], fields)
            )
        if answer_channel:
            spec = entry.get("spec")
            if isinstance(spec, dict):
                spec.pop("outliers", None)
        rendered_results[key] = entry
    data["results"] = rendered_results
    if row_limit is not None:
        coverage = data.get("coverage")
        if isinstance(coverage, dict) and isinstance(coverage.get("missing_cases"), dict):
            _shrink_snapshot_page(
                coverage["missing_cases"],
                row_limit,
                offset=int(stored["missing_offset"]),
                cursor_base=str(
                    natural_cursor_base if answer_channel else stored["coverage_cursor_base"]
                ),
                fields=fields,
            )
            if coverage["missing_cases"]["next_cursor"] is not None:
                response_budget.append_hint(
                    data,
                    "coverage.missing_cases is truncated; call analyze_results with "
                    "continue={result_set_id, cursor: "
                    "coverage.missing_cases.next_cursor} for the next page of "
                    "missing cases (no work is replayed).",
                )
    if answer_channel:
        data.pop("signals_available", None)
        hashes = data.get("source_hashes")
        if isinstance(hashes, list):
            data["source_hashes"] = [
                _pick(item, _RUN_IDENTITY_KEYS) for item in hashes if isinstance(item, dict)
            ]
        if data["next"] is not None:
            reason = (
                "an artifact item was deferred intact"
                if stored.get("natural_deferred")
                else "the call budget ended"
            )
            response_budget.append_hint(
                data,
                f"Analysis is partial because {reason}; call analyze_results with "
                "continue={result_set_id, cursor} from 'next'.",
            )
        coverage = data.get("coverage")
        missing_page = coverage.get("missing_cases") if isinstance(coverage, dict) else None
        if isinstance(missing_page, dict) and missing_page.get("next_cursor") is not None:
            response_budget.append_hint(
                data,
                "coverage.missing_cases is truncated; call analyze_results with "
                "continue={result_set_id, cursor: coverage.missing_cases.next_cursor} "
                "for the next page of missing cases (no work is replayed).",
            )
        failures = data.get("failures")
        has_failures = isinstance(failures, list) and bool(failures)
        missing_total = int(missing_page.get("total", 0)) if isinstance(missing_page, dict) else 0
        data["outcome"] = (
            "failed"
            if not rendered_results and has_failures and data["next"] is None
            else "partial"
            if has_failures or missing_total or data["next"] is not None
            else "complete"
        )
    return data


def columnarize_analysis_view(data: dict[str, Any]) -> None:
    """Render every declared attached-analysis row surface positionally."""
    for entry in data.get("results", {}).values():
        response_budget.columnarize(entry, "reduced")
        response_budget.columnarize(entry, "values")
        per_run = entry.get("per_run")
        if isinstance(per_run, dict):
            response_budget.columnarize(per_run, "items")
        spec = entry.get("spec")
        if isinstance(spec, dict):
            response_budget.columnarize(spec["fail_cases"], "items")
    coverage = data.get("coverage")
    if isinstance(coverage, dict) and isinstance(coverage.get("missing_cases"), dict):
        response_budget.columnarize(coverage["missing_cases"], "items")


def analysis_rows(data: dict[str, Any]) -> list[Any]:
    """Every row this response is currently showing, across all row surfaces.

    Every surface :meth:`_Limits.scaled` shrinks appears here, and nothing else
    does. A spec-heavy or group_by-heavy call is otherwise the size driver the
    shrink rung neither measures nor touches, which degrades to budget_not_met
    on a response the ladder could in fact have fitted.
    """
    rows: list[Any] = []
    for entry in data["results"].values():
        rows.extend(entry.get("reduced", []))
        rows.extend(entry.get("values", []))
        rows.extend(entry.get("groups", []))
        per_run = entry.get("per_run")
        if isinstance(per_run, dict):
            rows.extend(per_run["items"])
        spec = entry.get("spec")
        if isinstance(spec, dict):
            rows.extend(spec["fail_cases"]["items"])
    rows.extend(data["coverage"]["missing_cases"]["items"])
    return rows


# Rung 0's allowlist, declared as data rather than spelled inside the ``if``
# that applies it: a rung that exempts content is the one place a checker can
# silently lose coverage, so it has to be a list a test can read. ``_REMOVE_``
# names optional keys (dropped only when empty), ``_EMPTY_`` required ones
# (emptied in place, never deleted) — pinned against this tool's own schema by
# tests/test_response_budget.py.
_TRIM_REMOVE_RESULT: tuple[str, ...] = ("groups", "values")
_TRIM_REMOVE_ENVELOPE: tuple[str, ...] = ("signals_available",)
# The identity echo is the audit trail, not the way back — rows carry their own
# source label and case_id, and a run is addressed by manifest_id and job_id.
_TRIM_EMPTY_ENVELOPE: tuple[str, ...] = ("source_hashes",)


def _degrade_analysis(
    data: dict[str, Any],
    rung: response_budget.Rung,
    *,
    preserve_provenance: bool = False,
) -> None:
    """Apply the budget ladder's in-place presentation rungs to this envelope.

    The answer rung and the shrink rung are not here: revoking an opt-in changes
    what gets computed, and shrinking a page has to happen before its cursor is
    minted, so both are inputs to :func:`_assemble` instead. Nothing below
    touches failures, observations, warnings, completeness or spec verdicts.

    Idempotent, so the ladder may re-apply it to an envelope it already degraded
    on the way down.
    """
    if rung.trim:
        for entry in data["results"].values():
            response_budget.apply_trim(entry, remove=_TRIM_REMOVE_RESULT)
        # An explicit include.provenance is a caller opt-in, and the trim rung's
        # charter is to revoke none — so below the answer rung (the rung whose
        # documented job IS revoking opt-ins) an enriched identity echo
        # survives. Once the answer rung has revoked the opt-in, emptying the
        # echo is the ladder working as specified, not a second revocation.
        keep = preserve_provenance and not rung.answer_channel
        empty = () if keep else _TRIM_EMPTY_ENVELOPE
        response_budget.apply_trim(data, remove=_TRIM_REMOVE_ENVELOPE, empty=empty)
    if rung.columnar:
        for entry in data["results"].values():
            response_budget.columnarize(entry, "reduced")
            response_budget.columnarize(entry, "values")
            per_run = entry.get("per_run")
            if isinstance(per_run, dict):
                response_budget.columnarize(per_run, "items")
            spec = entry.get("spec")
            if isinstance(spec, dict):
                response_budget.columnarize(spec["fail_cases"], "items")
        response_budget.columnarize(data["coverage"]["missing_cases"], "items")


#: This tool's budget epilogue. No hint mirror: an analyze ``hint`` is the resume
#: route for a partial call, and the ladder's own note reaches the caller on
#: ``observations`` without displacing it.
_BUDGET_NOTES = response_budget.Notes(
    cut=(
        "presentation was reduced; every recipe that produced a result "
        "still has one, and its reductions and spec verdict are intact."
    ),
    route=(
        "Ask again with a larger 'budget' for the full presentation, or continue "
        "with continue={result_set_id, cursor}."
    ),
)


async def _negotiate_analysis(
    budget: ResponseBudget, a: AnalysisEvaluation
) -> types.CallToolResult:
    """Assemble this analysis at the mildest ladder rung that fits ``budget``."""
    base = _Limits.of(a.include)
    text = ""
    rendered: dict[str, Any] = {}
    # What the standing assembly was built for. Only two things change what
    # :func:`_assemble` produces — the answer channel and the limits — so a rung
    # that changes neither is the previous rung degraded one step further, not a
    # second pass over the same finished work.
    built_for: tuple[bool, _Limits] | None = None

    async def render(rung: response_budget.Rung) -> dict[str, Any]:
        nonlocal text, rendered, built_for
        limits = (
            base.scaled(response_budget.RowMeasure.of(analysis_rows(rendered)), rung)
            if rung.shrink
            else base
        )
        if built_for != (rung.answer_channel, limits):
            rendered, text = _assemble(a, rung, limits)
            built_for = (rung.answer_channel, limits)
        _degrade_analysis(rendered, rung, preserve_provenance=a.include.provenance)
        return rendered

    assert budget.tokens is not None  # the undegraded path never reaches here
    result = await response_budget.negotiate(
        budget.tokens, render, _BUDGET_NOTES, max_rung=budget.max_rung
    )
    response_budget.attach_notes(result, _BUDGET_NOTES)
    return format_response(text, result.data)


async def _evaluate_unit(
    recipe: Recipe,
    key: str,
    selected_runs: list[_ResolvedRun],
    runs: list[_ResolvedRun],
    precheck: dict[str, SourceFault],
    manifests: dict[str, dict[str, Any]],
    state: SessionState,
    item: result_store.ResultSet,
    item_deadline: float,
    step_cache: dict[str, list[dict[str, Any]]],
    *,
    per_run_offset: int,
    position: int,
) -> WorkUnit:
    """Evaluate one recipe over the runs that survived the source precheck.

    A source the precheck rejected is reported as this unit's failure and not
    read: the recipe still produces an entry from the runs that are intact,
    which is what makes one drifted source a partial result instead of a lost
    call.
    """
    selected_ids = _selected_manifest_ids(recipe, runs)
    failures = [
        Failure(
            code=fault.code,
            stage="source_precheck",
            where=manifest_id,
            message=fault.message,
        )
        for manifest_id, fault in precheck.items()
        if manifest_id in selected_ids
    ]
    eligible_runs = [run for run in selected_runs if run.manifest_id not in precheck]
    # max_points bounds the INLINE series only; a csv artifact is written at
    # full fidelity over the requested window. Say so when the caller set it
    # on a csv recipe, so a silently inert argument becomes a stated fact.
    # Carried on the unit, so it travels with the result it describes.
    observations: list[Observation] = []
    if (
        isinstance(recipe, WaveformRecipe)
        and recipe.format == "csv"
        and "max_points" in recipe.model_fields_set
    ):
        observations.append(
            Observation(
                code="max_points_not_applied",
                kind="provenance",
                detail=(
                    f"Recipe {key!r} sets max_points, which bounds format='inline' "
                    "series only; the csv artifact holds every sample in the "
                    "requested window. Narrow the recipe window to write fewer rows."
                ),
            )
        )

    records, item_failures, pending = await _evaluate_item(
        recipe,
        eligible_runs,
        manifests,
        state,
        item,
        item_deadline,
        step_cache,
    )
    failures.extend(item_failures)
    observations.extend(_absence_observations(recipe, key, records))
    return WorkUnit(
        key=key,
        recipe=recipe,
        records=records,
        failures=failures,
        pending=pending,
        eligible_ids={run.manifest_id for run in eligible_runs},
        per_run_offset=per_run_offset,
        position=position,
        observations=observations,
    )


def _discard_drifted(
    processed: list[WorkUnit],
    pending: list[_PendingArtifact],
    postcheck: dict[str, SourceFault],
) -> list[_PendingArtifact]:
    """Drop every record and artifact derived from a source that drifted.

    Drift is discovered after the reads, so this runs over finished work: the
    records go, the temp artifacts are left for the caller to remove, and each
    unit that used the source gains the failure that says why its rows thinned.
    """
    failed_ids = set(postcheck)
    for unit in processed:
        relevant = {
            manifest_id: fault
            for manifest_id, fault in postcheck.items()
            if manifest_id in unit.eligible_ids
        }
        if not relevant:
            continue
        unit.records = [record for record in unit.records if record.manifest_id not in failed_ids]
        unit.failures.extend(
            Failure(
                code=fault.code,
                stage="source_postcheck",
                where=manifest_id,
                message=fault.message,
            )
            for manifest_id, fault in relevant.items()
        )
    return [artifact for artifact in pending if artifact.manifest_id not in failed_ids]


async def _publish_unit_artifacts(
    processed: list[WorkUnit], pending: list[_PendingArtifact], deadline: float
) -> None:
    """Publish every surviving artifact once and hand each record its handle.

    A record's ``artifact`` block is written before the file has a digest or a
    size, because those are only known once the temp is complete; merging the
    handles here fills them in on the dicts the entries will render. A publish
    that fails takes the temps with it and marks only the units that had one.
    """
    try:
        handles = await _publish(pending, deadline)
    except (LTSpiceMCPError, OSError) as exc:
        await asyncio.to_thread(_remove_pending, pending)
        code = (
            "analysis_deadline"
            if isinstance(exc, AnalysisDeadlineExceeded)
            else "artifact_publish_failed"
        )
        for unit in processed:
            if unit.pending:
                unit.failures.append(
                    Failure(code=code, stage="publish", where=unit.key, message=str(exc))
                )
        return
    by_path = {handle["path"]: handle for handle in handles}
    for unit in processed:
        for record in unit.records:
            artifact = record.value.get("artifact")
            if isinstance(artifact, dict) and artifact["path"] in by_path:
                artifact.update(by_path[artifact["path"]])


@dataclass(frozen=True)
class _PageStop:
    """Stop the drive once one per_run page of a recipe is filled.

    One object rather than a flag plus a limit, because the limit means nothing
    without the stop: a reservoir size handed to a drive that runs to
    completion would silently change nothing. ``reservoir`` overrides the
    caller's requested page size for THIS evaluation only — the result set
    keeps the requested limit, so the cursor it mints resumes that view.
    """

    reservoir: int | None = None


@dataclass
class _DriveStart:
    """Where a drive begins: the immutable set and the position within it."""

    item: result_store.ResultSet
    position: int
    intra_item: int
    missing_offset: int
    include: AnalyzeInclude


async def _resolve_drive_start(
    args: AnalyzeResultsInput,
    state: SessionState,
    continuation: AnalysisContinuationPosition | None,
    loaded: result_store.ResultSet | None,
    call_deadline: float,
    digest_cache: _DigestCache,
) -> _DriveStart:
    """Load or create the result set this drive runs over, and find its start.

    Four ways in — a neutral continuation, a fresh request, a caller cursor, and
    a per_run page cursor — and each decides both the set and the position
    inside it. Kept together because they are one decision: which of the four a
    call is determines what may be validated (a page cursor must match the
    request that minted it) and which view the drive inherits.
    """
    page_cursor = (
        args.include.per_run.cursor
        if args.continuation is None and args.include.per_run is not None
        else None
    )
    cursor_has_view = False
    cursor_fields: list[str] | None = None
    if continuation is not None:
        # The set is immutable, so a caller driving successive continuations can
        # hand back the one it already holds instead of re-reading it per drive.
        item = (
            loaded
            if loaded is not None and loaded.result_set_id == continuation.result_set_id
            else await asyncio.to_thread(
                result_store.load,
                continuation.result_set_id,
                state.working_dir,
            )
        )
        position = continuation.work_index
        intra_item = continuation.row_offset
        missing_offset = continuation.missing_offset
        cursor_has_view = continuation.has_explicit_view
        cursor_fields = (
            list(continuation.view_fields) if continuation.view_fields is not None else None
        )
    elif args.continuation is None and page_cursor is None:
        item = await _create_result_set(args, state, call_deadline, digest_cache)
        position = 0
        intra_item = 0
        missing_offset = 0
    elif args.continuation is not None:
        item = await asyncio.to_thread(
            result_store.load,
            args.continuation.result_set_id,
            state.working_dir,
        )
        position, intra_item, missing_offset = result_store.decode_cursor(
            args.continuation.cursor, item
        )
        cursor_has_view, cursor_fields = result_store.cursor_view(args.continuation.cursor)
    else:
        assert page_cursor is not None
        item = await asyncio.to_thread(
            result_store.load,
            result_store.cursor_result_set_id(page_cursor),
            state.working_dir,
        )
        cursor_has_view, cursor_fields = result_store.cursor_view(page_cursor)
        stored_include = AnalyzeInclude.model_validate(item.inputs.get("include", {}))
        inherited_fields = cursor_fields if cursor_has_view else stored_include.fields
        if "fields" in args.include.model_fields_set and args.include.fields != inherited_fields:
            raise ResultError(
                "The per_run cursor carries a different include.fields view; replay "
                "page 1 with the new fields projection instead of changing fields on "
                "a cursor call."
            )
        request_matches = item.inputs.get("request_hash") == _request_hash(args)
        if not request_matches and not cursor_has_view:
            legacy_include = args.include.model_copy(update={"fields": stored_include.fields})
            legacy_args = args.model_copy(update={"include": legacy_include})
            request_matches = item.inputs.get("request_hash") == _request_hash(
                legacy_args,
                include_fields=True,
            )
        if not request_matches:
            raise ResultError(
                "The per_run cursor does not match these sources, recipes, grouping, "
                "and include options."
            )
        position, intra_item, missing_offset = result_store.decode_cursor(page_cursor, item)

    include = AnalyzeInclude.model_validate(item.inputs.get("include", {}))
    if continuation is not None or args.continuation is not None or page_cursor is not None:
        include = include.model_copy(
            update={"fields": cursor_fields if cursor_has_view else include.fields}
        )
    return _DriveStart(item, position, intra_item, missing_offset, include)


async def _evaluate_analysis_drive(
    args: AnalyzeResultsInput,
    state: SessionState,
    *,
    continuation: AnalysisContinuationPosition | None = None,
    loaded: result_store.ResultSet | None = None,
    page_stop: _PageStop | None = None,
) -> AnalysisEvaluation:
    """Implementation shared by the neutral seam and MCP's paged presentation."""
    loop = asyncio.get_running_loop()
    call_deadline = loop.time() + state.config.analysis_budget_s
    # Per-call caches: one file hash per (path, mtime, size); one step-table
    # parse per log path. Shared across manifest build, verification and steps.
    digest_cache: _DigestCache = {}
    step_cache: dict[str, list[dict[str, Any]]] = {}

    start = await _resolve_drive_start(
        args, state, continuation, loaded, call_deadline, digest_cache
    )
    item = start.item
    position, intra_item, missing_offset = start.position, start.intra_item, start.missing_offset
    include = start.include

    runs = _deserialize_runs(item, state)
    manifests = {str(manifest["manifest_id"]): manifest for manifest in item.source_manifests}
    missing = list(item.inputs.get("missing", []))
    # Skipped work carries the position it was skipped at: the budget ladder can
    # stop this response short of where the loop ended, and a rejection for work
    # that now resumes unread would be reported twice.
    skipped: list[tuple[int, Failure]] = []
    observations = [Observation.of(data) for data in item.inputs.get("observations", [])]
    group_by = list(item.inputs.get("group_by", []))
    declared_labels = {
        str(source.get("label"))
        for source in item.inputs.get("sources", [])
        if isinstance(source, dict)
    }
    work_done = False
    deferred = False

    def _skip(failure: Failure) -> None:
        """Record a per-recipe rejection and advance past it (item is done)."""
        nonlocal position, intra_item, work_done
        skipped.append((position, failure))
        position += 1
        intra_item = 0
        work_done = True

    # Source-drift precheck (once per call): verify every direct source before
    # any recipe reads it. The shared digest cache makes this free for the
    # sources a fresh set just hashed; a continuation re-checks them against the
    # manifest.
    precheck_deadline = loop.time() + max(_MIN_ITEM_DEADLINE_S, call_deadline - loop.time())
    precheck = await _verify_direct_sources(
        item.source_manifests,
        {run.manifest_id for run in runs},
        precheck_deadline,
        digest_cache,
    )

    processed: list[WorkUnit] = []
    all_pending: list[_PendingArtifact] = []
    evaluated_ids: set[str] = set()

    while position < len(item.work):
        work_item = item.work[position]
        raw_recipe = work_item["recipe"]
        key = str(raw_recipe.get("key") or f"recipe_{work_item['index']}")
        try:
            recipe = validate_recipe(raw_recipe)
        except (ValueError, TypeError) as exc:
            _skip(
                Failure(
                    code="recipe_invalid",
                    stage="validate",
                    where=f"recipes[{work_item['index']}]",
                    message=recipe_error(exc),
                )
            )
            continue

        unknown_labels = set(recipe.sources or ()) - declared_labels
        if unknown_labels:
            _skip(
                Failure(
                    code="recipe_invalid",
                    stage="validate",
                    where=f"recipes[{work_item['index']}]",
                    message=(
                        "recipe sources name unknown labels: " + ", ".join(sorted(unknown_labels))
                    ),
                )
            )
            continue

        selected_runs = [
            run for run in runs if recipe.sources is None or run.label in set(recipe.sources)
        ]
        estimate = _artifact_estimate(recipe, selected_runs)
        remaining = call_deadline - loop.time()
        if estimate > state.config.analysis_budget_s * _ARTIFACT_SAFETY_FACTOR:
            _skip(
                Failure(
                    code="artifact_too_large",
                    stage="preflight",
                    where=key,
                    message=(
                        "Artifact estimate exceeds the per-call safety bound. The "
                        "bound is computed from raw file size and signal count "
                        "before any data is read, so request fewer signals or "
                        "select fewer/smaller runs; narrowing the window or "
                        "lowering max_points does not move it. A single large "
                        "raw read for a single signal has no request-side lever "
                        "— raise [analysis] analysis_budget_s "
                        "(LTSPICE_MCP_ANALYSIS_BUDGET_S) instead."
                    ),
                )
            )
            continue
        if estimate > remaining and work_done:
            deferred = True
            break
        if remaining <= 0 and work_done:
            break
        item_deadline = loop.time() + max(_MIN_ITEM_DEADLINE_S, remaining)
        unit = await _evaluate_unit(
            recipe,
            key,
            selected_runs,
            runs,
            precheck,
            manifests,
            state,
            item,
            item_deadline,
            step_cache,
            per_run_offset=intra_item,
            position=position,
        )
        evaluated_ids.update(unit.eligible_ids)
        all_pending.extend(unit.pending)
        processed.append(unit)

        # Per_run pagination: decided on record count so the break can stop the
        # call here; the page and its cursor are built during assembly below,
        # which re-decides this against whatever limit it assembles at.
        per_run_limit = include.per_run.limit if include.per_run is not None else None
        if page_stop is not None and page_stop.reservoir is not None and per_run_limit is not None:
            per_run_limit = page_stop.reservoir
        if (
            page_stop is not None
            and per_run_limit is not None
            and (unit.records or not unit.failures)
            and intra_item + per_run_limit < len(unit.records)
        ):
            intra_item += per_run_limit
            work_done = True
            break
        position += 1
        intra_item = 0
        work_done = True

    # Source-drift postcheck (once per call): verify every evaluated source
    # before any artifact is published or results are returned. Drift discards
    # the records and artifacts derived from that source across every recipe
    # that used it.
    postcheck_deadline = loop.time() + max(_MIN_ITEM_DEADLINE_S, call_deadline - loop.time())
    postcheck = await _verify_direct_sources(
        item.source_manifests,
        evaluated_ids,
        postcheck_deadline,
        digest_cache,
    )
    if postcheck:
        failed_ids = set(postcheck)
        await asyncio.to_thread(
            _remove_pending,
            [artifact for artifact in all_pending if artifact.manifest_id in failed_ids],
        )
        all_pending = _discard_drifted(processed, all_pending, postcheck)

    await _publish_unit_artifacts(processed, all_pending, postcheck_deadline)

    signals: dict[str, list[str]] | None = None
    if include.signals_available:
        signals = {}
        for run in runs:
            try:
                raw = await services.load_raw(run.source.raw, state)
                signals[run.manifest_id] = list(raw.get_trace_names())
            except LTSpiceMCPError:
                signals[run.manifest_id] = []

    return AnalysisEvaluation(
        item=item,
        processed=processed,
        runs=runs,
        missing=missing,
        missing_offset=missing_offset,
        skipped=skipped,
        base_observations=observations,
        include=include,
        group_by=group_by,
        natural_position=position,
        natural_intra=intra_item,
        deferred=deferred,
        signals=signals,
    )


async def evaluate_analysis_results(
    args: AnalyzeResultsInput,
    state: SessionState,
    *,
    continuation: AnalysisContinuationPosition | None = None,
    loaded: result_store.ResultSet | None = None,
) -> AnalysisEvaluation:
    """Run one bounded neutral analysis drive.

    The returned rows, reductions, facts, failures, and missing cases are not
    projected or capped.  ``continuation`` is the evaluator's internal position;
    callers that need a complete result can pass it back repeatedly without
    decoding or merging rendered MCP pages.  Passing the previous drive's
    ``item`` as ``loaded`` skips re-reading the immutable set from disk.
    """
    return await _evaluate_analysis_drive(
        args,
        state,
        continuation=continuation,
        loaded=loaded,
    )


def complete_analysis_evaluations(drives: list[AnalysisEvaluation]) -> dict[str, Any]:
    """Render complete Python-door data from bounded neutral evaluator drives."""
    if not drives:
        raise ValueError("At least one analysis evaluation drive is required")

    first = drives[0]
    last = drives[-1]
    processed = [unit for drive in drives for unit in drive.processed]
    skipped = [failure for drive in drives for failure in drive.skipped]
    signals: dict[str, list[str]] | None = None
    if any(drive.signals is not None for drive in drives):
        signals = {}
        for drive in drives:
            if drive.signals is not None:
                signals.update(drive.signals)
    combined = replace(
        first,
        processed=processed,
        skipped=skipped,
        natural_position=last.natural_position,
        natural_intra=last.natural_intra,
        deferred=any(drive.deferred for drive in drives),
        signals=signals,
    )
    max_rows = max([len(combined.missing), *(len(unit.records) for unit in processed), 1])
    limits = _Limits(
        per_run=max_rows if combined.include.per_run is not None else None,
        rows=max_rows,
        fail_cases=max_rows,
        groups=None,
    )
    data, _text = _assemble(combined, None, limits)
    data["failures"] = [failure.wire() for failure in combined.failure_inventory]
    data["observations"] = [
        observation
        for observation in data["observations"]
        if observation.get("code") != "failures_truncated"
    ]
    return data


async def capture_attached_analysis(
    args: AnalyzeResultsInput, state: SessionState
) -> dict[str, Any]:
    """Evaluate once and retain the neutral bounded snapshot for a job sidecar."""
    assembly = await _evaluate_analysis_drive(
        args,
        state,
        page_stop=_PageStop(reservoir=MAX_PAGE_SIZE),
    )
    return await asyncio.to_thread(_snapshot_from_assembly, assembly)


@registry.tool(
    name="analyze_results",
    description=(
        "Measure finished simulation results: apply typed recipes to completed "
        "jobs and/or .raw files and get values, reductions, group splits and spec "
        "verdicts attributed to case, run and .step. One call spans many sources "
        "and many metrics, so batch them instead of calling per metric. Work is "
        "bounded by a compute budget; a partial response returns a result_set_id "
        "and cursor to resume with 'continue'. On a wide sweep set include.fields "
        "to return only the numbers you need."
    ),
    input_model=AnalyzeResultsInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
    output_schema=OUTPUT_SCHEMA,
)
async def handle_analyze_results(
    args: AnalyzeResultsInput, state: SessionState
) -> types.CallToolResult:
    assembly = await _evaluate_analysis_drive(args, state, page_stop=_PageStop())
    budget = resolve_response_budget(args.budget, state)
    if budget.tokens is None:
        data, text = _assemble(assembly, None, _Limits.of(assembly.include))
        return format_response(text, data)
    return await _negotiate_analysis(budget, assembly)
