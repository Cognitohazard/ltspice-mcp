"""Consolidated analysis over immutable result-source manifests.

One evaluation (:func:`evaluate_analysis`) runs every recipe of a request and
gives its complete answer (:func:`_assemble`); nothing in it knows a page, a
cursor or a reply size. The Python API returns that answer
(:func:`complete_analysis`); an MCP reply is presented from it by the reply
layer (``lib/presenter.py``), which stores a cut answer and reads its pages
back without evaluating again (:func:`_read_on`).
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import functools
import json
import math
import os
import re
import statistics
import time
from collections.abc import Callable, Collection, Mapping, MutableMapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path, PurePath
from stat import S_ISREG
from typing import Annotated, Any, NamedTuple
from weakref import WeakValueDictionary

import numpy as np
from mcp import types
from pydantic import BeforeValidator, Field, PrivateAttr, SkipValidation, model_validator

from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    InvalidCursorError,
    LTSpiceMCPError,
    PathSecurityError,
    ResultError,
    ResultFileNotFoundError,
    python_route_text,
)
from ltspice_mcp.lib import (
    O_BINARY,
    analysis_snapshot,
    fsync_dir,
    fsync_fd,
    metrics,
    not_shown,
    replace_file,
    response_budget,
    result_store,
    services,
)
from ltspice_mcp.lib.continuation import Unfilled, call_to_continue
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedRaw, PlotDescriptor
from ltspice_mcp.lib.format import cap_note, format_spice_value, held_to_cap, unique_name
from ltspice_mcp.lib.log_parser import diagnostic_collapse_key, printed_once_refusal
from ltspice_mcp.lib.log_types import given_no_value, measurement_named, run_measurement_facts
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.points import RowIdentity, point_identity
from ltspice_mcp.lib.presenter import (
    Annotate,
    Answer,
    ListCut,
    Series,
    StoredAnswer,
    join_pointer,
    resolve_pointer,
    split_pointer,
    stored_cursor,
)
from ltspice_mcp.lib.projection import (
    absent_path_warnings,
    keep_plan,
    project_row,
    resolve_field_paths,
)
from ltspice_mcp.lib.raw_parser import (
    get_step_count,
    may_hold_compressed_transient,
    safe_magnitude_db,
)
from ltspice_mcp.lib.recipes import (
    WAVEFORM_MAX_POINTS_CAP,
    KeyedRecipe,
    MeasurementsRecipe,
    MultiRecipe,
    OperatingPointRecipe,
    PlotRecipe,
    Recipe,
    ScalarRecipe,
    StepSelectionFields,
    StepSelector,
    WaveformRecipe,
    validate_recipe,  # noqa: TID251 - recipe_of_request, its owner, is in this module
)
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS, NEVER, rows, series
from ltspice_mcp.lib.response_budget import estimate_tokens
from ltspice_mcp.lib.result_kinds import AC_SHAPED
from ltspice_mcp.lib.result_observations import (
    coarse_waveform_observation,
    compressed_xvii_observation,
    deck_observation_inputs,
    measurements_failed_in_every_run,
)
from ltspice_mcp.lib.signal_analysis import downsample_minmax_traces
from ltspice_mcp.lib.simulator_build import Producer
from ltspice_mcp.lib.step_values import step_axis_key
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    CUT_SCHEMA,
    DIAGNOSTICS_COVERAGE_SCHEMA,
    HINT_SCHEMA,
    NOT_ESTABLISHED_SCHEMA,
    PRINTED_ONCE_SCHEMA,
    ROWS_WITHOUT_TARGET,
    RUN_DIAGNOSTICS_SCHEMA,
    STORED_ANSWER_SCHEMA,
    UNREADABLE_SCHEMA,
    CaseSelection,
    NotedModel,
    OptionalRawSelectionFields,
    OtherRawFields,
    RunsSelection,
    StrictModel,
    ToolInput,
    check_runs,
    format_response,
    open_stored_page,
    outcome_of,
    page_schema,
    presented,
    recovery_hint,
    registry,
    resolve_response_budget,
    sanitize_payload,
    say_over_target,
    stored_page,
)
from ltspice_mcp.tools.reference_index import validation_error_detail

_ARTIFACT_SAFETY_FACTOR = 4.0
_MIN_ITEM_DEADLINE_S = 0.05
# Public: the largest per_run page this tool will return. Any surface that lets
# a caller ask for a page size — including run_experiments' attached-analysis
# request — bounds itself on THIS name, so a limit accepted at submission is a
# limit the analysis can actually serve.
MAX_PAGE_SIZE = 100
_FAIL_CASE_PAGE_CAP = 100
_FAILURE_CAP = 100

_NO_ROWS_WARNING = "No rows: this recipe failed wherever it was read; see failures at {places}."
"""Said on the entry of a recipe that produced no rows because it failed, so
the entry points at the failures that explain it."""


def _no_rows_warning(failures: Sequence[Failure]) -> str:
    places = list(dict.fromkeys(failure.where for failure in failures))
    shown = ", ".join(places[:_FAILURE_WHERE_CAP])
    more = (
        f" and {len(places) - _FAILURE_WHERE_CAP} more" if len(places) > _FAILURE_WHERE_CAP else ""
    )
    return _NO_ROWS_WARNING.format(places=shown + more)


def _names_a_log(path: str) -> bool:
    """A source path ending ``.log``, in any case, is a log read alone."""
    return PurePath(path).suffix.casefold() == ".log"


class AnalyzeSourceInput(OptionalRawSelectionFields, OtherRawFields, NotedModel):
    job_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Analyze the results of a job this server ran. Exactly one of "
            "job_id or path. A job still running is reported under "
            "coverage.missing_cases instead of failing the call."
        ),
    )
    path: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Analyze a file this server did not run: a .raw, read with the .log "
            "of its name, or a .log alone for its .meas results. Its rows carry "
            "deck_sha256: null. Exactly one of job_id or path."
        ),
    )
    runs: RunsSelection = Field(
        default="all",
        description=(
            "Which runs of this source to read: 'all', a list of run indices, "
            "or {case_ids: [...]} for an experiment job."
        ),
    )
    label: str = Field(
        default="",
        min_length=1,
        description=(
            "Short unique name for this source; it tags every returned row and is "
            "what a recipe's own 'sources' list refers to. Name the condition "
            "('nominal', 'hot'). Default: the job_id, or the file's stem."
        ),
    )

    @property
    def is_log(self) -> bool:
        """``path`` names a log, read alone; any other path names a RAW, read
        with the ``.log`` of its name."""
        return self.path is not None and _names_a_log(self.path)

    @model_validator(mode="after")
    def _one_source(self) -> AnalyzeSourceInput:
        if (self.job_id is None) == (self.path is None):
            raise ValueError("provide exactly one of job_id or path")
        if self.is_log and self.plot_index is not None:
            raise ValueError("a .log path does not select a RAW plot")
        check_runs(self)
        # Beside a path, other_raw and a case selection are refused as every
        # reader of a path refuses them. A list of runs is not: a run the file
        # does not hold is reported missing, as for a job.
        if self.job_id is None and (
            refusal := services.run_selectors_beside_path(
                other_raw=self.other_raw, case_ids=isinstance(self.runs, CaseSelection)
            )
        ):
            raise ValueError(refusal)
        if "label" not in self.model_fields_set:
            # One source, or several that name themselves: the label only has
            # to tell this source's rows apart, and its job or file does that.
            # AnalyzeResultsInput makes a repeat unique.
            derived = self.job_id or Path(str(self.path)).stem
            object.__setattr__(self, "label", derived)
        return self


@dataclass
class _PendingArtifact:
    pending: Path
    final: Path
    content_type: str
    manifest_id: str


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
    """One item that did not produce a value, and what stage lost it.

    ``python_route`` marks a message ending in an ``api`` snippet; only an MCP
    page, which knows the session's tools, says where the snippet runs.
    """

    code: str
    stage: str
    where: str
    message: str
    python_route: bool = False

    @functools.cached_property
    def reason(self) -> tuple[str, str, str]:
        """What a collapsed failure row groups this one under, read once: a
        budget assembles the same failures once per rung it tries."""
        return (self.code, self.stage, diagnostic_collapse_key(self.message))

    def wire(self, served: Collection[str] | None = None) -> dict[str, Any]:
        message = self.message
        if self.python_route and served is not None:
            message = f"{message} {python_route_text(served)}"
        return {
            "code": self.code,
            "stage": self.stage,
            "where": self.where,
            "message": message,
        }


_FAILURE_WHERE_CAP = 10
"""How many places a collapsed failure row names before deferring to ``count``."""

_ROUTE_KEY = "python_route"
"""Marks a failure of a complete answer whose message ends in an ``api``
snippet: each door words it for itself (``for_session``, ``for_api``)."""

_DIGIT_RUN = re.compile(r"(\d+)")


def _failures_truncated_detail(counts: Mapping[str, int], served: Collection[str] | None) -> str:
    """The ``failures_truncated`` note for a page that lists ``_FAILURE_CAP`` of
    ``counts["rows"]`` failure rows, which stand for ``counts["records"]``
    records. ``served`` says where the ``api`` call it names runs, as
    :func:`python_route_text` does; None leaves that out (the Python API).

    The one text of the note: a page builds it here, and the render of a
    stored analysis builds it again here once it knows which session shows it.
    """
    rows, records = counts["rows"], counts["records"]
    standing_for = f", which stand for {records} failures" if records != rows else ""
    rest = "api.analyze_results returns every record."
    if served is not None:
        rest = f"{rest} {python_route_text(served)}"
    return (
        "failures: "
        + cap_note(_FAILURE_CAP, rows, f"failure records{standing_for}", rest)
        + " Coverage and recipe result presence still reflect the full call."
    )


def _place_order(where: str) -> tuple[Any, ...]:
    """``where`` in reading order: ``recipes[2]`` before ``recipes[10]``."""
    return tuple(
        int(part) if index % 2 else part for index, part in enumerate(_DIGIT_RUN.split(where))
    )


def _collapse_failures(
    failures: Sequence[Failure], served: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """Failure rows in wire shape, one per reason rather than one per place.

    A recipe that fails one way fails that way on every run — a signal no run
    carries is one message per run, identical but for ``where`` — so a wide
    sweep returned a row per run in a channel the budget ladder may never trim.
    Rows sharing ``(code, stage, message)`` therefore become one row naming its
    places, the way the experiment receipt already collapses case failures.

    The message keys through :func:`diagnostic_collapse_key`, which folds
    numbers: a message carrying the run's own numeric state differs in every
    case of a Monte Carlo, and a verbatim key would group none of them. The row
    is one member verbatim, the first by ``where`` so that two identical calls
    pick the same one. ``count`` is the true number of failure records and
    ``wheres`` the distinct places, capped, so a capped list reports its own
    shortfall rather than rounding it away. The Python API's complete result
    keeps every record, uncollapsed.
    """
    grouped: dict[tuple[str, str, str], list[Failure]] = {}
    for failure in failures:
        grouped.setdefault(failure.reason, []).append(failure)
    rows: list[dict[str, Any]] = []
    for members in grouped.values():
        ordered = sorted(members, key=lambda failure: _place_order(failure.where))
        row = ordered[0].wire(served)
        if len(ordered) > 1:
            places = list(dict.fromkeys(failure.where for failure in ordered))
            row["wheres"] = places[:_FAILURE_WHERE_CAP]
            row["count"] = len(ordered)
        rows.append(row)
    return rows


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
    #: Index of this unit in the result set's work list.
    position: int
    observations: list[Observation]


# Reduced/spec rows omit circuit/deck provenance while retaining plot selection.
_IDENTITY_KEYS: tuple[str, ...] = tuple(f.name for f in fields(RowIdentity))
_ATTRIBUTION_KEYS: tuple[str, ...] = (*_IDENTITY_KEYS[:5], "plot_index", "dialect", "snapshot_id")
# Every key a per_run/values row carries, in emission order. ``include.fields``
# paths are rooted here, so this one list is both the projector's alphabet and
# the answer a caller gets when a path names something that does not exist.
_ROW_KEYS: tuple[str, ...] = ("source", *_IDENTITY_KEYS, "value")


def resolve_row_fields(paths: list[str]) -> list[str]:
    """``include.fields`` as a projection reads them, on every surface that
    takes the argument: rooted at a row's own keys, a bare name read under
    ``value``. ``ValueError`` for a path no row can carry."""
    return resolve_field_paths(paths, _ROW_KEYS, argument="include.fields", bare_root="value")


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


class CappedPerRunLimit(StrictModel):
    """A per-run page size, held to ``MAX_PAGE_SIZE`` rather than refused.

    The cap bounds one MCP page, not what the caller may read, so a larger limit
    is served at the cap and said so through :meth:`limit_note`, which only the
    MCP page reports: the Python API returns every row whatever the page size.
    The held value is what the model carries, so a request identity built from
    it names the page actually served.
    """

    limit: int = Field(default=50, ge=1, description="Rows per page, held to 100.")
    _limit_note: str | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _hold_limit_to_cap(self) -> CappedPerRunLimit:
        limit, self._limit_note = held_to_cap("limit", self.limit, MAX_PAGE_SIZE)
        # Past validate_assignment, which would run this validator again.
        object.__setattr__(self, "limit", limit)
        return self

    def limit_note(self) -> str | None:
        """The warning that the limit was held to the cap, or None."""
        return self._limit_note


class PerRunInclude(CappedPerRunLimit):
    cursor: str | None = Field(
        default=None,
        description=(
            "Read on from per_run.next_cursor, with the request that gave it, "
            "include.fields included; another request is refused."
        ),
    )


class AnalyzeInclude(NotedModel):
    """Optional response blocks. The default carries reductions, groups and spec
    verdicts; per-run rows, outliers and signal listings are opt-in."""

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
            "reductions and one without inlines up to 100 rows."
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
            "manifest id. Costs one raw load per run; use it to find signal names, "
            "then turn it off."
        ),
    )
    provenance: bool = Field(
        default=False,
        description=(
            "Add artifact paths and content digests to each 'source_hashes' "
            "entry. Off by default: runs are addressed by manifest_id and "
            "job_id, which are always present."
        ),
    )
    fields: list[str] | None = Field(
        default=None,
        min_length=1,
        max_length=32,
        description=(
            "Keep only these dotted row paths on per_run/values rows. Roots: "
            f"{', '.join(_ROW_KEYS)}; a bare name reads under 'value' "
            "('phase_margin_worst_deg' is 'value.phase_margin_worst_deg'). Escape a "
            r"dot inside a key's own name as '\.', as in 'value.voltages.v(x1\.out)'."
        ),
    )

    @model_validator(mode="after")
    def _fields_name_real_row_keys(self) -> AnalyzeInclude:
        # A projection that silently keeps nothing is worse than no projection
        # at all, so a path that cannot be rooted in a row is rejected here
        # rather than returning empty rows the caller has to explain.
        if self.fields is None:
            return self
        resolved = resolve_row_fields(self.fields)
        if resolved != self.fields:
            object.__setattr__(self, "fields", resolved)  # not re-validated
        # A path named twice, or once bare and once under 'value', keeps one copy.
        self.keep_first("fields")
        return self


def include_flag_coercer(model: type[StrictModel]) -> Callable[[Any], Any]:
    """Build the ``include=['outliers', 'per_run']`` -> ``{name: True}`` reader.

    The list is what a caller writes first; the dict is the shape the recipes
    want, and rejecting the list bought nothing. The accepted names are read
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


_include_flags = include_flag_coercer(AnalyzeInclude)


def coerce_include_flags(value: Any) -> Any:
    """The include reader, with null read as omitted, as the attached analyze
    block reads it."""
    return AnalyzeInclude() if value is None else _include_flags(value)


class ContinueInput(StrictModel):
    result_set_id: str = Field(
        description="The result_set_id of the cut reply being read on.",
    )
    cursor: str = Field(
        description=(
            "A cut entry's next, a page's next_cursor or a rows_not_shown row's "
            "cursor, from that reply. A page is read from the stored answer; "
            "nothing is evaluated again."
        ),
    )


class AnalyzeResultsInput(StepSelectionFields, ToolInput, NotedModel):
    sources: list[AnalyzeSourceInput] | None = Field(
        default=None,
        max_length=64,
        description=(
            "What to read — up to 64 jobs, RAW or log files, each under a "
            "unique label. Every recipe runs against every source unless it "
            "names a subset. Required unless 'continue' is given."
        ),
    )
    # SkipValidation preserves the strict A.2 union in JSON Schema while
    # allowing the handler to validate each item independently.
    recipes: list[SkipValidation[Recipe]] | None = Field(
        default=None,
        max_length=256,
        description=(
            "The measurements to take, up to 256, each returned under its own "
            "unique 'key'. Ask for every metric in one call; a recipe that "
            "fails is reported under 'failures' and does not fail the others."
        ),
    )
    group_by: list[str] = Field(
        default_factory=list,
        description=(
            "Split each recipe's reductions along these dimensions: 'circuit', or "
            "the name of a parameter varied by a variation or within a run. Empty "
            "gives one reduction over every row."
        ),
    )
    include: Annotated[
        AnalyzeInclude,
        BeforeValidator(
            coerce_include_flags,
            json_schema_input_type=AnalyzeInclude | list[str] | None,
        ),
    ] = Field(
        default_factory=AnalyzeInclude,
        description=(
            "Opt-in response blocks. A bare list of names switches on per_run "
            "rows, outliers, signals_available and provenance. The row projection "
            "takes paths, so it is the object form only: include={'fields': "
            "['value.<name>', ...]}. Each block grows the payload."
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
            "Read on in a cut reply: a page of the stored answer, evaluated once. "
            "Request fields sent with it must be the whole request that gave it."
        ),
    )

    @model_validator(mode="after")
    def _new_or_continue(self) -> AnalyzeResultsInput:
        # A continuation reads a page of a stored answer. Fields sent with it
        # are normalized like a new request's here and, when sent, compared
        # with the request the stored answer answers (_read_on): an echo of the
        # original call is accepted, and a change is refused rather than
        # dropped, because raising include.per_run.limit on a page is the
        # obvious thing to try and ignoring it hands back a page the caller
        # did not ask for.
        if self.continuation is None and (not self.sources or not self.recipes):
            raise ValueError("a new analysis requires non-empty sources and recipes")
        sources = self.sources or []
        given = [source.label for source in sources if "label" in source.model_fields_set]
        if len(set(given)) != len(given):
            raise ValueError("source labels must be unique")
        # A label the caller left out is its source's job or file, which two
        # sources can share (one job read twice, for different runs); a -2, -3
        # suffix tells them apart, after every label the caller wrote.
        taken = set(given)
        for source in sources:
            if "label" in source.model_fields_set:
                continue
            label = unique_name(source.label, taken)
            taken.add(label)
            object.__setattr__(source, "label", label)
        keys = [
            str(item.get("key", "")) if isinstance(item, dict) else str(getattr(item, "key", ""))
            for item in self.recipes or []
        ]
        nonempty = [key for key in keys if key]
        if len(set(nonempty)) != len(nonempty):
            raise ValueError("recipe keys must be unique")
        self.keep_first("group_by")
        return self


class _CaptureKey(NamedTuple):
    """One whole-file capture, independent of its source label and RAW plot."""

    raw: Path | None
    log: Path | None
    console: Path | None
    dialect: str | None
    explicit_dialect: str | None
    snapshot_id: str | None
    trusted_job_artifact: bool


def _capture_key(source: services.AnalysisSource) -> _CaptureKey:
    return _CaptureKey(
        source.raw,
        source.log,
        source.console,
        source.dialect,
        source.explicit_dialect,
        (source.identity or {}).get("snapshot_id"),
        source.trusted_job_artifact,
    )


async def _captured_source(
    source: services.AnalysisSource,
    state: SessionState,
    captures: MutableMapping[_CaptureKey, ParsedArtifacts],
    *,
    require_raw: bool,
) -> services.AnalysisSource:
    """Share resident facts across recipes, aliases and plots in one drive.

    The first read still captures content in the contained worker and checks
    the manifest snapshot. Prechecks and postchecks always use fresh sources;
    this map is never persisted or carried into a continuation.
    """
    key = _capture_key(source)
    artifacts = captures.get(key)
    if artifacts is None or (require_raw and artifacts.raw is None):
        artifacts = await services.load_artifacts(source, state, require_raw=require_raw)
        captures[key] = artifacts
    return replace(source, captured=artifacts)


@dataclass(frozen=True)
class _ResolvedRun:
    manifest_id: str
    label: str
    source: services.AnalysisSource
    job_id: str | None


def _whole_page(items: list[Any]) -> dict[str, Any]:
    """``items`` whole, in the page shape a reply's ``per_run``,
    ``coverage.missing_cases`` and ``spec.fail_cases`` keep until the next
    breaking release; what a reply holds of it is said there when it is cut."""
    return {"items": items, "total": len(items), "returned": len(items), "truncated": False}


def _pick(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Project ``keys`` out of ``source`` (identity/attribution subsets)."""
    return {key: source[key] for key in keys}


#: The fields of a row's value that state a fact about the row: a
#: measurement that gave no value and why, and what the simulator said about
#: the run. A lean row and a projected row keep each one the value carries
#: (:func:`_row_renderer`), and a row left out is named for each
#: (``rows_not_shown``).
_FACT_FIELDS: tuple[str, ...] = (
    "failed_measurements",
    "not_finite",
    "unreadable",
    "answer_unknown",
    "not_established",
    "printed_once",
    "diagnostics",
    "diagnostics_coverage",
)


#: Nested value blocks the lean row keeps anyway. The rule is narrow on
#: purpose: a block belongs here only when the caller cannot get it back from
#: anything else in the response. An artifact handle names a file already
#: written to disk — drop it and the file is unreachable; a fact about the row
#: (``_FACT_FIELDS``) is what a reader must not miss.
_LEAN_KEPT_BLOCKS: frozenset[str] = frozenset({"artifact", *_FACT_FIELDS})


def _scalar_leaves(value: dict[str, Any]) -> dict[str, Any]:
    """``value`` flattened to its scalar leaves, the default lean value.

    The promoted headlines and the simple facts stay; the nested curve/list
    detail is reachable by name, since include.fields=["value"] returns the
    full block. If flattening would empty the value, the full dict stays: lean
    never trades data for absence.

    An ``artifact`` handle survives the flattening. It is a dict, so the
    scalar-leaves rule dropped it — and it is the only thing a caller cannot
    recompute from the response, because it names a file this call has already
    written. A plot recipe rendered that way came back as a series count and
    nothing else, which is not a lean answer to "plot this", it is no answer.
    """
    flat = {
        key: item
        for key, item in value.items()
        if key in _LEAN_KEPT_BLOCKS or not isinstance(item, (dict, list))
    }
    return flat if flat else value


#: Row keys the lean row leaves out: the value, rendered on its own below, and
#: the two digests.
_LEAN_ROW_DROPPED: frozenset[str] = frozenset({"value", "deck_sha256", "snapshot_id"})


def _lean_row(
    row: dict[str, Any], lean_value: Callable[[dict[str, Any]], dict[str, Any]]
) -> dict[str, Any]:
    """Default row rendering — the answer channel.

    Drops the per-row deck and file digests (provenance, reachable via
    include.fields; the file digest is also on the reply's source_hashes), and
    renders ``value`` by the metric's own ``lean_value`` (see _LEAN_VALUES).
    Attribution keys that carry nothing (a null step_index, empty step_values)
    are left out with every other such field, when the reply is sent
    (``tools/_base.reply_payload``).
    """
    out: dict[str, Any] = {key: item for key, item in row.items() if key not in _LEAN_ROW_DROPPED}
    value = row.get("value")
    out["value"] = lean_value(value) if isinstance(value, dict) else value
    return out


def _row_renderer(
    fields: list[str] | None,
    *,
    metric: str,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build one reusable renderer for every row of ``metric`` on a result surface.

    One plan serves both row surfaces, so projection never depends on an
    unrelated pagination choice.
    """
    plan = keep_plan(fields) if fields else None
    lean_value = _LEAN_VALUES.get(metric, _scalar_leaves)

    def render(row: dict[str, Any]) -> dict[str, Any]:
        if plan is None:
            return _lean_row(row, lean_value)
        projected = project_row(row, plan)
        return _with_facts(projected, row)

    return render


def _with_facts(projected: dict[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """A projected row with every fact its value states (``_FACT_FIELDS``),
    whatever ``include.fields`` named: a projection chooses the numbers a
    reply carries, never whether a fact about the row is said."""
    value = row.get("value")
    if not isinstance(value, Mapping):
        return projected
    facts = {key: value[key] for key in _FACT_FIELDS if value.get(key)}
    if not facts:
        return projected
    kept = projected.get("value")
    kept = dict(kept) if isinstance(kept, Mapping) else {}
    return {**projected, "value": {**kept, **{k: v for k, v in facts.items() if k not in kept}}}


def _projection_warnings(records: list[dict[str, Any]], fields: list[str]) -> list[str]:
    """One warning per requested path that no row of this recipe carries.

    It can miss legitimately, because ``include.fields`` is call-global while
    each metric has its own value shape.
    """
    return absent_path_warnings(
        records, fields, argument="include.fields", rows_name="row of this recipe"
    )


def _serialize_run(run: _ResolvedRun) -> dict[str, Any]:
    return {
        "manifest_id": run.manifest_id,
        "label": run.label,
        "raw": str(run.source.raw) if run.source.raw is not None else None,
        "log": str(run.source.log) if run.source.log else None,
        "netlist": str(run.source.netlist) if run.source.netlist else None,
        "console": str(run.source.console) if run.source.console else None,
        "dialect": run.source.dialect,
        "explicit_dialect": run.source.explicit_dialect,
        "plot_index": run.source.plot_index,
        "identity": run.source.identity,
        "trusted_job_artifact": run.source.trusted_job_artifact,
        "job_id": run.job_id,
        "producer": asdict(run.source.producer) if run.source.producer is not None else None,
        "device_table_withheld": run.source.device_table_withheld,
    }


def _stored_producer(data: Any) -> Producer | None:
    """A resolved run's recorded producer; None for a result set stored before
    runs recorded one, or a record that is not one, which reads as a bare path's."""
    if not isinstance(data, dict) or set(data) != {"family", "build", "program"}:
        return None
    if not all(value is None or isinstance(value, str) for value in data.values()):
        return None
    return Producer(**data)


def _deserialize_runs(item: result_store.ResultSet, state: SessionState) -> list[_ResolvedRun]:
    runs: list[_ResolvedRun] = []
    for data in item.inputs.get("resolved_runs", []):
        raw = Path(data["raw"]) if data["raw"] is not None else None
        dialect = data.get("dialect")
        runs.append(
            _ResolvedRun(
                manifest_id=str(data["manifest_id"]),
                label=str(data["label"]),
                source=services.AnalysisSource(
                    raw=raw,
                    log=Path(data["log"]) if data.get("log") else None,
                    netlist=Path(data["netlist"]) if data.get("netlist") else None,
                    console=Path(data["console"]) if data.get("console") else None,
                    dialect=dialect,
                    explicit_dialect=data.get("explicit_dialect"),
                    plot_index=data.get("plot_index", 0),
                    identity=dict(data.get("identity") or {}),
                    trusted_job_artifact=bool(data.get("trusted_job_artifact")),
                    producer=_stored_producer(data.get("producer")),
                    device_table_withheld=data.get("device_table_withheld"),
                ),
                job_id=data.get("job_id"),
            )
        )
    return runs


def _unresolved_source(
    label: str, run_index: int | None, exc: Exception, state: SessionState
) -> dict[str, Any]:
    """The coverage row of a source that names nothing this call can read.

    Coded and hinted as every other surface reports the same failure: the
    error's own code (``job_not_found``, ``result_file_not_found``,
    ``path_denied``) and its kind's recovery step (``recovery_hint``; a refused
    path's is the sandbox guidance, since the file may be perfectly readable).
    """
    error = exc if isinstance(exc, LTSpiceMCPError) else ResultError(str(exc))
    row: dict[str, Any] = {
        "label": label,
        "case_id": None,
        "run_index": run_index,
        "code": error.code,
        "detail": str(exc),
    }
    hint = recovery_hint(error, state)
    if hint is not None:
        row["hint"] = hint
    return row


async def _resolve_sources(
    inputs: list[AnalyzeSourceInput], state: SessionState
) -> tuple[list[_ResolvedRun], list[dict[str, Any]], dict[str, str | None], list[Observation]]:
    runs: list[_ResolvedRun] = []
    missing: list[dict[str, Any]] = []
    source_jobs: dict[str, str | None] = {}
    observations: list[Observation] = []
    for source_input in inputs:
        imported_path = source_input.path
        if imported_path is not None:
            requested_indices = {0} if source_input.runs == "all" else set(source_input.runs)
            for index in sorted(requested_indices - {0}):
                missing.append(
                    {
                        "label": source_input.label,
                        "case_id": None,
                        "run_index": index,
                        "code": "run_not_found",
                        "detail": "An imported source has exactly one outer run (index 0).",
                    }
                )
            if 0 not in requested_indices:
                continue
            try:
                resolved = services.resolve_analysis_source(
                    state,
                    raw_file=None if source_input.is_log else imported_path,
                    log_file=imported_path if source_input.is_log else None,
                    plot_index=source_input.plot_index or 0,
                    dialect=source_input.dialect,
                )
            except (LTSpiceMCPError, OSError) as exc:
                missing.append(_unresolved_source(source_input.label, 0, exc, state))
                continue
            # The file the source names, not only one read with it: a .raw path
            # whose .log is there still names nothing to read signals from.
            named = resolved.raw if resolved.raw is not None else resolved.log
            if named is not None and not await asyncio.to_thread(os.path.isfile, named):
                missing.append(
                    _unresolved_source(
                        source_input.label,
                        0,
                        ResultFileNotFoundError(f"No result file at {named}"),
                        state,
                    )
                )
                continue
            source = replace(
                resolved,
                identity={
                    "case_id": None,
                    "run_index": 0,
                    "assignments": {},
                    "circuit": Path(imported_path).stem,
                    "deck_sha256": None,
                    "step_index": None,
                    "step_values": {},
                    "requested_plot_index": source_input.plot_index,
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
                    code="path_without_deck_provenance",
                    kind="provenance",
                    detail=(
                        f"Source {source_input.label!r} is an imported result path; "
                        "its rows have no deck_sha256 because no producing job was supplied."
                    ),
                )
            )
            continue

        assert source_input.job_id is not None
        try:
            job = await services.resolve_job_async(source_input.job_id, state)
        except LTSpiceMCPError as exc:
            missing.append(_unresolved_source(source_input.label, None, exc, state))
            continue
        record = job.store_path
        source_jobs[job.job_id] = str(record) if record.is_file() else None

        # Whether each case can be read is its own answer (``case_refusal``),
        # whatever the job's status: a case still running costs only itself.
        selected, unmatched = services.select_cases(job, source_input.runs)
        missing.extend(
            {
                "label": source_input.label,
                "case_id": run.case_id,
                "run_index": run.run_index,
                "code": run.code,
            }
            for run in unmatched
        )
        for case in selected:
            ctx = await services.read_selected_case(
                job, case, state, require_raw=False, other_raw=source_input.other_raw
            )
            if isinstance(ctx, services.CaseUnread):
                row: dict[str, Any] = {
                    "label": source_input.label,
                    "case_id": case.case_id,
                    "run_index": case.run_index,
                    "code": ctx.code,
                    "detail": ctx.detail,
                }
                if ctx.evidence is not None:
                    row["evidence"] = ctx.evidence
                missing.append(row)
                continue
            resolved = services.source_for_run(
                ctx, plot_index=source_input.plot_index or 0, dialect=source_input.dialect
            )
            observations.extend(Observation.of(item) for item in resolved.observations)
            resolved = replace(
                resolved,
                identity={
                    **(resolved.identity or {}),
                    "requested_plot_index": source_input.plot_index,
                },
            )
            runs.append(
                _ResolvedRun(
                    f"{source_input.label}:{case.case_id}",
                    source_input.label,
                    resolved,
                    job.job_id,
                )
            )
            await state.note_recent_circuit(case.circuit_path.resolve())

    return runs, missing, source_jobs, observations


_SOLVE_FAILURE_RUN_CAP = 10
"""How many run labels a relayed solve failure names before deferring to ``runs``."""


async def _capture_ltspice_raws(
    runs: list[_ResolvedRun],
    state: SessionState,
    captures: MutableMapping[_CaptureKey, ParsedArtifacts],
    deadline: float | None,
) -> None:
    """Read the raw of each run that may be LTspice's compressed transient,
    for a recipe that reads only the log, so the reply can say the run was
    left coarse. LTspice XVII computes its own .meas results from points its
    compressed raw does not hold (``raw/compress_1mv``: within 0.25% of the
    uncompressed run's, where values read from the stored points were up to
    1.15% off), and the observation says so by build. Read before the log
    is, the capture serves both. A raw whose header says it is no compressed
    transient is not read, and one that is absent or cannot be read is not
    this recipe's failure."""
    for run in runs:
        raw = run.source.raw
        dialect = run.source.explicit_dialect or run.source.dialect
        if raw is None or dialect not in (None, "ltspice"):
            continue
        if run.source.plot_index == 0 and not await asyncio.to_thread(
            may_hold_compressed_transient, raw
        ):
            continue
        with contextlib.suppress(ResultError), services.analysis_deadline(deadline):
            await _captured_source(run.source, state, captures, require_raw=True)


def _coarse_waveform_observations(
    runs: list[_ResolvedRun],
    captures: Mapping[_CaptureKey, ParsedArtifacts],
    *,
    samples: bool = True,
) -> list[Observation]:
    """The transient runs a unit read that waveform compression left coarse,
    and the LTspice XVII ones it saved compressed that are not coarse
    (``services.compression_drops_points``).

    Every value read from such a run is interpolated across its gaps. Only a
    run whose samples were read has them to look at; the evidence is the first
    such run's. A unit that read only the log (the measurements recipe) took
    no samples, so it is told only of a coarse run.
    """
    coarse: list[tuple[str, dict[str, float]]] = []
    dropped: list[str] = []
    for run in runs:
        artifacts = captures.get(_capture_key(run.source))
        if artifacts is None or artifacts.raw is None:
            continue
        with contextlib.suppress(ResultError):
            raw = services.select_raw(artifacts, run.source.plot_index)
            if (found := services.coarse_waveform(run.source, raw)) is not None:
                coarse.append((run.label, found))
            elif samples and services.compression_drops_points(run.source, raw):
                dropped.append(run.label)
    observations = []
    if coarse:
        observations.append(
            Observation.of(coarse_waveform_observation(coarse, run_cap=_SOLVE_FAILURE_RUN_CAP))
        )
    if dropped:
        observations.append(
            Observation.of(compressed_xvii_observation(dropped, run_cap=_SOLVE_FAILURE_RUN_CAP))
        )
    return observations


def _measurements_failed_everywhere(
    runs: list[_ResolvedRun], captured_logs: Mapping[str, DecodedLog]
) -> list[Observation]:
    """The ``.meas`` that gave no value in any run this analysis read
    (``result_observations.measurements_failed_in_every_run``), for each
    set of runs whose decks request the same names. A run whose log's
    measurements were not read (``services.measurements_unread``) is not
    counted; ``_relay_solve_failures`` already says a log was not read."""
    groups: dict[tuple[str, ...] | None, tuple[list[list[str]], set[str]]] = {}
    for run in runs:
        logs = captured_logs.get(run.manifest_id)
        if logs is None or services.source_measurements_unread(run.source, logs) is not None:
            continue
        section = logs.section("measurements")
        requested = (
            deck_observation_inputs(run.source.netlist)[0]
            if run.source.netlist is not None
            else None
        )
        key = tuple(requested["meas"]) if requested is not None else None
        facts = run_measurement_facts(section["value"] or {}, key)
        read, circuits = groups.setdefault(key, ([], set()))
        read.append(given_no_value(facts))
        circuits.add(str((run.source.identity or {}).get("circuit") or run.label))
    observations: list[Observation] = []
    for key, (read, circuits) in groups.items():
        circuit = ", ".join(sorted(circuits)) if len(groups) > 1 else None
        found = measurements_failed_in_every_run(read, key, circuit=circuit)
        if found is not None:
            observations.append(Observation.of(found))
    return observations


def _relay_solve_failures(
    runs: list[_ResolvedRun], captured_logs: Mapping[str, DecodedLog]
) -> list[Observation]:
    """Relay solve failures from the capture that established each manifest.

    The one chokepoint for this rule on this surface: a run that produced a raw
    despite a failed solve is analyzed and reported like any other, so unless
    the simulator's own line is relayed here the caller reads a number with no
    way to know the solve behind it collapsed. Numeric metrics enforce the
    same rule through ``services.solve_failure_lines`` so they cannot drift.

    One observation per distinct cause rather than per run: a sweep that fails
    to converge fails the same way in every case, and the run labels are what
    distinguishes them. Distinct is measured by
    :func:`diagnostic_collapse_key`, because the simulator's line ends in the
    run's own numbers (``time = 4.4e-05, timestep = 1.2e-19``) and no two runs
    of a sweep abort at the same instant; the relayed line is the first run's,
    verbatim. Log facts the contained parser could not read are reported as
    such — an unread log is a gap in coverage, not an absence of failures.
    """
    grouped: dict[str, tuple[str, list[str]]] = {}
    unread: list[str] = []
    for run in runs:
        if run.source.log is None and run.source.console is None:
            continue
        try:
            logs = captured_logs.get(run.manifest_id)
            if logs is None:
                unread.append(run.label)
                continue
            section = logs.section("diagnostics")
            if section["status"] == "error":
                raise ResultError(section["error"]["message"])
            diagnostics = section["value"]
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
                    f"{len(unread)} run log(s) could not be read or parsed, "
                    "so solve failures may be missing from this report."
                ),
                evidence={
                    "runs": unread[:_SOLVE_FAILURE_RUN_CAP],
                    "run_count": len(unread),
                },
            )
        )
    return relayed


async def _digest(path: Path, deadline: float) -> str:
    return await services.bounded_parse(
        path,
        lambda: result_store.sha256_file(path),
        timeout_s=max(_MIN_ITEM_DEADLINE_S, deadline - asyncio.get_running_loop().time()),
    )


def _manifest_for(run: _ResolvedRun, artifacts: ParsedArtifacts) -> dict[str, Any]:
    files = {item.role: item for item in artifacts.logs.captured.files}
    digests = {
        role: files[role].sha256 if role in files else None for role in ("raw", "log", "console")
    }
    composite = result_store.composite_digest(
        raw_sha256=digests["raw"], log_sha256=digests["log"], console_sha256=digests["console"]
    )
    # This is request selection, not a claim that RAW has decoded successfully.
    plot_index = run.source.plot_index if "raw" in files else None
    dialect = run.source.explicit_dialect or run.source.dialect
    manifest: dict[str, Any] = {
        "manifest_id": run.manifest_id,
        "label": run.label,
        "composite_sha256": composite,
        "plot_index": plot_index,
        "dialect": dialect,
        "producing_dialect": run.source.dialect,
        "explicit_dialect": run.source.explicit_dialect,
        "snapshot_id": artifacts.snapshot_id,
        "selection_sha256": result_store.canonical_hash(
            {
                "composite_sha256": composite,
                "plot_index": plot_index,
                "dialect": dialect,
                "snapshot_id": artifacts.snapshot_id,
            }
        ),
        "trusted_job_artifact": run.source.trusted_job_artifact,
        "job_id": run.job_id,
    }
    for role in ("raw", "log", "console"):
        path = getattr(run.source, role)
        manifest[f"{role}_path"] = str(path) if path is not None else None
        manifest[f"{role}_sha256"] = digests[role]
        manifest[f"{role}_present"] = role in files
        manifest[f"{role}_bytes"] = files[role].size_bytes if role in files else None
    return manifest


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


def _request_inputs(args: AnalyzeResultsInput) -> dict[str, Any]:
    """The request as a result set records it, and what a request's
    identity is read from (:func:`_request_digest`).

    ``work`` is the recipes as work items, and the per_run cursor is cleared,
    since it is a position in the result rather than part of the request.
    """
    include = args.include.model_dump(mode="json")
    if isinstance(include.get("per_run"), dict):
        include["per_run"]["cursor"] = None
    return {
        "sources": [source.model_dump(mode="json") for source in args.sources or []],
        "work": _work_items(list(args.recipes or [])),
        "group_by": list(args.group_by),
        **StepSelection.of(args).as_inputs(),
        "include": include,
    }


async def _create_result_set(args: AnalyzeResultsInput, state: SessionState, deadline: float):
    assert args.sources is not None and args.recipes is not None
    loop = asyncio.get_running_loop()
    initialization_deadline = max(deadline, loop.time() + _MIN_ITEM_DEADLINE_S)
    runs, missing, source_jobs, observations = await _resolve_sources(args.sources, state)
    manifests: list[dict[str, Any]] = []
    captured_logs: dict[str, DecodedLog] = {}
    captures: dict[_CaptureKey, ParsedArtifacts] = {}
    for index, run in enumerate(runs):
        try:
            capture_key = _capture_key(run.source)
            artifacts = captures.get(capture_key)
            if artifacts is None:
                with services.analysis_deadline(initialization_deadline):
                    artifacts = await services.load_artifacts(run.source, state, require_raw=False)
                captures[capture_key] = artifacts
            manifest = _manifest_for(run, artifacts)
            manifests.append(manifest)
            captured_logs[run.manifest_id] = artifacts.logs
            runs[index] = replace(
                run,
                source=replace(
                    run.source,
                    identity={
                        **(run.source.identity or {}),
                        "dialect": manifest["dialect"],
                        "snapshot_id": manifest["snapshot_id"],
                    },
                ),
            )
        except AnalysisDeadlineExceeded as exc:
            raise AnalysisDeadlineExceeded(
                "Analysis initialization exceeded its deadline; no result set was created. "
                "Retry the original request with fewer sources or a larger "
                "[analysis] analysis_budget_s."
            ) from exc
        except (LTSpiceMCPError, OSError) as exc:
            manifests.append(
                {
                    "manifest_id": run.manifest_id,
                    "label": run.label,
                    "raw_path": str(run.source.raw) if run.source.raw is not None else None,
                    "log_path": str(run.source.log) if run.source.log else None,
                    "trusted_job_artifact": run.source.trusted_job_artifact,
                    "job_id": run.job_id,
                    "plot_index": run.source.plot_index if run.source.raw is not None else None,
                    "dialect": run.source.explicit_dialect or run.source.dialect,
                    "snapshot_id": None,
                    "digest_code": "source_unavailable",
                    "digest_error": str(exc),
                }
            )
    observations.extend(_relay_solve_failures(runs, captured_logs))
    observations.extend(
        await asyncio.to_thread(_measurements_failed_everywhere, runs, captured_logs)
    )
    request = _request_inputs(args)
    work = request.pop("work")
    inputs = {
        "working_dir": str(state.working_dir),
        **request,
        "request_hash": _request_digest(args),
        "resolved_runs": [_serialize_run(run) for run in runs],
        "missing": missing,
        "observations": [observation.wire() for observation in observations],
    }
    if loop.time() >= initialization_deadline:
        raise AnalysisDeadlineExceeded(
            "Analysis initialization exceeded its deadline; no result set was created. "
            "Retry the original request with fewer sources or a larger "
            "[analysis] analysis_budget_s."
        )
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
    *,
    state: SessionState,
) -> dict[str, SourceFault]:
    """The selected direct sources that no longer read as their manifest recorded.

    Recapture validates the worker snapshot, including artifact presence and
    console bytes. Equal size and modification time alone cannot prove that
    the bytes are unchanged. No parent-side content read is required.
    """
    failures: dict[str, SourceFault] = {}
    checked: dict[_CaptureKey, SourceFault | None] = {}
    for manifest in manifests:
        manifest_id = str(manifest["manifest_id"])
        if manifest_id not in selected_ids or manifest.get("digest_error"):
            continue
        source = services.AnalysisSource(
            raw=Path(manifest["raw_path"]) if manifest["raw_path"] is not None else None,
            log=Path(manifest["log_path"]) if manifest["log_path"] is not None else None,
            console=Path(manifest["console_path"])
            if manifest["console_path"] is not None
            else None,
            netlist=None,
            dialect=manifest["producing_dialect"],
            explicit_dialect=manifest["explicit_dialect"],
            plot_index=manifest["plot_index"] or 0,
            identity={"snapshot_id": manifest["snapshot_id"]},
            trusted_job_artifact=bool(manifest["trusted_job_artifact"]),
        )
        key = _capture_key(source)
        if key not in checked:
            try:
                with services.analysis_deadline(deadline):
                    await services.load_artifacts(source, state, require_raw=False)
                checked[key] = None
            except AnalysisDeadlineExceeded as exc:
                checked[key] = SourceFault("analysis_deadline", str(exc))
            except (LTSpiceMCPError, OSError) as exc:
                checked[key] = SourceFault("source_drift", str(exc))
        failure = checked[key]
        if failure is not None:
            failures[manifest_id] = failure
    return failures


def _spice(value: float | str | None) -> str | None:
    if value is None:
        return None
    return format_spice_value(value)


def _window_fields(window: Any) -> tuple[str | None, str | None]:
    if window is None:
        return None, None
    return _spice(window.start), _spice(window.end)


class StepSelection(NamedTuple):
    """The step choice a request declared, as the evaluator reads it.

    ``StepSelectionFields`` is where the two fields and their rule are
    declared; this is the resolved pair the evaluation loop carries. It is
    stored in the result set's inputs beside ``group_by``, so a continuation
    replays the same steps.
    """

    step: StepSelector | None = None
    all_steps: bool = False

    @classmethod
    def of(cls, args: StepSelectionFields) -> StepSelection:
        return cls(args.step, args.all_steps)

    @classmethod
    def from_inputs(cls, inputs: Mapping[str, Any]) -> StepSelection:
        raw = inputs.get("step")
        return cls(
            StepSelector.model_validate(raw) if isinstance(raw, Mapping) else None,
            bool(inputs.get("all_steps")),
        )

    def as_inputs(self) -> dict[str, Any]:
        return {
            "step": self.step.model_dump(mode="json") if self.step is not None else None,
            "all_steps": self.all_steps,
        }

    def plan(self, count: int, step_values: Sequence[Mapping[str, Any]]) -> StepPlan:
        """The steps this choice reads of a run of ``count`` steps, each with
        its stepped values: every step, the one ``step`` picks, or the first."""
        if self.all_steps:
            return [
                (index, dict(step_values[index]) if index < len(step_values) else {})
                for index in range(count)
            ]
        if self.step is None:
            return [(0, dict(step_values[0]) if step_values else {})]
        index = self.step.pick(step_values)
        return [(index, dict(step_values[index]))]


#: A resolved step plan: ``(step index, that step's axis values)`` per step.
StepPlan = list[tuple[int, dict[str, Any]]]


@dataclass(frozen=True)
class _RawStepPlan:
    steps: StepPlan
    descriptor: PlotDescriptor


async def _step_plan(
    steps: StepSelection,
    source: services.AnalysisSource,
    state: SessionState,
    plan_cache: dict[tuple[_CaptureKey, int], _RawStepPlan],
) -> _RawStepPlan:
    """The steps one source contributes, built once per source per call.

    The step choice belongs to the call rather than to a recipe, so every
    recipe over the same run resolves to the same plan — the whole plan is
    cached, not just the log parse behind it, and the read is shared rather
    than repeated per (recipe, run). Treat the result as read-only.
    """
    key = (_capture_key(source), source.plot_index)
    cached = plan_cache.get(key)
    if cached is not None:
        return cached
    plan_cache[key] = plan = await _build_step_plan(steps, source, state)
    return plan


async def _build_step_plan(
    steps: StepSelection,
    source: services.AnalysisSource,
    state: SessionState,
) -> _RawStepPlan:
    raw = await services.load_raw(source, state)
    step_values = [dict(step.parameters) for step in raw.descriptor.steps]
    return _RawStepPlan(steps.plan(get_step_count(raw), step_values), raw.descriptor)


async def _measurement_step_plan(
    steps: StepSelection,
    source: services.AnalysisSource,
    state: SessionState,
) -> list[tuple[int | None, dict[str, Any]]]:
    """The steps the measurements recipe reads of one source's log: with no
    step chosen, every step at once, as statistics over them; else the chosen
    rows, each with the stepped values of its ``.step`` line
    (``metrics.measurement_step_values``)."""
    if steps.step is None and not steps.all_steps:
        return [(None, {})]
    logs = await services.load_logs(source, state)
    parsed = services.read_measurements(source, logs)
    if not parsed or not parsed["measurements"]:
        # The metric reads the log again and says why it holds no results.
        return [(0, {})]
    if "op_solves" in parsed:
        # One value printed for a run of several steps
        # (log_parser.answered_measurements). A step chosen by its values has
        # no .step line to be found by: the refusal says what the log holds
        # (log_parser.printed_once_refusal). Every step at once, where the
        # value's step is not established, is one row that names no step.
        answered = await services.answer_measurements(source, parsed)
        if steps.step is not None:
            refusal = printed_once_refusal(
                answered, f"The step {steps.step.axis} = {steps.step.value!r}"
            )
            if refusal is not None:
                raise ResultError(refusal, show_hint=False)
        elif any(
            "step" not in entry.get("printed_once", {"step": 0})
            for entry in answered["measurements"].values()
        ):
            return [(None, {})]
    step_values = metrics.measurement_step_values(parsed, logs.value("steps") or [])
    return list(steps.plan(len(step_values), step_values))


async def _adapter_value(
    recipe: Recipe,
    source: services.AnalysisSource,
    step: int,
    state: SessionState,
) -> dict[str, Any]:
    """One RAW recipe's value, read by the metric function registered for its
    class in ``lib.metrics.METRICS``. Whole-log measurements are read for
    every step of a source at once (``metrics.measurement_rows``).

    Non-finite floats are nulled here rather than inside the metric, because
    that substitution is a property of what can cross the wire — a caller
    reading a metric in-process should see what the maths produced. Doing it at
    this edge also keeps the substitution note attached to the value it
    describes rather than to the whole response.
    """
    return sanitize_payload(await metrics.METRICS[type(recipe)](source, recipe, step, state))


def _log_precision_observations(records: list[Record]) -> list[Observation]:
    """Say that device operating points read here came from LTspice's log
    table, and to how many digits: once for a recipe, whatever it reduces or
    projects its rows to, the way a run's data facts are said."""
    marked = [record for record in records if "device_op_significant_digits" in record.value]
    if not marked:
        return []
    digits = marked[0].value["device_op_significant_digits"]
    runs = sorted({record.source for record in marked})
    return [
        Observation(
            code="device_op_log_precision",
            kind="value",
            detail=(
                f"Device operating points read here come from LTspice's log table, "
                f"which prints every value to {digits} significant digits: each is "
                "rounded by up to half a unit in its last digit. Neither "
                ".options numdgt nor .save @m1[gm] gives more on LTspice."
            ),
            evidence={
                "significant_digits": digits,
                "runs": runs[:_SOLVE_FAILURE_RUN_CAP],
                "run_count": len(runs),
            },
        )
    ]


#: The kinds of a recipe's observation that limit or qualify the number beside
#: it, which the reply carries at its top level whatever fields it shows: a
#: ``reconciliation`` (the number is not of the quantity asked about), a
#: ``coverage`` (a check that did not run, or a part not read) and a ``value``
#: (a fact about the data the number was read from: constant over the window,
#: not finite, corners merged at this sweep density).
_LIFTED_KINDS = frozenset({"reconciliation", "coverage", "value"})

#: The kinds that stay in the row's value, and why: what a reply leaves out
#: with the value does not change what its numbers mean.
_KEPT_KINDS: dict[str, str] = {
    "description": "says how the number was computed, not that it is limited",
    "relay": (
        "a line the simulator printed, which the summary's own errors and warnings "
        "and the reply's relayed solve failures carry"
    ),
}


#: How many observations of one code a recipe lifts to the top level, where
#: the list is never cut: one that names a run's own numbers (an
#: ``extreme_value``) groups with no other, so a large run set would lift one
#: per run. The rest stay in the rows' values, and the last lifted one says
#: how many.
_LIFTED_PER_CODE = 5

#: One lifted observation's identity: its code, kind, detail and evidence.
_Group = tuple[str, str, str, str]


def _value_observations(key: str, records: list[Record]) -> list[Observation]:
    """The observations a recipe's metric made of its rows' values that limit
    or qualify the number beside them (``_LIFTED_KINDS``). Each is given once,
    naming the recipe and every row it was made of, and is taken out of the
    rows' values, so no observation appears twice in one reply whichever
    fields it shows. At most ``_LIFTED_PER_CODE`` of one code are lifted; the
    others stay in the values they were made of, and the reply says how many.

    A lean row leaves the lists of its value out, so this is where they reach
    the reply. A value's other observations (``_KEPT_KINDS``: a fit's
    description, a relayed log line) stay in the value. The values are each
    evaluation's own, built fresh by its metric, so taking an observation out
    of one changes no other reply.
    """
    grouped: dict[_Group, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    sites: list[tuple[Record, _Group | None, Any]] = []
    for position, record in enumerate(records):
        found = record.value.get("observations") if isinstance(record.value, dict) else None
        for item in found or []:
            if not isinstance(item, Mapping) or item.get("kind") not in _LIFTED_KINDS:
                sites.append((record, None, item))
                continue
            evidence = dict(item.get("evidence") or {})
            group = (
                str(item.get("code", "")),
                str(item.get("kind", "")),
                str(item.get("detail", "")),
                json.dumps(evidence, sort_keys=True, default=str),
            )
            # The row's position in its recipe's rows, which a cut reply
            # turns into the cursor that starts at it (say_what_was_cut).
            where: dict[str, Any] = {"source": record.source, "index": position}
            for name in ("case_id", "step_index"):
                if (value := getattr(record.identity, name)) is not None:
                    where[name] = value
            grouped.setdefault(group, (evidence, []))[1].append(where)
            sites.append((record, group, item))
    lifted: list[_Group] = []
    left: dict[str, int] = {}
    for group in grouped:
        if sum(1 for kept in lifted if kept[0] == group[0]) < _LIFTED_PER_CODE:
            lifted.append(group)
        else:
            left[group[0]] = left.get(group[0], 0) + 1
    chosen = set(lifted)
    for record in {id(record): record for record, _, _ in sites}.values():
        kept = [
            item
            for owner, group, item in sites
            if owner is record and (group is None or group not in chosen)
        ]
        if kept:
            record.value["observations"] = kept
        else:
            record.value.pop("observations", None)
    observations = []
    for index, group in enumerate(lifted):
        code, kind, detail, _ = group
        evidence, rows = grouped[group]
        last = all(other[0] != code for other in lifted[index + 1 :])
        if last and left.get(code):
            detail = (
                f"{detail} ({left[code]} more {code} observation(s) of this recipe are in "
                "its rows' values; read them with include.fields=['value'].)"
            )
        observations.append(
            Observation(
                code=code,
                kind=kind,
                detail=detail,
                evidence={**evidence, "recipe": key, "rows": rows},
            )
        )
    return observations


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
    # A run whose empty device table has a reason of its own says that
    # instead (services.device_table_unread): once per reason, naming its runs.
    unread: dict[tuple[str, str], list[str]] = {}
    for record in records:
        if (reason := record.value.pop("device_table_unread", None)) is not None:
            unread.setdefault((reason["code"], reason["detail"]), []).append(record.source)
    observations = [
        Observation(
            code=code,
            kind="coverage",
            detail=f"Recipe {key!r} read a device table that is empty. {detail}",
            evidence={"recipe": key, "runs": len(runs)},
        )
        for (code, detail), runs in unread.items()
    ]
    values = [
        record.value
        for record in records
        if not any(record.source in runs for runs in unread.values())
    ]
    if not values or any(value.get("device_op_points") for value in values):
        return observations
    if not any(metrics.has_active_device(value.get("currents") or {}) for value in values):
        return observations
    return [
        *observations,
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
        ),
    ]


def _artifact_estimate(recipe: Recipe, runs: list[_ResolvedRun]) -> float:
    if not isinstance(recipe, WaveformRecipe) or recipe.format != "csv":
        return 0.0
    size = 0
    for run in runs:
        if run.source.raw is not None:
            with contextlib.suppress(OSError):
                size += run.source.raw.stat().st_size
    return 0.05 + size / 25_000_000 * max(1, len(recipe.signals))


def _sample_list(samples: np.ndarray, *, single: bool) -> list[float]:
    """``samples`` as wire floats, each in the shortest form of its own precision.

    A 32-bit sample widened to a double prints seventeen digits of which nine
    are the value (0.9999544620513916 for 0.99995446). Writing it in its 32-bit
    shortest form reads back as the very same 32-bit value, so nothing is lost.
    ``single`` says the samples hold 32-bit values, which decimation does not
    see: it returns doubles, but only ever ones it picked from the samples.
    """
    if not single:
        return samples.tolist()
    return samples.astype(np.float32).astype(str).astype(np.float64).tolist()


def _trace_components(wave: np.ndarray, analysis: str) -> dict[str, np.ndarray]:
    """One trace's inline components, using AC magnitude/phase only for AC plots."""
    if not np.iscomplexobj(wave):
        return {"y": wave}
    if analysis in AC_SHAPED:
        return {
            "magnitude_db": safe_magnitude_db(wave),
            "phase_deg": np.degrees(np.angle(wave)),
        }
    return {"real": np.real(wave), "imag": np.imag(wave)}


def _inline_waveform(
    raw: DecodedRaw,
    signals: list[services.Signal],
    step: int,
    start: str | None,
    end: str | None,
    limit: int,
) -> dict[str, Any]:
    """The inline samples of ``signals`` at one step, at most ``limit`` points each.

    Every trace is read on the same axis through the same window, and
    decimation buckets the axis alone, so the axis is sent once beside the
    series rather than in each.
    """
    axis = metrics.guarded_axis(raw, step)
    lo, hi = 0, len(axis)
    if start is not None or end is not None:
        lo, hi = metrics.window_indices(
            axis, metrics.parse_time(start, "start"), metrics.parse_time(end, "end")
        )
    axis = axis[lo:hi]
    traces = [
        (signal.name, _trace_components(signal.wave(raw, step)[lo:hi], raw.descriptor.analysis))
        for signal in signals
    ]
    samples = [part for _name, parts in traces for part in parts.values()]
    single = [part.dtype == np.float32 for part in samples]
    x = axis
    if len(axis) > limit:
        x, samples = downsample_minmax_traces(axis, samples, limit)
    written = iter(
        [_sample_list(part, single=flag) for part, flag in zip(samples, single, strict=True)]
    )
    return {
        "format": "inline",
        "x": _sample_list(x, single=x.dtype == np.float32),
        "series": [
            {"signal": name, **{component: next(written) for component in parts}}
            for name, parts in traces
        ],
        "points_returned": len(x),
        "points_total": len(axis),
    }


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

    raw = await services.load_raw(run.source, state)
    assert run.source.raw is not None
    start, end = _window_fields(recipe.window)
    if recipe.format == "inline":
        if raw.descriptor.axis is None:
            raise ResultError(
                "This plot has no sampled axis; use inspect results table or CSV export."
            )
        signals = [services.resolve_signal(raw, requested) for requested in recipe.signals]
        max_points, held = held_to_cap(
            "max_points",
            recipe.max_points,
            min(WAVEFORM_MAX_POINTS_CAP, state.config.max_points_returned),
        )
        # Decimating and writing every sample is CPU work proportional to the
        # points kept, so it runs off the event loop as the csv export does.
        inline = await asyncio.to_thread(
            _inline_waveform, raw, signals, step, start, end, max_points
        )
        return (
            {
                **inline,
                # A record's warnings become its recipe's, with how many runs said it.
                **({"warnings": [held]} if held is not None else {}),
                **point_identity(
                    run.source, step, step_values, plot_index=run.source.plot_index
                ).wire(_IDENTITY_KEYS),
            },
            [],
        )

    axis_name = raw.descriptor.axis.name if raw.descriptor.axis is not None else None
    cols: list[services.Signal] = []
    for requested in recipe.signals:
        signal = services.resolve_signal(raw, requested)
        if signal.name == axis_name:
            raise ResultError(f"{requested!r} is the sweep axis, not a signal")
        if signal not in cols:
            cols.append(signal)
    analysis_type = raw.descriptor.analysis
    recipe_hash = result_store.canonical_hash(recipe.model_dump(mode="json"))
    pending, final = result_store.artifact_paths(
        item,
        recipe_key=recipe.key,
        source_digest=manifest["selection_sha256"],
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
        [dict(step.parameters) for step in raw.descriptor.steps],
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
            **point_identity(
                run.source,
                step if export_steps is None or len(export_steps) == 1 else None,
                step_values if export_steps is None or len(export_steps) == 1 else {},
                plot_index=run.source.plot_index,
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

    raw = await services.load_raw(run.source, state)
    assert run.source.raw is not None
    if raw.descriptor.axis is None:
        raise ResultError(
            "This plot has no sampled axis; use inspect results table or CSV export."
        )
    axis_name = raw.descriptor.axis.name
    cols = [services.resolve_signal(raw, signal) for signal in recipe.signals]
    if axis_name in (col.name for col in cols):
        raise ResultError("The sweep axis cannot be plotted as a signal")
    analysis_type = raw.descriptor.analysis
    x_is_log = raw.descriptor.axis is not None and raw.descriptor.axis.quantity == "frequency"
    recipe_hash = result_store.canonical_hash(recipe.model_dump(mode="json"))
    pending, final = result_store.artifact_paths(
        item,
        recipe_key=recipe.key,
        source_digest=manifest["selection_sha256"],
        recipe_hash=recipe_hash,
        suffix="html",
    )
    pending.parent.mkdir(parents=True, exist_ok=True)
    span = recipe.span
    plan = await asyncio.to_thread(
        an.plan_plot,
        raw,
        [cols],
        split_by_unit=True,
        netlist=run.source.netlist,
        steps=[step for step, _ in steps],
        step_dicts=[dict(step.parameters) for step in raw.descriptor.steps],
        analysis_type=analysis_type,
        x_is_log=x_is_log if recipe.log_x is None else recipe.log_x,
        ts=metrics.parse_time(_spice(span.start) if span else None, "span.start"),
        te=metrics.parse_time(_spice(span.end) if span else None, "span.end"),
    )
    facts = await asyncio.to_thread(
        an.build_plot_file,
        raw,
        run.source.raw,
        plan,
        min(100_000, WAVEFORM_MAX_POINTS_CAP),
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
            **point_identity(
                run.source,
                steps[0][0] if len(steps) == 1 else None,
                steps[0][1] if len(steps) == 1 else {},
                plot_index=run.source.plot_index,
            ).wire(_IDENTITY_KEYS),
        },
        [_PendingArtifact(pending, final, "text/html", run.manifest_id)],
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


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


def _measurements_flat(value: dict[str, Any]) -> dict[str, Any]:
    return {
        name: entry.get("mean")
        for name, entry in value.get("stats", {}).items()
        if isinstance(entry, dict)
    }


def _measurements_compact(value: dict[str, Any]) -> dict[str, Any]:
    """A run's ``.meas`` numbers by name, when the run reported one of each.

    One case of a sweep with no ``.step`` holds a single sample per name, so its
    stats block restates that sample as min, max, mean, median, p10 and p90
    beside a zero spread and an empty histogram: about fifteen fields to carry
    one number. Here the number is keyed by name under ``measured``, read by the
    reducer's own extractor so a row and a reduction over it cannot disagree;
    the point a crossing or ``AT`` reported with it goes under ``at``. A name
    the log holds no number for keeps its key with null, so it never reads as
    a missing name, and the value beside says why, each null in one place,
    all of which come with the metric's value (``metrics.measurement_rows``):
    ``failed_measurements`` (LTspice printed it failed), ``unreadable``
    (LTspice printed something other than a finite number, quoted: LTspice 26
    prints a stepped FRA's gain margin ``infdB``, recorded as
    ``raw/fra_step``), ``answer_unknown`` (which field holds the result is not
    known) and ``not_established`` (the printed value is not established as
    the result). A stepped run's per-name spread is its answer, so it gets
    none of this, and neither does an empty selection.
    """
    stats = value.get("stats")
    if not isinstance(stats, dict) or not stats:
        return {}
    if not all(
        isinstance(entry, dict) and entry.get("total_count") == 1 for entry in stats.values()
    ):
        return {}
    measured = _measurements_flat(value)
    compact: dict[str, Any] = {"measured": measured}
    at = {name: entry["at"] for name, entry in stats.items() if "at" in entry}
    if at:
        compact["at"] = at
    return compact


# Metrics whose headline number lives only inside a list (points[]/crossings[])
# where dotted ``include.fields`` projection cannot reach — 13-22x the size of
# the equivalent shell output for a 12-case sweep table, because the caller
# could not name the one leaf it wanted. Promote that number to a flat ``value`` leaf
# at row-build time. bode_point's extractor is the reducer's own, so the
# projected leaf and a reduce over that recipe can never disagree;
# bode_crossing is a variable-length recipe whose category rejects ``reduce``
# at validation, so its rule lives only here. stability shipped its worst-case
# margins flat but left the crossover frequency in a list; this is that rule
# applied uniformly. measurements is the keyed form of it: a run holding one
# sample per .meas gets those samples by name (_measurements_compact).
_HEADLINE_LEAVES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "bode_point": lambda value: dict([_bode_point_sample(value)]),
    # No crossing COUNT here: the adapter caps its list (max_results, default
    # 10) and reports no truncation, so a count would silently saturate — a
    # wrong number dressed as a fact. Null first_crossing_hz carries "never
    # crossed"; ambiguity is visible in the list itself.
    "bode_crossing": lambda value: dict([_crossing_sample(value)]),
    "stability": lambda value: dict([_unity_gain_sample(value)]),
    "measurements": _measurements_compact,
}


def _promote_headlines(metric: str, value: dict[str, Any]) -> dict[str, Any]:
    """``value`` with the metric's headline leaves added; existing keys win."""
    promote = _HEADLINE_LEAVES.get(metric)
    if promote is None or not isinstance(value, dict):
        return value
    for name, leaf in promote(value).items():
        value.setdefault(name, leaf)
    return value


def _measurements_lean(value: dict[str, Any]) -> dict[str, Any]:
    """The answer channel's measurements value: without its stats block when
    ``measured`` carries every number in it. ``include.fields=["value"]``
    returns the block."""
    if "measured" not in value:
        return value
    return {key: item for key, item in value.items() if key != "stats"}


# How each keyed metric flattens its value dict into a {name: number} map.
_KEYED_EXTRACTORS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "measurements": _measurements_flat,
    "operating_point": metrics.operating_point_flat,
}


def _keyed_name(recipe: KeyedRecipe, name: str, flat: Mapping[str, Any]) -> str:
    """The key of a row's flat map that a caller's ``name`` reads: a
    measurement in any case, as the log's reader holds it lower-cased."""
    if isinstance(recipe, MeasurementsRecipe):
        return measurement_named(name, flat) or name
    return name


def _whole(value: dict[str, Any]) -> dict[str, Any]:
    return value


def _without_row_identity(value: dict[str, Any]) -> dict[str, Any]:
    """A waveform or plot value without the run identity it carries.

    Those values name their run inside themselves (``_artifact_record`` reads
    the row's identity back out of them), so on a row that already names it the
    copy in the value says everything twice, digests included.
    """
    return {key: item for key, item in value.items() if key not in _IDENTITY_KEYS}


# How the answer channel renders a metric's value; a metric not listed keeps
# its scalar leaves (_scalar_leaves). Applied to the lean row only, never to a
# projected one. A metric whose row value IS its nested structure keeps it
# whole. A waveform's value is the curve the caller asked for, less the run
# identity the row around it already states. A keyed metric's
# value is a map addressed by name — and operating_point's only FLAT leaves are
# ``step``/``step_count``/``device``, so leaning it returns the row's
# bookkeeping and drops every number: a bias-point read that answers "complete"
# and carries nothing. Derived from _KEYED_EXTRACTORS rather than listed, so a
# new keyed metric cannot be added without this rule following it. A
# measurements value sheds the stats block its 'measured' map restates.
_LEAN_VALUES: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "waveform": _without_row_identity,
    "plot": lambda value: _scalar_leaves(_without_row_identity(value)),
    **dict.fromkeys(_KEYED_EXTRACTORS, _whole),
    "measurements": _measurements_lean,
}


@dataclass
class _FieldTally:
    """One reducible number across a recipe's rows: the rows that carried it,
    and how many rows carried none."""

    samples: list[tuple[Record, float]] = field(default_factory=list)
    no_value: int = 0


def _tallies(recipe: Recipe, records: list[Record]) -> dict[str, _FieldTally]:
    """Each number a reduction or a spec over ``recipe`` reads, tallied over
    every row of it.

    A row with no number for a field (a null, a missing key, a value that is
    not finite) is counted rather than skipped. Leaving it out made a
    measurement that failed in one run of three a statistic over two with
    nothing saying so, a spec pass over the two it could judge, and one that
    failed in every run a name absent from the result.
    """
    if not (getattr(recipe, "reduce", None) or getattr(recipe, "spec", None)):
        return {}
    # The reducer category is the base the recipe inherits (exactly one); a
    # variable-length recipe matches none and yields no tallies.
    out: dict[str, _FieldTally] = {}

    def tally(name: str, candidate: Any, record: Record) -> None:
        entry = out.get(name)
        if entry is None:
            entry = out[name] = _FieldTally()
        number = _number(candidate)
        if number is None:
            entry.no_value += 1
        else:
            entry.samples.append((record, number))

    if isinstance(recipe, ScalarRecipe):
        nested = _SCALAR_NESTED.get(recipe.metric)
        names = _SCALAR_FIELDS.get(recipe.metric, ())
        for record in records:
            value = record.value
            if nested is not None:
                name, candidate = nested(value)
            else:
                name = next((key for key in names if key in value), names[0] if names else "value")
                candidate = value.get(name)
            tally(name, candidate, record)
    elif isinstance(recipe, MultiRecipe):
        pairs = recipe.reduction_fields()
        for record in records:
            for name, actual in pairs:
                tally(recipe.field_for_row(name, record.value), record.value.get(actual), record)
        if recipe.field is None:
            # A bare reduction covers the recipe's whole fixed table, which
            # holds fields a given circuit has none of: a lowpass's low cutoff,
            # a step response's disturbance figures. A field no row has a
            # number for is that, not a measurement that failed, so only a
            # field the caller named is reported with no samples.
            out = {name: entry for name, entry in out.items() if entry.samples}
    elif isinstance(recipe, KeyedRecipe):
        # 'field' means one thing on every category: the single number both a
        # reduction and a spec read. Absent, a keyed recipe covers every key
        # any row carries; a key is a name the run itself asked for (a .meas,
        # a bias-point quantity), so a row without a number for it is counted.
        extract = _KEYED_EXTRACTORS[recipe.metric]
        for record in records:
            flat = extract(record.value)
            if recipe.field is None:
                for name, candidate in flat.items():
                    tally(name, candidate, record)
            else:
                tally(recipe.field, flat.get(_keyed_name(recipe, recipe.field, flat)), record)
        # A key appears at most once in a row, so the rows with no number for
        # it, null or absent, are every row its samples do not account for.
        for entry in out.values():
            entry.no_value = len(records) - len(entry.samples)
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


def _reduce(recipe: Recipe, tallies: dict[str, _FieldTally]) -> list[dict[str, Any]]:
    stats = list(getattr(recipe, "reduce", []))
    reduced: list[dict[str, Any]] = []
    for field_name, tally in tallies.items():
        values = [value for _, value in tally.samples]
        # Present only when some row had no number: absent, every row the
        # reduction covers carried one.
        no_value = {"no_value_count": tally.no_value} if tally.no_value else {}
        for stat in stats:
            reduced.append(
                {
                    "field": field_name,
                    "stat": stat,
                    "value": _stat(stat, values),
                    **_attribution(stat, tally.samples),
                    **no_value,
                }
            )
    return reduced


def _spec(
    recipe: Recipe,
    tallies: dict[str, _FieldTally],
    rows: int,
    *,
    incomplete: bool,
    include_outliers: bool,
) -> dict[str, Any] | None:
    limits = getattr(recipe, "spec", None)
    if limits is None:
        return None
    # One spelling: the recipe's own 'field' names the number both a reduction
    # and a spec read. A scalar recipe declares none because it produces one
    # number, so its spec falls through to that single field.
    field = getattr(recipe, "field", None)
    if field is None and len(tallies) == 1:
        field = next(iter(tallies))
    # A row with no number for the field is a run the spec did not judge, so
    # it stands beside the passes and fails rather than outside them; with no
    # field to read, no row was judged.
    tally = tallies.get(field, _FieldTally()) if field is not None else _FieldTally(no_value=rows)
    samples, no_value = tally.samples, tally.no_value
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
    if not samples or ((incomplete or no_value) and not limits.allow_incomplete):
        verdict = "indeterminate"
    else:
        verdict = "fail" if failed else "pass"
    result = {
        "field": field,
        "min": limits.min,
        "max": limits.max,
        "pass_count": pass_count,
        "fail_count": len(failed),
        "no_value_count": no_value,
        "fail_cases": _whole_page(failed),
        "verdict": verdict,
        "allow_incomplete": limits.allow_incomplete,
    }
    if include_outliers:
        result["outliers"] = list(failed)
    return result


def _group_values(
    recipe: Recipe, records: list[Record], dimensions: list[str]
) -> list[dict[str, Any]]:
    """Every group, reduced, in first-appearance order."""
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
                stepped = identity.step_values
                value = identity.assignments.get(
                    dimension, stepped.get(step_axis_key(dimension, stepped))
                )
            values.append((dimension, value))
        grouped.setdefault(tuple(values), []).append(record)
    return [
        {
            "by": dict(group),
            "reduced": _reduce(recipe, _tallies(recipe, group_records)),
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
    step_plans: dict[tuple[_CaptureKey, int], _RawStepPlan],
    steps: StepSelection,
    captures: MutableMapping[_CaptureKey, ParsedArtifacts],
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
                if isinstance(recipe, MeasurementsRecipe):
                    if (
                        run.source.raw is None
                        and (run.source.identity or {}).get("requested_plot_index") is not None
                    ):
                        raise ResultError("Cannot select a RAW plot when RAW is absent")
                    run = replace(
                        run,
                        source=await _captured_source(
                            run.source, state, captures, require_raw=False
                        ),
                    )
                    log_source = replace(
                        run.source,
                        identity={
                            **(run.source.identity or {}),
                            "dialect": run.source.dialect,
                        },
                    )
                    plan = await _measurement_step_plan(steps, run.source, state)
                    values = await metrics.measurement_rows(
                        run.source, recipe, [step for step, _ in plan], state
                    )
                    for (step, step_values), value in zip(plan, values, strict=True):
                        records.append(
                            Record(
                                manifest_id=run.manifest_id,
                                source=run.label,
                                identity=point_identity(
                                    log_source, step, step_values, plot_index=None
                                ),
                                value=_promote_headlines(recipe.metric, sanitize_payload(value)),
                            )
                        )
                    continue
                run = replace(
                    run,
                    source=await _captured_source(run.source, state, captures, require_raw=True),
                )
                planned = await _step_plan(steps, run.source, state, step_plans)
                step_plan = planned.steps
                run = replace(
                    run,
                    source=replace(
                        run.source,
                        identity={
                            **(run.source.identity or {}),
                            "dialect": planned.descriptor.dialect,
                            "dialect_evidence": list(planned.descriptor.dialect_evidence),
                        },
                    ),
                )
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
                            identity=point_identity(
                                run.source, step, step_values, plot_index=run.source.plot_index
                            ),
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
                    python_route=getattr(exc, "python_route", False),
                )
            )
    return records, failures, pending


def _rename_artifacts(pending: list[_PendingArtifact]) -> None:
    for artifact in pending:
        artifact.final.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(artifact.pending, os.O_RDWR | O_BINARY)
        try:
            fsync_fd(fd)
        finally:
            os.close(fd)
        replace_file(artifact.pending, artifact.final)
        fsync_dir(artifact.final.parent)


def _published_size(path: Path) -> int | None:
    """The size of the regular file published at ``path``, or None."""
    try:
        status = path.stat()
    except OSError:
        return None
    return status.st_size if S_ISREG(status.st_mode) else None


async def _publish(pending: list[_PendingArtifact], deadline: float) -> list[dict[str, Any]]:
    """Publish each pending artifact at its final path, and its handle.

    A final path holds only what :func:`_rename_artifacts` moved there: a
    written and flushed file, renamed into place whole, so a file at a final
    path is a complete earlier publish. Its name is a digest of the result
    set (the request and the sources' digests), the recipe and the source,
    so a repeat of the request writes to the same path. When the file there
    already holds the bytes just written (same size and SHA-256) it is kept
    as it is: never replaced or opened for writing, so a program holding it
    open (a CSV in a spreadsheet, on Windows) does not fail the repeat. When
    its bytes differ (a writer of another release, or a file changed after it
    was published) it is replaced, and a refusal to replace it fails the
    recipe as before."""
    prepared: list[dict[str, Any]] = []
    moved: list[_PendingArtifact] = []
    kept: list[_PendingArtifact] = []
    loop = asyncio.get_running_loop()

    async def digest_of(path: Path) -> str:
        return await services.bounded_parse(
            path,
            lambda: result_store.sha256_file(path),
            timeout_s=max(_MIN_ITEM_DEADLINE_S, deadline - loop.time()),
        )

    for artifact in pending:
        digest = await digest_of(artifact.pending)
        size = await asyncio.to_thread(lambda path=artifact.pending: path.stat().st_size)
        published = await asyncio.to_thread(_published_size, artifact.final)
        same = False
        if published == size:
            # An unreadable published file is replaced, as one that differs.
            with contextlib.suppress(OSError):
                same = await digest_of(artifact.final) == digest
        (kept if same else moved).append(artifact)
        prepared.append(
            {
                "path": str(artifact.final),
                "content_type": artifact.content_type,
                "sha256": digest,
                "bytes": size,
            }
        )
    await asyncio.to_thread(_rename_artifacts, moved)
    await asyncio.to_thread(_remove_pending, kept)
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
    per_run: bool,
    include_outliers: bool,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """The one complete result entry for ``recipe``: every row, group and
    failing case. What a reply holds of it is the presenter's to decide."""
    incomplete = bool(failures or missing)
    # One tally serves the reduction and the spec.
    tallies = _tallies(recipe, records)
    entry: dict[str, Any] = {
        "metric": recipe.metric,
        "reduced": _reduce(recipe, tallies),
        "warnings": _record_warnings(records),
    }
    groups = _group_values(recipe, records, group_by)
    if groups:
        entry["groups"] = groups
    spec = _spec(
        recipe,
        tallies,
        len(records),
        incomplete=incomplete,
        include_outliers=include_outliers,
    )
    if spec is not None:
        entry["spec"] = spec
    render = _row_renderer(fields, metric=recipe.metric)
    if per_run:
        entry["per_run"] = _whole_page([render(record.wire()) for record in records])
    elif not getattr(recipe, "reduce", []):
        entry["values"] = [render(record.wire()) for record in records]
    # Only report unresolved paths where rows were actually emitted: a
    # reduce-only recipe has no row surface by construction, so its fields did
    # not fail to resolve — they had nothing to apply to.
    if fields and records and ("per_run" in entry or "values" in entry):
        entry["warnings"].extend(
            _projection_warnings([record.wire() for record in records], fields)
        )
    return entry


# What a caller addresses a run by, versus the audit trail proving what it ran
# against. In one measured response the trail was a sixth of the whole receipt
# (1,174 of 7,835 characters), naming files the analysis tools already resolve
# by id.
_RUN_IDENTITY_KEYS = ("manifest_id", "label", "plot_index", "dialect", "snapshot_id")
_RUN_PROVENANCE_KEYS = (
    "explicit_dialect",
    "producing_dialect",
    "raw_present",
    "log_present",
    "console_present",
    "raw_bytes",
    "log_bytes",
    "console_bytes",
    "console_path",
    "console_sha256",
    "job_id",
    "raw_path",
    "raw_sha256",
    "log_path",
    "log_sha256",
    "composite_sha256",
    "selection_sha256",
)


def _source_hashes(
    item: result_store.ResultSet, *, provenance: bool = False
) -> list[dict[str, Any]]:
    keys = _RUN_IDENTITY_KEYS + (_RUN_PROVENANCE_KEYS if provenance else ())
    return [{key: manifest.get(key) for key in keys} for manifest in item.source_manifests]


def _page_of(item_schema: dict[str, Any], cut: Any) -> dict[str, Any]:
    """The shared offset page, with its rows narrowed to ``item_schema`` and
    cut as ``cut`` declares (``lib/reply_cuts.py``).

    Closed to extra keys: a page this tool renders carries the five page keys
    and nothing else, so anything else on it is a schema failure rather than a
    silent pass. A fresh dict per call — the surfaces that share this shape
    each declare their own row type.
    """
    return {
        **page_schema({"type": "array", "items": item_schema, CUT: cut}),
        "additionalProperties": False,
    }


_PAGE_SCHEMA: dict[str, Any] = _page_of({"type": "object"}, ITEMS)

#: spec.fail_cases: a page with no next_cursor of its own. Cut short, its hint
#: names the call that reads the rest from the stored answer.
_FAIL_CASES_SCHEMA: dict[str, Any] = {
    **_PAGE_SCHEMA,
    "properties": {
        **{
            key: value for key, value in _PAGE_SCHEMA["properties"].items() if key != "next_cursor"
        },
        "hint": HINT_SCHEMA,
    },
    "required": [key for key in _PAGE_SCHEMA["required"] if key != "next_cursor"],
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
        "plot_index": {"type": ["integer", "null"], "minimum": 0},
        "dialect": {"type": ["string", "null"]},
        "snapshot_id": {"type": ["string", "null"]},
        "case_id": {"type": ["string", "null"]},
        "run_index": {"type": ["integer", "null"]},
        "step_index": {"type": ["integer", "null"]},
        "step_values": {"type": "object"},
        "assignments": {"type": "object"},
        "circuit": {"type": ["string", "null"]},
        "deck_sha256": {"type": ["string", "null"]},
        "value": {
            "type": "object",
            "description": (
                "The recipe's value. A measurements row whose run holds one value "
                "per .meas carries 'measured' {name: value} and 'at' {name: AT point "
                "printed beside a result that is its value}. A null in 'measured' is "
                "named once by why: 'failed_measurements' (LTspice printed it failed), "
                "'unreadable' {name: {line, reason, steps}} (it printed something "
                "other than a finite number for the result, quoted in reason), "
                "'answer_unknown' {name: reason} (which field holds the result is not "
                "known) or 'not_established' {name: {steps, reason}} (what it printed "
                "is not established as the result). 'printed_once' {name: "
                "{step_count, step, reason}} names a value the log printed once for a "
                "run of several steps (a stepped .op), with the step whose value it is "
                "where that is established. Its 'stats' block is left off the default "
                "row and returned by include.fields=['value']. A stepped run's row is "
                "'stats', with 'failed_measurements' and 'unreadable' over all its "
                "steps."
            ),
            "properties": {
                "diagnostics": RUN_DIAGNOSTICS_SCHEMA,
                "diagnostics_coverage": DIAGNOSTICS_COVERAGE_SCHEMA,
                "measured": {
                    "type": "object",
                    "additionalProperties": {"type": ["number", "null"]},
                },
                "at": {
                    "type": "object",
                    "additionalProperties": {"type": ["number", "null"]},
                },
                "failed_measurements": {
                    "type": "array",
                    "items": {"type": "string"},
                    CUT: NEVER,
                    "description": (
                        "The measurements LTspice printed as failed at the step(s) "
                        "the row reads: FAIL'ed, or 'failed' in a stepped table. A "
                        "summary's measurements are the whole run's at any step, so "
                        "its list names those that failed at any step."
                    ),
                },
                "unreadable": {
                    "type": "object",
                    "additionalProperties": UNREADABLE_SCHEMA,
                },
                "printed_once": {
                    "type": "object",
                    "additionalProperties": PRINTED_ONCE_SCHEMA,
                    "description": (
                        "The measurements the log printed once for a run of several "
                        "steps, by name (a stepped .op)."
                    ),
                },
                "answer_unknown": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": (
                        "The measurements whose 'measured' is null because which "
                        "field of what the log printed holds the result is not "
                        "known, with why."
                    ),
                },
                "not_established": {
                    "type": "object",
                    "additionalProperties": NOT_ESTABLISHED_SCHEMA,
                    "description": (
                        "The measurements with a step whose printed value is not "
                        "established as the result, such as a 0 LTspice XVII prints "
                        "for a failed step: no number is given for those steps."
                    ),
                },
                # An inline waveform's samples: the axis and each signal's
                # values, thinned together when a reply is too long for them.
                "x": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                    CUT: series("waveform"),
                },
                "series": {
                    "type": "array",
                    CUT: series("waveform"),
                    "items": {
                        "type": "object",
                        "properties": {"signal": {"type": "string"}},
                        "additionalProperties": {
                            "type": "array",
                            "items": {"type": ["number", "null"]},
                            CUT: NEVER,
                        },
                    },
                },
                "points_returned": {"type": "integer", "minimum": 0},
                "points_total": {"type": "integer", "minimum": 0},
            },
        },
    },
    "additionalProperties": False,
}

#: How a list of attributed rows is cut: whole rows, each named by the source,
#: case, run and step it reads, which include.fields may project out.
_ATTRIBUTED_ROWS_CUT: dict[str, Any] = rows(
    "source", "case_id", "run_index", "step_index", projected_by="include.fields"
)

_REDUCED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "field": {"type": "string"},
        "stat": {"type": "string"},
        "value": {"type": ["number", "integer", "null"]},
        "plot_index": {"type": ["integer", "null"], "minimum": 0},
        "dialect": {"type": ["string", "null"]},
        "snapshot_id": {"type": ["string", "null"]},
        "case_id": {"type": ["string", "null"]},
        "run_index": {"type": ["integer", "null"]},
        "step_index": {"type": ["integer", "null"]},
        "step_values": {"type": "object"},
        "assignments": {"type": "object"},
        "no_value_count": {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Rows of this recipe with no number for this field (a failed "
                ".meas, a null, a missing key), left out of the statistic. "
                "Absent when every row had one."
            ),
        },
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
        "no_value_count": {
            "type": "integer",
            "minimum": 0,
            "description": (
                "Rows with no number for the field, so neither passed nor failed; "
                "pass + fail + no_value is every row. Any makes the verdict "
                "indeterminate unless allow_incomplete. Absent only on a verdict "
                "an earlier build stored with a job."
            ),
        },
        "fail_cases": _FAIL_CASES_SCHEMA,
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail", "indeterminate"],
        },
        "allow_incomplete": {"type": "boolean"},
        "outliers": {"type": "array", "items": {"type": "object"}, CUT: ITEMS},
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
        # Never cut: the request bounds it (one row per field and stat it
        # asks for), and a summary is never missing while its rows are shown.
        "reduced": {"type": "array", "items": _REDUCED_SCHEMA, CUT: NEVER},
        "groups": {"type": "array", "items": {"type": "object"}, CUT: ITEMS},
        "spec": _SPEC_SCHEMA,
        "per_run": _page_of(_ATTRIBUTED_VALUE_SCHEMA, _ATTRIBUTED_ROWS_CUT),
        "values": {"type": "array", "items": _ATTRIBUTED_VALUE_SCHEMA, CUT: _ATTRIBUTED_ROWS_CUT},
        "warnings": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
    },
    "required": ["metric", "reduced", "warnings"],
    "additionalProperties": False,
}

#: ``rows_not_shown``: by recipe, what the rows a reply left out carry.
ROWS_NOT_SHOWN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Present when rows of a recipe were left out and some carry a fact: by "
        "recipe key, counts per kind ('failed_measurements', 'unreadable', "
        "'diagnostics.<code>', ...) over the rows after those sent, and rows naming "
        "a few by source, case, run and step, each with a cursor that starts at it "
        "(pass it as continue.cursor). A kind every row carries is given by its "
        "count. '*' sums the recipes the part's bound left out "
        "(analyze.ROWS_NOT_SHOWN_BOUND_TOKENS). Never cut."
    ),
    "additionalProperties": {
        "type": "object",
        "properties": {
            "counts": {"type": "object", "additionalProperties": {"type": "integer"}},
            "groups": {"type": "integer", "minimum": 1},
            "rows": {
                "type": "array",
                CUT: NEVER,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "case_id": {"type": ["string", "null"]},
                        "run_index": {"type": ["integer", "null"]},
                        "step_index": {"type": ["integer", "null"]},
                        "kinds": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
                        "cursor": {"type": "string"},
                    },
                    "required": ["kinds"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["counts"],
        "additionalProperties": False,
    },
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
                # How many of the runs requested have not finished: a value
                # over the runs analyzed is not the job's result yet.
                "runs_running": {"type": "integer", "minimum": 1},
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
            CUT: NEVER,
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
            CUT: NEVER,
            "items": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "stage": {"type": "string"},
                    "where": {"type": "string"},
                    "message": {"type": "string"},
                    # Present on a row that stands for several failures with one
                    # cause: the distinct places, capped, and the true count.
                    "wheres": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
                    "count": {"type": "integer", "minimum": 2},
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
                CUT: ITEMS,
            },
        },
        "source_hashes": {
            "type": "array",
            CUT: ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "manifest_id": {"type": "string"},
                    "label": {"type": "string"},
                    "plot_index": {"type": ["integer", "null"], "minimum": 0},
                    "dialect": {"type": ["string", "null"]},
                    "snapshot_id": {"type": ["string", "null"]},
                    "explicit_dialect": {"type": ["string", "null"]},
                    "producing_dialect": {"type": ["string", "null"]},
                    "raw_path": {"type": ["string", "null"]},
                    "raw_sha256": {"type": ["string", "null"]},
                    "log_path": {"type": ["string", "null"]},
                    "raw_present": {"type": ["boolean", "null"]},
                    "log_present": {"type": ["boolean", "null"]},
                    "console_present": {"type": ["boolean", "null"]},
                    "raw_bytes": {"type": ["integer", "null"]},
                    "log_bytes": {"type": ["integer", "null"]},
                    "console_bytes": {"type": ["integer", "null"]},
                    "console_path": {"type": ["string", "null"]},
                    "console_sha256": {"type": ["string", "null"]},
                    "log_sha256": {"type": ["string", "null"]},
                    "composite_sha256": {"type": ["string", "null"]},
                    "selection_sha256": {"type": ["string", "null"]},
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
        "rows_not_shown": ROWS_NOT_SHOWN_SCHEMA,
        "cut": CUT_SCHEMA,
        "stored_answer": STORED_ANSWER_SCHEMA,
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
class AnalysisEvaluation:
    """Everything the complete answer is built from, once every recipe ran.

    One evaluation runs every recipe of a request; nothing about it knows a
    page, a cursor or a reply size. What a reply holds of the answer is the
    reply layer's to decide (``tools/_base.presented``).
    """

    item: result_store.ResultSet
    processed: list[WorkUnit]
    runs: list[_ResolvedRun]
    #: Runs a source named but could not deliver, already in wire shape: they
    #: are stored verbatim in the result set and go straight into
    #: ``coverage.missing_cases``.
    missing: list[dict[str, Any]]
    #: Rejections, each with the work position it was rejected at.
    skipped: list[tuple[int, Failure]]
    base_observations: list[Observation]
    include: AnalyzeInclude
    group_by: list[str]
    signals: dict[str, list[str]] | None
    #: What this call's arguments asked for that was read differently (a
    #: repeat read once, a page size held to its cap), each said in the hint.
    argument_notes: tuple[str, ...] = ()
    #: The request's identity: a page read with ``include.per_run.cursor``
    #: must name the same request.
    request: str = ""

    @property
    def failure_inventory(self) -> tuple[Failure, ...]:
        """Every failure of the evaluation, one per place."""
        failures = [failure for _position, failure in self.skipped]
        failures.extend(failure for unit in self.processed for failure in unit.failures)
        return tuple(failures)

    def snapshot(self) -> dict[str, Any]:
        """What the answer was computed from: the result set, the request,
        and a digest of every source's manifest id and the parsed snapshot it
        read, whose size does not grow with the sources. A page of a stored
        answer carries it, so a caller can tell whether a source has changed
        since: ``source_hashes`` with ``include.provenance`` names each."""
        return {
            "result_set_id": self.item.result_set_id,
            "request": self.request,
            "per_run_limit": self.include.per_run.limit if self.include.per_run else None,
            "sources": result_store.canonical_hash(
                [
                    [str(manifest.get("manifest_id")), manifest.get("snapshot_id")]
                    for manifest in self.item.source_manifests
                ]
            )[:32],
        }


def _assemble(a: AnalysisEvaluation) -> dict[str, Any]:
    """The complete answer of one evaluation: every recipe's entry with every
    row, every missing case and every failure, one per place, each marked
    under ``_ROUTE_KEY`` when its message ends in an ``api`` snippet. Each
    door gives it as it delivers (:func:`for_session`, :func:`for_api`)."""
    item = a.item
    observations: list[Observation] = []
    for observation in (
        *a.base_observations,
        *(o for unit in a.processed for o in unit.observations),
    ):
        # A fact about a run, such as its compression, comes from each unit
        # that read it; it is said once.
        if observation not in observations:
            observations.append(observation)
    results: dict[str, Any] = {}
    # Where each entry's recipe sits in the work list, so the entries keep the
    # order the recipes were asked in, failed ones included.
    entry_at: dict[str, int] = {}
    # The entries that answered: rows, or no failure to explain their absence.
    delivered: list[str] = []
    analyzed_identities: set[tuple[Any, Any, Any]] = set()
    per_run = a.include.per_run is not None

    for at, failure in a.skipped:
        # A recipe refused before it ran keeps an entry under the key it
        # asked for, so reading results[key] says what happened instead of
        # raising: one entry per recipe, whatever the reason it has no rows.
        work_item = item.work[at]
        raw_recipe = work_item["recipe"]
        key = str(raw_recipe.get("key") or f"recipe_{work_item['index']}")
        if key in results:
            continue
        empty_rows: dict[str, Any] = {"per_run": _whole_page([])} if per_run else {"values": []}
        results[key] = {
            "metric": str(raw_recipe.get("metric") or ""),
            "reduced": [],
            "warnings": [_no_rows_warning([failure])],
            **empty_rows,
        }
        entry_at[key] = at

    for unit in a.processed:
        relevant_missing = [
            case
            for case in a.missing
            if unit.recipe.sources is None or case.get("label") in set(unit.recipe.sources)
        ]
        entry = _result_entry(
            unit.recipe,
            unit.records,
            unit.failures,
            a.group_by,
            relevant_missing,
            per_run,
            a.include.outliers,
            a.include.fields,
        )
        if unit.records or not unit.failures:
            delivered.append(unit.key)
        else:
            entry["warnings"].append(_no_rows_warning(unit.failures))
        results[unit.key] = entry
        entry_at[unit.key] = unit.position
        for record in unit.records:
            analyzed_identities.add(
                (record.source, record.identity.case_id, record.identity.run_index)
            )

    results = {key: results[key] for key in sorted(results, key=entry_at.__getitem__)}
    failures = a.failure_inventory
    coverage: dict[str, Any] = {
        "runs_requested": len(a.runs) + len(a.missing),
        "runs_analyzed": len(analyzed_identities),
        "missing_cases": _whole_page(list(a.missing)),
    }
    # Counted over every missing row: the one statement that a value over the
    # runs read is not yet the result of the runs asked.
    running = sum(row.get("code") == "run_not_finished" for row in a.missing)
    if running:
        coverage["runs_running"] = running
    data: dict[str, Any] = {
        "outcome": outcome_of(list(failures), partial=bool(a.missing), delivered=bool(delivered)),
        "coverage": coverage,
        "results": results,
        "observations": [observation.wire() for observation in observations],
        "failures": [_stored_failure(failure) for failure in failures],
        "source_hashes": _source_hashes(item, provenance=a.include.provenance),
        "result_set_id": item.result_set_id,
        "cursor": None,
        "next": None,
    }
    if a.signals is not None:
        data["signals_available"] = a.signals
    hints: list[str] = []
    if any(case.get("code") == PathSecurityError.code for case in a.missing):
        hints.append(
            "A source path was refused; its coverage.missing_cases row's "
            "hint says why and what to change."
        )
    hints.extend(a.argument_notes)
    if hints:
        data["hint"] = " ".join(hints)
    return data


def _stored_failure(failure: Failure) -> dict[str, Any]:
    """One failure of the complete answer: its plain message, marked when it
    ends in an ``api`` snippet, which each door words for itself."""
    row = failure.wire()
    if failure.python_route:
        row[_ROUTE_KEY] = True
    return row


def _failure_of(row: Mapping[str, Any]) -> Failure:
    return Failure(
        code=str(row["code"]),
        stage=str(row["stage"]),
        where=str(row["where"]),
        message=str(row["message"]),
        python_route=bool(row.get(_ROUTE_KEY)),
    )


def for_api(answer: Mapping[str, Any]) -> dict[str, Any]:
    """The complete answer as the Python API returns it: every failure, one
    per place, each message as written."""
    data = dict(answer)
    data["failures"] = [
        {key: value for key, value in row.items() if key != _ROUTE_KEY}
        for row in answer.get("failures") or ()
    ]
    return data


def for_session(answer: Mapping[str, Any], served: Collection[str]) -> dict[str, Any]:
    """The complete answer as an MCP session receives it: its failures
    collapsed to one row per reason and held to ``_FAILURE_CAP`` rows, with
    the note that names the rest, and each ``api`` snippet's message saying
    where it runs (``served``, the session's tools)."""
    data = dict(answer)
    failures = [_failure_of(row) for row in answer.get("failures") or ()]
    rows = _collapse_failures(failures, served)
    if len(rows) > _FAILURE_CAP:
        counts = {"rows": len(rows), "records": len(failures)}
        data["observations"] = [
            *answer.get("observations", []),
            Observation(
                code="failures_truncated",
                kind="coverage",
                detail=_failures_truncated_detail(counts, served),
            ).wire(),
        ]
        rows = rows[:_FAILURE_CAP]
    data["failures"] = rows
    data["observations"] = _bounded_evidence(data)
    return data


def _bounded_evidence(answer: Mapping[str, Any]) -> list[Any]:
    """The answer's observations, each lifted from a recipe's rows
    (:func:`_value_observations`) naming at most :data:`not_shown.NAMED` of
    them, with ``row_count`` the number it was made of; one every row of its
    recipe carries names none. ``observations`` is never cut, so this is its
    bound on an MCP reply; the Python API keeps every row."""
    counts = {_recipe_of(pointer): len(rows) for pointer, rows in _row_lists(answer).items()}
    bounded: list[Any] = []
    for observation in answer.get("observations") or ():
        evidence = observation.get("evidence") if isinstance(observation, Mapping) else None
        if not isinstance(evidence, Mapping):
            bounded.append(observation)
            continue
        rows, recipe = evidence.get("rows"), evidence.get("recipe")
        if not isinstance(rows, list) or not isinstance(recipe, str):
            bounded.append(observation)
            continue
        kept: dict[str, Any] = dict(evidence)
        kept["row_count"] = len(rows)
        if rows and len(rows) >= counts.get(recipe, len(rows) + 1):
            kept.pop("rows")
            kept["every_row"] = True
        else:
            kept["rows"] = rows[: not_shown.NAMED]
        bounded.append({**observation, "evidence": kept})
    return bounded


@dataclass(frozen=True)
class _StoredSamples:
    """One row's inline samples, read into arrays once for every limit a fit tries."""

    value: dict[str, Any]
    x: np.ndarray
    #: Each series component that holds samples: series index, name, samples.
    parts: tuple[tuple[int, str, np.ndarray], ...]

    @classmethod
    def of(cls, value: dict[str, Any]) -> _StoredSamples:
        return cls(
            value,
            np.asarray(value["x"], dtype=float),
            tuple(
                (index, name, np.asarray(item, dtype=float))
                for index, entry in enumerate(value["series"])
                for name, item in entry.items()
                if isinstance(item, list)
            ),
        )

    def at(self, limit: int) -> dict[str, Any]:
        """The value decimated again, to at most ``limit`` points.

        Bucketing min/max pairs by their minimum and maximum keeps every extreme
        the first decimation kept. The samples kept are ones the value already
        held, so their written form does not change.
        """
        if len(self.x) <= limit or not self.parts:
            return self.value
        kept, samples = downsample_minmax_traces(
            self.x, [part for _index, _name, part in self.parts], limit
        )
        series = [dict(entry) for entry in self.value["series"]]
        for (index, name, _part), thinned in zip(self.parts, samples, strict=True):
            series[index][name] = thinned.tolist()
        return {**self.value, "x": kept.tolist(), "series": series, "points_returned": len(kept)}


# ---------------------------------------------------------------------------
# What a reply says of the rows it left out
# ---------------------------------------------------------------------------


def row_kinds(row: Mapping[str, Any]) -> list[str]:
    """The kinds of fact a row's value states: each diagnostic's code, what
    of its log was not read, and each field naming a measurement that gave no
    value."""
    value = row.get("value")
    if not isinstance(value, Mapping):
        return []
    kinds = [
        f"diagnostics.{entry.get('code')}"
        for entry in value.get("diagnostics") or ()
        if isinstance(entry, Mapping)
    ]
    if value.get("diagnostics_coverage"):
        kinds.append(f"diagnostics_coverage.{value['diagnostics_coverage']}")
    kinds.extend(key for key in _FACT_FIELDS[:6] if value.get(key))
    return kinds


def _row_identity(_index: int, row: Mapping[str, Any]) -> dict[str, Any]:
    """What names a row left out: the source, case, run and step it reads."""
    return {
        key: row.get(key)
        for key in ("source", "case_id", "run_index", "step_index")
        if row.get(key) is not None
    }


def _kind_vocabulary() -> list[str]:
    """Every kind :func:`row_kinds` can give a row."""
    from ltspice_mcp.lib import log_conditions

    return [
        *(f"diagnostics.{code}" for code in sorted(log_conditions.diagnostic_codes())),
        *(f"diagnostics_coverage.{value}" for value in log_conditions.COVERAGE),
        *_FACT_FIELDS[:6],
    ]


#: The most ``rows_not_shown`` takes, in estimated tokens: every kind counted
#: in one recipe and in the recipes summed after it, and three rows named by
#: identities and a cursor of ordinary length (``lib/not_shown.py``).
ROWS_NOT_SHOWN_BOUND_TOKENS: int = not_shown.worst_bound(
    _kind_vocabulary(),
    identity={"source": "s" * 32, "case_id": "c" * 48, "run_index": 999_999, "step_index": 9999},
    cursor="c" * 160,
    field="rows_not_shown",
    entries="rows",
    group="k" * 32,
    folded=True,
)

#: What a longer case id or count of a later page may add to a reserve.
_RESERVE_MARGIN = 16


def _row_lists(answer: Mapping[str, Any]) -> dict[str, list[Any]]:
    """Each recipe's rows, by the pointer of the list that holds them."""
    found: dict[str, list[Any]] = {}
    results = answer.get("results")
    for key, entry in results.items() if isinstance(results, Mapping) else ():
        if not isinstance(entry, Mapping):
            continue
        per_run = entry.get("per_run")
        if isinstance(per_run, Mapping) and isinstance(per_run.get("items"), list):
            found[join_pointer(("results", key, "per_run", "items"))] = per_run["items"]
        elif isinstance(entry.get("values"), list):
            found[join_pointer(("results", key, "values"))] = entry["values"]
    return found


def _recipe_of(pointer: str) -> str:
    """The recipe key a ``/results/<key>/...`` pointer names."""
    return split_pointer(pointer)[1]


# ---------------------------------------------------------------------------
# How a reply of this tool is presented
# ---------------------------------------------------------------------------

TOOL = "analyze_results"

#: The lists of a waveform's value the presenter thins together.
_SERIES_KEYS = ("x", "series")

#: How each list this tool cuts is said on the page object that holds it.
_PAGED = ("/per_run/items", "/missing_cases/items", "/fail_cases/items")

#: The call that reads the rest of a cut reply: a cut entry's next, with the
#: reply's result_set_id.
_READ_ON = call_to_continue(
    "analyze_results",
    {
        "continue": {
            "result_set_id": Unfilled("result_set_id"),
            "cursor": Unfilled("a cut entry's next"),
        }
    },
)

_THINNED_WARNING = (
    "Inline series thinned to {shown} of {total} points per signal, keeping each "
    "bucket's minimum and maximum, to fit the reply. analyze_results with "
    "budget {whole} sends every point, and format 'csv' writes every sample to a file."
)


def _say_how_to_read_fail_cases(entry: ListCut) -> str | None:
    """What a cut ``spec.fail_cases``, which has no next_cursor of its own,
    says beside its flag: the call that reads the rest."""
    if entry.next is None:
        return None
    left = entry.total - entry.offset - entry.returned
    return f"{left} more failing case(s); read them with {_READ_ON}."


def answer_caps(
    answer: Mapping[str, Any], snapshot: Mapping[str, Any], *, target: int | None
) -> dict[str, int]:
    """The most entries an MCP reply sends of each list of ``answer``: a
    recipe's ``per_run`` page at the request's ``include.per_run.limit``
    (``snapshot``), which is part of the question, and, with no target, each
    other list of rows at :data:`MAX_PAGE_SIZE`, so a reply with no target
    still has a bound."""
    caps: dict[str, int] = {}
    limit = snapshot.get("per_run_limit")
    results = answer.get("results")
    for key, entry in results.items() if isinstance(results, Mapping) else ():
        if not isinstance(entry, Mapping):
            continue
        if isinstance(limit, int) and isinstance(entry.get("per_run"), Mapping):
            caps[join_pointer(("results", key, "per_run", "items"))] = limit
        if target is None:
            caps[join_pointer(("results", key, "values"))] = MAX_PAGE_SIZE
            caps[join_pointer(("results", key, "spec", "fail_cases", "items"))] = (
                _FAIL_CASE_PAGE_CAP
            )
            caps.setdefault(
                join_pointer(("results", key, "per_run", "items")), ROWS_WITHOUT_TARGET
            )
    if target is None:
        caps["/coverage/missing_cases/items"] = MAX_PAGE_SIZE
    return caps


def _row_facts(whole: Mapping[str, Any]) -> dict[str, not_shown.Facts]:
    return {
        pointer: not_shown.Facts.of(rows, row_kinds, _row_identity)
        for pointer, rows in _row_lists(whole).items()
    }


def say_what_was_cut(
    whole: Mapping[str, Any],
) -> tuple[Annotate, dict[str, int]]:
    """The words a cut of ``whole``, a complete answer of this tool, adds,
    and the bound ``rows_not_shown`` reserves of the target.

    Each page object (a recipe's ``per_run``, ``coverage.missing_cases``,
    ``spec.fail_cases``) says what the cut holds of it; the reply's ``next``
    and ``cursor`` name the first cut list's cursor; ``rows_not_shown``
    names, by recipe, the rows left out that carry a fact, each with a
    cursor that starts at it (``lib/not_shown.py``); a thinned series says
    so on its recipe's warnings."""
    facts = _row_facts(whole)

    @functools.cache
    def whole_size() -> int:
        # The target at which the presenter sends this answer whole.
        return estimate_tokens(dict(whole))

    def part(cut: Mapping[str, ListCut], stored: StoredAnswer | None) -> dict[str, Any]:
        groups: dict[str, not_shown.LeftOut] = {}
        for pointer, table in facts.items():
            entry = cut.get(pointer)
            if entry is None or entry.offset + entry.returned >= entry.total:
                continue
            groups[_recipe_of(pointer)] = not_shown.LeftOut(
                table,
                entry.offset + entry.returned,
                (lambda index, at=pointer: stored_cursor(stored, at, index))
                if stored is not None
                else (lambda _index: None),
            )
        return not_shown.not_shown(
            groups, bound=ROWS_NOT_SHOWN_BOUND_TOKENS, field="rows_not_shown", entries="rows"
        )

    def annotate(
        reply: dict[str, Any],
        cut: Mapping[str, ListCut],
        stored: StoredAnswer | None,
        paged: str | None,
    ) -> None:
        # The reply's next: on a page, the cursor that reads on in the list it
        # reads, absent on its last page, so following it ends; on a first
        # reply, the first cut list's.
        read_on: str | None = None
        if paged is not None:
            entry = cut.get(paged)
            read_on = entry.next if entry is not None else None
        else:
            read_on = next((entry.next for entry in cut.values() if entry.next), None)
        for pointer, entry in cut.items():
            keys = split_pointer(pointer)
            holder = join_pointer(keys[:-1])
            if pointer.endswith(_PAGED):
                page = resolve_pointer(reply, holder, default=None)
                if isinstance(page, dict):
                    page = _owned(reply, holder)
                    page["returned"] = entry.returned
                    page["truncated"] = entry.next is not None
                    if "next_cursor" in page or pointer.endswith(_PAGED[:2]):
                        page["next_cursor"] = entry.next
                    if pointer.endswith("/fail_cases/items"):
                        said = _say_how_to_read_fail_cases(entry)
                        if said:
                            page["hint"] = said
                        else:
                            page.pop("hint", None)
            elif keys[-2:-1] == ["value"] and keys[-1] in _SERIES_KEYS:
                # A series thinned: said once on its recipe.
                recipe = _owned(reply, join_pointer(keys[:2]))
                said = _THINNED_WARNING.format(
                    shown=entry.returned, total=entry.total, whole=whole_size()
                )
                warnings = list(recipe.get("warnings", []))
                if not any(text.startswith("Inline series thinned") for text in warnings):
                    recipe["warnings"] = [*warnings, said]
        left_out = part(cut, stored)
        if left_out:
            reply["rows_not_shown"] = left_out
        if stored is not None:
            _point_evidence_at_rows(reply, cut, stored)
        if read_on is not None:
            reply["cursor"] = read_on
            reply["next"] = {"result_set_id": reply["result_set_id"], "cursor": read_on}
            sentence = f"This reply is cut: read the rest with {_READ_ON}."
            hint = reply.get("hint")
            reply["hint"] = f"{sentence} {hint}" if hint else sentence

    reserve = 0
    first = part({p: ListCut(len(rows), 1) for p, rows in _row_lists(whole).items()}, None)
    if first:
        reserve = min(
            estimate_tokens({"rows_not_shown": first}) + _RESERVE_MARGIN,
            ROWS_NOT_SHOWN_BOUND_TOKENS,
        )
    return annotate, ({"rows_not_shown": reserve} if reserve else {})


def _point_evidence_at_rows(
    reply: dict[str, Any], cut: Mapping[str, ListCut], stored: StoredAnswer
) -> None:
    """Give each row an observation names (``evidence.rows``) the cursor that
    starts at it, when its recipe's rows are cut."""
    lists = {_recipe_of(pointer): pointer for pointer in cut if pointer in _row_pointers(reply)}
    if not lists:
        return
    observations = []
    for observation in reply.get("observations") or ():
        evidence = observation.get("evidence") if isinstance(observation, Mapping) else None
        if not isinstance(evidence, Mapping):
            observations.append(observation)
            continue
        pointer = lists.get(str(evidence.get("recipe")))
        rows = evidence.get("rows")
        if pointer is None or not isinstance(rows, list):
            observations.append(observation)
            continue
        named = [
            {**row, "cursor": stored_cursor(stored, pointer, row["index"])}
            if isinstance(row, Mapping) and isinstance(row.get("index"), int)
            else row
            for row in rows
        ]
        observations.append({**observation, "evidence": {**evidence, "rows": named}})
    reply["observations"] = observations


def _row_pointers(reply: Mapping[str, Any]) -> set[str]:
    """The pointers of the lists of rows a reply of this tool may hold."""
    found: set[str] = set()
    results = reply.get("results")
    for key, entry in results.items() if isinstance(results, Mapping) else ():
        if isinstance(entry, Mapping):
            found.add(join_pointer(("results", key, "per_run", "items")))
            found.add(join_pointer(("results", key, "values")))
    return found


def _owned(reply: dict[str, Any], pointer: str) -> dict[str, Any]:
    """The object at ``pointer`` in ``reply``, copied with every object on
    the way to it, so writing it leaves the answer it was cut from as it is."""
    node = reply
    for key in split_pointer(pointer):
        node[key] = dict(node[key])
        node = node[key]
    return node


def redraw_series(holder: dict[str, Any], points: int) -> dict[str, Any]:
    """One inline waveform value at no more than ``points`` points, its axis
    and every series thinned together, each bucket's minimum and maximum
    kept (``Series``)."""
    if not isinstance(holder.get("x"), list) or not isinstance(holder.get("series"), list):
        return holder
    return _StoredSamples.of(holder).at(points)


#: The fewest points a reply thins an inline series to: sixteen buckets, each
#: its minimum and maximum, still draw the shape of a transient.
THIN_FLOOR_POINTS = 32

#: How the presenter thins this tool's waveforms.
SERIES = Series(redraw_series, THIN_FLOOR_POINTS)


def analysis_answer(
    answer: Mapping[str, Any],
    snapshot: dict[str, Any],
    *,
    root: str = "",
    target: int | None,
    whole: bool = False,
) -> Answer:
    """``answer``, a complete answer of this tool as a session receives it,
    as one answer the presenter presents at ``root``: its caps (none for the
    Python API's ``whole``), its words and its reserve."""
    annotate, apart = say_what_was_cut(answer)
    return Answer(
        root,
        OUTPUT_SCHEMA,
        TOOL,
        snapshot,
        annotate=annotate,
        caps=None if whole else answer_caps(answer, snapshot, target=target),
        apart=apart,
    )


@dataclass(frozen=True)
class StoredAnalysis:
    """An attached analysis as a job keeps it: the complete answer with its
    rows unprojected, what it was computed from, and the request it answers
    without its row view, which each delivery chooses."""

    answer: dict[str, Any]
    snapshot: dict[str, Any]
    request: dict[str, Any]


def stored_analysis(stored: Any) -> StoredAnalysis:
    """The attached analysis a job's record holds, or ``ResultError`` for one
    an earlier release stored in another shape."""
    if analysis_snapshot.classify(stored) != "snapshot" or not isinstance(
        stored.get("answer"), dict
    ):
        raise ResultError(
            "The attached analysis was stored by an earlier release in a shape this "
            "build cannot render — re-run analyze_results against the job."
        )
    return StoredAnalysis(
        stored["answer"], dict(stored.get("snapshot") or {}), dict(stored.get("request") or {})
    )


def attached_snapshot(stored: Any, fields: list[str] | None) -> dict[str, Any]:
    """What an attached analysis delivered in the row view ``fields`` was
    computed from: its snapshot, naming the request an ``analyze_results``
    call over the job with that view sends."""
    held = stored_analysis(stored)
    request = copy.deepcopy(held.request)
    include = request.get("include")
    if isinstance(include, dict):
        include["fields"] = fields
    return {**held.snapshot, "request": result_store.canonical_hash(request)[:32]}


def render_attached_analysis(
    stored: Any,
    *,
    fields: list[str] | None = None,
    served: Collection[str] | None = None,
) -> dict[str, Any]:
    """The attached analysis a job's record holds, as a door gives it: its
    rows in the row view ``fields`` (None: the lean rows), to an MCP session
    (``served``, its tools) or, with None, to the Python API. It is the
    complete answer a later ``analyze_results`` call with the same request
    over the finished job gives."""
    held = stored_analysis(stored)
    answer = copy.deepcopy(held.answer)
    for entry in (answer.get("results") or {}).values():
        if not isinstance(entry, dict):
            continue
        render = _row_renderer(fields, metric=str(entry.get("metric") or ""))
        per_run = entry.get("per_run")
        rows = per_run.get("items") if isinstance(per_run, dict) else entry.get("values")
        if not isinstance(rows, list):
            continue
        shown = [render(row) for row in rows]
        if isinstance(per_run, dict):
            per_run["items"] = shown
        else:
            entry["values"] = shown
        if fields and rows:
            entry["warnings"] = [*entry.get("warnings", []), *_projection_warnings(rows, fields)]
    return for_api(answer) if served is None else for_session(answer, served)


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
    step_plans: dict[tuple[_CaptureKey, int], _RawStepPlan],
    steps: StepSelection,
    captures: MutableMapping[_CaptureKey, ParsedArtifacts],
    *,
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

    if isinstance(recipe, MeasurementsRecipe):
        await _capture_ltspice_raws(eligible_runs, state, captures, item_deadline)
    records, item_failures, pending = await _evaluate_item(
        recipe,
        eligible_runs,
        manifests,
        state,
        item,
        item_deadline,
        step_plans,
        steps,
        captures,
    )
    failures.extend(item_failures)
    observations.extend(_absence_observations(recipe, key, records))
    observations.extend(_value_observations(key, records))
    observations.extend(_log_precision_observations(records))
    observations.extend(
        await asyncio.to_thread(
            functools.partial(
                _coarse_waveform_observations,
                samples=not isinstance(recipe, MeasurementsRecipe),
            ),
            eligible_runs,
            captures,
        )
    )
    return WorkUnit(
        key=key,
        recipe=recipe,
        records=records,
        failures=failures,
        pending=pending,
        eligible_ids={run.manifest_id for run in eligible_runs},
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
        # An observation names its rows by their position in the unit's rows
        # (_value_observations), so the positions follow the rows kept.
        positions = {
            old: new
            for new, old in enumerate(
                index
                for index, record in enumerate(unit.records)
                if record.manifest_id not in failed_ids
            )
        }
        for observation in unit.observations:
            evidence = observation.evidence
            if isinstance(evidence, dict) and isinstance(evidence.get("rows"), list):
                evidence["rows"] = [
                    {**row, "index": positions[row["index"]]}
                    for row in evidence["rows"]
                    if isinstance(row, dict) and row.get("index") in positions
                ]
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


def _request_digest(args: AnalyzeResultsInput) -> str:
    """The request a stored answer answers, by digest: its sources, recipes,
    grouping, step selection and include options, row view included
    (:func:`attached_snapshot` names an attached analysis' the same way)."""
    return result_store.canonical_hash(_request_inputs(args))[:32]


async def evaluate_analysis(args: AnalyzeResultsInput, state: SessionState) -> AnalysisEvaluation:
    """Evaluate every recipe of a new request over its sources.

    ``[analysis] analysis_budget_s`` bounds the whole call: reading the
    sources, the check before the reads, every recipe, the check after them
    and the signal listing. A step gets what is left of the bound (at least
    ``_MIN_ITEM_DEADLINE_S``, so a check of what was read always runs); a
    recipe that runs out of it fails with ``analysis_deadline``, and each
    recipe the bound leaves no time for fails alone with the same code,
    saying to ask for it in a call of its own. Nothing here knows a page, a
    cursor or a reply size.
    """
    loop = asyncio.get_running_loop()
    budget_s = state.config.analysis_budget_s
    call_deadline = loop.time() + budget_s

    def step_deadline() -> float:
        return loop.time() + max(_MIN_ITEM_DEADLINE_S, call_deadline - loop.time())

    # RAW recipes share one resolved step plan per source per call.
    step_plans: dict[tuple[_CaptureKey, int], _RawStepPlan] = {}
    # Share only while the bounded result cache or an active reader owns it.
    captures: WeakValueDictionary[_CaptureKey, ParsedArtifacts] = WeakValueDictionary()

    item = await _create_result_set(args, state, call_deadline)
    include = AnalyzeInclude.model_validate(item.inputs.get("include", {}))
    runs = _deserialize_runs(item, state)
    manifests = {str(manifest["manifest_id"]): manifest for manifest in item.source_manifests}
    missing = list(item.inputs.get("missing", []))
    skipped: list[tuple[int, Failure]] = []
    observations = [Observation.of(data) for data in item.inputs.get("observations", [])]
    group_by = list(item.inputs.get("group_by", []))
    steps = StepSelection.from_inputs(item.inputs)
    declared_labels = {
        str(source.get("label"))
        for source in item.inputs.get("sources", [])
        if isinstance(source, dict)
    }

    # Source-drift precheck: verify every direct source before any recipe
    # reads it. Recapture validates the stored parser identity, including
    # console bytes and explicitly absent companions.
    precheck = await _verify_direct_sources(
        item.source_manifests,
        {run.manifest_id for run in runs},
        step_deadline(),
        state=state,
    )

    processed: list[WorkUnit] = []
    all_pending: list[_PendingArtifact] = []
    evaluated_ids: set[str] = set()

    for position, work_item in enumerate(item.work):
        raw_recipe = work_item["recipe"]
        key = str(raw_recipe.get("key") or f"recipe_{work_item['index']}")
        try:
            recipe = recipe_of_request(raw_recipe, declared_labels)
        except (ValueError, TypeError) as exc:
            skipped.append(
                (
                    position,
                    Failure(
                        code="recipe_invalid",
                        stage="validate",
                        where=f"recipes[{work_item['index']}]",
                        message=validation_error_detail("analyze_results", exc),
                    ),
                )
            )
            continue
        selected_runs = [
            run for run in runs if recipe.sources is None or run.label in set(recipe.sources)
        ]
        if _artifact_estimate(recipe, selected_runs) > budget_s * _ARTIFACT_SAFETY_FACTOR:
            skipped.append(
                (
                    position,
                    Failure(
                        code="artifact_too_large",
                        stage="preflight",
                        where=key,
                        message=(
                            "Artifact estimate exceeds the per-recipe safety bound. The "
                            "bound uses raw file size and signal count rather than "
                            "decoded sample counts, so request fewer signals or "
                            "select fewer/smaller runs; narrowing the window or "
                            "lowering max_points does not move it. A single large "
                            "raw read for a single signal has no request-side lever "
                            "— raise [analysis] analysis_budget_s "
                            "(LTSPICE_MCP_ANALYSIS_BUDGET_S) instead."
                        ),
                    ),
                )
            )
            continue
        if loop.time() >= call_deadline:
            skipped.append((position, _left_by_the_bound(key, budget_s)))
            continue
        unit = await _evaluate_unit(
            recipe,
            key,
            selected_runs,
            runs,
            precheck,
            manifests,
            state,
            item,
            step_deadline(),
            step_plans,
            steps,
            captures,
            position=position,
        )
        evaluated_ids.update(unit.eligible_ids)
        all_pending.extend(unit.pending)
        processed.append(unit)

    # Source-drift postcheck: verify every evaluated source before any
    # artifact is published or results are returned. Drift discards the
    # records and artifacts derived from that source across every recipe
    # that used it.
    postcheck_deadline = step_deadline()
    postcheck = await _verify_direct_sources(
        item.source_manifests,
        evaluated_ids,
        postcheck_deadline,
        state=state,
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
        signals_deadline = call_deadline
        signals = {run.manifest_id: [] for run in runs}
        for run in runs:
            if loop.time() >= signals_deadline:
                break
            if (
                not manifests[run.manifest_id].get("raw_present")
                or run.manifest_id in precheck
                or run.manifest_id in postcheck
            ):
                continue
            try:
                with services.analysis_deadline(signals_deadline):
                    source = await _captured_source(run.source, state, captures, require_raw=True)
                    raw = await services.load_raw(source, state)
                signals[run.manifest_id] = list(raw.get_trace_names())
            except AnalysisDeadlineExceeded:
                break
            except LTSpiceMCPError:
                continue

    return AnalysisEvaluation(
        item=item,
        processed=processed,
        runs=runs,
        missing=missing,
        skipped=skipped,
        base_observations=observations,
        include=include,
        group_by=group_by,
        signals=signals,
        argument_notes=tuple(args.argument_notes()),
        request=_request_digest(args),
    )


def _left_by_the_bound(key: str, budget_s: float) -> Failure:
    """The failure of a recipe the call's bound left no time for."""
    return Failure(
        code="analysis_deadline",
        stage="deadline",
        where=key,
        message=(
            f"The call's work bound ([analysis] analysis_budget_s, {budget_s:g} s) ended "
            "before this recipe was evaluated. Ask for it in a call of its own; the "
            "recipes before it are answered."
        ),
    )


def recipe_of_request(raw_recipe: Any, labels: Collection[str]) -> Recipe:
    """``raw_recipe`` as a recipe of a request whose sources carry ``labels``.

    The one answer to whether a recipe can run in a request: its own fields
    validate, and every label its ``sources`` names is a source's. Both
    analyze_results, which fails that one recipe, and run_experiments, which
    refuses an attached block at submission, ask here, so a block accepted at
    submission is one the analysis after the runs can read. Raises
    ``ValueError`` (a ``ValidationError`` for the recipe's own fields) or
    ``TypeError``.
    """
    recipe = validate_recipe(raw_recipe)
    unknown = set(recipe.sources or ()) - set(labels)
    if unknown:
        raise ValueError(
            "recipe sources name unknown labels: "
            + ", ".join(sorted(unknown))
            + ". This request's sources are labelled "
            + ", ".join(sorted(labels))
            + "; leave sources out to read every source."
        )
    return recipe


async def complete_analysis(args: AnalyzeResultsInput, state: SessionState) -> dict[str, Any]:
    """The complete answer the Python API returns: every recipe, every row
    and every failure, one per place."""
    evaluation = await evaluate_analysis(args, state)
    return for_api(await asyncio.to_thread(_assemble, evaluation))


async def capture_attached_analysis(
    args: AnalyzeResultsInput, state: SessionState
) -> dict[str, Any]:
    """Evaluate an analysis attached to a run, for the job's record: its
    complete answer, the one a later ``analyze_results`` call with the same
    request over the job gives, and what it was computed from."""
    evaluation = await evaluate_analysis(args, state)
    # Its rows unprojected: each delivery renders them in the row view it
    # asks for (render_attached_analysis), which is not part of the job.
    wide = replace(
        evaluation, include=evaluation.include.model_copy(update={"fields": list(_ROW_KEYS)})
    )
    answer = await asyncio.to_thread(_assemble, wide)
    request = _request_inputs(args)
    return analysis_snapshot.envelope(
        {"answer": answer, "snapshot": evaluation.snapshot(), "request": request}
    )


#: What a cursor whose stored answer is gone adds: how to get the answer again.
_GONE = "Send the request again without the cursor to evaluate it again."


async def _read_on(args: AnalyzeResultsInput, state: SessionState) -> types.CallToolResult:
    """A page of a stored answer: the list a cursor names, from its entry on,
    sliced from the stored answer. Nothing is evaluated and no source is
    read, so a page is never refused or emptied because a source moved: it
    carries the snapshot of the sources its answer was computed from."""
    by_page = args.continuation is None
    if args.continuation is not None:
        cursor = args.continuation.cursor
    else:
        assert args.include.per_run is not None and args.include.per_run.cursor is not None
        cursor = args.include.per_run.cursor
    stored, pointer, offset = await open_stored_page(
        cursor, tool=TOOL, schema=OUTPUT_SCHEMA, state=state, gone=_GONE
    )
    if args.continuation is not None and (
        args.continuation.result_set_id != stored.answer.get("result_set_id")
    ):
        raise InvalidCursorError(
            "This cursor belongs to another analysis than result_set_id "
            f"{args.continuation.result_set_id!r}; pass the result_set_id of the reply "
            "that gave the cursor."
        )
    request = stored.snapshot.get("request")
    if by_page or _REQUEST_ECHO & args.model_fields_set:
        if not (args.sources and args.recipes):
            raise ResultError(
                "'continue' reads a page of the stored answer: send the whole request "
                "with it (sources and recipes), or none of it."
            )
        if request != _request_digest(args.model_copy(update={"continuation": None})):
            raise InvalidCursorError(
                "The cursor reads the answer to another request: these sources, "
                "recipes, grouping, step selection or include options (fields "
                "included) differ from it. Send the request that gave the cursor, "
                "or send this one without a cursor to evaluate it."
            )
    target = resolve_response_budget(args.budget, state).tokens
    # Read over the whole stored answer, off the loop.
    (annotate, apart), caps = await asyncio.to_thread(
        lambda: (
            say_what_was_cut(stored.answer),
            answer_caps(stored.answer, stored.snapshot, target=target),
        )
    )
    reply = await stored_page(
        stored,
        pointer,
        offset,
        schema=OUTPUT_SCHEMA,
        state=state,
        annotate=annotate,
        cap=caps.get(pointer),
        target=target,
        over=say_over_target,
        apart=apart,
        series=SERIES,
    )
    return format_response(reply)


#: The request fields a ``continue`` call may echo: the request that gave
#: the cursor, sent again whole.
_REQUEST_ECHO = frozenset({"sources", "recipes", "include", "group_by", "step", "all_steps"})


@registry.tool(
    name="analyze_results",
    title="Analyze Results",
    description=(
        "Measure finished simulation results: apply typed recipes to completed "
        "jobs, RAW or log files and get values, reductions, group splits and spec "
        "verdicts attributed to case, run and .step. One call spans many sources "
        "and many metrics, so batch them instead of calling per metric. Every "
        "recipe is evaluated in the call; a reply too long for its target is cut, "
        "and 'continue' with a cut entry's next reads the rest. On a wide sweep "
        "set include.fields to return only the numbers you need. "
        # The recipe roster, with the plain words a caller searches for. A host
        # matching a request against tool descriptions sees only this text, so a
        # metric absent from it is a metric nothing can route to; and the
        # compact listing strips the per-branch schema prose, which leaves this
        # as the only place the names appear at all.
        # tests/test_consolidated_contracts.py pins every discriminant here.
        "Recipes: stability (phase margin, gain margin, unity-gain crossover, DC "
        "gain), bode_filter (cutoff, bandwidth, Q), bode_point, bode_slope "
        "(dB/decade), bode_crossing, ac_structure (poles, zeros), resonance, "
        "return_loss (VSWR), thd (distortion), tone (amplitude and phase at one "
        "frequency), noise_integral, signal_stats (RMS, "
        "ripple, peak-to-peak, percentiles), edges (rise/fall time), timing "
        "(propagation delay), periodic (duty cycle), transient_response (overshoot, settling, "
        "load step), time_steps (smallest time step, where), operating_point (bias point, gm/gds/vth), measurements "
        "(.meas), value (one trace at one axis point), summary, waveform "
        "(samples, CSV), plot (chart). inspect(kind='reference', query=...) "
        "searches these by plain words and returns a recipe's fields. A signal is "
        "one trace or V(a,b), not an expression; other trace math is numpy "
        "(guide section 'signals')."
    ),
    input_model=AnalyzeResultsInput,
    annotations=types.ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    ),
    output_schema=OUTPUT_SCHEMA,
)
async def handle_analyze_results(
    args: AnalyzeResultsInput, state: SessionState
) -> types.CallToolResult:
    if args.continuation is not None or (
        args.include.per_run is not None and args.include.per_run.cursor is not None
    ):
        return await _read_on(args, state)
    evaluation = await evaluate_analysis(args, state)
    held = args.include.per_run.limit_note() if args.include.per_run else None
    if held:
        evaluation = replace(
            evaluation,
            argument_notes=(*evaluation.argument_notes, f"include.per_run.{held}"),
        )
    target = resolve_response_budget(args.budget, state).tokens
    served = frozenset(state.tool_dispatch)

    def delivered() -> tuple[dict[str, Any], tuple[Annotate, dict[str, int]], dict[str, Any]]:
        # Work over the whole answer: off the loop.
        answer = for_session(_assemble(evaluation), served)
        return answer, say_what_was_cut(answer), evaluation.snapshot()

    answer, (annotate, apart), snapshot = await asyncio.to_thread(delivered)
    reply = await presented(
        answer,
        tool=TOOL,
        schema=OUTPUT_SCHEMA,
        snapshot=snapshot,
        state=state,
        annotate=annotate,
        target=target,
        caps=answer_caps(answer, snapshot, target=target),
        over=say_over_target,
        apart=apart,
        series=SERIES,
    )
    return format_response(reply)
