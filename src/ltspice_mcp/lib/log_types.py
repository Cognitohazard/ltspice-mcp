"""Log result shapes shared without importing dependency parsers."""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal, NotRequired, get_args

from typing_extensions import TypedDict

#: The sections a captured log decodes into, in the order the decoder writes
#: them: the one list the decoder and the validated view share.
LogSectionName = Literal[
    "diagnostics",
    "measurements",
    "device_op",
    "steps",
    "op_iterations",
    "temperatures",
    "fourier",
    "native_tables",
    "fra_margins",
    "error_context",
]
LOG_SECTIONS: tuple[LogSectionName, ...] = get_args(LogSectionName)


class LogDecodeError(ValueError):
    """A captured log identity or parsed section is invalid."""


class LogLimitError(LogDecodeError):
    """A log cannot be decoded completely within the supplied finite limits."""


@dataclass(frozen=True)
class LogLimits:
    log_bytes: int
    line_bytes: int
    lines: int
    section_entries: int
    metadata_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            if type(value := getattr(self, field.name)) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a finite positive integer")


def log_section(
    operation: Callable[[], Any], limits: LogLimits, *, present: bool = True
) -> dict[str, Any]:
    """Convert one bounded section to plain facts with explicit errors/nulls."""
    nonfinite = 0
    entries = 0

    def plain(value: Any) -> Any:
        nonlocal nonfinite, entries
        entries += 1
        if entries > limits.section_entries:
            raise LogLimitError("section_entries limit exceeded")
        if value is None or type(value) in (str, bool, int):
            return value
        if type(value) is float:
            if not math.isfinite(value):
                nonfinite += 1
                return None
            return value
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, dict) and all(type(key) is str for key in value):
            return {key: plain(item) for key, item in value.items()}
        raise LogDecodeError("Log section contains a non-JSON value")

    try:
        value = plain(operation())
    except LogLimitError:
        raise
    except Exception as exc:
        return {
            "status": "error",
            "value": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "nonfinite_count": nonfinite,
        }
    return {
        "status": "parsed" if present else "absent",
        "value": value,
        "error": None,
        "nonfinite_count": nonfinite,
    }


NativeRepresentation = Literal["real", "complex"]
NativeAnalysisLabel = Literal[
    "Sensitivity Analysis",
    "DISTORTION - 2nd harmonic",
    "DISTORTION - 3rd harmonic",
    "DISTORTION - IM: f1+f2",
    "DISTORTION - IM: f1-f2",
    "DISTORTION - IM: 2f1-f2",
]


class NativeListing(TypedDict):
    id: str
    analysis_label: str
    line: int


class _NativeBlock(TypedDict):
    ordinal: int
    line_start: int
    line_end: int
    printed_analysis_label: NativeAnalysisLabel | None
    printed_title_line: str | None
    listing_current: NativeListing | None
    closed_by: Literal["form_feed", "next_print_header", "ngspice_done"]
    analysis_extent: Literal["unknown"]


class NativeScalarEntry(TypedDict):
    line: int
    label: str
    representation: NativeRepresentation
    real: float | None
    imag: float | None
    unit: None


class NativeScalarBlock(_NativeBlock):
    layout: Literal["scalar_print"]
    axis: None
    entries: list[NativeScalarEntry]


class NativeCoordinate(TypedDict):
    label: Literal["frequency"]
    unit: None
    convention: Literal["printed"]


class NativeColumn(TypedDict):
    label: str
    representation: NativeRepresentation
    unit: None


class NativeFrequencyRow(TypedDict):
    line: int
    index: int
    frequency: float | None
    real: float | None
    imag: float | None


class NativeFrequencyBlock(_NativeBlock):
    layout: Literal["frequency_table"]
    coordinate: NativeCoordinate
    column: NativeColumn
    rows: list[NativeFrequencyRow]


NativeTableBlock = NativeScalarBlock | NativeFrequencyBlock


class MeasErrorEntry(TypedDict):
    """One .MEAS parse failure with an optional fix suggestion."""

    directive: str
    raw_block: str
    suggestion: str | None


class LogCondition(TypedDict):
    """One condition a run's output named, counted under one set of rules
    (``log_conditions``)."""

    code: str
    count: int
    first_text: str
    channel: Literal["log", "console"]
    severity: NotRequired[str]


class LogDiagnostics(TypedDict):
    """Return shape of :func:`extract_log_diagnostics`."""

    warnings: list[str]
    errors: list[str]
    meas_errors: list[MeasErrorEntry]
    #: What the log and console say under each set of rules that found
    #: anything, by the set's name (``log_conditions.classify_channels``).
    conditions: dict[str, list[LogCondition]]


class UnreadableValue(TypedDict):
    """Why a measurement's printed value was not read: the log line that
    printed it (None where no line names it) and the reason."""

    line: str | None
    reason: str


class NotEstablished(TypedDict):
    """The steps of a measurement whose printed value is not established as
    its result, from 0, and why."""

    steps: list[int]
    reason: str


class _MeasurementMetadata(TypedDict, total=False):
    """Optional metadata folded into a .MEAS result.

    Each metadata field is either a scalar (when the value is constant
    across .step iterations — e.g. a literal ``FROM=2m``) or a list of
    per-step values (when LTspice computed a different marker per step,
    e.g. TRIG/TARG times of a per-step rise time).
    """

    range_from: float | list[float | None] | None
    range_to: float | list[float | None] | None
    at: float | list[float | None] | None
    unreadable: UnreadableValue
    failed_steps: list[int]
    #: How the log printed the measurement (``log_parser.PrintedForm``), as the
    #: decoder found it; a reader turns it into ``answer`` or
    #: ``answer_unknown`` (``log_parser.answered_measurements``) and never
    #: passes it on.
    printed_form: str
    #: The field holding the measurement's result, ``"value"`` (its
    #: ``values``) or ``"at"``, as ``log_parser.measurement_answer`` decides.
    answer: str
    #: Why the field holding the result is not known, where it is not.
    answer_unknown: str
    #: The steps, from 0, whose printed value is not established as the
    #: result, and why (``log_parser.answered_measurements``): their values
    #: are null.
    not_established: NotEstablished


class MeasurementEntry(_MeasurementMetadata):
    """One .MEAS result, with optional range/at metadata folded in.

    ``values`` is one entry per .step iteration (length 1 for unstepped runs).
    ``range_from`` / ``range_to`` carry the FROM/TO bounds for windowed measurements.
    ``at`` carries the AT/WHEN time/freq for point measurements. Missing when
    not applicable (use ``.get`` rather than ``[]`` to access). When the
    underlying value varies per .step, the field is a list aligned with
    ``values`` rather than a single scalar. ``unreadable`` is set when the log
    printed something other than a number for the value, such as a WHEN's
    threshold written as a parameter or an expression: the value is null, and
    only this entry fails. ``failed_steps`` lists the steps, from 0, whose row
    of a stepped table LTspice printed ``failed``; their values are null
    (:func:`measurement_failed`). ``answer`` names the field holding the
    result: ``at`` for a WHEN with no FIND printed on its own line, whose
    ``values`` hold the level it searched for, and ``value`` otherwise;
    ``answer_unknown`` says why it is not known in its place.
    ``not_established`` names the steps whose printed value is not
    established as the result (XVII prints 0 for a failed step), null in
    ``values``.
    """

    values: list[float | None]


def measurement_named(name: str, names: Iterable[str]) -> str | None:
    """Which of a log's measurement ``names`` a caller's ``name`` refers to,
    or None for none of them.

    The log reader holds a measurement under the name lower-cased, whatever
    case the deck gave it, so a name is matched ignoring case; one held
    exactly as spelled is preferred.
    """
    held = names if isinstance(names, Collection) else list(names)
    if name in held:
        return name
    folded = name.casefold()
    return next((key for key in held if key.casefold() == folded), None)


class MeasurementsOutput(TypedDict):
    """Return shape of :func:`parse_measurements`.

    ``errors`` is populated only on the empty-measurements path (the log
    had no .MEAS results and the parser surfaced why). Always present in
    the return value — ``None`` when the measurement parse succeeded.

    ``measurements`` is keyed by .meas name; each entry is a structured
    :class:`MeasurementEntry` with ``values`` plus folded-in ``range_from``,
    ``range_to``, and ``at`` metadata. The flat ``name_from`` / ``name_to`` /
    ``name_at`` keys that spicelib emits are not surfaced separately.

    ``failed_measurements`` names, in the order the log printed them, the
    measurements LTspice printed as failed: ``Measurement "name" FAIL'ed``, or
    ``failed`` in a row of the measurement's stepped table. It means that and
    nothing else; a value that is null for another reason (``unreadable``, or a
    number that is not finite) is not listed.
    """

    measurements: dict[str, MeasurementEntry]
    step_count: int
    errors: list[str] | None
    warnings: list[str] | None
    failed_measurements: list[str]
    #: The LTspice build the log's first line names, as the decoder found it
    #: (``simulator_build.log_banner_build``); a reader turns it into the
    #: producer its entries are answered for
    #: (``log_parser.answered_measurements``) and never passes it on.
    printed_by: NotRequired[str]


def measurement_failed(table: Mapping[str, Any], name: str, step: int | None = None) -> bool:
    """Whether LTspice printed the measurement ``name`` of a log's measurement
    ``table`` (:class:`MeasurementsOutput`) as failed: at ``step`` (from 0),
    or with no step at every step it was taken.

    The one reading of the log decoder's answer. A name ``failed_measurements``
    lists failed at the steps its entry's ``failed_steps`` gives, where its
    stepped table printed ``failed``; without them, LTspice printed it
    ``FAIL'ed`` as a whole and it failed at every step.
    """
    if name not in (table.get("failed_measurements") or ()):
        return False
    entry = (table.get("measurements") or {}).get(name)
    if not isinstance(entry, Mapping) or entry.get("failed_steps") is None:
        return True
    steps = entry["failed_steps"]
    if step is not None:
        return step in steps
    return set(range(len(entry.get("values") or ()))) <= set(steps)


def run_measurement_facts(
    table: Mapping[str, Any], requested: Sequence[str] | None
) -> dict[str, Any]:
    """What one run's measurement ``table`` (the log's, answered by
    ``log_parser.answered_measurements`` where it was) says the run did not
    give of the ``.meas`` its deck requests (``requested``, from
    ``result_observations.parse_requested_outputs``; None where the deck was
    not read), as a run keeps it.

    ``failed``: the names LTspice printed as failed (:func:`measurement_failed`),
    in the order the log printed them. ``failed_steps``: for a name that
    failed at some steps of a stepped run only, those steps from 0.
    ``not_established``: for a name whose printed value is not established as
    its result at some steps (XVII prints 0 for a failed step), those steps.
    ``not_printed``: each requested name the log printed nothing for, not
    even a failure (LTspice 26 on ``deck/save_omits_meas``; ngspice skips a
    top-level ``.meas`` in batch mode). Each only when it names something, so
    facts without them say every requested ``.meas`` was given. ``requested``
    is kept whenever the deck's own text requests a ``.meas``: for
    :func:`given_no_value`'s callers to say whether a failure is every one
    the deck requests, and for a receipt to say the deck has ``.meas`` cards
    without reading it again."""
    facts: dict[str, Any] = {}
    failed = list(table.get("failed_measurements") or ())
    entries = table.get("measurements") or {}
    if failed:
        facts["failed"] = failed
        partly = {
            name: list(entries[name]["failed_steps"])
            for name in failed
            if not measurement_failed(table, name)
        }
        if partly:
            facts["failed_steps"] = partly
    unestablished = {
        name: list(entry["not_established"]["steps"])
        for name, entry in entries.items()
        if isinstance(entry, Mapping) and entry.get("not_established")
    }
    if unestablished:
        facts["not_established"] = unestablished
    printed = list(entries)
    missing = [name for name in requested or () if measurement_named(name, printed) is None]
    if missing:
        facts["not_printed"] = missing
    if requested:
        facts["requested"] = list(requested)
    return facts


def given_no_value(facts: Mapping[str, Any]) -> list[str]:
    """The names a run's :func:`run_measurement_facts` say it gave no value
    for at any step: printed failed at every step, or not printed at all."""
    partly = facts.get("failed_steps") or {}
    return [
        *(name for name in facts.get("failed") or () if name not in partly),
        *(facts.get("not_printed") or ()),
    ]
