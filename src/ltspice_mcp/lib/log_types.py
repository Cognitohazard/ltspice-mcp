"""Log result shapes shared without importing dependency parsers."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any, Literal

from typing_extensions import TypedDict


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


class LogDiagnostics(TypedDict):
    """Return shape of :func:`extract_log_diagnostics`."""

    warnings: list[str]
    errors: list[str]
    meas_errors: list[MeasErrorEntry]


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


class MeasurementEntry(_MeasurementMetadata):
    """One .MEAS result, with optional range/at metadata folded in.

    ``values`` is one entry per .step iteration (length 1 for unstepped runs).
    ``range_from`` / ``range_to`` carry the FROM/TO bounds for windowed measurements.
    ``at`` carries the AT/WHEN time/freq for point measurements. Missing when
    not applicable (use ``.get`` rather than ``[]`` to access). When the
    underlying value varies per .step, the field is a list aligned with
    ``values`` rather than a single scalar.
    """

    values: list[float | None]


class MeasurementsOutput(TypedDict):
    """Return shape of :func:`parse_measurements`.

    ``errors`` is populated only on the empty-measurements path (the log
    had no .MEAS results and the parser surfaced why). Always present in
    the return value — ``None`` when the measurement parse succeeded.

    ``measurements`` is keyed by .meas name; each entry is a structured
    :class:`MeasurementEntry` with ``values`` plus folded-in ``range_from``,
    ``range_to``, and ``at`` metadata. The flat ``name_from`` / ``name_to`` /
    ``name_at`` keys that spicelib emits are not surfaced separately.
    """

    measurements: dict[str, MeasurementEntry]
    step_count: int
    errors: list[str] | None
    warnings: list[str] | None
    failed_measurements: list[str]
