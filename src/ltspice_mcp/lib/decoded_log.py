"""Validated resident log facts without dependency parser imports or file reads.

The caller supplies an already bounded worker manifest. Mutable accessors return
copies; captured identities are frozen. The fact snapshot identity is separate
from the worker operation/cache identity used for artifact continuations.
"""

from __future__ import annotations

import copy
import json
import math
import re
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_types import LogDiagnostics, MeasurementsOutput, NativeTableBlock
from ltspice_mcp.lib.parser_capture import CapturedInputs

LogSectionName = Literal[
    "diagnostics",
    "measurements",
    "device_op",
    "steps",
    "op_iterations",
    "temperatures",
    "fourier",
    "native_tables",
    "error_context",
]
_NAMES = {"raw": "input.raw", "log": "input.log", "console": "input.exe.log"}
_SECTIONS: tuple[LogSectionName, ...] = (
    "diagnostics",
    "measurements",
    "device_op",
    "steps",
    "op_iterations",
    "temperatures",
    "fourier",
    "native_tables",
    "error_context",
)
_Value = TypeVar("_Value")


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Error(_Record):
    type: str = Field(min_length=1)
    message: str


class _Section(_Record, Generic[_Value]):
    status: Literal["parsed", "absent", "error"]
    value: _Value | None
    error: _Error | None
    nonfinite_count: int = Field(ge=0)


class _Scan(_Record):
    complete: bool
    capturedbytes: int = Field(ge=0)


class _Iterations(_Record):
    attempts: int = Field(ge=0)
    succeeded: int = Field(ge=0)


class _Temperatures(_Record):
    temp_c: float | None
    tnom_c: float | None


class _Harmonic(_Record):
    number: int | None
    frequency: float | None
    magnitude: float | None
    phase: float | None


class _Fourier(_Record):
    signal: str
    thd: float | None
    thd_unit: Literal["%"]
    phd: float | None
    fundamental_frequency: float | None
    harmonics: list[_Harmonic]


class _Manifest(_Record):
    version: int = Field(ge=1, le=1)
    capture_facts: CapturedInputs
    scan: _Scan
    diagnostics: _Section[LogDiagnostics]
    measurements: _Section[MeasurementsOutput]
    device_op: _Section[dict[str, float | None]]
    steps: _Section[list[dict[str, float | None]]]
    op_iterations: _Section[_Iterations]
    temperatures: _Section[_Temperatures]
    fourier: _Section[list[_Fourier]]
    native_tables: _Section[list[NativeTableBlock]]
    error_context: _Section[str]


def _check_json(value: Any, active: set[int]) -> None:
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Log manifest numbers must be finite; use null and nonfinite_count")
        return
    if type(value) not in (dict, list):
        raise ValueError("Log manifest must contain only plain JSON values")
    if id(value) in active:
        raise ValueError("Log manifest cannot contain cyclic containers")
    if len(active) >= 16:
        raise ValueError("Log manifest exceeds the nesting of supported fact shapes")
    active.add(id(value))
    try:
        if type(value) is dict:
            if any(type(key) is not str for key in value):
                raise ValueError("Log manifest keys must be strings")
            values = value.values()
        else:
            values = value
        for item in values:
            _check_json(item, active)
    finally:
        active.remove(id(value))


def _numeric_nulls(name: LogSectionName, value: Any) -> int:
    if value is None or name in ("diagnostics", "op_iterations", "error_context"):
        return 0
    if name in ("device_op", "temperatures"):
        return sum(item is None for item in value.values())
    if name == "steps":
        return sum(item is None for row in value for item in row.values())
    if name == "measurements":
        count = 0
        for entry in value["measurements"].values():
            count += sum(item is None for item in entry["values"])
            for key in ("range_from", "range_to", "at"):
                if key in entry:
                    field = entry[key]
                    count += (
                        sum(item is None for item in field)
                        if isinstance(field, list)
                        else field is None
                    )
        return count
    if name == "native_tables":
        count = 0
        for block in value:
            if block["layout"] == "scalar_print":
                for entry in block["entries"]:
                    count += entry["real"] is None
                    count += entry["representation"] == "complex" and entry["imag"] is None
            else:
                for row in block["rows"]:
                    count += row["frequency"] is None
                    count += row["real"] is None
                    count += block["column"]["representation"] == "complex" and row["imag"] is None
        return count
    return sum(
        sum(block[key] is None for key in ("thd", "phd", "fundamental_frequency"))
        + sum(
            sum(harmonic[key] is None for key in ("frequency", "magnitude", "phase"))
            for harmonic in block["harmonics"]
        )
        for block in value
    )


def _check_native_blocks(blocks: list[dict[str, Any]], log_bytes: int) -> None:
    previous_end = 0
    for ordinal, block in enumerate(blocks):
        start, end = block["line_start"], block["line_end"]
        if block["ordinal"] != ordinal or not previous_end < start <= end <= log_bytes + 1:
            raise ValueError("Native block ordinals/source lines must retain captured order")
        previous_end = end
        if block["closed_by"] == "next_print_header" and ordinal + 1 == len(blocks):
            raise ValueError("Native next-header closure needs a following block")
        current = block["listing_current"]
        if current is not None and (
            not current["id"] or not current["analysis_label"] or not 0 < current["line"] < start
        ):
            raise ValueError("Native listing context must precede its printed block")
        scalar = block["layout"] == "scalar_print"
        if scalar:
            if (
                block["printed_analysis_label"] is not None
                or block["printed_title_line"] is not None
            ):
                raise ValueError("Scalar print cannot invent a printed analysis title")
            rows = block["entries"]
        else:
            label, title = block["printed_analysis_label"], block["printed_title_line"]
            if (
                label is None
                or title is None
                or not re.match(re.escape(label) + r"[ \t]", title.lstrip())
            ):
                raise ValueError("Native frequency tables require their matching printed title")
            if not block["column"]["label"] or re.search(r"\s", block["column"]["label"]):
                raise ValueError("Native column labels must be literal nonempty tokens")
            rows = block["rows"]
        if not rows:
            raise ValueError("Native printed blocks cannot be empty")
        previous_line = start - 1 if scalar else start
        for index, row in enumerate(rows):
            if not previous_line < row["line"] <= end:
                raise ValueError("Native row source lines must remain ordered within their block")
            previous_line = row["line"]
            if scalar:
                if not row["label"] or re.search(r"[\s=]", row["label"]):
                    raise ValueError("Native scalar labels must be literal nonempty tokens")
                representation = row["representation"]
            else:
                if row["index"] != index:
                    raise ValueError("Native printed row indices must retain their original order")
                representation = block["column"]["representation"]
            if representation == "real" and row["imag"] is not None:
                raise ValueError("Native real syntax cannot carry an imaginary component")
        if rows[-1]["line"] != end or (scalar and rows[0]["line"] != start):
            raise ValueError("Native block range must match its observed numeric rows")


def _check_sections(facts: dict[str, Any], captured: CapturedInputs) -> None:
    present = {item.role for item in captured.files}
    for name in _SECTIONS:
        section = facts[name]
        status, value, error, count = (
            section["status"],
            section["value"],
            section["error"],
            section["nonfinite_count"],
        )
        if status == "error":
            if value is not None or error is None:
                raise ValueError(f"{name} error requires null value and an error record")
        elif error is not None:
            raise ValueError(f"{name} successful/absent section cannot carry an error")
        if status == "absent" and count:
            raise ValueError(f"{name} absent section cannot count nonfinite observations")
        if name in ("diagnostics", "op_iterations", "error_context") and count:
            raise ValueError(f"{name} cannot contain nonfinite numeric facts")
        has_input = (
            bool(present & {"log", "console"}) if name == "diagnostics" else "log" in present
        )
        if not has_input and status != "absent":
            raise ValueError(f"{name} input is absent")
        if status == "error":
            continue
        if count > _numeric_nulls(name, value):
            raise ValueError(f"{name} nonfinite_count exceeds numeric null fields")
        if name == "diagnostics":
            if value is None or (status == "parsed") != has_input:
                raise ValueError("diagnostics status contradicts captured presence")
            if status == "absent" and any(value.values()):
                raise ValueError("Absent diagnostics must be empty")
        elif name == "measurements":
            if not has_input:
                if value is not None:
                    raise ValueError("Absent measurement input requires null value")
                continue
            if value is None or (status == "parsed") != bool(value["measurements"]):
                raise ValueError("Measurement status contradicts measurement entries")
            if value["step_count"] < 0:
                raise ValueError("Measurement step_count must be nonnegative")
            for entry in value["measurements"].values():
                for key in ("range_from", "range_to", "at"):
                    field = entry.get(key)
                    if isinstance(field, list) and len(field) != len(entry["values"]):
                        raise ValueError("Measurement metadata must align with values")
        elif name in ("device_op", "steps"):
            if value is None or (status == "parsed") != bool(value):
                raise ValueError(f"{name} status contradicts section entries")
            if name == "steps" and any(not row for row in value):
                raise ValueError("Parsed step rows require numeric parameters")
        elif name == "op_iterations":
            if value is None or value["succeeded"] > value["attempts"]:
                raise ValueError("OP successes cannot exceed attempts")
            if (status == "parsed") != bool(value["attempts"]):
                raise ValueError("OP iteration status contradicts attempts")
        elif name == "temperatures":
            if value is None or (status == "parsed") != (
                any(v is not None for v in value.values()) or count > 0
            ):
                raise ValueError("Temperature status contradicts observations")
        elif name == "error_context":
            if not has_input:
                if value is not None:
                    raise ValueError("Absent error_context input requires null value")
            elif status != "parsed" or value is None:
                raise ValueError("error_context requires a parsed excerpt for present input")
        elif name == "native_tables":
            if not has_input:
                if value is not None:
                    raise ValueError("Absent native-table input requires null value")
                continue
            if value is None or (status == "parsed") != bool(value):
                raise ValueError("Native-table status contradicts its physical print blocks")
            if count != _numeric_nulls(name, value):
                raise ValueError("Native nonfinite counts must match observed numeric nulls")
            _check_native_blocks(
                value, next(item.size_bytes for item in captured.files if item.role == "log")
            )
        elif not has_input:
            if value is not None:
                raise ValueError("Absent Fourier input requires null value")
        elif value is None or (status == "parsed") != bool(value):
            raise ValueError("Fourier status contradicts decoded blocks")


@dataclass(frozen=True, init=False, slots=True)
class DecodedLog:
    """Strictly validated, copy-isolated facts from one complete captured scan."""

    _captured: CapturedInputs
    _facts: dict[str, Any] = field(repr=False)
    _snapshot_id: str

    def __init__(self, metadata: dict[str, Any]) -> None:
        if type(metadata) is not dict:
            raise ValueError("Log manifest must be a plain JSON object")
        _check_json(metadata, set())
        validated = _Manifest.model_validate_json(
            json.dumps(metadata, allow_nan=False), strict=True
        )
        captured = validated.capture_facts
        roles = [item.role for item in captured.files] + list(captured.absent)
        if len(roles) != 3 or set(roles) != set(_NAMES):
            raise ValueError("Each captured role must occur exactly once, present or absent")
        for item in captured.files:
            if (
                item.name != _NAMES[item.role]
                or item.size_bytes < 0
                or not re.fullmatch(r"[0-9a-f]{64}", item.sha256)
            ):
                raise ValueError("Invalid captured filename, size or digest")
        if not validated.scan.complete or validated.scan.capturedbytes != sum(
            item.size_bytes for item in captured.files if item.role != "raw"
        ):
            raise ValueError("Log scan must cover all captured log/console bytes completely")
        facts = validated.model_dump(mode="json")
        _check_sections(facts, captured)
        object.__setattr__(self, "_facts", facts)
        object.__setattr__(self, "_captured", captured)
        object.__setattr__(
            self,
            "_snapshot_id",
            captured.cache_key(dialect=None, producing_dialect=None, revision="log-facts-v1"),
        )

    @property
    def captured(self) -> CapturedInputs:
        return self._captured

    @property
    def snapshot_id(self) -> str:
        """Captured fact identity; separate from parser cache/continuation identity."""
        return self._snapshot_id

    @property
    def capture_facts(self) -> dict[str, Any]:
        return copy.deepcopy(self._facts["capture_facts"])

    @property
    def scan(self) -> dict[str, Any]:
        return copy.deepcopy(self._facts["scan"])

    def section(self, name: LogSectionName) -> dict[str, Any]:
        if name not in _SECTIONS:
            raise ValueError(f"Unknown log section: {name}")
        return copy.deepcopy(self._facts[name])

    def value(self, name: LogSectionName) -> Any:
        """Return detached parsed/absent facts, raising on a section decode error."""
        section = self.section(name)
        if section["status"] == "error":
            raise ResultError(f"Log section {name!r} failed: {section['error']['message']}")
        return section["value"]

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._facts)
