"""Validate a parser's plain manifest before allocating resident arrays."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw, StepValue, transient_time_offset
from ltspice_mcp.lib.parser_capture import parser_cache_key
from ltspice_mcp.lib.raw_header import RawHeader, RawLimits
from ltspice_mcp.lib.store import parser_file_in

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_log import DecodedLog
    from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts

_HEADER = TypeAdapter(RawHeader)


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class _Array(_Record):
    plot_index: int = Field(ge=0)
    trace_index: int = Field(ge=0)
    file: str
    dtype: Literal["<f4", "<f8", "<c16"]
    count: int = Field(gt=0)
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class _Range(_Record):
    step_index: int = Field(ge=0)
    offset: int = Field(ge=0)
    length: int = Field(gt=0)
    log_row: int | None = Field(default=None, ge=0)


class _Plot(_Record):
    plot_index: int = Field(ge=0)
    decoder_has_axis: bool
    data_convention: Literal["stored"]
    step_status: Literal["unstepped", "matched", "unresolved", "mismatch"]
    step_ranges: list[_Range] | None


class _Parameter(_Record):
    name: str
    token: str
    value: float | int | None


class _LogRow(_Record):
    ordinal: int = Field(ge=0)
    line_number: int = Field(gt=0)
    text: str
    parameters: list[_Parameter]


class _StepLog(_Record):
    present: bool
    size_bytes: int | None = Field(ge=0)
    sha256: str | None
    encoding: str | None
    status: Literal["absent", "parsed", "unsupported"]
    rows: list[_LogRow]


class _RawPayload(_Record):
    header: dict
    arrays: list[_Array]
    plots: list[_Plot]
    step_log: _StepLog


class _Reply(_RawPayload):
    version: Literal[1]
    status: Literal["ok"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")


class _ArtifactsReply(_Record):
    version: Literal[1]
    status: Literal["ok"]
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw: _RawPayload | None
    logs: dict


def read_parsed_artifacts(
    metadata: dict,
    directory: Path,
    *,
    limits: RawLimits,
    require_raw: bool,
    request: dict,
) -> ParsedArtifacts:
    """Validate shared capture identity and log facts before reading arrays."""
    from ltspice_mcp.lib.decoded_log import DecodedLog
    from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts

    if type(metadata.get("version")) is not int:
        raise ValueError("Parser protocol version must be an integer")
    reply = _ArtifactsReply.model_validate(metadata)
    if (reply.raw is not None) != require_raw:
        raise ValueError("Parser result does not match the requested operation")
    logs = DecodedLog(reply.logs)
    key = parser_cache_key(
        logs.captured,
        dialect=request["dialect"],
        producing_dialect=request["producing_dialect"],
        limits=request["limits"],
    )
    if key != reply.cache_key:
        raise ValueError("Parser capture identity disagrees with its result")
    raw = None
    if reply.raw is not None:
        raw_metadata = reply.raw
        captures = {item.role: item for item in logs.captured.files}
        captured_raw = captures.get("raw")
        if (
            captured_raw is None
            or raw_metadata.header.get("size_bytes") != captured_raw.size_bytes
        ):
            raise ValueError("RAW header does not match its captured file")
        captured_log = captures.get("log")
        if raw_metadata.step_log.present != (captured_log is not None) or (
            captured_log is not None
            and (
                raw_metadata.step_log.size_bytes != captured_log.size_bytes
                or raw_metadata.step_log.sha256 != captured_log.sha256
            )
        ):
            raise ValueError("RAW steps and log facts refer to different captures")
        steps = logs.section("steps")
        if raw_metadata.step_log.status == "parsed" and steps["status"] != "error":
            rows = [
                {parameter.name: parameter.value for parameter in row.parameters}
                for row in raw_metadata.step_log.rows
            ]
            if rows != steps["value"]:
                raise ValueError("Step values disagree between RAW metadata and log facts")
        raw = _materialize_raw(raw_metadata, directory, limits=limits, cache_key=key, logs=logs)
    return ParsedArtifacts(snapshot_id=key, raw=raw, logs=logs)


def _validate_step_log(log: _StepLog) -> None:
    if log.status == "absent":
        if (
            log.present
            or log.rows
            or any(value is not None for value in (log.size_bytes, log.sha256, log.encoding))
        ):
            raise ValueError("Step logs declared absent cannot carry captured data")
        return
    if (
        not log.present
        or log.size_bytes is None
        or not log.encoding
        or log.sha256 is None
        or re.fullmatch(r"[0-9a-f]{64}", log.sha256) is None
    ):
        raise ValueError("Present step logs require size, digest and encoding")
    if log.status == "unsupported" and log.rows:
        raise ValueError("Step metadata cannot come from an unsupported log")
    previous_line = 0
    for index, row in enumerate(log.rows):
        if row.ordinal != index or row.line_number <= previous_line:
            raise ValueError("Step log rows must retain their recorded order")
        previous_line = row.line_number
        if len({p.name.casefold() for p in row.parameters}) != len(row.parameters):
            raise ValueError("Step parameters must have unique names")


def _step_values(
    plot: _Plot, points: int, log: _StepLog
) -> tuple[list[int] | None, list[dict[str, StepValue]] | None]:
    if plot.step_ranges is None:
        if plot.step_status not in {"unresolved", "mismatch"}:
            raise ValueError("Resolved steps require complete ranges")
        return None, None
    offset = 0
    starts = []
    values = []
    referenced_rows = set()
    for index, part in enumerate(plot.step_ranges):
        if part.step_index != index or part.offset != offset:
            raise ValueError("Step ranges must partition the stored trace in order")
        starts.append(offset)
        offset += part.length
        if part.log_row is not None:
            if part.log_row >= len(log.rows) or part.log_row in referenced_rows:
                raise ValueError("Step range references an absent log row")
            referenced_rows.add(part.log_row)
            row = log.rows[part.log_row]
            if len({p.name for p in row.parameters}) != len(row.parameters):
                raise ValueError("Step parameters must have unique names")
            values.append(
                {p.name: p.value if p.value is not None else p.token for p in row.parameters}
            )
    if not starts or offset != points:
        raise ValueError("Step ranges must cover every stored point exactly once")
    if plot.step_status == "unstepped" and (len(starts) != 1 or referenced_rows):
        raise ValueError("Step metadata contradicts the unstepped plot")
    if values and len(values) != len(starts):
        raise ValueError("Step metadata must bind every range or none")
    if plot.step_status == "matched":
        if log.status != "parsed" or not values or referenced_rows != set(range(len(log.rows))):
            raise ValueError("Step matches must bind every captured row exactly once")
    elif referenced_rows:
        raise ValueError("Step rows can bind only a matched plot")
    return starts, values or None


def read_decoded_raw(
    metadata: dict, directory: Path, *, limits: RawLimits, logs: DecodedLog | None = None
) -> DecodedRaw:
    """Read exact typed numeric bytes after confirmed worker-tree exit.

    Numeric storage has no executable or dtype-object encoding. Hash the actual
    resident bytes, so a file change between stat and read cannot silently
    substitute a different array. No RAW/log dependency parser runs here.
    """
    if type(metadata.get("version")) is not int:
        raise ValueError("Parser protocol version must be an integer")
    reply = _Reply.model_validate(metadata)
    return _materialize_raw(
        reply,
        directory,
        limits=limits,
        cache_key=reply.cache_key,
        logs=logs,
    )


def _materialize_raw(
    reply: _RawPayload,
    directory: Path,
    *,
    limits: RawLimits,
    cache_key: str,
    logs: DecodedLog | None,
) -> DecodedRaw:
    """Admit every plot and file before constructing resident numeric views."""
    _validate_step_log(reply.step_log)
    header = _HEADER.validate_json(json.dumps(reply.header), strict=True, extra="forbid")
    if not 0 < len(header.plots) <= limits.plots or len(reply.plots) != len(header.plots):
        raise ValueError("Parser plot inventory exceeds its bound or disagrees")
    if (
        not 0 < header.size_bytes <= limits.input_bytes
        or not 0 < header.header_bytes <= limits.header_bytes
    ):
        raise ValueError("Parser input/header size exceeds its bound")
    if not 0 < header.numeric_bytes <= limits.numeric_bytes:
        raise ValueError("Parser numeric allocation exceeds its byte limit")
    expected_bytes = 0
    expected_arrays = {}
    step_views = []
    for index, plot in enumerate(header.plots):
        if plot.index != index or reply.plots[index].plot_index != index:
            raise ValueError("Parser plot identities are inconsistent")
        if (
            not 0 < plot.variable_count <= limits.variables
            or len(plot.variables) != plot.variable_count
        ):
            raise ValueError("Parser variable inventory is invalid")
        if (
            not 0 < plot.point_count <= limits.points
            or len(plot.value_bytes) != plot.variable_count
        ):
            raise ValueError("Parser declared shape is invalid")
        if plot.numeric_bytes != plot.point_count * sum(plot.value_bytes):
            raise ValueError("Parser plot numeric byte count disagrees")
        transient_time_offset(plot)
        if (
            "stepped" in {flag.casefold() for flag in plot.flags}
            and reply.plots[index].step_status == "unstepped"
        ):
            raise ValueError("Unstepped status contradicts stored step evidence")
        step_views.append(_step_values(reply.plots[index], plot.point_count, reply.step_log))
        for trace, width in enumerate(plot.value_bytes):
            if width not in {4, 8, 16} or plot.variables[trace].index != trace:
                raise ValueError("Parser trace layout is invalid")
            expected_arrays[index, trace] = (plot.point_count, width)
            expected_bytes += plot.point_count * width
    if expected_bytes != header.numeric_bytes or len(reply.arrays) != len(expected_arrays):
        raise ValueError("Parser numeric inventory disagrees with its allocation")
    entries = {}
    for array in reply.arrays:
        identity = array.plot_index, array.trace_index
        if identity not in expected_arrays or identity in entries:
            raise ValueError("Parser arrays must cover every trace exactly once")
        count, width = expected_arrays[identity]
        expected_dtype = {4: "<f4", 8: "<f8", 16: "<c16"}[width]
        if (
            array.count != count
            or array.dtype != expected_dtype
            or array.byte_size != count * width
        ):
            raise ValueError("Parser numeric dtype, count or byte size disagrees")
        if array.file != f"p{array.plot_index}_t{array.trace_index}.bin":
            raise ValueError("Parser array filename disagrees with its identity")
        entries[identity] = array
    # Admit every file before the first resident read. Recheck length/digest
    # after each read as well, because files can change after this inventory.
    for entry in entries.values():
        path = directory / entry.file
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path != parser_file_in(directory, entry.file):
            raise ValueError("Parser arrays must be contained regular files")
        if info.st_size != entry.byte_size:
            raise ValueError("Parser numeric file size disagrees")
    plots = []
    for index, plot in enumerate(header.plots):
        waves = []
        for trace in range(plot.variable_count):
            entry = entries[index, trace]
            path = directory / entry.file
            with path.open("rb") as handle:
                wave = np.fromfile(handle, dtype=entry.dtype, count=entry.count)
                extra = handle.read(1)
            if len(wave) != entry.count or extra:
                raise ValueError("Parser numeric file changed while reading")
            if hashlib.sha256(wave.data.cast("B")).hexdigest() != entry.sha256:
                raise ValueError("Parser numeric file digest disagrees")
            waves.append(wave)
        ranges, steps = step_views[index]
        plots.append(
            DecodedPlot(
                plot,
                waves,
                snapshot_id=cache_key,
                steps=steps,
                step_offsets=ranges,
                step_status=reply.plots[index].step_status,
            )
        )
    return DecodedRaw(plots, logs=logs)
