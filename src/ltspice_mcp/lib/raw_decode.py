"""Fully materialize captured RAW files inside an owned finite parser worker.

The caller owns capture, process deadlines, memory limits, manifest admission
and directory cleanup. This decoder verifies admitted input identities, uses
preflight facts before dependency allocation, and writes positional numeric
files only. Stored samples (including signed time and complex trace zero) stay
unchanged; engineering axes, units and time rebasing belong to the facade.
"""

from __future__ import annotations

import codecs
import functools
import hashlib
import io
import math
import os
import stat
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from spicelib.raw.plot_data import PlotData

from ltspice_mcp.lib.decoded_raw import deck_time, transient_time_offset
from ltspice_mcp.lib.encoding import decode_spice_bytes_strictly, encode_spice_text
from ltspice_mcp.lib.log_parser import read_step_line
from ltspice_mcp.lib.parser_capture import CAPTURED_NAMES, CapturedFile, CapturedInputs
from ltspice_mcp.lib.raw_header import (
    RawHeader,
    RawLimitError,
    RawLimits,
    RawPlotHeader,
    RawTooLargeError,
    leading_parameters,
    preflight_raw,
)
from ltspice_mcp.lib.result_kinds import (
    STEPPED_SWEEP,
    Analysis,
    header_analysis,
    sampled_axis,
    stepped_kind,
)
from ltspice_mcp.lib.signal_analysis import UnorderedAxisError, window_span
from ltspice_mcp.lib.store import is_sha256, parser_file_in

_CHUNK_BYTES = 64 * 1024
_DTYPES = {4: "<f4", 8: "<f8", 16: "<c16"}


class RawDecodeError(ValueError):
    """Capture identity or dependency materialization did not satisfy the contract."""


class WindowRefusedError(ValueError):
    """A window of a plot's axis that keeps nothing a reader could use.

    ``parser_code`` is the code the parser worker reports it under, so the
    caller gets this message rather than a failed parse.
    """

    parser_code = "window_refused"


@dataclass(frozen=True)
class AxisWindow:
    """The span ``[start, end]`` of plot ``plot_index``'s axis a read keeps, in
    the axis's own unit (a transient's deck time); an open end keeps all on
    its side. Points are kept by ``signal_analysis.window_span``, step by step."""

    plot_index: int
    start: float | None
    end: float | None

    def text(self) -> str:
        """The window as a caller wrote it."""
        return f"({_bound(self.start)}, {_bound(self.end)})"


def _bound(value: float | None) -> str:
    return "None" if value is None else f"{value:g}"


def numbers_held(header: RawHeader, chosen: RawHeader) -> int:
    """The most bytes of numbers ``decode_raw`` holds reading ``chosen``'s
    traces of ``header``.

    spicelib reads a plot stored a trace at a time (LTspice's Fast Access)
    one trace after another, so it holds only the traces kept. A plot stored
    a point at a time it reads whole whichever traces are asked for: binary
    into one buffer, text into every trace (``docs/spicelib_bugs.md``, Bug
    40). Plots are decoded one after
    another, so the sum over them bounds what is held at once.
    """
    return sum(
        kept.numeric_bytes if plot.storage_order == "trace" else plot.numeric_bytes
        for plot, kept in zip(header.plots, chosen.plots, strict=True)
    )


class _PlotView:
    """An absolute-position read view ending at one preflight-validated plot.

    In particular, spicelib's ASCII trailing-line loop sees EOF instead of
    the next plot's Title. Binary lazy reads still reopen the admitted path;
    their lengths and offsets have already been validated by preflight.
    """

    def __init__(
        self, handle: BinaryIO, start: int, end: int, *, header_end: int, encoding: str
    ) -> None:
        self.handle = handle
        self.start = start
        self.end = end
        self.header_end = header_end
        self.encoding = encoding
        handle.seek(start)

    def tell(self) -> int:
        return self.handle.tell()

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        if whence == os.SEEK_CUR:
            offset += self.tell()
        elif whence == os.SEEK_END:
            offset += self.end
        elif whence != os.SEEK_SET:
            raise RawDecodeError("Invalid decoder seek mode")
        if not self.start <= offset <= self.end:
            raise RawDecodeError("Dependency read outside validated plot")
        return self.handle.seek(offset)

    def read(self, size: int = -1) -> bytes:
        position = self.tell()
        remaining = self.end - self.tell()
        data = self.handle.read(remaining if size < 0 else min(size, remaining))
        # PlotData decodes each header read independently. Supply one whole
        # encoded character instead of splitting UTF-8 or a surrogate pair.
        # Payload reads and absolute byte positions remain unchanged.
        extra = 0
        if position < self.header_end:
            if self.encoding == "utf_8" and size == 1 and data and data[0] >= 0xC2:
                width = 2 if data[0] < 0xE0 else 3 if data[0] < 0xF0 else 4
                extra = width - 1
            elif self.encoding == "utf_16_le" and size == 2 and len(data) == 2:
                if 0xD800 <= int.from_bytes(data, "little") <= 0xDBFF:
                    extra = 2
        if extra:
            data += self.handle.read(min(extra, self.header_end - self.tell()))
        return data

    def readline(self, size: int = -1) -> bytes:
        remaining = self.end - self.tell()
        return self.handle.readline(remaining if size < 0 else min(size, remaining))


class _WorkerPlot(PlotData):
    def _load_step_information(self, filename: Path) -> None:
        # Dependency construction calls this before arrays exist. Loading
        # captured step facts separately avoids hidden companion reads and
        # allocation/slicing driven by an unverified log-row count.
        self._steps = None


def _positive_limit(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a finite positive integer")


def _verified_inputs(
    captured: CapturedInputs, directory: Path, limits: RawLimits, step_log_bytes: int
) -> dict[str, CapturedFile]:
    roles = [item.role for item in captured.files] + list(captured.absent)
    if len(roles) != 3 or set(roles) != set(CAPTURED_NAMES):
        raise RawDecodeError("Captured input roles must occur exactly once, present or absent")
    records = {item.role: item for item in captured.files}
    if "raw" not in records:
        raise RawDecodeError("RAW decoding requires a captured raw file")
    for item in captured.files:
        if item.name != CAPTURED_NAMES[item.role]:
            raise RawDecodeError("Invalid captured input filename")
        if type(item.size_bytes) is not int or item.size_bytes < 0:
            raise RawDecodeError("Invalid captured input size")
        if not is_sha256(item.sha256):
            raise RawDecodeError("Invalid captured input digest")
    if records["raw"].size_bytes > limits.input_bytes:
        raise RawLimitError("input_bytes limit exceeded")
    if sum(item.size_bytes for role, item in records.items() if role != "raw") > step_log_bytes:
        raise RawLimitError("step_log_bytes limit exceeded")
    for role, name in CAPTURED_NAMES.items():
        path = parser_file_in(directory, name)
        if role not in records:
            if path.exists() or path.is_symlink():
                raise RawDecodeError("Captured companion absence changed")
            continue
        item = records[role]
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise RawDecodeError("Captured inputs must be regular nonsymlink files")
        if info.st_size != item.size_bytes:
            raise RawDecodeError("Captured input size changed")
        digest = hashlib.sha256()
        total = 0
        with path.open("rb") as handle:
            while block := handle.read(min(_CHUNK_BYTES, item.size_bytes - total + 1)):
                total += len(block)
                if total > item.size_bytes:
                    raise RawDecodeError("Captured input size changed")
                digest.update(block)
        if total != item.size_bytes:
            raise RawDecodeError("Captured input size changed")
        if digest.hexdigest() != item.sha256:
            raise RawDecodeError("Captured input digest changed")
    return records


def _step_log(
    record: CapturedFile | None,
    directory: Path,
    header: RawHeader,
    limits: RawLimits,
    byte_limit: int,
    row_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "present": record is not None,
        "size_bytes": record.size_bytes if record else None,
        "sha256": record.sha256 if record else None,
        "encoding": None,
        "status": "absent",
        "rows": [],
    }
    if record is None:
        return result
    with parser_file_in(directory, record.name).open("rb") as handle:
        data = handle.read(byte_limit + 1)
    if len(data) != record.size_bytes:
        raise RawDecodeError("Captured step log size changed")
    try:
        # Never with a replacement character: one must not decide whether a
        # step marker exists.
        text, encoding = decode_spice_bytes_strictly(data)
    except UnicodeError as exc:
        raise RawDecodeError("Invalid captured step log encoding") from exc
    result["encoding"] = encoding
    result["status"] = "parsed" if header.plots[0].dialect == "ltspice" else "unsupported"
    if result["status"] == "unsupported":
        return result
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(io.StringIO(text), 1):
        original = line.rstrip("\r\n")
        step = read_step_line(original)
        if step is None:
            continue
        if len(encode_spice_text(original, encoding)) > limits.line_bytes:
            raise RawLimitError("line_bytes limit exceeded in step log row")
        if len(rows) == row_limit:
            raise RawLimitError("step_rows limit exceeded in companion log")
        parameters = [
            {
                "name": parameter.name,
                "token": parameter.token,
                "value": (
                    parameter.number
                    if parameter.number is not None and math.isfinite(parameter.number)
                    else None
                ),
            }
            for parameter in step.parameters
        ]
        if len({p["name"].casefold() for p in parameters}) != len(parameters):
            raise RawDecodeError(f"Duplicate step parameter in companion log line {line_number}")
        rows.append(
            {
                "ordinal": len(rows),
                "line_number": line_number,
                "text": original,
                "parameters": parameters,
            }
        )
    result["rows"] = rows
    return result


def _prepared_plot(
    path: Path, directory: Path, header: RawPlotHeader
) -> tuple[Path, int, int, int, str]:
    fields = {key.lower(): value for key, value in header.fields}
    if any(key.startswith(".") for key in fields):
        raise RawDecodeError("RAW alias/parameter directives are not supported")
    with path.open("rb") as handle:
        handle.seek(header.header_offset)
        data = handle.read(header.payload_offset - header.header_offset)
    text = data.decode(header.encoding, errors="strict")
    bom_bytes = (3 if header.encoding == "utf-8" else 2) if text.startswith("\ufeff") else 0
    needs_normalization = (
        header.encoding == "utf-16-be"
        or (header.storage == "values" and header.encoding != "utf-8")
        or "Variables:" not in text.splitlines()
        or fields["no. variables"] != str(header.variable_count)
        or fields["no. points"] != str(header.point_count)
    )
    if not needs_normalization:
        return (
            path,
            header.header_offset + bom_bytes,
            header.payload_offset,
            header.payload_end,
            header.encoding.replace("-", "_"),
        )
    normalized = parser_file_in(directory, f"p{header.index}_decode.raw")
    lines = []
    for key, value in header.fields:
        if key.lower() == "no. variables":
            value = str(header.variable_count)
        elif key.lower() == "no. points":
            value = str(header.point_count)
        lines.append(f"{key.title()}: {value}\n")
    lines.append("Variables:\n")
    for variable in header.variables:
        row = [str(variable.index), variable.name, variable.declared_type, *variable.attributes]
        lines.append("\t" + "\t".join(row) + "\n")
    lines.append("Binary:\n" if header.storage == "binary" else "Values:\n")
    normalized_header = "".join(lines).encode("utf-8")
    # UTF-16 -> UTF-8 expands by at most three bytes per source byte;
    # canonical counts shrink. Bound scratch from the validated source span.
    maximum = 3 * (header.payload_end - header.header_offset)
    written = len(normalized_header)
    with normalized.open("xb") as writer, path.open("rb") as reader:
        writer.write(normalized_header)
        reader.seek(header.payload_offset)
        remaining = header.payload_bytes
        decoder = codecs.getincrementaldecoder(header.encoding)(errors="strict")
        while remaining:
            block = reader.read(min(_CHUNK_BYTES, remaining))
            if not block:
                raise RawDecodeError("Captured RAW truncated during normalization")
            remaining -= len(block)
            converted = (
                decoder.decode(block, final=remaining == 0).encode("utf-8")
                if header.storage == "values"
                else block
            )
            written += len(converted)
            if written > maximum:
                raise RawDecodeError("Normalized RAW exceeded its bounded source expansion")
            writer.write(converted)
    return normalized, 0, len(normalized_header), written, "utf_8"


def _validated_step_starts(
    header: RawPlotHeader, analysis: Analysis, data: np.ndarray, row_limit: int
) -> list[int] | None:
    """Admit the native restart convention only for unambiguous ordered runs."""
    if not np.isfinite(data[0]):
        return None
    starts = []
    for offset in range(0, len(data), _CHUNK_BYTES):
        matches = np.flatnonzero(data[offset : offset + _CHUNK_BYTES] == data[0])
        if len(starts) + len(matches) > row_limit:
            raise RawLimitError("step_rows limit exceeded in RAW ranges")
        starts.extend(offset + int(index) for index in matches)
    ends = [*starts[1:], header.point_count]
    direction = 0 if analysis == "dc" else 1
    time_signs = analysis == "transient"
    for start, end in zip(starts, ends, strict=True):
        # A singleton cannot distinguish a real step from a repeated start
        # coordinate. Refuse instead of assigning the plateau to a log row.
        if end - start < 2:
            return None
        previous = None
        for offset in range(start, end, _CHUNK_BYTES):
            block = data[offset : min(offset + _CHUNK_BYTES, end)]
            if np.iscomplexobj(block) and np.any(np.imag(block) != 0):
                return None
            coordinate = np.real(block)
            if time_signs:
                # Validate native LT time ordering on a bounded temporary;
                # retain stored signs and Offset unchanged in output arrays.
                coordinate = np.abs(coordinate)
            if not np.all(np.isfinite(coordinate)):
                return None
            if direction == 0:
                if coordinate[1] == coordinate[0]:
                    return None
                direction = 1 if coordinate[1] > coordinate[0] else -1
            if previous is not None:
                if direction > 0 and coordinate[0] <= previous:
                    return None
                if direction < 0 and coordinate[0] >= previous:
                    return None
            if direction > 0:
                ordered = coordinate[1:] > coordinate[:-1]
            else:
                ordered = coordinate[1:] < coordinate[:-1]
            if not np.all(ordered):
                return None
            previous = coordinate[-1]
    return starts


def _step_facts(
    header: RawPlotHeader,
    data: np.ndarray,
    has_axis: bool,
    log: dict[str, Any],
    plot_count: int,
    row_limit: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "plot_index": header.index,
        "decoder_has_axis": has_axis,
        "data_convention": "stored",
        "step_status": "unstepped",
        "step_ranges": [
            {"step_index": 0, "offset": 0, "length": header.point_count, "log_row": None}
        ],
    }
    if not header.stepped:
        return result
    result["step_status"] = "unresolved"
    result["step_ranges"] = None
    kind = stepped_kind(header)
    if kind is not None and kind.stepping == "point_a_step":
        # The raw's own step values are the record (result_kinds).
        result["step_status"] = "matched"
        result["step_ranges"] = [
            {"step_index": index, "offset": index, "length": 1, "log_row": None}
            for index in range(header.point_count)
        ]
        result["step_parameters"] = leading_parameters(header)
        return result
    if kind is not STEPPED_SWEEP or not has_axis:
        if header.point_count == 1 and log["status"] == "parsed" and len(log["rows"]) > 1:
            result["step_status"] = "mismatch"
        return result
    starts = _validated_step_starts(header, header_analysis(header), data, row_limit)
    if starts is None or log["status"] != "parsed" or not log["rows"] or plot_count != 1:
        return result
    for row in log["rows"]:
        step = read_step_line(row["text"])
        if step is None or not step.only_parameters:
            return result
    ends = [*starts[1:], header.point_count]
    ranges = [
        {"step_index": index, "offset": start, "length": end - start, "log_row": None}
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]
    result["step_ranges"] = ranges
    if len(log["rows"]) != len(ranges):
        result["step_status"] = "mismatch"
    else:
        result["step_status"] = "matched"
        for index, item in enumerate(ranges):
            item["log_row"] = index
    return result


def _write_array(
    data: np.ndarray, directory: Path, header: RawPlotHeader, index: int
) -> dict[str, Any]:
    dtype = _DTYPES[header.value_bytes[index]]
    expected = np.dtype(dtype)
    if (
        data.shape != (header.point_count,)
        or data.dtype.kind != expected.kind
        or data.dtype.itemsize != expected.itemsize
    ):
        raise RawDecodeError("Dependency trace shape or dtype contradicts RAW preflight")
    name = f"p{header.index}_t{index}.bin"
    path = parser_file_in(directory, name)
    with path.open("xb") as writer:
        data.astype(expected, copy=False).tofile(writer)
    size = header.point_count * expected.itemsize
    if path.stat().st_size != size:
        raise RawDecodeError("Numeric output byte count disagrees with RAW preflight")
    digest = hashlib.sha256()
    with path.open("rb") as reader:
        while block := reader.read(_CHUNK_BYTES):
            digest.update(block)
    return {
        "plot_index": header.index,
        "trace_index": index,
        "file": name,
        "dtype": dtype,
        "count": header.point_count,
        "byte_size": size,
        "sha256": digest.hexdigest(),
    }


def decode_raw(
    captured: CapturedInputs,
    directory: Path,
    *,
    limits: RawLimits,
    dialect: str | None = None,
    producing_dialect: str | None = None,
    step_log_bytes: int,
    step_rows: int,
    preflight: RawHeader | None = None,
    chosen: RawHeader | None = None,
    window: AxisWindow | None = None,
) -> dict[str, Any]:
    """Return header/arrays/plots/step_log only; no dependency object escapes.

    Invoke inside the finite worker. An optional preflight is an internal,
    trusted header for these same captured bytes, already checked under the
    current policy by the worker dispatcher; it avoids another ASCII scan.
    Input size/digest verification still runs. step_log_bytes bounds both
    captured logs together (console is verified but not interpreted); step_rows
    bounds log rows and RAW step ranges. The RAW line budget also bounds step
    rows. Scratch and output files stay for the caller to clean after exit.

    ``chosen`` is the preflight cut to the traces a read keeps
    (``raw_header.select_traces``): the dependency is still checked against
    the whole file, only those traces are asked of it and written, and the
    result is the cut header with their arrays. What the dependency holds
    while it reads them is ``numbers_held``.

    ``window`` keeps only the points of one plot's axis in a span
    (``_windowed``); that plot's header then counts the points kept, and the
    numeric limit holds what every plot keeps, checked before any array is
    written.
    """
    _positive_limit(step_log_bytes, "step_log_bytes")
    _positive_limit(step_rows, "step_rows")
    records = _verified_inputs(captured, directory, limits, step_log_bytes)
    path = parser_file_in(directory, records["raw"].name)
    header = (
        preflight
        if preflight is not None
        else preflight_raw(
            path, limits=limits, dialect=dialect, producing_dialect=producing_dialect
        )
    )
    if header.size_bytes != records["raw"].size_bytes:
        raise RawDecodeError("Supplied RAW preflight does not match captured size")
    log = _step_log(records.get("log"), directory, header, limits, step_log_bytes, step_rows)
    kept = header if chosen is None else chosen
    if len(kept.plots) != len(header.plots) or kept.size_bytes != header.size_bytes:
        raise RawDecodeError("Chosen traces do not belong to this RAW preflight")
    if window is not None and window.plot_index < len(kept.plots):
        _check_windowed(kept.plots[window.plot_index], window)
    arrays = []
    plots = []
    kept_plots = []
    for plot, keep in zip(header.plots, kept.plots, strict=True):
        source, start, header_end, end, encoding = _prepared_plot(path, directory, plot)
        with source.open("rb") as handle:
            view = _PlotView(handle, start, end, header_end=header_end, encoding=encoding)
            reader = _WorkerPlot(view, source, plot.index + 1, encoding, plot.dialect, False)  # pyright: ignore[reportArgumentType]
        names = [variable.name for variable in plot.variables]
        if (
            reader.nPoints != plot.point_count
            or reader.nVariables != plot.variable_count
            or reader.get_trace_names() != names
            or reader.aliases
        ):
            raise RawDecodeError("Dependency plot metadata contradicts RAW preflight")
        # Only the kept traces are asked for; spicelib reads the axis with
        # them, and a plot stored a point at a time whole (numbers_held).
        reader.read_trace_data([variable.name for variable in keep.variables])
        first = np.asarray(reader.get_trace(0).data)
        facts = _step_facts(keep, first, reader.has_axis, log, len(header.plots), step_rows)
        spans = None
        if window is not None and plot.index == window.plot_index:
            spans, windowed, facts = _windowed(keep, first, facts, window)
            numbers = kept.numeric_bytes - keep.numeric_bytes + windowed.numeric_bytes
            if numbers > limits.numeric_bytes:
                raise RawTooLargeError(numbers, limits.numeric_bytes, chosen=True)
            keep = windowed
        positions = {variable.name: variable.index for variable in plot.variables}
        for variable in keep.variables:
            trace = reader.get_trace(positions[variable.name])
            if trace.name != variable.name or trace.whattype != variable.declared_type:
                raise RawDecodeError("Dependency trace metadata contradicts RAW preflight")
            data = np.asarray(trace.data)
            if spans is not None:
                data = np.concatenate([data[lo:hi] for lo, hi in spans])
            arrays.append(_write_array(data, directory, keep, variable.index))
        plots.append(facts)
        kept_plots.append(keep)
    kept = replace(
        kept,
        numeric_bytes=sum(plot.numeric_bytes for plot in kept_plots),
        plots=tuple(kept_plots),
    )
    if sum(item["byte_size"] for item in arrays) != kept.numeric_bytes:
        raise RawDecodeError("Numeric output aggregate disagrees with RAW preflight")
    return {"header": asdict(kept), "arrays": arrays, "plots": plots, "step_log": log}


def _check_windowed(plot: RawPlotHeader, window: AxisWindow) -> None:
    """Refuse, before decoding, a window of a plot that has no axis to window."""
    if sampled_axis(header_analysis(plot), plot.first_type) is None:
        raise WindowRefusedError(
            f"window={window.text()} reads a span of a plot's axis, and plot "
            f"{plot.index} ({plot.plot_name}) has none: it holds one point "
            "(of each step), not a sweep. Read it without window."
        )


def _windowed(
    plot: RawPlotHeader, first: np.ndarray, facts: dict[str, Any], window: AxisWindow
) -> tuple[list[tuple[int, int]], RawPlotHeader, dict[str, Any]]:
    """The spans of each step of ``plot`` that ``window`` keeps
    (``window_span`` on the axis as a reader sees it), and the plot's header
    and step facts for the points kept.

    A step that keeps no point is refused, and so is an axis that decreases
    within a step, where no span is one stretch of it.
    """
    coordinate: Callable[[np.ndarray], np.ndarray] | None = None
    if header_analysis(plot) == "transient":
        offset = transient_time_offset(plot)
        coordinate = functools.partial(deck_time, offset=offset)

    ranges = facts["step_ranges"] or [
        {"step_index": 0, "offset": 0, "length": plot.point_count, "log_row": None}
    ]
    spans: list[tuple[int, int]] = []
    kept_ranges: list[dict[str, Any]] = []
    for part in ranges:
        begin = part["offset"]
        values = first[begin : begin + part["length"]]
        try:
            lo, hi = window_span(values, window.start, window.end, coordinate=coordinate)
        except UnorderedAxisError as exc:
            raise WindowRefusedError(
                f"window={window.text()} keeps a span of plot {plot.index}'s axis, and the "
                "axis decreases within a step (a sweep that runs downwards, or steps whose "
                "boundaries were not found), so no span of it is one stretch. Read it "
                "without window."
            ) from exc
        if hi <= lo:
            ends = values[[0, -1]] if coordinate is None else coordinate(values[[0, -1]])
            where = "" if facts["step_ranges"] is None else f"step {part['step_index']}'s "
            raise WindowRefusedError(
                f"window={window.text()} keeps no point of plot {plot.index}: "
                f"{where}axis runs from {float(np.real(ends[0])):g} to "
                f"{float(np.real(ends[1])):g}. Choose a window that meets it."
            )
        kept_ranges.append(
            {**part, "offset": sum(end - start for start, end in spans), "length": hi - lo}
        )
        spans.append((begin + lo, begin + hi))
    points = sum(end - start for start, end in spans)
    windowed = replace(plot, point_count=points, numeric_bytes=points * sum(plot.value_bytes))
    kept_facts = {**facts, "step_ranges": kept_ranges if facts["step_ranges"] else None}
    return spans, windowed, kept_facts
