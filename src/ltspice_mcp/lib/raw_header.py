"""Bounded, dependency-free preflight of a complete RAW snapshot.

This module validates every plot before a decoder can allocate trace arrays.
It preserves header facts without interpreting engineering units or axes.
The caller supplies finite budgets, an authorized immutable file, and any
known producing dialect. Capture, deadlines, worker memory, companion logs,
and dependency decoding belong to the caller. A successful preflight does
not make an inline third-party decoder safe.

Binary widths follow spicelib 1.5.1's PlotData trace construction: LTspice
uses an eight-byte first real variable and four-byte remaining variables
(all eight with ``double``); complex variables occupy sixteen bytes.
ngspice/Xyce use eight-byte real or sixteen-byte complex variables. QSPICE
uses eight bytes for the first complex variable and sixteen for the rest.
FastAccess changes ordering, not total length. LTspice's ordinary layouts
are checked against recorded fixtures, including stepped transient and AC.
Double, FastAccess, QSPICE and Xyce have source-defined synthetic coverage;
unrecognized flags, LTspice analysis layouts and Xyce text footers refuse.
"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass, fields
from pathlib import Path
from typing import BinaryIO


class RawHeaderError(ValueError):
    """The complete RAW input is malformed or unsupported."""


class RawLimitError(RawHeaderError):
    """A caller-supplied preflight budget was exceeded."""


class RawDialectError(RawHeaderError):
    """Dialect evidence is ambiguous or contradictory."""


class RawLayoutError(RawHeaderError):
    """A storage layout has no allowlisted byte formula."""


@dataclass(frozen=True)
class RawLimits:
    """Positive integer budgets; no production defaults or unlimited values.

    Input, header and numeric byte budgets cover the entire file. Header
    bytes include BOMs and line endings. Line bytes include line endings in
    headers and text payloads. Variables and points are per-plot maxima.
    Numeric bytes count all declared trace arrays at the decoder's scalar
    widths, including variable zero. They do not bound decoder temporaries,
    metadata objects, downstream dtype conversions, or process memory.
    """

    input_bytes: int
    header_bytes: int
    line_bytes: int
    plots: int
    variables: int
    points: int
    numeric_bytes: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field.name} must be a finite positive integer")


@dataclass(frozen=True)
class RawVariable:
    index: int
    name: str
    declared_type: str
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class RawPlotHeader:
    index: int
    header_offset: int
    encoding: str
    fields: tuple[tuple[str, str], ...]
    plot_name: str
    flags: tuple[str, ...]
    variable_count: int
    point_count: int
    variables: tuple[RawVariable, ...]
    dialect: str
    dialect_evidence: tuple[str, ...]
    storage: str
    storage_order: str
    value_bytes: tuple[int, ...]
    payload_offset: int
    payload_end: int
    payload_bytes: int
    numeric_bytes: int


@dataclass(frozen=True)
class RawHeader:
    size_bytes: int
    header_bytes: int
    numeric_bytes: int
    plots: tuple[RawPlotHeader, ...]


_DIALECTS = ("ltspice", "ngspice", "qspice", "xyce")
_WRITER = re.compile(r"(?<![a-z0-9_])(ltspice|ngspice|qspice|xyce)(?![a-z0-9_])", re.I)
_INTEGER = re.compile(r"[0-9]+\Z")
_NUMBER = re.compile(
    r"[+-]?(?:(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?|inf(?:inity)?|nan)\Z",
    re.I,
)
_FLAGS = {"real", "complex", "forward", "log", "linear", "stepped", "double", "fastaccess"}
_LT_REAL_PLOTS = {
    "transient analysis",
    "dc transfer characteristic",
    "operating point",
    "noise spectral density - (v/hz½ or a/hz½)",
}


class _Scanner:
    def __init__(self, handle: BinaryIO, size: int, limits: RawLimits) -> None:
        self.handle = handle
        self.size = size
        self.limits = limits
        self.plot_index = 0
        self.header_bytes = 0
        self.numeric_bytes = 0

    def error(self, cls: type[RawHeaderError], message: str) -> RawHeaderError:
        return cls(f"RAW plot {self.plot_index}, byte {self.handle.tell()}: {message}")

    def check_limit(self, name: str, value: int) -> None:
        if value > getattr(self.limits, name):
            raise self.error(RawLimitError, f"{name} limit exceeded ({value})")

    def encoding(self) -> str:
        start = self.handle.tell()
        prefix = self.handle.read(min(14, self.size - start))
        for bom, encoding in (
            (b"\xff\xfe", "utf-16-le"),
            (b"\xfe\xff", "utf-16-be"),
            (b"\xef\xbb\xbf", "utf-8"),
            (b"", "utf-16-le"),
            (b"", "utf-16-be"),
            (b"", "utf-8"),
        ):
            if prefix.startswith(bom + "Title:".encode(encoding)):
                self.handle.seek(start + len(bom))
                self.header_bytes += len(bom)
                self.check_limit("header_bytes", self.header_bytes)
                return encoding
        self.handle.seek(start)
        raise self.error(RawHeaderError, "invalid encoding or missing Title: at plot boundary")

    def line(self, encoding: str, *, header: bool = False) -> str:
        budget = self.limits.line_bytes
        if header:
            budget = min(budget, self.limits.header_bytes - self.header_bytes)
        if encoding == "utf-8":
            data = self.handle.readline(min(budget + 1, self.size - self.handle.tell()))
        else:
            # Read code units, not text-mode lines: UTF-16 LE's LF ends in
            # a NUL byte, and BE can contain LF bytes within another unit.
            data = bytearray()
            newline = "\n".encode(encoding)
            while len(data) <= budget:
                unit = self.handle.read(min(2, self.size - self.handle.tell()))
                data.extend(unit)
                if not unit or unit == newline:
                    break
        self.check_limit("line_bytes", len(data))
        if header:
            self.header_bytes += len(data)
            self.check_limit("header_bytes", self.header_bytes)
        if not data or not data.endswith("\n".encode(encoding)):
            raise self.error(RawHeaderError, "truncated line or missing line ending")
        try:
            text = data.decode(encoding, errors="strict").removesuffix("\n").removesuffix("\r")
        except UnicodeDecodeError as exc:
            raise self.error(RawHeaderError, "invalid header/text payload encoding") from exc
        if any(ord(char) < 32 and char != "\t" for char in text):
            raise self.error(RawHeaderError, "invalid control character in header/text payload")
        return text

    def integer(self, text: str, name: str, maximum: int, *, positive: bool = True) -> int:
        if not _INTEGER.fullmatch(text):
            raise self.error(RawHeaderError, f"invalid {name}: expected decimal integer")
        digits = text.lstrip("0") or "0"
        bound = str(maximum)
        if len(digits) > len(bound) or (len(digits) == len(bound) and digits > bound):
            raise self.error(RawLimitError, f"{name} limit exceeded")
        value = int(digits)
        if positive and value == 0:
            raise self.error(RawHeaderError, f"invalid {name}: expected positive count")
        return value

    def text_payload(self, encoding: str, points: int, widths: tuple[int, ...]) -> None:
        for point in range(points):
            for index, width in enumerate(widths):
                line = self.line(encoding).strip()
                while not line:
                    line = self.line(encoding).strip()
                if index == 0:
                    number, sep, line = line.partition("\t")
                    if (
                        not sep
                        or self.integer(number, "point index", points, positive=False) != point
                    ):
                        raise self.error(RawHeaderError, "invalid text payload point index")
                    line = line.strip()
                parts = line.split(",")
                expected = 2 if width == 16 else 1
                if len(parts) != expected or any(not _NUMBER.fullmatch(p.strip()) for p in parts):
                    raise self.error(
                        RawHeaderError,
                        f"invalid numeric text payload at point {point}, variable {index}",
                    )

    def skip_text_separators(self, encoding: str) -> None:
        separators = {char.encode(encoding) for char in " \t\r\n"}
        unit_bytes = 1 if encoding == "utf-8" else 2
        while self.handle.tell() < self.size:
            start = self.handle.tell()
            unit = self.handle.read(unit_bytes)
            self.handle.seek(start)
            # The next plot can use a different encoding. Only interpret
            # actual separator lines with the preceding payload's codec.
            if unit not in separators:
                return
            line = self.line(encoding)
            if line.strip():
                self.handle.seek(start)
                return


def _dialect_argument(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.lower() not in _DIALECTS:
        raise RawDialectError(f"invalid {name}; expected one of {', '.join(_DIALECTS)}")
    return value.lower()


def _resolve_dialect(
    scanner: _Scanner,
    fields: dict[str, str],
    variables: tuple[RawVariable, ...],
    encoding: str,
    explicit: str | None,
    producing: str | None,
    previous: str | None,
) -> tuple[str, tuple[str, ...]]:
    writers = set(_WRITER.findall(fields.get("command", "").lower()))
    if len(writers) > 1:
        raise scanner.error(RawDialectError, "ambiguous header Command: names multiple writers")
    writer = next(iter(writers), None)
    facts = [
        ("explicit_dialect", explicit),
        ("producing_dialect", producing),
        ("header:Command", writer),
    ]
    candidates = {value for _, value in facts if value is not None}
    if previous is not None:
        candidates.add(previous)
    if len(candidates) > 1:
        raise scanner.error(
            RawDialectError,
            "dialect conflict between explicit, producing, header or previous plot evidence",
        )
    if candidates:
        evidence = tuple(label for label, value in facts if value is not None)
        return next(iter(candidates)), evidence or ("previous_plot",)
    # Narrow compatibility profiles, labelled as inference rather than writer
    # declarations. Neither ASCII nor UTF-16 alone establishes a dialect.
    if "command" not in fields:
        flags = fields["flags"].lower().split()
        if encoding.startswith("utf-16") and "offset" in fields and "forward" in flags:
            return "ltspice", ("legacy:utf16-offset-forward",)
        if (
            encoding == "utf-8"
            and fields["plotname"] == "Noise Spectral Density Curves"
            and flags == ["real"]
            and variables[0].name == "frequency"
            and "grid=3" in variables[0].attributes
            and any(v.declared_type == "voltage-density" for v in variables[1:])
        ):
            return "ngspice", ("legacy:ngspice-noise-grid",)
    raise scanner.error(RawDialectError, "ambiguous dialect; supply explicit or producing dialect")


def _layout(
    scanner: _Scanner, dialect: str, fields: dict[str, str], storage: str, count: int
) -> tuple[tuple[int, ...], str]:
    flags = fields["flags"].lower().split()
    flag_set = set(flags)
    if len(flags) != len(flag_set) or len(flag_set & {"real", "complex"}) != 1:
        raise scanner.error(
            RawHeaderError,
            "invalid Flags: require exactly one of real/complex, without duplicates",
        )
    unknown = flag_set - _FLAGS
    if unknown:
        raise scanner.error(
            RawLayoutError, f"unsupported layout flags: {', '.join(sorted(unknown))}"
        )
    if "log" in flags and "linear" in flags:
        raise scanner.error(RawHeaderError, "invalid Flags: log and linear conflict")
    if "double" in flags and (dialect != "ltspice" or "complex" in flags):
        raise scanner.error(RawLayoutError, "unsupported double flag for this layout")
    if "fastaccess" in flags and (dialect != "ltspice" or storage != "binary"):
        raise scanner.error(RawLayoutError, "unsupported fastaccess flag for this layout")
    if "stepped" in flags and dialect != "ltspice":
        raise scanner.error(RawLayoutError, "unsupported stepped flag for this layout")
    plot = fields["plotname"].lower()
    if plot == "ac analysis" and "complex" not in flags:
        raise scanner.error(RawLayoutError, "AC Analysis requires a complex layout")
    if dialect == "ltspice" and storage == "binary":
        supported = plot == "ac analysis" if "complex" in flags else plot in _LT_REAL_PLOTS
        if not supported:
            raise scanner.error(
                RawLayoutError,
                "unsupported LTspice binary analysis layout; no validated byte formula",
            )
    if "complex" in flags:
        widths = ((8,) + (16,) * (count - 1)) if dialect == "qspice" else (16,) * count
    elif dialect == "ltspice" and "double" not in flags:
        widths = (8,) + (4,) * (count - 1)
    else:
        widths = (8,) * count
    return widths, "trace" if "fastaccess" in flags else "point"


def _read_plot(
    scanner: _Scanner, explicit: str | None, producing: str | None, previous: str | None
) -> RawPlotHeader:
    start = scanner.handle.tell()
    encoding = scanner.encoding()
    original_fields: list[tuple[str, str]] = []
    header: dict[str, str] = {}
    while True:
        line = scanner.line(encoding, header=True)
        key, sep, value = line.partition(":")
        key = key.strip()
        normalized = key.lower()
        if not sep or not key or normalized in header:
            raise scanner.error(RawHeaderError, "invalid or duplicate header field")
        if not header and key != "Title":
            raise scanner.error(RawHeaderError, "missing Title: at plot boundary")
        header[normalized] = value.strip()
        if normalized == "variables":
            if value.strip():
                raise scanner.error(
                    RawHeaderError, "Variables: must introduce indexed variable rows"
                )
            break
        original_fields.append((key, value.strip()))
    for required in ("title", "plotname", "flags", "no. variables", "no. points"):
        if required not in header or (required != "title" and not header[required]):
            raise scanner.error(RawHeaderError, f"missing or empty header field {required}")
    count = scanner.integer(
        header["no. variables"], "No. Variables / variables", scanner.limits.variables
    )
    points = scanner.integer(header["no. points"], "No. Points / points", scanner.limits.points)
    variables: list[RawVariable] = []
    names: set[str] = set()
    for index in range(count):
        parts = scanner.line(encoding, header=True).lstrip().split("\t")
        if len(parts) < 3 or not parts[1] or not parts[2]:
            raise scanner.error(RawHeaderError, f"invalid variable row {index}")
        if scanner.integer(parts[0], "variable index", count, positive=False) != index:
            raise scanner.error(RawHeaderError, f"inconsistent variable index at row {index}")
        if parts[1] in names:
            raise scanner.error(RawHeaderError, f"duplicate variable name at row {index}")
        names.add(parts[1])
        variables.append(RawVariable(index, parts[1], parts[2], tuple(parts[3:])))
    marker = scanner.line(encoding, header=True).lower()
    if marker not in ("binary:", "values:"):
        raise scanner.error(
            RawHeaderError, "variable count mismatch or missing Binary:/Values: marker"
        )
    storage = marker[:-1]
    resolved, evidence = _resolve_dialect(
        scanner, header, tuple(variables), encoding, explicit, producing, previous
    )
    widths, order = _layout(scanner, resolved, header, storage, count)
    numeric_bytes = points * sum(widths)
    scanner.numeric_bytes += numeric_bytes
    scanner.check_limit("numeric_bytes", scanner.numeric_bytes)
    payload_offset = scanner.handle.tell()
    if storage == "binary":
        end = payload_offset + numeric_bytes
        if end > scanner.size:
            raise scanner.error(
                RawHeaderError, "truncated binary payload; declared layout exceeds input bytes"
            )
        scanner.handle.seek(end)
    else:
        scanner.text_payload(encoding, points, widths)
    payload_end = scanner.handle.tell()
    return RawPlotHeader(
        index=scanner.plot_index,
        header_offset=start,
        encoding=encoding,
        fields=tuple(original_fields),
        plot_name=header["plotname"],
        flags=tuple(header["flags"].split()),
        variable_count=count,
        point_count=points,
        variables=tuple(variables),
        dialect=resolved,
        dialect_evidence=evidence,
        storage=storage,
        storage_order=order,
        value_bytes=widths,
        payload_offset=payload_offset,
        payload_end=payload_end,
        payload_bytes=payload_end - payload_offset,
        numeric_bytes=numeric_bytes,
    )


def preflight_raw(
    path: str | Path,
    *,
    limits: RawLimits,
    dialect: str | None = None,
    producing_dialect: str | None = None,
) -> RawHeader:
    """Validate all plots of an authorized complete snapshot, without decoding.

    Offsets are absolute bytes; payload_end is exclusive and excludes any
    inter-plot text blank lines. Binary payloads are skipped by an allowlisted
    formula, never scanned for metadata. Text values are checked one bounded
    line at a time, retaining no samples. Numeric budgets are checked before
    walking payloads. Dialect arguments and each plot's Command must agree;
    narrow legacy profiles are used only in the absence of declared evidence.

    Raises RawHeaderError (or its limit/dialect/layout subclasses) for invalid
    input. Filesystem errors remain OSError. Reading uses binary file handles
    on all platforms. A size/mtime change is refused, but same-size rewrite
    races require the caller's immutable capture; this is not a snapshotter.
    """
    explicit = _dialect_argument(dialect, "dialect")
    producing = _dialect_argument(producing_dialect, "producing_dialect")
    path = Path(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise RawHeaderError("RAW input must be a regular file")
    if before.st_size > limits.input_bytes:
        raise RawLimitError("input_bytes limit exceeded")
    with path.open("rb") as handle:
        initial = os.fstat(handle.fileno())
        if not stat.S_ISREG(initial.st_mode):
            raise RawHeaderError("RAW input must be a regular file")
        scanner = _Scanner(handle, initial.st_size, limits)
        scanner.check_limit("input_bytes", initial.st_size)
        plots: list[RawPlotHeader] = []
        while handle.tell() < initial.st_size:
            scanner.plot_index = len(plots)
            scanner.check_limit("plots", len(plots) + 1)
            plot = _read_plot(scanner, explicit, producing, plots[-1].dialect if plots else None)
            plots.append(plot)
            if plot.storage == "values":
                scanner.skip_text_separators(plot.encoding)
        if not plots:
            raise RawHeaderError("RAW input contains no complete plots")
        final = os.fstat(handle.fileno())
        if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
            raise RawHeaderError("RAW input changed during preflight; use an immutable snapshot")
    return RawHeader(initial.st_size, scanner.header_bytes, scanner.numeric_bytes, tuple(plots))
