"""Plain log facts decoded only from full bounded captures in owned scratch.

The caller supplies process memory/deadline containment and owns directory
cleanup. This operation neither reads live sources nor publishes service data.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ltspice_mcp.lib import log_parser
from ltspice_mcp.lib.encoding import decode_spice_bytes_strictly
from ltspice_mcp.lib.log_types import LogDecodeError as LogDecodeError
from ltspice_mcp.lib.log_types import LogLimitError as LogLimitError
from ltspice_mcp.lib.log_types import LogLimits as LogLimits
from ltspice_mcp.lib.log_types import log_section as _section
from ltspice_mcp.lib.native_log_tables import decode_native_tables
from ltspice_mcp.lib.parser_capture import CapturedInputs
from ltspice_mcp.lib.store import parser_file_in

_NAMES = {"raw": "input.raw", "log": "input.log", "console": "input.exe.log"}
_SECTIONS = (
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


def _read_captures(captured: CapturedInputs, directory: Path, limits: LogLimits) -> dict[str, str]:
    roles = [item.role for item in captured.files] + list(captured.absent)
    if len(roles) != 3 or set(roles) != set(_NAMES):
        raise LogDecodeError("Each captured role must occur exactly once, present or absent")
    records = {item.role: item for item in captured.files}
    for item in captured.files:
        if (
            item.name != _NAMES[item.role]
            or type(item.size_bytes) is not int
            or item.size_bytes < 0
        ):
            raise LogDecodeError("Invalid captured log filename or size")
        if not re.fullmatch(r"[0-9a-f]{64}", item.sha256):
            raise LogDecodeError("Invalid captured log digest")
    logs = [item for item in captured.files if item.role != "raw"]
    if sum(item.size_bytes for item in logs) > limits.log_bytes:
        raise LogLimitError("log_bytes limit exceeded")
    # Existing helpers have a head-only cap. Refuse before calling them rather
    # than label their truncated read complete when a caller permits more.
    if any(item.size_bytes > log_parser.log_read_limit() for item in logs):
        raise LogLimitError("Shared log reader byte cap exceeded")
    result = {}
    line_count = 0
    for role, name in _NAMES.items():
        if role == "raw":
            continue
        path = parser_file_in(directory, name)
        item = records.get(role)
        if item is None:
            if path.exists() or path.is_symlink():
                raise LogDecodeError("Captured log absence changed")
            continue
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != item.size_bytes:
            raise LogDecodeError("Captured log must remain a regular file of its recorded size")
        with path.open("rb") as handle:
            data = handle.read(item.size_bytes + 1)
        if len(data) != item.size_bytes or hashlib.sha256(data).hexdigest() != item.sha256:
            raise LogDecodeError("Captured log digest or size changed")
        try:
            text = decode_spice_bytes_strictly(data)[0]
        except UnicodeError as exc:
            raise LogDecodeError("Invalid captured log encoding") from exc
        if "\x00" in text:
            raise LogDecodeError("Captured log contains binary NUL characters")
        for line in io.StringIO(text):
            line_count += 1
            if line_count > limits.lines:
                raise LogLimitError("lines limit exceeded")
            if len(line.encode("utf-8")) > limits.line_bytes:
                raise LogLimitError("line_bytes limit exceeded (decoded UTF-8)")
        result[role] = text
    return result


def _fourier(reader: Any, log_path: Path, text: str) -> list[dict[str, Any]]:
    # The shared reader retries NaN/Inf distortion lines as 0.0%. Restore those
    # literal source facts on the owned reader before numeric extraction; the
    # section's plain conversion emits null with an explicit nonfinite count.
    originals: dict[str, list[dict[str, float]]] = {}
    current = None
    for line in io.StringIO(text):
        if line.startswith("Fourier components of"):
            signal = line.rstrip("\r\n").split(" of ")[-1].strip()
            current = {}
            originals.setdefault(signal, []).append(current)
        elif current is not None:
            # LTspice 24 and later print each figure on a line of its own
            # ("Total Harmonic Distortion:   13.617501%"). LTspice XVII prints
            # one line with a second figure in parentheses
            # ("Total Harmonic Distortion: 13.603246%(13.610258%)"); the first
            # is the one read, as the shared reader reads it.
            match = re.match(r"(Total|Partial) Harmonic Distortion:\s*([^\s%(]+)", line)
            if match:
                current["thd" if match[1] == "Total" else "phd"] = float(match[2])
    for signal, blocks in originals.items():
        decoded = reader.fourier.get(signal, [])
        if len(decoded) != len(blocks):
            raise LogDecodeError("Fourier blocks were not completely decoded")
        for parsed, block in zip(decoded, blocks, strict=True):
            for key, value in block.items():
                setattr(parsed, key, value)
    result = log_parser.parse_fourier_data(log_path, reader=reader, strict=True)
    # PHD is a printed fact, retained to account for sanitized nonfinite PHD.
    for entry, parsed in zip(
        result, [block for group in reader.fourier.values() for block in group], strict=True
    ):
        entry["phd"] = float(parsed.phd) if parsed.phd is not None else None
    return result


def _temperatures(text: str) -> dict[str, float | None]:
    ambient, nominal = log_parser.parse_temperatures(text=text)
    result = {"temp_c": ambient, "tnom_c": nominal}
    seen = set()
    for line in io.StringIO(text):
        if not re.match(r"^\s*(?:temp\s*=|tnom\s*=|Doing analysis at\b)", line, re.I):
            continue
        for match in re.finditer(r"\b(temp|tnom)\s*=\s*([^\s]+)", line, re.I):
            key = "temp_c" if match[1].lower() == "temp" else "tnom_c"
            if key not in seen:
                result[key] = float(match[2])
                seen.add(key)
    return result


def _steps(text: str) -> list[dict[str, float]]:
    markers = 0
    for line in io.StringIO(text):
        match = log_parser._RE_STEP_LINE.match(line.strip())  # pyright: ignore[reportPrivateUsage]
        if match is None:
            continue
        markers += 1
        seen: set[str] = set()
        for parameter in log_parser._RE_STEP_KV.finditer(match[1]):  # pyright: ignore[reportPrivateUsage]
            name = parameter[1].casefold()
            if name in seen:
                raise LogDecodeError(f"Duplicate step parameter {parameter[1]!r}")
            seen.add(name)
            try:
                float(parameter[2])
            except ValueError as exc:
                raise LogDecodeError("Step parameter is not numeric") from exc
    rows = log_parser.parse_step_iterations(text=text)
    if len(rows) != markers:
        raise LogDecodeError("Step rows were not completely decoded")
    return rows


def _check_device_rows(text: str) -> None:
    inside, devices = False, 0
    for line in io.StringIO(text):
        if line.startswith("Semiconductor Device Operating Points:"):
            inside = True
            continue
        if not inside:
            continue
        if re.match(r"^\s*--- .* ---\s*$", line):
            devices = 0
        tokens = line.split()
        if not tokens:
            # A blank line ends a group of devices. LTspice XVII goes on to
            # "Date:" and "Total elapsed time:" after the last one, which are
            # labelled lines of the log and not rows of the block.
            devices = 0
            continue
        if tokens[0] == "Name:":
            devices = len(tokens) - 1
        elif (
            devices
            and (tokens[0].endswith(":") or tokens[0] == "Gmb")
            and len(tokens) != devices + 1
        ):
            raise LogDecodeError("Device operating-point row has missing columns")


def _error_context(log_path: Path) -> str:
    excerpt = log_parser.extract_error_context(log_path, max_lines=20)
    if excerpt == "(Log file not found)" or excerpt.startswith("(Error reading log file:"):
        raise LogDecodeError(excerpt)
    return excerpt


def decode_logs(captured: CapturedInputs, directory: Path, *, limits: LogLimits) -> dict[str, Any]:
    """Decode general log facts; input/metadata excess refuses, never truncates.

    Line bytes are measured in normalized UTF-8 including the newline. The
    section_entries budget counts container nodes and scalar values. The caller
    must run this synchronous operation in its killable parser worker.
    """
    text = _read_captures(captured, directory, limits)
    log_path = parser_file_in(directory, "input.log")
    body = text.get("log", "")
    sections = {}
    sections["error_context"] = (
        _section(lambda: _error_context(log_path), limits)
        if "log" in text
        else _section(lambda: None, limits, present=False)
    )
    sections["native_tables"] = (
        decode_native_tables(body, limits=limits)
        if "log" in text
        else _section(lambda: None, limits, present=False)
    )
    sections["diagnostics"] = _section(
        lambda: log_parser.extract_log_diagnostics(log_path), limits, present=bool(text)
    )
    _, attempts = log_parser.scan_op_step_log(text=body)
    sections["steps"] = _section(
        lambda: _steps(body), limits, present=bool(re.search(r"^\s*\.step\s+", body, re.M | re.I))
    )
    sections["op_iterations"] = _section(
        lambda: {"attempts": attempts, "succeeded": log_parser.count_op_iterations(text=body)},
        limits,
        present=bool(attempts),
    )
    sections["temperatures"] = _section(lambda: _temperatures(body), limits)
    if (
        sections["temperatures"]["status"] == "parsed"
        and all(value is None for value in sections["temperatures"]["value"].values())
        and not sections["temperatures"]["nonfinite_count"]
    ):
        sections["temperatures"]["status"] = "absent"
    device_present = "Semiconductor Device Operating Points:" in body

    def device_op() -> dict[str, float]:
        if not device_present:
            return {}
        _check_device_rows(body)
        values = log_parser.read_device_op_points(log_path, scratch_dir=directory, strict=True)
        if not values:
            raise LogDecodeError("Device operating-point block was not decoded")
        return values

    sections["device_op"] = _section(device_op, limits, present=device_present)
    diagnostics = sections["diagnostics"]["value"] or {}
    if "log" not in text:
        for name in ("measurements", "fourier"):
            sections[name] = _section(lambda: None, limits, present=False)
    elif diagnostics.get("errors") and not log_parser.has_circuit_line(body):
        # A run the simulator refused before it began writes no "Circuit:" line
        # and no measurement or Fourier block. The measurements are the empty
        # table a log with no .meas gives, carrying the reason the run failed.
        sections["measurements"] = _section(
            lambda: log_parser.empty_measurements(diagnostics), limits, present=False
        )
        sections["fourier"] = _section(lambda: [], limits, present=False)
    else:
        try:
            with log_parser.normalized_log(body, directory) as normalized:
                reader = log_parser.make_log_reader(normalized, scratch_dir=directory)

            def measurements() -> Any:
                for name in reader.get_measure_names():
                    if name.lower() == "circuit":
                        continue
                    for value in reader.dataset.get(name.lower(), []):
                        if (
                            value is not None
                            and not isinstance(value, complex)
                            and str(value).upper() != "FAILED"
                        ):
                            try:
                                float(value)
                            except (ValueError, TypeError) as exc:
                                raise LogDecodeError(
                                    f"Measurement {name!r} is not numeric"
                                ) from exc
                value = log_parser.parse_measurements(log_path, reader=reader)
                named = re.findall(r"^Measurement:\s*(\S+)", body, re.M)
                if any(name.lower() not in value["measurements"] for name in named):
                    raise LogDecodeError("Measurement blocks were not completely decoded")
                return value

            sections["measurements"] = _section(measurements, limits)
            if (
                sections["measurements"]["status"] == "parsed"
                and not sections["measurements"]["value"]["measurements"]
            ):
                sections["measurements"]["status"] = "absent"
            sections["fourier"] = _section(
                lambda: _fourier(reader, log_path, body), limits, present=bool(reader.fourier)
            )
        except LogLimitError:
            raise
        except Exception as exc:
            for name in ("measurements", "fourier"):
                sections[name] = {
                    "status": "error",
                    "value": None,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                    "nonfinite_count": 0,
                }
    result = {
        "version": 1,
        "capture_facts": asdict(captured),
        "scan": {
            "complete": True,
            "capturedbytes": sum(item.size_bytes for item in captured.files if item.role != "raw"),
        },
        **{name: sections[name] for name in _SECTIONS},
    }
    # Streaming byte accounting prevents a second oversized result allocation.
    size = 0
    for chunk in json.JSONEncoder(allow_nan=False, separators=(",", ":")).iterencode(result):
        size += len(chunk.encode("utf-8"))
        if size > limits.metadata_bytes:
            raise LogLimitError("metadata_bytes limit exceeded")
    return json.loads(json.dumps(result, allow_nan=False))
