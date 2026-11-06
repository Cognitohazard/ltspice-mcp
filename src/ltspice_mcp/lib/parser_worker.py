"""Fixed raw/log parser operations, invoked only after process containment.

The bootstrap sets memory and lifetime bounds before importing this module.
All untrusted source reads and third-party decoding stay in that process.

The decoders are imported only on the branch that decodes. Most calls find
their content already parsed and only capture and hash it, and the decoders'
imports (spicelib, which brings NumPy) are most of what a parser process
costs to start.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.log_types import LogLimits
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs, parser_cache_key
from ltspice_mcp.lib.raw_header import (
    RawHeader,
    RawLimits,
    RawTooLargeError,
    preflight_raw,
    select_traces,
)
from ltspice_mcp.lib.store import is_sha256, parser_file_in


def _positive(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a finite positive integer")
    return value


def _source_files(value: Any) -> SourceFiles:
    if not isinstance(value, dict) or set(value) != {"raw", "log", "console"}:
        raise ValueError("Parser sources must name raw, log and console roles")
    paths = {}
    for role, name in value.items():
        if name is None:
            paths[role] = None
        elif isinstance(name, str) and Path(name).is_absolute():
            paths[role] = Path(name)
        else:
            raise ValueError("Parser sources must be admitted absolute paths or absent")
    return SourceFiles(**paths)


def _signals(value: Any) -> list[str] | None:
    """The signals a read keeps the traces of, or None for every trace."""
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or not 0 < len(value) <= 64
        or any(not isinstance(item, str) or not 0 < len(item) <= 256 for item in value)
    ):
        raise ValueError("Parser signals must be 1 to 64 names of up to 256 characters")
    return value


def parse_request(request: dict[str, Any], directory: Path) -> None:
    """Capture and decode the fixed operation, writing one bounded result file."""
    expected = {
        "version",
        "op",
        "sources",
        "dialect",
        "producing_dialect",
        "limits",
        "signals",
        "existing_cache_keys",
    }
    if set(request) != expected or type(request["version"]) is not int or request["version"] != 1:
        raise ValueError("Unsupported parser request shape")
    if request["op"] not in {"load_raw", "load_logs"}:
        raise ValueError("Unsupported parser operation")
    require_raw = request["op"] == "load_raw"
    sources = _source_files(request["sources"])
    if require_raw and sources.raw is None:
        raise ValueError("RAW parsing requires a raw source")
    limits = request["limits"]
    if not isinstance(limits, dict) or set(limits) != {
        "raw",
        "log",
        "step_rows",
        "metadata_bytes",
    }:
        raise ValueError("Parser limits must be explicit and complete")
    raw_limits = RawLimits(**limits["raw"])
    signals = _signals(request["signals"])
    if signals is not None and not require_raw:
        raise ValueError("Only a RAW read chooses signals")
    log_limits = LogLimits(**limits["log"])
    step_rows = _positive(limits["step_rows"], "step_rows")
    metadata_bytes = _positive(limits["metadata_bytes"], "metadata_bytes")
    keys = request["existing_cache_keys"]
    if not isinstance(keys, list) or len(keys) > 32 or any(not is_sha256(key) for key in keys):
        raise ValueError("Parser cache keys must be bounded content identities")
    captured = capture_inputs(
        sources,
        directory,
        input_bytes=raw_limits.input_bytes,
        log_bytes=log_limits.log_bytes,
        require_raw=require_raw,
    )
    header = (
        preflight_raw(
            parser_file_in(directory, "input.raw"),
            # A read of chosen signals holds the numeric budget to what it
            # keeps (below), not to the whole file. A file's numbers take no
            # more than four bytes for each byte it stores them in: as many
            # in binary, fewer in text, which spends at least two characters
            # on a value it keeps in eight bytes.
            limits=raw_limits
            if signals is None
            else replace(raw_limits, numeric_bytes=4 * raw_limits.input_bytes),
            dialect=request["dialect"],
            producing_dialect=request["producing_dialect"],
        )
        if require_raw
        else None
    )
    key = parser_cache_key(
        captured,
        dialect=request["dialect"],
        producing_dialect=request["producing_dialect"],
        limits=limits,
        signals=signals,
    )
    result: dict[str, Any] = {"version": 1, "status": "cached", "cache_key": key}
    if key not in keys:
        from ltspice_mcp.lib.log_decode import decode_logs

        result["logs"] = decode_logs(captured, directory, limits=log_limits)
        result["raw"] = None
        if require_raw:
            from ltspice_mcp.lib.raw_decode import decode_raw

            assert header is not None  # preflight ran for every RAW read
            result["raw"] = decode_raw(
                captured,
                directory,
                limits=raw_limits,
                dialect=request["dialect"],
                producing_dialect=request["producing_dialect"],
                step_log_bytes=log_limits.log_bytes,
                step_rows=step_rows,
                preflight=header,
                chosen=None if signals is None else _chosen(header, signals, raw_limits),
            )
        result["status"] = "ok"
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > metadata_bytes:
        raise ValueError("Parser result metadata exceeds its byte limit")
    parser_file_in(directory, "result.json").write_bytes(payload)


def _chosen(header: RawHeader, signals: list[str], limits: RawLimits) -> RawHeader:
    """``header`` cut to the traces ``signals`` read (``trace_names.chosen_traces``),
    within the numeric budget a whole-file read is held to."""
    from ltspice_mcp.lib.simulator_build import is_cp1252_ltspice_build
    from ltspice_mcp.lib.trace_names import chosen_traces

    def xvii(plot: Any) -> bool:
        # As device_names.writes_xvii_names reads a decoded raw's Command.
        command = next((value for key, value in plot.fields if key.lower() == "command"), "")
        return is_cp1252_ltspice_build(command)

    chosen = select_traces(header, chosen_traces(header.plots, signals, xvii=xvii))
    if chosen.numeric_bytes > limits.numeric_bytes:
        raise RawTooLargeError(chosen.numeric_bytes, limits.numeric_bytes, chosen=True)
    return chosen
