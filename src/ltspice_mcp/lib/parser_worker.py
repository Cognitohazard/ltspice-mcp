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
import re
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.log_types import LogLimits
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs, parser_cache_key
from ltspice_mcp.lib.raw_header import RawLimits, preflight_raw
from ltspice_mcp.lib.store import parser_file_in

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


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


def parse_request(request: dict[str, Any], directory: Path) -> None:
    """Capture and decode the fixed operation, writing one bounded result file."""
    expected = {
        "version",
        "op",
        "sources",
        "dialect",
        "producing_dialect",
        "limits",
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
    log_limits = LogLimits(**limits["log"])
    step_rows = _positive(limits["step_rows"], "step_rows")
    metadata_bytes = _positive(limits["metadata_bytes"], "metadata_bytes")
    keys = request["existing_cache_keys"]
    if (
        not isinstance(keys, list)
        or len(keys) > 32
        or any(not isinstance(key, str) or _DIGEST.fullmatch(key) is None for key in keys)
    ):
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
            limits=raw_limits,
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
    )
    result: dict[str, Any] = {"version": 1, "status": "cached", "cache_key": key}
    if key not in keys:
        from ltspice_mcp.lib.log_decode import decode_logs

        result["logs"] = decode_logs(captured, directory, limits=log_limits)
        result["raw"] = None
        if require_raw:
            from ltspice_mcp.lib.raw_decode import decode_raw

            result["raw"] = decode_raw(
                captured,
                directory,
                limits=raw_limits,
                dialect=request["dialect"],
                producing_dialect=request["producing_dialect"],
                step_log_bytes=log_limits.log_bytes,
                step_rows=step_rows,
                preflight=header,
            )
        result["status"] = "ok"
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > metadata_bytes:
        raise ValueError("Parser result metadata exceeds its byte limit")
    parser_file_in(directory, "result.json").write_bytes(payload)
