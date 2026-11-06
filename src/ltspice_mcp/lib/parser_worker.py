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
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ltspice_mcp.lib.log_types import LogLimits
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs, parser_cache_key
from ltspice_mcp.lib.raw_header import (
    RawHeader,
    RawLimits,
    RawTooLargeError,
    RawTooLargeToDecodeError,
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


@dataclass(frozen=True)
class _Chosen:
    """What a selective read keeps (its signals' traces, a window of its
    plot's axis, or both), and how much it may decode."""

    signals: list[str] | None
    window: tuple[float | None, float | None] | None
    plot_index: int
    decode_bytes: int


def _chosen_request(value: Any) -> _Chosen | None:
    """The request's choice of signals and window, or None for a whole read."""
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {
        "signals",
        "window",
        "plot_index",
        "decode_bytes",
    }:
        raise ValueError(
            "Parser chosen read must name signals, window, plot_index and decode_bytes"
        )
    signals, window = value["signals"], value["window"]
    if signals is None and window is None:
        raise ValueError("Parser chosen read must choose signals or a window")
    if signals is not None and (
        not isinstance(signals, list)
        or not 0 < len(signals) <= 64
        or any(not isinstance(item, str) or not 0 < len(item) <= 256 for item in signals)
    ):
        raise ValueError("Parser signals must be 1 to 64 names of up to 256 characters")
    plot_index = value["plot_index"]
    if type(plot_index) is not int or plot_index < 0:
        raise ValueError("Parser plot_index must be a nonnegative integer")
    decode_bytes = _positive(value["decode_bytes"], "decode_bytes")
    return _Chosen(signals, _window(window), plot_index, decode_bytes)


def _window(value: Any) -> tuple[float | None, float | None] | None:
    """A window ``[start, end]`` of finite bounds, either one open, or None."""
    if value is None:
        return None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or value == [None, None]
        or any(
            bound is not None and (isinstance(bound, bool) or not isinstance(bound, int | float))
            for bound in value
        )
        or any(bound is not None and not math.isfinite(bound) for bound in value)
    ):
        raise ValueError("Parser window must be [start, end] of finite numbers, one open at most")
    start, end = value
    if start is not None and end is not None and start >= end:
        raise ValueError("Parser window must start before it ends")
    return (None if start is None else float(start), None if end is None else float(end))


def parse_request(request: dict[str, Any], directory: Path) -> None:
    """Capture and decode the fixed operation, writing one bounded result file."""
    expected = {
        "version",
        "op",
        "sources",
        "dialect",
        "producing_dialect",
        "limits",
        "chosen",
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
    chosen = _chosen_request(request["chosen"])
    if chosen is not None and not require_raw:
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
            # A selective read holds the numeric budget to what it keeps and
            # what it decodes (_cut, decode_raw), not to the whole file. A
            # file's numbers take no more than four bytes for each byte it
            # stores them in: as many in binary, fewer in text, which spends
            # at least four characters on a value it keeps in sixteen bytes;
            # and each point takes at least a byte. What is kept still holds
            # every plot's axis, at eight bytes a point or more, so no more
            # points than the budget's own limit on them.
            limits=raw_limits
            if chosen is None
            else replace(
                raw_limits,
                numeric_bytes=4 * raw_limits.input_bytes,
                points=raw_limits.input_bytes,
            ),
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
        chosen=None if chosen is None else request["chosen"],
    )
    result: dict[str, Any] = {"version": 1, "status": "cached", "cache_key": key}
    if key not in keys:
        from ltspice_mcp.lib.log_decode import decode_logs

        result["logs"] = decode_logs(captured, directory, limits=log_limits)
        result["raw"] = None
        if require_raw:
            from ltspice_mcp.lib.raw_decode import AxisWindow, decode_raw, numbers_held

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
                chosen=None if chosen is None else _cut(header, chosen, raw_limits, numbers_held),
                window=None
                if chosen is None or chosen.window is None
                else AxisWindow(chosen.plot_index, *chosen.window),
            )
        result["status"] = "ok"
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    if len(payload) > metadata_bytes:
        raise ValueError("Parser result metadata exceeds its byte limit")
    parser_file_in(directory, "result.json").write_bytes(payload)


def _cut(
    header: RawHeader,
    chosen: _Chosen,
    limits: RawLimits,
    held: Callable[[RawHeader, RawHeader], int],
) -> RawHeader:
    """``header`` cut to the traces ``chosen`` reads (``trace_names.chosen_traces``),
    every trace when it chooses none, with no more numbers held while decoding
    (``held``) than ``chosen`` allows. Without a window it keeps no more than
    the numeric budget a whole-file read is held to; with one, ``decode_raw``
    holds the points the window keeps to it."""
    from ltspice_mcp.lib.trace_names import chosen_traces

    cut = (
        header
        if chosen.signals is None
        else select_traces(
            header, chosen_traces(header.plots, chosen.signals, plot_index=chosen.plot_index)
        )
    )
    if chosen.window is None and cut.numeric_bytes > limits.numeric_bytes:
        raise RawTooLargeError(cut.numeric_bytes, limits.numeric_bytes, chosen=True)
    decoded = held(header, cut)
    if decoded > chosen.decode_bytes:
        raise RawTooLargeToDecodeError(decoded, chosen.decode_bytes)
    return cut
