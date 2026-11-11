"""Source admission, finite parser calls, and fully resident result caching.

All source content reads/hashes happen in the owned worker; path authorization
and source stamps are filesystem metadata work in the parent's off-loop thread.
A source whose files still carry the stat stamps recorded when their content
was parsed is served from the cache without a worker (``_source_stamps``,
``_settled``); any other read is captured and hashed in the worker. Only
validated numeric payloads are read by the parent, after confirmed tree exit.
Initial production limits are conservative: configured aggregate input bytes,
512 MiB stored numeric arrays, 2 GiB worker memory, and bounded
header/log/manifest records.
A corrected transient axis can add up to another 512 MiB in the parent. Cache
admission counts that storage; these component limits are not a total RSS cap.
"""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING
from uuid import uuid4

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError, ResultFileNotFoundError
from ltspice_mcp.lib.decoded_log import DecodedLog
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.file_stamp import ABSENT, file_stamp
from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
from ltspice_mcp.lib.parser_capture import unavailable_inputs
from ltspice_mcp.lib.parser_process import (
    JsonValue,
    ParserProcessError,
    ParserProcessLimits,
    run_parser_sync,
)
from ltspice_mcp.lib.parser_protocol import read_parsed_artifacts
from ltspice_mcp.lib.raw_header import RawLimits
from ltspice_mcp.lib.result_cache import ParserCleanupError
from ltspice_mcp.lib.store import is_sha256

if TYPE_CHECKING:
    from ltspice_mcp.lib.services import AnalysisSource
    from ltspice_mcp.state import SessionState

NUMERIC_BYTES = 512 * 1024 * 1024
LOG_BYTES = 16 * 1024 * 1024
STEP_ROWS = 4096
METADATA_BYTES = 16 * 1024 * 1024
PROCESS_LIMITS = ParserProcessLimits(
    memory_bytes=2 * 1024 * 1024 * 1024,
    request_bytes=64 * 1024,
    metadata_bytes=METADATA_BYTES,
    error_bytes=16 * 1024,
    cleanup_grace_s=3.0,
)
SIGNALS_BYTES = 20 * 1024
"""The most a read's chosen signals may take of the parser request, as JSON:
64 names of 256 plain characters, within the request's limit beside its paths.
A character outside ASCII takes up to six bytes there."""
DECODE_BYTES = PROCESS_LIMITS.memory_bytes // 2
"""The most numbers a read of chosen signals may have the parser hold while it
decodes them (``raw_decode.numbers_held``): half its memory, the rest left to
the interpreter and the arithmetic on the axis."""
_DIALECTS = {"ltspice", "ngspice", "qspice", "xyce"}
_ROLES = ("raw", "log", "console")
_SETTLE_NS = 100_000_000
"""How far in the past a source's last write must be before its stamp is
recorded: longer than any clock tick a filesystem stamps writes with."""
_COARSE_SETTLE_NS = 2_000_000_000
"""The same for a whole-second timestamp, which marks a filesystem with coarse
timestamps (FAT keeps two seconds)."""
_now_ns = time.time_ns
"""The wall clock the settle rule compares file timestamps against."""


def log_limits() -> dict[str, JsonValue]:
    """Plain LogLimits fields; dependency log decoding remains worker-only."""
    return {
        "log_bytes": LOG_BYTES,
        "line_bytes": 64 * 1024,
        "lines": 500_000,
        "section_entries": 250_000,
        "metadata_bytes": METADATA_BYTES,
    }


def raw_limits(max_raw_mb: int) -> RawLimits:
    if type(max_raw_mb) is not int or max_raw_mb <= 0:
        raise ResultError("Configured max_raw_mb must be a finite positive integer")
    return RawLimits(
        input_bytes=max_raw_mb * 1024 * 1024,
        header_bytes=2 * 1024 * 1024,
        line_bytes=64 * 1024,
        plots=64,
        variables=4096,
        points=NUMERIC_BYTES // 8,
        numeric_bytes=NUMERIC_BYTES,
    )


def _dialect(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value.strip().lower() not in _DIALECTS:
        raise ResultError("Unknown artifact dialect")
    return value.strip().lower()


def _check_active(deadline: float, cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise InterruptedError("Parser call was cancelled")
    if time.monotonic() >= deadline:
        raise AnalysisDeadlineExceeded("Artifact parsing exceeded the analysis deadline")


def load_raw_sync(
    source: AnalysisSource,
    state: SessionState,
    *,
    deadline: float,
    cancel: threading.Event | None = None,
) -> DecodedRaw:
    """Capture unless the source's stamps are unchanged; select after verification."""
    artifacts = load_artifacts_sync(
        source, state, deadline=deadline, cancel=cancel, require_raw=True
    )
    return select_raw(artifacts, source.plot_index)


def select_raw(artifacts: ParsedArtifacts, plot_index: int) -> DecodedRaw:
    """Select a plot from a validated whole-file cache entry."""
    raw = artifacts.raw
    if raw is None:
        raise ResultError("Captured artifacts do not contain resident RAW data")
    try:
        return raw if plot_index == raw.plot_index else raw.select_plot(plot_index)
    except IndexError as exc:
        raise ResultError(f"RAW plot_index {plot_index} is outside this artifact") from exc


def load_logs_sync(
    source: AnalysisSource,
    state: SessionState,
    *,
    deadline: float,
    cancel: threading.Event | None = None,
) -> DecodedLog:
    """Load complete log facts without requiring a valid RAW payload."""
    return load_artifacts_sync(
        source, state, deadline=deadline, cancel=cancel, require_raw=False
    ).logs


def load_artifacts_sync(
    source: AnalysisSource,
    state: SessionState,
    *,
    deadline: float,
    cancel: threading.Event | None = None,
    require_raw: bool,
) -> ParsedArtifacts:
    """Shared admission, stamp check, capture, snapshot checking and cleanup.

    Paths are authorized on every call. A source whose files carry the stamps
    recorded when their content was parsed returns that content without a
    parser process; anything else is captured and hashed in the worker.
    """
    if require_raw and source.raw is None:
        raise ResultError(
            "This analysis source has no RAW artifact; use log analysis for this source"
        )
    if type(source.plot_index) is not int or source.plot_index < 0:
        raise ResultError("RAW plot_index must be a nonnegative integer")
    limits = raw_limits(state.config.max_raw_mb)
    chosen: dict[str, JsonValue] | None = (
        {
            # In one order, so a read of the same names is one parse.
            "signals": None if source.signals is None else [*sorted(source.signals)],
            "window": None if source.window is None else [*source.window],
            "plot_index": source.plot_index,
            "decode_bytes": DECODE_BYTES,
        }
        if require_raw and (source.signals is not None or source.window is not None)
        else None
    )
    policy: dict[str, JsonValue] = {
        "dialect": _dialect(source.explicit_dialect),
        "producing_dialect": _dialect(source.dialect),
        "limits": {
            "raw": asdict(limits),
            "log": log_limits(),
            "step_rows": STEP_ROWS,
            "metadata_bytes": METADATA_BYTES,
        },
        "chosen": chosen,
    }
    try:
        _check_active(deadline, cancel)
        admitted = _admit(source, state)
        stamped_at = _now_ns()
        stamps = _source_stamps(admitted)
        if not source.trusted_job_artifact:
            _refuse_missing(admitted, stamps, require_raw=require_raw)
        stamp_key = (
            None if stamps is None else json.dumps([admitted, stamps, policy], sort_keys=True)
        )
        if stamp_key is not None:
            cached = state.results.stamped(stamp_key)
            if cached is not None and (not require_raw or cached.raw is not None):
                _check_active(deadline, cancel)
                return _same_snapshot(source, cached)
        with state.results.parse_slot(deadline=deadline, cancel=cancel):
            artifacts = _load(
                source, state, admitted, policy, limits, deadline, cancel, require_raw
            )
    except TimeoutError as exc:
        raise AnalysisDeadlineExceeded("Artifact parser admission exceeded its deadline") from exc
    except InterruptedError as exc:
        raise ResultError("Artifact parser call was cancelled after cleanup") from exc
    if (
        stamp_key is not None
        and stamps is not None
        and _settled(stamps, stamped_at)
        and _source_stamps(admitted) == stamps
    ):
        state.results.record_stamp(stamp_key, artifacts.snapshot_id)
    return _same_snapshot(source, artifacts)


def _admit(source: AnalysisSource, state: SessionState) -> dict[str, JsonValue]:
    """Each role's authorized absolute path, or None for a role not requested."""
    paths = {"raw": source.raw, "log": source.log, "console": source.console}
    admitted: dict[str, JsonValue] = {}
    for role, path in paths.items():
        if path is None:
            admitted[role] = None
        elif source.trusted_job_artifact:
            admitted[role] = str(path.absolute())
        else:
            admitted[role] = str(state.resolve_path(str(path)))
    return admitted


def _refuse_missing(
    admitted: dict[str, JsonValue], stamps: tuple[object, ...] | None, *, require_raw: bool
) -> None:
    """Refuse a path a caller named when the files a capture needs are not there.

    The capture's own rule (``parser_capture.unavailable_inputs``), asked of the
    stamps just taken, so the failure is the caller's missing file, with that
    kind's recovery, rather than a parse that failed. A role whose file could
    not be stamped is left to the capture, as present. A job's own artifacts
    are not asked here: a run whose file is gone is a fact about the run, and
    the capture reports it as one.
    """
    found: dict[str, bool | None] = {}
    for index, role in enumerate(_ROLES):
        path = admitted[role]
        if path is None:
            found[role] = None
            continue
        stamp = stamps[index] if stamps is not None else file_stamp(str(path))
        found[role] = stamp != ABSENT
    if unavailable_inputs(found, require_raw=require_raw) is not None:
        named = next(admitted[role] for role in _ROLES if admitted[role] is not None)
        raise ResultFileNotFoundError(f"No result file at {named}")


def _source_stamps(admitted: dict[str, JsonValue]) -> tuple[object, ...] | None:
    """Each role's file stamp (``lib.file_stamp``), ``"absent"`` for a missing
    file, None for a role not requested; None overall when a source cannot be
    stamped, which leaves the read to the worker."""
    stamps: list[object] = []
    for role in _ROLES:
        path = admitted[role]
        if path is None:
            stamps.append(None)
            continue
        stamp = file_stamp(str(path))
        if stamp is None:
            return None
        stamps.append(stamp)
    return tuple(stamps)


def _settled(stamps: tuple[object, ...], now_ns: int) -> bool:
    """Whether every present source's modification and change times are
    safely in the past.

    A stamp that matches says nothing about a rewrite landing in the same
    timestamp tick as the change it records: that rewrite can keep every
    field. So a stamp is recorded only once that tick has passed (git calls
    the other case "racily clean"), after which any write moves it.
    """
    for stamp in stamps:
        if isinstance(stamp, tuple):
            for moment in (int(stamp[3]), int(stamp[4])):
                margin = _COARSE_SETTLE_NS if moment % 1_000_000_000 == 0 else _SETTLE_NS
                if now_ns - moment < margin:
                    return False
    return True


def _same_snapshot(source: AnalysisSource, artifacts: ParsedArtifacts) -> ParsedArtifacts:
    expected = (source.identity or {}).get("snapshot_id")
    if expected is not None and expected != artifacts.snapshot_id:
        raise ResultError("Analysis source snapshot changed; reload the source before continuing")
    return artifacts


def _load(
    source: AnalysisSource,
    state: SessionState,
    admitted: dict[str, JsonValue],
    policy: dict[str, JsonValue],
    limits: RawLimits,
    deadline: float,
    cancel: threading.Event | None,
    require_raw: bool,
) -> ParsedArtifacts:
    _check_active(deadline, cancel)
    label = source.raw or source.log or source.console or "analysis source"
    snapshot = state.results.snapshot(require_raw=require_raw)
    directory = state.store.parser_dir(uuid4().hex)
    directory.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir()
    reaped = False
    try:
        request: dict[str, JsonValue] = {
            "version": 1,
            "op": "load_raw" if require_raw else "load_logs",
            "sources": admitted,
            **policy,
            "existing_cache_keys": list(snapshot),
        }
        reply = run_parser_sync(
            request,
            work_dir=directory,
            deadline=deadline,
            limits=PROCESS_LIMITS,
            cancel=cancel,
            warm=state.results.warm_parser(PROCESS_LIMITS, directory.parent),
        )
        reaped = True
        _check_active(deadline, cancel)
        metadata = reply.metadata
        key = metadata.get("cache_key")
        if (
            type(metadata.get("version")) is not int
            or metadata["version"] != 1
            or not isinstance(key, str)
            or not is_sha256(key)
        ):
            raise ResultError("Parser returned an invalid version or content identity")
        if metadata.get("status") == "cached":
            if set(metadata) != {"version", "status", "cache_key"} or key not in snapshot:
                raise ResultError("Parser cache reply does not name a retained snapshot")
            artifacts = state.results.get(key)
            if artifacts is None or (require_raw and artifacts.raw is None):
                artifacts = snapshot[key]
        else:
            artifacts = read_parsed_artifacts(
                metadata, directory, limits=limits, require_raw=require_raw, request=request
            )
            _check_active(deadline, cancel)
            state.results.put(key, artifacts)
        return artifacts
    except ParserProcessError as exc:
        reaped = exc.reaped
        if not reaped:
            failure = ParserCleanupError(directory, worker_pid=exc.worker_pid)
            state.results.retain_parser_slot(directory, worker_pid=exc.worker_pid)
            raise failure from exc
        if exc.code == "deadline":
            raise AnalysisDeadlineExceeded("Artifact parsing exceeded its deadline") from exc
        if exc.code in ("signals_not_found", "window_refused"):
            raise ResultError(f"{label}: {exc}", show_hint=False) from exc
        if exc.code == "too_large":
            chosen = policy.get("chosen")
            raise ResultError(
                _too_large(label, str(exc), chosen=chosen if isinstance(chosen, dict) else None),
                show_hint=False,
                # A whole read's refusal ends in the Python read that keeps less.
                python_route=chosen is None,
            ) from exc
        if exc.code == "too_large_to_decode":
            raise ResultError(_too_large_to_decode(label, str(exc)), show_hint=False) from exc
        raise ResultError(f"Failed to parse result file {label}: {exc}") from exc
    except (ValueError, OSError) as exc:
        if not reaped:
            failure = ParserCleanupError(directory)
            state.results.retain_parser_slot(directory)
            raise failure from exc
        raise ResultError(f"Failed to validate parsed result {label}: {exc}") from exc
    finally:
        if reaped:
            try:
                shutil.rmtree(directory)
            except OSError as exc:
                raise ParserCleanupError(directory) from exc


def _too_large(label: object, reason: str, *, chosen: Mapping[str, JsonValue] | None) -> str:
    """Why a result was not read, as the worker counted it, and what reads less."""
    if chosen is not None:
        fewer = (
            "Choose a shorter window, or keep only the traces you need with signals=[...]"
            if chosen.get("signals") is None
            else "Choose fewer signals or a shorter window (window=(start, end) keeps "
            "the points of the axis between them), or read them in more than one load"
        )
        return f"{label} was not read: {reason}. {fewer}."
    return (
        f"{label} was not read: {reason}. Make the run write fewer numbers: .save only "
        "the nodes you need, shorten the .tran, or measure with .meas instead of keeping "
        "the waveform. Or read only the traces you need, the stretch of the axis you "
        "need, or both, with the axis and any step parameters, from a file holding up "
        f"to {DECODE_BYTES // 2**30} GiB of numbers or a larger one in LTspice's Fast "
        "Access layout: r = api.load_raw(..., signals=['V(out)', 'I(R1)'], "
        "window=('1m', '2m'))"
    )


def _too_large_to_decode(label: object, reason: str) -> str:
    """Why a selective read was not made: the worker's reason names what reads less."""
    return (
        f"{label} was not read: {reason}. Or make the run write fewer numbers: .save only "
        "the nodes you need, shorten the .tran, or measure with .meas instead of keeping "
        "the waveform."
    )
