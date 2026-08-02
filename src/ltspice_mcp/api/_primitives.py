"""Step-aware raw-result and measurement primitives for the Python API."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.log_parser import parse_measurements, parse_step_iterations
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.lib.raw_parser import (
    detect_sim_type,
    get_step_count,
    is_ac_analysis,
    is_dc_analysis,
    is_noise_analysis,
)
from ltspice_mcp.state import SessionState


def _analysis_type(raw: RawRead) -> str:
    sim_type = detect_sim_type(raw)
    if is_noise_analysis(sim_type):
        return "noise"
    if is_ac_analysis(sim_type):
        return "ac"
    if is_dc_analysis(sim_type):
        return "dc"
    if "transient" in sim_type.lower():
        return "transient"
    return "unknown"


def _raw_step_metadata(raw: RawRead) -> list[dict[str, Any]]:
    try:
        rows = raw.steps
    except Exception:
        return []
    if not rows:
        return []
    return [dict(row) if isinstance(row, dict) else {} for row in rows]


def _has_step_parameters(rows: list[dict[str, Any]]) -> bool:
    return any(any(key.lower() != "run" for key in row) for row in rows)


async def _aligned_steps(raw: RawRead, raw_path: Path) -> tuple[int, list[dict[str, Any]]]:
    raw_rows = _raw_step_metadata(raw)
    rows = raw_rows
    if not _has_step_parameters(rows):
        log_path = raw_path.with_suffix(".log")
        log_rows = await services.bounded_parse(
            log_path,
            lambda: parse_step_iterations(log_path),
            timeout_s=services.RAW_PARSE_TIMEOUT_S,
        )
        if log_rows:
            rows = [dict(row) for row in log_rows]

    step_count = max(get_step_count(raw), len(rows), 1)
    aligned = [dict(rows[index]) if index < len(rows) else {} for index in range(step_count)]
    return step_count, aligned


class RawResult:
    """Public, detached-array view over one cached SPICE raw result."""

    def __init__(
        self,
        raw: RawRead,
        *,
        source: Path,
        dialect: str | None,
        step_count: int,
        steps: list[dict[str, Any]],
    ) -> None:
        self._raw = raw
        self._signals = tuple(str(name) for name in raw.get_trace_names())
        self._steps = tuple(copy.deepcopy(steps))
        self._step_count = step_count
        self._analysis_type = _analysis_type(raw)
        self._dialect = getattr(raw, "dialect", None) or dialect
        self._source = source

    @property
    def signals(self) -> list[str]:
        """Trace names available in this result, including its primary axis."""
        return list(self._signals)

    def trace(self, name: str, *, step: int = 0) -> np.ndarray:
        """Return one step of ``name`` as a detached array."""
        canonical = services.validate_signal(self._raw, name)
        self._validate_step(step)
        return np.array(self._raw.get_wave(canonical, step=step), copy=True)

    def axis(self, *, step: int = 0) -> np.ndarray:
        """Return one step's real-valued time or frequency axis as a detached array."""
        self._validate_step(step)
        axis = np.array(self._raw.get_axis(step=step), copy=True)
        if np.iscomplexobj(axis):
            return np.real(axis).copy()
        return axis

    def _validate_step(self, step: int) -> None:
        if step < 0 or step >= self._step_count:
            raise ResultError(
                f"Step {step} out of range. Valid range: 0 to {self._step_count - 1}"
            )

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def steps(self) -> list[dict[str, Any]]:
        """Parameter metadata aligned one-for-one with the result steps."""
        return copy.deepcopy(list(self._steps))

    @property
    def analysis_type(self) -> str:
        return self._analysis_type

    @property
    def dialect(self) -> str | None:
        return self._dialect

    @property
    def source(self) -> Path:
        return self._source


async def load_raw_result(
    *,
    state: SessionState,
    raw_path: str | Path | None,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
) -> RawResult:
    """Resolve and bounded-parse one raw result on the API event loop."""
    dialect: str | None
    if raw_path is not None:
        resolved = resolve_safe_path(str(raw_path), state.config.allowed_paths)
        dialect = services.raw_dialect_for(resolved, state)
    else:
        assert job_id is not None
        job = await services.resolve_job_async(job_id, state)
        if isinstance(job, ExperimentJob):
            context = services.resolve_experiment_run(
                job_id,
                state,
                run_index=run_index,
                case_id=case_id,
            )
            resolved = context.raw
            dialect = context.dialect
            state.raw_dialect_hints[resolved] = dialect
        else:
            if case_id is not None:
                raise TypeError("case_id is only valid when job_id identifies an experiment job")
            resolved = services.resolve_raw_file(job_id, state, run_index)
            dialect = services.raw_dialect_for(resolved, state)

    raw = await services.load_raw(resolved, state)
    step_count, steps = await _aligned_steps(raw, resolved)
    return RawResult(
        raw,
        source=resolved,
        dialect=dialect,
        step_count=step_count,
        steps=steps,
    )


async def load_measurement_results(
    *,
    state: SessionState,
    job_id: str,
    run_index: int,
    case_id: str | None,
) -> dict[str, Any]:
    """Resolve and bounded-parse one legacy run or experiment-case log."""
    job = await services.resolve_job_async(job_id, state)
    if isinstance(job, ExperimentJob):
        context = services.resolve_experiment_run(
            job_id,
            state,
            run_index=run_index,
            case_id=case_id,
        )
        if context.log is None:
            raise ResultError(f"Experiment case {context.identity['case_id']!r} has no log file")
        log_path = context.log
    else:
        if case_id is not None:
            raise TypeError("case_id is only valid when job_id identifies an experiment job")
        log_path = services.resolve_log_file(job_id, state, run_index)

    parsed = await services.bounded_parse(
        log_path,
        lambda: parse_measurements(log_path),
        timeout_s=services.RAW_PARSE_TIMEOUT_S,
    )
    return copy.deepcopy(dict(parsed))
