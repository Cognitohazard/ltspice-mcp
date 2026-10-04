"""Step-aware raw-result and measurement primitives for the Python API."""

from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.pathutil import resolve_safe_path
from ltspice_mcp.state import SessionState


class RawResult:
    """Public, detached-array view over one cached SPICE raw result."""

    def __init__(
        self,
        raw: DecodedRaw,
        *,
        source: Path,
        dialect: str | None,
        step_count: int,
        steps: list[dict[str, Any]],
    ) -> None:
        self._raw = raw
        self._signals = tuple(str(name) for name in raw.get_trace_names())
        # Takes ownership of the rows the caller just built; the detachment
        # contract is the per-read copy on the ``steps`` property.
        self._steps = tuple(steps)
        self._step_count = step_count
        self._analysis_type = raw.descriptor.analysis
        self._dialect = raw.descriptor.dialect
        self._source = source

    @property
    def signals(self) -> list[str]:
        """Trace names available in this result, including its primary axis."""
        return list(self._signals)

    def trace(self, name: str, *, step: int = 0) -> np.ndarray:
        """Return one step of ``name`` as a detached array.

        ``name`` is one trace, or a node pair ``V(a,b)``, read as
        ``V(a) - V(b)`` from this step. Other trace math is ordinary numpy on
        the traces it combines.
        """
        signal = services.resolve_signal(self._raw, name)
        self._validate_step(step)
        return np.array(signal.wave(self._raw, step), copy=True)

    def axis(self, *, step: int = 0) -> np.ndarray:
        """Return the selected plot's real sampled coordinates as a detached array."""
        self._validate_step(step)
        descriptor = self._raw.descriptor.axis
        if descriptor is not None and not descriptor.real_coordinates:
            raise ResultError("The selected plot's axis has non-real coordinates.")
        axis = np.array(self._raw.get_axis(step=step), copy=True)
        if np.iscomplexobj(axis):
            return np.real(axis).copy()
        return axis

    def _validate_step(self, step: int) -> None:
        if not self._raw.descriptor.steps:
            raise ResultError("Step boundaries are unresolved; this plot supports inventory only.")
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

    @property
    def plot_index(self) -> int:
        return self._raw.plot_index

    @property
    def descriptor(self) -> dict[str, Any]:
        """Detached metadata for the selected plot."""
        return asdict(self._raw.descriptor)

    @property
    def plots(self) -> list[dict[str, Any]]:
        """Detached plot inventory, in original artifact order."""
        return [asdict(plot.descriptor) for plot in self._raw.plots]

    def table(self, *, step: int = 0) -> list[dict[str, Any]]:
        """Plain native quantities from a plot with no sampled axis."""
        if self._raw.descriptor.axis is not None:
            raise ResultError("table() requires a plot with no sampled axis")
        self._validate_step(step)
        rows: list[dict[str, Any]] = []
        for trace in self._raw.descriptor.traces:
            for index, sample in enumerate(self._raw.get_wave(trace.name, step=step)):
                value: float | dict[str, float] = (
                    {"real": float(np.real(sample)), "imag": float(np.imag(sample))}
                    if np.iscomplexobj(sample)
                    else float(sample)
                )
                rows.append(
                    {
                        "signal": trace.name,
                        "step_index": step,
                        "sample_index": index,
                        "value": value,
                        "unit": trace.unit,
                    }
                )
        return rows


async def load_raw_result(
    *,
    state: SessionState,
    raw_path: str | Path | None,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    plot_index: int = 0,
    dialect: str | None = None,
) -> RawResult:
    """Resolve and bounded-parse one raw result on the API event loop."""
    if raw_path is not None:
        resolved = resolve_safe_path(str(raw_path), state.allowed_paths())
        source = services.source_for_raw_path(
            resolved, state, plot_index=plot_index, dialect=dialect
        )
    else:
        assert job_id is not None
        job = await services.resolve_job_async(job_id, state)
        context = services.experiment_run_context(job, state, run_index=run_index, case_id=case_id)
        source = services.source_for_run(context, plot_index=plot_index, dialect=dialect)
        resolved = source.raw

    raw = await services.load_raw(source, state)
    assert resolved is not None
    step_count = len(raw.descriptor.steps)
    steps = [dict(step.parameters) for step in raw.descriptor.steps]
    return RawResult(
        raw,
        source=resolved,
        dialect=raw.descriptor.dialect,
        step_count=step_count,
        steps=steps,
    )


async def load_measurement_results(
    *,
    state: SessionState,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    log_path: str | Path | None = None,
) -> dict[str, Any]:
    """Read detached measurement facts through the shared log capture."""
    if log_path is not None:
        source = services.resolve_analysis_source(state, log_file=str(log_path))
        absent_message = "Measurement source has no log file"
    else:
        assert job_id is not None
        job = await services.resolve_job_async(job_id, state)
        context = services.experiment_run_context(
            job, state, run_index=run_index, case_id=case_id, require_raw=False
        )
        absent_message = f"Experiment case {context.identity['case_id']!r} has no log file"
        if context.log is None:
            raise ResultError(absent_message)
        source = services.source_for_run(context)
    logs = await services.load_logs(source, state)
    section = logs.section("measurements")
    if section["status"] == "error":
        raise ResultError(section["error"]["message"])
    if section["value"] is None:
        raise ResultError(absent_message)
    return dict(section["value"])
