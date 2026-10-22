"""Step-aware raw-result and measurement primitives for the Python API."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_log import LogSectionName
from ltspice_mcp.lib.decoded_raw import AC_SHAPED, DecodedRaw
from ltspice_mcp.lib.log_parser import split_device_key
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

    def value(self, name: str, *, step: int = 0) -> float | complex:
        """The one value of ``name`` in this step, for a plot of one point a
        step, such as an operating point (``.op``, one point for each step of a
        stepped one): a float, or a complex for a complex trace.

        ``name`` is read as :meth:`trace` reads it. A trace of more than one
        point in the step is refused rather than cut to its first; read it
        with :meth:`trace`.
        """
        samples = self.trace(name, step=step)
        if samples.size != 1:
            raise ResultError(
                f"{name!r} has {samples.size} points in step {step}; value() reads a "
                "plot of one point a step, such as an operating point's. Read this "
                "one with trace()."
            )
        return samples.item()

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

    def _compared_signals(self) -> dict[str, str]:
        """This plot's traces but its axis, by name without case."""
        axis = self._raw.descriptor.axis
        return {
            trace.name.casefold(): trace.name
            for trace in self._raw.descriptor.traces
            if axis is None or trace.name != axis.name
        }

    def compare(
        self,
        other: RawResult,
        *,
        signals: Sequence[str] | None = None,
        step: int = 0,
        other_step: int | None = None,
    ) -> dict[str, Any]:
        """How far ``other``'s traces are from this result's, signal by signal.

        Signals are matched by name without case, as SPICE reads them;
        ``signals`` names the ones to compare, and by default every signal
        both hold is, the axis aside. Each is compared at this result's
        points, ``other`` read there by interpolation when its axis differs
        (a compressed LTspice raw stores few points, and its values between
        them are the line drawn through them): linearly on a time or sweep
        axis, in log frequency, dB and unwrapped phase on a frequency axis.
        Only the span both axes cover is compared. A plot of no axis, such as
        an operating point, compares point by point. Complex values compare as
        complex numbers.

        ``other_step`` defaults to the step of ``other`` whose parameters are
        this ``step``'s, or step 0 of a result that is not stepped.

        Returns ``{"signals": {name: {"max_abs", "max_rel", "at", "value",
        "other"}}, "only_here", "only_there", "span", "points"}``: per signal
        the largest difference, its share of this result's largest magnitude
        of that signal (None when that is 0), the axis value it is at, and
        the two values there. How large a difference may be is the caller's.
        """
        self._validate_step(step)
        if other_step is None:
            other_step = _paired_step(self.steps[step], other)
        other._validate_step(other_step)
        here, there = self._compared_signals(), other._compared_signals()
        if signals is None:
            keys = [key for key in here if key in there]
        else:
            keys = [name.casefold() for name in signals]
            missing = [name for name in signals if name.casefold() not in here.keys() & there]
            if missing:
                raise ResultError(f"Not a signal of both results: {', '.join(missing)}")
        if (self._raw.descriptor.axis is None) != (other._raw.descriptor.axis is None):
            raise ResultError("One result has a sampled axis and the other has none.")
        compared: dict[str, Any] = {}
        span: list[float] | None = None
        points: int
        other_axis = None
        if self._raw.descriptor.axis is None:
            at = None
            select = slice(None)
            points = len(self.trace(next(iter(here.values())), step=step)) if here else 0
        else:
            axis, other_axis = self.axis(step=step), other.axis(step=other_step)
            low = max(axis.min(), other_axis.min())
            high = min(axis.max(), other_axis.max())
            select = (axis >= low) & (axis <= high)
            points = int(np.count_nonzero(select))
            if points == 0:
                raise ResultError("The two results' axes share no point to compare at.")
            at = axis[select]
            span = [float(low), float(high)]
        log_frequency = self._raw.descriptor.analysis in AC_SHAPED
        for key in keys:
            ours = self.trace(here[key], step=step)[select]
            theirs = other.trace(there[key], step=other_step)
            if at is None:
                if theirs.size != ours.size:
                    raise ResultError(
                        f"{here[key]!r} has {ours.size} points here and {theirs.size} in the "
                        "other result; a plot of no axis compares point by point."
                    )
            elif other_axis is not None and not np.array_equal(at, other_axis):
                theirs = _resample(other_axis, theirs, at, log_frequency=log_frequency)
            difference = np.abs(ours - theirs)
            worst = int(np.argmax(difference))
            peak = float(np.max(np.abs(ours)))
            compared[here[key]] = {
                "max_abs": float(difference[worst]),
                "max_rel": float(difference[worst]) / peak if peak else None,
                "at": None if at is None else float(at[worst]),
                "value": ours[worst].item(),
                "other": theirs[worst].item(),
            }
        return {
            "signals": compared,
            "only_here": sorted(name for key, name in here.items() if key not in there),
            "only_there": sorted(name for key, name in there.items() if key not in here),
            "span": span,
            "points": points,
        }


def _paired_step(parameters: dict[str, Any], other: RawResult) -> int:
    """The step of ``other`` whose parameters are ``parameters``; step 0 of a
    result that is not stepped."""
    if other.step_count == 1 and not other.steps[0]:
        return 0
    for index, theirs in enumerate(other.steps):
        if theirs == parameters:
            return index
    raise ResultError(
        f"No step of the other result has this step's parameters ({parameters or 'none'}); "
        "pass other_step."
    )


def _resample(
    axis: np.ndarray, values: np.ndarray, at: np.ndarray, *, log_frequency: bool
) -> np.ndarray:
    """``values`` on ``axis`` read at ``at`` by linear interpolation: in log
    frequency, and for a complex value in log magnitude and unwrapped phase, on a
    frequency axis."""
    order = np.argsort(axis, kind="stable")
    axis, values = axis[order], values[order]
    if log_frequency and axis[0] > 0 and at.min() > 0:
        axis, at = np.log10(axis), np.log10(at)
        if np.iscomplexobj(values) and np.all(values != 0):
            magnitude = np.exp(np.interp(at, axis, np.log(np.abs(values))))
            phase = np.interp(at, axis, np.unwrap(np.angle(values)))
            return magnitude * np.exp(1j * phase)
    if np.iscomplexobj(values):
        return np.interp(at, axis, np.real(values)) + 1j * np.interp(at, axis, np.imag(values))
    return np.interp(at, axis, values)


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
        resolved = state.resolve_path(str(raw_path))
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


async def load_log_section(
    name: LogSectionName,
    *,
    state: SessionState,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    log_path: str | Path | None,
) -> Any:
    """One section of a log, read through the shared log capture: of the log at
    ``log_path``, or of one experiment case's. None when the log has no such
    section."""
    if log_path is not None:
        source = services.resolve_analysis_source(state, log_file=str(log_path))
    else:
        assert job_id is not None
        job = await services.resolve_job_async(job_id, state)
        context = services.experiment_run_context(
            job, state, run_index=run_index, case_id=case_id, require_raw=False
        )
        if context.log is None:
            raise ResultError(f"Experiment case {context.identity['case_id']!r} has no log file")
        source = services.source_for_run(context)
    logs = await services.load_logs(source, state)
    return logs.value(name)


def measurements_of(section: Any) -> dict[str, Any]:
    """The ``measurements`` section as ``Api.measurements`` returns it."""
    if section is None:
        raise ResultError("The log holds no measurements")
    return dict(section)


def device_op_points_of(section: Any) -> dict[str, dict[str, float | None]]:
    """The ``device_op`` section by device and parameter, as
    ``Api.device_op_points`` returns it. A value the log printed as not a
    finite number is None."""
    by_device: dict[str, dict[str, float | None]] = {}
    for key, number in (section or {}).items():
        device, param = split_device_key(key)
        by_device.setdefault(device, {})[param] = number
    return by_device
