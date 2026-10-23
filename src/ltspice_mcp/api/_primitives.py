"""Step-aware raw-result and measurement primitives for the Python API."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.ac_analysis import log_interp_at, log_interp_complex_at
from ltspice_mcp.lib.decoded_log import LogSectionName
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.device_names import (
    TraceKey,
    parse_device_param,
    path_text,
    table_device_path,
    trace_key,
    writes_xvii_names,
)
from ltspice_mcp.lib.raw_parser import OtherRawKind
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
        observations: tuple[dict[str, Any], ...] = (),
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
        self._observations = observations

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
    def observations(self) -> list[dict[str, Any]]:
        """What this result says of itself as it was read: ``partial_raw`` when
        it is a stopped case's kept raw, whose values end where the run was
        stopped. Empty for a whole result."""
        return copy.deepcopy(list(self._observations))

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

    def _keyed_traces(self) -> dict[TraceKey, int]:
        """This plot's traces but its axis, by ``trace_key``, each to its index;
        a key two traces share is left out, never guessed between."""
        xvii = writes_xvii_names(self._raw)
        axis = self._raw.descriptor.axis
        keyed: dict[TraceKey, int] = {}
        shared: set[TraceKey] = set()
        for index, trace in enumerate(self._raw.descriptor.traces):
            if axis is not None and trace.name == axis.name:
                continue
            key = trace_key(trace.name, xvii=xvii)
            shared.update([key] if key in keyed else [])
            keyed[key] = index
        return {key: index for key, index in keyed.items() if key not in shared}

    def _step_with(self, parameters: dict[str, Any]) -> int:
        """The step whose parameters are ``parameters``; step 0 of a result
        that is not stepped."""
        if self._step_count == 1 and not self._steps[0]:
            return 0
        try:
            return self._steps.index(parameters)
        except ValueError:
            raise ResultError(
                f"No step of the other result has this step's parameters "
                f"({parameters or 'none'}); pass other_step."
            ) from None

    def compare(
        self,
        other: RawResult,
        *,
        signals: Sequence[str] | None = None,
        step: int = 0,
        other_step: int | None = None,
    ) -> dict[str, Any]:
        """How far ``other``'s traces are from this result's, signal by signal.

        ``signals`` names the ones to compare, each found in each result as
        :meth:`trace` finds it. By default every signal both hold is, the axis
        aside, paired by name without case and by the path to a node or device
        inside subcircuits however each build spells it (LTspice 26's
        ``V(xa:xin:mid)`` is XVII's ``V(a:in:mid)``). Each is compared at this result's
        points, ``other`` read there by interpolation when its axis differs
        (a compressed LTspice raw stores few points, and its values between
        them are the line drawn through them): linearly on a time or sweep
        axis, in log frequency, dB and unwrapped phase on a frequency axis
        (``log_interp_complex``).
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
            other_step = other._step_with(self._steps[step])
        other._validate_step(other_step)
        here, there = self._keyed_traces(), other._keyed_traces()
        names = [trace.name for trace in self._raw.descriptor.traces]
        other_names = [trace.name for trace in other._raw.descriptor.traces]
        if signals is None:
            pairs = [(index, there[key]) for key, index in here.items() if key in there]
        else:
            found = [(self._find(signal), other._find(signal)) for signal in signals]
            missing = [
                s for s, (a, b) in zip(signals, found, strict=True) if a is None or b is None
            ]
            if missing:
                raise ResultError(f"Not a signal of both results: {', '.join(missing)}")
            pairs = [(names.index(a), other_names.index(b)) for a, b in found if a and b]
        has_axis = self._raw.descriptor.axis is not None
        if has_axis != (other._raw.descriptor.axis is not None):
            raise ResultError("One result has a sampled axis and the other has none.")
        select: slice | np.ndarray = slice(None)
        at = span = resample = None
        if has_axis:
            axis, other_axis = self.axis(step=step), other.axis(step=other_step)
            low = max(axis.min(), other_axis.min())
            high = min(axis.max(), other_axis.max())
            select = (axis >= low) & (axis <= high)
            at = axis[select]
            if not at.size:
                raise ResultError("The two results' axes share no point to compare at.")
            span = [float(low), float(high)]
            if not np.array_equal(at, other_axis):
                assert self._raw.descriptor.axis is not None
                resample = _Resampler(
                    other_axis, at, log=self._raw.descriptor.axis.quantity == "frequency"
                )
        compared: dict[str, Any] = {}
        points = 0 if at is None else at.size
        for index, other_index in pairs:
            ours = self._raw.get_wave(index, step=step)[select]
            points = ours.size
            theirs = other._raw.get_wave(other_index, step=other_step)
            if resample is not None:
                theirs = resample(theirs)
            elif theirs.size != ours.size:
                raise ResultError(
                    f"{names[index]!r} has {ours.size} points here and {theirs.size} in the "
                    "other result; a plot of no axis compares point by point."
                )
            difference = np.abs(ours - theirs)
            worst = int(np.argmax(difference))
            peak = float(np.max(np.abs(ours)))
            compared[names[index]] = {
                "max_abs": float(difference[worst]),
                "max_rel": float(difference[worst]) / peak if peak else None,
                "at": None if at is None else float(at[worst]),
                "value": ours[worst].item(),
                "other": theirs[worst].item(),
            }
        return {
            "signals": compared,
            "only_here": sorted(names[i] for key, i in here.items() if key not in there),
            "only_there": sorted(other_names[i] for key, i in there.items() if key not in here),
            "span": span,
            "points": points,
        }

    def _find(self, signal: str) -> str | None:
        """The trace ``signal`` names in this result, found as :meth:`trace`
        finds one."""
        return services.find_trace(
            [trace.name for trace in self._raw.descriptor.traces],
            signal,
            xvii=writes_xvii_names(self._raw),
        )


class _Resampler:
    """Another result's values read at this one's axis points, the other's
    axis sorted once and, on a frequency axis, its logarithm taken once."""

    def __init__(self, axis: np.ndarray, at: np.ndarray, *, log: bool) -> None:
        self._order = None if np.all(np.diff(axis) >= 0) else np.argsort(axis, kind="stable")
        self._axis = axis if self._order is None else axis[self._order]
        self._at = at
        self._log = log and self._axis[0] > 0 and at.min() > 0

    def __call__(self, values: np.ndarray) -> np.ndarray:
        if self._order is not None:
            values = values[self._order]
        if not self._log:
            return np.interp(self._at, self._axis, values)
        if np.iscomplexobj(values):
            return log_interp_complex_at(self._axis, values, self._at)
        return log_interp_at(self._axis, values, self._at)


async def load_raw_result(
    *,
    state: SessionState,
    raw_path: str | Path | None,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    plot_index: int = 0,
    dialect: str | None = None,
    other_raw: OtherRawKind | None = None,
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
        context = services.experiment_run_context(
            job, state, run_index=run_index, case_id=case_id, other_raw=other_raw
        )
        source = services.source_for_run(context, plot_index=plot_index, dialect=dialect)
        resolved = source.raw

    raw = await services.load_raw(source, state)
    observations = await asyncio.to_thread(services.read_observations, source, raw)
    assert resolved is not None
    step_count = len(raw.descriptor.steps)
    steps = [dict(step.parameters) for step in raw.descriptor.steps]
    return RawResult(
        raw,
        source=resolved,
        dialect=raw.descriptor.dialect,
        step_count=step_count,
        steps=steps,
        observations=observations,
    )


_V = TypeVar("_V")


class LogResult(dict[str, _V]):
    """A section of a log as the Python API returns it: a plain ``dict``, and
    what the read says of it, as ``RawResult.observations`` says of a raw."""

    def __init__(
        self, data: Mapping[str, _V], *, observations: tuple[dict[str, Any], ...] = ()
    ) -> None:
        super().__init__(data)
        self._observations = observations

    @property
    def observations(self) -> list[dict[str, Any]]:
        """What this log says of itself as it was read: ``partial_raw`` when it
        is the log of a stopped case whose raw was kept, whose results end
        where the run was stopped. Empty for a whole run's log."""
        return copy.deepcopy(list(self._observations))


async def load_log_section(
    name: LogSectionName,
    *,
    state: SessionState,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    log_path: str | Path | None,
) -> tuple[Any, tuple[dict[str, Any], ...]]:
    """One section of a log, read through the shared log capture: of the log at
    ``log_path``, or of one experiment case's, and what the read says of it
    (``AnalysisSource.observations``). The section is None when the log has no
    such section."""
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
    return logs.value(name), source.observations


def measurements_of(section: Any, observations: tuple[dict[str, Any], ...] = ()) -> LogResult[Any]:
    """The ``measurements`` section as ``Api.measurements`` returns it."""
    if section is None:
        raise ResultError("The log holds no measurements")
    return LogResult(section, observations=observations)


def device_op_points_of(
    section: Any, observations: tuple[dict[str, Any], ...] = ()
) -> LogResult[dict[str, float | None]]:
    """The ``device_op`` section by device and parameter, as
    ``Api.device_op_points`` returns it: a device inside a subcircuit by its
    path (``xa.xin.mtail``), however the build that wrote the table names it.
    A value the log printed as not a finite number is None."""
    by_device: dict[str, dict[str, float | None]] = {}
    for key, number in (section or {}).items():
        if (parsed := parse_device_param(key)) is None:
            continue
        device, param = parsed
        by_device.setdefault(path_text(table_device_path(device)), {})[param] = number
    return LogResult(by_device, observations=observations)
