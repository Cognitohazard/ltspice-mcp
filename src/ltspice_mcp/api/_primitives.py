"""Step-aware raw-result and measurement primitives for the Python API."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import numpy as np

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.ac_analysis import log_interp_at, log_interp_complex_at
from ltspice_mcp.lib.decoded_raw import DecodedRaw
from ltspice_mcp.lib.device_names import (
    TraceKey,
    parse_device_param,
    path_text,
    table_device_path,
    trace_key,
    writes_xvii_names,
)
from ltspice_mcp.lib.log_types import LogSectionName
from ltspice_mcp.lib.raw_parser import OtherRawKind, get_step_count
from ltspice_mcp.lib.step_values import StepValues, listed_steps, step_values_text, step_with
from ltspice_mcp.state import SessionState


class RawResult:
    """Public, detached-array view over one cached SPICE raw result.

    ``signals`` names its traces, ``steps`` holds one parameter dict a
    ``.step`` iteration and ``step_count`` counts them. ``axis(step=k)``,
    ``trace(name, step=k)`` and ``value(name, step=k)`` read samples,
    ``table(step=k)`` a plot with no sampled axis, and ``compare(other)`` how
    far another run's traces are from these. ``analysis_type``, ``dialect``,
    ``descriptor``, ``plots``, ``plot_index``, ``source`` and ``observations``
    say what was read. Of a plot whose steps cannot be told apart, every
    read of its steps, ``step_count`` and ``steps`` included, is refused
    with the reason, as the tools refuse it.
    """

    def __init__(
        self,
        raw: DecodedRaw,
        *,
        source: Path,
        dialect: str | None,
        observations: tuple[dict[str, Any], ...] = (),
        chosen: tuple[str, ...] | None = None,
    ) -> None:
        self._raw = raw
        self._chosen = chosen
        self._signals = tuple(str(name) for name in raw.get_trace_names())
        # The detachment contract is the per-read copy on the ``steps`` property.
        self._steps = tuple(StepValues(step.parameters) for step in raw.descriptor.steps)
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
        signal = self._resolve(name)
        self._validate_step(step)
        return np.array(signal.wave(self._raw, step), copy=True)

    def _resolve(self, name: str) -> services.Signal:
        """``name`` resolved against this result (``services.resolve_signal``).

        A result read for chosen signals holds only their traces, so a trace
        it lacks may be in the file: its refusal says how to read it. Any other
        refusal (an expression, a step value, a noise pair) holds for the file
        too, and is given as it is.
        """
        read = services.read_signal(self._raw, name)
        if read.refused is None:
            return services.Signal(read.name, read.plus, read.minus)
        refused = services.signal_refused(self._raw, name, read)
        if self._chosen is not None and read.refused == "missing":
            raise ResultError(
                f"{refused} This result was read with signals={list(self._chosen)}, so it "
                f"holds only their traces; load_raw again with {name!r} among them.",
                show_hint=refused.show_hint,
                python_route=refused.python_route,
            )
        raise refused

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
        # Refused, with why, when the plot's steps cannot be told apart.
        services.validate_step(self._raw, step)

    @property
    def step_count(self) -> int:
        """How many steps the plot holds, 1 for a run that is not stepped:
        counted as the tools count them (``raw_parser.get_step_count``), so
        a plot whose steps cannot be told apart, such as the operating-point
        raw LTspice 26 writes beside a stepped ``.tran`` or ``.ac``, is
        refused with the reason rather than counted as none."""
        return get_step_count(self._raw)

    @property
    def steps(self) -> list[dict[str, Any]]:
        """Parameter metadata aligned one-for-one with the result steps,
        refused as :attr:`step_count` is.

        Each entry finds a parameter as a step selector does, whatever the
        case of its name: LTspice writes ``rd`` for a deck's ``RD``, and
        ``steps[0]["RD"]`` reads it (:class:`~ltspice_mcp.lib.step_values.StepValues`).
        """
        get_step_count(self._raw)
        return copy.deepcopy(list(self._steps))

    @property
    def analysis_type(self) -> str:
        return self._analysis_type

    @property
    def dialect(self) -> str | None:
        return self._dialect

    @property
    def source(self) -> Path:
        """The ``.raw`` file this result was read from: the one the case's run
        wrote (its other raw, with ``other_raw``) when loaded by ``job_id``,
        else the path given, resolved."""
        return self._source

    @property
    def observations(self) -> list[dict[str, Any]]:
        """What this result says of itself as it was read: ``partial_raw`` when
        it is a stopped case's kept raw, whose values end where the run was
        stopped, ``coarse_compressed_waveform`` when waveform compression left a
        transient with few points, ``signals_chosen`` when it was read for
        chosen signals and ``window_chosen`` for a window of its axis. Empty
        when none applies."""
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
        for trace in self._raw.descriptor.signals():
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
        """This plot's signals but its axis, by ``trace_key``, each to its
        index; a key two traces share is left out, never guessed between."""
        xvii = writes_xvii_names(self._raw)
        descriptor = self._raw.descriptor
        keyed: dict[TraceKey, int] = {}
        shared: set[TraceKey] = set()
        for index, trace in enumerate(descriptor.traces):
            if trace.name in descriptor.step_variables or (
                descriptor.axis is not None and trace.name == descriptor.axis.name
            ):
                continue
            key = trace_key(trace.name, xvii=xvii)
            shared.update([key] if key in keyed else [])
            keyed[key] = index
        return {key: index for key, index in keyed.items() if key not in shared}

    def _step_with(self, parameters: Mapping[str, Any]) -> int:
        """The step whose parameters are ``parameters``, each value matched as
        a step selector matches it (``step_values.step_with``); step 0 of a
        result that is not stepped."""
        if self.step_count == 1 and not self._steps[0]:
            return 0
        found = step_with(self._steps, parameters)
        if found is None:
            raise ResultError(
                f"No step of the other result has this step's parameters "
                f"({step_values_text(parameters)}): its steps are "
                f"{listed_steps(self._steps)}; pass other_step."
            )
        return found

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
        this ``step``'s, each the same value as a step selector reads it
        (``1k`` is 1000, to a part in a million), or step 0 of a result that is
        not stepped. When no step of ``other`` has them, the refusal lists the
        steps it has.

        Returns ``{"signals": {name: {"max_abs", "max_rel", "at", "value",
        "other"}}, "only_here", "only_there", "span", "points"}``: per signal
        the largest difference, its share of this result's largest magnitude
        of that signal (None when that is 0), the axis value it is at, and
        the two values there. How large a difference may be is the caller's.
        A result read for chosen signals (``load_raw(signals=...)``) holds no
        other trace, so a trace only the other result holds was not read, and
        is not known to be missing: when ``other`` was read so, those of this
        result's traces are ``unread_there`` and ``only_here`` is empty, and
        when this result was, the other's are ``unread_here`` and
        ``only_there`` is empty.
        """
        self._validate_step(step)
        if other_step is None:
            other_step = other._step_with(self._steps[step])
        other._validate_step(other_step)
        here, there = self._keyed_traces(), other._keyed_traces()
        names = [trace.name for trace in self._raw.descriptor.traces]
        other_names = [trace.name for trace in other._raw.descriptor.traces]
        pairs: list[tuple[str, _Read, _Read]]
        if signals is None:
            pairs = [
                (names[index], _trace_at(index), _trace_at(there[key]))
                for key, index in here.items()
                if key in there
            ]
        else:
            pairs, missing, reasons = [], [], []
            for signal in signals:
                try:
                    ours = self._resolve(signal)
                    theirs = other._resolve(signal)
                except ResultError as exc:
                    missing.append(signal)
                    reasons.append(str(exc))
                    continue
                pairs.append((ours.name, ours.wave, theirs.wave))
            if missing:
                raise ResultError(
                    f"Not a signal of both results: {', '.join(missing)}. {reasons[0]}"
                )
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
        for name, read_ours, read_theirs in pairs:
            ours = np.asarray(read_ours(self._raw, step))[select]
            points = ours.size
            theirs = np.asarray(read_theirs(other._raw, other_step))
            if resample is not None:
                theirs = resample(theirs)
            elif theirs.size != ours.size:
                raise ResultError(
                    f"{name!r} has {ours.size} points here and {theirs.size} in the "
                    "other result; a plot of no axis compares point by point."
                )
            difference = np.abs(ours - theirs)
            worst = int(np.argmax(difference))
            peak = float(np.max(np.abs(ours)))
            compared[name] = {
                "max_abs": float(difference[worst]),
                "max_rel": float(difference[worst]) / peak if peak else None,
                "at": None if at is None else float(at[worst]),
                "value": ours[worst].item(),
                "other": theirs[worst].item(),
            }
        only_here = sorted(names[i] for key, i in here.items() if key not in there)
        only_there = sorted(other_names[i] for key, i in there.items() if key not in here)
        result: dict[str, Any] = {
            "signals": compared,
            "only_here": only_here,
            "only_there": only_there,
            "span": span,
            "points": points,
        }
        if other._chosen is not None:
            result["only_here"], result["unread_there"] = [], only_here
        if self._chosen is not None:
            result["only_there"], result["unread_here"] = [], only_there
        return result


#: One step of a signal of a result: ``read(raw, step)``.
_Read = Callable[[DecodedRaw, int], np.ndarray]


def _trace_at(index: int) -> _Read:
    """A reader of the trace at ``index`` of a result's selected plot."""
    return lambda raw, step: np.asarray(raw.get_wave(index, step=step))


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
    signals: tuple[str, ...] | None = None,
    window: tuple[float | None, float | None] | None = None,
) -> RawResult:
    """Resolve and bounded-parse one raw result on the API event loop.

    With ``signals``, the parser keeps only the traces they read, with the
    plot's first variable and any step parameters
    (``trace_names.chosen_traces``); with ``window``, only the points of the
    plot's axis in it (``raw_decode.AxisWindow``). It holds its numeric
    budget to what it keeps."""
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

    source = dataclasses.replace(source, signals=signals, window=window)
    raw = await services.load_raw(source, state)
    observations = await asyncio.to_thread(services.read_observations, source, raw)
    assert resolved is not None
    return RawResult(
        raw,
        source=resolved,
        dialect=raw.descriptor.dialect,
        observations=observations,
        chosen=signals,
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
    section = logs.value(name)
    if name == "measurements" and section is not None:
        # Each measurement names the field holding its result.
        section = await services.answer_measurements(source, section)
    return section, source.observations


def measurements_of(section: Any, observations: tuple[dict[str, Any], ...]) -> LogResult[Any]:
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
