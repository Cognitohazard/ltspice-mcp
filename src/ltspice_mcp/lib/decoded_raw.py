"""Fully resident RAW plots, with header facts separate from physical meaning.

The decoder transfers ownership of unrebased, full-length NumPy arrays in
header-variable order. It must not retain writable aliases. Construction marks
these arrays read-only without copying. Only transient coordinate correction
can allocate a replacement; every accessor then uses that same corrected axis.
This module neither reads files nor constructs third-party readers.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from ltspice_mcp.errors import NoAxisError, ResultError
from ltspice_mcp.lib.raw_header import RawPlotHeader

if TYPE_CHECKING:
    from ltspice_mcp.lib.decoded_log import DecodedLog

StepValue = str | int | float
Analysis = Literal[
    "op", "transient", "ac", "dc", "noise", "tf", "pz", "sens_dc", "sens_ac", "disto", "unknown"
]
Representation = Literal["real", "complex"]
StepStatus = Literal["unstepped", "matched", "unresolved", "mismatch"]
Monotonicity = Literal["nondecreasing", "nonincreasing", "constant", "nonmonotonic", "unknown"]


class RawTrace(Protocol):
    """Trace metadata and numeric access required by shared numerical helpers."""

    @property
    def name(self) -> str: ...

    @property
    def whattype(self) -> str: ...

    def get_wave(self, step: int = 0) -> NDArray: ...


class RawData(Protocol):
    """Reader access used by numerical helpers, independent of plot descriptors.

    Structural typing also accepts existing spicelib test readers. Implementing
    this interface alone does not establish residency or parser safety; the
    production loader supplies those guarantees through DecodedRaw.
    """

    @property
    def dialect(self) -> str | None: ...

    @property
    def steps(self) -> Sequence[Mapping[str, Any]] | None: ...

    @property
    def raw_params(self) -> Mapping[str, Any]: ...

    def get_raw_property(self, property_name: str | None = None) -> Any: ...

    def get_trace_names(self) -> list[str]: ...

    def get_trace(self, trace_ref: str | int) -> RawTrace: ...

    def get_wave(self, trace_ref: str | int, step: int = 0) -> NDArray: ...

    def get_axis(self, step: int = 0) -> NDArray | list[float]: ...

    def get_steps(self, **kwargs: StepValue) -> Sequence[int]: ...


@dataclass(frozen=True)
class TraceDescriptor:
    name: str
    declared_type: str
    representation: Representation
    unit: str | None
    unit_evidence: str | None
    normalization: str | None


@dataclass(frozen=True)
class AxisDescriptor:
    trace_index: int
    name: str
    declared_type: str
    unit: str | None
    quantity: str
    convention: str
    real_coordinates: bool
    monotonicity: Monotonicity
    time_offset: float


@dataclass(frozen=True)
class StepDescriptor:
    index: int
    parameters: tuple[tuple[str, StepValue], ...]


@dataclass(frozen=True)
class PlotDescriptor:
    snapshot_id: str
    plot_index: int
    original_plot_name: str
    analysis: Analysis
    layout: Literal["sampled", "table"]
    axis: AxisDescriptor | None
    traces: tuple[TraceDescriptor, ...]
    steps: tuple[StepDescriptor, ...]
    completeness: Literal["complete", "step_metadata_missing"]
    dialect: str
    dialect_evidence: tuple[str, ...]
    step_status: StepStatus = "unstepped"


_UNITS = {
    "voltage": "V",
    "current": "A",
    "device_current": "A",
    "time": "s",
    "frequency": "Hz",
    "hertz": "Hz",
    "admittance": "S",
    "impedance": "Ω",
    "capacitance": "F",
    "voltage-density": "V/√Hz",
    "current-density": "A/√Hz",
}
_ROOT_LABEL = re.compile(r"(?:v\()?((?:pole|zero)\([0-9]+\))\)?\Z", re.I)
_ANALYSIS_NAMES: dict[str, Analysis] = {
    "operating point": "op",
    "transfer function": "tf",
    "pole-zero analysis": "pz",
    "transient analysis": "transient",
    "ac analysis": "ac",
    "dc transfer characteristic": "dc",
    "integrated noise": "noise",
}


def _analysis(header: RawPlotHeader) -> Analysis:
    label = header.plot_name.strip().lower()
    if label == "sensitivity analysis":
        first = header.variables[0]
        return "sens_ac" if first.declared_type.lower() in ("frequency", "hertz") else "sens_dc"
    if label.startswith("distortion -"):
        return "disto"
    if label.startswith("noise spectral density"):
        return "noise"
    return _ANALYSIS_NAMES.get(label, "unknown")


def _trace_descriptor(
    header: RawPlotHeader, index: int, wave: NDArray, analysis: Analysis
) -> TraceDescriptor:
    variable = header.variables[index]
    declared = variable.declared_type.lower()
    name = variable.name.lower()
    unit = _UNITS.get(declared)
    evidence = "declared_type" if unit else None
    normalization = None
    # ngspice's native analyses emit generic voltage types for derived facts.
    # TF input/output resistances and PZ Laplace roots have different dimensions.
    if analysis == "unknown":
        unit, evidence = None, None
    elif analysis == "tf":
        unit, evidence = None, None
        if header.dialect == "ngspice" and (
            "#input_impedance)" in name or name.startswith("v(output_impedance_at_")
        ):
            unit, evidence = "Ω", "ngspice:tf_resistance_label"
        elif declared == "impedance":
            unit, evidence = "Ω", "declared_type"
    elif analysis == "pz":
        unit, evidence = None, None
        if header.dialect == "ngspice" and _ROOT_LABEL.fullmatch(variable.name):
            unit, evidence = "s^-1", "ngspice:pz_laplace_root"
    elif analysis in ("sens_dc", "sens_ac") and not (analysis == "sens_ac" and index == 0):
        # cktsens.c divides the perturbed output by delta_var, without relative
        # normalization. Header labels do not establish parameter/output units.
        unit, evidence = None, None
        if header.dialect == "ngspice":
            normalization = "absolute"
    elif analysis == "noise" and name in {
        "inoise",
        "v(inoise)",
        "i(inoise)",
        "inoise_spectrum",
        "inoise_total",
    }:
        unit, evidence = None, "input_source_unresolved"
    elif (
        analysis == "noise"
        and header.plot_name.lower() != "integrated noise"
        and declared in ("voltage", "current", "device_current")
        and index != 0
    ):
        unit = "V/√Hz" if declared == "voltage" else "A/√Hz"
        evidence = "noise_spectral_density:declared_type"
    return TraceDescriptor(
        variable.name,
        variable.declared_type,
        "complex" if np.iscomplexobj(wave) else "real",
        unit,
        evidence,
        normalization,
    )


def _step_descriptors(
    points: int, rows: Sequence[Mapping[str, StepValue]] | None, offsets: Sequence[int] | None
) -> tuple[tuple[StepDescriptor, ...], tuple[int, ...]]:
    starts = tuple(offsets) if offsets is not None else (0,)
    if (
        not starts
        or starts[0] != 0
        or any(type(s) is not int or not 0 <= s < points for s in starts)
    ):
        raise ValueError("Step starts must be integer sample offsets beginning at zero")
    if any(a >= b for a, b in pairwise(starts)):
        raise ValueError("Step starts must be strictly increasing")
    if rows is not None and len(rows) != len(starts):
        raise ValueError("Step metadata and sample offsets must have the same count")
    descriptors = []
    for index in range(len(starts)):
        parameters = tuple(rows[index].items()) if rows is not None else ()
        for key, value in parameters:
            if not isinstance(key, str) or type(value) not in (str, int, float):
                raise ValueError(
                    "Step parameters must contain string keys and plain scalar values"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("Step parameters must be finite")
        descriptors.append(StepDescriptor(index, parameters))
    return tuple(descriptors), (*starts, points)


def _axis_kind(header: RawPlotHeader, analysis: Analysis) -> tuple[str, str] | None:
    declared = header.variables[0].declared_type.lower()
    if analysis == "transient" and declared == "time":
        return "time", "deck_time"
    if analysis in ("ac", "sens_ac", "disto", "noise") and declared in ("frequency", "hertz"):
        return "frequency", "swept_f1" if analysis == "disto" else "frequency"
    if analysis == "dc":
        return "swept_variable", "sweep"
    return None


def transient_time_offset(header: RawPlotHeader) -> float:
    """Validate the time offset using header facts before numeric allocation."""
    analysis = _analysis(header)
    if analysis != "transient" or _axis_kind(header, analysis) is None:
        return 0.0
    offset = float(next((value for key, value in header.fields if key.lower() == "offset"), "0"))
    if not math.isfinite(offset):
        raise ValueError("Transient time offset must be finite")
    return offset


def _coordinate_facts(wave: NDArray, boundaries: tuple[int, ...]) -> tuple[bool, Monotonicity]:
    real = bool(
        np.all(np.isfinite(wave)) and (not np.iscomplexobj(wave) or np.all(np.imag(wave) == 0))
    )
    if not real:
        return False, "unknown"
    # Check within each step, never across the reset between steps.
    increasing, decreasing = True, True
    for start, end in pairwise(boundaries):
        values = np.real(wave)[start:end]
        increasing &= bool(np.all(values[1:] >= values[:-1]))
        decreasing &= bool(np.all(values[1:] <= values[:-1]))
    if increasing and decreasing:
        return True, "constant"
    if increasing:
        return True, "nondecreasing"
    if decreasing:
        return True, "nonincreasing"
    return True, "nonmonotonic"


@dataclass(frozen=True)
class DecodedTrace:
    """Original trace metadata and a read-only step slice, with no lazy loader."""

    descriptor: TraceDescriptor
    _values: NDArray
    _boundaries: tuple[int, ...] | None

    @property
    def name(self) -> str:
        return self.descriptor.name

    @property
    def whattype(self) -> str:
        return self.descriptor.declared_type

    @property
    def numerical_type(self) -> str:
        if self.descriptor.representation == "complex":
            return "complex"
        return "double" if self._values.dtype.itemsize == 8 else "real"

    def get_wave(self, step: int = 0) -> NDArray:
        if self._boundaries is None:
            raise ResultError("Step boundaries are unresolved; this plot supports inventory only")
        if type(step) is not int or not 0 <= step < len(self._boundaries) - 1:
            raise IndexError(f"Step {step} is outside this plot")
        return self._values[self._boundaries[step] : self._boundaries[step + 1]]


@dataclass(frozen=True, init=False)
class DecodedPlot:
    header: RawPlotHeader
    descriptor: PlotDescriptor
    _traces: tuple[DecodedTrace, ...]
    _has_step_metadata: bool
    _stored_waves: tuple[NDArray, ...]

    def __init__(
        self,
        header: RawPlotHeader,
        waves: Sequence[NDArray],
        *,
        snapshot_id: str,
        steps: Sequence[Mapping[str, StepValue]] | None = None,
        step_offsets: Sequence[int] | None = None,
        step_status: StepStatus | None = None,
    ) -> None:
        if (
            not header.variables
            or len(waves) != header.variable_count
            or len(header.variables) != len(waves)
        ):
            raise ValueError("Every header variable must have one resident array")
        if any(
            not isinstance(w, np.ndarray)
            or w.ndim != 1
            or len(w) != header.point_count
            or w.dtype.kind not in "fciu"
            for w in waves
        ):
            raise ValueError(
                "Resident arrays must be one-dimensional numeric arrays of the declared length"
            )
        descriptors, boundaries = _step_descriptors(header.point_count, steps, step_offsets)
        stepped = any(flag.lower() == "stepped" for flag in header.flags)
        if step_status is None:
            if step_offsets is None and stepped:
                step_status = "unresolved"
            elif steps is not None or len(descriptors) > 1 or stepped:
                step_status = "matched" if steps is not None else "unresolved"
            else:
                step_status = "unstepped"
        if step_status not in ("unstepped", "matched", "unresolved", "mismatch"):
            raise ValueError("Unknown step reconciliation status")
        if step_status == "unstepped" and (len(descriptors) != 1 or stepped):
            raise ValueError("Unstepped status contradicts stored step evidence")
        if step_status == "matched" and steps is None:
            raise ValueError("Matched steps require associated metadata")
        unresolved_boundaries = step_offsets is None and step_status in ("unresolved", "mismatch")
        if unresolved_boundaries:
            if steps is not None:
                raise ValueError("Step metadata cannot be associated without validated boundaries")
            descriptors = ()
        analysis = _analysis(header)
        kind = _axis_kind(header, analysis)
        arrays = list(waves)
        offset = transient_time_offset(header)
        if kind is not None and analysis == "transient":
            if np.iscomplexobj(arrays[0]):
                raise ValueError("Transient time must be real and its offset finite")
            if offset or np.any(arrays[0] < 0):
                corrected = np.abs(arrays[0], dtype=np.result_type(arrays[0].dtype, offset))
                corrected += offset
                arrays[0] = corrected
        traces = tuple(
            _trace_descriptor(header, i, wave, analysis) for i, wave in enumerate(arrays)
        )
        axis = None
        if kind is not None:
            real, monotonicity = _coordinate_facts(arrays[0], boundaries)
            if unresolved_boundaries:
                monotonicity = "unknown"
            first = traces[0]
            axis = AxisDescriptor(
                0, first.name, first.declared_type, first.unit, *kind, real, monotonicity, offset
            )
        missing_steps = steps is None and step_status != "unstepped"
        descriptor = PlotDescriptor(
            snapshot_id,
            header.index,
            header.plot_name,
            analysis,
            "sampled" if axis else "table",
            axis,
            traces,
            descriptors,
            "step_metadata_missing" if missing_steps else "complete",
            header.dialect,
            header.dialect_evidence,
            step_status,
        )
        for wave in waves:
            wave.setflags(write=False)
        if arrays[0] is not waves[0]:
            arrays[0].setflags(write=False)
        object.__setattr__(self, "header", header)
        object.__setattr__(self, "descriptor", descriptor)
        object.__setattr__(self, "_has_step_metadata", steps is not None)
        object.__setattr__(self, "_stored_waves", tuple(waves))
        object.__setattr__(
            self,
            "_traces",
            tuple(
                DecodedTrace(t, w, None if unresolved_boundaries else boundaries)
                for t, w in zip(traces, arrays, strict=True)
            ),
        )

    @property
    def dialect(self) -> str:
        return self.header.dialect

    @property
    def time_offset(self) -> float:
        return self.descriptor.axis.time_offset if self.descriptor.axis else 0.0

    @property
    def raw_params(self) -> dict[str, str | list[str]]:
        params: dict[str, str | list[str]] = dict(self.header.fields)
        params["Variables"] = self.get_trace_names()
        return params

    @property
    def steps(self) -> list[dict[str, StepValue]] | None:
        return (
            [dict(s.parameters) for s in self.descriptor.steps]
            if self._has_step_metadata
            else None
        )

    def get_raw_property(
        self, property_name: str | None = None
    ) -> str | list[str] | dict[str, str | list[str]]:
        params = self.raw_params
        if property_name is None:
            return params
        for key, value in params.items():
            if key.lower() == property_name.lower():
                return value
        raise ValueError(f"Unknown RAW property: {property_name}")

    def get_trace_names(self) -> list[str]:
        return [t.name for t in self._traces]

    def get_trace(self, trace_ref: str | int) -> DecodedTrace:
        if type(trace_ref) is int and 0 <= trace_ref < len(self._traces):
            return self._traces[trace_ref]
        if isinstance(trace_ref, str):
            for trace in self._traces:
                if trace.name.lower() == trace_ref.lower():
                    return trace
        raise IndexError(f"Unknown trace: {trace_ref}")

    def get_wave(self, trace_ref: str | int, step: int = 0) -> NDArray:
        return self.get_trace(trace_ref).get_wave(step)

    def get_axis(self, step: int = 0) -> NDArray:
        if self.descriptor.axis is None:
            raise NoAxisError(
                f"Plot {self.header.index} ({self.header.plot_name}) has no sampled axis"
            )
        return self.get_wave(self.descriptor.axis.trace_index, step)

    def get_steps(self, **kwargs: StepValue) -> list[int]:
        if not self.descriptor.steps:
            raise ResultError("Step boundaries are unresolved; this plot supports inventory only")
        indices = []
        for step in self.descriptor.steps:
            parameters = dict(step.parameters)
            if all(
                key in parameters and parameters[key] == value for key, value in kwargs.items()
            ):
                indices.append(step.index)
        return indices


@dataclass(frozen=True, init=False)
class DecodedRaw:
    """One selected plot with an immutable inventory of all resident plots."""

    plots: tuple[DecodedPlot, ...]
    plot_index: int
    logs: DecodedLog | None

    def __init__(
        self, plots: Sequence[DecodedPlot], *, plot_index: int = 0, logs: DecodedLog | None = None
    ) -> None:
        inventory = tuple(plots)
        if not inventory or any(p.header.index != i for i, p in enumerate(inventory)):
            raise ValueError("Resident plots must form a nonempty inventory in header-index order")
        if type(plot_index) is not int or not 0 <= plot_index < len(inventory):
            raise IndexError(f"Plot {plot_index} is outside this artifact")
        object.__setattr__(self, "plots", inventory)
        object.__setattr__(self, "plot_index", plot_index)
        object.__setattr__(self, "logs", logs)

    @property
    def descriptor(self) -> PlotDescriptor:
        return self.plots[self.plot_index].descriptor

    @property
    def dialect(self) -> str:
        return self.plots[self.plot_index].dialect

    @property
    def raw_params(self) -> dict[str, str | list[str]]:
        return self.plots[self.plot_index].raw_params

    @property
    def time_offset(self) -> float:
        return self.plots[self.plot_index].time_offset

    @property
    def steps(self) -> list[dict[str, StepValue]] | None:
        return self.plots[self.plot_index].steps

    def select_plot(self, plot_index: int) -> DecodedRaw:
        return DecodedRaw(self.plots, plot_index=plot_index, logs=self.logs)

    def get_plot_names(self) -> list[str]:
        return [p.header.plot_name for p in self.plots]

    def get_nr_plots(self) -> int:
        return len(self.plots)

    def get_raw_property(
        self, property_name: str | None = None
    ) -> str | list[str] | dict[str, str | list[str]]:
        return self.plots[self.plot_index].get_raw_property(property_name)

    def get_trace_names(self) -> list[str]:
        return self.plots[self.plot_index].get_trace_names()

    def get_trace(self, trace_ref: str | int) -> DecodedTrace:
        return self.plots[self.plot_index].get_trace(trace_ref)

    def get_wave(self, trace_ref: str | int, step: int = 0) -> NDArray:
        return self.plots[self.plot_index].get_wave(trace_ref, step)

    def get_axis(self, step: int = 0) -> NDArray:
        return self.plots[self.plot_index].get_axis(step)

    def get_steps(self, **kwargs: StepValue) -> list[int]:
        return self.plots[self.plot_index].get_steps(**kwargs)
