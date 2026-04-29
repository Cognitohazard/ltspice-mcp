"""Simulation-result analysis tools.

All tools in this module consume simulation output files (.raw, .log) and
return derived metrics. Organized by what the tool answers:

    Scalar summaries:
        ltspice_signal_stats        — mean/RMS/pk-pk/etc for one signal
        ltspice_query_value         — value at a specific time/frequency
        ltspice_operating_point     — DC node voltages + branch currents

    Waveform metrics (transient only, reject AC):
        ltspice_edge_metrics        — rise/fall time + slew rate
        ltspice_pulse_response      — overshoot/undershoot/settling
        ltspice_timing_between      — signed delay between two signals
        ltspice_periodic_metrics    — period/frequency/duty/jitter

    .MEAS extraction:
        ltspice_measurement_stats   — aggregate .MEAS across sweep/MC
                                       (single-run .MEAS values are folded
                                        into ltspice_simulation_summary)

    High-level overview:
        ltspice_simulation_summary  — sim type, signals, warnings, key metrics
"""

import contextlib
import math
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from pydantic import Field

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.ac_analysis import (
    CrossingWithQuantity,
    FilterMetricsOutput,
    GainAtPoint,
    Quantity,
    ResonancesOutput,
    RollOffOutput,
    SearchDirection,
    StabilityMetricsOutput,
    compute_filter_metrics,
    compute_resonances,
    compute_roll_off,
    compute_stability_metrics,
    find_crossings_any_quantity,
    gain_at_frequencies,
    prepare_ac_arrays,
)
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.raw_parser import (
    OperatingPointOutput,
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    detect_sim_type,
    extract_operating_point,
    is_ac_analysis,
    query_point_value,
    safe_magnitude_db,
)
from ltspice_mcp.lib.signal_analysis import (
    EdgeMetricsOutput,
    MeasurementStatsEntry,
    PeriodicMetricsOutput,
    PulseResponseOutput,
    TimingBetweenOutput,
    analyze_edge,
    analyze_periodic,
    analyze_pulse_response,
    analyze_timing_between,
    compute_measurement_stats,
    compute_signal_stats,
    window_and_clean,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    MEAS_ERRORS_SCHEMA,
    RO_ANNOTATIONS,
    ToolInput,
    format_meas_errors,
    format_response,
    registry,
    safe_path,
)

FormatField = Literal["json", "text"] | None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _parse_time(s: str | None, name: str) -> float | None:
    """Parse a SPICE-notation time value; return None if input is None."""
    if s is None:
        return None
    try:
        v = parse_spice_value(s)
    except ValueError as e:
        raise ResultError(f"Invalid {name} value: {e}") from e
    if not math.isfinite(v):
        raise ResultError(f"{name} must be finite, got {s!r}")
    return v


def _reject_ac(raw) -> None:
    sim_type = detect_sim_type(raw)
    if is_ac_analysis(sim_type):
        raise ResultError(
            f"This tool requires transient analysis data; got {sim_type!r}. "
            "Use ltspice_signal_stats or ltspice_simulation_summary for "
            "frequency-domain analysis."
        )


def _load_real_signal(
    raw_file: str, signal: str, step: int, state: SessionState
) -> tuple[np.ndarray, np.ndarray]:
    """Load (axis, wave) for a signal, rejecting AC/complex data."""
    raw_path = safe_path(raw_file, state)
    raw = services.load_raw(raw_path, state)
    _reject_ac(raw)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)
    axis = np.asarray(raw.get_axis(step=step))
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    wave = np.asarray(raw.get_wave(signal, step=step))
    if np.iscomplexobj(wave):
        raise ResultError(
            f"Signal {signal!r} contains complex values; this tool requires "
            "real-valued transient data."
        )
    return axis, wave


def _window(
    axis: np.ndarray,
    wave: np.ndarray,
    t_start: str | None,
    t_end: str | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    ts = _parse_time(t_start, "t_start")
    te = _parse_time(t_end, "t_end")
    try:
        return window_and_clean(axis, wave, ts, te)
    except ValueError as e:
        raise ResultError(str(e)) from e


def _run(compute, *args, **kwargs) -> dict:
    """Invoke a pure-compute function, re-raising ValueError as ResultError."""
    try:
        return compute(*args, **kwargs)
    except ValueError as e:
        raise ResultError(str(e)) from e


def _warning_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["", "Warnings:", *(f"  - {w}" for w in warnings)]


class SignalStatsInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    step: int = Field(default=0, description="Step index for .step directives")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start in SPICE notation (e.g. '1m', '100u'). Transient only. "
            "Strongly recommended when computing RMS or average — the startup "
            "transient otherwise biases the result. Ignored for AC analysis."
        ),
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation. Transient only; ignored for AC.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class QueryValueInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    at: str = Field(
        description="Time or frequency to query in SPICE notation (e.g., '1m', '100u', '1G', '2.5k')"
    )
    step: int = Field(default=0, description="Step index for .step directives")
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class OperatingPointInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    step: int = Field(
        default=0,
        description=(
            "Step index for stepped .OP runs (e.g. ``.step temp ...`` + ``.op``). "
            "Default 0 returns the first step. Out-of-range values raise a "
            "structured error rather than silently returning the wrong step."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


class SimulationSummaryInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    log_file: str | None = Field(
        default=None,
        description=(
            "Optional path to .log file. Defaults to ``raw_file`` with the "
            "extension swapped to ``.log`` — pass an explicit value only if "
            "the log lives somewhere unusual."
        ),
    )
    signal: str | None = Field(
        default=None,
        description="Signal for AC bandwidth metrics (e.g., 'V(outp)'). Required for AC analysis.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="ltspice_signal_stats",
    description=(
        "Scalar summary of one signal in a .raw result. Use this when you need "
        "a single number per metric (average, RMS, peak, etc.) — not a waveform "
        "or a trend.\n\n"
        "Transient: time-weighted mean, RMS, std, abs-mean, and min/max/pk-pk "
        "using trapezoidal integration (RMS = sqrt(∫ y² dt / T)). This is "
        "correct on LTspice's adaptive timestep — simple np.mean(y) would "
        "overweight densely sampled regions. Optionally restrict to "
        "[t_start, t_end]; passing no window averages the whole waveform "
        "including any startup transient, which is usually wrong for RMS/mean.\n\n"
        "DC: returns min/max/pk-pk and the simple/abs mean over the swept "
        "axis, plus ``sweep_start_used``/``sweep_end_used``/``sweep_span``. "
        "RMS and std are deliberately omitted — they're meaningless on a "
        "non-time axis. Use t_start/t_end to restrict the sweep range.\n\n"
        "AC: returns magnitude (dB) min/max/mean and phase (deg) min/max. "
        "t_start/t_end are ignored for AC — use ltspice_query_value for a "
        "point at a specific frequency.\n\n"
        "Related tools: for rise/fall times use ltspice_edge_metrics; for "
        "overshoot/settling use ltspice_pulse_response; for period/duty use "
        "ltspice_periodic_metrics; to aggregate .MEAS values across a sweep "
        "use ltspice_measurement_stats."
    ),
    input_model=SignalStatsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            "analysis_type": {"type": "string"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "mean": {"type": "number"},
            "rms": {"type": "number"},
            "std": {"type": "number"},
            "abs_mean": {"type": "number"},
            "peak_to_peak": {"type": "number"},
            "point_count": {"type": "integer"},
            # Transient-only window metadata
            "t_start_used": {"type": ["number", "null"]},
            "t_end_used": {"type": ["number", "null"]},
            "duration": {"type": ["number", "null"]},
            # DC-sweep window metadata (axis is the swept variable, not time)
            "sweep_start_used": {"type": ["number", "null"]},
            "sweep_end_used": {"type": ["number", "null"]},
            "sweep_span": {"type": ["number", "null"]},
            # AC-only fields
            "min_db": {"type": "number"},
            "max_db": {"type": "number"},
            "mean_db": {"type": "number"},
            "min_phase": {"type": "number"},
            "max_phase": {"type": "number"},
        },
    },
)
async def handle_signal_stats(args: SignalStatsInput, state: SessionState):
    raw_path = safe_path(args.raw_file, state)
    signal = args.signal
    step = args.step
    fmt = args.format

    raw = services.load_raw(raw_path, state)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)

    try:
        wave = raw.get_wave(signal, step=step)
    except Exception as e:
        raise ResultError(f"Failed to read signal {signal!r}: {e}") from e
    if len(wave) == 0:
        raise ResultError(
            f"Signal {signal!r} has no data points at step {step}; cannot compute statistics."
        )

    if np.iscomplexobj(wave):
        if args.t_start is not None or args.t_end is not None:
            raise ResultError(
                "t_start/t_end windowing is not supported for AC analysis. "
                "Use ltspice_query_value to look up a specific frequency."
            )
        magnitude_db = safe_magnitude_db(wave)
        phase_deg = np.angle(wave, deg=True)
        stats = {
            "analysis_type": "ac",
            "min_db": float(np.min(magnitude_db)),
            "max_db": float(np.max(magnitude_db)),
            "mean_db": float(np.mean(magnitude_db)),
            "min_phase": float(np.min(phase_deg)),
            "max_phase": float(np.max(phase_deg)),
            "point_count": len(wave),
        }
        lines = [
            f"Signal: {signal} (AC Analysis)",
            "",
            "Magnitude (dB):",
            f"  Min: {stats['min_db']:.2f} dB",
            f"  Max: {stats['max_db']:.2f} dB",
            f"  Mean: {stats['mean_db']:.2f} dB",
            "",
            "Phase:",
            f"  Min: {stats['min_phase']:.2f} deg",
            f"  Max: {stats['max_phase']:.2f} deg",
            "",
            f"Data Points: {stats['point_count']}",
        ]
        return format_response("\n".join(lines), {"signal": signal, **stats}, fmt)

    axis = np.asarray(raw.get_axis(step=step))
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    wave_real = np.asarray(wave)

    # Bug E: distinguish DC sweep (axis = sweep variable, e.g. ``temp``)
    # from transient (axis = time). Trapezoidal mean/RMS over a sweep axis
    # is mathematically meaningless; the t_start/t_end labels are misleading
    # too since the units aren't seconds.
    sim_type = detect_sim_type(raw).lower()
    is_dc_sweep = "dc transfer" in sim_type or "dc " in sim_type

    ts = _parse_time(args.t_start, "t_start")
    te = _parse_time(args.t_end, "t_end")
    try:
        t_win, y_win, _ = window_and_clean(axis, wave_real, ts, te)
    except ValueError as e:
        raise ResultError(str(e)) from e

    try:
        core = compute_signal_stats(t_win, y_win)
    except ValueError as e:
        raise ResultError(str(e)) from e

    if is_dc_sweep:
        stats = {
            "analysis_type": "dc",
            "min": core["min"],
            "max": core["max"],
            "mean": core["mean"],
            "abs_mean": core["abs_mean"],
            "peak_to_peak": core["pk_pk"],
            "point_count": core["num_samples"],
            "sweep_start_used": core["t_start"],
            "sweep_end_used": core["t_end"],
            "sweep_span": core["duration"],
        }
    else:
        stats = {
            "analysis_type": "transient",
            "min": core["min"],
            "max": core["max"],
            "mean": core["mean"],
            "rms": core["rms"],
            "std": core["std"],
            "abs_mean": core["abs_mean"],
            "peak_to_peak": core["pk_pk"],
            "point_count": core["num_samples"],
            "t_start_used": core["t_start"],
            "t_end_used": core["t_end"],
            "duration": core["duration"],
        }
    window_note = (
        f" (window [{core['t_start']:.6g}, {core['t_end']:.6g}] s)"
        if ts is not None or te is not None
        else ""
    )
    lines = [
        f"Signal: {signal}{window_note}",
        f"Min:          {stats['min']:.6g}",
        f"Max:          {stats['max']:.6g}",
        f"Peak-to-Peak: {stats['peak_to_peak']:.6g}",
        f"Mean:         {stats['mean']:.6g}",
    ]
    if "rms" in stats:
        lines.append(f"RMS:          {stats['rms']:.6g}")
    if "std" in stats:
        lines.append(f"Std:          {stats['std']:.6g}")
    lines.append(f"Abs mean:     {stats['abs_mean']:.6g}")
    if "duration" in stats:
        lines.append(f"Duration:     {stats['duration']:.6g} s  ({stats['point_count']} samples)")
    elif "sweep_span" in stats:
        lines.append(f"Sweep span:   {stats['sweep_span']:.6g}  ({stats['point_count']} samples)")
    return format_response("\n".join(lines), {"signal": signal, **stats}, fmt)


@registry.tool(
    name="ltspice_query_value",
    description=(
        "Look up the value of a signal at a specific time point (transient) or "
        "frequency (AC). Returns the nearest data point without interpolation."
    ),
    input_model=QueryValueInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            "requested_x": {"type": "number"},
            "actual_x": {"type": "number"},
            "value": {"type": "number"},
            "magnitude_db": {"type": "number"},
            "phase_deg": {"type": "number"},
        },
    },
)
async def handle_query_value(args: QueryValueInput, state: SessionState):
    """Query signal value at a specific time or frequency."""
    raw_path = safe_path(args.raw_file, state)
    signal = args.signal
    at_str = args.at
    step = args.step
    fmt = args.format

    try:
        target_x = parse_spice_value(at_str)
    except ValueError as e:
        raise ResultError(f"Invalid 'at' value: {e}") from e

    # np.searchsorted treats NaN as greater than everything and returns the
    # last index, which looks like a valid result but isn't.
    if not math.isfinite(target_x):
        raise ResultError(f"'at' value must be finite, got {at_str!r} (parsed as {target_x})")

    raw = services.load_raw(raw_path, state)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)

    try:
        result_data = query_point_value(raw, signal, target_x, step)
    except Exception as e:
        raise ResultError(f"Failed to query value: {e}") from e

    sim_type = detect_sim_type(raw)
    x_unit = "f" if is_ac_analysis(sim_type) else "t"

    if "magnitude_db" in result_data:
        lines = [
            f"Signal: {signal} at {x_unit}={result_data['requested_x']:.6g}",
            f"Requested: {result_data['requested_x']:.6g}",
            f"Nearest point: {result_data['actual_x']:.6g}",
            f"Magnitude: {result_data['magnitude_db']:.2f} dB",
            f"Phase: {result_data['phase_deg']:.2f} deg",
        ]
    else:
        lines = [
            f"Signal: {signal} at {x_unit}={result_data['requested_x']:.6g}",
            f"Requested: {result_data['requested_x']:.6g}",
            f"Nearest point: {result_data['actual_x']:.6g}",
            f"Value: {result_data['value']:.6g}",
        ]

    return format_response("\n".join(lines), {"signal": signal, **result_data}, fmt)


def _format_measurements(
    measurements: dict, step_count: int, errors: list[str] | None = None
) -> str:
    """Format .MEAS results for display. Shared between handlers.

    Accepts the new structured shape (``{name: {"values": [...], ...}}``)
    where each entry may carry ``range_from`` / ``range_to`` / ``at`` metadata.
    """
    if not measurements:
        if errors:
            lines = ["No .MEAS results — errors in log:", ""]
            for err in errors:
                lines.append(f"  {err}")
            return "\n".join(lines)
        return "No .MEAS results found in log file"

    def _meta_suffix(entry: dict) -> str:
        bits: list[str] = []
        if entry.get("range_from") is not None or entry.get("range_to") is not None:
            lo = entry.get("range_from")
            hi = entry.get("range_to")
            if lo is not None and hi is not None:
                bits.append(f"FROM={lo:g} TO={hi:g}")
            elif lo is not None:
                bits.append(f"FROM={lo:g}")
            elif hi is not None:
                bits.append(f"TO={hi:g}")
        if entry.get("at") is not None:
            bits.append(f"AT={entry['at']:g}")
        return f"  ({', '.join(bits)})" if bits else ""

    if step_count <= 1:
        lines = [".MEAS Results:", ""]
        for name, entry in measurements.items():
            values = entry.get("values", [])
            value = values[0] if values else None
            suffix = _meta_suffix(entry)
            if value is None:
                lines.append(f"  {name} = FAILED{suffix}")
            else:
                lines.append(f"  {name} = {value:.6g}{suffix}")
    else:
        lines = [f".MEAS Results ({step_count} steps):", ""]
        for name, entry in measurements.items():
            values = entry.get("values", [])
            value_strs: list[str] = []
            for val in values:
                if val is None:
                    value_strs.append("FAILED")
                else:
                    value_strs.append(f"{val:.6g}")
            suffix = _meta_suffix(entry)
            lines.append(f"  {name}: [{', '.join(value_strs)}]{suffix}")

    return "\n".join(lines)


@registry.tool(
    name="ltspice_operating_point",
    description=("Read DC operating point data showing all node voltages and branch currents."),
    input_model=OperatingPointInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=OperatingPointOutput,
)
async def handle_operating_point(args: OperatingPointInput, state: SessionState):
    """Read DC operating point data (all node voltages and branch currents)."""
    raw_path = safe_path(args.raw_file, state)
    fmt = args.format
    raw = services.load_raw(raw_path, state)

    sim_type = detect_sim_type(raw)
    # ``extract_operating_point`` reads ``wave[step]`` for every trace. That's
    # the DC bias point only for ``.OP`` (and ``.DC`` — point 0 is the
    # sweep's starting bias). For AC/Noise it's the magnitude at the first
    # frequency point — the "voltages" returned would be AC magnitudes
    # (e.g. ``V(in)=1`` from an ``AC 1`` source). For Transient it's t=0
    # which may include initial conditions, not the converged op-point.
    sim_lower = sim_type.lower()
    if "ac" in sim_lower.split() or "noise" in sim_lower:
        raise ResultError(
            f"Cannot extract DC operating point from {sim_type!r}: the first "
            "point in an AC/Noise raw is the magnitude at the lowest frequency, "
            "not the bias. Run a separate ``.OP`` analysis to capture the bias "
            "point, or read it from the simulation .log."
        )
    if "transient" in sim_lower:
        raise ResultError(
            f"Cannot extract DC operating point from {sim_type!r}: the first "
            "transient point is at t=0 and reflects initial conditions, not "
            "the converged DC bias. Run a separate ``.OP`` analysis."
        )

    services.validate_step(raw, args.step)
    op_step_count = services.get_step_count(raw)
    op_data: dict

    try:
        op_data = dict(extract_operating_point(raw, step=args.step))
    except Exception as e:
        raise ResultError(f"Failed to extract operating point: {e}") from e

    op_data["step"] = args.step
    op_data["step_count"] = op_step_count

    lines = ["DC Operating Point", ""]
    if op_step_count > 1:
        lines.append(
            f"Step {args.step} of {op_step_count} (use step=N to read other "
            "iterations of stepped .OP runs)"
        )
        lines.append("")

    if op_data["voltages"]:
        lines.append("Node Voltages:")
        for name, value in op_data["voltages"].items():
            lines.append(f"  {name} = {value:.6g}")
        lines.append("")

    if op_data["currents"]:
        lines.append("Branch Currents:")
        for name, value in op_data["currents"].items():
            lines.append(f"  {name} = {value:.6g}")

    return format_response("\n".join(lines), op_data, fmt)


@registry.tool(
    name="ltspice_simulation_summary",
    description=(
        "Get a comprehensive simulation summary including type, signal list, data size, "
        ".MEAS results, Fourier analysis, AC bandwidth metrics, and warnings."
    ),
    input_model=SimulationSummaryInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "sim_type": {"type": "string"},
            "range": {"type": "object"},
            "point_count": {"type": "integer"},
            "step_count": {"type": "integer"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "measurements": {
                "type": "object",
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "values": {
                            "type": "array",
                            "items": {"type": ["number", "null"]},
                        },
                        "range_from": {"type": ["number", "null"]},
                        "range_to": {"type": ["number", "null"]},
                        "at": {"type": ["number", "null"]},
                    },
                    "required": ["values"],
                },
            },
            "fourier": {"type": "array", "items": {"type": "object"}},
            "ac_bandwidth_metrics": {
                "type": "object",
                "properties": {
                    "bandwidth_3db": {"type": ["number", "null"]},
                    "unity_gain_freq": {"type": ["number", "null"]},
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
            "errors": {"type": "array", "items": {"type": "string"}},
            "meas_errors": MEAS_ERRORS_SCHEMA,
        },
    },
)
async def handle_simulation_summary(args: SimulationSummaryInput, state: SessionState):
    """Get comprehensive simulation summary."""
    raw_path = safe_path(args.raw_file, state)
    fmt = args.format
    log_path = None
    if args.log_file is not None:
        log_path = safe_path(args.log_file, state)
    else:
        # Friction A: every analysis call shouldn't have to plumb both
        # ``raw_file`` and the adjacent ``.log``. Auto-derive when missing.
        derived = raw_path.with_suffix(".log")
        if derived.exists():
            log_path = derived

    raw = services.load_raw(raw_path, state)

    try:
        summary = build_simulation_summary(raw, log_path, None)
    except Exception as e:
        raise ResultError(f"Failed to build summary: {e}") from e

    suggestions = services.suggestions_from_errors(summary.get("errors"), state.libraries)
    if suggestions:
        summary["suggestions"] = suggestions

    # Compute AC bandwidth metrics only when signal is explicitly specified
    ac_metrics = None
    if is_ac_analysis(summary["sim_type"]) and args.signal:
        with contextlib.suppress(Exception):
            ac_metrics = compute_ac_bandwidth_metrics(raw, args.signal, 0)

    json_data = dict(summary)
    if ac_metrics:
        json_data["ac_bandwidth_metrics"] = ac_metrics

    if fmt == "json":
        return format_response("", json_data, fmt)

    # Text formatting (skip when json)
    lines = [f"Simulation Summary: {summary['sim_type']}", ""]

    if "time_start" in summary["range"]:
        lines.append(
            f"Time span: {summary['range']['time_start']:.6g} to {summary['range']['time_end']:.6g}"
        )
    elif "freq_start" in summary["range"]:
        lines.append(
            f"Frequency range: {summary['range']['freq_start']:.6g} to {summary['range']['freq_end']:.6g}"
        )
    elif "sweep_start" in summary["range"]:
        lines.append(
            f"DC sweep: {summary['range']['sweep_start']:.6g} to {summary['range']['sweep_end']:.6g}"
        )

    lines.append(
        f"Data points: {summary['point_count']} per signal, {summary['step_count']} step(s)"
    )
    lines.append("")

    lines.append(f"Signals ({len(summary['signals'])}):")
    for signal in summary["signals"]:
        lines.append(f"  - {signal}")
    lines.append("")

    if "measurements" in summary:
        lines.append(
            _format_measurements(
                summary["measurements"],
                summary.get("step_count", 1),
            )
        )
        lines.append("")

    if "fourier" in summary:
        lines.append("Fourier Analysis:")
        for fourier in summary["fourier"]:
            lines.append(f"  Signal: {fourier['signal']}")
            if fourier["thd"] is not None:
                lines.append(f"  THD: {fourier['thd']:.2f}%")
            if fourier["fundamental_frequency"] is not None:
                lines.append(f"  Fundamental: {fourier['fundamental_frequency']:.6g} Hz")
            if fourier["harmonics"]:
                lines.append("  Harmonics:")
                for harm in fourier["harmonics"][:10]:
                    lines.append(
                        f"    {harm['number']}: {harm['frequency']:.6g} Hz, "
                        f"{harm['magnitude']:.6g}, {harm['phase']:.2f} deg"
                    )
                if len(fourier["harmonics"]) > 10:
                    lines.append(f"    ... ({len(fourier['harmonics'])} total)")
        lines.append("")

    if ac_metrics:
        lines.append("AC Bandwidth Metrics:")
        if ac_metrics["bandwidth_3db"] is not None:
            lines.append(f"  -3dB point: {ac_metrics['bandwidth_3db']:.6g} Hz")
        if ac_metrics["unity_gain_freq"] is not None:
            lines.append(f"  Unity-gain frequency: {ac_metrics['unity_gain_freq']:.6g} Hz")
        lines.append("")

    if "errors" in summary:
        lines.append(f"Errors ({len(summary['errors'])}):")
        for error in summary["errors"]:
            lines.append(f"  {error}")
        lines.append("")

    if "warnings" in summary:
        lines.append(f"Warnings ({len(summary['warnings'])}):")
        for warning in summary["warnings"]:
            lines.append(f"  {warning}")
        lines.append("")

    meas_lines = format_meas_errors(summary.get("meas_errors", []))
    if meas_lines:
        lines.extend(meas_lines)
        lines.append("")

    return format_response("\n".join(lines), json_data, fmt)


class EdgeMetricsInput(ToolInput):
    raw_file: str = Field(description="Path to .raw transient result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start time in SPICE notation (e.g. '1m', '100u'). Strongly "
            "recommended when the transient contains startup transients or "
            "multiple edges — otherwise the first edge in the full waveform is "
            "measured (often the power-up glitch)."
        ),
    )
    t_end: str | None = Field(default=None, description="Window end time in SPICE notation")
    edge: Literal["rising", "falling", "auto"] = Field(
        default="auto",
        description="Edge direction. 'auto' infers from window endpoints.",
    )
    edge_index: int = Field(
        default=0,
        description="Which matching edge in the window (0 = first). Use with tight t_start/t_end for determinism.",
    )
    low_pct: float = Field(default=10.0, description="Low threshold percent (default 10%)")
    high_pct: float = Field(default=90.0, description="High threshold percent (default 90%)")
    format: FormatField = Field(default=None, description="'json' or 'text'")


class PulseResponseInput(ToolInput):
    raw_file: str = Field(description="Path to .raw transient result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description="Window start — ideally the stimulus edge. Defaults to full transient.",
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    initial_value: float | None = Field(
        default=None,
        description="Pre-step steady value. Auto = mean of first 10% of window. Set explicitly if the start is contaminated by ringing.",
    )
    final_value: float | None = Field(
        default=None,
        description="Post-step steady value. Auto = mean of last 10% of window.",
    )
    settling_tolerance_pct: float = Field(
        default=2.0,
        description="Settling band as percent of |final - initial|. 2% is standard; 1% or 5% also common.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class TimingBetweenInput(ToolInput):
    raw_file: str = Field(description="Path to .raw transient result file")
    signal_a: str = Field(description="Reference signal (e.g. 'V(in)')")
    signal_b: str = Field(description="Delayed signal (e.g. 'V(out)'). delay = t_b - t_a.")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(default=None, description="Window start in SPICE notation")
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    threshold_a: float | None = Field(
        default=None,
        description="Absolute threshold for signal_a. If omitted, threshold_pct of signal_a's range is used.",
    )
    threshold_b: float | None = Field(
        default=None,
        description="Absolute threshold for signal_b. If omitted, threshold_pct of signal_b's range is used.",
    )
    threshold_pct: float = Field(
        default=50.0,
        description="Threshold percent applied PER SIGNAL (not shared) — asymmetric for CMOS with different rails.",
    )
    direction_a: Literal["rising", "falling"] = Field(default="rising")
    direction_b: Literal["rising", "falling"] = Field(default="rising")
    format: FormatField = Field(default=None, description="'json' or 'text'")


class PeriodicMetricsInput(ToolInput):
    raw_file: str = Field(description="Path to .raw transient result file")
    signal: str = Field(description="Signal name (e.g. 'V(clk)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description="Window start — recommended to skip the startup transient.",
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    threshold: float | None = Field(
        default=None,
        description="Absolute threshold level. Auto = midpoint of window min/max. For drifting signals, set explicitly.",
    )
    min_periods: int = Field(
        default=2,
        description="Minimum complete periods required; error if window has fewer.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class MeasurementStatsInput(ToolInput):
    log_file: str | None = Field(
        default=None,
        description=(
            "Path to .log file from a single ``.step`` run that already "
            "concatenates every step's .MEAS results. For Monte Carlo / "
            "multi-run sweep jobs that emit one log per run, pass ``job_id`` "
            "instead and the aggregator walks every run's log."
        ),
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Batch job ID from ``run_montecarlo`` / ``run_sweep``. The tool "
            "loads each completed run's log, concatenates the .MEAS results "
            "(one row per run), and aggregates. Mutually exclusive with "
            "``log_file``."
        ),
    )
    measurement: str | None = Field(
        default=None,
        description="If given, stats for only this .MEAS; otherwise all measurements.",
    )
    histogram_bins: int = Field(
        default=10,
        description="Histogram bin count. Set to 0 to skip histogram computation.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


# ---------------------------------------------------------------------------
# Response TypedDicts — compose lib output + per-tool metadata (signal names)
# ---------------------------------------------------------------------------


class EdgeMetricsResponse(EdgeMetricsOutput):
    signal: str


class PulseResponseResponse(PulseResponseOutput):
    signal: str


class TimingBetweenResponse(TimingBetweenOutput):
    signal_a: str
    signal_b: str


class PeriodicMetricsResponse(PeriodicMetricsOutput):
    signal: str


AggregatedField = Literal["value", "at"]


class MeasurementStatsResponseEntry(MeasurementStatsEntry):
    """Per-measurement stats entry as returned by the MCP layer.

    Adds ``aggregated_field`` to the lib-level :class:`MeasurementStatsEntry`
    so callers can tell which per-run scalar (the trigger level or the
    WHEN-clause crossing point) the stats describe.
    """

    aggregated_field: AggregatedField


class MeasurementStatsResponse(TypedDict):
    stats: dict[str, MeasurementStatsResponseEntry]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


@registry.tool(
    name="ltspice_edge_metrics",
    description=(
        "Use when you need to quantify HOW FAST one transition happened: rise "
        "time, fall time, slew rate. Inputs a transient .raw plus a time "
        "window around the edge of interest.\n\n"
        "Returns: transition_time (10→90% by default, configurable via "
        "low_pct/high_pct), slew_rate (V/s or A/s), detected low/high levels, "
        "and the three crossing times.\n\n"
        "Levels are auto-estimated from the first/last 10% of the window — "
        "NOT global min/max — so overshoot/undershoot doesn't poison the "
        "level estimate. Crossings are sub-sample-accurate via linear "
        "interpolation. Rejects AC analysis.\n\n"
        "PICK THE WINDOW. If the transient has startup glitches or multiple "
        "edges, set t_start/t_end tightly around the edge you care about — "
        "otherwise you get the first edge in the full waveform, which is "
        "often the power-up artifact. Use edge_index only when multiple "
        "edges in the window are intentional.\n\n"
        "For settling/overshoot after the edge, use ltspice_pulse_response. "
        "For delay between two signals' edges, use ltspice_timing_between."
    ),
    input_model=EdgeMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=EdgeMetricsResponse,
)
async def handle_edge_metrics(args: EdgeMetricsInput, state: SessionState):
    axis, wave = _load_real_signal(args.raw_file, args.signal, args.step, state)
    t, y, _ = _window(axis, wave, args.t_start, args.t_end)
    data = _run(
        analyze_edge,
        t,
        y,
        edge=args.edge,
        edge_index=args.edge_index,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
    )
    data["signal"] = args.signal

    label = "Rise time" if data["is_rise_time"] else "Fall time"
    lines = [
        f"Edge Metrics: {args.signal} ({data['edge_direction']} edge "
        f"{args.edge_index} of {data['num_edges_in_window']})",
        "",
        f"{label} ({data['low_pct']:.0f}%-{data['high_pct']:.0f}%): "
        f"{data['transition_time']:.6g} s",
        f"Slew rate: {data['slew_rate']:.6g} (units/s)",
        f"Low level: {data['low_level']:.6g}",
        f"High level: {data['high_level']:.6g}",
        f"t(low): {data['t_low_crossing']:.6g} s",
        f"t(high): {data['t_high_crossing']:.6g} s",
        f"t(mid): {data['t_mid_crossing']:.6g} s",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_pulse_response",
    description=(
        "Use when you need step-response quality metrics: overshoot %, "
        "undershoot %, settling time, peak value and peak time. Inputs a "
        "transient .raw covering ONE step transition — ideally with the "
        "stimulus edge near t_start and enough tail to see settling.\n\n"
        "Returns: direction (rising/falling), initial/steady-state values, "
        "peak (absolute and pct), settling_time (to within "
        "settling_tolerance_pct band; null if never settled in window).\n\n"
        "Definitions: overshoot is excursion BEYOND final in the step "
        "direction; undershoot is excursion beyond initial opposite the "
        "step direction. overshoot_pct = 0 means MEASURED overdamped, not "
        "missing data. settling_tolerance_pct defaults to 2% of "
        "|final - initial|; 1% and 5% are also common.\n\n"
        "If the auto-detected initial/final (mean of first/last 10% of "
        "window) is contaminated by ringing, pass explicit initial_value/"
        "final_value. Rejects AC analysis.\n\n"
        "For just rise/fall time without overshoot, use ltspice_edge_metrics."
    ),
    input_model=PulseResponseInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=PulseResponseResponse,
)
async def handle_pulse_response(args: PulseResponseInput, state: SessionState):
    axis, wave = _load_real_signal(args.raw_file, args.signal, args.step, state)
    t, y, _ = _window(axis, wave, args.t_start, args.t_end)
    data = _run(
        analyze_pulse_response,
        t,
        y,
        initial_value=args.initial_value,
        final_value=args.final_value,
        settling_tolerance_pct=args.settling_tolerance_pct,
    )
    data["signal"] = args.signal

    settle = (
        "never (within window)"
        if data["settling_time"] is None
        else f"{data['settling_time']:.6g} s"
    )
    lines = [
        f"Pulse Response: {args.signal} ({data['direction']} step)",
        "",
        f"Initial: {data['initial_value']:.6g}",
        f"Final:   {data['steady_state_value']:.6g}",
        f"Peak:    {data['peak_value']:.6g} at t={data['peak_time']:.6g} s",
        f"Overshoot:  {data['overshoot_pct']:.3f} %",
        f"Undershoot: {data['undershoot_pct']:.3f} %",
        f"Settling time (±{data['settling_tolerance_pct']:.2f}%): {settle}",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_timing_between",
    description=(
        "Use when you need propagation delay / skew between TWO signals — "
        "e.g. input-to-output delay, clock-to-Q, input-skew. Inputs one "
        "transient .raw containing both signals on a shared time axis.\n\n"
        "Returns: signed delay = t_b - t_a where t_a and t_b are the FIRST "
        "threshold crossings of signal_a and signal_b in the window. "
        "Negative delay means signal_b leads signal_a.\n\n"
        "Thresholds default to 50% of EACH signal's own min-max range in the "
        "window — intentional for asymmetric CMOS where V_in and V_out have "
        "different rails. Override per-signal via threshold_a / threshold_b "
        "if you need absolute thresholds (e.g. VIH/VIL at fixed voltages). "
        "Set direction_a / direction_b independently (e.g. rising input → "
        "falling output for an inverter).\n\n"
        "Picks only the FIRST crossing of each signal in the window — if "
        "both signals have multiple edges, tighten t_start/t_end around the "
        "specific edge pair you want. Rejects AC analysis."
    ),
    input_model=TimingBetweenInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=TimingBetweenResponse,
)
async def handle_timing_between(args: TimingBetweenInput, state: SessionState):
    raw_path = safe_path(args.raw_file, state)
    raw = services.load_raw(raw_path, state)
    _reject_ac(raw)
    sig_a = services.validate_signal(raw, args.signal_a)
    sig_b = services.validate_signal(raw, args.signal_b)
    services.validate_step(raw, args.step)

    axis = np.asarray(raw.get_axis(step=args.step))
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    ya_full = np.asarray(raw.get_wave(sig_a, step=args.step))
    yb_full = np.asarray(raw.get_wave(sig_b, step=args.step))
    if np.iscomplexobj(ya_full) or np.iscomplexobj(yb_full):
        raise ResultError(
            "Signals contain complex values; this tool requires real-valued transient data."
        )

    ts = _parse_time(args.t_start, "t_start")
    te = _parse_time(args.t_end, "t_end")

    try:
        t_a_arr, ya, _ = window_and_clean(axis, ya_full, ts, te)
        t_b_arr, yb, _ = window_and_clean(axis, yb_full, ts, te)
    except ValueError as e:
        raise ResultError(str(e)) from e

    # Both windows come from the same axis; indices match, but re-confirm
    # defensively against downstream shape drift.
    if len(t_a_arr) != len(t_b_arr):
        raise ResultError(
            "Internal error: windowed axes have different lengths for the two signals"
        )

    data = _run(
        analyze_timing_between,
        t_a_arr,
        ya,
        yb,
        threshold_a=args.threshold_a,
        threshold_b=args.threshold_b,
        threshold_pct=args.threshold_pct,
        direction_a=args.direction_a,
        direction_b=args.direction_b,
    )
    data["signal_a"] = args.signal_a
    data["signal_b"] = args.signal_b

    lines = [
        f"Timing: {args.signal_a} ({data['direction_a']}) → "
        f"{args.signal_b} ({data['direction_b']})",
        "",
        f"t({args.signal_a}) = {data['t_a']:.6g} s @ threshold={data['threshold_a_used']:.6g}",
        f"t({args.signal_b}) = {data['t_b']:.6g} s @ threshold={data['threshold_b_used']:.6g}",
        f"Delay (t_b - t_a): {data['delay']:.6g} s",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_periodic_metrics",
    description=(
        "Use for an oscillating transient signal (clock, oscillator output, "
        "switching waveform) when you need period, frequency, duty cycle, "
        "pulse widths, and period-to-period jitter.\n\n"
        "Returns: period (mean across measured periods), frequency (1/period), "
        "jitter_rms (std-dev of period lengths — timing jitter, NOT signal "
        "amplitude variance), duty_cycle_pct, mean high/low pulse widths, "
        "edge counts. duty_cycle_pct / pulse_widths are null if no full "
        "periods could be paired.\n\n"
        "Uses threshold crossings; threshold defaults to the midpoint of "
        "window min/max. For a signal with DC drift, set an explicit "
        "threshold — the auto midpoint moves with the drift and the edge "
        "detection gets unstable. min_periods guards against accidentally "
        "running on 1-edge windows.\n\n"
        "Skip the startup transient via t_start/t_end; the first cycle is "
        "often wider than steady state. Rejects AC analysis. For a single "
        "edge (not periodic), use ltspice_edge_metrics."
    ),
    input_model=PeriodicMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=PeriodicMetricsResponse,
)
async def handle_periodic_metrics(args: PeriodicMetricsInput, state: SessionState):
    axis, wave = _load_real_signal(args.raw_file, args.signal, args.step, state)
    t, y, _ = _window(axis, wave, args.t_start, args.t_end)
    data = _run(
        analyze_periodic,
        t,
        y,
        threshold=args.threshold,
        min_periods=args.min_periods,
    )
    data["signal"] = args.signal

    duty = f"{data['duty_cycle_pct']:.3f} %" if data["duty_cycle_pct"] is not None else "n/a"
    high_w = f"{data['pulse_width_high']:.6g} s" if data["pulse_width_high"] is not None else "n/a"
    low_w = f"{data['pulse_width_low']:.6g} s" if data["pulse_width_low"] is not None else "n/a"
    lines = [
        f"Periodic Metrics: {args.signal}",
        "",
        f"Period:      {data['period']:.6g} s",
        f"Frequency:   {data['frequency']:.6g} Hz",
        f"Duty cycle:  {duty}",
        f"High width:  {high_w}",
        f"Low width:   {low_w}",
        f"Jitter RMS:  {data['jitter_rms']:.6g} s",
        f"Threshold:   {data['threshold_used']:.6g}",
        f"Edges: {data['num_rising_edges']} rising / "
        f"{data['num_falling_edges']} falling "
        f"({data['num_periods_measured']} period(s))",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


class _MeasSamples(TypedDict):
    """Per-name accumulator used inside :func:`_aggregate_job_measurements`."""

    values: list[float | None]
    ats: list[float | None]


def _aggregate_job_measurements(
    job_id: str, state: SessionState
) -> tuple[dict[str, list[float | None]], int, dict[str, AggregatedField]]:
    """Walk every completed run's .log and concatenate ``.MEAS`` results.

    The MC engine emits one log per run; this reconciles by collecting
    per-run scalar values keyed by .MEAS name.

    For ``WHEN``-style .MEAS, the per-run scalar in ``values`` is the
    trigger level (constant across runs by definition) — the interesting
    per-run axis lives in the folded ``at`` field. When that pattern is
    detected (constant ``values``, varying ``at``) the aggregator swaps to
    the ``at`` axis automatically.

    Returns ``(flat_values, run_count, axis_map)`` where ``axis_map[name]``
    is ``"value"`` or ``"at"`` describing which field was aggregated.
    """
    batch_job = services.resolve_batch_job(job_id, state)

    if not batch_job.run_results:
        raise ResultError(
            f"Batch job {job_id!r} has no completed runs yet — wait for it "
            "to finish (use ltspice_check_job to monitor)."
        )

    samples: dict[str, _MeasSamples] = {}
    runs_processed = 0
    for run_index in sorted(batch_job.run_results.keys()):
        run = batch_job.run_results[run_index]
        log_path_str = run.get("log_file")
        if not log_path_str:
            continue
        try:
            data = parse_measurements(Path(log_path_str))
        except Exception:
            # Missing/unreadable per-run log — skip silently. Aggregation
            # over partial runs is the documented behaviour.
            continue
        runs_processed += 1
        for name, entry in data.get("measurements", {}).items():
            row = entry.get("values", [])
            scalar = row[0] if row else None
            at_raw = entry.get("at")
            at_val = float(at_raw) if isinstance(at_raw, int | float) else None
            bucket = samples.get(name)
            if bucket is None:
                bucket = _MeasSamples(
                    values=[None] * (runs_processed - 1),
                    ats=[None] * (runs_processed - 1),
                )
                samples[name] = bucket
            bucket["values"].append(scalar)
            bucket["ats"].append(at_val)
        # Backfill any names that didn't appear in this run.
        for bucket in samples.values():
            if len(bucket["values"]) < runs_processed:
                bucket["values"].append(None)
                bucket["ats"].append(None)

    flat_values: dict[str, list[float | None]] = {}
    axis_map: dict[str, AggregatedField] = {}
    for name, bucket in samples.items():
        vals = bucket["values"]
        ats = bucket["ats"]
        # Swap to ``at`` when the level is constant (or all-None) and the
        # per-run frequency varies — the WHEN-style case.
        valid_vals = [v for v in vals if v is not None]
        valid_ats = [a for a in ats if a is not None]
        levels_constant = len({round(v, 12) for v in valid_vals}) <= 1
        ats_vary = len({round(a, 12) for a in valid_ats}) > 1
        if levels_constant and ats_vary:
            flat_values[name] = ats
            axis_map[name] = "at"
        else:
            flat_values[name] = vals
            axis_map[name] = "value"

    return flat_values, runs_processed, axis_map


@registry.tool(
    name="ltspice_measurement_stats",
    description=(
        "Use to AGGREGATE .MEAS scalar results across a .step sweep or Monte "
        "Carlo run. Answers questions like 'across 100 MC trials, what's the "
        "worst-case rise time?' or 'how does gain vary as R sweeps 1k..10k?'. "
        "Inputs the .log file produced by the run.\n\n"
        "Returns per-measurement: min, max, mean, median, std, p10, p90, "
        "best_step_index (argmin) and worst_step_index (argmax), failure "
        "count, and an optional histogram (set histogram_bins=0 to skip).\n\n"
        "Requires either a parametric sweep (.step) or Monte Carlo. On a "
        "single-run simulation there's only one value per measurement, so "
        "stats collapse to trivial values — use ltspice_simulation_summary "
        "instead to just read the scalars.\n\n"
        "Works with .MEAS from any analysis type (.tran/.ac/.dc/.op) — the "
        "measurement directives themselves embed the analysis context. Pass "
        "measurement=NAME to aggregate just one; otherwise returns all "
        ".MEAS in the log."
    ),
    input_model=MeasurementStatsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=MeasurementStatsResponse,
)
async def handle_measurement_stats(args: MeasurementStatsInput, state: SessionState):
    if args.log_file is not None and args.job_id is not None:
        raise ResultError(
            "Pass either ``log_file`` (single .step log) or ``job_id`` "
            "(walk every run's log of a Monte Carlo / sweep batch), not both."
        )
    if args.log_file is None and args.job_id is None:
        raise ResultError("Provide either ``log_file`` or ``job_id``.")

    axis_map: dict[str, AggregatedField] = {}
    if args.job_id is not None:
        flat_values, run_count, axis_map = _aggregate_job_measurements(args.job_id, state)
        if not flat_values:
            raise ResultError(f"No .MEAS results found across the runs of job {args.job_id!r}.")
        steps_label = f"{run_count} run(s)"
    elif args.log_file is not None:
        log_path = safe_path(args.log_file, state)
        try:
            meas_data = parse_measurements(log_path)
        except ResultError:
            raise
        except Exception as e:
            raise ResultError(f"Failed to parse log file: {e}") from e

        measurements = meas_data.get("measurements", {})
        if not measurements:
            errors = meas_data.get("errors") or []
            err_block = (
                "\n".join(f"  {e}" for e in errors)
                if errors
                else "  (log contained no .MEAS results and no diagnostics)"
            )
            raise ResultError(f"No .MEAS results in log:\n{err_block}")

        flat_values = {name: list(entry.get("values", [])) for name, entry in measurements.items()}
        axis_map = dict.fromkeys(flat_values, "value")
        steps_label = f"{meas_data.get('step_count', 1)} step(s)"
    else:  # unreachable — earlier guard rejects this combination
        raise ResultError("Provide either ``log_file`` or ``job_id``.")

    stats = _run(
        compute_measurement_stats,
        flat_values,
        histogram_bins=args.histogram_bins,
        measurement=args.measurement,
    )
    # Surface which field each stat block was computed from so a downstream
    # consumer can tell "this is the level (constant)" from "this is the
    # WHEN-clause crossing frequency".
    for name, entry in stats.items():
        entry["aggregated_field"] = axis_map.get(name, "value")

    lines = [
        f"Measurement Stats ({steps_label})",
        "",
    ]
    for name, entry in stats.items():
        lines.append(f"{name}:")
        field = entry.get("aggregated_field", "value")
        lines.append(
            f"  valid {entry['valid_count']}/{entry['total_count']} "
            f"(failed {entry['failure_count']})  field={field}"
        )
        if entry["valid_count"] > 0:
            lines.append(
                f"  min={entry['min']:.6g}  max={entry['max']:.6g}  "
                f"mean={entry['mean']:.6g}  median={entry['median']:.6g}  "
                f"std={entry['std']:.6g}"
            )
            lines.append(
                f"  p10={entry['p10']:.6g}  p90={entry['p90']:.6g}  "
                f"argmin step={entry['best_step_index']}  "
                f"argmax step={entry['worst_step_index']}"
            )
        lines.append("")

    return format_response("\n".join(lines).rstrip(), {"stats": stats}, args.format)


# ---------------------------------------------------------------------------
# AC analysis tools
# ---------------------------------------------------------------------------


def _parse_freq(s: str, name: str = "frequency") -> float:
    """Parse a SPICE-notation frequency into a finite positive float."""
    try:
        v = parse_spice_value(s)
    except ValueError as e:
        raise ResultError(f"Invalid {name} value {s!r}: {e}") from e
    if not math.isfinite(v):
        raise ResultError(f"{name} must be finite, got {s!r}")
    if v <= 0:
        raise ResultError(f"{name} must be positive, got {s!r} ({v})")
    return v


def _load_ac_signal(
    raw_file: str, signal: str, step: int, state: SessionState
) -> tuple[np.ndarray, np.ndarray]:
    """Load (freqs, H) for an AC signal. Rejects transient data."""
    raw_path = safe_path(raw_file, state)
    raw = services.load_raw(raw_path, state)
    sim_type = detect_sim_type(raw)
    if not is_ac_analysis(sim_type):
        raise ResultError(
            f"This tool requires AC analysis data; got {sim_type!r}. "
            "Use ltspice_signal_stats (transient) or run a .AC sweep first."
        )
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)
    axis = np.asarray(raw.get_axis(step=step))
    wave = np.asarray(raw.get_wave(signal, step=step))
    try:
        return prepare_ac_arrays(axis, wave)
    except ValueError as e:
        raise ResultError(str(e)) from e


# ---- Input models ---------------------------------------------------------


class FindCrossingInput(ToolInput):
    raw_file: str = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    quantity: Quantity = Field(
        description=(
            "What to cross: 'magnitude_db' (dB), 'magnitude_linear' (absolute |H|), "
            "or 'phase_deg' (UNWRAPPED phase in degrees)."
        ),
    )
    level: float = Field(
        description="Level to cross at, in the units of `quantity`. e.g. 0 for 0 dB, -180 for phase margin.",
    )
    direction: SearchDirection = Field(default="any")
    f_start: str | None = Field(
        default=None,
        description="Lower frequency bound in SPICE notation (e.g. '10k'). Defaults to sweep start.",
    )
    f_end: str | None = Field(
        default=None,
        description="Upper frequency bound in SPICE notation. Defaults to sweep end.",
    )
    max_results: int = Field(default=10, description="Cap on returned crossings (1..100).")
    min_separation_decades: float = Field(
        default=0.0,
        description="Merge crossings within this many decades; useful when gain grazes the level.",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class GainAtInput(ToolInput):
    raw_file: str = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    frequencies: list[str] = Field(
        description=(
            "Frequencies to query, each in SPICE notation (e.g. ['100', '1k', '10k']). "
            "Log-axis interpolation is used — queries between sample points are exact "
            "under a log-scale linear assumption, which matches .AC DEC spacing."
        ),
    )
    include_unwrapped_phase: bool = Field(
        default=False,
        description="Also return cumulative unwrapped phase (handy for delay / margin prep).",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class FilterMetricsInput(ToolInput):
    raw_file: str = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    ref_db: float = Field(
        default=-3.0,
        description=(
            "Cutoff reference BELOW passband in dB (must be negative). "
            "Standard -3 for half-power; use -1 for tighter passband specs "
            "or -6 for voltage-half."
        ),
    )
    flatness_db: float = Field(
        default=1.0,
        description="Passband flatness tolerance in dB used for auto-detecting the passband range.",
    )
    passband_range: list[str] | None = Field(
        default=None,
        description=(
            "Optional [f_lo, f_hi] SPICE-notation override for the passband. "
            "If omitted, auto-detected from the flat region near the peak."
        ),
    )
    stopband_range: list[str] | None = Field(
        default=None,
        description=(
            "Optional [f_lo, f_hi] SPICE-notation stopband region. If given, "
            "stopband_rejection is the worst-case attenuation in that range."
        ),
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class StabilityMetricsInput(ToolInput):
    raw_file: str = Field(description="Path to loop-gain AC analysis .raw file")
    signal: str = Field(description="Loop-gain signal (e.g. 'V(loop)')")
    min_separation_decades: float = Field(
        default=0.1,
        description="Merge near-duplicate crossovers closer than this many decades.",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class RollOffInput(ToolInput):
    raw_file: str = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    f_low: str = Field(description="Low frequency bound (SPICE notation)")
    f_high: str = Field(description="High frequency bound (SPICE notation)")
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class ResonanceInput(ToolInput):
    raw_file: str = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    min_prominence_db: float = Field(
        default=3.0,
        description=(
            "Minimum peak prominence in dB. Smaller = more sensitive but also "
            "catches gentle humps. 3 dB rejects filter-passband shoulders."
        ),
    )
    min_separation_decades: float = Field(
        default=0.2,
        description="Merge peaks closer than this many decades (find_peaks can emit duplicates on shoulders).",
    )
    max_peaks: int = Field(default=20, description="Maximum peaks returned (1..100)")
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


# ---- Output schemas -------------------------------------------------------


class FilterMetricsResponse(FilterMetricsOutput):
    """Tool-layer response = lib output + the signal name the user asked about."""

    signal: str


class StabilityMetricsResponse(StabilityMetricsOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


class FindCrossingResponse(TypedDict):
    """Tool-layer response for :func:`handle_find_crossing`."""

    signal: str
    quantity: Quantity
    level: float
    direction: SearchDirection
    crossings: list[CrossingWithQuantity]
    warnings: list[str]


class GainAtResponse(TypedDict):
    """Tool-layer response for :func:`handle_gain_at`."""

    signal: str
    points: list[GainAtPoint]
    warnings: list[str]


class RollOffResponse(RollOffOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


class ResonancesResponse(ResonancesOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


# ---- Handlers -------------------------------------------------------------


@registry.tool(
    name="ltspice_find_crossing",
    description=(
        "Low-level primitive: find all frequencies where a signal's magnitude "
        "(dB or linear) or phase crosses a given level. This is the escape "
        "hatch when the opinionated tools (filter_metrics, stability_metrics) "
        "don't match your question.\n\n"
        "Examples:\n"
        "  - 0 dB crossing of V(out) → unity-gain frequency\n"
        "  - -180° phase crossing → gain margin frequency\n"
        "  - -20 dB crossing → custom stopband edge\n"
        "  - -135° phase crossing → 45° phase-margin frequency\n\n"
        "Log-axis interpolation between sample points. Returns crossings in "
        "increasing frequency. For phase queries, phase is UNWRAPPED first "
        "(continuous, no ±180° jumps) — so 'level=-180' really means -180° "
        "in absolute phase even on systems whose true phase goes past -360°.\n\n"
        "Use the bundled tools for common questions: ltspice_filter_metrics "
        "for -3 dB cutoffs, ltspice_stability_metrics for all unity-gain and "
        "-180° crossings with margins, ltspice_gain_at for point queries "
        "without a crossing search."
    ),
    input_model=FindCrossingInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=FindCrossingResponse,
)
async def handle_find_crossing(args: FindCrossingInput, state: SessionState):
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    f_start = _parse_freq(args.f_start, "f_start") if args.f_start else None
    f_end = _parse_freq(args.f_end, "f_end") if args.f_end else None
    if args.max_results < 1 or args.max_results > 1000:
        raise ResultError(f"max_results must be in [1, 1000], got {args.max_results}")

    try:
        crossings, warnings = find_crossings_any_quantity(
            freqs,
            H,
            quantity=args.quantity,
            level=args.level,
            direction=args.direction,
            f_start=f_start,
            f_end=f_end,
            max_results=args.max_results,
            min_separation_decades=args.min_separation_decades,
        )
    except ValueError as e:
        raise ResultError(str(e)) from e

    data = {
        "signal": args.signal,
        "quantity": args.quantity,
        "level": args.level,
        "direction": args.direction,
        "crossings": crossings,
        "warnings": warnings,
    }

    unit = crossings[0]["units"] if crossings else ""
    lines = [
        f"Crossings of {args.signal}.{args.quantity} at {args.level:g}{unit}:",
        "",
    ]
    if not crossings:
        lines.append("  (none found in window)")
    else:
        for c in crossings:
            lines.append(f"  {c['frequency_hz']:.6g} Hz ({c['direction']})")
    lines += _warning_lines(warnings)
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_gain_at",
    description=(
        "Query magnitude (dB + linear) and phase at a list of frequencies. "
        "Use this instead of calling ltspice_query_value N times — one "
        "simulation load, log-axis interpolation, consistent phase handling.\n\n"
        "Phase is reported wrapped to (-180°, 180°] by default (what you'd "
        "read off a Bode plot). Set include_unwrapped_phase=true to also "
        "get the continuous unwrapped phase — useful for phase-margin prep "
        "or group-delay estimation.\n\n"
        "Frequencies outside the sweep range are clamped to the nearest "
        "endpoint and a warning is emitted — don't silently extrapolate.\n\n"
        "For filter characterization use ltspice_filter_metrics; for "
        "stability margins use ltspice_stability_metrics; for custom "
        "crossing searches use ltspice_find_crossing."
    ),
    input_model=GainAtInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=GainAtResponse,
)
async def handle_gain_at(args: GainAtInput, state: SessionState):
    if not args.frequencies:
        raise ResultError("frequencies list is empty")
    if len(args.frequencies) > 1000:
        raise ResultError(f"Too many frequencies ({len(args.frequencies)}); cap is 1000")
    freqs_q = [_parse_freq(f, "frequency") for f in args.frequencies]
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    points, warnings = _run(
        gain_at_frequencies,
        freqs,
        H,
        freqs_q,
        include_unwrapped_phase=args.include_unwrapped_phase,
    )
    data = {"signal": args.signal, "points": points, "warnings": warnings}

    lines = [f"Gain/phase of {args.signal}:", ""]
    header = "  {:>14s}  {:>10s}  {:>10s}".format("Frequency (Hz)", "Mag (dB)", "Phase (°)")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for p in points:
        lines.append(
            f"  {p['frequency_hz']:>14.6g}  {p['magnitude_db']:>10.3f}  {p['phase_deg']:>10.2f}"
        )
    lines += _warning_lines(warnings)
    return format_response("\n".join(lines), data, args.format)


def _parse_freq_pair(pair: list[str] | None, name: str) -> tuple[float, float] | None:
    if pair is None:
        return None
    if len(pair) != 2:
        raise ResultError(f"{name} must have exactly 2 elements, got {len(pair)}")
    lo = _parse_freq(pair[0], f"{name}[0]")
    hi = _parse_freq(pair[1], f"{name}[1]")
    if lo >= hi:
        raise ResultError(f"{name}: low ({lo}) must be less than high ({hi})")
    return lo, hi


@registry.tool(
    name="ltspice_filter_metrics",
    description=(
        "Characterize a filter response: LPF / HPF / BPF / BSF type, cutoffs, "
        "passband gain & ripple, stopband rejection, transition bandwidth, "
        "rough pole-order estimate.\n\n"
        "Cutoffs are reported at `ref_db` BELOW the passband (not an "
        "absolute -3 dB gain) — so for a BPF with 20 dB passband gain, "
        "`ref_db=-3` gives cutoffs at 17 dB absolute, matching datasheet "
        "conventions. Passband is auto-detected from the flat region near "
        "the peak (within `flatness_db` of max); override with "
        "`passband_range`.\n\n"
        "Classification heuristic: endpoint gain vs peak location. The "
        "classifier is deliberately conservative — ambiguous sweeps (shallow "
        "roll-off, lopsided endpoints, sharp under-sampled notches) return "
        "`filter_type='unknown'` rather than a best guess, and a warning "
        "spells out what was ambiguous. Notch (BSF) detection requires "
        "dense sampling near the null; if the minimum sample is flanked by "
        "points more than a few dB higher, stopband_rejection_db is a "
        "lower bound only and the tool warns accordingly.\n\n"
        "Order estimate: measures slope in the ASYMPTOTIC region (1-2 "
        "decades past cutoff). Returned only when slope is within ±2 dB/dec "
        "of an integer multiple of 20 dB/dec, else null — reports raw "
        "slope regardless.\n\n"
        "For stability / loop-gain questions use ltspice_stability_metrics; "
        "for resonant peaks & Q use ltspice_resonance."
    ),
    input_model=FilterMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=FilterMetricsResponse,
)
async def handle_filter_metrics(args: FilterMetricsInput, state: SessionState):
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    pb = _parse_freq_pair(args.passband_range, "passband_range")
    sb = _parse_freq_pair(args.stopband_range, "stopband_range")
    data = _run(
        compute_filter_metrics,
        freqs,
        H,
        ref_db=args.ref_db,
        flatness_db=args.flatness_db,
        passband_range=pb,
        stopband_range=sb,
    )
    data["signal"] = args.signal

    fc_lo = "-" if data["cutoff_low_hz"] is None else f"{data['cutoff_low_hz']:.6g} Hz"
    fc_hi = "-" if data["cutoff_high_hz"] is None else f"{data['cutoff_high_hz']:.6g} Hz"
    rej = (
        "-" if data["stopband_rejection_db"] is None else f"{data['stopband_rejection_db']:.2f} dB"
    )
    slope = (
        "-"
        if data["rolloff_slope_db_per_decade"] is None
        else f"{data['rolloff_slope_db_per_decade']:.2f} dB/dec"
    )
    order = "-" if data["estimated_order"] is None else f"{data['estimated_order']}"
    tbw = (
        "-"
        if data["transition_bandwidth_hz"] is None
        else f"{data['transition_bandwidth_hz']:.6g} Hz"
    )
    lines = [
        f"Filter Metrics: {args.signal}",
        "",
        f"Type:                {data['filter_type']}",
        f"Passband:            "
        f"[{data['passband_low_hz']:.6g}, {data['passband_high_hz']:.6g}] Hz "
        f"@ {data['passband_gain_db']:.2f} dB",
        f"Passband ripple:     {data['passband_ripple_db']:.3f} dB",
        f"Cutoff (ref {args.ref_db:+.1f} dB): low={fc_lo}  high={fc_hi}",
        f"Stopband rejection:  {rej}",
        f"Transition BW:       {tbw}",
        f"Roll-off slope:      {slope}",
        f"Estimated order:     {order}",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_stability_metrics",
    description=(
        "Find EVERY unity-gain and -180° phase crossover in a loop-gain AC "
        "sweep, report phase margin at each unity-gain crossing and gain "
        "margin at each -180° crossing. Replaces the single-crossing "
        "approximation in ltspice_simulation_summary, which returns wrong "
        "margins on conditionally-stable systems.\n\n"
        "Run this on a LOOP-GAIN signal (typically a dedicated middlebrook "
        "probe or .AC of the open loop). Running on a closed-loop output "
        "gives meaningless margins.\n\n"
        "Returns: dc_gain_db, high_freq_gain_db, stability classification "
        "(stable / unstable / conditional / unconditional / "
        "always_below_unity), all crossings, per-crossing margins, and the "
        "worst-case values.\n\n"
        "Nuances:\n"
        "  - Phase is UNWRAPPED first, so systems whose phase drops past "
        "-360° are handled correctly (otherwise the raw wrap hides the "
        "crossing).\n"
        "  - If phase NEVER crosses -180°, gain margin is 'infinite' "
        "(returned as null with stability='unconditional'). That's stable, "
        "not an error.\n"
        "  - If gain NEVER reaches unity, phase margin is undefined "
        "(returned as null with stability='always_below_unity').\n"
        "  - Multiple crossovers trigger stability='conditional' and a "
        "warning — each one needs its own review.\n\n"
        "For -3 dB filter cutoffs use ltspice_filter_metrics; for custom "
        "crossings use ltspice_find_crossing."
    ),
    input_model=StabilityMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=StabilityMetricsResponse,
)
async def handle_stability_metrics(args: StabilityMetricsInput, state: SessionState):
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    data = _run(
        compute_stability_metrics,
        freqs,
        H,
        min_separation_decades=args.min_separation_decades,
    )
    data["signal"] = args.signal

    pm_worst = (
        "-" if data["phase_margin_worst_deg"] is None else f"{data['phase_margin_worst_deg']:.2f}°"
    )
    gm_worst = (
        "-" if data["gain_margin_worst_db"] is None else f"{data['gain_margin_worst_db']:.2f} dB"
    )
    lines = [
        f"Stability: {args.signal}",
        "",
        f"DC gain:          {data['dc_gain_db']:.2f} dB",
        f"High-freq gain:   {data['high_freq_gain_db']:.2f} dB",
        f"Classification:   {data['stability']}",
        f"PM (worst):       {pm_worst}",
        f"GM (worst):       {gm_worst}",
        "",
        f"Unity-gain crossings ({len(data['unity_gain_crossovers'])}):",
    ]
    for c, m in zip(data["unity_gain_crossovers"], data["phase_margins"], strict=True):
        lines.append(
            f"  {c['frequency_hz']:.6g} Hz ({c['direction']})  PM={m['margin_deg']:+.2f}°"
        )
    lines.append(f"Phase -180° crossings ({len(data['phase_180_crossovers'])}):")
    for c, m in zip(data["phase_180_crossovers"], data["gain_margins"], strict=True):
        lines.append(
            f"  {c['frequency_hz']:.6g} Hz ({c['direction']})  GM={m['margin_db']:+.2f} dB"
        )
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_roll_off",
    description=(
        "Magnitude slope between two frequencies, reported in dB/decade and "
        "dB/octave. Useful for sanity-checking the asymptotic slope of a "
        "filter's stopband or an amplifier's high-frequency roll-off.\n\n"
        "Pick the endpoints in the ASYMPTOTIC region (≥1 decade past any "
        "knee) — measuring across the knee understates the slope. "
        "A rounded pole-order estimate is returned only when the slope is "
        "within ±2 dB/dec of an integer multiple of 20 dB/dec; noisy or "
        "non-asymptotic slopes get a null order and raw slope.\n\n"
        "For a full filter characterization use ltspice_filter_metrics; for "
        "resonance peaks use ltspice_resonance."
    ),
    input_model=RollOffInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=RollOffResponse,
)
async def handle_roll_off(args: RollOffInput, state: SessionState):
    f_lo = _parse_freq(args.f_low, "f_low")
    f_hi = _parse_freq(args.f_high, "f_high")
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    data = _run(compute_roll_off, freqs, H, f_low=f_lo, f_high=f_hi)
    data["signal"] = args.signal

    order = (
        "-"
        if data["nearest_pole_order_estimate"] is None
        else str(data["nearest_pole_order_estimate"])
    )
    lines = [
        f"Roll-off: {args.signal}",
        "",
        f"Span: [{data['f_low_hz']:.6g}, {data['f_high_hz']:.6g}] Hz "
        f"({data['span_decades']:.2f} decades)",
        f"Gain:  {data['gain_low_db']:.2f} → {data['gain_high_db']:.2f} dB "
        f"(Δ = {data['delta_db']:+.2f} dB)",
        f"Slope: {data['slope_db_per_decade']:.2f} dB/decade "
        f"({data['slope_db_per_octave']:.2f} dB/octave)",
        f"Estimated nearest pole order: {order}",
    ]
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="ltspice_resonance",
    description=(
        "Detect magnitude peaks in an AC sweep and estimate Q factor + "
        "-3 dB bandwidth for each. Useful for RLC resonators, crystal "
        "oscillators, peaking amps, or any response with distinct resonant "
        "modes.\n\n"
        "Q = f_peak / Δf(-3 dB from peak). Q is returned as null for peaks "
        "without two flanking -3 dB crossings inside the swept range — "
        "widen the sweep if you need Q for a boundary peak.\n\n"
        "`min_prominence_db=3` rejects the gentle hump of a filter's "
        "passband (which isn't a resonance). Tight resonances (Q > 30) "
        "need dense sampling near f_peak — log sweeps with <50 pts/decade "
        "will under-sample the peak and give inflated Q/bandwidth.\n\n"
        "For overall filter characterization use ltspice_filter_metrics; "
        "for stability margins use ltspice_stability_metrics."
    ),
    input_model=ResonanceInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=ResonancesResponse,
)
async def handle_resonance(args: ResonanceInput, state: SessionState):
    if args.max_peaks < 1 or args.max_peaks > 1000:
        raise ResultError(f"max_peaks must be in [1, 1000], got {args.max_peaks}")
    freqs, H = _load_ac_signal(args.raw_file, args.signal, args.step, state)
    data = _run(
        compute_resonances,
        freqs,
        H,
        min_prominence_db=args.min_prominence_db,
        min_separation_decades=args.min_separation_decades,
        max_peaks=args.max_peaks,
    )
    data["signal"] = args.signal

    lines = [f"Resonances: {args.signal}", "", f"Peaks detected: {data['num_peaks_detected']}"]
    for p in data["peaks"]:
        q = "-" if p["q_factor"] is None else f"{p['q_factor']:.2f}"
        bw = "-" if p["bandwidth_3db_hz"] is None else f"{p['bandwidth_3db_hz']:.6g} Hz"
        lines.append(
            f"  f={p['frequency_hz']:.6g} Hz  gain={p['magnitude_db']:.2f} dB  "
            f"Q={q}  BW-3dB={bw}  phase={p['phase_deg']:+.2f}°"
        )
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)
