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
        ltspice_measurements        — raw .MEAS values per step
        ltspice_measurement_stats   — aggregate .MEAS across sweep/MC

    High-level overview:
        ltspice_simulation_summary  — sim type, signals, warnings, key metrics
"""

import contextlib
import math
from typing import Literal

import numpy as np
from pydantic import Field

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    detect_sim_type,
    extract_operating_point,
    is_ac_analysis,
    query_point_value,
    safe_magnitude_db,
)
from ltspice_mcp.lib.signal_analysis import (
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
    RO_ANNOTATIONS,
    ToolInput,
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
    services.validate_signal(raw, signal)
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
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class QueryValueInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    at: str = Field(
        description="Time or frequency to query in SPICE notation (e.g., '1m', '100u', '1G', '2.5k')"
    )
    step: int = Field(default=0, description="Step index for .step directives")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class MeasurementsInput(ToolInput):
    log_file: str = Field(description="Path to .log file from simulation")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class OperatingPointInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


class SimulationSummaryInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    log_file: str | None = Field(default=None, description="Optional path to .log file")
    signal: str | None = Field(
        default=None,
        description="Signal for AC bandwidth metrics (e.g., 'V(outp)'). Required for AC analysis.",
    )
    format: Literal["json", "text"] | None = Field(default=None, description="Response format: 'json' for structured data, 'text' for human-readable")


@registry.tool(
    name="ltspice_signal_stats",
    description=(
        "Scalar summary of one signal in a .raw result. Use this when you need "
        "a single number per metric (average, RMS, peak, etc.) — not a waveform "
        "or a trend.\n\n"
        "Transient/DC: returns time-weighted mean, RMS, std, abs-mean, and "
        "min/max/pk-pk using trapezoidal integration (RMS = sqrt(∫ y² dt / T)). "
        "This is correct on LTspice's adaptive timestep — simple np.mean(y) "
        "would overweight densely sampled regions. Optionally restrict to a "
        "[t_start, t_end] window; passing no window averages the whole "
        "waveform including any startup transient, which is usually wrong for "
        "RMS/mean. Rejects AC analysis for the windowed path.\n\n"
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
            "t_start_used": {"type": ["number", "null"]},
            "t_end_used": {"type": ["number", "null"]},
            "duration": {"type": ["number", "null"]},
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
    services.validate_signal(raw, signal)
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
        f"RMS:          {stats['rms']:.6g}",
        f"Std:          {stats['std']:.6g}",
        f"Abs mean:     {stats['abs_mean']:.6g}",
        f"Duration:     {stats['duration']:.6g} s  ({stats['point_count']} samples)",
    ]
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
        raise ResultError(
            f"'at' value must be finite, got {at_str!r} (parsed as {target_x})"
        )

    raw = services.load_raw(raw_path, state)
    services.validate_signal(raw, signal)
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
    """Format .MEAS results for display. Shared between handlers."""
    if not measurements:
        if errors:
            lines = ["No .MEAS results — errors in log:", ""]
            for err in errors:
                lines.append(f"  {err}")
            return "\n".join(lines)
        return "No .MEAS results found in log file"

    if step_count <= 1:
        lines = [".MEAS Results:", ""]
        for name, values in measurements.items():
            value = values[0] if values else None
            if value is None:
                lines.append(f"  {name} = FAILED")
            else:
                lines.append(f"  {name} = {value:.6g}")
    else:
        lines = [f".MEAS Results ({step_count} steps):", ""]
        for name, values in measurements.items():
            value_strs = []
            for val in values:
                if val is None:
                    value_strs.append("FAILED")
                else:
                    value_strs.append(f"{val:.6g}")
            lines.append(f"  {name}: [{', '.join(value_strs)}]")

    return "\n".join(lines)


@registry.tool(
    name="ltspice_measurements",
    description=(
        "Extract .MEAS measurement results from a simulation log file. "
        "Returns all measurements exactly as computed by the simulator."
    ),
    input_model=MeasurementsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "measurements": {
                "type": "object",
                "additionalProperties": {
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                },
            },
            "step_count": {"type": "integer"},
        },
    },
)
async def handle_measurements(args: MeasurementsInput, state: SessionState):
    """Extract .MEAS measurement results from simulation log file."""
    log_path = safe_path(args.log_file, state)
    fmt = args.format

    try:
        meas_data = parse_measurements(log_path)
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(f"Failed to parse log file: {e}") from e

    return format_response(
        _format_measurements(
            meas_data["measurements"], meas_data["step_count"], meas_data.get("errors")
        ),
        meas_data,
        fmt,
    )


@registry.tool(
    name="ltspice_operating_point",
    description=(
        "Read DC operating point data showing all node voltages and branch currents."
    ),
    input_model=OperatingPointInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "voltages": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
            "currents": {
                "type": "object",
                "additionalProperties": {"type": "number"},
            },
        },
    },
)
async def handle_operating_point(args: OperatingPointInput, state: SessionState):
    """Read DC operating point data (all node voltages and branch currents)."""
    raw_path = safe_path(args.raw_file, state)
    fmt = args.format
    raw = services.load_raw(raw_path, state)

    try:
        op_data = extract_operating_point(raw)
    except Exception as e:
        raise ResultError(f"Failed to extract operating point: {e}") from e

    lines = ["DC Operating Point", ""]

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
                    "type": "array",
                    "items": {"type": ["number", "null"]},
                },
            },
            "fourier": {"type": "array", "items": {"type": "object"}},
            "ac_bandwidth_metrics": {
                "type": "object",
                "properties": {
                    "bandwidth_3db": {"type": ["number", "null"]},
                    "unity_gain_freq": {"type": ["number", "null"]},
                    "phase_margin": {"type": ["number", "null"]},
                    "gain_margin": {"type": ["number", "null"]},
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
            "errors": {"type": "array", "items": {"type": "string"}},
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
        if ac_metrics["phase_margin"] is not None:
            lines.append(f"  Phase margin: {ac_metrics['phase_margin']:.2f} deg")
        if ac_metrics["gain_margin"] is not None:
            lines.append(f"  Gain margin: {ac_metrics['gain_margin']:.2f} dB")
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
    log_file: str = Field(description="Path to .log file from a .step or Monte Carlo run")
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
# Output schemas
# ---------------------------------------------------------------------------

_WARNINGS_SCHEMA = {"type": "array", "items": {"type": "string"}}

_EDGE_METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string"},
        "transition_time": {"type": "number"},
        "slew_rate": {"type": "number"},
        "low_level": {"type": "number"},
        "high_level": {"type": "number"},
        "t_low_crossing": {"type": "number"},
        "t_high_crossing": {"type": "number"},
        "t_mid_crossing": {"type": "number"},
        "edge_direction": {"type": "string"},
        "is_rise_time": {"type": "boolean"},
        "low_pct": {"type": "number"},
        "high_pct": {"type": "number"},
        "num_edges_in_window": {"type": "integer"},
        "warnings": _WARNINGS_SCHEMA,
    },
}

_PULSE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string"},
        "direction": {"type": "string"},
        "initial_value": {"type": "number"},
        "steady_state_value": {"type": "number"},
        "peak_value": {"type": "number"},
        "peak_time": {"type": "number"},
        "overshoot_pct": {"type": "number"},
        "undershoot_pct": {"type": "number"},
        "settling_time": {"type": ["number", "null"]},
        "settling_tolerance_pct": {"type": "number"},
        "warnings": _WARNINGS_SCHEMA,
    },
}

_TIMING_BETWEEN_SCHEMA = {
    "type": "object",
    "properties": {
        "signal_a": {"type": "string"},
        "signal_b": {"type": "string"},
        "t_a": {"type": "number"},
        "t_b": {"type": "number"},
        "delay": {"type": "number"},
        "threshold_a_used": {"type": "number"},
        "threshold_b_used": {"type": "number"},
        "direction_a": {"type": "string"},
        "direction_b": {"type": "string"},
        "num_crossings_a": {"type": "integer"},
        "num_crossings_b": {"type": "integer"},
        "warnings": _WARNINGS_SCHEMA,
    },
}

_PERIODIC_METRICS_SCHEMA = {
    "type": "object",
    "properties": {
        "signal": {"type": "string"},
        "period": {"type": "number"},
        "frequency": {"type": "number"},
        "jitter_rms": {"type": "number"},
        "duty_cycle_pct": {"type": ["number", "null"]},
        "pulse_width_high": {"type": ["number", "null"]},
        "pulse_width_low": {"type": ["number", "null"]},
        "num_rising_edges": {"type": "integer"},
        "num_falling_edges": {"type": "integer"},
        "num_periods_measured": {"type": "integer"},
        "threshold_used": {"type": "number"},
        "warnings": _WARNINGS_SCHEMA,
    },
}

_MEASUREMENT_STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "stats": {
            "type": "object",
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "total_count": {"type": "integer"},
                    "valid_count": {"type": "integer"},
                    "failure_count": {"type": "integer"},
                    "min": {"type": ["number", "null"]},
                    "max": {"type": ["number", "null"]},
                    "mean": {"type": ["number", "null"]},
                    "median": {"type": ["number", "null"]},
                    "std": {"type": ["number", "null"]},
                    "p10": {"type": ["number", "null"]},
                    "p90": {"type": ["number", "null"]},
                    "best_step_index": {"type": ["integer", "null"]},
                    "worst_step_index": {"type": ["integer", "null"]},
                    "histogram": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "bin_start": {"type": "number"},
                                "bin_end": {"type": "number"},
                                "count": {"type": "integer"},
                            },
                        },
                    },
                },
            },
        }
    },
}


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
    output_schema=_EDGE_METRICS_SCHEMA,
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
    output_schema=_PULSE_RESPONSE_SCHEMA,
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
    output_schema=_TIMING_BETWEEN_SCHEMA,
)
async def handle_timing_between(args: TimingBetweenInput, state: SessionState):
    raw_path = safe_path(args.raw_file, state)
    raw = services.load_raw(raw_path, state)
    _reject_ac(raw)
    services.validate_signal(raw, args.signal_a)
    services.validate_signal(raw, args.signal_b)
    services.validate_step(raw, args.step)

    axis = np.asarray(raw.get_axis(step=args.step))
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    ya_full = np.asarray(raw.get_wave(args.signal_a, step=args.step))
    yb_full = np.asarray(raw.get_wave(args.signal_b, step=args.step))
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
    output_schema=_PERIODIC_METRICS_SCHEMA,
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
        "stats collapse to trivial values — use ltspice_measurements "
        "instead to just read the scalars.\n\n"
        "Works with .MEAS from any analysis type (.tran/.ac/.dc/.op) — the "
        "measurement directives themselves embed the analysis context. Pass "
        "measurement=NAME to aggregate just one; otherwise returns all "
        ".MEAS in the log."
    ),
    input_model=MeasurementStatsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema=_MEASUREMENT_STATS_SCHEMA,
)
async def handle_measurement_stats(args: MeasurementStatsInput, state: SessionState):
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

    stats = _run(
        compute_measurement_stats,
        measurements,
        histogram_bins=args.histogram_bins,
        measurement=args.measurement,
    )

    lines = [
        f"Measurement Stats ({meas_data.get('step_count', len(next(iter(measurements.values()))))} step(s))",
        "",
    ]
    for name, entry in stats.items():
        lines.append(f"{name}:")
        lines.append(
            f"  valid {entry['valid_count']}/{entry['total_count']} "
            f"(failed {entry['failure_count']})"
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
