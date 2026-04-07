"""Waveform analysis tools. (Phase 4)"""

import contextlib
import math
from typing import Literal

from pydantic import Field

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    compute_signal_stats,
    detect_sim_type,
    extract_operating_point,
    is_ac_analysis,
    query_point_value,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    RO_ANNOTATIONS,
    ToolInput,
    format_response,
    registry,
    safe_path,
)


class SignalStatsInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    step: int = Field(default=0, description="Step index for .step directives")
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
    name="ltspice_get_signal_stats",
    description=(
        "Get statistical summary of a signal/trace. For transient/DC analysis: "
        "returns min, max, mean, RMS, and peak-to-peak values. For AC analysis: "
        "returns magnitude (dB) and phase statistics."
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
            "peak_to_peak": {"type": "number"},
            "point_count": {"type": "integer"},
            "min_db": {"type": "number"},
            "max_db": {"type": "number"},
            "mean_db": {"type": "number"},
            "min_phase": {"type": "number"},
            "max_phase": {"type": "number"},
        },
    },
)
async def handle_get_signal_stats(arguments: SignalStatsInput, state: SessionState):
    """Get statistics for a signal/trace."""
    raw_path = safe_path(arguments.raw_file, state)
    signal = arguments.signal
    step = arguments.step
    fmt = arguments.format

    raw = services.load_raw(raw_path, state)
    services.validate_signal(raw, signal)
    services.validate_step(raw, step)

    try:
        stats = compute_signal_stats(raw, signal, step)
    except Exception as e:
        raise ResultError(f"Failed to compute statistics: {e}") from e

    if stats["analysis_type"] == "ac":
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
    else:
        lines = [
            f"Signal: {signal}",
            f"Min: {stats['min']:.6g}",
            f"Max: {stats['max']:.6g}",
            f"Mean: {stats['mean']:.6g}",
            f"RMS: {stats['rms']:.6g}",
            f"Peak-to-Peak: {stats['peak_to_peak']:.6g}",
            f"Data Points: {stats['point_count']}",
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
async def handle_query_value(arguments: QueryValueInput, state: SessionState):
    """Query signal value at a specific time or frequency."""
    raw_path = safe_path(arguments.raw_file, state)
    signal = arguments.signal
    at_str = arguments.at
    step = arguments.step
    fmt = arguments.format

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
    name="ltspice_get_measurements",
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
async def handle_get_measurements(arguments: MeasurementsInput, state: SessionState):
    """Extract .MEAS measurement results from simulation log file."""
    log_path = safe_path(arguments.log_file, state)
    fmt = arguments.format

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
    name="ltspice_get_operating_point",
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
async def handle_get_operating_point(arguments: OperatingPointInput, state: SessionState):
    """Read DC operating point data (all node voltages and branch currents)."""
    raw_path = safe_path(arguments.raw_file, state)
    fmt = arguments.format
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
    name="ltspice_get_simulation_summary",
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
async def handle_get_simulation_summary(arguments: SimulationSummaryInput, state: SessionState):
    """Get comprehensive simulation summary."""
    raw_path = safe_path(arguments.raw_file, state)
    fmt = arguments.format
    log_path = None
    if arguments.log_file is not None:
        log_path = safe_path(arguments.log_file, state)

    raw = services.load_raw(raw_path, state)

    try:
        summary = build_simulation_summary(raw, log_path, None)
    except Exception as e:
        raise ResultError(f"Failed to build summary: {e}") from e

    # Compute AC bandwidth metrics only when signal is explicitly specified
    ac_metrics = None
    if is_ac_analysis(summary["sim_type"]) and arguments.signal:
        with contextlib.suppress(Exception):
            ac_metrics = compute_ac_bandwidth_metrics(raw, arguments.signal, 0)

    # Build JSON data dict (always needed for json mode, cheap to build)
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
