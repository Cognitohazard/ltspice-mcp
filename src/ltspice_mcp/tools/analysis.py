"""Waveform analysis tools. (Phase 4)"""

from mcp import types

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_parser import parse_measurements
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    compute_signal_stats,
    detect_sim_type,
    extract_operating_point,
    query_point_value,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    load_raw,
    run_sync,
    safe_path,
    text_response,
    validate_signal,
    validate_step,
)


async def handle_get_signal_stats(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get statistics for a signal/trace."""
    raw_path = safe_path(arguments["raw_file"], state)
    signal = arguments["signal"]
    step = arguments.get("step", 0)

    raw = await load_raw(raw_path, state)
    await validate_signal(raw, signal)
    await validate_step(raw, step)

    try:
        stats = await run_sync(compute_signal_stats, raw, signal, step)
    except Exception as e:
        raise ResultError(f"Failed to compute statistics: {e}")

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

    return text_response("\n".join(lines))


async def handle_query_value(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Query signal value at a specific time or frequency."""
    raw_path = safe_path(arguments["raw_file"], state)
    signal = arguments["signal"]
    at_str = arguments["at"]
    step = arguments.get("step", 0)

    try:
        target_x = parse_spice_value(at_str)
    except ValueError as e:
        raise ResultError(f"Invalid 'at' value: {e}")

    raw = await load_raw(raw_path, state)
    await validate_signal(raw, signal)
    await validate_step(raw, step)

    try:
        result_data = await run_sync(query_point_value, raw, signal, target_x, step)
    except Exception as e:
        raise ResultError(f"Failed to query value: {e}")

    sim_type = await run_sync(detect_sim_type, raw)
    x_unit = "f" if "AC" in sim_type.upper() else "t"

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

    return text_response("\n".join(lines))


def _format_measurements(measurements: dict, step_count: int) -> str:
    """Format .MEAS results for display. Shared between handlers."""
    if not measurements:
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


async def handle_get_measurements(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Extract .MEAS measurement results from simulation log file."""
    log_path = safe_path(arguments["log_file"], state)

    try:
        meas_data = await run_sync(parse_measurements, log_path)
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(f"Failed to parse log file: {e}")

    return text_response(
        _format_measurements(meas_data["measurements"], meas_data["step_count"])
    )


async def handle_get_operating_point(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Read DC operating point data (all node voltages and branch currents)."""
    raw_path = safe_path(arguments["raw_file"], state)
    raw = await load_raw(raw_path, state)

    try:
        op_data = await run_sync(extract_operating_point, raw)
    except Exception as e:
        raise ResultError(f"Failed to extract operating point: {e}")

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

    return text_response("\n".join(lines))


async def handle_get_simulation_summary(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get comprehensive simulation summary."""
    raw_path = safe_path(arguments["raw_file"], state)
    log_path = None
    if "log_file" in arguments:
        log_path = safe_path(arguments["log_file"], state)

    raw = await load_raw(raw_path, state)

    try:
        summary = await run_sync(build_simulation_summary, raw, log_path, None)
    except Exception as e:
        raise ResultError(f"Failed to build summary: {e}")

    # Compute AC bandwidth metrics if applicable
    ac_metrics = None
    if "AC" in summary["sim_type"].upper():
        voltage_signals = [s for s in summary["signals"] if s.startswith("V(")]
        if voltage_signals:
            try:
                ac_metrics = await run_sync(
                    compute_ac_bandwidth_metrics, raw, voltage_signals[0], 0
                )
            except Exception:
                pass

    # Format response
    lines = [f"Simulation Summary: {summary['sim_type']}", ""]

    # Range information
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

    # Measurements (uses shared formatter)
    if "measurements" in summary:
        lines.append(_format_measurements(
            summary["measurements"],
            summary.get("step_count", 1),
        ))
        lines.append("")

    # Fourier analysis
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

    # AC bandwidth metrics
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

    # Warnings
    if "warnings" in summary:
        lines.append(f"Warnings ({len(summary['warnings'])}):")
        for warning in summary["warnings"]:
            lines.append(f"  {warning}")
        lines.append("")

    return text_response("\n".join(lines))


# Tool definitions
_RO_ANNOTATIONS = types.ToolAnnotations(
    readOnlyHint=True,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
)

TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="ltspice_get_signal_stats",
        description="Get statistical summary of a signal/trace. For transient/DC analysis: returns min, max, mean, RMS, and peak-to-peak values. For AC analysis: returns magnitude (dB) and phase (degrees) statistics. All values are computed from the full waveform data.",
        inputSchema={
            "type": "object",
            "properties": {
                "raw_file": {
                    "type": "string",
                    "description": "Path to .raw result file from simulation",
                },
                "signal": {
                    "type": "string",
                    "description": "Signal/trace name (e.g., 'V(out)', 'I(R1)'). Use ltspice_get_simulation_summary to see available signals.",
                },
                "step": {
                    "type": "integer",
                    "description": "Step index for .step directives (default 0)",
                },
            },
            "required": ["raw_file", "signal"],
        },
        annotations=_RO_ANNOTATIONS,
    ),
    types.Tool(
        name="ltspice_query_value",
        description="Query the value of a signal at a specific time (transient) or frequency (AC). Returns the nearest data point without interpolation. Accepts SPICE notation for the 'at' parameter: k=1e3, Meg=1e6, m=1e-3, u=1e-6, n=1e-9, p=1e-12, f=1e-15 (e.g., '1k' for 1kHz, '10m' for 10ms).",
        inputSchema={
            "type": "object",
            "properties": {
                "raw_file": {
                    "type": "string",
                    "description": "Path to .raw result file from simulation",
                },
                "signal": {
                    "type": "string",
                    "description": "Signal/trace name (e.g., 'V(out)', 'I(R1)'). Use ltspice_get_simulation_summary to see available signals.",
                },
                "at": {
                    "type": "string",
                    "description": "Time or frequency value to query. Accepts numbers or SPICE notation (e.g., '1k', '10Meg', '100m')",
                },
                "step": {
                    "type": "integer",
                    "description": "Step index for .step directives (default 0)",
                },
            },
            "required": ["raw_file", "signal", "at"],
        },
        annotations=_RO_ANNOTATIONS,
    ),
    types.Tool(
        name="ltspice_get_measurements",
        description="Extract .MEAS measurement results from a simulation log file. Returns all measurements exactly as computed by the simulator. For stepped simulations, returns values for each step.",
        inputSchema={
            "type": "object",
            "properties": {
                "log_file": {
                    "type": "string",
                    "description": "Path to .log file from simulation",
                },
            },
            "required": ["log_file"],
        },
        annotations=_RO_ANNOTATIONS,
    ),
    types.Tool(
        name="ltspice_get_operating_point",
        description="Read DC operating point data showing all node voltages and branch currents. Returns values directly (DC operating point data is small). Works best with .OP simulation results.",
        inputSchema={
            "type": "object",
            "properties": {
                "raw_file": {
                    "type": "string",
                    "description": "Path to .raw result file from simulation",
                },
            },
            "required": ["raw_file"],
        },
        annotations=_RO_ANNOTATIONS,
    ),
    types.Tool(
        name="ltspice_get_simulation_summary",
        description="Get a comprehensive simulation summary including type, signal list, data size, .MEAS results, Fourier analysis, AC bandwidth metrics, and all warnings. Type-aware: AC shows frequency range and bandwidth metrics, transient shows time span, DC shows sweep range.",
        inputSchema={
            "type": "object",
            "properties": {
                "raw_file": {
                    "type": "string",
                    "description": "Path to .raw result file from simulation",
                },
                "log_file": {
                    "type": "string",
                    "description": "Optional path to .log file for measurements and warnings",
                },
            },
            "required": ["raw_file"],
        },
        annotations=_RO_ANNOTATIONS,
    ),
]

# Handler mapping
TOOL_HANDLERS: dict[str, object] = {
    "ltspice_get_signal_stats": handle_get_signal_stats,
    "ltspice_query_value": handle_query_value,
    "ltspice_get_measurements": handle_get_measurements,
    "ltspice_get_operating_point": handle_get_operating_point,
    "ltspice_get_simulation_summary": handle_get_simulation_summary,
}
