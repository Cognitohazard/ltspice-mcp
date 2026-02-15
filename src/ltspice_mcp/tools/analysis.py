"""Waveform analysis tools. (Phase 4)"""

from mcp import types
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.result_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    compute_signal_stats,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    get_trace_names,
    parse_fourier_data,
    parse_measurements,
    query_point_value,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import run_sync, safe_path


async def _get_raw_reader(raw_path, state: SessionState) -> RawRead:
    """Load and cache a RawRead instance.

    Uses state.results FileCache to avoid re-parsing the same file.
    RawRead is loaded with traces_to_read="*" for full data access.

    Args:
        raw_path: Path to .raw file
        state: Session state with results cache

    Returns:
        Cached RawRead instance
    """
    return await run_sync(
        state.results.get, raw_path, lambda p: RawRead(str(p), traces_to_read="*")
    )


async def handle_list_signals(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """List all signal/trace names in a simulation result file.

    Args:
        arguments: Contains 'raw_file' (path to .raw file)
        state: Session state with results cache

    Returns:
        List with single TextContent containing simulation type, step count, and signal names

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found or parse error
    """
    raw_path = safe_path(arguments["raw_file"], state)

    # Verify file exists
    if not await run_sync(raw_path.exists):
        raise ResultError(f"Result file not found: {raw_path}")

    # Load RawRead via cache
    try:
        raw = await _get_raw_reader(raw_path, state)
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )

    # Get simulation metadata
    sim_type = await run_sync(detect_sim_type, raw)
    step_count = await run_sync(get_step_count, raw)
    trace_names = await run_sync(get_trace_names, raw)

    # Format response
    lines = [
        f"Simulation Type: {sim_type}",
        f"Steps: {step_count}",
        f"Signals: {len(trace_names)}",
        "",
    ]

    # List all signals (no truncation per user decision)
    for name in trace_names:
        lines.append(f"  - {name}")

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


async def handle_get_signal_stats(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get statistics for a signal/trace.

    Returns min/max/mean/RMS/peak-to-peak for transient analysis.
    Returns magnitude (dB) and phase (degrees) stats for AC analysis.

    Args:
        arguments: Contains 'raw_file' (path), 'signal' (trace name), optional 'step' (int)
        state: Session state with results cache

    Returns:
        List with single TextContent containing signal statistics

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found, signal not found, or parse error
    """
    raw_path = safe_path(arguments["raw_file"], state)
    signal = arguments["signal"]
    step = arguments.get("step", 0)

    # Verify file exists
    if not await run_sync(raw_path.exists):
        raise ResultError(f"Result file not found: {raw_path}")

    # Load RawRead via cache
    try:
        raw = await _get_raw_reader(raw_path, state)
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )

    # Validate signal exists
    trace_names = await run_sync(get_trace_names, raw)
    if signal not in trace_names:
        available = ", ".join(trace_names[:10])
        if len(trace_names) > 10:
            available += f", ... ({len(trace_names)} total)"
        raise ResultError(
            f"Signal '{signal}' not found. Available signals: {available}"
        )

    # Validate step range
    step_count = await run_sync(get_step_count, raw)
    if step < 0 or step >= step_count:
        raise ResultError(
            f"Step {step} out of range. Valid range: 0 to {step_count - 1}"
        )

    # Compute statistics
    try:
        stats = await run_sync(compute_signal_stats, raw, signal, step)
    except Exception as e:
        raise ResultError(f"Failed to compute statistics: {e}")

    # Format response based on analysis type
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
        # Transient or DC analysis
        lines = [
            f"Signal: {signal}",
            f"Min: {stats['min']:.6g}",
            f"Max: {stats['max']:.6g}",
            f"Mean: {stats['mean']:.6g}",
            f"RMS: {stats['rms']:.6g}",
            f"Peak-to-Peak: {stats['peak_to_peak']:.6g}",
            f"Data Points: {stats['point_count']}",
        ]

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


async def handle_query_value(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Query signal value at a specific time or frequency.

    Returns the nearest data point (no interpolation).
    Accepts SPICE notation for the 'at' parameter (e.g., '1k', '10Meg', '100m').

    Args:
        arguments: Contains 'raw_file' (path), 'signal' (trace name), 'at' (time/freq), optional 'step' (int)
        state: Session state with results cache

    Returns:
        List with single TextContent containing requested and actual x values, and signal value

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found, signal not found, invalid 'at' value, or parse error
    """
    raw_path = safe_path(arguments["raw_file"], state)
    signal = arguments["signal"]
    at_str = arguments["at"]
    step = arguments.get("step", 0)

    # Parse 'at' value (handles SPICE notation like '1k', '10Meg')
    try:
        target_x = parse_spice_value(at_str)
    except ValueError as e:
        raise ResultError(f"Invalid 'at' value: {e}")

    # Verify file exists
    if not await run_sync(raw_path.exists):
        raise ResultError(f"Result file not found: {raw_path}")

    # Load RawRead via cache
    try:
        raw = await _get_raw_reader(raw_path, state)
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )

    # Validate signal exists
    trace_names = await run_sync(get_trace_names, raw)
    if signal not in trace_names:
        available = ", ".join(trace_names[:10])
        if len(trace_names) > 10:
            available += f", ... ({len(trace_names)} total)"
        raise ResultError(
            f"Signal '{signal}' not found. Available signals: {available}"
        )

    # Validate step range
    step_count = await run_sync(get_step_count, raw)
    if step < 0 or step >= step_count:
        raise ResultError(
            f"Step {step} out of range. Valid range: 0 to {step_count - 1}"
        )

    # Query point value
    try:
        result_data = await run_sync(query_point_value, raw, signal, target_x, step)
    except Exception as e:
        raise ResultError(f"Failed to query value: {e}")

    # Format response based on data type
    sim_type = await run_sync(detect_sim_type, raw)
    x_unit = "f" if "AC" in sim_type.upper() else "t"

    if "magnitude_db" in result_data:
        # AC analysis
        lines = [
            f"Signal: {signal} at {x_unit}={result_data['requested_x']:.6g}",
            f"Requested: {result_data['requested_x']:.6g}",
            f"Nearest point: {result_data['actual_x']:.6g}",
            f"Magnitude: {result_data['magnitude_db']:.2f} dB",
            f"Phase: {result_data['phase_deg']:.2f} deg",
        ]
    else:
        # Transient/DC analysis
        lines = [
            f"Signal: {signal} at {x_unit}={result_data['requested_x']:.6g}",
            f"Requested: {result_data['requested_x']:.6g}",
            f"Nearest point: {result_data['actual_x']:.6g}",
            f"Value: {result_data['value']:.6g}",
        ]

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


async def handle_get_measurements(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Extract .MEAS measurement results from simulation log file.

    Args:
        arguments: Contains 'log_file' (path to .log file)
        state: Session state

    Returns:
        List with single TextContent containing .MEAS results

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found or parse error
    """
    log_path = safe_path(arguments["log_file"], state)

    # Verify file exists
    if not await run_sync(log_path.exists):
        raise ResultError(f"Log file not found: {log_path}")

    # Parse measurements
    try:
        meas_data = await run_sync(parse_measurements, log_path)
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(f"Failed to parse log file: {e}")

    measurements = meas_data["measurements"]
    step_count = meas_data["step_count"]

    # Format response
    if not measurements:
        result = "No .MEAS results found in log file"
    elif step_count <= 1:
        # Single-step simulation
        lines = [".MEAS Results:", ""]
        for name, values in measurements.items():
            value = values[0] if values else None
            if value is None:
                lines.append(f"  {name} = FAILED")
            else:
                lines.append(f"  {name} = {value:.6g}")
        result = "\n".join(lines)
    else:
        # Multi-step simulation
        lines = [f".MEAS Results ({step_count} steps):", ""]
        for name, values in measurements.items():
            # Format as array
            value_strs = []
            for val in values:
                if val is None:
                    value_strs.append("FAILED")
                else:
                    value_strs.append(f"{val:.6g}")
            lines.append(f"  {name}: [{', '.join(value_strs)}]")
        result = "\n".join(lines)

    return [types.TextContent(type="text", text=result)]


async def handle_get_operating_point(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Read DC operating point data (all node voltages and branch currents).

    Args:
        arguments: Contains 'raw_file' (path to .raw file)
        state: Session state with results cache

    Returns:
        List with single TextContent containing operating point data

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found or parse error
    """
    raw_path = safe_path(arguments["raw_file"], state)

    # Verify file exists
    if not await run_sync(raw_path.exists):
        raise ResultError(f"Result file not found: {raw_path}")

    # Load RawRead via cache
    try:
        raw = await _get_raw_reader(raw_path, state)
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )

    # Extract operating point data
    try:
        op_data = await run_sync(extract_operating_point, raw)
    except Exception as e:
        raise ResultError(f"Failed to extract operating point: {e}")

    # Format response
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

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


async def handle_get_simulation_summary(
    arguments: dict, state: SessionState
) -> list[types.TextContent]:
    """Get comprehensive simulation summary.

    Args:
        arguments: Contains 'raw_file' (path), optional 'log_file' (path)
        state: Session state with results cache

    Returns:
        List with single TextContent containing comprehensive summary

    Raises:
        PathSecurityError: Path outside sandbox
        ResultError: File not found or parse error
    """
    raw_path = safe_path(arguments["raw_file"], state)
    log_path = None
    if "log_file" in arguments:
        log_path = safe_path(arguments["log_file"], state)

    # Verify raw file exists
    if not await run_sync(raw_path.exists):
        raise ResultError(f"Result file not found: {raw_path}")

    # Verify log file exists if provided
    if log_path and not await run_sync(log_path.exists):
        raise ResultError(f"Log file not found: {log_path}")

    # Load RawRead via cache
    try:
        raw = await _get_raw_reader(raw_path, state)
    except Exception as e:
        raise ResultError(
            f"Failed to parse result file: {e}. "
            "File may be corrupted or not a valid SPICE .raw file"
        )

    # Build summary
    try:
        summary = await run_sync(build_simulation_summary, raw, log_path, None)
    except Exception as e:
        raise ResultError(f"Failed to build summary: {e}")

    # Compute AC bandwidth metrics if applicable
    ac_metrics = None
    if "AC" in summary["sim_type"].upper():
        # Find first voltage signal for bandwidth analysis
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

    # Data size
    lines.append(
        f"Data points: {summary['point_count']} per signal, {summary['step_count']} step(s)"
    )
    lines.append("")

    # Signals
    lines.append(f"Signals ({len(summary['signals'])}):")
    for signal in summary["signals"]:
        lines.append(f"  - {signal}")
    lines.append("")

    # Measurements
    if "measurements" in summary:
        lines.append(".MEAS Results:")
        for name, values in summary["measurements"].items():
            if len(values) == 1:
                val = values[0]
                if val is None:
                    lines.append(f"  {name} = FAILED")
                else:
                    lines.append(f"  {name} = {val:.6g}")
            else:
                value_strs = []
                for val in values:
                    if val is None:
                        value_strs.append("FAILED")
                    else:
                        value_strs.append(f"{val:.6g}")
                lines.append(f"  {name}: [{', '.join(value_strs)}]")
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
                for harm in fourier["harmonics"][:10]:  # Show first 10
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

    result = "\n".join(lines)
    return [types.TextContent(type="text", text=result)]


# Tool definitions
TOOL_DEFS: list[types.Tool] = [
    types.Tool(
        name="list_signals",
        description="List all signal/trace names in a simulation result file (.raw). Returns simulation type, step count, and complete list of available signals. Use this to discover what signals can be analyzed before calling get_signal_stats or query_value.",
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
    ),
    types.Tool(
        name="get_signal_stats",
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
                    "description": "Signal/trace name (e.g., 'V(out)', 'I(R1)'). Use list_signals to see available signals.",
                },
                "step": {
                    "type": "integer",
                    "description": "Step index for .step directives (default 0)",
                },
            },
            "required": ["raw_file", "signal"],
        },
    ),
    types.Tool(
        name="query_value",
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
                    "description": "Signal/trace name (e.g., 'V(out)', 'I(R1)'). Use list_signals to see available signals.",
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
    ),
    types.Tool(
        name="get_measurements",
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
    ),
    types.Tool(
        name="get_operating_point",
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
    ),
    types.Tool(
        name="get_simulation_summary",
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
    ),
]

# Handler mapping
TOOL_HANDLERS: dict[str, object] = {
    "list_signals": handle_list_signals,
    "get_signal_stats": handle_get_signal_stats,
    "query_value": handle_query_value,
    "get_measurements": handle_get_measurements,
    "get_operating_point": handle_get_operating_point,
    "get_simulation_summary": handle_get_simulation_summary,
}
