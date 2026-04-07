""".raw file parsing and waveform analysis.

Provides core functions for parsing .raw files, extracting trace names,
computing statistics, and querying data points. All functions work with
spicelib's RawRead objects and return Python primitives (no numpy types).

For .log file parsing (measurements, Fourier data), see log_parser.py.

Functions are synchronous — callers invoke them directly (see concurrency contract in tools/_base.py).
"""

import contextlib
import re
from pathlib import Path

import numpy as np
from spicelib.log.ltsteps import LTSpiceLogReader
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.lib.log_parser import (
    extract_log_diagnostics,
    parse_fourier_data,
    parse_measurements,
)

# Smallest positive normal float — floor for magnitude before log10 to avoid -inf
_FLOAT_TINY = np.finfo(float).tiny

# Word-boundary simulation-type matchers. Substring matching would false-positive
# on phrases like "DC transfer characteristic" (contains "AC") or "backup" (also
# contains "AC"), so detection is anchored to whole words.
_RE_TRANSIENT = re.compile(r"\bTRANSIENT\b", re.IGNORECASE)
_RE_AC = re.compile(r"\bAC\b", re.IGNORECASE)
_RE_DC = re.compile(r"\bDC\b", re.IGNORECASE)


def _safe_magnitude_db(wave: np.ndarray) -> np.ndarray:
    """Convert complex waveform to magnitude in dB, clamping zeros to avoid -inf."""
    magnitude = np.abs(wave)
    magnitude = np.where(magnitude > 0, magnitude, _FLOAT_TINY)
    return 20 * np.log10(magnitude)


def detect_sim_type(raw: RawRead) -> str:
    """Detect simulation type from raw file metadata.

    Args:
        raw: Loaded RawRead instance

    Returns:
        Simulation type string (e.g., "Transient Analysis", "AC Analysis")
        or "Unknown" if detection fails
    """
    try:
        plot_name = raw.get_raw_property("Plotname")
        if plot_name:
            return str(plot_name)
    except Exception:
        pass
    return "Unknown"


def is_ac_analysis(sim_type: str) -> bool:
    """Check if simulation type is AC analysis.

    Uses a word-boundary match on "AC" so substrings in unrelated words
    (e.g. "characteristic", "backup", "BACK") don't false-positive.
    """
    return bool(_RE_AC.search(sim_type))


def get_trace_names(raw: RawRead) -> list[str]:
    """Get list of all trace/signal names in the result file.

    Args:
        raw: Loaded RawRead instance

    Returns:
        List of trace names (e.g., ["time", "V(out)", "I(R1)"])
    """
    return raw.get_trace_names()


def get_step_count(raw: RawRead) -> int:
    """Get number of simulation steps (for .step directives).

    Args:
        raw: Loaded RawRead instance

    Returns:
        Number of steps (defaults to 1 if detection fails)
    """
    try:
        return len(raw.get_steps())
    except Exception:
        return 1


def compute_signal_stats(raw: RawRead, trace_name: str, step: int = 0) -> dict:
    """Compute statistics for a single trace.

    For transient/DC (real data): min, max, mean, RMS, peak-to-peak
    For AC (complex data): magnitude (dB) and phase (degrees) stats

    Args:
        raw: Loaded RawRead instance
        trace_name: Name of trace to analyze
        step: Step index (default 0)

    Returns:
        Dictionary with stats and analysis_type field.
        All values are Python float (not numpy scalars).

    Raises:
        ValueError: If the requested trace contains no data points.
    """
    wave = raw.get_wave(trace_name, step=step)

    if len(wave) == 0:
        raise ValueError(
            f"Signal '{trace_name}' has no data points at step {step}; "
            "cannot compute statistics."
        )

    # Detect if this is AC data (complex array)
    if np.iscomplexobj(wave):
        # AC Analysis - compute magnitude and phase stats
        magnitude_db = _safe_magnitude_db(wave)
        phase_deg = np.angle(wave, deg=True)

        return {
            "analysis_type": "ac",
            "min_db": float(np.min(magnitude_db)),
            "max_db": float(np.max(magnitude_db)),
            "mean_db": float(np.mean(magnitude_db)),
            "min_phase": float(np.min(phase_deg)),
            "max_phase": float(np.max(phase_deg)),
            "point_count": len(wave),
        }
    else:
        # Transient/DC Analysis - compute standard stats
        return {
            "analysis_type": "transient",
            "min": float(np.min(wave)),
            "max": float(np.max(wave)),
            "mean": float(np.mean(wave)),
            "rms": float(np.sqrt(np.mean(wave**2))),
            "peak_to_peak": float(np.ptp(wave)),
            "point_count": len(wave),
        }


def query_point_value(raw: RawRead, trace_name: str, target_x: float, step: int = 0) -> dict:
    """Query signal value at a specific time/frequency (nearest neighbor).

    Uses binary search for O(log n) lookup. No interpolation - returns
    the nearest data point to the requested value.

    Args:
        raw: Loaded RawRead instance
        trace_name: Name of trace to query
        target_x: Time or frequency value to query
        step: Step index (default 0)

    Returns:
        Dictionary with trace name, requested/actual x values, and signal value.
        For AC data, includes magnitude_db and phase_deg.
        All values are Python float (not numpy scalars).

    Raises:
        ValueError: If the trace contains no data points.
    """
    axis = raw.get_axis(step=step)
    wave = raw.get_wave(trace_name, step=step)

    if len(axis) == 0 or len(wave) == 0:
        raise ValueError(
            f"Signal '{trace_name}' has no data points at step {step}; "
            "cannot query value."
        )

    # Binary search for nearest point
    idx = np.searchsorted(axis, target_x)

    # Handle edge cases and find closest point
    if idx == 0:
        closest_idx = 0
    elif idx == len(axis):
        closest_idx = len(axis) - 1
    else:
        # Choose closer of idx-1 or idx
        closest_idx = idx - 1 if abs(axis[idx - 1] - target_x) < abs(axis[idx] - target_x) else idx

    actual_x = float(axis[closest_idx])

    # Build result based on data type
    result = {
        "trace": trace_name,
        "requested_x": float(target_x),
        "actual_x": actual_x,
    }

    if np.iscomplexobj(wave):
        # AC data - return magnitude and phase
        value = wave[closest_idx]
        result["magnitude_db"] = float(20 * np.log10(np.abs(value)))
        result["phase_deg"] = float(np.angle(value, deg=True))
    else:
        # Real data - return raw value
        result["value"] = float(wave[closest_idx])

    return result


def extract_operating_point(raw: RawRead) -> dict:
    """Extract DC operating point data (all node voltages and branch currents).

    Works best with Operating Point (.OP) simulations, but can extract
    first-point values from any simulation type.

    Args:
        raw: Loaded RawRead instance

    Returns:
        Dictionary with 'voltages' and 'currents' dicts mapping trace names to values.
        All values are Python float.
    """
    trace_names = get_trace_names(raw)

    voltages = {}
    currents = {}

    for trace in trace_names:
        # Get first data point (OP has exactly one point, others we take first)
        wave = raw.get_wave(trace, step=0)
        if len(wave) == 0:
            # Skip traces with no data — happens on truncated/aborted runs
            continue
        value = float(wave[0])

        # SPICE node names are case-insensitive; spicelib may return either case.
        trace_upper = trace.upper()
        if trace_upper.startswith("V("):
            voltages[trace] = value
        elif trace_upper.startswith("I("):
            currents[trace] = value

    return {"voltages": voltages, "currents": currents}


def compute_ac_bandwidth_metrics(raw: RawRead, trace_name: str, step: int = 0) -> dict:
    """Compute AC bandwidth metrics (best-effort).

    Calculates -3dB point, unity-gain frequency, phase margin, and gain margin
    for AC analysis. Returns None for metrics that cannot be computed.

    Phase is unwrapped before crossing detection so that systems whose true
    phase drops below -180° (e.g. 3-pole loops) are correctly handled — the
    raw ``np.angle`` output wraps from -179° to +179° at every -180° crossing,
    which would otherwise hide the crossing entirely.

    Args:
        raw: Loaded RawRead instance
        trace_name: Name of voltage trace to analyze
        step: Step index (default 0)

    Returns:
        Dictionary with bandwidth_3db, unity_gain_freq, phase_margin, gain_margin.
        Each value is Python float or None if not computable.
    """
    axis = raw.get_axis(step=step)
    wave = raw.get_wave(trace_name, step=step)

    # Convert to magnitude (dB) and unwrapped phase (degrees)
    magnitude_db = _safe_magnitude_db(wave)
    phase_rad = np.unwrap(np.angle(wave))
    phase_deg = np.rad2deg(phase_rad)

    metrics: dict[str, float | None] = {
        "bandwidth_3db": None,
        "unity_gain_freq": None,
        "phase_margin": None,
        "gain_margin": None,
    }

    # 1. -3dB bandwidth
    try:
        # Find max magnitude
        max_db = np.max(magnitude_db)
        target_db = max_db - 3.0

        # If gain is monotonically decreasing, use first point as reference
        if magnitude_db[0] == max_db or np.all(np.diff(magnitude_db) <= 0):
            target_db = magnitude_db[0] - 3.0

        # Find first crossing below -3dB
        crossings = np.where(magnitude_db < target_db)[0]
        if len(crossings) > 0:
            metrics["bandwidth_3db"] = float(axis[crossings[0]])
    except Exception:
        pass

    # 2. Unity-gain frequency (0dB crossing)
    try:
        # Find where magnitude crosses 0dB from positive to negative
        sign_changes = np.diff(np.sign(magnitude_db))
        # Look for -2 (positive to negative crossing)
        crossings = np.where(sign_changes < 0)[0]
        if len(crossings) > 0:
            # Use first crossing
            idx = crossings[0]
            # Linear interpolation for better accuracy
            if idx + 1 < len(axis):
                x0, x1 = axis[idx], axis[idx + 1]
                y0, y1 = magnitude_db[idx], magnitude_db[idx + 1]
                # Interpolate to find exact 0dB crossing
                if y1 != y0:
                    unity_freq = x0 + (0 - y0) * (x1 - x0) / (y1 - y0)
                    metrics["unity_gain_freq"] = float(unity_freq)

                    # 3. Phase margin at unity-gain frequency
                    ugf_idx = np.searchsorted(axis, unity_freq)
                    if ugf_idx < len(phase_deg):
                        phase_at_ugf = phase_deg[ugf_idx]
                        metrics["phase_margin"] = float(180 + phase_at_ugf)
    except Exception:
        pass

    # 4. Gain margin at -180 degree phase crossing
    try:
        # Find where phase crosses -180 degrees
        phase_target = -180
        # Look for crossings near -180
        crossings = np.where((phase_deg[:-1] > phase_target) & (phase_deg[1:] <= phase_target))[0]
        if len(crossings) > 0:
            idx = crossings[0]
            # Read gain at that frequency
            gain_at_crossing = magnitude_db[idx]
            metrics["gain_margin"] = float(-gain_at_crossing)
    except Exception:
        pass

    return metrics


def build_simulation_summary(
    raw: RawRead, log_path: Path | None, duration: float | None = None
) -> dict:
    """Build comprehensive, type-aware simulation summary.

    Args:
        raw: Loaded RawRead instance
        log_path: Optional path to .log file for measurements/warnings
        duration: Optional simulation duration in seconds

    Returns:
        Dictionary with sim_type, range info, signals, point_count, step_count,
        optional measurements, warnings, Fourier data, and duration.
        All numpy types converted to Python float.
    """
    # Get basic metadata
    sim_type = detect_sim_type(raw)
    trace_names = get_trace_names(raw)
    step_count = get_step_count(raw)

    # Get axis to determine range and point count
    axis = raw.get_axis(step=0)
    point_count = len(axis)

    # Word-boundary matching avoids false positives like "DC transfer
    # characteristic" being classified as AC because "characteristic"
    # contains the substring "AC".
    range_info: dict = {}
    if point_count > 0:
        if _RE_TRANSIENT.search(sim_type):
            range_info = {"time_start": float(axis[0]), "time_end": float(axis[-1])}
        elif is_ac_analysis(sim_type):
            # AC axis values may be complex (frequency + j0); take real part
            range_info = {
                "freq_start": float(axis[0].real),
                "freq_end": float(axis[-1].real),
            }
        elif _RE_DC.search(sim_type):
            range_info = {"sweep_start": float(axis[0]), "sweep_end": float(axis[-1])}
        # Operating Point has no range (single point)

    summary = {
        "sim_type": sim_type,
        "range": range_info,
        "point_count": point_count,
        "step_count": step_count,
        "signals": trace_names,
    }

    # Add optional data from log file
    if log_path and log_path.exists():
        # Create a single LTSpiceLogReader for both measurements and Fourier
        log_reader: LTSpiceLogReader | None = None
        with contextlib.suppress(Exception):
            log_reader = LTSpiceLogReader(str(log_path))

        # Parse measurements
        if log_reader is not None:
            try:
                meas_data = parse_measurements(log_path, reader=log_reader)
                if meas_data["measurements"]:
                    summary["measurements"] = meas_data["measurements"]
            except Exception:
                pass

        # Parse warnings and errors from log
        try:
            diagnostics = extract_log_diagnostics(log_path)
            if diagnostics["warnings"]:
                summary["warnings"] = diagnostics["warnings"]
            if diagnostics["errors"]:
                summary["errors"] = diagnostics["errors"]
        except Exception:
            pass

        # Parse Fourier data
        if log_reader is not None:
            try:
                fourier_data = parse_fourier_data(log_path, reader=log_reader)
                if fourier_data:
                    summary["fourier"] = fourier_data
            except Exception:
                pass

    # Add duration if provided
    if duration is not None:
        summary["duration"] = float(duration)

    return summary
