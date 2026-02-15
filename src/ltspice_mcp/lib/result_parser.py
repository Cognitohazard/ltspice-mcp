"""Result file parsing and waveform analysis.

Provides core functions for parsing .raw files, extracting trace names,
computing statistics, and querying data points. All functions work with
spicelib's RawRead objects and return Python primitives (no numpy types).

Functions are synchronous and CPU-bound - callers must wrap in run_sync().
"""

import numpy as np
from spicelib.raw.raw_read import RawRead


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

    Args:
        sim_type: Simulation type string from detect_sim_type

    Returns:
        True if AC analysis, False otherwise
    """
    return "AC" in sim_type.upper()


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
        return raw.get_steps()
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
    """
    wave = raw.get_wave(trace_name, step=step)

    # Detect if this is AC data (complex array)
    if np.iscomplexobj(wave):
        # AC Analysis - compute magnitude and phase stats
        magnitude_db = 20 * np.log10(np.abs(wave))
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
    """
    axis = raw.get_axis(step=step)
    wave = raw.get_wave(trace_name, step=step)

    # Binary search for nearest point
    idx = np.searchsorted(axis, target_x)

    # Handle edge cases and find closest point
    if idx == 0:
        closest_idx = 0
    elif idx == len(axis):
        closest_idx = len(axis) - 1
    else:
        # Choose closer of idx-1 or idx
        if abs(axis[idx - 1] - target_x) < abs(axis[idx] - target_x):
            closest_idx = idx - 1
        else:
            closest_idx = idx

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
