""".raw file parsing and waveform analysis.

Provides core functions for parsing .raw files, extracting trace names,
computing statistics, and querying data points. All functions work with
spicelib's RawRead objects and return Python primitives (no numpy types).

For .log file parsing (measurements, Fourier data), see log_parser.py.

Functions are synchronous — callers invoke them directly (see concurrency contract in tools/_base.py).
"""

from __future__ import annotations

import contextlib
import re
from pathlib import Path
from typing import TypedDict

import numpy as np
from spicelib.log.ltsteps import LTSpiceLogReader
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.lib.log_parser import (
    extract_log_diagnostics,
    parse_fourier_data,
    parse_measurements,
)


class OperatingPointOutput(TypedDict):
    """Return shape of :func:`extract_operating_point`."""

    voltages: dict[str, float]
    currents: dict[str, float]

# Smallest positive normal float — floor for magnitude before log10 to avoid -inf
_FLOAT_TINY = np.finfo(float).tiny

# Word-boundary simulation-type matchers. Substring matching would false-positive
# on phrases like "DC transfer characteristic" (contains "AC") or "backup" (also
# contains "AC"), so detection is anchored to whole words.
_RE_TRANSIENT = re.compile(r"\bTRANSIENT\b", re.IGNORECASE)
_RE_AC = re.compile(r"\bAC\b", re.IGNORECASE)
_RE_DC = re.compile(r"\bDC\b", re.IGNORECASE)


def safe_magnitude_db(wave: np.ndarray) -> np.ndarray:
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

    if idx == 0:
        closest_idx = 0
    elif idx == len(axis):
        closest_idx = len(axis) - 1
    else:
        # Choose closer of idx-1 or idx
        closest_idx = idx - 1 if abs(axis[idx - 1] - target_x) < abs(axis[idx] - target_x) else idx

    actual_x = float(axis[closest_idx])

    result = {
        "trace": trace_name,
        "requested_x": float(target_x),
        "actual_x": actual_x,
    }

    if np.iscomplexobj(wave):
        value = wave[closest_idx]
        result["magnitude_db"] = float(20 * np.log10(np.abs(value)))
        result["phase_deg"] = float(np.angle(value, deg=True))
    else:
        result["value"] = float(wave[closest_idx])

    return result


def extract_operating_point(raw: RawRead) -> OperatingPointOutput:
    """Extract DC operating point data (all node voltages and branch currents).

    Works best with Operating Point (.OP) simulations, but can extract
    first-point values from any simulation type.

    Args:
        raw: Loaded RawRead instance

    Returns:
        Dictionary with 'voltages' and 'currents' dicts mapping trace names to values.
        All values are Python float.
    """
    trace_names = raw.get_trace_names()

    voltages = {}
    currents = {}

    for trace in trace_names:
        wave = raw.get_wave(trace, step=0)
        if len(wave) == 0:
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
    """Compute AC bandwidth metrics used by ``simulation_summary``.

    Thin wrapper over :mod:`ltspice_mcp.lib.ac_analysis`. Returns a dict
    with ``bandwidth_3db``, ``unity_gain_freq``, ``phase_margin``, and
    ``gain_margin`` — each a Python float or None. For all crossovers,
    per-crossing margins, and the stability classification, call
    ``ltspice_stability_metrics`` directly.
    """
    # Deferred import — ac_analysis imports raw_parser at module load so
    # the edge in the other direction has to stay late-bound.
    from ltspice_mcp.lib.ac_analysis import (
        compute_stability_metrics,
        detect_crossings,
        prepare_ac_arrays,
    )

    metrics: dict[str, float | None] = {
        "bandwidth_3db": None,
        "unity_gain_freq": None,
        "phase_margin": None,
        "gain_margin": None,
    }

    try:
        axis_raw = raw.get_axis(step=step)
        wave_raw = raw.get_wave(trace_name, step=step)
        freqs, H = prepare_ac_arrays(np.asarray(axis_raw), np.asarray(wave_raw))
    except Exception:
        return metrics

    # -3 dB bandwidth relative to the low-frequency (DC) gain. For LPFs
    # this is the cutoff; for HPFs there's no such crossing and the value
    # stays None; for BPFs it reports the first -3 dB crossing above DC
    # (the low cutoff), matching the previous behavior.
    try:
        mag_db = safe_magnitude_db(H)
        ref_db = float(mag_db[0])
        crossings = detect_crossings(freqs, mag_db, ref_db - 3.0, direction="falling")
        if crossings:
            metrics["bandwidth_3db"] = float(crossings[0]["frequency_hz"])
    except Exception:
        pass

    try:
        stability = compute_stability_metrics(freqs, H)
        # Pick the crossover that produced the worst-case margin so the
        # reported unity_gain_freq and phase_margin come from the SAME
        # crossing (matters on conditionally-stable amps where the first
        # unity-gain crossing is fine but a later one defines stability).
        pm_entries = stability["phase_margins"]
        if pm_entries:
            worst_pm = min(pm_entries, key=lambda m: abs(m["margin_deg"]))
            metrics["unity_gain_freq"] = float(worst_pm["frequency_hz"])
            metrics["phase_margin"] = float(worst_pm["margin_deg"])
        gm_entries = stability["gain_margins"]
        if gm_entries:
            worst_gm = min(gm_entries, key=lambda m: abs(m["margin_db"]))
            metrics["gain_margin"] = float(worst_gm["margin_db"])
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
    sim_type = detect_sim_type(raw)
    trace_names = raw.get_trace_names()
    step_count = get_step_count(raw)

    axis = raw.get_axis(step=0)
    point_count = len(axis)

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

    if log_path and log_path.exists():
        log_reader: LTSpiceLogReader | None = None
        with contextlib.suppress(Exception):
            log_reader = LTSpiceLogReader(str(log_path))

        if log_reader is not None:
            try:
                meas_data = parse_measurements(log_path, reader=log_reader)
                if meas_data["measurements"]:
                    summary["measurements"] = meas_data["measurements"]
            except Exception:
                pass

        try:
            diagnostics = extract_log_diagnostics(log_path)
            if diagnostics["warnings"]:
                summary["warnings"] = diagnostics["warnings"]
            if diagnostics["errors"]:
                summary["errors"] = diagnostics["errors"]
        except Exception:
            pass

        if log_reader is not None:
            try:
                fourier_data = parse_fourier_data(log_path, reader=log_reader)
                if fourier_data:
                    summary["fourier"] = fourier_data
            except Exception:
                pass

    if duration is not None:
        summary["duration"] = float(duration)

    return summary
