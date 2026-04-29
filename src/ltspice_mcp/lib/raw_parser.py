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


class _OperatingPointStepMeta(TypedDict, total=False):
    """Optional step metadata populated by the tool layer."""

    step: int
    step_count: int


class OperatingPointOutput(_OperatingPointStepMeta):
    """Return shape of :func:`extract_operating_point`.

    ``step`` / ``step_count`` are present only when the result is built by
    the tool layer for a stepped .OP run.
    """

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
            f"Signal '{trace_name}' has no data points at step {step}; cannot query value."
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


def extract_operating_point(raw: RawRead, step: int = 0) -> OperatingPointOutput:
    """Extract DC operating point data (all node voltages and branch currents).

    Works best with Operating Point (.OP) simulations, but can extract
    first-point values from any simulation type. ``step`` selects which
    iteration of a stepped .OP run to return (0 by default).

    Args:
        raw: Loaded RawRead instance
        step: Step index for stepped .OP / .DC runs.

    Returns:
        Dictionary with 'voltages' and 'currents' dicts mapping trace names to values.
        All values are Python float.
    """
    trace_names = raw.get_trace_names()

    voltages = {}
    currents = {}

    for trace in trace_names:
        wave = raw.get_wave(trace, step=step)
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
    """Compute -3 dB bandwidth and unity-gain frequency for AC simulations.

    Returns a dict with ``bandwidth_3db`` and ``unity_gain_freq`` (each a
    Python float or None). The bandwidth is the first −3 dB crossing
    relative to DC gain (low cutoff for LPFs, low edge for BPFs). The
    unity-gain frequency is the worst-case 0 dB crossover from the full
    stability sweep — meaningful for amplifier-shaped responses.

    Margins (phase, gain) are NOT reported here because they only have
    semantic meaning when the supplied signal is a loop gain, which this
    function can't verify. For full stability analysis with all
    crossovers, per-crossing margins, and a stability classification,
    call ``ltspice_stability_metrics`` directly on a loop-gain signal.
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
        # Worst-case unity-gain crossover: meaningful for amp-shaped
        # responses; stability_metrics returns this even for non-loop-gain
        # signals (it's just a 0 dB crossing).
        pm_entries = stability["phase_margins"]
        if pm_entries:
            worst_pm = min(pm_entries, key=lambda m: abs(m["margin_deg"]))
            metrics["unity_gain_freq"] = float(worst_pm["frequency_hz"])
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

    # Bug D: stepped ``.op`` raw files have no axis — spicelib raises
    # "This RAW file does not have an axis." Treat that as a valid degenerate
    # case (no range, no point_count beyond step_count) instead of aborting
    # the whole summary.
    try:
        axis = raw.get_axis(step=0)
        point_count = len(axis)
        has_axis = True
    except Exception:
        axis = None  # type: ignore[assignment]
        point_count = step_count
        has_axis = False

    range_info: dict = {}
    if has_axis and point_count > 0 and axis is not None:
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

    # ``point_count`` is the per-step axis length (sweep points on .AC/.DC,
    # samples on .tran). ``step_count`` is the number of ``.step`` iterations
    # — 1 for unstepped runs. Together they describe the raw shape unambiguously;
    # don't surface alias keys.
    summary = {
        "sim_type": sim_type,
        "range": range_info,
        "point_count": point_count,
        "step_count": step_count,
        "signals": trace_names,
    }

    if log_path and log_path.exists():
        from ltspice_mcp.lib.log_parser import make_log_reader, parse_step_iterations

        log_reader: LTSpiceLogReader | None = None
        with contextlib.suppress(Exception):
            log_reader = make_log_reader(log_path)

        if log_reader is not None:
            try:
                meas_data = parse_measurements(log_path, reader=log_reader)
                if meas_data["measurements"]:
                    summary["measurements"] = meas_data["measurements"]
            except Exception:
                pass

        warnings: list[str] = []
        try:
            diagnostics = extract_log_diagnostics(log_path)
            warnings = list(diagnostics["warnings"])
            if diagnostics["errors"]:
                summary["errors"] = diagnostics["errors"]
            if diagnostics.get("meas_errors"):
                summary["meas_errors"] = diagnostics["meas_errors"]
        except Exception:
            pass

        # Stepped ``.op`` runs the bias point per step, but LTspice only
        # writes step 0 to the .raw — leaving the user thinking it's the
        # only step. Only worth checking when an .op summary has a single
        # raw step; the .step lookup is otherwise a wasted log read.
        if step_count <= 1 and "operating" in sim_type.lower():
            try:
                log_steps = parse_step_iterations(log_path)
            except Exception:
                log_steps = []
            if len(log_steps) > 1:
                param_name = next(iter(log_steps[0].keys()), "param")
                warnings.append(
                    f"Stepped .op detected: log shows {len(log_steps)} iterations "
                    f"of {param_name!r} but the .raw only carries step 0. Convert "
                    f"to '.dc {param_name} START STOP STEP' to access every bias "
                    "point."
                )

        if warnings:
            summary["warnings"] = warnings

        if log_reader is not None:
            try:
                fourier_data = parse_fourier_data(log_path, reader=log_reader)
                if fourier_data:
                    # Drop entries with neither THD nor harmonics — the empty
                    # stub from a -nan recovery isn't worth surfacing.
                    fourier_data = [
                        f for f in fourier_data if f.get("thd") is not None or f.get("harmonics")
                    ]
                    if fourier_data:
                        summary["fourier"] = fourier_data
            except Exception:
                pass

    if duration is not None:
        summary["duration"] = float(duration)

    return summary
