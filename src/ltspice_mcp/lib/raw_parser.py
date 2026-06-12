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
from ltspice_mcp.lib.result_observations import surface_observations


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
_RE_NOISE = re.compile(r"\bNOISE\b", re.IGNORECASE)


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


def is_noise_analysis(sim_type: str) -> bool:
    """Check if the sim type is a Noise Spectral Density run.

    LTspice's Plotname is ``Noise Spectral Density - (V/Hz½ or A/Hz½)``;
    the word-boundary match avoids false positives on hypothetical
    composite types like "DC NOISE ANALYSIS".
    """
    return bool(_RE_NOISE.search(sim_type))


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


def real_axis(axis: np.ndarray) -> np.ndarray:
    """Return the real part of a SPICE axis. AC frequency axes are stored
    as ``complex(freq, 0)`` by spicelib — strip the imaginary tag for
    ordering / nearest-neighbour lookups. Real axes pass through unchanged.
    """
    return np.real(axis) if np.iscomplexobj(axis) else axis


def nearest_index(axis: np.ndarray, target: float) -> int:
    """Return the axis index nearest to ``target`` (binary search, O(log N)).

    SPICE sweep axes are monotonic so ``np.searchsorted`` finds the bracket
    in one call; the closer-of-pair check picks the better neighbour.
    """
    ins = int(np.searchsorted(axis, target))
    if ins == 0:
        return 0
    if ins >= len(axis):
        return len(axis) - 1
    return ins - 1 if abs(axis[ins - 1] - target) < abs(axis[ins] - target) else ins


def sample_to_dict(sample: complex | float | np.generic) -> dict[str, float]:
    """Convert a wave sample to a JSON-friendly dict.

    Complex AC samples emit ``magnitude_db`` + ``magnitude_linear`` +
    ``phase_deg`` (dB uses the same zero-floor as :func:`safe_magnitude_db`;
    ``magnitude_linear`` is the absolute |value| for currents/ratios where dB
    is awkward, matching ``bode_metrics(mode='point'/'crossing')``). Real
    samples emit ``value``.
    """
    if np.iscomplexobj(sample):
        return {
            "magnitude_db": float(safe_magnitude_db(np.asarray([sample]))[0]),
            "magnitude_linear": float(np.abs(complex(sample))),  # type: ignore[arg-type]
            "phase_deg": float(np.angle(complex(sample), deg=True)),  # type: ignore[arg-type]
        }
    return {"value": float(np.real(sample))}


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
    axis = real_axis(np.asarray(raw.get_axis(step=step)))
    wave = raw.get_wave(trace_name, step=step)

    if axis.size == 0 or len(wave) == 0:
        raise ValueError(
            f"Signal '{trace_name}' has no data points at step {step}; cannot query value."
        )

    closest_idx = nearest_index(axis, target_x)
    return {
        "trace": trace_name,
        "requested_x": float(target_x),
        "actual_x": float(axis[closest_idx]),
        **sample_to_dict(wave[closest_idx]),
    }


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
    call ``stability_metrics`` directly on a loop-gain signal.
    """
    # Deferred import — ac_analysis imports raw_parser at module load so
    # the edge in the other direction has to stay late-bound.
    from ltspice_mcp.lib.ac_analysis import (
        HALF_POWER_DB,
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
        crossings = detect_crossings(freqs, mag_db, ref_db + HALF_POWER_DB, direction="falling")
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
    raw: RawRead,
    log_path: Path | None,
    duration: float | None = None,
    *,
    step: int = 0,
    requested: dict[str, list[str]] | None = None,
    value_scan: str = "off",
) -> dict:
    """Build comprehensive, type-aware simulation summary.

    Args:
        raw: Loaded RawRead instance
        log_path: Optional path to .log file for measurements/warnings
        duration: Optional simulation duration in seconds
        step: Which .step iteration to summarize (axis/range/point_count).
            Defaults to 0. Callers exposing a step (simulation_summary) thread
            it through so the range reflects the chosen step, not always step 0.
        requested: Parsed ``.meas``/``.four`` names from the deck, for the
            requested-vs-produced reconciliation in the observation surfacer.
            None when the caller has no netlist (skips reconciliation).
        value_scan: Coverage decision for value surfacing — ``"scan"`` (this
            ``raw`` has traces loaded; scan them), ``"skipped_large"`` (traces
            not loaded; surface the coverage gap), or ``"off"``.

    Returns:
        Dictionary with sim_type, range info, signals, point_count, step_count,
        optional measurements, warnings, Fourier data, duration, and an
        always-present ``observations`` list (see ``result_observations``).
        All numpy types converted to Python float.
    """
    sim_type = detect_sim_type(raw)
    trace_names = raw.get_trace_names()
    step_count = get_step_count(raw)

    # Stepped ``.op`` raw files have no axis — spicelib raises
    # "This RAW file does not have an axis." Treat that as a valid degenerate
    # case (no range, no point_count beyond step_count) instead of aborting
    # the whole summary.
    try:
        axis = raw.get_axis(step=step)
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
        elif is_ac_analysis(sim_type) or is_noise_analysis(sim_type):
            # AC and noise both sweep over frequency. Axis values may be
            # complex (frequency + j0); take real part.
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
        from ltspice_mcp.lib.log_parser import make_log_reader, scan_op_step_log

        log_reader: LTSpiceLogReader | None = None
        with contextlib.suppress(Exception):
            log_reader = make_log_reader(log_path)

        if log_reader is not None:
            try:
                meas_data = parse_measurements(log_path, reader=log_reader)
                if meas_data["measurements"]:
                    summary["measurements"] = meas_data["measurements"]
                # FAIL'ed measurements aren't in get_measure_names(); surface
                # them as a separate list so consumers can distinguish "did
                # not trigger" from "did not parse".
                failed = meas_data.get("failed_measurements") or []
                if failed:
                    summary["failed_measurements"] = list(failed)
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
        # only step. Two signals: ``.step name=value`` lines (only emitted
        # for ``.tran``/``.ac`` steps) and the Newton-iteration counter
        # (the only signal for stepped ``.op``). Single log walk picks up
        # both.
        if step_count <= 1 and "operating" in sim_type.lower():
            log_steps, op_iters = scan_op_step_log(log_path)
            iteration_count = max(len(log_steps), op_iters)
            if iteration_count > 1:
                if log_steps:
                    param_name = next(iter(log_steps[0].keys()), "param")
                    suggestion = (
                        f"Convert to '.dc {param_name} START STOP STEP' to "
                        "access every bias point."
                    )
                else:
                    suggestion = (
                        "Convert the parametric .op to '.dc <param> START STOP "
                        "STEP' or wrap the .op inside a .tran to capture every "
                        "bias point."
                    )
                warnings.append(
                    f"Stepped .op detected: log shows {iteration_count} bias-"
                    "point iterations but the .raw only carries step 0. " + suggestion
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

    # Surface observations (a "surfacer", not a "judger" — see
    # ``result_observations``). Always present, possibly empty. Value traces are
    # extracted here only when the caller signalled they're loaded; on the
    # bounded success path they are not, and the surfacer records that gap.
    value_traces: dict | None = None
    if value_scan == "scan":
        # The sweep axis (time / frequency / DC source) is trace 0 and isn't a
        # signal worth scanning. Skip it only when the raw actually HAS an axis:
        # an operating-point raw has none, so ITS trace 0 is a real node, and
        # skipping it there hid a degenerate first-sorted value (e.g. a floating
        # node at ~1e30) — exactly the case this scan exists to catch.
        axis_name = trace_names[0] if (has_axis and trace_names) else None
        value_traces = {}
        for name in trace_names:
            if name == axis_name:
                continue
            try:
                value_traces[name] = np.asarray(raw.get_wave(name, step=step))
            except Exception:
                continue
    summary["observations"] = surface_observations(
        summary, requested=requested, value_traces=value_traces, value_scan=value_scan
    )

    return summary
