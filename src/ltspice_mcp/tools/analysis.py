"""Simulation-result analysis tools.

All tools in this module consume simulation output files (.raw, .log) and
return derived metrics. Organized by what the tool answers:

    Scalar summaries:
        signal_stats        — mean/RMS/pk-pk/etc for one signal
        query_value         — value at a specific time/frequency
        operating_point     — DC node voltages + branch currents

    Waveform metrics (transient only, reject AC):
        edge_metrics        — rise/fall time + slew rate
        pulse_response      — overshoot/undershoot/settling
        timing_between      — signed delay between two signals
        periodic_metrics    — period/frequency/duty/jitter

    .MEAS extraction:
        measurement_stats   — aggregate .MEAS across sweep/MC
                                       (single-run .MEAS values are folded
                                        into simulation_summary)

    High-level overview:
        simulation_summary  — sim type, signals, warnings, key metrics
"""

import asyncio
import contextlib
import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
from mcp import types
from pydantic import Field

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import atomic_write, desktop, services
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
    unwrap_phase_safe,
)
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.job_store import SIDECAR_DIRNAME
from ltspice_mcp.lib.log_parser import parse_measurements, parse_step_iterations
from ltspice_mcp.lib.plot_html import build_plot_html
from ltspice_mcp.lib.raw_parser import (
    OperatingPointOutput,
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    is_ac_analysis,
    is_dc_analysis,
    is_noise_analysis,
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
    downsample_minmax,
    stat_envelope,
    window_and_clean,
)
from ltspice_mcp.state import BatchJob, SessionState
from ltspice_mcp.tools._base import (
    MEAS_ERRORS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    RO_ANNOTATIONS,
    ToolInput,
    format_meas_errors,
    format_observations,
    format_response,
    registry,
    safe_path,
    schema_from_typeddict,
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


def _reject_non_transient(raw) -> None:
    """Reject AC / DC / noise raws from a transient-only metric tool.

    edge/pulse/periodic/timing read rise-times, periods, and delays off a time
    axis; an AC sweep (frequency axis, complex data), a .DC sweep (voltage
    axis), or a noise spectrum (frequency axis) all produce meaningless numbers
    here, so refuse them with a pointer to the right tool. A .op raw has no axis
    and is not a sweep; the caller's get_axis call is guarded to surface a clean
    ResultError pointing at operating_point.
    """
    sim_type = detect_sim_type(raw)
    if is_ac_analysis(sim_type) or is_dc_analysis(sim_type) or is_noise_analysis(sim_type):
        raise ResultError(
            f"This tool requires transient analysis (.tran) data; got {sim_type!r}. "
            "For a .DC sweep use query_value or signal_stats; for frequency-domain "
            "(.AC / .noise) use bode_metrics or signal_stats."
        )


def _guarded_axis(raw, step: int) -> np.ndarray:
    """Real-valued sweep axis for ``step``, or a clean error for a no-axis raw.

    spicelib's ``get_axis`` raises when a result has no axis (e.g. an Operating
    Point run); convert that into a friendly ResultError pointing at
    ``operating_point`` rather than letting it surface as a generic internal
    error. AC frequency axes come back complex — strip to the real part.
    """
    try:
        axis = np.asarray(raw.get_axis(step=step))
    except Exception as e:
        raise ResultError(
            f"This result has no data axis ({e}). Use operating_point for a .op result."
        ) from e
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    return axis


def _classify_analysis(raw) -> tuple[str, str, str, bool]:
    """``(plotname, analysis_type, axis_unit, x_is_log)`` for a result raw.

    ``analysis_type`` is one of ``transient``/``ac``/``dc``/``noise``;
    ``axis_unit`` is the x unit (``s``/``Hz``/``""``); ``x_is_log`` marks a
    log-frequency x (ac/noise). Shared by ``get_waveform``, ``export_waveform``,
    and ``plot_waveform`` so the classification lives in one place.
    """
    sim_type = detect_sim_type(raw)
    if is_noise_analysis(sim_type):
        return sim_type, "noise", "Hz", True
    if is_ac_analysis(sim_type):
        return sim_type, "ac", "Hz", True
    if is_dc_analysis(sim_type):
        return sim_type, "dc", "", False
    return sim_type, "transient", "s", False


async def _load_real_signal(
    raw_file: str, signal: str, step: int, state: SessionState
) -> tuple[np.ndarray, np.ndarray]:
    """Load (axis, wave) for a signal, rejecting AC/complex data."""
    raw_path = safe_path(raw_file, state)
    raw = await services.load_raw(raw_path, state)
    _reject_non_transient(raw)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)
    axis = _guarded_axis(raw, step)
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


# Header line for a rendered warnings block. Shared so _strip_warning_block
# (which splits rendered text on it) can't drift from what _warning_lines emits.
_WARNINGS_HEADER = "Warnings:"


def _warning_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["", _WARNINGS_HEADER, *(f"  - {w}" for w in warnings)]


class SignalStatsInput(ToolInput):
    raw_file: str = Field(description="Path to .raw result file from simulation")
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    step: int = Field(default=0, description="Step index for .step directives")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start in SPICE notation (e.g. '1m', '100u'). Transient only. "
            "Strongly recommended when computing RMS or average — the startup "
            "transient otherwise biases the result. Rejected for AC analysis (time-windowing a frequency sweep is an error)."
        ),
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation. Transient only; rejected for AC.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


def _effective_raw_path(
    raw_file: str | None, job_id: str | None, run_index: int, state: SessionState
) -> Path:
    """Resolve the .raw to analyze from EITHER a user ``raw_file`` OR a job run.

    A user-supplied ``raw_file`` is untrusted input → validated via ``safe_path``.
    A job run's ``raw_file`` is a server-generated artifact (the same trust model
    ``batch_results`` uses for ``run_results`` paths), so it is used directly and
    may legitimately live outside ``allowed_paths`` (e.g. a WSL temp dir). This is
    what lets a sweep/MC run be analyzed by the same tools as a standalone raw.
    """
    # Truthiness, not identity: an empty/whitespace raw_file (StrictModel strips
    # to "") must count as absent, else it slips past and safe_path("") resolves
    # to the working dir → a confusing "not a valid .raw" error downstream.
    if bool(raw_file) == bool(job_id):
        raise ResultError("Pass exactly one of 'raw_file' or 'job_id'.")
    if job_id:
        run = services.resolve_run(job_id, state, run_index)
        if run.raw_file is None:
            raise ResultError(f"Run {run_index} of job {job_id!r} has no raw file yet.")
        return run.raw_file
    assert raw_file  # truthy per the guard above
    return safe_path(raw_file, state)


class QueryValueInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``. Lets you query a sweep run "
            "the same way you'd query a standalone raw."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    at: str | None = Field(
        default=None,
        description=(
            "Time or frequency to query in SPICE notation (e.g., '1m', '100u', "
            "'1G', '2.5k'). Required unless ``step_axis`` is given (then it picks "
            "the inner-axis point within the chosen step; optional)."
        ),
    )
    step: int = Field(
        default=0,
        description="Step index for .step directives (ignored when ``step_axis`` is used).",
    )
    step_axis: str | None = Field(
        default=None,
        description=(
            "Select the step by a .step/.DC sweep-axis VALUE instead of an index: "
            "the parameter name (e.g. 'temp', 'Rval'). Pair with ``step_value``. "
            "The nearest step is chosen and flagged with ``exact_match``."
        ),
    )
    step_value: str | None = Field(
        default=None,
        description="Target value of ``step_axis`` in SPICE notation (e.g. '27', '1k'). "
        "Required when ``step_axis`` is given.",
    )
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
    step: int = Field(
        default=0,
        description=(
            "Step index for ac_bandwidth_metrics on a stepped (.step) run. "
            "Default 0 (first step). On a multi-step run the metric is computed "
            "for this step only — a warning notes it."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="signal_stats",
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
        "t_start/t_end are rejected for AC — use query_value for a "
        "point at a specific frequency.\n\n"
        "Noise: returns min/max/pk-pk and the simple/abs mean of the noise "
        "spectral density over the frequency axis, plus ``freq_start_used``/"
        "``freq_end_used``. RMS/std/duration are omitted; t_start/t_end are "
        "rejected — pass them via query_value at specific frequencies "
        "instead.\n\n"
        "Related tools: for rise/fall times use edge_metrics; for "
        "overshoot/settling use pulse_response; for period/duty use "
        "periodic_metrics; to aggregate .MEAS values across a sweep "
        "use measurement_stats."
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
            # Noise-only window metadata (axis is frequency)
            "freq_start_used": {"type": ["number", "null"]},
            "freq_end_used": {"type": ["number", "null"]},
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

    raw = await services.load_raw(raw_path, state)
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
                "Use query_value to look up a specific frequency.",
                show_hint=False,
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

    # Distinguish DC sweep (axis = sweep variable, e.g. ``temp``)
    # from transient (axis = time). Trapezoidal mean/RMS over a sweep axis
    # is mathematically meaningless; the t_start/t_end labels are misleading
    # too since the units aren't seconds.
    sim_type_raw = detect_sim_type(raw)
    sim_type = sim_type_raw.lower()
    is_dc_sweep = "dc transfer" in sim_type or "dc " in sim_type
    is_noise = is_noise_analysis(sim_type_raw)

    if is_noise and (args.t_start is not None or args.t_end is not None):
        raise ResultError(
            "t_start/t_end windowing is not supported for Noise analysis (axis is "
            "frequency, not time). Use query_value to look up a specific "
            "frequency.",
            show_hint=False,
        )

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

    if is_noise:
        # Noise spectral density (V/√Hz) over a (usually log-spaced) frequency
        # axis. A plain arithmetic mean is dominated by wherever the samples
        # cluster and depends on the sweep span, not the circuit — it is not a
        # meaningful figure of merit, so it is deliberately omitted.
        # min/max is the useful "worst-case noise density" reading.
        stats = {
            "analysis_type": "noise",
            "min": core["min"],
            "max": core["max"],
            "peak_to_peak": core["pk_pk"],
            "point_count": core["num_samples"],
            "freq_start_used": core["t_start"],
            "freq_end_used": core["t_end"],
        }
    elif is_dc_sweep:
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
    elif "freq_start_used" in stats:
        lines.append(
            f"Frequency:    {stats['freq_start_used']:.6g}..{stats['freq_end_used']:.6g} Hz"
            f"  ({stats['point_count']} samples)"
        )
    return format_response("\n".join(lines), {"signal": signal, **stats}, fmt)


_DEFAULT_WAVEFORM_BUCKETS = 200


def _busiest_bucket(buckets: list[dict]) -> int | None:
    """Index of the bucket with the largest peak-to-peak (a where-to-look fact)."""
    if not buckets:
        return None
    return max(range(len(buckets)), key=lambda i: buckets[i]["pk_pk"])


def _format_waveform_text(
    signal: str,
    sim_type: str,
    axis_unit: str,
    env: dict,
    observations: list[dict],
) -> list[str]:
    buckets = env["buckets"]
    g_min = min((b["min"] for b in buckets), default=float("nan"))
    g_max = max((b["max"] for b in buckets), default=float("nan"))
    unit = f" {axis_unit}" if axis_unit else ""
    lines = [
        f"Waveform envelope: {signal} ({sim_type})",
        f"  Window:  [{env['x_start']:.6g}, {env['x_end']:.6g}]{unit}",
        f"  Points:  {env['point_count']} -> {env['bucket_count']} buckets"
        + ("  (decimated)" if env["decimated"] else ""),
        f"  Range:   min {g_min:.6g}   max {g_max:.6g}",
        "",
        "Per-bucket envelope (x_start, x_end, min, max, mean, rms, pk_pk, "
        "crest_factor) is in structuredContent; narrow [t_start, t_end] to zoom.",
    ]
    lines.extend(format_observations(observations))
    return lines


class GetWaveformInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Decimate a specific run of a completed sweep/MC (or single) job "
            "instead of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal/trace name (e.g., 'V(out)', 'I(R1)').")
    step: int = Field(default=0, description="Step index for .step directives.")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start in SPICE notation (e.g. '1m', '100u'). Narrow the window "
            "and re-request to zoom into a region of interest."
        ),
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation.",
    )
    buckets: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Number of equal-time envelope buckets (overview resolution). Defaults "
            "to 200; capped at the server's max_points_returned ceiling and at the "
            "sample count."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="get_waveform",
    description=(
        "Decimated waveform egress: a min/max-preserving stat-envelope of one "
        "real-valued signal over a time/sweep/frequency window — for when a scalar "
        "isn't enough and you need to see the SHAPE (switching nodes, amplifier "
        "internal nodes, startup transients).\n\n"
        "Splits the window into equal-time buckets; each bucket reports the raw "
        "sample min/max (a narrow spike or ringing peak is never averaged away), "
        "time-weighted trapezoidal mean/rms (correct on LTspice's adaptive "
        "timestep), pk_pk, and crest_factor (peak/rms — high = impulsive/spiky). "
        "Scalar-guided zoom: read the envelope, then re-request a narrower "
        "[t_start, t_end] to resolve a region at higher resolution (same call, "
        "tighter window). The ``observations`` list surfaces FACTS, not verdicts "
        "(decimation coverage, dropped non-finite samples, which bucket has the "
        "largest pk-to-pk) — you decide what the shape means.\n\n"
        "Works on transient (.tran), DC sweep (.dc), and noise (.noise) results. "
        "For complex AC data use bode_metrics; for a single scalar use "
        "signal_stats; for one point value use query_value."
    ),
    input_model=GetWaveformInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            "analysis_type": {"type": "string"},
            "axis_unit": {"type": "string"},
            "window_start_used": {"type": "number"},
            "window_end_used": {"type": "number"},
            "point_count": {"type": "integer"},
            "bucket_count": {"type": "integer"},
            "max_points_ceiling": {"type": "integer"},
            "decimated": {"type": "boolean"},
            "buckets": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "x_start": {"type": "number"},
                        "x_end": {"type": "number"},
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "mean": {"type": "number"},
                        "rms": {"type": "number"},
                        "pk_pk": {"type": "number"},
                        "crest_factor": {"type": ["number", "null"]},
                        "num_samples": {"type": "integer"},
                    },
                },
            },
            "observations": OBSERVATIONS_SCHEMA,
        },
    },
)
async def handle_get_waveform(args: GetWaveformInput, state: SessionState):
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    fmt = args.format
    step = args.step

    raw = await services.load_raw(raw_path, state)
    signal = services.validate_signal(raw, args.signal)
    services.validate_step(raw, step)

    try:
        wave = np.asarray(raw.get_wave(signal, step=step))
    except Exception as e:
        raise ResultError(f"Failed to read signal {signal!r}: {e}") from e
    if wave.size == 0:
        raise ResultError(f"Signal {signal!r} has no data points at step {step}.")
    if np.iscomplexobj(wave):
        raise ResultError(
            "get_waveform returns a real-valued (time/sweep-domain) envelope; "
            f"signal {signal!r} is complex (AC analysis). Use bode_metrics for "
            "magnitude/phase vs frequency, or query_value at a specific frequency.",
            show_hint=False,
        )

    axis = _guarded_axis(raw, step)

    sim_type_raw, analysis_type, axis_unit, _ = _classify_analysis(raw)

    x_win, y_win, dropped = _window(axis, wave, args.t_start, args.t_end)

    ceiling = state.config.max_points_returned
    requested = args.buckets if args.buckets is not None else _DEFAULT_WAVEFORM_BUCKETS
    n_buckets = max(1, min(requested, ceiling))

    env = _run(stat_envelope, x_win, y_win, n_buckets)

    # Surface FACTS, not verdicts (result-trust doctrine): decimation coverage,
    # dropped non-finite samples, and a where-to-look pointer to the busiest
    # bucket. The model decides what the shape is and where to zoom next.
    observations: list[dict] = []
    if env["decimated"]:
        observations.append(
            {
                "code": "decimated",
                "kind": "coverage",
                "detail": (
                    f"{env['point_count']} samples reduced to {env['bucket_count']} "
                    "equal-time buckets; sub-bucket detail is not represented. "
                    "Re-request a narrower [t_start, t_end] to resolve a region."
                ),
            }
        )
    if dropped:
        observations.append(
            {
                "code": "non_finite",
                "kind": "value",
                "detail": (
                    f"{dropped} non-finite sample(s) dropped from the window before bucketing."
                ),
            }
        )
    busiest = _busiest_bucket(env["buckets"])
    if busiest is not None:
        b = env["buckets"][busiest]
        unit = f" {axis_unit}" if axis_unit else ""
        observations.append(
            {
                "code": "max_pk_pk_bucket",
                "kind": "value",
                "detail": (
                    f"Bucket {busiest} ([{b['x_start']:.6g}, {b['x_end']:.6g}]{unit}) "
                    f"has the largest peak-to-peak ({b['pk_pk']:.6g}) in the window; "
                    "narrow the window there to resolve it."
                ),
                "evidence": {
                    "bucket_index": busiest,
                    "x_start": b["x_start"],
                    "x_end": b["x_end"],
                    "pk_pk": b["pk_pk"],
                },
            }
        )

    data = {
        "signal": signal,
        "analysis_type": analysis_type,
        "axis_unit": axis_unit,
        "window_start_used": env["x_start"],
        "window_end_used": env["x_end"],
        "point_count": env["point_count"],
        "bucket_count": env["bucket_count"],
        "max_points_ceiling": ceiling,
        "decimated": env["decimated"],
        "buckets": env["buckets"],
        "observations": observations,
    }
    lines = _format_waveform_text(signal, sim_type_raw, axis_unit, env, observations)
    return format_response("\n".join(lines), data, fmt)


# ---------------------------------------------------------------------------
# export_waveform — full-fidelity CSV egress to disk
# ---------------------------------------------------------------------------

WAVEFORMS_SUBDIR = "waveforms"

# Generous backstop against a pathological export exhausting memory/disk. Full
# fidelity is the contract, so this is high and RAISES with guidance to window —
# never silently truncates (no silent caps).
_EXPORT_MAX_ROWS = 20_000_000

# x-column header per analysis type (unit-tagged so the CSV is self-describing).
_X_HEADER = {
    "transient": "time_s",
    "ac": "freq_Hz",
    "noise": "freq_Hz",
    "dc": "sweep",
}


def _window_indices(axis: np.ndarray, ts: float | None, te: float | None) -> tuple[int, int]:
    """``[lo, hi)`` sample indices for the ``[ts, te]`` window on a real axis.

    Unlike ``window_and_clean`` this keeps non-finite samples (full-fidelity
    egress) and imposes no minimum sample count. It DOES reproduce the
    monotonicity requirement: ``searchsorted`` returns silently-wrong indices on
    a descending sweep, so refuse one rather than write a corrupt window.

    An empty selection (``lo >= hi``) is returned, not raised: a stepped export
    may legitimately have one step whose axis does not reach the window, and the
    caller decides whether to skip that step or fail the whole export.
    """
    if ts is None and te is None:
        return 0, int(axis.size)
    if axis.size > 1 and not bool(np.all(np.diff(axis) >= 0)):
        raise ResultError(
            "Cannot apply a [t_start, t_end] window to a non-monotonic axis "
            "(e.g. a descending sweep); omit the window to export the full axis."
        )
    if ts is not None and te is not None and ts >= te:
        raise ResultError(f"t_start ({ts:g}) must be < t_end ({te:g}).")
    lo = 0 if ts is None else int(np.searchsorted(axis, ts, side="left"))
    hi = int(axis.size) if te is None else int(np.searchsorted(axis, te, side="right"))
    return lo, hi


def _complex_columns(
    name: str, wave: np.ndarray, complex_format: str
) -> tuple[list[str], list[np.ndarray]]:
    """Expand one complex AC trace into (column names, real-valued arrays).

    Phase is the WRAPPED ``np.angle`` in degrees — the lossless primitive
    matching query_value/bode_metrics; a consumer who wants a continuous curve
    runs ``np.unwrap`` themselves. Magnitude uses the shared ``safe_magnitude_db``
    (floored to avoid -inf at exact zeros).
    """
    if complex_format == "re_im":
        return [f"{name}_re", f"{name}_im"], [np.real(wave), np.imag(wave)]
    if complex_format == "both":
        return (
            [f"{name}_mag_dB", f"{name}_phase_deg", f"{name}_re", f"{name}_im"],
            [safe_magnitude_db(wave), np.degrees(np.angle(wave)), np.real(wave), np.imag(wave)],
        )
    return (
        [f"{name}_mag_dB", f"{name}_phase_deg"],
        [safe_magnitude_db(wave), np.degrees(np.angle(wave))],
    )


def _export_filename(
    raw_path: Path, analysis_type: str, job_id: str | None, run_index: int
) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    run = f"_run{run_index}" if job_id else ""
    return f"{raw_path.stem}_{analysis_type}{run}_{stamp}.csv"


def _build_and_write(
    raw,
    raw_path: Path,
    cols: list[str],
    n_steps: int,
    analysis_type: str,
    ts: float | None,
    te: float | None,
    complex_format: str,
    out_path: Path,
) -> dict:
    """Assemble tidy/long rows for every step, render CSV, write it atomically.

    Runs entirely in a worker thread: heavy numpy reads, O(N) row assembly, and
    file I/O. Returns only FACTS — the handler turns them into observations and
    builds the response on the event loop (the concurrency contract keeps
    response building off worker threads).
    """
    stepped = n_steps > 1
    x_header = _X_HEADER[analysis_type]

    # Per-step .step parameter values for the step_value column. spicelib's
    # get_steps() carries only integers, not the name=value map, so recover it
    # from the sibling .log (same source query_value(step_axis=) uses).
    step_dicts: list[dict[str, float]] = []
    if stepped:
        log_path = raw_path.with_suffix(".log")
        if log_path.exists():
            step_dicts = parse_step_iterations(log_path)

    header: list[str] | None = None
    row_count = 0
    non_finite = 0
    had_complex = False
    empty_steps: list[int] = []
    win_lo: float | None = None
    win_hi: float | None = None

    # Stream rows straight into the atomic temp file: one step's data is the most
    # held in memory at once (no global rows list, no whole-CSV StringIO copy), so
    # a huge full-fidelity export does not balloon RAM. Any exception here unlinks
    # the temp and leaves the destination untouched.
    with atomic_write(out_path) as f:
        csv_writer = csv.writer(f)
        for step in range(n_steps):
            axis = _guarded_axis(raw, step)
            lo, hi = _window_indices(axis, ts, te)
            if lo >= hi:
                # This step's axis does not intersect the window (a step may end
                # earlier than its siblings). Skip it; surfaced as a fact.
                empty_steps.append(step)
                continue
            axis_w = axis[lo:hi]
            col_names: list[str] = []
            col_arrays: list[np.ndarray] = []
            for name in cols:
                wave = np.asarray(raw.get_wave(name, step=step))
                if wave.size == 0:
                    raise ResultError(f"Signal {name!r} has no data points at step {step}.")
                wave_w = wave[lo:hi]
                non_finite += int(np.count_nonzero(~np.isfinite(wave_w)))
                # Key on the trace's own dtype, not the run type: an AC raw can hold
                # a real trace, and a stray complex trace must not become one column.
                if np.iscomplexobj(wave_w):
                    had_complex = True
                    names, arrays = _complex_columns(name, wave_w, complex_format)
                else:
                    names, arrays = [name], [wave_w]
                col_names.extend(names)
                col_arrays.extend(arrays)

            if header is None:
                prefix = ["step_index", "step_value"] if stepped else []
                header = [*prefix, x_header, *col_names]
                csv_writer.writerow(header)

            lo0, hi0 = float(axis_w[0]), float(axis_w[-1])
            win_lo = lo0 if win_lo is None else min(win_lo, lo0)
            win_hi = hi0 if win_hi is None else max(win_hi, hi0)

            # .tolist() converts numpy -> python floats (full round-trippable repr)
            # at C speed; zip transposes columns into tidy/long rows.
            columns = [axis_w.tolist(), *(a.tolist() for a in col_arrays)]
            if stepped:
                label = (
                    ";".join(f"{k}={v:g}" for k, v in step_dicts[step].items())
                    if step < len(step_dicts)
                    else ""
                )
                csv_writer.writerows(
                    [step, label, *values] for values in zip(*columns, strict=True)
                )
            else:
                csv_writer.writerows(zip(*columns, strict=True))
            row_count += len(axis_w)
            if row_count > _EXPORT_MAX_ROWS:
                raise ResultError(
                    f"Export exceeds the {_EXPORT_MAX_ROWS:,}-row safety cap "
                    f"({row_count:,}+ rows). Narrow [t_start, t_end] or export fewer signals."
                )

        if header is None:
            # Every step was skipped — the window selected no samples anywhere.
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {n_steps} steps." if stepped else ".")
            )

    return {
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "n_steps": n_steps,
        "window_used": [win_lo, win_hi] if win_lo is not None else [],
        "non_finite": non_finite,
        "had_complex": had_complex,
        "empty_steps": empty_steps,
        "step_values_available": bool(step_dicts) if stepped else None,
    }


class ExportWaveformInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Export a specific run of a completed sweep/MC (or single) job "
            "instead of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    signals: list[str] | Literal["all"] = Field(
        default="all",
        description=(
            "Trace names to export (e.g. ['V(out)', 'I(R1)']) or 'all' for every non-axis trace."
        ),
    )
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start in SPICE notation (e.g. '1m', '100u', '1k'). Bounds the "
            "export by windowing, not decimation — full fidelity inside the window."
        ),
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation.",
    )
    complex_format: Literal["mag_phase", "re_im", "both"] = Field(
        default="mag_phase",
        description=(
            "How complex AC traces become columns: 'mag_phase' = magnitude(dB) + "
            "phase(deg) [default], 're_im' = real + imag, 'both' = all four. Ignored "
            "for real-valued (.tran/.dc/.noise) traces."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="export_waveform",
    description=(
        "Full-fidelity waveform egress: write every sample of one or more signals "
        "to a CSV file on disk and return its path — for when you want to compute on "
        "the raw data yourself (FFT, custom metrics, cross-correlation) rather than "
        "read a scalar or a decimated envelope.\n\n"
        "Lossless within the chosen window (no decimation — that is get_waveform's "
        "job). Works on transient (.tran), DC sweep (.dc), AC (.ac), and noise "
        "(.noise). Complex AC traces are written as magnitude(dB)+phase(deg) by "
        "default (``complex_format`` selects re/im or both); phase is the wrapped "
        "np.angle — run np.unwrap yourself for a continuous curve. A stepped "
        "(.step / Monte-Carlo) run is written tidy/long: one row per (step, sample) "
        "with leading step_index/step_value columns, because each transient step has "
        "its own time vector. The ``observations`` list surfaces FACTS (rows written, "
        "window coverage, non-finite samples KEPT, the complex format used) — not "
        "verdicts.\n\n"
        "Returns the CSV path plus row/column counts; read the file with your own "
        "tools. For a quick in-context shape use get_waveform; for a single scalar "
        "use signal_stats/query_value."
    ),
    input_model=ExportWaveformInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "row_count": {"type": "integer"},
            "column_count": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "signals": {"type": "array", "items": {"type": "string"}},
            "analysis_type": {"type": "string", "enum": ["transient", "ac", "dc", "noise"]},
            "n_steps": {"type": "integer"},
            "window_used": {"type": "array", "items": {"type": "number"}},
            "complex_format": {"type": ["string", "null"]},
            "observations": OBSERVATIONS_SCHEMA,
        },
    },
)
async def handle_export_waveform(args: ExportWaveformInput, state: SessionState):
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    fmt = args.format
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")

    raw = await services.load_raw(raw_path, state)
    # A .op raw has no sweep axis to tabulate — refuse early with the clean
    # pointer to operating_point, not by failing mid-write inside the worker.
    _guarded_axis(raw, 0)

    _, analysis_type, _, _ = _classify_analysis(raw)

    # Signal columns (canonical names), excluding the axis (trace 0).
    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]
    if args.signals == "all":
        cols = list(trace_names[1:])
    else:
        seen: set[str] = set()
        cols = []
        for s in args.signals:
            canon = services.validate_signal(raw, s)
            if canon == axis_name:
                raise ResultError(f"{s!r} is the sweep axis, not a signal column.")
            if canon not in seen:
                seen.add(canon)
                cols.append(canon)
    if not cols:
        raise ResultError("No signal traces to export (the result has only an axis).")

    n_steps = get_step_count(raw)
    ts = _parse_time(args.t_start, "t_start")
    te = _parse_time(args.t_end, "t_end")

    # Destination: a Linux-side sidecar next to the CIRCUIT, never next to the
    # raw — a job-run raw can live in a Windows temp under /mnt/c that the client
    # cannot Read. Output is a server artifact, so it is not run through safe_path.
    if args.job_id:
        dest_anchor = services.resolve_job(args.job_id, state).netlist.parent
    else:
        dest_anchor = safe_path(args.raw_file, state).parent  # type: ignore[arg-type]
    out_path = (
        dest_anchor
        / SIDECAR_DIRNAME
        / WAVEFORMS_SUBDIR
        / _export_filename(raw_path, analysis_type, args.job_id, args.run_index)
    ).resolve()
    # Refuse to follow a symlinked sidecar out of the circuit directory: if
    # .ltspice-mcp/ or waveforms/ is a symlink, .resolve() would point the write
    # outside the anchor. Server-artifact paths skip safe_path, so enforce
    # containment explicitly here.
    if not out_path.is_relative_to(dest_anchor.resolve()):
        raise ResultError(
            "Refusing to write the export outside the circuit directory "
            "(a symlinked .ltspice-mcp/ or waveforms/ would redirect it)."
        )

    try:
        facts = await asyncio.to_thread(
            _build_and_write,
            raw,
            raw_path,
            cols,
            n_steps,
            analysis_type,
            ts,
            te,
            args.complex_format,
            out_path,
        )
    except ValueError as e:
        raise ResultError(
            f"Failed to assemble the export (corrupt or truncated .raw?): {e}"
        ) from e

    # Surface FACTS, not verdicts (result-trust doctrine).
    observations: list[dict] = []
    step_note = " with step_index/step_value columns (tidy/long)" if facts["n_steps"] > 1 else ""
    observations.append(
        {
            "code": "export_written",
            "kind": "coverage",
            "detail": (
                f"Wrote {facts['row_count']} row(s) x {facts['column_count']} column(s) "
                f"for {facts['n_steps']} step(s){step_note}."
            ),
        }
    )
    if args.t_start is not None or args.t_end is not None:
        observations.append(
            {
                "code": "window_applied",
                "kind": "coverage",
                "detail": (
                    f"Exported the windowed range {facts['window_used']} "
                    f"({_X_HEADER[analysis_type]}); samples outside it were excluded."
                ),
            }
        )
    if facts["empty_steps"]:
        observations.append(
            {
                "code": "window_empty_steps",
                "kind": "coverage",
                "detail": (
                    f"{len(facts['empty_steps'])} step(s) had no samples in the window "
                    f"and were omitted: {facts['empty_steps']}."
                ),
            }
        )
    if facts["non_finite"]:
        observations.append(
            {
                "code": "non_finite",
                "kind": "value",
                "detail": (
                    f"{facts['non_finite']} non-finite sample(s) are present and were "
                    "KEPT in the CSV (full fidelity), not dropped."
                ),
            }
        )
    if facts["step_values_available"] is False:
        observations.append(
            {
                "code": "step_value_unavailable",
                "kind": "value",
                "detail": (
                    "step_value column left blank: no .step parameter map found in the "
                    "sibling .log."
                ),
            }
        )
    if facts["had_complex"]:
        observations.append(
            {
                "code": "complex_format_used",
                "kind": "value",
                "detail": (
                    f"Complex AC traces written as {args.complex_format!r}; phase is "
                    "the WRAPPED np.angle in degrees (run np.unwrap for a continuous curve)."
                ),
            }
        )

    data = {
        "path": str(out_path),
        "row_count": facts["row_count"],
        "column_count": facts["column_count"],
        "columns": facts["columns"],
        "signals": cols,
        "analysis_type": analysis_type,
        "n_steps": facts["n_steps"],
        "window_used": facts["window_used"],
        "complex_format": args.complex_format if facts["had_complex"] else None,
        "observations": observations,
    }
    lines = [
        f"Exported {facts['row_count']} row(s) to {out_path}",
        f"Columns: {', '.join(facts['columns'])}",
        *format_observations(observations),
    ]
    return format_response("\n".join(lines), data, fmt)


@registry.tool(
    name="query_value",
    description=(
        "Look up the value of a signal at a specific time point (transient) or "
        "frequency (AC). Returns the nearest data point without interpolation.\n\n"
        "To pick a step of a .step/.DC sweep by its axis VALUE (rather than a "
        "raw step index), pass ``step_axis`` + ``step_value`` (e.g. "
        "step_axis='temp', step_value='27'); ``at`` then selects the inner-axis "
        "point within that step (optional). AC samples also return "
        "``magnitude_linear`` alongside ``magnitude_db``/``phase_deg``.\n\n"
        "To query a run of a completed sweep/MC job, pass ``job_id`` + "
        "``run_index`` instead of ``raw_file`` — the run is analyzed like any "
        "standalone raw."
    ),
    input_model=QueryValueInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            # direct (at) path
            "requested_x": {"type": "number"},
            "actual_x": {"type": "number"},
            "value": {"type": "number"},
            "magnitude_db": {"type": "number"},
            "magnitude_linear": {"type": "number"},
            "phase_deg": {"type": "number"},
            # step_axis path (delegated to the step lookup); keys are optional
            "axis": {"type": "string"},
            "requested_value": {"type": "number"},
            "actual_value": {"type": "number"},
            "exact_match": {"type": "boolean"},
            "step_index": {"type": "integer"},
            "requested_at": {"type": "number"},
            "actual_at": {"type": "number"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    },
)
async def handle_query_value(args: QueryValueInput, state: SessionState):
    """Query signal value at a specific time/frequency, or at a chosen sweep step."""
    # Step-by-axis-value mode folds in the former step_get tool. It selects a
    # step WITHIN a single .step raw, so it is raw_file-only — ``job_id`` already
    # selects the run, so the two selection mechanisms are mutually exclusive.
    if args.step_axis is not None:
        if args.job_id is not None:
            raise ResultError(
                "query_value: 'step_axis' selects a step of a .step raw and can't be "
                "combined with 'job_id' (the run is already selected — pass 'at').",
                show_hint=False,
            )
        step_raw = args.raw_file
        if step_raw is None:
            raise ResultError("query_value: 'step_axis' requires 'raw_file'.", show_hint=False)
        if args.step_value is None:
            raise ResultError(
                "query_value: 'step_value' is required when 'step_axis' is given.",
                show_hint=False,
            )
        from ltspice_mcp.tools.circuit import StepGetInput, handle_step_get

        return await handle_step_get(
            StepGetInput(
                raw_file=step_raw,
                axis=args.step_axis,
                value=args.step_value,
                signal=args.signal,
                at=args.at,
                format=args.format,
            ),
            state,
        )

    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    signal = args.signal
    if args.at is None:
        raise ResultError(
            "query_value: 'at' is required (or use step_axis + step_value).", show_hint=False
        )
    at_str = args.at
    step = args.step
    fmt = args.format

    try:
        target_x = parse_spice_value(at_str)
    except ValueError as e:
        raise ResultError(f"Invalid 'at' value: {e}", show_hint=False) from e

    # np.searchsorted treats NaN as greater than everything and returns the
    # last index, which looks like a valid result but isn't.
    if not math.isfinite(target_x):
        raise ResultError(
            f"'at' value must be finite, got {at_str!r} (parsed as {target_x})", show_hint=False
        )

    raw = await services.load_raw(raw_path, state)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)

    try:
        result_data = query_point_value(raw, signal, target_x, step)
    except Exception as e:
        # Operating-point raws have no time/frequency axis; spicelib raises
        # "This RAW file does not have an axis." Give a precise, actionable
        # message instead of the generic failure + misleading check_job hint.
        if "does not have an axis" in str(e).lower():
            raise ResultError(
                "This is an Operating Point result (no time/frequency axis to "
                "query). Use operating_point to read node voltages and branch "
                "currents.",
                show_hint=False,
            ) from e
        raise ResultError(f"Failed to query value: {e}") from e

    sim_type = detect_sim_type(raw)
    x_unit = "f" if is_ac_analysis(sim_type) else "t"

    if "magnitude_db" in result_data:
        lines = [
            f"Signal: {signal} at {x_unit}={result_data['requested_x']:.6g}",
            f"Requested: {result_data['requested_x']:.6g}",
            f"Nearest point: {result_data['actual_x']:.6g}",
            f"Magnitude: {result_data['magnitude_db']:.2f} dB "
            f"({result_data['magnitude_linear']:.6g})",
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

    def _fmt_meta(value: object) -> str:
        # Per-step lists are summarised as ``[lo..hi]`` rather than
        # echoed in full — the per-step values already accompany them in
        # the entry's ``values`` field.
        if isinstance(value, list):
            nums = [v for v in value if isinstance(v, int | float)]
            if not nums:
                return "[…]"
            return f"[{min(nums):g}..{max(nums):g}]"
        if isinstance(value, int | float):
            return f"{value:g}"
        return str(value)

    def _meta_suffix(entry: dict) -> str:
        bits: list[str] = []
        if entry.get("range_from") is not None or entry.get("range_to") is not None:
            lo = entry.get("range_from")
            hi = entry.get("range_to")
            if lo is not None and hi is not None:
                bits.append(f"FROM={_fmt_meta(lo)} TO={_fmt_meta(hi)}")
            elif lo is not None:
                bits.append(f"FROM={_fmt_meta(lo)}")
            elif hi is not None:
                bits.append(f"TO={_fmt_meta(hi)}")
        if entry.get("at") is not None:
            bits.append(f"AT={_fmt_meta(entry['at'])}")
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
    name="operating_point",
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
    raw = await services.load_raw(raw_path, state)

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

    # A DC sweep raw has no single "operating point": point 0 is just the
    # sweep's starting value (e.g. V1=0). Flag it so an all-zeros / start-of-
    # sweep result isn't mistaken for a degenerate circuit.
    dc_sweep_note: str | None = None
    if "transfer" in sim_lower or "dc" in sim_lower.split():
        dc_sweep_note = (
            "This is a DC sweep raw; the values are sweep point "
            f"{args.step} (the sweep's starting bias), not a chosen operating "
            "point. Use query_value at a specific sweep value, or run a .OP."
        )
        op_data.setdefault("warnings", []).append(dc_sweep_note)

    lines = ["DC Operating Point", ""]
    if dc_sweep_note:
        lines.append(f"⚠ {dc_sweep_note}")
        lines.append("")
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
    name="simulation_summary",
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
                        # Scalar when the bound is constant across .step
                        # iterations; list (one entry per step) when it
                        # varies (e.g. TRIG/TARG marker times).
                        "range_from": {
                            "type": ["number", "array", "null"],
                            "items": {"type": ["number", "null"]},
                        },
                        "range_to": {
                            "type": ["number", "array", "null"],
                            "items": {"type": ["number", "null"]},
                        },
                        "at": {
                            "type": ["number", "array", "null"],
                            "items": {"type": ["number", "null"]},
                        },
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
            "failed_measurements": {"type": "array", "items": {"type": "string"}},
            "observations": OBSERVATIONS_SCHEMA,
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

    raw = await services.load_raw(raw_path, state)
    # Honor ``step`` for the summary itself (range/point_count), not just for
    # ac_bandwidth_metrics. Validate up front so an out-of-range step errors
    # clearly instead of being silently ignored.
    services.validate_step(raw, args.step)

    try:
        # ``raw`` here is fully loaded (services.load_raw reads all traces), so
        # the value scan is affordable and surfaces NaN/extreme-value facts.
        summary = build_simulation_summary(raw, log_path, None, step=args.step, value_scan="scan")
    except Exception as e:
        # Suppress the generic ResultError hint — it points at simulation_summary,
        # which is the tool that just failed (self-referential).
        raise ResultError(f"Failed to build summary: {e}", show_hint=False) from e

    suggestions = services.suggestions_from_errors(summary.get("errors"), state.libraries)
    if suggestions:
        summary["suggestions"] = suggestions

    # Compute AC bandwidth metrics on AC raws. When ``signal`` is omitted,
    # auto-pick the first V(...) trace and warn — the previous behaviour
    # silently dropped ac_bandwidth_metrics from the response, leaving
    # users to wonder why their AC summary had no metrics.
    ac_metrics = None
    ac_signal_used: str | None = None
    if is_ac_analysis(summary["sim_type"]):
        ac_signal_used = args.signal
        if ac_signal_used is None:
            ac_signal_used = next(
                (t for t in summary["signals"] if t.upper().startswith("V(")), None
            )
            if ac_signal_used is not None:
                summary.setdefault("warnings", []).append(
                    f"AC summary built without an explicit ``signal``; "
                    f"defaulted to {ac_signal_used!r} for ac_bandwidth_metrics. "
                    "Pass ``signal=`` to choose a different trace."
                )
        if ac_signal_used:
            with contextlib.suppress(Exception):
                ac_metrics = compute_ac_bandwidth_metrics(raw, ac_signal_used, args.step)
            # On a multi-step run the metric is for one step only; the bare
            # number next to step_count=N otherwise reads as the whole-run
            # answer (it isn't — it's wrong for the other N-1 steps).
            if ac_metrics is not None and summary.get("step_count", 1) > 1:
                ac_metrics["step"] = args.step
                summary.setdefault("warnings", []).append(
                    f"ac_bandwidth_metrics is for step {args.step} of "
                    f"{summary['step_count']}; pass step=N for other steps."
                )

    json_data = dict(summary)
    if ac_metrics:
        json_data["ac_bandwidth_metrics"] = ac_metrics
    if ac_signal_used and ac_signal_used != args.signal:
        json_data["ac_signal_used"] = ac_signal_used

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

    if summary.get("failed_measurements"):
        lines.append("FAIL'ed measurements (logged but did not trigger):")
        for name in summary["failed_measurements"]:
            lines.append(f"  {name}")
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

    obs_lines = format_observations(summary.get("observations", []))
    if obs_lines:
        lines.extend(obs_lines)
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
    low_level: float | None = Field(
        default=None,
        description=(
            "Absolute low rail level, overriding auto-detection. Use when the "
            "auto estimate (mean of first/last 10%) is biased — e.g. a "
            "rise-from-rail where early samples cluster in the fast ramp."
        ),
    )
    high_level: float | None = Field(
        default=None,
        description="Absolute high rail level, overriding auto-detection.",
    )
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
            "Job ID. For a batch job (``run_montecarlo`` / ``run_sweep``) the "
            "tool loads each completed run's log, concatenates the .MEAS "
            "results (one row per run), and aggregates. For a completed "
            "single-simulation job it aggregates that run's log (per-step "
            "values for a .step run). Mutually exclusive with ``log_file``."
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
    name="edge_metrics",
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
        "For settling/overshoot after the edge, use pulse_response. "
        "For delay between two signals' edges, use timing_between."
    ),
    input_model=EdgeMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=EdgeMetricsResponse,
)
async def handle_edge_metrics(args: EdgeMetricsInput, state: SessionState):
    axis, wave = await _load_real_signal(args.raw_file, args.signal, args.step, state)
    t, y, _ = _window(axis, wave, args.t_start, args.t_end)
    data = _run(
        analyze_edge,
        t,
        y,
        edge=args.edge,
        edge_index=args.edge_index,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
        low_level=args.low_level,
        high_level=args.high_level,
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
    name="pulse_response",
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
        "For just rise/fall time without overshoot, use edge_metrics."
    ),
    input_model=PulseResponseInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=PulseResponseResponse,
)
async def handle_pulse_response(args: PulseResponseInput, state: SessionState):
    axis, wave = await _load_real_signal(args.raw_file, args.signal, args.step, state)
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

    # overshoot/undershoot/settling are None when the window is a full pulse
    # (metrics undefined, not authoritative). Disambiguate that from a genuine
    # "never settled within the window" (settling_time None on a valid window).
    undefined = "net_step_small_vs_swing" in data.get("quality", [])

    def _pct(value: float | None) -> str:
        return "undefined (full-pulse window)" if value is None else f"{value:.3f} %"

    if data["settling_time"] is not None:
        settle = f"{data['settling_time']:.6g} s"
    elif undefined:
        settle = "undefined (full-pulse window)"
    else:
        settle = "never (within window)"
    lines = [
        f"Pulse Response: {args.signal} ({data['direction']} step)",
        "",
        f"Initial: {data['initial_value']:.6g}",
        f"Final:   {data['steady_state_value']:.6g}",
        f"Peak:    {data['peak_value']:.6g} at t={data['peak_time']:.6g} s",
        f"Overshoot:  {_pct(data['overshoot_pct'])}",
        f"Undershoot: {_pct(data['undershoot_pct'])}",
        f"Settling time (±{data['settling_tolerance_pct']:.2f}%): {settle}",
    ]
    if data.get("quality"):
        lines.append(f"Quality flags: {', '.join(data['quality'])}")
    lines += _warning_lines(data["warnings"])
    return format_response("\n".join(lines), data, args.format)


@registry.tool(
    name="timing_between",
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
    raw = await services.load_raw(raw_path, state)
    _reject_non_transient(raw)
    sig_a = services.validate_signal(raw, args.signal_a)
    sig_b = services.validate_signal(raw, args.signal_b)
    services.validate_step(raw, args.step)

    axis = _guarded_axis(raw, args.step)
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
    name="periodic_metrics",
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
        "edge (not periodic), use edge_metrics."
    ),
    input_model=PeriodicMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=PeriodicMetricsResponse,
)
async def handle_periodic_metrics(args: PeriodicMetricsInput, state: SessionState):
    axis, wave = await _load_real_signal(args.raw_file, args.signal, args.step, state)
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


# Most distinct per-run diagnostic lines relayed in the no-measurements
# error before truncating to a "... and N more" summary line.
_MAX_RELAYED_RUN_DIAGNOSTICS = 8


def _diagnostics_block(diags: list[str], empty_note: str) -> str:
    """Indent diagnostic lines for an error payload, or fall back to
    ``empty_note`` when there is nothing to relay."""
    if not diags:
        return f"  {empty_note}"
    return "\n".join(f"  {d}" for d in diags)


def _aggregate_job_measurements(
    batch_job: BatchJob,
) -> tuple[dict[str, list[float | None]], int, dict[str, AggregatedField], list[str]]:
    """Walk every completed run's .log and concatenate ``.MEAS`` results.

    The MC engine emits one log per run; this reconciles by collecting
    per-run scalar values keyed by .MEAS name. The caller has already
    resolved ``batch_job`` from the job store.

    For ``WHEN``-style .MEAS, the per-run scalar in ``values`` is the
    trigger level (constant across runs by definition) — the interesting
    per-run axis lives in the folded ``at`` field. When that pattern is
    detected (constant ``values``, varying ``at``) the aggregator swaps to
    the ``at`` axis automatically.

    Returns ``(flat_values, run_count, axis_map, diagnostics)`` where
    ``axis_map[name]`` is ``"value"`` or ``"at"`` describing which field was
    aggregated, and ``diagnostics`` carries deduplicated per-run log
    errors/warnings explaining missing measurements (e.g. ngspice's
    batch-mode .meas skip) so an empty aggregate can relay the cause.
    """
    if not batch_job.run_results:
        raise ResultError(
            f"Batch job {batch_job.job_id!r} has no completed runs yet — wait for it "
            "to finish (use check_job to monitor)."
        )

    samples: dict[str, _MeasSamples] = {}
    diagnostics: list[str] = []
    seen_diagnostics: set[str] = set()
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
        # parse_measurements only populates errors/warnings on its
        # no-measurements branch, so this collects exactly the lines that
        # explain an absence. Deduplicated: every run of a batch typically
        # repeats the same simulator diagnostic verbatim.
        for diag in list(data.get("errors") or []) + list(data.get("warnings") or []):
            if diag not in seen_diagnostics:
                seen_diagnostics.add(diag)
                diagnostics.append(diag)
        for name, entry in data.get("measurements", {}).items():
            row = entry.get("values", [])
            scalar = row[0] if row else None
            at_raw = entry.get("at")
            # Per-step lists collapse to the first scalar: batch runs
            # usually have step_count=1 so this is a no-op, but guard anyway.
            if isinstance(at_raw, list):
                at_raw = next((v for v in at_raw if v is not None), None)
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

    return flat_values, runs_processed, axis_map, diagnostics


def _aggregate_log_measurements(
    log_path: Path,
) -> tuple[dict[str, list[float | None]], dict[str, AggregatedField], str]:
    """Aggregate the .MEAS results of ONE log file.

    The interesting case is a ``.step`` log, where each .MEAS name carries
    one value per step; a plain single-run log yields one value per name
    (honest n=1 stats). Shared by the ``log_file`` input branch and the
    single-simulation ``job_id`` branch, which read the same physical shape.

    Returns ``(flat_values, axis_map, steps_label)``.
    """
    try:
        meas_data = parse_measurements(log_path)
    except ResultError:
        raise
    except Exception as e:
        raise ResultError(f"Failed to parse log file: {e}") from e

    measurements = meas_data.get("measurements", {})
    if not measurements:
        # Surface BOTH errors and warnings — the reason measurements are
        # missing is often a warning (e.g. ngspice "No .measure possible in
        # batch mode"), not an error. Reporting "no diagnostics" while every
        # other tool shows the cause is misleading.
        diags = list(meas_data.get("errors") or []) + list(meas_data.get("warnings") or [])
        err_block = _diagnostics_block(
            diags, "(log contained no .MEAS results and no diagnostics)"
        )
        raise ResultError(f"No .MEAS results in log:\n{err_block}")

    flat_values = {name: list(entry.get("values", [])) for name, entry in measurements.items()}
    axis_map: dict[str, AggregatedField] = dict.fromkeys(flat_values, "value")
    steps_label = f"{meas_data.get('step_count', 1)} step(s)"
    return flat_values, axis_map, steps_label


@registry.tool(
    name="measurement_stats",
    description=(
        "Use to AGGREGATE .MEAS scalar results across a .step sweep or Monte "
        "Carlo run. Answers questions like 'across 100 MC trials, what's the "
        "worst-case rise time?' or 'how does gain vary as R sweeps 1k..10k?'. "
        "Inputs the .log file produced by the run.\n\n"
        "Returns per-measurement: min, max, mean, median, std, p10, p90, "
        "best_step_index (argmin) and worst_step_index (argmax), failure "
        "count, and an optional histogram (set histogram_bins=0 to skip).\n\n"
        "Accepts any job id: a sweep/MC batch aggregates across its runs; a "
        "single-simulation job aggregates its own log (one value per step "
        "for a .step run). Axis choice differs by shape: a batch detects "
        "WHEN-style .MEAS (constant level, varying crossing) and swaps to "
        "aggregating the 'at' field; a stepped single-run log always "
        "aggregates the 'value' field. The aggregated_field output says "
        "which was used. On a plain single run there's only one value per "
        "measurement, so stats collapse to n=1 — use simulation_summary "
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
        job = services.resolve_job(args.job_id, state)
        if isinstance(job, BatchJob):
            flat_values, run_count, axis_map, run_diags = _aggregate_job_measurements(job)
            if not flat_values:
                if run_count == 0:
                    raise ResultError(
                        f"No .MEAS results found across the runs of job {args.job_id!r} "
                        "— none of the per-run log files could be read."
                    )
                # Relay WHY from the per-run logs, in the same indented format
                # as the single-log branch — the cause is often stated there
                # verbatim (e.g. ngspice's batch-mode .meas skip). Capped:
                # run-unique lines (timestamps, values) survive deduplication,
                # which would otherwise grow the message one line per run on
                # a large Monte Carlo batch.
                shown = run_diags[:_MAX_RELAYED_RUN_DIAGNOSTICS]
                hidden = len(run_diags) - len(shown)
                if hidden:
                    shown.append(f"... and {hidden} more distinct diagnostic lines")
                err_block = _diagnostics_block(
                    shown, "(run logs contained no .MEAS results and no diagnostics)"
                )
                raise ResultError(
                    f"No .MEAS results found across the runs of job {args.job_id!r}:\n{err_block}"
                )
            steps_label = f"{run_count} run(s)"
        else:
            # Single-simulation job: aggregate its one log, which is the same
            # physical shape as the ``log_file`` input (a .step run carries one
            # value per step; a plain run collapses to n=1 stats).
            # ``resolve_log_file`` gates on completed status like every other
            # job-id-addressed read; the path is a trusted server artifact.
            log_path = services.resolve_log_file(args.job_id, state)
            flat_values, axis_map, steps_label = _aggregate_log_measurements(log_path)
    elif args.log_file is not None:
        flat_values, axis_map, steps_label = _aggregate_log_measurements(
            safe_path(args.log_file, state)
        )
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
    """Parse a SPICE-notation frequency into a finite positive float.

    Tolerates a trailing ``Hz`` unit — ``'159Hz'`` and ``'15.9kHz'`` are the
    natural way to write a frequency, but the SPICE value parser only knows SI
    prefixes (k, meg, …). Strip a trailing ``hz`` before parsing so the unit is
    accepted rather than rejected with a confusing error.
    """
    cleaned = s.strip()
    if cleaned[-2:].lower() == "hz":
        cleaned = cleaned[:-2].strip()
    try:
        v = parse_spice_value(cleaned)
    except ValueError as e:
        raise ResultError(f"Invalid {name} value {s!r}: {e}", show_hint=False) from e
    if not math.isfinite(v):
        raise ResultError(f"{name} must be finite, got {s!r}")
    if v <= 0:
        raise ResultError(f"{name} must be positive, got {s!r} ({v})")
    return v


async def _load_ac_signal(
    raw_file: str | Path, signal: str, step: int, state: SessionState
) -> tuple[np.ndarray, np.ndarray]:
    """Load (freqs, H) for an AC signal. Rejects transient data.

    A ``Path`` is treated as an already-resolved, trusted artifact (a job run's
    raw, resolved via the read-model) and loaded directly; a ``str`` is untrusted
    user input and validated via ``safe_path``.
    """
    raw_path = raw_file if isinstance(raw_file, Path) else safe_path(raw_file, state)
    raw = await services.load_raw(raw_path, state)
    sim_type = detect_sim_type(raw)
    if not is_ac_analysis(sim_type):
        raise ResultError(
            f"This tool requires AC analysis data; got {sim_type!r}. "
            "Use signal_stats (transient) or run a .AC sweep first."
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
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
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
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
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
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
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
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
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
    max_peaks: int = Field(default=20, description="Maximum peaks returned (1..1000)")
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


# Internal compute adapter — exposed publicly via bode_metrics(mode="crossing").
async def handle_find_crossing(args: FindCrossingInput, state: SessionState):
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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


# Internal compute adapter — exposed publicly via bode_metrics(mode="point").
async def handle_gain_at(args: GainAtInput, state: SessionState):
    if not args.frequencies:
        raise ResultError("frequencies list is empty")
    if len(args.frequencies) > 1000:
        raise ResultError(f"Too many frequencies ({len(args.frequencies)}); cap is 1000")
    freqs_q = [_parse_freq(f, "frequency") for f in args.frequencies]
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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


# Internal compute adapter — exposed publicly via bode_metrics(mode="filter").
async def handle_filter_metrics(args: FilterMetricsInput, state: SessionState):
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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
    name="stability_metrics",
    description=(
        "Find EVERY unity-gain and -180° phase crossover in a loop-gain AC "
        "sweep, report phase margin at each unity-gain crossing and gain "
        "margin at each -180° crossing. Replaces the single-crossing "
        "approximation in simulation_summary, which returns wrong "
        "margins on conditionally-stable systems.\n\n"
        "Run this on a LOOP-GAIN signal (typically a dedicated middlebrook "
        "probe or .AC of the open loop). Running on a closed-loop output "
        "gives meaningless margins — if the DC phase starts near ±180° (a "
        "closed-loop / inverting output rather than a loop probe, which "
        "starts near 0°), a warning says so in ``warnings``.\n\n"
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
        "For -3 dB filter cutoffs use bode_metrics(mode='filter'); for custom "
        "crossings use bode_metrics(mode='crossing')."
    ),
    input_model=StabilityMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=StabilityMetricsResponse,
)
async def handle_stability_metrics(args: StabilityMetricsInput, state: SessionState):
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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


# Internal compute adapter — exposed publicly via bode_metrics(mode="slope").
async def handle_roll_off(args: RollOffInput, state: SessionState):
    f_lo = _parse_freq(args.f_low, "f_low")
    f_hi = _parse_freq(args.f_high, "f_high")
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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


BodeMode = Literal["filter", "slope", "point", "crossing"]


def _bode_output_schema() -> dict:
    """Union object schema across the four mode shapes for client introspection.

    Merges the per-mode response TypedDicts so the published ``outputSchema``
    accurately documents every key a mode can return (each is optional — the
    actual key set depends on ``mode``). Reusing the TypedDicts keeps the
    mode shapes single-sourced with the internal compute adapters.
    """
    merged: dict = {}
    for td in (FilterMetricsResponse, RollOffResponse, GainAtResponse, FindCrossingResponse):
        merged.update(schema_from_typeddict(td).get("properties", {}))
    # all_steps mode wraps per-step results under ``steps``; a step entry is a
    # mode result plus its ``step`` index (or an ``error`` if that step failed).
    step_item = {
        "type": "object",
        "properties": {
            **merged,
            "step": {"type": "integer"},
            "error": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "properties": {
            **merged,
            "mode": {"type": "string"},
            "signal": {"type": "string"},
            "all_steps": {"type": "boolean"},
            "step_count": {"type": "integer"},
            "steps": {"type": "array", "items": step_item},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
    }


class BodeMetricsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to AC analysis .raw result file. Pass this OR ``job_id``, not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``. Combine with ``all_steps`` "
            "to sweep the .step axis WITHIN that run."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    mode: BodeMode = Field(
        description=(
            "Which view of the AC response to compute:\n"
            "  'filter'   — LPF/HPF/BPF/BSF type, cutoffs, ripple, rejection "
            "(args: ref_db, flatness_db, passband_range, stopband_range)\n"
            "  'slope'    — magnitude slope between two frequencies "
            "(args: f_low, f_high — both required)\n"
            "  'point'    — magnitude (dB + linear) and phase at specific "
            "frequencies (args: frequencies — required; include_unwrapped_phase)\n"
            "  'crossing' — every frequency where magnitude/phase crosses a "
            "level (args: quantity + level — required; direction, f_start, "
            "f_end, max_results, min_separation_decades)"
        )
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    all_steps: bool = Field(
        default=False,
        description=(
            "Compute the metric for EVERY step of a stepped (.step) sweep in one "
            "call, instead of the single `step`. Returns `steps`: a list of "
            "per-step results (each tagged with its `step` index). A step whose "
            "computation fails is returned with an `error` field rather than "
            "aborting the whole call. On a non-stepped raw this returns a single "
            "entry. Use this for 'give me the cutoff/slope/gain at every step'."
        ),
    )
    # mode="crossing"
    quantity: Quantity | None = Field(
        default=None,
        description="crossing: 'magnitude_db' | 'magnitude_linear' | 'phase_deg'.",
    )
    level: float | None = Field(
        default=None, description="crossing: level to cross, in the units of `quantity`."
    )
    direction: SearchDirection = Field(default="any", description="crossing: edge direction.")
    f_start: str | None = Field(default=None, description="crossing: lower frequency bound.")
    f_end: str | None = Field(default=None, description="crossing: upper frequency bound.")
    max_results: int = Field(default=10, description="crossing: cap on returned crossings.")
    min_separation_decades: float = Field(
        default=0.0, description="crossing: merge crossings within this many decades."
    )
    # mode="point"
    frequencies: list[str] | None = Field(
        default=None, description="point: frequencies to query (SPICE notation)."
    )
    include_unwrapped_phase: bool = Field(
        default=False, description="point: also return cumulative unwrapped phase."
    )
    # mode="filter"
    ref_db: float = Field(
        default=-3.0, description="filter: cutoff reference below passband (dB)."
    )
    flatness_db: float = Field(
        default=1.0, description="filter: passband flatness tolerance (dB)."
    )
    passband_range: list[str] | None = Field(
        default=None, description="filter: optional [f_lo, f_hi] passband override."
    )
    stopband_range: list[str] | None = Field(
        default=None, description="filter: optional [f_lo, f_hi] stopband region."
    )
    # mode="slope"
    f_low: str | None = Field(default=None, description="slope: low frequency bound (required).")
    f_high: str | None = Field(default=None, description="slope: high frequency bound (required).")
    format: FormatField = Field(default=None)


@registry.tool(
    name="bode_metrics",
    description=(
        "AC / Bode-plot analysis in one tool, selected by `mode`. The response "
        "shape depends on the mode:\n"
        "  mode='filter'   — filter type, cutoffs (at `ref_db` below passband), "
        "passband gain/ripple, stopband rejection, transition BW, pole-order.\n"
        "  mode='slope'    — magnitude slope (dB/decade + dB/octave) between "
        "`f_low` and `f_high`; pick endpoints ≥1 decade past any knee.\n"
        "  mode='point'    — magnitude (dB + linear) and phase at each of "
        "`frequencies` (log-axis interpolation; out-of-range clamps + warns).\n"
        "  mode='crossing' — every frequency where `quantity` crosses `level` "
        "(phase is UNWRAPPED first); the escape hatch for custom queries like "
        "unity-gain (0 dB) or phase-margin (-180°) frequencies.\n\n"
        "Pass `all_steps=true` to compute the chosen mode for every step of a "
        ".step sweep in one call (returns a `steps` list instead of a single "
        "result) — e.g. the -3 dB cutoff at every value of a stepped component.\n\n"
        "To analyze a run of a completed sweep/MC job, pass `job_id` + "
        "`run_index` instead of `raw_file` (combine with `all_steps` to also "
        "sweep the .step axis within that run).\n\n"
        "For loop-gain stability margins use stability_metrics; for resonant "
        "peaks & Q use resonance."
    ),
    input_model=BodeMetricsInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_schema=_bode_output_schema(),
)
async def handle_bode_metrics(args: BodeMetricsInput, state: SessionState):
    """Dispatch to the per-mode AC compute adapters (one shared AC load each)."""
    _validate_bode_mode_args(args)
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    if not args.all_steps:
        return await _bode_dispatch(args, args.step, state, raw_path)

    # all_steps: compute the metric for every step of the sweep.
    raw = await services.load_raw(raw_path, state)
    step_count = get_step_count(raw)

    steps_out: list[dict] = []
    step_texts: list[str] = []
    # Distinct per-step warning -> the step indices that emitted it. A warning
    # that fires identically on every step (e.g. the no-stopband_range sweep-
    # endpoint note) is surfaced ONCE at the top level with its step coverage,
    # not repeated per step in both the structured 'steps' and the text.
    warning_steps: dict[str, list[int]] = {}
    first_error: ResultError | None = None
    for i in range(step_count):
        try:
            res = await _bode_dispatch(args, i, state, raw_path)
            sc = _structured(res)
            for w in sc.pop("warnings", None) or []:
                warning_steps.setdefault(w, []).append(i)
            steps_out.append({"step": i, **sc})
            step_texts.append(f"── step {i} ──\n{_strip_warning_block(_result_text(res))}")
        except ResultError as e:
            if first_error is None:
                first_error = e
            steps_out.append({"step": i, "error": str(e)})
            step_texts.append(f"── step {i} ── error: {e}")

    errored = [s for s in steps_out if "error" in s]
    if step_count and len(errored) == step_count and first_error is not None:
        # Every step failed (e.g. a non-AC raw fed to all_steps) — re-raise the
        # ORIGINAL error so its show_hint/suggestions survive, instead of a
        # "success" full of buried per-step errors.
        raise first_error

    data: dict = {
        "mode": args.mode,
        "signal": args.signal,
        "all_steps": True,
        "step_count": step_count,
        "steps": steps_out,
    }
    warnings: list[str] = []
    if step_count == 1:
        warnings.append("Raw is not stepped (step_count=1); 'steps' has a single entry.")
    for w, idxs in warning_steps.items():
        warnings.append(f"{w} ({_warning_coverage(idxs, step_count)})")
    if errored:
        warnings.append(f"{len(errored)} of {step_count} steps failed (see per-step 'error').")
    if warnings:
        data["warnings"] = warnings

    header = [
        f"bode_metrics(mode={args.mode!r}, all_steps) — {args.signal}",
        f"Steps: {step_count}",
        *(f"⚠ {w}" for w in warnings),
        "",
    ]
    return format_response("\n".join(header) + "\n".join(step_texts), data, args.format)


def _validate_bode_mode_args(args: BodeMetricsInput) -> None:
    """Raise for missing per-mode required args. Called once up front so a
    caller mistake surfaces immediately instead of being swallowed per-step in
    ``all_steps`` mode."""
    if args.mode == "crossing" and (args.quantity is None or args.level is None):
        raise ResultError("bode_metrics mode='crossing' requires 'quantity' and 'level'.")
    if args.mode == "point" and not args.frequencies:
        raise ResultError("bode_metrics mode='point' requires 'frequencies'.")
    if args.mode == "slope" and (args.f_low is None or args.f_high is None):
        raise ResultError("bode_metrics mode='slope' requires 'f_low' and 'f_high'.")


async def _bode_dispatch(
    args: BodeMetricsInput, step: int, state: SessionState, raw_path: Path
) -> types.CallToolResult:
    """Build the per-mode input for ``step`` and dispatch to its compute adapter.

    ``raw_path`` is the already-resolved .raw (from raw_file or a job run) — the
    adapters load it directly (trusted Path), so a sweep run is analyzed by the
    same AC machinery as a standalone raw. Assumes ``_validate_bode_mode_args``
    has already validated required args.
    """
    if args.mode == "crossing":
        assert args.quantity is not None and args.level is not None
        return await handle_find_crossing(
            FindCrossingInput(
                raw_file=raw_path,
                signal=args.signal,
                quantity=args.quantity,
                level=args.level,
                direction=args.direction,
                f_start=args.f_start,
                f_end=args.f_end,
                max_results=args.max_results,
                min_separation_decades=args.min_separation_decades,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "point":
        assert args.frequencies
        return await handle_gain_at(
            GainAtInput(
                raw_file=raw_path,
                signal=args.signal,
                frequencies=args.frequencies,
                include_unwrapped_phase=args.include_unwrapped_phase,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "filter":
        return await handle_filter_metrics(
            FilterMetricsInput(
                raw_file=raw_path,
                signal=args.signal,
                ref_db=args.ref_db,
                flatness_db=args.flatness_db,
                passband_range=args.passband_range,
                stopband_range=args.stopband_range,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "slope":
        assert args.f_low is not None and args.f_high is not None
        return await handle_roll_off(
            RollOffInput(
                raw_file=raw_path,
                signal=args.signal,
                f_low=args.f_low,
                f_high=args.f_high,
                step=step,
                format=args.format,
            ),
            state,
        )
    raise ResultError(f"Unknown bode_metrics mode {args.mode!r}")


def _structured(result: types.CallToolResult) -> dict:
    """structuredContent of an adapter result as a dict (``{}`` if absent)."""
    return dict(result.structuredContent) if result.structuredContent else {}


def _result_text(result: types.CallToolResult) -> str:
    block = result.content[0] if result.content else None
    return block.text if isinstance(block, types.TextContent) else ""


def _strip_warning_block(text: str) -> str:
    """Drop the trailing ``_warning_lines`` block from an adapter's rendered
    text. In all_steps mode the per-step warnings are hoisted (deduped) to the
    top level, so repeating them inside every step's text is noise."""
    return text.split(f"\n\n{_WARNINGS_HEADER}\n", 1)[0]


def _warning_coverage(step_indices: list[int], step_count: int) -> str:
    """Describe which steps a hoisted all_steps warning applies to. Lists the
    actual step indices for ANY partial subset — never a bare count — so a
    consumer can still identify exactly which sweep cases emitted it; only the
    every-step case collapses to a compact label."""
    if len(step_indices) == step_count:
        return f"all {step_count} steps"
    return "steps " + ",".join(str(i) for i in step_indices)


@registry.tool(
    name="resonance",
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
        "For overall filter characterization use bode_metrics(mode='filter'); "
        "for stability margins use stability_metrics."
    ),
    input_model=ResonanceInput,
    annotations=RO_ANNOTATIONS,
    profiles=("full", "agentic"),
    output_model=ResonancesResponse,
)
async def handle_resonance(args: ResonanceInput, state: SessionState):
    if args.max_peaks < 1 or args.max_peaks > 1000:
        raise ResultError(f"max_peaks must be in [1, 1000], got {args.max_peaks}")
    freqs, H = await _load_ac_signal(args.raw_file, args.signal, args.step, state)
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


# ---------------------------------------------------------------------------
# plot_waveform — interactive HTML chart opened on the local desktop
# ---------------------------------------------------------------------------

PLOTS_SUBDIR = "plots"
_DEFAULT_PLOT_MAX_POINTS = 100_000
_PLOT_MAX_POINTS_CEILING = 2_000_000
# Global backstop on the rendered panel size. ``max_points`` caps each series,
# but null-padding every series onto a union x (distinct per-step axes) multiplies
# series-count by union-length — so a many-step distinct-axis run could blow far
# past the per-series cap. Refuse with guidance before materializing, rather than
# allocate a giant payload / write an unopenable HTML (no silent truncation).
_PLOT_MAX_CELLS = 10_000_000

_X_LABEL = {
    "transient": "Time (s)",
    "ac": "Frequency (Hz)",
    "noise": "Frequency (Hz)",
    "dc": "Sweep",
}


def _plot_filename(raw_path: Path, analysis_type: str, job_id: str | None, run_index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    run = f"_run{run_index}" if job_id else ""
    return f"{raw_path.stem}_{analysis_type}{run}_{stamp}.html"


def _to_json_floats(arr: np.ndarray) -> list[float | None]:
    """Array -> JSON-safe list; non-finite samples become ``None`` (a uPlot gap).

    Pairs with ``json.dumps(allow_nan=False)`` in the renderer: a literal NaN/Inf
    token would make the browser's ``JSON.parse`` reject the whole blob, silently
    blanking the chart — so non-finite values are converted to null here.
    """
    return [v if math.isfinite(v) else None for v in np.asarray(arr, dtype=float).tolist()]


def _plot_cells_exceeded(n_rows: int, x_len: int) -> ResultError:
    return ResultError(
        f"This plot would render ~{n_rows * x_len:,} cells ({n_rows - 1} series x "
        f"{x_len:,} x-points), over the {_PLOT_MAX_CELLS:,} cap. Stepped runs with "
        "distinct per-step axes inflate the shared x-axis — plot fewer signals, a "
        "single step (step=N), or narrow [t_start, t_end]."
    )


def _union_panel(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    x_scale: str,
    x_label: str,
    y_label: str,
) -> tuple[dict, bool]:
    """Build a uPlot panel (shared x + N y-series) from per-series ``(x, y, label)``.

    Series whose x differs from the others — e.g. a transient ``.step`` run where
    each step has its own adaptive time vector — are null-padded onto the union x
    so each renders as a clean gap off its own support (uPlot's data model is one
    shared x-row + N y-series). Returns ``(panel, unioned)``.

    Guards the rendered size in two cheap stages so neither the concat nor the pad
    can blow up: the longest single series is a lower bound on the union (stage 1,
    before concatenating — also bounds the concat to <= the cap), and the actual
    union length is the exact size (stage 2, before padding).
    """
    n_rows = len(series) + 1  # the x row plus one row per series
    longest = max(len(s[0]) for s in series)
    if n_rows * longest > _PLOT_MAX_CELLS:
        raise _plot_cells_exceeded(n_rows, longest)
    union = series[0][0] if len(series) == 1 else np.unique(np.concatenate([s[0] for s in series]))
    if n_rows * len(union) > _PLOT_MAX_CELLS:
        raise _plot_cells_exceeded(n_rows, len(union))
    data: list[list[float | None]] = [_to_json_floats(union)]
    labels: list[dict[str, str]] = []
    unioned = False
    for x, y, label in series:
        if len(x) == len(union) and np.array_equal(x, union):
            data.append(_to_json_floats(y))
        else:
            unioned = True
            col = np.full(len(union), np.nan)
            col[np.searchsorted(union, x)] = y
            data.append(_to_json_floats(col))
        labels.append({"label": label})
    panel = {
        "x_scale": x_scale,
        "x_label": x_label,
        "y_label": y_label,
        "series": labels,
        "data": data,
    }
    return panel, unioned


def _build_plot_and_write(
    raw,
    raw_path: Path,
    cols: list[str],
    steps_to_plot: list[int],
    step_dicts: list[dict[str, float]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    max_points: int,
    out_path: Path,
    title: str,
) -> dict:
    """Build per-series plot data, assemble the HTML, write it atomically.

    Runs in a worker thread (heavy numpy + HTML string build + file I/O) and
    returns only FACTS — the handler turns them into observations and builds the
    response on the event loop (the concurrency contract keeps response building
    off worker threads).
    """
    is_ac = analysis_type == "ac"
    x_label = _X_LABEL[analysis_type]
    multi = len(steps_to_plot) > 1

    def _label(col: str, step: int) -> str:
        if not multi:
            return col
        sv = (
            ";".join(f"{k}={v:g}" for k, v in step_dicts[step].items())
            if step < len(step_dicts)
            else ""
        )
        return f"{col} [{sv}]" if sv else f"{col} [step {step}]"

    empty_steps: list[int] = []
    non_finite = 0
    downsampled = False
    points_per_series: list[int] = []
    phase_warnings: list[str] = []
    win_lo: float | None = None
    win_hi: float | None = None

    def _track_window(x: np.ndarray) -> None:
        nonlocal win_lo, win_hi
        lo0, hi0 = float(x[0]), float(x[-1])
        win_lo = lo0 if win_lo is None else min(win_lo, lo0)
        win_hi = hi0 if win_hi is None else max(win_hi, hi0)

    if is_ac:
        mag_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        phase_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        for step in steps_to_plot:
            axis = _guarded_axis(raw, step)
            lo, hi = _window_indices(axis, ts, te)
            if lo >= hi:
                empty_steps.append(step)
                continue
            for col in cols:
                wave = np.asarray(raw.get_wave(col, step=step))[lo:hi]
                freq, h = prepare_ac_arrays(axis[lo:hi], wave)
                mag = safe_magnitude_db(h)
                phase, warns = unwrap_phase_safe(h)
                phase_warnings.extend(warns)
                non_finite += int(np.count_nonzero(~np.isfinite(mag)))
                non_finite += int(np.count_nonzero(~np.isfinite(phase)))
                if len(freq) > max_points:
                    downsampled = True
                    f_ds, mag = downsample_minmax(freq, mag, max_points)
                    _, phase = downsample_minmax(freq, phase, max_points)
                    freq = f_ds
                _track_window(freq)
                points_per_series.append(len(freq))
                label = _label(col, step)
                mag_series.append((freq, mag, label))
                phase_series.append((freq, phase, label))
        if not mag_series:
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {len(steps_to_plot)} steps." if multi else ".")
            )
        mag_panel, u1 = _union_panel(mag_series, "log", x_label, "Magnitude (dB)")
        phase_panel, u2 = _union_panel(phase_series, "log", x_label, "Phase (deg)")
        unioned = u1 or u2
        spec = {"analysis_type": analysis_type, "bode": True, "panels": [mag_panel, phase_panel]}
    else:
        plot_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        for col in cols:
            for step in steps_to_plot:
                axis = _guarded_axis(raw, step)
                lo, hi = _window_indices(axis, ts, te)
                if lo >= hi:
                    if step not in empty_steps:
                        empty_steps.append(step)
                    continue
                axis_w = axis[lo:hi]
                wave = np.asarray(raw.get_wave(col, step=step))[lo:hi]
                if np.iscomplexobj(wave):
                    # Defensive: a stray complex trace in a non-AC raw.
                    wave = np.real(wave)
                non_finite += int(np.count_nonzero(~np.isfinite(wave)))
                x_arr, y_arr = axis_w, wave
                if len(y_arr) > max_points:
                    downsampled = True
                    x_arr, y_arr = downsample_minmax(axis_w, wave, max_points)
                _track_window(x_arr)
                points_per_series.append(len(y_arr))
                plot_series.append((x_arr, y_arr, _label(col, step)))
        if not plot_series:
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {len(steps_to_plot)} steps." if multi else ".")
            )
        y_label = ", ".join(cols) if len(cols) <= 3 else f"{len(cols)} signals"
        panel, unioned = _union_panel(
            plot_series, "log" if x_is_log else "linear", x_label, y_label
        )
        spec = {"analysis_type": analysis_type, "bode": False, "panels": [panel]}

    series_count = sum(len(p["series"]) for p in spec["panels"])
    summary = f"{raw_path.stem} — {analysis_type}: {series_count} series"
    html_str = build_plot_html(spec, title=title, summary=summary)
    with atomic_write(out_path) as f:
        f.write(html_str)

    return {
        "panels": len(spec["panels"]),
        "series_count": series_count,
        "points_per_series": points_per_series,
        "downsampled": downsampled,
        "unioned": unioned,
        "empty_steps": sorted(set(empty_steps)),
        "non_finite": non_finite,
        "phase_unwrapped": is_ac,
        "phase_warnings": phase_warnings,
        "window_used": [win_lo, win_hi] if win_lo is not None else [],
        "step_values_available": (bool(step_dicts) if multi else None),
    }


class PlotWaveformInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Plot a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    signals: list[str] | Literal["all"] = Field(
        default="all",
        description="Trace names to plot (e.g. ['V(out)', 'I(R1)']) or 'all' for every non-axis trace.",
    )
    step: int | None = Field(
        default=None,
        description=(
            "For a .step run: omit to overlay ALL steps as separate traces, or give "
            "a 0-based step index to plot just that one."
        ),
    )
    t_start: str | None = Field(
        default=None,
        description="Window start in SPICE notation (e.g. '1m', '1k'); bounds the plotted range.",
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation.",
    )
    max_points: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Per-series point budget before a min/max-preserving downsample engages "
            f"(default {_DEFAULT_PLOT_MAX_POINTS}). Full fidelity below this; spikes "
            "are preserved when it engages."
        ),
    )
    open: bool = Field(
        default=True,
        description="Open the written HTML in the local browser (terminal clients). Set false to only write it.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description="Response format: 'json' for structured data, 'text' for human-readable",
    )


@registry.tool(
    name="plot_waveform",
    description=(
        "Render an INTERACTIVE chart of one or more signals and open it on your "
        "desktop — for a human to see the SHAPE and zoom/pan/hover, the co-design "
        "complement to the numeric tools.\n\n"
        "Picks the chart from the run type: transient (V/I vs time), DC sweep, AC "
        "Bode (stacked magnitude-dB + phase-deg vs log frequency), noise (vs log "
        "frequency); a .step / Monte-Carlo run overlays every step as a labelled "
        "trace (or pass ``step`` for one). Full fidelity by default, with a "
        "min/max-preserving downsample above ``max_points`` (spikes survive; "
        "surfaced as a fact). Writes a self-contained HTML file next to the circuit "
        "and returns its path; on a terminal client it also opens it locally.\n\n"
        "For numbers to compute on use export_waveform (CSV) or the scalar tools "
        "(signal_stats, bode_metrics); this tool is for looking, not measuring."
    ),
    input_model=PlotWaveformInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    profiles=("full", "agentic"),
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "analysis_type": {"type": "string", "enum": ["transient", "ac", "dc", "noise"]},
            "signals": {"type": "array", "items": {"type": "string"}},
            "n_steps": {"type": "integer"},
            "steps_plotted": {"type": "integer"},
            "panels": {"type": "integer"},
            "series_count": {"type": "integer"},
            "points_per_series": {"type": "array", "items": {"type": "integer"}},
            "max_points": {"type": "integer"},
            "downsampled": {"type": "boolean"},
            "window_used": {"type": "array", "items": {"type": "number"}},
            "opened": {"type": "boolean"},
            "opener": {"type": ["string", "null"]},
            "observations": OBSERVATIONS_SCHEMA,
        },
    },
)
async def handle_plot_waveform(args: PlotWaveformInput, state: SessionState):
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    fmt = args.format
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")

    raw = await services.load_raw(raw_path, state)
    # A .op raw has no sweep axis to plot — refuse early with the clean pointer.
    _guarded_axis(raw, 0)

    _, analysis_type, _, x_is_log = _classify_analysis(raw)

    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]
    if args.signals == "all":
        cols = list(trace_names[1:])
    else:
        seen: set[str] = set()
        cols = []
        for s in args.signals:
            canon = services.validate_signal(raw, s)
            if canon == axis_name:
                raise ResultError(f"{s!r} is the sweep axis, not a signal column.")
            if canon not in seen:
                seen.add(canon)
                cols.append(canon)
    if not cols:
        raise ResultError("No signal traces to plot (the result has only an axis).")

    n_steps = get_step_count(raw)
    if args.step is not None:
        services.validate_step(raw, args.step)
        steps_to_plot = [args.step]
    else:
        steps_to_plot = list(range(n_steps))

    ts = _parse_time(args.t_start, "t_start")
    te = _parse_time(args.t_end, "t_end")

    step_dicts: list[dict[str, float]] = []
    if len(steps_to_plot) > 1:
        log_path = raw_path.with_suffix(".log")
        if log_path.exists():
            step_dicts = parse_step_iterations(log_path)

    max_points = min(args.max_points or _DEFAULT_PLOT_MAX_POINTS, _PLOT_MAX_POINTS_CEILING)

    # Destination: a Linux-side sidecar next to the CIRCUIT, never next to a raw
    # that may live in a Windows temp under /mnt/c. Output is a server artifact,
    # so it is not run through safe_path — but the resolved path must stay under
    # the anchor (a symlinked sidecar would otherwise redirect it out).
    if args.job_id:
        dest_anchor = services.resolve_job(args.job_id, state).netlist.parent
    else:
        dest_anchor = safe_path(args.raw_file, state).parent  # type: ignore[arg-type]
    out_path = (
        dest_anchor
        / SIDECAR_DIRNAME
        / PLOTS_SUBDIR
        / _plot_filename(raw_path, analysis_type, args.job_id, args.run_index)
    ).resolve()
    if not out_path.is_relative_to(dest_anchor.resolve()):
        raise ResultError(
            "Refusing to write the plot outside the circuit directory "
            "(a symlinked .ltspice-mcp/ or plots/ would redirect it)."
        )

    title = f"{raw_path.stem} — {analysis_type}"
    try:
        facts = await asyncio.to_thread(
            _build_plot_and_write,
            raw,
            raw_path,
            cols,
            steps_to_plot,
            step_dicts,
            analysis_type,
            x_is_log,
            ts,
            te,
            max_points,
            out_path,
            title,
        )
    except ValueError as e:
        raise ResultError(f"Failed to build the plot (corrupt or truncated .raw?): {e}") from e

    opened, opener = False, None
    if args.open:
        opened, opener = await asyncio.to_thread(desktop.open_in_desktop, out_path)

    # Surface FACTS, not verdicts (result-trust doctrine).
    observations: list[dict] = [
        {
            "code": "plot_written",
            "kind": "coverage",
            "detail": (
                f"Wrote an interactive {analysis_type} plot: {facts['panels']} panel(s), "
                f"{facts['series_count']} series ({len(steps_to_plot)} of {n_steps} step(s))."
            ),
        }
    ]
    if facts["downsampled"]:
        observations.append(
            {
                "code": "downsampled",
                "kind": "coverage",
                "detail": (
                    f"At least one series exceeded {max_points} points and was reduced by "
                    "min/max-preserving decimation (spikes preserved; sub-bucket detail not "
                    "shown). Pass a larger max_points or narrow [t_start, t_end] for more."
                ),
            }
        )
    if facts["phase_unwrapped"]:
        observations.append(
            {
                "code": "phase_unwrapped",
                "kind": "value",
                "detail": (
                    "Bode phase is UNWRAPPED for a readable continuous curve — this differs "
                    "from export_waveform, which keeps the wrapped np.angle as its lossless "
                    "primitive."
                ),
            }
        )
    for warn in facts["phase_warnings"]:
        observations.append({"code": "sparse_sweep", "kind": "value", "detail": warn})
    if facts["unioned"]:
        observations.append(
            {
                "code": "step_axis_unioned",
                "kind": "coverage",
                "detail": (
                    "Steps have different per-step x vectors; series were aligned onto a "
                    "union x (each renders as a gap off its own support)."
                ),
            }
        )
    if facts["empty_steps"]:
        observations.append(
            {
                "code": "window_empty_steps",
                "kind": "coverage",
                "detail": (
                    f"{len(facts['empty_steps'])} step(s) had no samples in the window and "
                    f"were omitted: {facts['empty_steps']}."
                ),
            }
        )
    if facts["non_finite"]:
        observations.append(
            {
                "code": "non_finite",
                "kind": "value",
                "detail": (
                    f"{facts['non_finite']} non-finite sample(s) are present; they render as "
                    "gaps in the plot."
                ),
            }
        )
    if facts["step_values_available"] is False:
        observations.append(
            {
                "code": "step_value_unavailable",
                "kind": "value",
                "detail": (
                    "Step legend labels left blank: no .step parameter map found in the "
                    "sibling .log."
                ),
            }
        )
    if not args.open:
        observations.append(
            {
                "code": "open_skipped",
                "kind": "coverage",
                "detail": "Local open skipped (open=false); open the returned path manually.",
            }
        )
    elif not opened:
        observations.append(
            {
                "code": "open_failed",
                "kind": "coverage",
                "detail": (
                    "Could not launch a local opener (headless or none found); open the "
                    "returned path manually."
                ),
            }
        )

    data = {
        "path": str(out_path),
        "analysis_type": analysis_type,
        "signals": cols,
        "n_steps": n_steps,
        "steps_plotted": len(steps_to_plot),
        "panels": facts["panels"],
        "series_count": facts["series_count"],
        "points_per_series": facts["points_per_series"],
        "max_points": max_points,
        "downsampled": facts["downsampled"],
        "window_used": facts["window_used"],
        "opened": opened,
        "opener": opener,
        "observations": observations,
    }
    head = f"Wrote interactive {analysis_type} plot to {out_path}"
    if opened:
        head += f" (opened with {opener})"
    lines = [head, *format_observations(observations)]
    return format_response("\n".join(lines), data, fmt)
