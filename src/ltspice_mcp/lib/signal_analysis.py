"""Pure-function signal-analysis primitives for transient .raw data.

Takes numpy arrays in, returns dicts of Python floats / None. No I/O, no
spicelib dependencies. Raises ``ValueError`` with user-facing messages on
domain errors — the tool layer re-raises these as ``ResultError``.

Depends on numpy + scipy.signal.find_peaks; no other third-party code.

These primitives operate on real-valued transient data only. The tool layer
rejects AC analysis before calling in.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import numpy as np
from scipy.signal import find_peaks

_LEVEL_EPSILON = 1e-12

CrossingDirection = Literal["rising", "falling"]


class EdgeMetricsOutput(TypedDict):
    """Return shape of :func:`analyze_edge`."""

    transition_time: float
    slew_rate: float
    low_level: float
    high_level: float
    t_low_crossing: float
    t_high_crossing: float
    t_mid_crossing: float
    edge_direction: CrossingDirection
    is_rise_time: bool
    low_pct: float
    high_pct: float
    num_edges_in_window: int
    warnings: list[str]


class PulseResponseOutput(TypedDict):
    """Return shape of :func:`analyze_pulse_response`."""

    direction: CrossingDirection
    initial_value: float
    steady_state_value: float
    peak_value: float
    peak_time: float
    overshoot_pct: float
    undershoot_pct: float
    settling_time: float | None
    settling_tolerance_pct: float
    warnings: list[str]


class TimingBetweenOutput(TypedDict):
    """Return shape of :func:`analyze_timing_between`."""

    t_a: float
    t_b: float
    delay: float
    threshold_a_used: float
    threshold_b_used: float
    direction_a: CrossingDirection
    direction_b: CrossingDirection
    num_crossings_a: int
    num_crossings_b: int
    warnings: list[str]


class PeriodicMetricsOutput(TypedDict):
    """Return shape of :func:`analyze_periodic`."""

    period: float
    frequency: float
    jitter_rms: float
    duty_cycle_pct: float | None
    pulse_width_high: float | None
    pulse_width_low: float | None
    num_rising_edges: int
    num_falling_edges: int
    num_periods_measured: int
    threshold_used: float
    warnings: list[str]


class SignalStatsOutput(TypedDict):
    """Return shape of :func:`compute_signal_stats`. Transient-only."""

    t_start: float
    t_end: float
    duration: float
    num_samples: int
    mean: float
    rms: float
    std: float
    abs_mean: float
    min: float
    max: float
    pk_pk: float


class HistogramBin(TypedDict):
    """One bin in a ``.MEAS`` value histogram."""

    bin_start: float
    bin_end: float
    count: int


class MeasurementStatsEntry(TypedDict):
    """Per-measurement aggregate stats in :func:`compute_measurement_stats`."""

    total_count: int
    valid_count: int
    failure_count: int
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    std: float | None
    p10: float | None
    p90: float | None
    best_step_index: int | None
    worst_step_index: int | None
    histogram: list[HistogramBin]


def window_and_clean(
    t: np.ndarray,
    y: np.ndarray,
    t_start: float | None,
    t_end: float | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slice to ``[t_start, t_end]`` and strip non-finite samples.

    Returns ``(t_clean, y_clean, dropped_nonfinite)``. Requires at least 3
    samples after cleaning.
    """
    if len(t) != len(y):
        raise ValueError(f"Axis and wave have different lengths: {len(t)} vs {len(y)}")
    if len(t) < 3:
        raise ValueError(f"Signal has only {len(t)} samples; need at least 3")
    if not np.all(np.diff(t) >= 0):
        raise ValueError("Time axis is not monotonically non-decreasing; cannot analyze")

    axis_min = float(t[0])
    axis_max = float(t[-1])

    if t_start is not None and t_end is not None and t_start >= t_end:
        raise ValueError(f"t_start ({t_start:.6g}) must be less than t_end ({t_end:.6g})")
    if t_start is not None and (t_start < axis_min or t_start > axis_max):
        raise ValueError(
            f"t_start={t_start:.6g} is outside axis range [{axis_min:.6g}, {axis_max:.6g}]"
        )
    if t_end is not None and (t_end < axis_min or t_end > axis_max):
        raise ValueError(
            f"t_end={t_end:.6g} is outside axis range [{axis_min:.6g}, {axis_max:.6g}]"
        )

    i0 = 0 if t_start is None else int(np.searchsorted(t, t_start, side="left"))
    i1 = len(t) if t_end is None else int(np.searchsorted(t, t_end, side="right"))

    t_win = t[i0:i1]
    y_win = y[i0:i1]

    mask = np.isfinite(t_win) & np.isfinite(y_win)
    dropped = int(len(y_win) - int(mask.sum()))
    t_clean = t_win[mask]
    y_clean = y_win[mask]

    if len(t_clean) < 3:
        raise ValueError(
            f"Window [{t_start}, {t_end}] has {len(t_clean)} valid samples "
            f"after cleaning; need at least 3. Axis range: "
            f"[{axis_min:.6g}, {axis_max:.6g}]"
        )

    return t_clean, y_clean, dropped


def _interp_crossings(
    t: np.ndarray, y: np.ndarray, threshold: float, direction: str
) -> list[float]:
    """Sub-sample-accurate threshold crossings via linear interpolation.

    ``direction`` is ``"rising"``, ``"falling"``, or ``"any"``. Returns crossing
    times in increasing order. Handles sample values that exactly equal the
    threshold by treating them as crossings only if the next sample is on the
    opposite side.
    """
    if direction not in ("rising", "falling", "any"):
        raise ValueError(f"direction must be 'rising', 'falling', or 'any', got {direction!r}")
    d = y - threshold
    # Strict sign change between consecutive samples
    sign_change = (d[:-1] * d[1:]) < 0
    # A sample sitting exactly on threshold "exits" on the next sample
    touch_exit = (d[:-1] == 0) & (d[1:] != 0)
    idx = np.where(sign_change | touch_exit)[0]
    if len(idx) == 0:
        return []

    y0 = y[idx]
    y1 = y[idx + 1]
    t0 = t[idx]
    t1 = t[idx + 1]
    denom = y1 - y0
    # frac = 0 where denom == 0 (can only happen for touch_exit with y0==y1==threshold,
    # which we filtered above; guard anyway)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom != 0, (threshold - y0) / denom, 0.0)
    tc = t0 + frac * (t1 - t0)

    rising_mask = y1 > y0
    if direction == "rising":
        tc = tc[rising_mask]
    elif direction == "falling":
        tc = tc[~rising_mask]

    return [float(x) for x in tc]


def _tail_windows(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (first 10% slice, last 10% slice) of a 1-D signal."""
    tail_n = max(1, len(y) // 10)
    return y[:tail_n], y[-tail_n:]


def _estimate_levels(y: np.ndarray) -> tuple[float, float]:
    """Return (start_level, end_level) averaged over first/last 10% of samples.

    Using the ends rather than global min/max makes the estimate resistant to
    overshoot/undershoot ringing in the transition region.
    """
    head, tail = _tail_windows(y)
    return float(np.mean(head)), float(np.mean(tail))


def _level_stability(y: np.ndarray) -> tuple[float, float]:
    """Stddev of the first/last 10% of samples. High stddev signals that
    the auto-level estimate is unreliable (window straddles an edge, or
    response hasn't settled).
    """
    head, tail = _tail_windows(y)
    return float(np.std(head)), float(np.std(tail))


# Auto-level estimate is rejected when leading/trailing 10% stddev exceeds
# this fraction of |final - initial|. 0.10 = 10% — generous enough to allow
# small ripple, tight enough to catch a window straddling the edge.
_AUTO_LEVEL_VARIANCE_THRESHOLD = 0.10


def analyze_edge(
    t: np.ndarray,
    y: np.ndarray,
    *,
    edge: str = "auto",
    edge_index: int = 0,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    low_level: float | None = None,
    high_level: float | None = None,
) -> EdgeMetricsOutput:
    """Compute transition time (rise or fall) and slew rate for one edge.

    Levels are estimated from the first/last 10% of the window (NOT global
    min/max — this keeps overshoot/undershoot out of the level estimate).
    Crossings are sub-sample-accurate via linear interpolation.

    ``low_level`` / ``high_level`` override the auto-detected rail levels with
    absolute values. Use them when the auto estimate is biased — e.g. a
    rise-from-rail where samples cluster in the fast early ramp pull the
    first-10% mean off zero (V7-P2-1). Direction is still inferred from the
    signal, not the overrides.
    """
    if len(t) < 3:
        raise ValueError("Need at least 3 samples to analyze an edge")
    if not (0 <= low_pct < high_pct <= 100):
        raise ValueError(
            f"Thresholds must satisfy 0 <= low_pct < high_pct <= 100, "
            f"got low_pct={low_pct}, high_pct={high_pct}"
        )
    if edge not in ("auto", "rising", "falling"):
        raise ValueError(f"edge must be 'auto', 'rising', or 'falling', got {edge!r}")

    warnings: list[str] = []
    start_level, end_level = _estimate_levels(y)

    if abs(end_level - start_level) < _LEVEL_EPSILON:
        raise ValueError(
            f"No edge detected — signal level is constant "
            f"(start={start_level:.6g}, end={end_level:.6g}). "
            "Widen the window or check the signal."
        )

    detected_direction = "rising" if end_level > start_level else "falling"
    resolved_low = min(start_level, end_level) if low_level is None else float(low_level)
    resolved_high = max(start_level, end_level) if high_level is None else float(high_level)
    if resolved_high <= resolved_low:
        raise ValueError(
            f"high_level ({resolved_high:.6g}) must exceed low_level ({resolved_low:.6g})"
        )

    # Auto-level bias advisory (mirrors analyze_pulse_response's per-rail
    # stability gate). A leading/trailing 10% window that straddles the
    # transition — e.g. an edge sitting at a window edge — biases the auto
    # rail estimate it feeds; that shifts low_thresh/high_thresh and so
    # propagates into transition_time/slew_rate, not just the reported levels.
    # Each rail is overridable independently, so the advisory is gated
    # per-rail: warn only when the rail a noisy window feeds is still
    # auto-detected. The window→rail mapping flips with direction — on a
    # rising edge the start window sets the low rail and the end window the
    # high rail; on a falling edge it's reversed. A combined "both auto" gate
    # would silently skip one-sided overrides.
    abs_delta = abs(end_level - start_level)
    start_std, end_std = _level_stability(y)
    threshold = _AUTO_LEVEL_VARIANCE_THRESHOLD * abs_delta
    start_rail_auto = (low_level if detected_direction == "rising" else high_level) is None
    end_rail_auto = (high_level if detected_direction == "rising" else low_level) is None
    if start_std > threshold and start_rail_auto:
        warnings.append(
            f"Leading-10% stddev ({start_std:.3g}) high vs |end-start| "
            f"({abs_delta:.3g}); the auto rail it sets may be biased (edge near "
            "window start?) — transition_time/slew inherit it. Pass an explicit "
            "level for that rail to suppress."
        )
    if end_std > threshold and end_rail_auto:
        warnings.append(
            f"Trailing-10% stddev ({end_std:.3g}) high vs |end-start| "
            f"({abs_delta:.3g}); the auto rail it sets may be biased (edge near "
            "window end?) — transition_time/slew inherit it. Pass an explicit "
            "level for that rail to suppress."
        )

    if edge == "auto":
        direction = detected_direction
    else:
        direction = edge
        if direction != detected_direction:
            warnings.append(
                f"Requested {edge} edge but window shows {detected_direction} "
                f"transition (start={start_level:.6g}, end={end_level:.6g})"
            )

    level_range = resolved_high - resolved_low
    mid_level = resolved_low + 0.5 * level_range
    low_thresh = resolved_low + (low_pct / 100.0) * level_range
    high_thresh = resolved_low + (high_pct / 100.0) * level_range

    mid_crossings = _interp_crossings(t, y, mid_level, direction=direction)
    if not mid_crossings:
        raise ValueError(f"No {direction} edge found in window at mid-level {mid_level:.6g}")
    if edge_index < 0 or edge_index >= len(mid_crossings):
        raise ValueError(
            f"edge_index={edge_index} out of range; found {len(mid_crossings)} {direction} edge(s)"
        )
    if len(mid_crossings) > 1:
        warnings.append(
            f"Found {len(mid_crossings)} {direction} edges in window; "
            f"using edge_index={edge_index} at t={mid_crossings[edge_index]:.6g}"
        )

    chosen_mid_t = mid_crossings[edge_index]

    low_crossings = _interp_crossings(t, y, low_thresh, direction=direction)
    high_crossings = _interp_crossings(t, y, high_thresh, direction=direction)
    if not low_crossings or not high_crossings:
        raise ValueError(
            f"Could not find {low_pct}% and {high_pct}% crossings near edge at "
            f"t={chosen_mid_t:.6g} (low_thresh={low_thresh:.6g}, "
            f"high_thresh={high_thresh:.6g})"
        )

    t_low = min(low_crossings, key=lambda x: abs(x - chosen_mid_t))
    t_high = min(high_crossings, key=lambda x: abs(x - chosen_mid_t))
    transition_time = abs(t_high - t_low)

    if transition_time < 1e-18:
        raise ValueError(
            f"Transition time effectively zero ({transition_time:.3e}); "
            "check sample density near edge"
        )

    # Slew rate uses the threshold-to-threshold voltage delta (not the full
    # level range) — standard 10-90 slew rate is ΔV_80% / Δt_10-90%.
    threshold_delta = high_thresh - low_thresh
    slew_rate = threshold_delta / transition_time

    return {
        "transition_time": float(transition_time),
        "slew_rate": float(slew_rate),
        "low_level": float(resolved_low),
        "high_level": float(resolved_high),
        "t_low_crossing": float(t_low),
        "t_high_crossing": float(t_high),
        "t_mid_crossing": float(chosen_mid_t),
        "edge_direction": direction,
        "is_rise_time": direction == "rising",
        "low_pct": float(low_pct),
        "high_pct": float(high_pct),
        "num_edges_in_window": len(mid_crossings),
        "warnings": warnings,
    }


def analyze_pulse_response(
    t: np.ndarray,
    y: np.ndarray,
    *,
    initial_value: float | None = None,
    final_value: float | None = None,
    settling_tolerance_pct: float = 2.0,
) -> PulseResponseOutput:
    """Compute overshoot, undershoot, settling time for a step response.

    - ``initial_value`` / ``final_value`` default to mean of first/last 10% of window.
    - Overshoot = max excursion beyond ``final_value`` in the step direction.
    - Undershoot = max excursion beyond ``initial_value`` opposite the step direction.
    - ``overshoot_pct = 0`` means overdamped (measured, not unknown).
    - ``settling_time = None`` means never settled within the window.
    """
    if len(t) < 3:
        raise ValueError("Need at least 3 samples")
    if settling_tolerance_pct <= 0:
        raise ValueError(f"settling_tolerance_pct must be positive, got {settling_tolerance_pct}")

    warnings: list[str] = []
    start_level, end_level = _estimate_levels(y)
    iv = float(initial_value) if initial_value is not None else start_level
    fv = float(final_value) if final_value is not None else end_level

    delta = fv - iv
    if abs(delta) < _LEVEL_EPSILON:
        raise ValueError(
            f"No step detected: |final - initial| = {abs(delta):.3e} "
            f"(initial={iv:.6g}, final={fv:.6g}). Widen the window or pass "
            "explicit initial_value/final_value."
        )

    # Auto-level estimate gating (Fr4): only HARD-fail when both ends are
    # noisy — that means the user picked a window with no quiet region and
    # we genuinely can't bootstrap. When only ONE end is noisy, fall back
    # to the boundary sample on that side (y[0] or y[-1]) and surface a
    # warning. The other end's stable mean is still trusted.
    abs_delta = abs(delta)
    start_std, end_std = _level_stability(y)
    threshold = _AUTO_LEVEL_VARIANCE_THRESHOLD * abs_delta
    start_noisy = initial_value is None and start_std > threshold
    end_noisy = final_value is None and end_std > threshold
    if start_noisy and end_noisy:
        raise ValueError(
            f"Auto-detected levels are unreliable on both ends of the window "
            f"(leading stddev {start_std:.3g}, trailing stddev {end_std:.3g}; "
            f"|final - initial| {abs_delta:.3g}). Pass explicit "
            f"initial_value AND final_value, or pick a window that includes "
            f"a quiet region on either the pre-step or post-step side."
        )
    if start_noisy:
        iv = float(y[0])
        delta = fv - iv
        abs_delta = abs(delta)
        warnings.append(
            f"Leading-10% stddev ({start_std:.3g}) high vs |final-initial| "
            f"({abs_delta:.3g}); using y[0]={iv:.6g} as initial_value. "
            "Pass an explicit initial_value to suppress."
        )
    if end_noisy:
        fv = float(y[-1])
        delta = fv - iv
        abs_delta = abs(delta)
        warnings.append(
            f"Trailing-10% stddev ({end_std:.3g}) high vs |final-initial| "
            f"({abs_delta:.3g}); using y[-1]={fv:.6g} as final_value. "
            "Response may not be fully settled."
        )
    if abs_delta < _LEVEL_EPSILON:
        raise ValueError(
            f"After fallback, |final - initial| collapsed to {abs_delta:.3e}; "
            "widen the window or pass explicit initial_value/final_value."
        )

    # Fr4: after the auto-level logic settles, refuse windows that contain
    # a full transition + return (e.g. a PULSE with rising AND falling
    # edges inside the window). The y peak-to-peak swing dwarfs the
    # initial→final delta, so overshoot_pct would explode (peak / |tiny
    # delta| → millions of percent). Skip on explicit-level path so the
    # caller can deliberately work near zero.
    if initial_value is None and final_value is None:
        y_pk_pk = float(np.max(y) - np.min(y))
        if y_pk_pk > _LEVEL_EPSILON and abs_delta < 0.1 * y_pk_pk:
            raise ValueError(
                f"Window contains a peak-to-peak swing of {y_pk_pk:.3g} but "
                f"|final - initial| is only {abs_delta:.3g} — the window "
                f"appears to capture a full pulse (rise AND fall) rather "
                f"than a single monotonic step. Narrow t_start/t_end to one "
                f"transition, or pass explicit initial_value/final_value if "
                f"this is intentional."
            )

    direction = "rising" if delta > 0 else "falling"

    if direction == "rising":
        overshoot_signal = y - fv
        undershoot_signal = iv - y
    else:
        overshoot_signal = fv - y
        undershoot_signal = y - iv

    over_peaks, _ = find_peaks(overshoot_signal)
    if len(over_peaks) > 0 and overshoot_signal[over_peaks[0]] > 0:
        peak_idx = int(over_peaks[0])
        overshoot_pct = float(overshoot_signal[peak_idx] / abs_delta * 100.0)
    else:
        peak_idx = int(np.argmax(y) if direction == "rising" else np.argmin(y))
        overshoot_pct = 0.0

    peak_value = float(y[peak_idx])
    peak_time = float(t[peak_idx])

    under_peaks, _ = find_peaks(undershoot_signal)
    if len(under_peaks) > 0 and undershoot_signal[under_peaks[0]] > 0:
        undershoot_pct = float(undershoot_signal[under_peaks[0]] / abs_delta * 100.0)
    else:
        undershoot_pct = 0.0

    tol = (settling_tolerance_pct / 100.0) * abs_delta
    outside = np.abs(y - fv) > tol
    outside_idx = np.where(outside)[0]
    settling_time: float | None
    if len(outside_idx) == 0:
        settling_time = 0.0
        warnings.append(
            f"Signal is already within ±{settling_tolerance_pct}% tolerance at "
            "window start; settling_time=0 (window may start after settling)"
        )
    elif outside_idx[-1] == len(y) - 1:
        settling_time = None
        warnings.append(
            f"Signal did not settle within ±{settling_tolerance_pct}% tolerance by end of window"
        )
    else:
        last_outside = int(outside_idx[-1])
        settling_time = float(t[last_outside + 1] - t[0])

    return {
        "direction": direction,
        "initial_value": iv,
        "steady_state_value": fv,
        "peak_value": peak_value,
        "peak_time": peak_time,
        "overshoot_pct": float(overshoot_pct),
        "undershoot_pct": float(undershoot_pct),
        "settling_time": settling_time,
        "settling_tolerance_pct": float(settling_tolerance_pct),
        "warnings": warnings,
    }


def analyze_timing_between(
    t: np.ndarray,
    ya: np.ndarray,
    yb: np.ndarray,
    *,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    threshold_pct: float = 50.0,
    direction_a: str = "rising",
    direction_b: str = "rising",
) -> TimingBetweenOutput:
    """Time delay between first threshold crossings of two signals on a shared axis.

    Thresholds are per-signal (default 50% of each signal's own range in the
    window) — intentional for asymmetric CMOS where V_in and V_out have
    different rails.
    """
    if len(ya) != len(yb) or len(ya) != len(t):
        raise ValueError(
            f"All three arrays must have equal length; got "
            f"len(t)={len(t)}, len(ya)={len(ya)}, len(yb)={len(yb)}"
        )
    if len(t) < 3:
        raise ValueError("Need at least 3 samples")
    if direction_a not in ("rising", "falling"):
        raise ValueError(f"direction_a must be 'rising' or 'falling', got {direction_a!r}")
    if direction_b not in ("rising", "falling"):
        raise ValueError(f"direction_b must be 'rising' or 'falling', got {direction_b!r}")
    if not (0 <= threshold_pct <= 100):
        raise ValueError(f"threshold_pct must be in [0, 100], got {threshold_pct}")

    def _auto_threshold(y: np.ndarray, name: str) -> float:
        lo = float(np.min(y))
        hi = float(np.max(y))
        if abs(hi - lo) < _LEVEL_EPSILON:
            raise ValueError(
                f"Signal {name} is constant in window (min={lo:.6g}, max={hi:.6g}); "
                "cannot place automatic threshold. Provide an explicit threshold."
            )
        return lo + (threshold_pct / 100.0) * (hi - lo)

    thresh_a = float(threshold_a) if threshold_a is not None else _auto_threshold(ya, "a")
    thresh_b = float(threshold_b) if threshold_b is not None else _auto_threshold(yb, "b")

    crossings_a = _interp_crossings(t, ya, thresh_a, direction=direction_a)
    crossings_b = _interp_crossings(t, yb, thresh_b, direction=direction_b)

    if not crossings_a:
        raise ValueError(f"No {direction_a} crossing of signal_a at threshold {thresh_a:.6g}")
    if not crossings_b:
        raise ValueError(f"No {direction_b} crossing of signal_b at threshold {thresh_b:.6g}")

    t_a = crossings_a[0]
    t_b = crossings_b[0]

    warnings: list[str] = []
    if len(crossings_a) > 1:
        warnings.append(
            f"signal_a has {len(crossings_a)} {direction_a} crossings in window; using first"
        )
    if len(crossings_b) > 1:
        warnings.append(
            f"signal_b has {len(crossings_b)} {direction_b} crossings in window; using first"
        )

    return {
        "t_a": float(t_a),
        "t_b": float(t_b),
        "delay": float(t_b - t_a),
        "threshold_a_used": float(thresh_a),
        "threshold_b_used": float(thresh_b),
        "direction_a": direction_a,
        "direction_b": direction_b,
        "num_crossings_a": len(crossings_a),
        "num_crossings_b": len(crossings_b),
        "warnings": warnings,
    }


def analyze_periodic(
    t: np.ndarray,
    y: np.ndarray,
    *,
    threshold: float | None = None,
    min_periods: int = 2,
) -> PeriodicMetricsOutput:
    """Period, frequency, duty cycle, jitter for an oscillating signal.

    Uses threshold crossings (default = midpoint of window min/max). For
    drifting or DC-offset signals, set an explicit threshold — the auto
    midpoint moves with the drift.
    """
    if len(t) < 3:
        raise ValueError("Need at least 3 samples")
    if min_periods < 1:
        raise ValueError(f"min_periods must be >= 1, got {min_periods}")

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    if abs(y_max - y_min) < _LEVEL_EPSILON:
        raise ValueError(
            f"Signal is constant in window (min={y_min:.6g}, max={y_max:.6g}); "
            "no periodic behavior to analyze"
        )

    thresh = (y_min + y_max) / 2.0 if threshold is None else float(threshold)
    if thresh <= y_min or thresh >= y_max:
        raise ValueError(
            f"threshold={thresh:.6g} must be strictly between signal min "
            f"({y_min:.6g}) and max ({y_max:.6g})"
        )

    rising = _interp_crossings(t, y, thresh, direction="rising")
    falling = _interp_crossings(t, y, thresh, direction="falling")

    warnings: list[str] = []
    if len(rising) < min_periods + 1:
        raise ValueError(
            f"Only {len(rising)} rising edge(s) found; need at least "
            f"{min_periods + 1} for {min_periods} period(s). Widen the window "
            f"or adjust threshold (currently {thresh:.6g})."
        )

    rising_arr = np.asarray(rising)
    periods = np.diff(rising_arr)
    period_mean = float(np.mean(periods))
    period_std = float(np.std(periods, ddof=0)) if len(periods) > 1 else 0.0
    frequency = 1.0 / period_mean if period_mean > 0 else float("nan")

    duty_cycles: list[float] = []
    high_widths: list[float] = []
    low_widths: list[float] = []
    if falling:
        falling_arr = np.asarray(falling)
        for i in range(len(rising_arr) - 1):
            t_rise = rising_arr[i]
            t_next_rise = rising_arr[i + 1]
            mask = (falling_arr > t_rise) & (falling_arr < t_next_rise)
            between = falling_arr[mask]
            if len(between) == 0:
                continue
            if len(between) > 1:
                # Multiple falling edges per period = non-monotonic within period
                # (e.g. ringy square wave). Use the first.
                pass
            t_fall = float(between[0])
            high = t_fall - t_rise
            low = t_next_rise - t_fall
            period = t_next_rise - t_rise
            if period > 0:
                duty_cycles.append(high / period)
                high_widths.append(high)
                low_widths.append(low)
    else:
        warnings.append("No falling edges found at threshold; duty cycle undefined")

    if duty_cycles:
        duty_pct: float | None = float(np.mean(duty_cycles)) * 100.0
        high_width_mean: float | None = float(np.mean(high_widths))
        low_width_mean: float | None = float(np.mean(low_widths))
    else:
        duty_pct = None
        high_width_mean = None
        low_width_mean = None
        if falling:
            warnings.append(
                "Could not pair rising/falling edges into full periods; duty cycle undefined"
            )

    return {
        "period": period_mean,
        "frequency": frequency,
        "jitter_rms": period_std,
        "duty_cycle_pct": duty_pct,
        "pulse_width_high": high_width_mean,
        "pulse_width_low": low_width_mean,
        "num_rising_edges": len(rising),
        "num_falling_edges": len(falling),
        "num_periods_measured": len(periods),
        "threshold_used": float(thresh),
        "warnings": warnings,
    }


def compute_signal_stats(
    t: np.ndarray,
    y: np.ndarray,
) -> SignalStatsOutput:
    """Time-weighted signal statistics over the window [t[0], t[-1]].

    Uses trapezoidal integration so non-uniform sample spacing is handled
    correctly (LTspice adaptively varies dt). For uniform axes the results
    match ``np.mean`` / ``sqrt(np.mean(y**2))``.

    Returns time-weighted ``mean``, ``rms``, ``std``, and ``abs_mean`` plus
    sample-space ``min`` / ``max`` / ``pk_pk`` and window metadata. If the
    window collapses to a single instant (``duration == 0``), falls back to
    unweighted statistics over the raw samples.
    """
    if len(t) != len(y):
        raise ValueError(f"Axis and wave have different lengths: {len(t)} vs {len(y)}")
    if len(t) < 1:
        raise ValueError("Signal has no samples")

    t_start = float(t[0])
    t_end = float(t[-1])
    duration = t_end - t_start

    y_min = float(np.min(y))
    y_max = float(np.max(y))

    if duration > 0 and len(t) >= 2:
        mean = float(np.trapezoid(y, t) / duration)
        mean_sq = float(np.trapezoid(y * y, t) / duration)
        abs_mean = float(np.trapezoid(np.abs(y), t) / duration)
        variance = max(mean_sq - mean * mean, 0.0)
        std = float(np.sqrt(variance))
        rms = float(np.sqrt(mean_sq))
    else:
        mean = float(np.mean(y))
        rms = float(np.sqrt(np.mean(y * y)))
        abs_mean = float(np.mean(np.abs(y)))
        std = float(np.std(y, ddof=0))

    return {
        "t_start": t_start,
        "t_end": t_end,
        "duration": float(duration),
        "num_samples": len(t),
        "mean": mean,
        "rms": rms,
        "std": std,
        "abs_mean": abs_mean,
        "min": y_min,
        "max": y_max,
        "pk_pk": y_max - y_min,
    }


def compute_measurement_stats(
    measurements: Mapping[str, Sequence[float | None]],
    *,
    histogram_bins: int = 10,
    measurement: str | None = None,
) -> dict[str, MeasurementStatsEntry]:
    """Aggregate stats across all steps of a ``.MEAS`` result dict.

    ``None`` entries (failed measurements) are counted but excluded from stats.
    If ``measurement`` is given, stats are computed for only that one.
    """
    if histogram_bins < 0:
        raise ValueError(f"histogram_bins must be >= 0, got {histogram_bins}")

    names = [measurement] if measurement is not None else list(measurements.keys())
    if measurement is not None and measurement not in measurements:
        raise ValueError(
            f"Measurement {measurement!r} not found. Available: "
            f"{', '.join(measurements.keys()) or '<none>'}"
        )

    result: dict[str, MeasurementStatsEntry] = {}
    for name in names:
        values = measurements[name]
        total = len(values)
        valid = [v for v in values if v is not None]
        valid_count = len(valid)

        if valid_count == 0:
            result[name] = {
                "total_count": total,
                "valid_count": valid_count,
                "failure_count": total - valid_count,
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "std": None,
                "p10": None,
                "p90": None,
                "best_step_index": None,
                "worst_step_index": None,
                "histogram": [],
            }
            continue

        arr = np.asarray(valid, dtype=float)
        e_min = float(np.min(arr))
        e_max = float(np.max(arr))

        best_step: int | None = None
        worst_step: int | None = None
        best_val: float | None = None
        worst_val: float | None = None
        for i, v in enumerate(values):
            if v is None:
                continue
            if best_val is None or v < best_val:
                best_val = v
                best_step = i
            if worst_val is None or v > worst_val:
                worst_val = v
                worst_step = i

        histogram: list[HistogramBin] = []
        if histogram_bins > 0 and valid_count >= 2 and e_min < e_max:
            counts, edges = np.histogram(arr, bins=histogram_bins)
            histogram = [
                {
                    "bin_start": float(edges[i]),
                    "bin_end": float(edges[i + 1]),
                    "count": int(counts[i]),
                }
                for i in range(len(counts))
            ]

        result[name] = {
            "total_count": total,
            "valid_count": valid_count,
            "failure_count": total - valid_count,
            "min": e_min,
            "max": e_max,
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "std": float(np.std(arr, ddof=0)),
            "p10": float(np.percentile(arr, 10)),
            "p90": float(np.percentile(arr, 90)),
            "best_step_index": best_step,
            "worst_step_index": worst_step,
            "histogram": histogram,
        }

    return result
