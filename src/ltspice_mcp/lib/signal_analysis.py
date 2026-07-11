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
from typing import Literal, NotRequired, TypedDict

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
    """Return shape of :func:`analyze_pulse_response`.

    ``overshoot_pct``/``undershoot_pct``/``settling_time`` are ``None`` when the
    window is degenerate enough that they are *undefined* (when the net step is
    tiny next to the window's swing, they divide by a near-zero baseline) — a
    machine-readable "not available" rather than an authoritative-looking nonsense
    number.

    ``quality`` carries machine-readable codes a consumer can gate on without
    parsing the free-text ``warnings`` (e.g. ``net_step_small_vs_swing``,
    ``levels_bootstrapped_from_boundary``). Each code names an *input condition or
    computation-provenance fact*, never a result verdict — see the repo-wide
    "Result-trust: surface, don't judge" in CLAUDE.md.
    """

    direction: CrossingDirection
    initial_value: float
    steady_state_value: float
    peak_value: float
    peak_time: float
    overshoot_pct: float | None
    undershoot_pct: float | None
    settling_time: float | None
    settling_tolerance_pct: float
    quality: list[str]
    warnings: list[str]


class DisturbanceResponseOutput(TypedDict):
    """Return shape of :func:`analyze_disturbance_response`.

    Measures a regulated output's excursion from — and recovery to — its own
    pre-disturbance baseline (LDO/PMIC load transient). Complements
    :func:`analyze_pulse_response`, which correctly nulls its step metrics when
    the signal returns to its starting level.

    ``min_value``/``max_value`` are the raw worst-case samples over the window;
    ``max_droop``/``max_overshoot`` are the excursions below/above ``baseline``,
    clamped to ``>= 0`` (an LDO/PMIC positive-rail convention). ``recovery_time``
    is ``None`` when the signal never re-enters the settle band before the window
    ends, or when the band is undefined (see ``quality``).
    """

    baseline: float
    baseline_source: Literal["explicit", "auto_leading_window"]
    min_value: float
    min_time: float
    max_value: float
    max_time: float
    max_droop: float
    max_overshoot: float
    recovery_time: float | None
    settle_band: float
    settle_band_pct: float | None
    quality: list[str]
    warnings: list[str]


class TimingBetweenOutput(TypedDict):
    """Return shape of :func:`analyze_timing_between`.

    ``t_a``/``t_b``/``delay`` describe the FIRST crossing of each signal
    (kept for the one-shot step/propagation case, and the only pairing that
    can go negative when b leads a). The ``pair_*``/``delay_*`` fields
    aggregate over ALL sequential edge pairs — what dead-time / minimum
    off-time audits over a pulse train need.
    """

    t_a: float
    t_b: float
    delay: float
    threshold_a_used: float
    threshold_b_used: float
    direction_a: CrossingDirection
    direction_b: CrossingDirection
    num_crossings_a: int
    num_crossings_b: int
    pair_count: int
    delay_min: float | None
    delay_max: float | None
    delay_mean: float | None
    delay_min_at: float | None
    delay_max_at: float | None
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
    t_at_min: float
    t_at_max: float


class WaveformBucket(TypedDict):
    """One equal-time bucket of a decimated waveform envelope."""

    x_start: float
    x_end: float
    min: float
    max: float
    mean: float
    rms: float
    pk_pk: float
    crest_factor: float | None
    num_samples: int


class StatEnvelopeOutput(TypedDict):
    """Return shape of :func:`stat_envelope`."""

    buckets: list[WaveformBucket]
    point_count: int
    bucket_count: int
    x_start: float
    x_end: float
    decimated: bool


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
    min_step_index: int | None
    max_step_index: int | None
    histogram: list[HistogramBin]


def window_and_clean(
    t: np.ndarray,
    y: np.ndarray,
    t_start: float | None,
    t_end: float | None,
    *,
    allow_descending: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Slice to ``[t_start, t_end]`` and strip non-finite samples.

    Returns ``(t_clean, y_clean, dropped_nonfinite)``. Requires at least 3
    samples after cleaning.

    ``allow_descending`` is for sweep axes (DC/noise) where a strictly
    decreasing axis is a legitimate ordering (e.g. ``.dc V1 5 0 -0.1``), not
    corruption: the axis and wave are flipped to ascending together so the
    windowing below stays correct. It stays off for time/frequency axes, where
    a non-monotonic axis means a corrupt result and must be refused.
    """
    if len(t) != len(y):
        raise ValueError(f"Axis and wave have different lengths: {len(t)} vs {len(y)}")
    if len(t) < 3:
        raise ValueError(f"Signal has only {len(t)} samples; need at least 3")
    dt = np.diff(t)
    if not np.all(dt >= 0):
        if allow_descending and np.all(dt <= 0):
            t = t[::-1]
            y = y[::-1]
        else:
            raise ValueError("Time axis is not monotonically non-decreasing; cannot analyze")

    axis_min = float(t[0])
    axis_max = float(t[-1])

    if t_start is not None and t_end is not None and t_start >= t_end:
        raise ValueError(f"t_start ({t_start:.6g}) must be less than t_end ({t_end:.6g})")

    # SI-suffix parsing can land a bound a few ulps past the axis end ('1500u'
    # parses to 1.5000000000000002e-3 against an axis ending at 1.5e-3); a
    # strict check then rejects it with a self-contradicting "t_end=0.0015 is
    # outside axis range [0, 0.0015]". Clamp near-miss bounds instead.
    tol = 1e-9 * max(abs(axis_min), abs(axis_max), axis_max - axis_min)

    def _clamp_bound(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        if axis_min - tol <= value <= axis_max + tol:
            return min(max(value, axis_min), axis_max)
        raise ValueError(
            f"{name}={value:.6g} is outside axis range [{axis_min:.6g}, {axis_max:.6g}]"
        )

    t_start = _clamp_bound(t_start, "t_start")
    t_end = _clamp_bound(t_end, "t_end")

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


# Level-stability gate: a leading/trailing-window stddev above this fraction
# of |final - initial| means the window straddles the transition, so the auto
# rail it bootstraps is unreliable. 0.10 = 10% — generous enough to allow small
# ripple, tight enough to catch a window straddling the edge. Shared by the edge
# bias advisory (analyze_edge) and the pulse-response bootstrap gate
# (analyze_pulse_response); both feed it the same _level_stability() stddev.
# This is the STDDEV-vs-step gate — kept distinct from _FULL_PULSE_DELTA_FRACTION
# below (step-vs-swing), so tuning one can't silently move the other even though
# they currently share the 10% value.
_AUTO_LEVEL_VARIANCE_THRESHOLD = 0.10

# Full-pulse reject: on the auto-level path, if the net |final - initial| step
# is smaller than this fraction of the window's peak-to-peak swing, the window
# almost certainly captured a full pulse (rise AND fall) rather than one
# monotonic edge — overshoot_pct would explode against the tiny net delta.
# Same 10% magnitude as the stddev gate above but a SEPARATE meaning; named
# separately so the two gates stay independently tunable.
_FULL_PULSE_DELTA_FRACTION = 0.10

# Minimum in-band dwell before the window end, in units of the NOMINAL
# trailing-window length (10% of the analyzed time span), for an auto-derived
# final value to support a settling_time. A tail can be perfectly flat (passing
# the noise gate) yet be nothing more than the last plateau of a still-ringing
# waveform; when the settle-band entry sits that close to the window end, the
# number is unsupported. The yardstick is a time fraction, NOT the
# last-10%-of-samples slice `_tail_windows` uses for level statistics: adaptive
# .tran stepping makes settled flat tails sparse, so the sample slice's time
# span can swell toward the whole window (which would suppress every genuine
# settle) or collapse to zero on short runs (which would never suppress). A
# time fraction is invariant to sample density. Measured bounds on uniform
# fixtures (where the two yardsticks coincide): a ringing staircase paused on
# its final plateau reaches ~1.25 trailing windows of dwell, while a genuine
# auto-settle sits at ~1.8; 1.5 splits them.
_SETTLE_MIN_DWELL_TAILS = 1.5

# The nominal trailing-window fraction of the analyzed time span — the
# time-domain twin of `_tail_windows`' last-10%-of-samples slice.
_SETTLE_DWELL_TAIL_FRACTION = 0.10


def _is_full_pulse(pk_pk: float, abs_delta: float) -> bool:
    """True if the net step is tiny vs the window's peak-to-peak swing — i.e. the
    window captured a full pulse (rise AND fall), not one monotonic edge, so its
    endpoint-derived direction/levels are meaningless."""
    return pk_pk > _LEVEL_EPSILON and abs_delta < _FULL_PULSE_DELTA_FRACTION * pk_pk


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
    first-10% mean off zero. Direction is still inferred from the
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
    #
    # Severity is intentionally asymmetric vs analyze_pulse_response, which
    # hard-fails on the same gate: a biased rail only shifts transition_time/slew
    # slightly here (still a usable number), whereas a biased pulse baseline makes
    # overshoot_pct explode against a tiny net delta — garbage, not a caveat. So
    # the edge path warns-and-continues; the pulse path raises.
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
        # The endpoint-derived direction is only meaningful for a single
        # monotonic edge. When the window captures a full pulse (rise AND fall),
        # the net |end-start| is tiny vs the peak-to-peak swing, start/end land
        # on the same rail, and detected_direction is noise — don't contradict
        # an explicitly requested edge with it. The requested direction is
        # honored below; if that edge truly isn't present, the crossing search
        # raises a clear "No <dir> edge found" error.
        full_pulse = _is_full_pulse(float(np.ptp(y)), abs_delta)
        if direction != detected_direction and not full_pulse:
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


def _largest_positive_peak(signal: np.ndarray) -> int | None:
    """Index of the largest positive local peak of ``signal``, or None.

    The LARGEST positive peak, not the first one find_peaks returns: with
    pre-edge ripple inside the window, the first local peak is the ripple
    (usually non-positive in over/undershoot coordinates), which silently
    reported 0% on a genuinely overshooting edge. find_peaks is still the
    gate — it excludes window endpoints, so a still-rising signal cut
    mid-transition doesn't count as overshoot.
    """
    peaks, _ = find_peaks(signal)
    positive = [int(p) for p in peaks if signal[p] > 0]
    if not positive:
        return None
    return max(positive, key=lambda p: float(signal[p]))


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
    quality: list[str] = []
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

    # Auto-level estimate gating: when a leading/trailing window is too noisy to
    # bootstrap a rail, fall back to the boundary sample (y[0] / y[-1]) on that
    # side and surface a warning rather than refusing. Surfacer, not judger — we
    # return a usable best-effort number plus the caveat and let the caller judge,
    # rather than hard-failing when both ends are noisy (the old behaviour). When
    # both are noisy both fall-backs fire and both warnings are emitted; the
    # genuine "no step exists" guards below still raise, because that's a fact,
    # not a judgement.
    abs_delta = abs(delta)
    start_std, end_std = _level_stability(y)
    threshold = _AUTO_LEVEL_VARIANCE_THRESHOLD * abs_delta
    start_noisy = initial_value is None and start_std > threshold
    end_noisy = final_value is None and end_std > threshold
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
    if start_noisy or end_noisy:
        # A rail was bootstrapped from a boundary sample rather than a stable
        # mean — finite and usable, but rougher. Machine-readable so a consumer
        # can weight it without parsing the warning text.
        quality.append("levels_bootstrapped_from_boundary")
    if abs_delta < _LEVEL_EPSILON:
        raise ValueError(
            f"After fallback, |final - initial| collapsed to {abs_delta:.3e}; "
            "widen the window or pass explicit initial_value/final_value."
        )

    # After the auto-level logic settles, a window that captures a full pulse
    # (rise AND fall) has a peak-to-peak swing that dwarfs the net initial→final
    # delta, so overshoot_pct is computed against a tiny baseline and balloons
    # (peak / |tiny delta| → millions of percent). Surfacer, not judger: surface
    # the condition with its evidence and still return the number, rather than
    # refusing — the caller (often an LLM that recognises an absurd overshoot)
    # judges. Only on the auto-level path; explicit levels mean the caller
    # deliberately chose the baseline.
    if initial_value is None and final_value is None:
        y_pk_pk = float(np.ptp(y))
        if _is_full_pulse(y_pk_pk, abs_delta):
            # The net step is a near-zero baseline, so overshoot/undershoot/
            # settling divide by ~0 and are *undefined*, not merely suspect. The
            # code names the observable condition (net step small vs the window's
            # swing — typically a full pulse), NOT a verdict; the return nulls
            # those metrics rather than emitting an authoritative-looking number.
            quality.append("net_step_small_vs_swing")
            warnings.append(
                f"Window peak-to-peak swing ({y_pk_pk:.3g}) dwarfs the net "
                f"|final - initial| ({abs_delta:.3g}) — the window appears to capture "
                f"a full pulse (rise AND fall), so overshoot/undershoot/settling are "
                f"undefined (returned as null). Narrow t_start/t_end to one transition, "
                f"or pass explicit initial_value/final_value, if this isn't intended."
            )

    direction = "rising" if delta > 0 else "falling"

    if direction == "rising":
        overshoot_signal = y - fv
        undershoot_signal = iv - y
    else:
        overshoot_signal = fv - y
        undershoot_signal = y - iv

    over_idx = _largest_positive_peak(overshoot_signal)
    if over_idx is not None:
        peak_idx = over_idx
        overshoot_pct = float(overshoot_signal[peak_idx] / abs_delta * 100.0)
    else:
        peak_idx = int(np.argmax(y) if direction == "rising" else np.argmin(y))
        overshoot_pct = 0.0

    peak_value = float(y[peak_idx])
    peak_time = float(t[peak_idx])

    under_idx = _largest_positive_peak(undershoot_signal)
    undershoot_pct = (
        float(undershoot_signal[under_idx] / abs_delta * 100.0) if under_idx is not None else 0.0
    )

    tol = (settling_tolerance_pct / 100.0) * abs_delta
    # Shared last-band-exit crossing (interpolated between the last out-of-band
    # sample and the first in-band one, so a coarse adaptive tail doesn't land
    # the time a full timestep late). Same helper the disturbance-recovery time
    # uses, so the edge handling stays single-sourced.
    settling_time, bracket_dt = _band_exit_time(t, y, fv, tol)
    # Deferred until after the suppression gates: a warning describing the
    # interpolation accuracy of a settling_time must not ship when that
    # settling_time is nulled in the output.
    interp_warning: str | None = None
    if settling_time == 0.0 and bracket_dt is None:
        warnings.append(
            f"Signal is already within ±{settling_tolerance_pct}% tolerance at "
            "window start; settling_time=0 (window may start after settling)"
        )
    elif settling_time is None:
        warnings.append(
            f"Signal did not settle within ±{settling_tolerance_pct}% tolerance by end of window"
        )
    elif settling_time > 0 and bracket_dt is not None and bracket_dt > 0.1 * settling_time:
        interp_warning = (
            f"settling_time interpolated across a coarse local timestep "
            f"(Δt≈{bracket_dt / settling_time * 100:.0f}% of the settle time); "
            "accuracy is bounded by the run's resolution near settling"
        )

    # A trailing window too noisy to trust as the settled rail (still ringing, or
    # the window ends mid-transition) means the final value was bootstrapped from a
    # single boundary sample. A settle band anchored to that sample can report a
    # definite-looking settling_time measured against an unknown asymptote — on a
    # ringing tail the band lands on a ripple plateau, not the DC value. Null it
    # rather than emit that false number; overshoot/undershoot stay (bounded error
    # vs an outright wrong "settled at T"), and an explicit final_value — which
    # clears end_noisy — bypasses this to measure against a known asymptote.
    if end_noisy and settling_time is not None:
        settling_time = None
        # Distinct flag so the renderer shows this null as UNKNOWN, not "never
        # settled" — names the condition (final value taken from a noisy tail),
        # not a verdict.
        quality.append("settling_final_value_from_noisy_tail")
        warnings.append(
            "settling_time suppressed: the trailing window is too noisy to establish "
            "the final value (still ringing, or the window ends mid-transition), so a "
            "settle band anchored to it is unreliable. Pass an explicit final_value "
            "to measure settling against a known asymptote."
        )

    # A quiet tail is not enough: a still-ringing waveform paused on its last
    # plateau has a perfectly flat trailing window (end_std ~0) yet enters the
    # settle band only moments before the window ends — and the auto-derived
    # final value comes from that same short tail, so band and "settled" stretch
    # are the same few samples. Require a minimum in-band dwell before the end.
    # Auto-final path only: an explicit final_value pins the asymptote, so a
    # genuinely late settle against it stands. Overshoot/undershoot stay, same
    # policy as the noisy-tail suppression above. settling_time is measured
    # from t[0], so the dwell is the window span minus it.
    metrics_undefined = "net_step_small_vs_swing" in quality
    if final_value is None and settling_time is not None and not metrics_undefined:
        span = float(t[-1] - t[0])
        # Nominal trailing window as a time fraction — sampling-invariant where
        # the last-10%-of-samples slice is not (see _SETTLE_MIN_DWELL_TAILS).
        tail_len = _SETTLE_DWELL_TAIL_FRACTION * span
        dwell = span - settling_time
        if span > 0 and dwell < _SETTLE_MIN_DWELL_TAILS * tail_len:
            settling_time = None
            quality.append("settling_dwell_near_window_end")
            warnings.append(
                f"settling_time suppressed: the signal entered the "
                f"±{settling_tolerance_pct}% settle band only ~{dwell / tail_len:.2g}x "
                "the trailing window before the end, and the auto-derived final "
                "value comes from that same short tail — indistinguishable from a "
                "still-ringing waveform paused on a plateau. Pass an explicit "
                "final_value, tighten t_start around the step edge, or extend "
                "the simulation window."
            )

    # A full-pulse window makes overshoot/undershoot/settling undefined (computed
    # against a ~0 baseline) — return null, not a nonsense magnitude. peak_value /
    # peak_time / levels stay valid (they're real samples). The escape hatch is
    # intact: the flag only fires on the auto-level path, so explicit
    # initial_value/final_value bypasses it and returns real metrics.
    if settling_time is not None and not metrics_undefined and interp_warning:
        warnings.append(interp_warning)
    return {
        "direction": direction,
        "initial_value": iv,
        "steady_state_value": fv,
        "peak_value": peak_value,
        "peak_time": peak_time,
        "overshoot_pct": None if metrics_undefined else float(overshoot_pct),
        "undershoot_pct": None if metrics_undefined else float(undershoot_pct),
        "settling_time": None if metrics_undefined else settling_time,
        "settling_tolerance_pct": float(settling_tolerance_pct),
        "quality": quality,
        "warnings": warnings,
    }


def _band_exit_time(
    t: np.ndarray, y: np.ndarray, center: float, tol: float
) -> tuple[float | None, float | None]:
    """Time (from ``t[0]``) of the LAST exit from the band ``[center ± tol]``.

    Interpolates the crossing between the last out-of-band sample and the first
    in-band one (same scheme as the pulse-response settle-band crossing), so a
    coarse adaptive tail doesn't land the time a full timestep late. Returns
    ``(time_from_t0, bracket_dt)``; ``(0.0, None)`` when the signal never leaves
    the band, ``(None, None)`` when it is still outside at the final sample.
    """
    outside_idx = np.where(np.abs(y - center) > tol)[0]
    if len(outside_idx) == 0:
        return 0.0, None
    if outside_idx[-1] == len(y) - 1:
        return None, None
    k = int(outside_idx[-1])
    tk, tk1 = float(t[k]), float(t[k + 1])
    y0, y1 = float(y[k]), float(y[k + 1])
    boundary = center + tol if y0 > center else center - tol
    frac = (boundary - y0) / (y1 - y0) if y1 != y0 else 1.0
    frac = min(max(frac, 0.0), 1.0)
    bracket_dt = tk1 - tk
    return tk + frac * bracket_dt - float(t[0]), bracket_dt


def analyze_disturbance_response(
    t: np.ndarray,
    y: np.ndarray,
    *,
    baseline: float | None = None,
    settle_band: float | None = None,
    settle_band_pct: float = 2.0,
) -> DisturbanceResponseOutput:
    """Excursion-and-recovery metrics for a regulated output under a load step.

    For an LDO/PMIC-style output that returns to its own level after a load
    transient, :func:`analyze_pulse_response` correctly nulls its step metrics
    (net step ≈ 0). This measures the complementary quantities: worst droop and
    overshoot relative to the pre-disturbance ``baseline``, and the time to
    recover to within a settle band of it.

    ``baseline`` defaults to the mean of the leading 10% of samples (the
    pre-disturbance steady state); pass it explicitly when the window does not
    start in steady state. Recovery is measured from ``t[0]``, so the window
    must begin at or just before the disturbance edge.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 3:
        raise ValueError("disturbance_response needs at least 3 samples in the window.")
    if settle_band is not None and settle_band <= 0:
        raise ValueError("settle_band must be positive.")
    if settle_band is None and settle_band_pct <= 0:
        raise ValueError("settle_band_pct must be positive.")

    quality: list[str] = []
    warnings: list[str] = []

    if baseline is None:
        baseline = _estimate_levels(y)[0]
        baseline_source: Literal["explicit", "auto_leading_window"] = "auto_leading_window"
    else:
        baseline = float(baseline)
        baseline_source = "explicit"

    min_idx = int(np.argmin(y))
    max_idx = int(np.argmax(y))
    min_value = float(y[min_idx])
    max_value = float(y[max_idx])
    # Droop/overshoot are the one-sided excursions from baseline, clamped to
    # >= 0 (a positive-rail LDO/PMIC convention); the raw min_value/max_value
    # carry the unclamped fact so a negative-rail node stays readable.
    max_droop = max(0.0, baseline - min_value)
    max_overshoot = max(0.0, max_value - baseline)

    if settle_band is not None:
        tol = float(settle_band)
        pct_out: float | None = None
    else:
        tol = (settle_band_pct / 100.0) * abs(baseline)
        pct_out = float(settle_band_pct)

    recovery_time: float | None
    if tol <= _LEVEL_EPSILON:
        # baseline ≈ 0 with no absolute band → the relative band collapses to a
        # point. Emit a null over a meaningless "recovered at ~0", naming the
        # input condition rather than inventing a band.
        recovery_time = None
        quality.append("recovery_band_undefined_baseline_zero")
        warnings.append(
            f"baseline≈{baseline:.3g} and no absolute settle_band given, so the "
            f"±{settle_band_pct:g}% relative band is ≈0 and recovery_time is undefined. "
            "Pass an explicit settle_band (in signal units)."
        )
    else:
        recovery_time, bracket_dt = _band_exit_time(t, y, baseline, tol)
        if recovery_time is None:
            warnings.append(
                f"Signal did not return to within ±{tol:.3g} of baseline by the end of "
                "the window; recovery_time is unavailable (extend the simulation window)."
            )
        elif bracket_dt is None:
            # Never left the band — no disturbance in this window. Return the
            # (small, real) droop/overshoot rather than refusing.
            quality.append("no_excursion_beyond_band")
            warnings.append(
                f"Signal stayed within ±{tol:.3g} of baseline for the whole window — no "
                "disturbance detected. Confirm the window spans the load-step edge."
            )
        elif recovery_time > 0 and bracket_dt > 0.1 * recovery_time:
            warnings.append(
                f"recovery_time interpolated across a coarse local timestep "
                f"(Δt≈{bracket_dt / recovery_time * 100:.0f}% of the recovery time); "
                "accuracy is bounded by the run's resolution near recovery."
            )

    return {
        "baseline": float(baseline),
        "baseline_source": baseline_source,
        "min_value": min_value,
        "min_time": float(t[min_idx]),
        "max_value": max_value,
        "max_time": float(t[max_idx]),
        "max_droop": float(max_droop),
        "max_overshoot": float(max_overshoot),
        "recovery_time": recovery_time,
        "settle_band": float(tol),
        "settle_band_pct": pct_out,
        "quality": quality,
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

    # All-edges pairing: each A crossing consumes the first unused B crossing
    # at or after it. A pulse train's dead-time / minimum-off audit needs the
    # min/max over ALL pairs — the first pair alone says nothing about the
    # worst edge.
    pairs: list[tuple[float, float]] = []
    j = 0
    for ta_edge in crossings_a:
        while j < len(crossings_b) and crossings_b[j] < ta_edge:
            j += 1
        if j == len(crossings_b):
            break
        pairs.append((ta_edge, crossings_b[j]))
        j += 1
    delays = [tb_edge - ta_edge for ta_edge, tb_edge in pairs]

    warnings: list[str] = []
    if len(crossings_a) > 1 or len(crossings_b) > 1:
        warnings.append(
            f"signal_a has {len(crossings_a)} {direction_a} and signal_b "
            f"{len(crossings_b)} {direction_b} crossing(s) in window; "
            "t_a/t_b/delay use the first of each — read delay_min/delay_max "
            "for the aggregate over all edge pairs"
        )

    delay_min = delay_max = delay_mean = delay_min_at = delay_max_at = None
    if delays:
        min_i = int(np.argmin(delays))
        max_i = int(np.argmax(delays))
        delay_min = float(delays[min_i])
        delay_max = float(delays[max_i])
        delay_mean = float(np.mean(delays))
        delay_min_at = float(pairs[min_i][0])
        delay_max_at = float(pairs[max_i][0])
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
        "pair_count": len(pairs),
        "delay_min": delay_min,
        "delay_max": delay_max,
        "delay_mean": delay_mean,
        "delay_min_at": delay_min_at,
        "delay_max_at": delay_max_at,
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

    # Ringing or glitches near the threshold add extra crossings per cycle, so
    # the edge count — and thus the reported frequency — can be ~2x the true
    # switching rate. When some periods are far shorter than the mean the spacing
    # is bimodal; surface that rather than auto-correct (the user retargets the
    # threshold or windows past the transient).
    if len(periods) >= 3 and period_mean > 0 and float(np.min(periods)) < 0.5 * period_mean:
        warnings.append(
            f"Uneven edge spacing (shortest period {float(np.min(periods)):.3g}s vs mean "
            f"{period_mean:.3g}s): ringing or glitches near the threshold may be adding "
            "crossings, so frequency/duty may be ~2x the true switching rate. Set an "
            "explicit threshold away from the ring, or window past the startup transient."
        )

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


class HarmonicEntry(TypedDict):
    """One harmonic in :func:`analyze_thd`.

    ``magnitude`` is the harmonic's sinusoid amplitude in the signal's own
    units (2/sum(window)-scaled one-sided spectrum), not a raw FFT bin.
    """

    n: int
    frequency: float
    magnitude: float
    db_rel: float


class ThdOutput(TypedDict):
    """Return shape of :func:`analyze_thd`.

    ``signal`` is not set by :func:`analyze_thd` itself — the tool layer adds it
    before returning — but it is declared here so the published output schema
    covers every key the ``thd`` tool emits.
    """

    signal: NotRequired[str]
    # Native unit of the signal (V, A, …), added by the tool layer. The
    # per-harmonic ``magnitude`` and the fundamental amplitude are in this unit;
    # the ratios/percentages/dB are dimensionless.
    unit: NotRequired[str]
    fundamental_hz: float
    fundamental_source: Literal["given", "detected"]
    thd_ratio: float
    thd_pct: float
    thd_db: float
    thd_n_ratio: float
    thd_n_pct: float
    harmonics: list[HarmonicEntry]
    n_harmonics_used: int
    window: str
    coherent: bool
    n_cycles: float
    n_fft: int
    fs_hz: float
    warnings: list[str]


def _next_pow2(n: int) -> int:
    return 1 << max(1, n - 1).bit_length()


def _detect_fundamental(t: np.ndarray, y: np.ndarray) -> float | None:
    """Fundamental frequency: the largest non-DC bin of a uniform-resample FFT,
    refined to sub-bin accuracy by quadratic interpolation around the peak.

    A bare argmax resolves the fundamental only to ~1/span Hz; the downstream
    coherent window built from it would then be a non-integer number of cycles
    and leak. Parabolic interpolation of the three bins around the peak pins the
    fundamental far more precisely, so the coherent path stays coherent.
    """
    n = _next_pow2(len(t))
    dt = (float(t[-1]) - float(t[0])) / n
    if dt <= 0:
        return None
    tu = np.linspace(float(t[0]), float(t[-1]), n, endpoint=False)
    yu = np.interp(tu, t, y)
    yu = yu - yu.mean()
    # Hann-window the detection FFT: a rectangular window leaks badly on a
    # non-integer-cycle record, biasing the parabolic peak fit; Hann's smooth
    # main lobe makes log-magnitude parabolic interpolation accurate to ~0.1%.
    spec = np.abs(np.fft.rfft(yu * np.hanning(n)))
    if spec.size < 2:
        return None
    freqs = np.fft.rfftfreq(n, d=dt)
    k = int(np.argmax(spec[1:]) + 1)
    bin_hz = float(freqs[1])
    if 1 <= k < spec.size - 1:
        # Quadratic interpolation on log-magnitude (Smith) for the sub-bin peak.
        a, b, c = (float(np.log(spec[k + d] + 1e-300)) for d in (-1, 0, 1))
        denom = a - 2.0 * b + c
        delta = 0.5 * (a - c) / denom if denom != 0 else 0.0
        delta = float(np.clip(delta, -0.5, 0.5))
        return (k + delta) * bin_hz
    return float(freqs[k])


def analyze_thd(
    t: np.ndarray,
    y: np.ndarray,
    *,
    fundamental: float | None = None,
    n_harmonics: int = 7,
    window: Literal["coherent", "hann"] = "coherent",
    max_fft: int = 1 << 18,
) -> ThdOutput:
    """Total harmonic distortion of a periodic transient signal via FFT.

    Defaults to COHERENT sampling: the record is trimmed to an integer number of
    fundamental periods and a rectangular window is used, so every harmonic lands
    exactly on an FFT bin — no spectral leakage, THD is exact. ``window='hann'``
    instead analyzes the full window with a Hann taper (use when the fundamental
    can't fit an integer number of cycles); the result is then approximate and a
    leakage warning is emitted.

    Every condition the number depends on is surfaced: the fundamental (and
    whether it was given or detected), the window kind, whether sampling was
    coherent, the cycles analyzed, the FFT length and sample rate, and the
    per-harmonic levels. THD = sqrt(Σ harmonic²) / fundamental; THD+N folds in
    all non-fundamental energy (noise included). SPICE's non-uniform timestep is
    resampled onto a uniform grid first (reported as ``fs_hz``).
    """
    if t.shape != y.shape:
        raise ValueError(f"t/y length mismatch: {t.size} vs {y.size}")
    if t.size < 8:
        raise ValueError(f"Need at least 8 samples for an FFT; got {t.size}")
    if not 1 <= n_harmonics <= 50:
        raise ValueError(f"n_harmonics must be 1..50; got {n_harmonics}")
    if window not in ("coherent", "hann"):
        raise ValueError(f"window must be 'coherent' or 'hann'; got {window!r}")

    warnings: list[str] = []
    span = float(t[-1]) - float(t[0])
    if span <= 0:
        raise ValueError("Time window has zero span.")

    if fundamental is not None:
        if fundamental <= 0:
            raise ValueError(f"fundamental must be > 0; got {fundamental}")
        f0 = float(fundamental)
        f0_source: Literal["given", "detected"] = "given"
    else:
        det = _detect_fundamental(t, y)
        if det is None or det <= 0:
            raise ValueError(
                "Could not detect a fundamental frequency; pass fundamental= "
                "(the signal may be aperiodic or too short)."
            )
        f0 = det
        f0_source = "detected"
        warnings.append(
            f"Fundamental auto-detected as {f0:g} Hz (largest FFT bin); pass "
            "fundamental= if that is wrong."
        )

    cycles_avail = f0 * span

    n_cyc = 0  # set in the coherent branch; the coherence guard only reads it there
    if window == "coherent":
        n_cyc = int(np.floor(cycles_avail + 1e-9))
        if n_cyc < 1:
            raise ValueError(
                f"Window spans {cycles_avail:.3g} fundamental cycles (< 1); cannot "
                "sample coherently. Widen [t_start, t_end] or use window='hann'."
            )
        t_end = float(t[0]) + n_cyc / f0
        n_fft = min(_next_pow2(max(t.size, 4 * n_harmonics * n_cyc + 1)), max_fft)
        tu = np.linspace(float(t[0]), t_end, n_fft, endpoint=False)
        win = np.ones(n_fft)
        coherent = True
        n_cycles = float(n_cyc)
        window_label = "coherent (rectangular)"
        fs = n_fft / (n_cyc / f0)
    else:
        n_fft = min(_next_pow2(t.size), max_fft)
        tu = np.linspace(float(t[0]), float(t[-1]), n_fft, endpoint=False)
        win = np.hanning(n_fft)
        coherent = False
        n_cycles = cycles_avail
        window_label = "hann"
        fs = n_fft / span
        warnings.append(
            "Hann window: harmonics may straddle bins (spectral leakage), so THD "
            "is approximate. Use window='coherent' for an exact result."
        )

    # np.interp is plain linear interpolation with no anti-alias filter. It only
    # ADDS points (no folding) while up-sampling, but if the FFT-length cap
    # forced n_fft below the window's own sample count we are DOWN-sampling, and
    # content above the new Nyquist folds into low bins and corrupts THD/THD+N.
    # Surface that rather than return a silently-aliased number.
    if n_fft < t.size:
        warnings.append(
            f"Resampled to {n_fft} points from {t.size} samples (FFT length cap "
            f"{max_fft}); content above the resample Nyquist may alias into the "
            "spectrum. Raise max_fft or narrow the window."
        )

    yu = np.interp(tu, t, y)
    yu = yu - yu.mean()
    # Scale to amplitude units (2/sum(win) = one-sided amplitude with window
    # coherent-gain correction; for the rectangular window this is 2/N). Raw
    # rfft bins are N/2 times the sinusoid amplitude — reporting them as
    # per-harmonic "magnitude" mislabels an 0.1 V harmonic as ~819 V at
    # n_fft=16384. Ratios (THD, db_rel) are unaffected by the common factor.
    spec = np.abs(np.fft.rfft(yu * win)) * (2.0 / float(np.sum(win)))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    nyq = fs / 2.0
    # Root-sum-squaring a windowed tone's main lobe overcounts its amplitude:
    # the coherent-gain scaling above already calibrates the PEAK bin, so the
    # half-height Hann neighbours add another sqrt(1.5) on top. By Parseval a
    # tone's lobe energy in these units is (amplitude * sqrt(N*sum(win^2)) /
    # sum(win))^2, so dividing a lobe RSS by this factor recovers the
    # amplitude (exactly 1 for the rectangular window).
    lobe_rss_norm = float(np.sqrt(n_fft * np.sum(win * win)) / np.sum(win))

    # Coherent sampling is only exact if the fundamental truly landed on its bin
    # (n_cyc). An imprecise f0 (e.g. auto-detected) makes the window a
    # non-integer number of cycles, spreading the fundamental into neighbouring
    # bins — single-bin reads are then wrong while still claiming exactness.
    # Detect that leakage and degrade to the leakage-tolerant peak-search
    # measurement (below) with a warning, instead of reporting a wrong THD as
    # coherent/exact.
    if coherent and 1 <= n_cyc < spec.size - 1:
        leak = max(float(spec[n_cyc - 1]), float(spec[n_cyc + 1]))
        peak_bin = n_cyc - 1 + int(np.argmax(spec[n_cyc - 1 : n_cyc + 2]))
        if peak_bin != n_cyc or (spec[n_cyc] > 0 and leak > 0.05 * float(spec[n_cyc])):
            coherent = False
            window_label = "near-coherent (residual leakage)"
            warnings.append(
                "Fundamental did not land exactly on an FFT bin (window is not an "
                "integer number of cycles — e.g. an imprecise fundamental); THD is "
                "approximate. Pass a precise fundamental= for an exact result."
            )

    def _bin_mag(target: float) -> tuple[float, float]:
        """(amplitude, frequency) near ``target`` Hz. Exact bin for coherent
        sampling; for a Hann window the energy spreads across the main lobe, so
        root-sum-square the ±2-bin lobe (the same width for every harmonic, so
        the THD ratio stays consistent), calibrate it back to amplitude via
        ``lobe_rss_norm``, and report the peak bin's frequency."""
        k = round(target / fs * n_fft)
        if k <= 0 or k >= spec.size:
            return 0.0, target
        if coherent:
            return float(spec[k]), float(freqs[k])
        lo = max(1, k - 2)
        hi = min(spec.size, k + 3)
        seg = spec[lo:hi]
        j = lo + int(np.argmax(seg))
        return float(np.sqrt(np.sum(seg**2)) / lobe_rss_norm), float(freqs[j])

    fund_mag, fund_freq = _bin_mag(f0)
    if fund_mag <= 0:
        raise ValueError(
            f"Fundamental at {f0:g} Hz has zero magnitude in the spectrum; the "
            "signal may be DC or the fundamental is wrong."
        )

    harmonics: list[HarmonicEntry] = []
    dropped = 0
    sum_sq = 0.0
    for h in range(2, n_harmonics + 1):
        fh = h * f0
        if fh >= nyq:
            dropped += 1
            continue
        mag, fr = _bin_mag(fh)
        sum_sq += mag * mag
        harmonics.append(
            {
                "n": h,
                "frequency": fr,
                "magnitude": mag,
                "db_rel": float(20.0 * np.log10(mag / fund_mag)) if mag > 0 else float("-inf"),
            }
        )
    if dropped:
        warnings.append(
            f"{dropped} requested harmonic(s) lie above Nyquist ({nyq:g} Hz) and "
            "were dropped; raise the sample density or lower n_harmonics."
        )

    thd_ratio = float(np.sqrt(sum_sq) / fund_mag)

    # THD+N: every non-DC, non-fundamental bin's energy relative to the
    # fundamental (Hann gets a ±2-bin guard around the fundamental for leakage).
    k_fund = round(f0 / fs * n_fft)
    mask = np.ones(spec.size, dtype=bool)
    mask[0] = False
    guard = 0 if coherent else 2
    mask[max(1, k_fund - guard) : k_fund + guard + 1] = False  # slice clamps past the end
    # Energy-over-energy ratio: the numerator is a per-bin RSS in raw spec
    # units, so undo the fundamental's lobe→amplitude calibration (a no-op on
    # the coherent/rectangular path, where lobe_rss_norm == 1).
    thd_n_ratio = float(np.sqrt(np.sum(spec[mask] ** 2)) / (fund_mag * lobe_rss_norm))

    if n_cycles < 3:
        warnings.append(
            f"Only {n_cycles:.3g} fundamental cycle(s) in the window; few-cycle "
            "estimates are unreliable — widen the window."
        )

    return {
        "fundamental_hz": fund_freq,
        "fundamental_source": f0_source,
        "thd_ratio": thd_ratio,
        "thd_pct": thd_ratio * 100.0,
        "thd_db": float(20.0 * np.log10(thd_ratio)) if thd_ratio > 0 else float("-inf"),
        "thd_n_ratio": thd_n_ratio,
        "thd_n_pct": thd_n_ratio * 100.0,
        "harmonics": harmonics,
        "n_harmonics_used": len(harmonics),
        "window": window_label,
        "coherent": coherent,
        "n_cycles": n_cycles,
        "n_fft": n_fft,
        "fs_hz": float(fs),
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
    # Named t_at_min/t_at_max (time OF the extremum) — t_min/t_max would read
    # as window bounds next to t_start/t_end.
    t_at_min = float(t[int(np.argmin(y))])
    t_at_max = float(t[int(np.argmax(y))])

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
        "t_at_min": t_at_min,
        "t_at_max": t_at_max,
    }


def _equal_time_buckets(t: np.ndarray, n_buckets: int) -> tuple[np.ndarray, np.ndarray, int]:
    """Tile ``[t[0], t[-1]]`` into up to ``n_buckets`` equal-time buckets.

    Returns ``(edges, bounds, n)``: ``edges`` are the ``n + 1`` bucket boundaries
    and ``bounds`` are ``n + 1`` cumulative sample counts, so bucket ``b`` owns the
    contiguous slice ``[bounds[b]:bounds[b + 1]]`` of a sorted ``t`` (empty buckets
    have ``bounds[b] == bounds[b + 1]``). A degenerate span — or ``n == 1`` —
    collapses to one bucket holding every sample. Assumes a non-decreasing axis.
    """
    x_start = float(t[0])
    x_end = float(t[-1])
    n = min(n_buckets, len(t))
    if x_end <= x_start or n == 1:
        edges = np.array([x_start, x_end], dtype=float)
        idx = np.zeros(len(t), dtype=np.intp)
        n = 1
    else:
        edges = np.linspace(x_start, x_end, n + 1)
        # Bucket index = count of interior edges <= the sample (side="right" so a
        # sample landing exactly on an edge falls into the upper bucket).
        idx = np.searchsorted(edges[1:-1], t, side="right")
    # idx is non-decreasing (t and edges are sorted), so each bucket is a
    # CONTIGUOUS slice; cumulative bincount gives its bounds in O(samples + n).
    return (
        edges,
        np.concatenate(([0], np.cumsum(np.bincount(idx, minlength=n)))).astype(np.intp),
        n,
    )


def stat_envelope(t: np.ndarray, y: np.ndarray, n_buckets: int) -> StatEnvelopeOutput:
    """Decimate a real waveform into up to ``n_buckets`` equal-time buckets.

    Each bucket carries the raw sample ``min``/``max`` over its time slice — a
    narrow spike is never averaged away (the point of an envelope vs a plain
    downsample) — plus time-weighted trapezoidal ``mean``/``rms`` (via
    :func:`compute_signal_stats`, correct on LTspice's adaptive timestep) and
    ``pk_pk``/``crest_factor`` facts. Buckets tile ``[t[0], t[-1]]`` by equal
    time width so each maps to a comparable real interval (like a plot column);
    intervals with no samples are skipped, so ``bucket_count`` may be less than
    the requested count. ``crest_factor`` is ``peak/rms`` (peak =
    max(|min|, |max|)), or ``None`` when ``rms`` is ~0 — a meaningless ratio is
    omitted rather than reported. Assumes a monotonically non-decreasing axis
    (as produced by :func:`window_and_clean`).
    """
    if len(t) != len(y):
        raise ValueError(f"Axis and wave have different lengths: {len(t)} vs {len(y)}")
    if len(t) < 1:
        raise ValueError("Signal has no samples")
    if n_buckets < 1:
        raise ValueError(f"n_buckets must be >= 1, got {n_buckets}")

    x_start = float(t[0])
    x_end = float(t[-1])
    edges, bounds, n = _equal_time_buckets(t, n_buckets)

    buckets: list[WaveformBucket] = []
    for b in range(n):
        lo = int(bounds[b])
        hi = int(bounds[b + 1])
        if hi == lo:
            continue
        core = compute_signal_stats(t[lo:hi], y[lo:hi])
        rms = core["rms"]
        peak = max(abs(core["min"]), abs(core["max"]))
        crest = float(peak / rms) if rms > _LEVEL_EPSILON else None
        buckets.append(
            {
                "x_start": float(edges[b]),
                "x_end": float(edges[b + 1]),
                "min": core["min"],
                "max": core["max"],
                "mean": core["mean"],
                "rms": rms,
                "pk_pk": core["pk_pk"],
                "crest_factor": crest,
                "num_samples": core["num_samples"],
            }
        )

    return {
        "buckets": buckets,
        "point_count": len(t),
        "bucket_count": len(buckets),
        "x_start": x_start,
        "x_end": x_end,
        "decimated": len(t) > len(buckets),
    }


def downsample_minmax(
    x: np.ndarray, y: np.ndarray, target_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample ``(x, y)`` to about ``target_points``, preserving min and max.

    Buckets the axis into equal-time spans (:func:`_equal_time_buckets`) and emits
    two points per non-empty bucket — its ``min`` at the left edge and ``max`` at
    the right — so a narrow glitch's amplitude survives display decimation, losing
    only sub-bucket timing. Returns flat ``(x, y)`` arrays in axis order. Computes
    only per-bucket min/max via ``reduceat`` over the contiguous slices (not the
    full envelope stats). For an unbucketed view pass ``target_points`` >= the
    sample count.

    Used by ``plot_waveform`` to bound the points handed to the browser; the
    full-resolution data stays on disk (``export_waveform``).
    """
    if len(x) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=float)
    # _equal_time_buckets needs an ascending axis; a high->low sweep (a .dc 5 0
    # or descending .noise) otherwise reads x_end <= x_start and collapses to a
    # single bucket — two output points for the whole curve. Flip to ascending,
    # bucket, then restore the caller's axis order.
    flipped = x.size > 1 and x[0] > x[-1]
    if flipped:
        x, y = x[::-1], y[::-1]
    edges, bounds, _ = _equal_time_buckets(x, max(1, target_points // 2))
    starts = bounds[:-1]
    nonempty = bounds[1:] > starts
    block_starts = starts[nonempty]
    xs = np.empty(2 * len(block_starts), dtype=float)
    ys = np.empty_like(xs)
    xs[0::2] = edges[:-1][nonempty]
    xs[1::2] = edges[1:][nonempty]
    ys[0::2] = np.minimum.reduceat(y, block_starts)
    ys[1::2] = np.maximum.reduceat(y, block_starts)
    if flipped:
        return xs[::-1], ys[::-1]
    return xs, ys


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
                "min_step_index": None,
                "max_step_index": None,
                "histogram": [],
            }
            continue

        arr = np.asarray(valid, dtype=float)
        e_min = float(np.min(arr))
        e_max = float(np.max(arr))

        # Neutral min/max, not best/worst: whether higher or lower is better is
        # the metric's polarity, which the aggregator can't know — the caller
        # judges. (Mirrors batch_results' max_case_run/min_case_run.)
        min_step: int | None = None
        max_step: int | None = None
        min_val: float | None = None
        max_val: float | None = None
        for i, v in enumerate(values):
            if v is None:
                continue
            if min_val is None or v < min_val:
                min_val = v
                min_step = i
            if max_val is None or v > max_val:
                max_val = v
                max_step = i

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
            "min_step_index": min_step,
            "max_step_index": max_step,
            "histogram": histogram,
        }

    return result
