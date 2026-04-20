"""Unit tests for pure waveform analysis primitives.

Synthetic arrays only — no I/O, no spicelib. Exercises analytical cases with
known answers and edge-case handling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ltspice_mcp.lib.signal_analysis import (
    _interp_crossings,
    analyze_edge,
    analyze_periodic,
    analyze_pulse_response,
    analyze_timing_between,
    compute_measurement_stats,
    compute_signal_stats,
    window_and_clean,
)

# ---------------------------------------------------------------------------
# window_and_clean
# ---------------------------------------------------------------------------


class TestWindowAndClean:
    def test_full_window(self):
        t = np.linspace(0, 1, 100)
        y = np.sin(2 * np.pi * t)
        t_out, _y_out, dropped = window_and_clean(t, y, None, None)
        assert len(t_out) == 100
        assert dropped == 0

    def test_slice_window(self):
        t = np.linspace(0, 1, 101)
        y = np.zeros_like(t)
        t_out, _, _ = window_and_clean(t, y, 0.2, 0.8)
        assert t_out[0] >= 0.2 - 1e-12
        assert t_out[-1] <= 0.8 + 1e-12

    def test_strips_nan(self):
        t = np.linspace(0, 1, 100)
        y = np.zeros_like(t)
        y[::10] = np.nan
        _, y_out, dropped = window_and_clean(t, y, None, None)
        assert dropped == 10
        assert np.all(np.isfinite(y_out))

    def test_strips_inf(self):
        t = np.linspace(0, 1, 100)
        y = np.zeros_like(t)
        y[5] = np.inf
        y[7] = -np.inf
        _, y_out, dropped = window_and_clean(t, y, None, None)
        assert dropped == 2
        assert np.all(np.isfinite(y_out))

    def test_too_few_samples(self):
        with pytest.raises(ValueError, match="at least 3"):
            window_and_clean(np.array([0.0, 1.0]), np.array([0.0, 1.0]), None, None)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="different lengths"):
            window_and_clean(np.zeros(10), np.zeros(11), None, None)

    def test_non_monotonic_axis(self):
        t = np.array([0.0, 1.0, 0.5, 2.0])
        y = np.zeros_like(t)
        with pytest.raises(ValueError, match="monotonically"):
            window_and_clean(t, y, None, None)

    def test_window_out_of_range(self):
        t = np.linspace(0, 1, 100)
        y = np.zeros_like(t)
        with pytest.raises(ValueError, match="outside axis range"):
            window_and_clean(t, y, None, 5.0)

    def test_inverted_window(self):
        t = np.linspace(0, 1, 100)
        y = np.zeros_like(t)
        with pytest.raises(ValueError, match="must be less than"):
            window_and_clean(t, y, 0.8, 0.2)


# ---------------------------------------------------------------------------
# _interp_crossings
# ---------------------------------------------------------------------------


class TestInterpCrossings:
    def test_linear_ramp(self):
        t = np.linspace(0, 1, 1001)
        y = t  # crosses 0.5 at t=0.5
        cross = _interp_crossings(t, y, 0.5, "rising")
        assert len(cross) == 1
        assert cross[0] == pytest.approx(0.5, abs=1e-3)

    def test_falling_ignored_when_rising_requested(self):
        t = np.linspace(0, 1, 1001)
        y = 1.0 - t
        assert _interp_crossings(t, y, 0.5, "rising") == []
        falling = _interp_crossings(t, y, 0.5, "falling")
        assert len(falling) == 1

    def test_multiple_crossings(self):
        t = np.linspace(0, 1, 10001)
        y = np.sin(2 * np.pi * 3 * t)  # 3 cycles → 3 rising zero-crossings
        cross = _interp_crossings(t, y, 0.0, "rising")
        assert len(cross) == 3

    def test_exact_threshold_sample(self):
        # Sample lands exactly on threshold, then rises: still register
        t = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([-1.0, 0.0, 1.0, 2.0])
        cross = _interp_crossings(t, y, 0.0, "rising")
        assert len(cross) == 1
        assert cross[0] == pytest.approx(1.0, abs=1e-9)

    def test_no_crossing(self):
        t = np.linspace(0, 1, 100)
        y = np.full_like(t, 2.0)
        assert _interp_crossings(t, y, 5.0, "rising") == []

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction must be"):
            _interp_crossings(np.zeros(3), np.zeros(3), 0.0, "sideways")


# ---------------------------------------------------------------------------
# analyze_edge
# ---------------------------------------------------------------------------


def _linear_edge(
    t0: float,
    t1: float,
    v0: float,
    v1: float,
    n: int = 2001,
    pre_pad: float = 0.1,
    post_pad: float = 0.1,
):
    """Build a transient that sits at v0, linearly ramps v0→v1 over [t0,t1], then sits at v1."""
    total_start = t0 - pre_pad
    total_end = t1 + post_pad
    t = np.linspace(total_start, total_end, n)
    y = np.where(
        t <= t0,
        v0,
        np.where(t >= t1, v1, v0 + (v1 - v0) * (t - t0) / (t1 - t0)),
    )
    return t, y


class TestAnalyzeEdge:
    def test_ideal_rising_10_90(self):
        # Linear rise 0→1 over [1ms, 2ms] → 10-90% rise time = 0.8 ms
        # Slew rate (standard): threshold_delta / transition_time = 0.8V / 0.8ms = 1000 V/s
        t, y = _linear_edge(1e-3, 2e-3, 0.0, 1.0)
        result = analyze_edge(t, y)
        assert result["is_rise_time"] is True
        assert result["transition_time"] == pytest.approx(0.8e-3, rel=1e-3)
        assert result["slew_rate"] == pytest.approx(1000.0, rel=1e-3)
        assert result["low_level"] == pytest.approx(0.0, abs=1e-6)
        assert result["high_level"] == pytest.approx(1.0, abs=1e-6)

    def test_ideal_falling(self):
        t, y = _linear_edge(1e-3, 2e-3, 1.0, 0.0)
        result = analyze_edge(t, y)
        assert result["is_rise_time"] is False
        assert result["edge_direction"] == "falling"
        assert result["transition_time"] == pytest.approx(0.8e-3, rel=1e-3)

    def test_20_80_thresholds(self):
        t, y = _linear_edge(1e-3, 2e-3, 0.0, 1.0)
        result = analyze_edge(t, y, low_pct=20, high_pct=80)
        assert result["transition_time"] == pytest.approx(0.6e-3, rel=1e-3)

    def test_no_edge_flat(self):
        t = np.linspace(0, 1, 1000)
        y = np.full_like(t, 5.0)
        with pytest.raises(ValueError, match="No edge detected"):
            analyze_edge(t, y)

    def test_invalid_thresholds(self):
        t, y = _linear_edge(0, 1, 0, 1)
        with pytest.raises(ValueError, match="low_pct < high_pct"):
            analyze_edge(t, y, low_pct=90, high_pct=10)

    def test_direction_mismatch_errors(self):
        # Pure rising edge: asking for falling finds no falling mid-crossings → error.
        t, y = _linear_edge(1e-3, 2e-3, 0.0, 1.0)
        with pytest.raises(ValueError, match="No falling edge"):
            analyze_edge(t, y, edge="falling")

    def test_direction_mismatch_bipolar(self):
        # Signal rises then falls; requesting falling with bad auto should still
        # find the falling section — but our algorithm uses window endpoints
        # for level detection, so if ends are equal, it errors. This documents
        # that behavior.
        t = np.linspace(0, 2e-3, 2001)
        y = np.where(t < 1e-3, t / 1e-3, 2.0 - t / 1e-3)  # triangle 0→1→0
        with pytest.raises(ValueError, match="No edge detected"):
            analyze_edge(t, y)

    def test_edge_index_out_of_range(self):
        t = np.linspace(0, 1, 1001)
        y = np.where(t < 0.5, 0.0, 1.0)
        with pytest.raises(ValueError, match="edge_index=5"):
            analyze_edge(t, y, edge_index=5)


# ---------------------------------------------------------------------------
# analyze_pulse_response
# ---------------------------------------------------------------------------


def _second_order_step(
    zeta: float,
    wn: float,
    t_end: float,
    n: int = 5001,
    final: float = 1.0,
    pre_pad: float | None = None,
):
    """Exact step response of an underdamped 2nd-order system 1/(s^2+2*zeta*wn*s+wn^2).

    Returns (t, y). Pre-pads with a flat 0 plateau so auto-level detection
    picks up initial=0 correctly. ``pre_pad`` defaults to ~10% of t_end.
    """
    if pre_pad is None:
        pre_pad = 0.1 * t_end
    n_pre = int(n * pre_pad / (pre_pad + t_end))
    n_step = n - n_pre
    t_pre = np.linspace(-pre_pad, 0, n_pre, endpoint=False) if n_pre > 0 else np.array([])
    t_step = np.linspace(0, t_end, n_step)
    if zeta < 1:
        wd = wn * math.sqrt(1 - zeta**2)
        phi = math.atan2(math.sqrt(1 - zeta**2), zeta)
        y_step = final * (
            1 - np.exp(-zeta * wn * t_step) / math.sqrt(1 - zeta**2) * np.sin(wd * t_step + phi)
        )
    elif zeta == 1:
        y_step = final * (1 - np.exp(-wn * t_step) * (1 + wn * t_step))
    else:
        r = math.sqrt(zeta**2 - 1)
        s1 = -wn * (zeta - r)
        s2 = -wn * (zeta + r)
        y_step = final * (1 + (s2 * np.exp(s1 * t_step) - s1 * np.exp(s2 * t_step)) / (s1 - s2))
    y_pre = np.zeros(n_pre)
    t = np.concatenate([t_pre, t_step])
    y = np.concatenate([y_pre, y_step])
    return t, y


class TestAnalyzePulseResponse:
    def test_underdamped_overshoot_matches_analytical(self):
        # Classic 2nd-order formula: Mp = exp(-pi*zeta / sqrt(1-zeta^2)) * 100
        zeta = 0.2
        wn = 2 * math.pi * 1000  # 1kHz natural
        expected_overshoot = math.exp(-math.pi * zeta / math.sqrt(1 - zeta**2)) * 100
        t, y = _second_order_step(zeta, wn, t_end=20e-3, n=20001)
        result = analyze_pulse_response(t, y)
        assert result["direction"] == "rising"
        assert result["overshoot_pct"] == pytest.approx(expected_overshoot, rel=0.05)
        assert result["steady_state_value"] == pytest.approx(1.0, abs=0.02)

    def test_overdamped_no_overshoot(self):
        t, y = _second_order_step(zeta=2.0, wn=2 * math.pi * 100, t_end=0.1)
        result = analyze_pulse_response(t, y)
        assert result["overshoot_pct"] == 0.0
        assert result["direction"] == "rising"

    def test_falling_step(self):
        # Invert a rising step to get a falling one
        t, y = _second_order_step(zeta=0.2, wn=2 * math.pi * 1000, t_end=20e-3)
        y_fall = 1.0 - y
        result = analyze_pulse_response(t, y_fall)
        assert result["direction"] == "falling"
        assert result["overshoot_pct"] > 0

    def test_no_step(self):
        t = np.linspace(0, 1, 1000)
        y = np.full_like(t, 3.3)
        with pytest.raises(ValueError, match="No step detected"):
            analyze_pulse_response(t, y)

    def test_explicit_initial_final(self):
        t, y = _linear_edge(1e-3, 2e-3, 0.0, 1.0)
        result = analyze_pulse_response(t, y, initial_value=0.0, final_value=1.0)
        assert result["initial_value"] == 0.0
        assert result["steady_state_value"] == 1.0

    def test_never_settles(self):
        # A signal that keeps oscillating without decay
        t = np.linspace(0, 10e-3, 5001)
        y = 1.0 + 0.5 * np.sin(2 * np.pi * 1000 * t)
        # Provide explicit initial/final: initial=1.0, final=1.0 → no step, so
        # inject a fake initial
        with pytest.raises(ValueError, match="No step detected"):
            analyze_pulse_response(t, y)

    def test_settles(self):
        # Light ringing that dies within window; pad pre-step plateau.
        t_pre = np.linspace(-1e-3, 0, 500, endpoint=False)
        t_step = np.linspace(0, 10e-3, 5001)
        y_pre = np.zeros_like(t_pre)
        y_step = 1.0 - np.exp(-500 * t_step) * np.cos(2 * np.pi * 500 * t_step)
        t = np.concatenate([t_pre, t_step])
        y = np.concatenate([y_pre, y_step])
        result = analyze_pulse_response(t, y, settling_tolerance_pct=2.0)
        assert result["settling_time"] is not None
        assert 0 < result["settling_time"] < 11e-3

    def test_invalid_tolerance(self):
        t, y = _linear_edge(1e-3, 2e-3, 0.0, 1.0)
        with pytest.raises(ValueError, match="must be positive"):
            analyze_pulse_response(t, y, settling_tolerance_pct=-1.0)


# ---------------------------------------------------------------------------
# analyze_timing_between
# ---------------------------------------------------------------------------


class TestAnalyzeTimingBetween:
    def test_known_delay(self):
        t = np.linspace(0, 1, 10001)
        ya = np.where(t < 0.3, 0.0, 1.0)
        yb = np.where(t < 0.5, 0.0, 1.0)
        result = analyze_timing_between(t, ya, yb)
        assert result["delay"] == pytest.approx(0.2, abs=1e-3)
        assert result["t_a"] == pytest.approx(0.3, abs=1e-3)
        assert result["t_b"] == pytest.approx(0.5, abs=1e-3)

    def test_signal_b_leads_negative_delay(self):
        t = np.linspace(0, 1, 10001)
        ya = np.where(t < 0.5, 0.0, 1.0)
        yb = np.where(t < 0.3, 0.0, 1.0)
        result = analyze_timing_between(t, ya, yb)
        assert result["delay"] == pytest.approx(-0.2, abs=1e-3)

    def test_asymmetric_rails(self):
        # V_in 0-3V, V_out 0-1.8V, 50% each should still find the transitions
        t = np.linspace(0, 1, 10001)
        ya = np.where(t < 0.3, 0.0, 3.0)
        yb = np.where(t < 0.4, 0.0, 1.8)
        result = analyze_timing_between(t, ya, yb)
        assert result["threshold_a_used"] == pytest.approx(1.5, abs=0.01)
        assert result["threshold_b_used"] == pytest.approx(0.9, abs=0.01)
        assert result["delay"] == pytest.approx(0.1, abs=1e-3)

    def test_no_crossing_a(self):
        t = np.linspace(0, 1, 1000)
        ya = np.full_like(t, 1.0)
        yb = np.where(t < 0.5, 0.0, 1.0)
        with pytest.raises(ValueError, match="constant"):
            analyze_timing_between(t, ya, yb)

    def test_explicit_absolute_threshold(self):
        t = np.linspace(0, 1, 10001)
        ya = np.where(t < 0.3, 0.0, 2.0)
        yb = np.where(t < 0.5, 0.0, 2.0)
        result = analyze_timing_between(t, ya, yb, threshold_a=1.0, threshold_b=1.0)
        assert result["threshold_a_used"] == 1.0
        assert result["threshold_b_used"] == 1.0

    def test_falling_direction(self):
        t = np.linspace(0, 1, 10001)
        ya = np.where(t < 0.3, 1.0, 0.0)
        yb = np.where(t < 0.5, 1.0, 0.0)
        result = analyze_timing_between(t, ya, yb, direction_a="falling", direction_b="falling")
        assert result["delay"] == pytest.approx(0.2, abs=1e-3)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            analyze_timing_between(np.zeros(5), np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# analyze_periodic
# ---------------------------------------------------------------------------


class TestAnalyzePeriodic:
    def test_square_wave_50_duty(self):
        freq = 1000.0
        period = 1.0 / freq
        t = np.linspace(0, 5 * period, 50001)  # 5 periods
        # Use sign to get a clean square wave
        y = np.where(np.sin(2 * np.pi * freq * t) >= 0, 1.0, 0.0)
        result = analyze_periodic(t, y)
        assert result["period"] == pytest.approx(period, rel=1e-3)
        assert result["frequency"] == pytest.approx(freq, rel=1e-3)
        assert result["duty_cycle_pct"] == pytest.approx(50.0, abs=0.5)

    def test_40_duty_cycle(self):
        freq = 1000.0
        period = 1.0 / freq
        t = np.linspace(0, 10 * period, 100001)
        phase = (t * freq) % 1.0
        y = np.where(phase < 0.4, 1.0, 0.0)
        result = analyze_periodic(t, y)
        assert result["duty_cycle_pct"] == pytest.approx(40.0, abs=1.0)

    def test_frequency_from_sine(self):
        freq = 500.0
        t = np.linspace(0, 10 / freq, 100001)
        y = np.sin(2 * np.pi * freq * t)
        result = analyze_periodic(t, y)
        assert result["frequency"] == pytest.approx(freq, rel=1e-3)

    def test_constant_signal_errors(self):
        t = np.linspace(0, 1, 1000)
        y = np.full_like(t, 2.5)
        with pytest.raises(ValueError, match="constant"):
            analyze_periodic(t, y)

    def test_too_few_periods(self):
        # Only half a period
        t = np.linspace(0, 0.5e-3, 1000)
        y = np.sin(2 * np.pi * 1000 * t)
        with pytest.raises(ValueError, match="rising edge"):
            analyze_periodic(t, y, min_periods=3)

    def test_threshold_out_of_range(self):
        freq = 1000.0
        t = np.linspace(0, 5 / freq, 5000)
        y = np.sin(2 * np.pi * freq * t)  # range [-1, 1]
        with pytest.raises(ValueError, match="strictly between"):
            analyze_periodic(t, y, threshold=5.0)

    def test_jitter_measurement(self):
        # Inject known period variation
        freq = 1000.0
        period = 1.0 / freq
        # Generate edges with known jitter
        rng = np.random.default_rng(42)
        n_periods = 50
        jitter_std = period * 0.01  # 1% jitter
        edges = np.cumsum(rng.normal(period, jitter_std, n_periods))
        t = np.linspace(0, edges[-1] + period, 200001)
        # Build square wave from edges
        y = np.zeros_like(t)
        for i in range(0, len(edges) - 1, 2):
            mask = (t >= edges[i]) & (t < edges[i + 1])
            y[mask] = 1.0
        result = analyze_periodic(t, y, min_periods=5)
        # Jitter std on rising edges ≈ sqrt(2) * underlying edge jitter for
        # period-to-period variation; just check it's in the right ballpark.
        assert result["jitter_rms"] > 0


# ---------------------------------------------------------------------------
# compute_measurement_stats
# ---------------------------------------------------------------------------


class TestComputeMeasurementStats:
    def test_basic(self):
        meas = {"fc": [1000.0, 1100.0, 900.0, 1050.0, 950.0]}
        result = compute_measurement_stats(meas)
        entry = result["fc"]
        assert entry["valid_count"] == 5
        assert entry["failure_count"] == 0
        assert entry["min"] == 900.0
        assert entry["max"] == 1100.0
        assert entry["mean"] == pytest.approx(1000.0)
        assert entry["median"] == 1000.0
        assert entry["best_step_index"] == 2  # min at index 2
        assert entry["worst_step_index"] == 1  # max at index 1

    def test_with_failures(self):
        meas = {"fc": [1000.0, None, 900.0, None, 1100.0]}
        result = compute_measurement_stats(meas)
        entry = result["fc"]
        assert entry["valid_count"] == 3
        assert entry["failure_count"] == 2
        assert entry["best_step_index"] == 2
        assert entry["worst_step_index"] == 4

    def test_all_failures(self):
        meas = {"fc": [None, None, None]}
        result = compute_measurement_stats(meas)
        entry = result["fc"]
        assert entry["valid_count"] == 0
        assert entry["failure_count"] == 3
        assert entry["min"] is None
        assert entry["histogram"] == []

    def test_single_value_no_histogram(self):
        meas = {"fc": [42.0]}
        result = compute_measurement_stats(meas)
        entry = result["fc"]
        assert entry["valid_count"] == 1
        assert entry["histogram"] == []  # too few for histogram

    def test_histogram_bins(self):
        meas = {"fc": [float(i) for i in range(100)]}
        result = compute_measurement_stats(meas, histogram_bins=5)
        entry = result["fc"]
        assert len(entry["histogram"]) == 5
        assert sum(bin_["count"] for bin_ in entry["histogram"]) == 100

    def test_histogram_bins_zero_skips(self):
        meas = {"fc": [1.0, 2.0, 3.0]}
        result = compute_measurement_stats(meas, histogram_bins=0)
        assert result["fc"]["histogram"] == []

    def test_select_single_measurement(self):
        meas = {"fc": [1.0, 2.0, 3.0], "vp": [10.0, 20.0, 30.0]}
        result = compute_measurement_stats(meas, measurement="fc")
        assert list(result.keys()) == ["fc"]

    def test_unknown_measurement(self):
        meas = {"fc": [1.0]}
        with pytest.raises(ValueError, match="not found"):
            compute_measurement_stats(meas, measurement="missing")

    def test_invalid_bins(self):
        with pytest.raises(ValueError, match="must be >= 0"):
            compute_measurement_stats({"x": [1.0]}, histogram_bins=-1)

    def test_std_single_value_is_zero(self):
        meas = {"fc": [42.0]}
        result = compute_measurement_stats(meas)
        assert result["fc"]["std"] == 0.0

    def test_std_multi_value(self):
        meas = {"fc": [0.0, 10.0]}
        result = compute_measurement_stats(meas)
        # std with ddof=0 of [0, 10] is 5.0
        assert result["fc"]["std"] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# compute_signal_stats
# ---------------------------------------------------------------------------


class TestComputeSignalStats:
    def test_dc_signal(self):
        t = np.linspace(0, 1e-3, 1001)
        y = np.full_like(t, 3.14)
        r = compute_signal_stats(t, y)
        assert r["mean"] == pytest.approx(3.14)
        assert r["rms"] == pytest.approx(3.14)
        assert r["std"] == pytest.approx(0.0, abs=1e-12)
        assert r["min"] == pytest.approx(3.14)
        assert r["max"] == pytest.approx(3.14)
        assert r["pk_pk"] == pytest.approx(0.0)
        assert r["abs_mean"] == pytest.approx(3.14)
        assert r["duration"] == pytest.approx(1e-3)
        assert r["num_samples"] == 1001

    def test_sine_rms_is_amplitude_over_sqrt2(self):
        # Integer number of periods so window RMS equals the analytical value.
        freq = 1000.0
        periods = 10
        t = np.linspace(0, periods / freq, 100001)
        amp = 5.0
        y = amp * np.sin(2 * np.pi * freq * t)
        r = compute_signal_stats(t, y)
        assert r["mean"] == pytest.approx(0.0, abs=1e-3)
        assert r["rms"] == pytest.approx(amp / math.sqrt(2), rel=1e-4)
        assert r["std"] == pytest.approx(amp / math.sqrt(2), rel=1e-4)
        assert r["min"] == pytest.approx(-amp, rel=1e-4)
        assert r["max"] == pytest.approx(amp, rel=1e-4)
        assert r["pk_pk"] == pytest.approx(2 * amp, rel=1e-4)
        # Average rectified value of a sine is 2*amp/pi
        assert r["abs_mean"] == pytest.approx(2 * amp / math.pi, rel=1e-3)

    def test_sine_with_dc_offset(self):
        freq = 1000.0
        t = np.linspace(0, 10 / freq, 100001)
        amp, offset = 2.0, 3.0
        y = offset + amp * np.sin(2 * np.pi * freq * t)
        r = compute_signal_stats(t, y)
        assert r["mean"] == pytest.approx(offset, rel=1e-3)
        # RMS of offset+sine: sqrt(offset^2 + amp^2/2)
        expected_rms = math.sqrt(offset**2 + amp**2 / 2)
        assert r["rms"] == pytest.approx(expected_rms, rel=1e-4)
        # Std strips the DC component
        assert r["std"] == pytest.approx(amp / math.sqrt(2), rel=1e-3)

    def test_square_wave_rms(self):
        # Unit-amplitude square wave (0/1): RMS = sqrt(duty), mean = duty
        t = np.linspace(0, 1.0, 100001)
        duty = 0.3
        phase = (t * 10.0) % 1.0  # 10 periods
        y = np.where(phase < duty, 1.0, 0.0)
        r = compute_signal_stats(t, y)
        assert r["mean"] == pytest.approx(duty, rel=1e-2)
        assert r["rms"] == pytest.approx(math.sqrt(duty), rel=1e-2)

    def test_nonuniform_axis_trapezoidal(self):
        # Irregular sampling: RMS should still be amplitude / sqrt(2).
        rng = np.random.default_rng(0)
        freq = 100.0
        t = np.sort(rng.uniform(0, 10 / freq, size=20001))
        amp = 2.0
        y = amp * np.sin(2 * np.pi * freq * t)
        r = compute_signal_stats(t, y)
        assert r["rms"] == pytest.approx(amp / math.sqrt(2), rel=5e-3)

    def test_single_sample_falls_back(self):
        r = compute_signal_stats(np.array([0.5]), np.array([7.0]))
        assert r["duration"] == 0.0
        assert r["num_samples"] == 1
        assert r["mean"] == pytest.approx(7.0)
        assert r["rms"] == pytest.approx(7.0)
        assert r["std"] == pytest.approx(0.0)

    def test_zero_duration_falls_back(self):
        t = np.array([1.0, 1.0, 1.0])
        y = np.array([2.0, 4.0, 6.0])
        r = compute_signal_stats(t, y)
        assert r["duration"] == 0.0
        assert r["mean"] == pytest.approx(4.0)
        assert r["pk_pk"] == pytest.approx(4.0)

    def test_negative_values(self):
        t = np.linspace(0, 1.0, 1001)
        y = -np.ones_like(t)
        r = compute_signal_stats(t, y)
        assert r["mean"] == pytest.approx(-1.0)
        assert r["rms"] == pytest.approx(1.0)
        assert r["abs_mean"] == pytest.approx(1.0)
        assert r["min"] == pytest.approx(-1.0)
        assert r["max"] == pytest.approx(-1.0)

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="different lengths"):
            compute_signal_stats(np.zeros(5), np.zeros(6))

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="no samples"):
            compute_signal_stats(np.array([]), np.array([]))
