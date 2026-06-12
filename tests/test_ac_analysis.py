"""Unit tests for AC analysis primitives.

Synthetic transfer functions only — no I/O, no spicelib. Closed-form filters
(Butterworth, biquad) give known answers to check against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from ltspice_mcp.lib.ac_analysis import (
    classify_filter,
    compute_filter_metrics,
    compute_resonances,
    compute_roll_off,
    compute_stability_metrics,
    detect_crossings,
    find_crossings_any_quantity,
    gain_at_frequencies,
    log_interp,
    log_interp_complex,
    magnitude_db,
    prepare_ac_arrays,
    unwrap_phase_safe,
)

# ---------------------------------------------------------------------------
# Synthetic response builders
# ---------------------------------------------------------------------------


def _log_freqs(decade_lo: int, decade_hi: int, n: int) -> np.ndarray:
    return np.logspace(decade_lo, decade_hi, n)


def _lpf_1pole(freqs: np.ndarray, fc: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    return wc / (s + wc)


def _lpf_2pole(freqs: np.ndarray, fc: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    return (wc * wc) / ((s + wc) ** 2)


def _hpf_2pole(freqs: np.ndarray, fc: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    return (s * s) / ((s + wc) ** 2)


def _bpf(freqs: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    return _lpf_1pole(freqs, f_hi) * (
        1 - _lpf_1pole(freqs, f_lo)
    )  # HPF = 1 - LPF of the same corner


def _biquad_resonator(freqs: np.ndarray, f0: float, Q: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    w0 = 2 * np.pi * f0
    return (w0 * w0) / (s * s + (w0 / Q) * s + w0 * w0)


def _notch(freqs: np.ndarray, f0: float, Q: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    w0 = 2 * np.pi * f0
    return (s * s + w0 * w0) / (s * s + (w0 / Q) * s + w0 * w0)


def _two_pole_loop(freqs: np.ndarray, A: float, p1: float, p2: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    w1 = 2 * np.pi * p1
    w2 = 2 * np.pi * p2
    return A / ((1 + s / w1) * (1 + s / w2))


def _three_pole_loop(freqs: np.ndarray, A: float, p1: float, p2: float, p3: float) -> np.ndarray:
    s = 1j * 2 * np.pi * freqs
    return A / (
        (1 + s / (2 * np.pi * p1)) * (1 + s / (2 * np.pi * p2)) * (1 + s / (2 * np.pi * p3))
    )


# ---------------------------------------------------------------------------
# prepare_ac_arrays
# ---------------------------------------------------------------------------


class TestPrepareAcArrays:
    def test_happy_path(self):
        f = _log_freqs(1, 5, 200)
        H = _lpf_1pole(f, 1000.0)
        f_out, H_out = prepare_ac_arrays(f, H)
        assert f_out.dtype == float
        assert H_out.dtype == np.complex128
        assert np.all(np.diff(f_out) > 0)

    def test_rejects_real_wave(self):
        f = _log_freqs(1, 5, 100)
        y = np.zeros_like(f)
        with pytest.raises(ValueError, match="transient"):
            prepare_ac_arrays(f, y)

    def test_rejects_non_positive_freq(self):
        f = np.array([0.0, 1.0, 10.0, 100.0])
        H = (1 + 0j) * np.ones_like(f)
        with pytest.raises(ValueError, match="zero or negative"):
            prepare_ac_arrays(f, H)

    def test_rejects_non_monotonic(self):
        f = np.array([1.0, 10.0, 5.0, 100.0])
        H = (1 + 0j) * np.ones_like(f)
        with pytest.raises(ValueError, match="strictly increasing"):
            prepare_ac_arrays(f, H)

    def test_drops_non_finite(self):
        f = _log_freqs(1, 5, 100)
        H = _lpf_1pole(f, 1000.0).astype(np.complex128)
        H[5] = np.nan + 0j
        H[7] = np.inf + 0j
        f_out, H_out = prepare_ac_arrays(f, H)
        assert len(f_out) == 98
        assert np.all(np.isfinite(np.real(H_out)))
        assert np.all(np.isfinite(np.imag(H_out)))


# ---------------------------------------------------------------------------
# log_interp
# ---------------------------------------------------------------------------


class TestLogInterp:
    def test_on_sample(self):
        f = _log_freqs(0, 6, 7)  # 1, 10, 100, ..., 1e6
        v = np.arange(len(f), dtype=float)
        assert log_interp(f, v, 100.0) == pytest.approx(2.0)
        assert log_interp(f, v, 1000.0) == pytest.approx(3.0)

    def test_midpoint_in_log(self):
        f = np.array([1.0, 100.0])  # spans 2 decades
        v = np.array([0.0, 20.0])
        # geometric mid = 10Hz → halfway in log → v=10
        assert log_interp(f, v, 10.0) == pytest.approx(10.0)

    def test_clamp_to_endpoints(self):
        f = np.array([10.0, 100.0, 1000.0])
        v = np.array([1.0, 2.0, 3.0])
        assert log_interp(f, v, 1.0) == 1.0
        assert log_interp(f, v, 1e9) == 3.0

    def test_rejects_non_positive(self):
        f = np.array([1.0, 10.0])
        v = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match="positive"):
            log_interp(f, v, 0.0)

    def test_log_interp_complex(self):
        f = _log_freqs(0, 6, 500)
        H = _lpf_1pole(f, 1000.0)
        # At f=1 kHz, magnitude should be ~ -3 dB, phase ~ -45°.
        c = log_interp_complex(f, H, 1000.0)
        db = 20 * math.log10(abs(c))
        assert db == pytest.approx(-3.0103, abs=0.05)
        phase = math.degrees(math.atan2(c.imag, c.real))
        assert phase == pytest.approx(-45.0, abs=0.5)


# ---------------------------------------------------------------------------
# detect_crossings + find_crossings_any_quantity
# ---------------------------------------------------------------------------


class TestDetectCrossings:
    def test_single_crossing(self):
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        v = np.array([10.0, 5.0, -5.0, -10.0])
        c = detect_crossings(f, v, 0.0)
        assert len(c) == 1
        # crossing lies between 10 and 100 Hz (log-axis midpoint ≈ 31.6).
        assert 10.0 < c[0]["frequency_hz"] < 100.0
        assert c[0]["direction"] == "falling"

    def test_multiple_crossings(self):
        f = _log_freqs(0, 4, 500)
        v = np.sin(np.log10(f) * 2 * np.pi)  # oscillates across 0 at each decade
        c = detect_crossings(f, v, 0.0)
        assert len(c) >= 3

    def test_min_separation(self):
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        v = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
        all_ = detect_crossings(f, v, 0.0, min_separation_decades=0.0)
        thinned = detect_crossings(f, v, 0.0, min_separation_decades=2.0)
        assert len(all_) == 4
        assert len(thinned) <= 1

    def test_direction_filter(self):
        f = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([-1.0, 1.0, -1.0, 1.0])
        rising = detect_crossings(f, v, 0.0, direction="rising")
        falling = detect_crossings(f, v, 0.0, direction="falling")
        assert all(c["direction"] == "rising" for c in rising)
        assert all(c["direction"] == "falling" for c in falling)
        assert len(rising) + len(falling) == 3


class TestFindCrossingsAnyQuantity:
    def test_magnitude_db(self):
        f = _log_freqs(0, 6, 500)
        H = _lpf_1pole(f, 1000.0)
        cx, _warn = find_crossings_any_quantity(f, H, quantity="magnitude_db", level=-3.0)
        assert cx
        # The -3 dB point of a 1-pole LPF is at fc; interpolation should land ~1 kHz.
        assert cx[0]["frequency_hz"] == pytest.approx(1000.0, rel=0.02)
        assert cx[0]["units"] == "dB"

    def test_phase_warning_on_sparse(self):
        # Construct a hand-crafted H whose wrapped phase jumps more than 90°
        # between adjacent samples — simulates an under-sampled fast-phase
        # region.
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        phase_deg = np.array([0.0, -120.0, -250.0, -370.0])
        H = np.exp(1j * np.deg2rad(phase_deg))
        _cx, warn = find_crossings_any_quantity(f, H, quantity="phase_deg", level=-180.0)
        assert any("Phase changes by up to" in w for w in warn)


# ---------------------------------------------------------------------------
# gain_at_frequencies
# ---------------------------------------------------------------------------


class TestGainAt:
    def test_batch_query(self):
        f = _log_freqs(0, 6, 500)
        H = _lpf_1pole(f, 1000.0)
        points, _warn = gain_at_frequencies(f, H, [100.0, 1000.0, 10000.0])
        assert len(points) == 3
        # Approximate 1-pole LPF gains.
        assert points[0]["magnitude_db"] == pytest.approx(-0.043, abs=0.1)
        assert points[1]["magnitude_db"] == pytest.approx(-3.0103, abs=0.05)
        assert points[2]["magnitude_db"] == pytest.approx(-20.04, abs=0.1)
        # Phase wrapping to (-180, 180].
        for p in points:
            assert -180.0 < p["phase_deg"] <= 180.0

    def test_clamps_out_of_range(self):
        f = _log_freqs(0, 3, 100)
        H = _lpf_1pole(f, 100.0)
        points, warn = gain_at_frequencies(f, H, [1e-3, 1e9])
        assert any("outside sweep range" in w for w in warn)
        assert len(points) == 2

    def test_phase_interpolated_across_wrap_seam(self):
        # Three coincident poles at 1 kHz: phase = -3*atan(f/f0), crossing
        # -180° at f0*tan(60°) ≈ 1732 Hz. On a coarse 5-points/decade grid
        # the samples straddling the seam sit at -173.1° and -204.9°
        # (wrapped: +155.1°). Interpolating the WRAPPED phase averages
        # straight through ≈ -9° — ~180° of silent error exactly where
        # phase matters most. Interpolation must run on unwrapped phase.
        f = np.logspace(2, 4.4, 13)  # exact 0.2-decade steps
        f0 = 1000.0
        H = (1.0 / (1.0 + 1j * f / f0)) ** 3
        points, _warn = gain_at_frequencies(f, H, [10**3.3], include_unwrapped_phase=True)
        # Log-midpoint of the unwrapped neighbors (-173.06°, -204.88°).
        assert points[0].get("phase_deg_unwrapped") == pytest.approx(-188.97, abs=0.3)
        # Reported wrapped phase: -188.97 + 360 = +171.03, NOT ≈ -9.
        assert points[0]["phase_deg"] == pytest.approx(171.03, abs=0.3)

    def test_unwrapped_phase_included_on_request(self):
        f = _log_freqs(0, 6, 300)
        H = _lpf_1pole(f, 1000.0)
        points, _ = gain_at_frequencies(f, H, [1000.0], include_unwrapped_phase=True)
        assert "phase_deg_unwrapped" in points[0]

    def test_empty_query_rejected(self):
        f = _log_freqs(0, 3, 100)
        H = _lpf_1pole(f, 100.0)
        with pytest.raises(ValueError, match="empty"):
            gain_at_frequencies(f, H, [])


# ---------------------------------------------------------------------------
# classify_filter + compute_filter_metrics
# ---------------------------------------------------------------------------


class TestClassifyFilter:
    def test_lpf(self):
        f = _log_freqs(0, 6, 400)
        H = _lpf_1pole(f, 1000.0)
        assert classify_filter(f, magnitude_db(H)) == "lowpass"

    def test_hpf(self):
        f = _log_freqs(0, 6, 400)
        H = _hpf_2pole(f, 1000.0)
        assert classify_filter(f, magnitude_db(H)) == "highpass"

    def test_bpf(self):
        f = _log_freqs(0, 6, 400)
        H = _bpf(f, 100.0, 10000.0)
        assert classify_filter(f, magnitude_db(H)) == "bandpass"

    def test_allpass(self):
        f = _log_freqs(0, 5, 200)
        H = np.ones_like(f, dtype=np.complex128) * 2.0
        assert classify_filter(f, magnitude_db(H)) == "allpass"

    def test_truncated_lpf_sweep_is_unknown(self):
        # Sweep ends before the filter reaches a clear stopband → classifier
        # must not guess.
        f = np.logspace(0, 2.5, 200)  # 1 Hz → ~316 Hz, fc = 1 kHz
        H = _lpf_1pole(f, 1000.0)
        # Over this range the gain drops only ~3 dB — insufficient for LPF.
        assert classify_filter(f, magnitude_db(H)) in ("allpass", "unknown")

    def test_lopsided_bpf_is_unknown(self):
        # BPF whose low cutoff is below the sweep start: only one slope shows.
        f = np.logspace(3, 6, 200)  # 1 kHz → 1 MHz
        H = _bpf(f, 100.0, 10000.0)  # low cutoff at 100 Hz — off-screen
        kind = classify_filter(f, magnitude_db(H))
        # Either "unknown" or a clean LPF label; must not be "bandpass".
        assert kind != "bandpass"


class TestComputeFilterMetrics:
    def test_1pole_lpf(self):
        f = _log_freqs(0, 6, 400)
        H = _lpf_1pole(f, 1000.0)
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "lowpass"
        assert m["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.03)
        assert m["estimated_order"] == 1
        assert m["rolloff_slope_db_per_decade"] == pytest.approx(-20.0, abs=1.0)

    def test_2pole_lpf(self):
        f = _log_freqs(0, 6, 400)
        H = _lpf_2pole(f, 1000.0)
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "lowpass"
        assert m["estimated_order"] == 2
        assert m["rolloff_slope_db_per_decade"] == pytest.approx(-40.0, abs=1.0)

    def test_2pole_hpf(self):
        f = _log_freqs(0, 6, 400)
        H = _hpf_2pole(f, 1000.0)
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "highpass"
        assert m["estimated_order"] == 2

    def test_bpf_cutoffs(self):
        f = _log_freqs(0, 6, 800)
        H = _bpf(f, 100.0, 10000.0)
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "bandpass"
        assert m["cutoff_low_hz"] is not None
        assert m["cutoff_high_hz"] is not None
        # Geometric center ≈ sqrt(100*10k) = 1 kHz.
        assert m["cutoff_low_hz"] < 1000 < m["cutoff_high_hz"]

    def test_notch_with_dense_sampling(self):
        # Q=10 gives a wider notch than Q=100, so the nearest sample lands
        # deeper into the null; Q=100 needs either exact-frequency sampling
        # or 10x more points to show full rejection, which is realistic for
        # real LTspice sweeps.
        f = _log_freqs(0, 6, 4000)
        H = _notch(f, 1000.0, 10.0)
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "bandstop"
        assert m["stopband_rejection_db"] is not None
        assert m["stopband_rejection_db"] > 15

    def test_rejects_positive_ref_db(self):
        f = _log_freqs(0, 6, 100)
        H = _lpf_1pole(f, 1000.0)
        with pytest.raises(ValueError, match="negative"):
            compute_filter_metrics(f, H, ref_db=3.0)


# ---------------------------------------------------------------------------
# compute_stability_metrics
# ---------------------------------------------------------------------------


class TestStabilityMetrics:
    def test_2pole_never_crosses_minus_180(self):
        f = _log_freqs(0, 8, 500)
        H = _two_pole_loop(f, 1000.0, 1000.0, 100000.0)
        r = compute_stability_metrics(f, H)
        assert r["stability"] == "unconditional"
        assert r["gain_margin_worst_db"] is None
        assert r["phase_margin_worst_deg"] is not None

    def test_3pole_has_gain_and_phase_margins(self):
        f = _log_freqs(0, 9, 2000)
        H = _three_pole_loop(f, 10000.0, 100.0, 10000.0, 1000000.0)
        r = compute_stability_metrics(f, H)
        assert r["gain_margin_worst_db"] is not None
        assert r["phase_margin_worst_deg"] is not None

    def test_gain_below_unity_everywhere(self):
        f = _log_freqs(0, 6, 200)
        H = 0.1 * _lpf_1pole(f, 1000.0)  # max gain = -20 dB
        r = compute_stability_metrics(f, H)
        assert r["stability"] == "always_below_unity"
        assert r["phase_margin_worst_deg"] is None
        assert not r["unity_gain_crossovers"]

    def test_dc_gain_reported(self):
        f = _log_freqs(0, 6, 200)
        A = 1000.0
        H = _two_pole_loop(f, A, 1000.0, 100000.0)
        r = compute_stability_metrics(f, H)
        assert r["dc_gain_db"] == pytest.approx(20 * np.log10(A), abs=0.01)

    def test_2pole_low_phase_margin_warns_marginal(self):
        # Regression: two coincident poles at 100 Hz with modest
        # DC gain push the unity crossover well above both poles → ~20° phase
        # margin, yet the phase never crosses -180° so the loop classifies
        # "unconditional". That label must carry a marginal-ringing advisory.
        f = _log_freqs(0, 7, 4000)
        H = _two_pole_loop(f, 32.0, 100.0, 100.0)
        r = compute_stability_metrics(f, H)
        assert r["stability"] == "unconditional"
        pm = r["phase_margin_worst_deg"]
        assert pm is not None and pm < 45.0
        assert any("marginal" in w.lower() for w in r["warnings"])

    def test_2pole_high_phase_margin_no_marginal_warning(self):
        # Dominant-pole design: crossover sits just past the dominant pole, so
        # phase margin is comfortable and no marginal advisory fires.
        f = _log_freqs(0, 7, 4000)
        H = _two_pole_loop(f, 10.0, 10.0, 1_000_000.0)
        r = compute_stability_metrics(f, H)
        assert r["stability"] == "unconditional"
        pm = r["phase_margin_worst_deg"]
        assert pm is not None and pm > 45.0
        assert not any("marginal" in w.lower() for w in r["warnings"])


# ---------------------------------------------------------------------------
# compute_roll_off
# ---------------------------------------------------------------------------


class TestRollOff:
    def test_1pole_asymptote(self):
        f = _log_freqs(0, 8, 500)
        H = _lpf_1pole(f, 100.0)
        r = compute_roll_off(f, H, f_low=1e4, f_high=1e6)
        assert r["slope_db_per_decade"] == pytest.approx(-20.0, abs=0.5)
        assert r["nearest_pole_order_estimate"] == 1

    def test_2pole_asymptote(self):
        f = _log_freqs(0, 8, 500)
        H = _lpf_2pole(f, 100.0)
        r = compute_roll_off(f, H, f_low=1e4, f_high=1e6)
        assert r["slope_db_per_decade"] == pytest.approx(-40.0, abs=0.5)
        assert r["nearest_pole_order_estimate"] == 2

    def test_narrow_window_warns(self):
        f = _log_freqs(0, 8, 500)
        H = _lpf_1pole(f, 100.0)
        r = compute_roll_off(f, H, f_low=1e4, f_high=1.5e4)
        assert r["warnings"]

    def test_rejects_bad_range(self):
        f = _log_freqs(0, 8, 100)
        H = _lpf_1pole(f, 100.0)
        with pytest.raises(ValueError, match="less than"):
            compute_roll_off(f, H, f_low=1e6, f_high=1e4)

    def test_rejects_out_of_sweep(self):
        f = _log_freqs(1, 4, 100)
        H = _lpf_1pole(f, 100.0)
        with pytest.raises(ValueError, match="outside"):
            compute_roll_off(f, H, f_low=0.1, f_high=10.0)


# ---------------------------------------------------------------------------
# compute_resonances
# ---------------------------------------------------------------------------


class TestResonance:
    def test_biquad_q(self):
        f = _log_freqs(1, 5, 4000)
        H = _biquad_resonator(f, 1000.0, 10.0)
        r = compute_resonances(f, H)
        assert len(r["peaks"]) == 1
        peak = r["peaks"][0]
        assert peak["frequency_hz"] == pytest.approx(1000.0, rel=0.05)
        assert peak["q_factor"] == pytest.approx(10.0, rel=0.05)
        assert peak["bandwidth_3db_hz"] == pytest.approx(100.0, rel=0.1)

    def test_no_peak_on_lpf(self):
        f = _log_freqs(1, 5, 1000)
        H = _lpf_1pole(f, 1000.0)
        r = compute_resonances(f, H)
        assert not r["peaks"]

    def test_rejects_bad_prominence(self):
        f = _log_freqs(1, 5, 1000)
        H = _biquad_resonator(f, 1000.0, 5.0)
        with pytest.raises(ValueError, match="positive"):
            compute_resonances(f, H, min_prominence_db=-1.0)


# ---------------------------------------------------------------------------
# unwrap + magnitude helpers
# ---------------------------------------------------------------------------


class TestUnwrapPhaseSafe:
    def test_unwrap_monotonic(self):
        f = _log_freqs(0, 8, 400)
        H = _three_pole_loop(f, 1000.0, 100.0, 10000.0, 1000000.0)
        p, _ = unwrap_phase_safe(H)
        # Phase should decrease monotonically from ~0° to ~-270°.
        assert p[0] > p[-1]
        assert p[-1] < -200

    def test_warning_on_sparse(self):
        phase_deg = np.array([0.0, -120.0, -250.0, -370.0])
        H = np.exp(1j * np.deg2rad(phase_deg))
        _, w = unwrap_phase_safe(H)
        assert any("unwrap" in msg.lower() for msg in w)


class TestMagnitudeDb:
    def test_zero_is_floored(self):
        H = np.array([0.0 + 0j, 1 + 0j])
        db = magnitude_db(H)
        assert np.isfinite(db).all()


class TestStabilityPhaseWarning:
    """stability_metrics warns when DC phase is near ±180°."""

    def test_warns_on_inverting_output(self):
        # Synthesize a CS-amp-style transfer: gain 1000 at DC, single pole at
        # 1 kHz, BUT with a sign inversion (phase starts at 180°).
        freqs = np.logspace(0, 7, 200)
        omega = 2 * np.pi * freqs
        H_lp = 1000 / (1 + 1j * omega / (2 * np.pi * 1e3))
        H = -H_lp  # inversion → phase starts at +180°
        out = compute_stability_metrics(freqs, H)
        assert any("doesn't look like a loop-gain probe" in w for w in out["warnings"])

    def test_silent_on_loop_probe(self):
        # Standard loop probe: phase starts at 0°.
        freqs = np.logspace(0, 7, 200)
        omega = 2 * np.pi * freqs
        H = 1000 / (1 + 1j * omega / (2 * np.pi * 1e3))
        out = compute_stability_metrics(freqs, H)
        assert not any("doesn't look like a loop-gain probe" in w for w in out["warnings"])


class TestPoleOrderTolerance:
    """Pole-order estimate accepts ±3 dB/dec around an integer."""

    def test_accepts_minus_18(self):
        # A real-world miller_ota slope was -17.97 dB/dec; the earlier ±2 cutoff
        # rejected it. ±3 should accept "1" as the order.
        from ltspice_mcp.lib.ac_analysis import _estimate_order_from_slope

        assert _estimate_order_from_slope(-17.97) == 1

    def test_rejects_far_from_integer(self):
        from ltspice_mcp.lib.ac_analysis import _estimate_order_from_slope

        # -10 dB/dec is half-way between order 0 and 1 — neither is a
        # confident answer, so return None.
        assert _estimate_order_from_slope(-10.0) is None


class TestGainAtPhaseUnwrappedOmitted:
    """phase_deg_unwrapped is absent when not requested."""

    def test_omitted_by_default(self):
        freqs = np.logspace(0, 6, 100)
        omega = 2 * np.pi * freqs
        H = 1.0 / (1 + 1j * omega / (2 * np.pi * 1e3))
        points, _ = gain_at_frequencies(freqs, H, [100.0, 1e4])
        for p in points:
            assert "phase_deg_unwrapped" not in p

    def test_present_when_requested(self):
        freqs = np.logspace(0, 6, 100)
        omega = 2 * np.pi * freqs
        H = 1.0 / (1 + 1j * omega / (2 * np.pi * 1e3))
        points, _ = gain_at_frequencies(freqs, H, [100.0, 1e4], include_unwrapped_phase=True)
        for p in points:
            assert "phase_deg_unwrapped" in p
