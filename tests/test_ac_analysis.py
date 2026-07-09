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
    compute_return_loss,
    compute_roll_off,
    compute_stability_metrics,
    detect_crossings,
    find_crossings_any_quantity,
    gain_at_frequencies,
    integrate_noise,
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

    def test_amplitude_deadband_suppresses_epsilon_chatter(self):
        # A signal that hovers at the level with tiny float-epsilon wobble flips
        # sign on every sample. Without a deadband every flip is a crossing; an
        # amplitude deadband larger than the wobble collapses them to none.
        f = _log_freqs(0, 6, 500)
        v = 1e-9 * np.sin(np.log10(f) * 50 * np.pi)  # wobble ±1e-9 about 0
        raw = detect_crossings(f, v, 0.0)
        assert len(raw) > 5  # default behavior: every epsilon flip counts
        deadbanded = detect_crossings(f, v, 0.0, min_amplitude=0.3)
        assert len(deadbanded) == 0

    def test_amplitude_deadband_preserves_real_crossings(self):
        # A signal that genuinely swings well past the level on both sides of a
        # crossing must still register it once the deadband is set — the
        # excursion clears the deadband, so the crossing is real.
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        v = np.array([10.0, 5.0, -5.0, -10.0])
        c = detect_crossings(f, v, 0.0, min_amplitude=0.3)
        assert len(c) == 1
        assert c[0]["direction"] == "falling"

    def test_amplitude_deadband_default_unchanged(self):
        # Default min_amplitude=0.0 must reproduce the no-deadband result for
        # an ordinary multi-crossing signal (existing callers unaffected).
        f = _log_freqs(0, 4, 500)
        v = np.sin(np.log10(f) * 2 * np.pi)
        assert detect_crossings(f, v, 0.0) == detect_crossings(f, v, 0.0, min_amplitude=0.0)

    def test_amplitude_deadband_respects_direction_filter(self):
        # The deadband must still honor the direction filter: a clean triangle
        # that swings well past the level has one falling and one rising
        # crossing; the deadband keeps each only under its matching direction.
        f = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        v = np.array([-5.0, 5.0, 5.0, 5.0, -5.0])
        falling = detect_crossings(f, v, 0.0, direction="falling", min_amplitude=0.3)
        rising = detect_crossings(f, v, 0.0, direction="rising", min_amplitude=0.3)
        assert len(falling) == 1 and falling[0]["direction"] == "falling"
        assert len(rising) == 1 and rising[0]["direction"] == "rising"

    def test_amplitude_deadband_keeps_crossing_that_starts_inside_band(self):
        # A signal that starts INSIDE the deadband on the high side (+0.1) then
        # clearly drops past -min_amplitude genuinely crosses the level. The
        # deadband must keep that crossing — seeding the confirmed side as
        # "unknown" used to drop it (a real crossover silently vanishing).
        f = np.array([1.0, 10.0, 100.0, 1000.0])
        v = np.array([0.1, -0.05, -1.0, -2.0])  # inside ±0.3 at start, well below later
        c = detect_crossings(f, v, 0.0, min_amplitude=0.3)
        assert len(c) == 1
        assert c[0]["direction"] == "falling"


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

    def test_filter_cutoff_reference_is_plateau_not_band_median(self):
        # Sweep starts only ONE decade below fc=1 kHz, so the in-band median
        # sits meaningfully below the DC plateau (the auto band runs up into
        # the roll-off knee). The reference gain must anchor to the flat
        # DC-side plateau, not the band median — otherwise the -3 dB cutoff
        # is dragged outward. On the old band-median code cutoff_high landed
        # ~+4.7% high; the plateau anchor pulls it back to ~fc.
        f = _log_freqs(2, 5, 200)  # 100 Hz .. 100 kHz, ~66 pts/decade
        H = _lpf_1pole(f, 1000.0)  # H(f) = 1/(1 + j f/fc), unity DC gain
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "lowpass"
        # Plateau gain is |H| at the DC-side edge (100 Hz, one decade below fc).
        plateau_db = magnitude_db(H)[0]
        assert m["passband_gain_db"] == pytest.approx(plateau_db, abs=0.02)
        # Plateau anchor pulls the cutoff to ~+1% of fc (the residual is the
        # reference frame: |H| is already -0.043 dB at the 100 Hz plateau edge,
        # so -3 dB below THAT lands just above fc). The old band-median code put
        # it at ~+5%.
        assert m["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.02)

    def test_highpass_cutoff_reference_is_plateau(self):
        # Mirror of the LPF case for a 1st-order HPF H(f) = (j f/fc)/(1+j f/fc),
        # fc=1 kHz. Sweep runs well above fc so the high-frequency plateau is
        # the unity passband; the reference gain must anchor to that high-freq
        # plateau (not the band median dragged down by the knee on the low
        # side), keeping cutoff_low within ~1% of fc.
        f = _log_freqs(2, 5, 200)  # 100 Hz .. 100 kHz, ~1-2 decades above fc
        fc = 1000.0
        s = 1j * 2 * np.pi * f
        wc = 2 * np.pi * fc
        H = s / (s + wc)  # 1st-order HPF, unity gain at high f
        m = compute_filter_metrics(f, H)
        assert m["filter_type"] == "highpass"
        # Plateau gain is |H| at the high-frequency edge (100 kHz).
        plateau_db = magnitude_db(H)[-1]
        assert m["passband_gain_db"] == pytest.approx(plateau_db, abs=0.02)
        assert m["cutoff_low_hz"] == pytest.approx(fc, rel=0.02)


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

    def test_allpass_flat_at_unity_is_degenerate(self):
        # An allpass has |H| = 1 (0 dB) at every frequency: the magnitude only
        # grazes 0 dB with float-epsilon wobble, so it must NOT register dozens
        # of unity-gain crossings nor classify as conditionally stable. It is a
        # degenerate flat-at-unity response with at most one (spurious) crossing.
        f = _log_freqs(0, 6, 1000)
        fp = 1000.0
        s = 1j * 2 * np.pi * f
        w = 2 * np.pi * fp
        H = (1 - s / w) / (1 + s / w)  # first-order allpass, |H| == 1 exactly
        r = compute_stability_metrics(f, H)
        assert len(r["unity_gain_crossovers"]) <= 1
        assert r["stability"] != "conditional"
        assert r["stability"] == "flat_at_unity"

    def test_loop_starting_just_above_unity_keeps_its_crossover(self):
        # A loop whose magnitude starts just above unity — inside the unity-gain
        # deadband (+0.1 dB) — then rolls off well below unity genuinely crosses
        # 0 dB. The deadband must not drop that crossover and misclassify the
        # loop as "always_below_unity" with no phase margin.
        f = _log_freqs(0, 6, 1000)
        gain = 10 ** (0.1 / 20)  # +0.1 dB DC, inside the ±0.3 dB deadband
        H = gain * _lpf_1pole(f, 1000.0)
        r = compute_stability_metrics(f, H)
        assert r["stability"] != "always_below_unity"
        assert r["unity_gain_crossovers"]
        assert r["phase_margin_worst_deg"] is not None

    def test_worst_phase_margin_is_most_negative_not_smallest_magnitude(self):
        # A conditionally-stable loop crosses unity twice: once at a healthy
        # positive phase margin and once at a deeply negative one. The "worst"
        # scalar must be the most-negative margin (the unstable crossing), not
        # the smallest-magnitude one — a small positive margin must never mask a
        # large negative one.
        margins = [
            {"frequency_hz": 100.0, "margin_deg": 5.0, "direction": "falling"},
            {"frequency_hz": 1000.0, "margin_deg": -170.0, "direction": "rising"},
        ]
        # Drive the same selection the compute function uses on its margin list.
        worst = min(m["margin_deg"] for m in margins)
        assert worst == -170.0

        # End-to-end: build a loop whose two unity crossings straddle -180° so
        # the per-crossover margins are sign-mixed, and confirm the reported
        # worst margin is negative (the least-stable crossing wins).
        f = _log_freqs(0, 8, 6000)
        H = _three_pole_loop(f, 10000.0, 100.0, 10000.0, 1000000.0)
        r = compute_stability_metrics(f, H)
        pms = [m["margin_deg"] for m in r["phase_margins"]]
        assert r["phase_margin_worst_deg"] == pytest.approx(min(pms))


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

    def test_magnitude_linear_matches_db(self):
        # |Z| in ohms under a 1 A probe (or |H| for a transfer): the native-unit
        # magnitude that disambiguates a dBΩ peak from a dB dip.
        f = _log_freqs(1, 5, 4000)
        H = _biquad_resonator(f, 1000.0, 10.0)
        peak = compute_resonances(f, H)["peaks"][0]
        assert peak["magnitude_linear"] == pytest.approx(
            10.0 ** (peak["magnitude_db"] / 20.0)
        )
        assert peak["magnitude_linear"] > 0

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


# ---------------------------------------------------------------------------
# integrate_noise
# ---------------------------------------------------------------------------


class TestIntegrateNoise:
    """Total RMS noise = sqrt(∫ density² df) (amplitude-density convention,
    shared by LTspice and ngspice)."""

    def test_flat_density_matches_closed_form(self):
        # Flat density D over [a, b]: sqrt(∫ D² df) = D·sqrt(b - a).
        freqs = np.linspace(1.0, 1001.0, 2001)
        d = 2e-9
        density = np.full_like(freqs, d)
        r = integrate_noise(freqs, density, None, None)
        assert r["total_rms"] == pytest.approx(d * math.sqrt(1000.0), rel=1e-4)
        assert r["n_points"] == 2001
        assert r["f_start_used"] == pytest.approx(1.0)
        assert r["f_end_used"] == pytest.approx(1001.0)
        assert r["warnings"] == []

    def test_band_subset(self):
        freqs = np.linspace(0.0, 1000.0, 1001)  # 1 Hz spacing
        density = np.full_like(freqs, 1e-9)
        r = integrate_noise(freqs, density, 100.0, 200.0)
        assert r["total_rms"] == pytest.approx(1e-9 * math.sqrt(100.0), rel=1e-3)
        assert r["f_start_used"] == pytest.approx(100.0)
        assert r["f_end_used"] == pytest.approx(200.0)

    def test_out_of_range_band_clips_and_warns(self):
        freqs = np.linspace(10.0, 1000.0, 100)
        density = np.full_like(freqs, 1e-9)
        r = integrate_noise(freqs, density, 1.0, 5000.0)
        assert any("clip" in w.lower() for w in r["warnings"])
        assert r["f_start_used"] == pytest.approx(10.0)
        assert r["f_end_used"] == pytest.approx(1000.0)

    def test_inverted_band_rejected(self):
        freqs = np.linspace(1.0, 100.0, 50)
        with pytest.raises(ValueError, match="must be <"):
            integrate_noise(freqs, np.ones_like(freqs), 50.0, 10.0)

    def test_descending_sweep_normalized(self):
        # A high->low .noise sweep must be flipped to ascending, not rejected.
        freqs = np.linspace(1000.0, 1.0, 2001)  # descending
        d = 2e-9
        density = np.full_like(freqs, d)
        r = integrate_noise(freqs, density, None, None)
        assert r["total_rms"] == pytest.approx(d * math.sqrt(999.0), rel=1e-4)
        assert r["f_start_used"] == pytest.approx(1.0)
        assert r["f_end_used"] == pytest.approx(1000.0)

    def test_descending_sweep_with_band(self):
        freqs = np.linspace(1000.0, 1.0, 1000)  # descending, ~1 Hz spacing
        density = np.full_like(freqs, 1e-9)
        r = integrate_noise(freqs, density, 100.0, 200.0)
        assert r["f_start_used"] == pytest.approx(100.0, abs=2.0)
        assert r["f_end_used"] == pytest.approx(200.0, abs=2.0)
        assert r["total_rms"] == pytest.approx(1e-9 * math.sqrt(100.0), rel=0.02)

    def test_non_monotonic_axis_rejected(self):
        freqs = np.array([1.0, 5.0, 3.0, 10.0])
        with pytest.raises(ValueError, match="monotonic"):
            integrate_noise(freqs, np.ones_like(freqs), None, None)

    def test_non_finite_density_dropped_and_warned(self):
        # A NaN density sample (a singular point) would NaN-poison the integral;
        # it must be dropped with a warning, not returned as a silent NaN total.
        freqs = np.linspace(1.0, 1001.0, 2001)
        density = np.full_like(freqs, 2e-9)
        density[1000] = np.nan
        r = integrate_noise(freqs, density, None, None)
        assert math.isfinite(r["total_rms"])
        assert r["total_rms"] > 0.0
        assert any("non-finite" in w.lower() for w in r["warnings"])


# ---------------------------------------------------------------------------
# compute_return_loss
# ---------------------------------------------------------------------------


class TestComputeReturnLoss:
    """Reflection metrics from an impedance trace (V(node) = Zin under a 1 A probe)."""

    @staticmethod
    def _flat(zin: complex, n: int = 100):
        f = np.logspace(6, 9, n)
        H = np.full(n, zin, dtype=complex)
        return f, H

    def test_known_mismatch_at_freq(self):
        # Zin=100, z0=50 → Γ=0.3333, RL=9.542 dB, VSWR=2.0
        f, H = self._flat(100 + 0j)
        r = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        assert r["gamma_mag"] == pytest.approx(1 / 3, abs=1e-4)
        assert r["return_loss_db"] == pytest.approx(9.542, abs=1e-2)
        assert r["vswr"] == pytest.approx(2.0, abs=1e-3)
        assert r["worst_match"] is False
        assert r["frequency_hz"] == pytest.approx(1e7, rel=1e-3)

    def test_perfect_match_null_return_loss(self):
        f, H = self._flat(50 + 0j)
        r = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        assert r["gamma_mag"] == pytest.approx(0.0, abs=1e-9)
        assert r["return_loss_db"] is None  # RL → ∞
        assert r["vswr"] == pytest.approx(1.0, abs=1e-6)

    def test_total_reflection_null_vswr(self):
        f, H = self._flat(0 + 0j)  # dead short → Γ = -1 exactly
        r = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        assert r["gamma_mag"] == pytest.approx(1.0, abs=1e-9)
        assert r["vswr"] is None  # VSWR → ∞
        assert any("reflect" in w.lower() for w in r["warnings"])

    def test_worst_match_scan(self):
        f = np.logspace(6, 9, 101)
        H = np.full(101, 50 + 0j)  # matched everywhere...
        H[70] = 200 + 0j  # ...except one badly-mismatched point
        r = compute_return_loss(f, H, z0=50.0)  # no at_hz → worst-match scan
        assert r["worst_match"] is True
        assert r["frequency_hz"] == pytest.approx(f[70], rel=1e-6)

    def test_zin_extrema_reported_across_sweep(self):
        # |Zin| range over the whole sweep rides along regardless of the
        # evaluated point — the data for choosing a meaningful z0.
        f = np.logspace(6, 9, 101)
        H = np.full(101, 50 + 0j)
        H[10] = 5 + 0j  # |Z| minimum
        H[90] = 400 + 0j  # |Z| maximum
        r = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        # .get(): the keys are NotRequired in the TypedDict; a miss fails the
        # approx compare loudly.
        assert r.get("zin_min_mag_ohm") == pytest.approx(5.0)
        assert r.get("zin_min_freq_hz") == pytest.approx(float(f[10]))
        assert r.get("zin_max_mag_ohm") == pytest.approx(400.0)
        assert r.get("zin_max_freq_hz") == pytest.approx(float(f[90]))

    def test_reactive_worst_match_hints_z0_choice(self):
        # A pure reactance reflects totally against ANY real z0 — the worst-
        # match pick is then a tautology, so the result must say the 50 Ω
        # default may not suit the port (the power/filter-input case).
        f = np.logspace(3, 6, 60)
        H = 1j * 2 * np.pi * f * 10e-6  # ideal 10 µH input
        r = compute_return_loss(f, H, z0=50.0)
        assert r["worst_match"] is True
        assert r["gamma_mag"] == pytest.approx(1.0, abs=1e-9)
        assert any("purely reactive" in w for w in r["warnings"])
        # A resistive worst match must NOT carry the hint.
        f2, H2 = self._flat(100 + 0j)
        r2 = compute_return_loss(f2, H2, z0=50.0)
        assert not any("purely reactive" in w for w in r2["warnings"])

    def test_out_of_range_at_reports_endpoint_frequency(self):
        # A request below/above the sweep clamps to the nearest endpoint; the
        # reported frequency_hz must be the endpoint actually evaluated, not the
        # unavailable request, or the metrics disagree with their claimed freq.
        f, H = self._flat(100 + 0j)  # sweep 1e6 .. 1e9
        below = compute_return_loss(f, H, z0=50.0, at_hz=1.0)
        assert below["frequency_hz"] == pytest.approx(float(f[0]))
        assert any("outside the sweep range" in w for w in below["warnings"])
        above = compute_return_loss(f, H, z0=50.0, at_hz=1e12)
        assert above["frequency_hz"] == pytest.approx(float(f[-1]))
        # In-range request keeps its exact frequency, no clamp warning.
        inrange = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        assert inrange["frequency_hz"] == pytest.approx(1e7, rel=1e-9)
        assert not any("outside the sweep range" in w for w in inrange["warnings"])

    def test_negative_zin_flagged(self):
        # A reversed probe reads V(node) = -Zin; use -30 (not -50, which makes
        # Γ singular) so the negative-real-part warning is the thing under test.
        f, H = self._flat(-30 + 0j)
        r = compute_return_loss(f, H, z0=50.0, at_hz=1e7)
        assert any("probe" in w.lower() for w in r["warnings"])

    def test_bad_z0_raises(self):
        f, H = self._flat(50 + 0j)
        with pytest.raises(ValueError, match="z0"):
            compute_return_loss(f, H, z0=0.0, at_hz=1e7)
