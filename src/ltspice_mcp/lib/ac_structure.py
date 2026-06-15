"""Structural reading of a Bode response: poles, zeros, order, and phase facts.

Given a complex transfer function ``H(jw)`` over a log-spaced frequency sweep,
this module extracts the design-relevant *structure* of the response — the net
high-frequency pole-zero excess, the located corner frequencies (real poles,
real zeros, complex pole/zero pairs), the low-frequency asymptote (so an
integrator / type-2 / type-3 compensator reads correctly), and two facts the
magnitude alone cannot reveal: whether the system is non-minimum-phase (an
RHP zero or transport delay) and any pure transport delay.

It surfaces FACTS, not verdicts (see ``lib/result_observations.py`` for the
doctrine): every located corner carries a frequency RANGE rather than a single
exact number, closely-spaced features are flagged as merged rather than split
into invented precise corners, and an observation always recommends checking
the result against the actual Bode plot.

Two complementary strategies are combined behind one entry point:

  - **rational fit** — a Levy least-squares fit of a rational ``B(s)/A(s)``
    with Sanathanan-Koerner reweighting, order chosen automatically. When the
    fit reproduces the sweep tightly its roots give the exact poles and zeros,
    so they are used directly.
  - **asymptotic reading** — when no low-order rational fit is faithful (e.g. a
    transport delay, which is not rational), fall back to reading the structure
    off the Bode asymptotes: breakpoints from the magnitude-slope staircase, a
    joint gain-phase corner reading, group delay for resonances and delay, and a
    gain-phase consistency (Bode) residual for non-minimum-phase content.

The single public function is :func:`analyze_ac_structure`.

Phase is differentiated only after going through ``unwrap_phase_safe`` — never
raw ``np.angle`` differencing, whose wrapped ±180° seam silently corrupts any
derivative (the classic naive-Bode failure).
"""

from __future__ import annotations

from itertools import pairwise
from typing import Literal, TypedDict

import numpy as np
from scipy.signal import find_peaks

from ltspice_mcp.lib.ac_analysis import (
    _slope_db_per_decade,  # pyright: ignore[reportPrivateUsage]  # shared slope primitive
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.raw_parser import safe_magnitude_db

# ---------------------------------------------------------------------------
# Public result types
# ---------------------------------------------------------------------------

CornerKind = Literal["real_pole", "real_zero", "complex_pair", "complex_zero_pair"]


class Corner(TypedDict):
    """One located corner of the response.

    ``f_lo``/``f_hi`` bracket the frequency range over which the corner sits
    (a point estimate sets ``f_lo == f_hi``). ``q`` is the resonance quality
    factor and is non-null only for a ``complex_pair``. ``merged`` is True when
    the corner is an under-resolved cluster of closely-spaced features whose
    individual frequencies could not be separated from this sweep — then
    ``[f_lo, f_hi]`` brackets the whole cluster.
    """

    kind: CornerKind
    f_lo: float
    f_hi: float
    q: float | None
    merged: bool


class Observation(TypedDict):
    """One surfaced fact about how the structure was read (not a verdict)."""

    code: str
    detail: str


class AcStructureResult(TypedDict):
    """Structured facts read from a Bode response by :func:`analyze_ac_structure`."""

    net_order: int | None
    corners: list[Corner]
    integrator: bool
    lf_slope_db_per_decade: float
    non_minimum_phase: bool
    phase_residual_deg: float | None
    transport_delay_s: float | None
    method: str
    fit_rel_err: float | None
    observations: list[Observation]


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Relative fit error below which the rational fit's roots are trusted directly.
_FIT_TRUST_REL_ERR = 1e-2

# One real pole/zero contributes 20 dB/dec of magnitude slope.
_DB_PER_ORDER = 20.0
# Half-decade smoothing window (in decades) for the running magnitude slope.
_SLOPE_WINDOW_DEC = 0.5
# A net magnitude-slope change below this (dB/dec) is noise, not a breakpoint.
_STEP_MIN_DB = 8.0
# A slope sample within this of an integer multiple of 20 dB/dec counts as
# sitting on an asymptotic plateau.
_PLATEAU_TOL_DB = 5.0
# Local slope-derivative (dB/dec per sample) below this counts as "flat".
_FLAT_DSLOPE = 1.2
# Two same-sign transitions whose bands lie within this many decades are one
# (merged) corner — closely-spaced poles/zeros, doublets.
_MERGE_GAP_DEC = 0.35
# A plateau must span at least this many decades to count; rejects the flat apex
# of a high-Q resonance overshoot, which would otherwise split a pair in two.
_PLATEAU_MIN_DEC = 0.2

# A complex pair's phase swing is ±180°; a first-order corner ±90°. Classify a
# phase lobe as a pair above this midpoint.
_PAIR_PHASE_DEG = 135.0
# Minimum phase swing (deg) for a lobe to count as a corner at all.
_MIN_LOBE_DEG = 25.0
# Local magnitude-slope step (dB/dec) required to confirm a minimum-phase zero.
_ZERO_MAG_RISE_DB = 10.0
# Minimum local phase rise (deg) to call the phase "recovering" at a zero.
_ZERO_PHASE_RISE_DEG = 30.0

# Group-delay peak prominence as a fraction of the robust scale of tau_g.
_GD_PEAK_PROMINENCE_FRAC = 0.15
# Half-height fraction used to bracket a group-delay peak.
_GD_WIDTH_REL_HEIGHT = 0.5
# HF window fraction scanned for a flat transport-delay floor.
_GD_HF_FRAC = 0.15
# np.gradient leaves a one-sided edge artifact; trim this many end samples.
_GD_EDGE_TRIM = 3
# Floor flatness: this fraction of HF samples must sit within the band below.
_GD_FLOOR_BAND = 0.25
_GD_FLOOR_FLAT_FRAC = 0.6
# tau_g below this (s) is treated as no transport delay (a decaying 1/w tail).
_GD_FLOOR_MIN_TAU_S = 1e-7
# Minimum Q for a group-delay peak to be promoted to a complex pair the
# breakpoint reader missed.
_GD_RESONANCE_MIN_Q = 0.7

# Fraction trimmed off each end of the gain-phase residual band (the Bode-kernel
# convolution rings where the magnitude slope past the band edge is unknown).
_RESID_TRIM_FRAC = 0.15
# Phase residual (deg) above which the response is non-minimum-phase.
_RESID_NMP_THRESHOLD_DEG = 15.0
# Uniform-grid size for the Bode-kernel convolution.
_RESID_UNIFORM_N = 4096
# A transport delay makes the residual grow linearly with frequency; flag it
# when a straight-line fit explains at least this much of the variance.
_RESID_DELAY_R2_MIN = 0.90

# A root with imaginary part above this fraction of its magnitude is a complex
# (conjugate-pair) root; a root with real part above this fraction is in the
# right half-plane (non-minimum-phase).
_ROOT_COMPLEX_FRAC = 0.02

# Corners (by geometric center) within this many decades are the same feature
# when fusing the asymptotic readers.
_MERGE_DEC = 0.25

# Low-frequency asymptote: measure the magnitude slope over the lowest this many
# decades of the sweep.
_LF_WINDOW_DEC = 0.5
# Slope at or below this (dB/dec) is integrator-like (a pole at the origin).
_LF_INTEGRATOR_SLOPE_DB = -10.0
# ...corroborated by the low-frequency phase sitting within this of -90°.
_LF_INTEGRATOR_PHASE_TOL_DEG = 35.0


# ---------------------------------------------------------------------------
# Low-frequency asymptote (integrator detection)
# ---------------------------------------------------------------------------


def _lf_asymptote(
    freqs: np.ndarray, mag_db: np.ndarray, phase_deg: np.ndarray
) -> tuple[float, bool]:
    """Magnitude slope (dB/dec) over the lowest part of the sweep, and whether
    it looks like a pole at the origin (integrator).

    A type-2 / type-3 compensator has an integrator: its magnitude falls at
    ~-20 dB/dec at the low end with the phase near -90°. Reading the
    low-frequency asymptote makes that structure visible — without it the
    response just looks like it starts mid-roll-off.
    """
    f_lo = float(freqs[0])
    f_hi = min(f_lo * 10.0**_LF_WINDOW_DEC, float(freqs[-1]))
    if f_hi <= f_lo * 1.05:
        # Sweep too short to read an asymptote; fall back to the first two points.
        slope = float((mag_db[1] - mag_db[0]) / (np.log10(freqs[1]) - np.log10(freqs[0])))
    else:
        slope = _slope_db_per_decade(freqs, mag_db, f_lo, f_hi)

    lf_mask = freqs <= f_hi
    lf_phase = float(np.median(phase_deg[lf_mask])) if lf_mask.any() else float(phase_deg[0])
    phase_near_minus_90 = abs(lf_phase + 90.0) <= _LF_INTEGRATOR_PHASE_TOL_DEG
    integrator = bool(slope <= _LF_INTEGRATOR_SLOPE_DB and phase_near_minus_90)
    return slope, integrator


# ---------------------------------------------------------------------------
# Asymptotic-slope breakpoint reader (magnitude staircase)
# ---------------------------------------------------------------------------


class _Breakpoint(TypedDict):
    kind: CornerKind
    f_lo: float
    f_hi: float
    order: int
    merged: bool
    resonant: bool


def _running_slope(log_f: np.ndarray, mag_db: np.ndarray, half_win_dec: float) -> np.ndarray:
    """Local magnitude slope (dB/decade) at each sample.

    Least-squares line fit of ``mag_db`` over a ±``half_win_dec`` window in
    log-frequency around each point — the smoothed-local generalization of the
    two-point :func:`ac_analysis._slope_db_per_decade`. Endpoint NaNs are
    filled from the nearest interior fit so the array stays full length.
    """
    n = log_f.size
    slope = np.full(n, np.nan)
    for i in range(n):
        mask = (log_f >= log_f[i] - half_win_dec) & (log_f <= log_f[i] + half_win_dec)
        if np.count_nonzero(mask) >= 2:
            slope[i] = np.polyfit(log_f[mask], mag_db[mask], 1)[0]
    valid = np.where(np.isfinite(slope))[0]
    if valid.size:
        slope[: valid[0]] = slope[valid[0]]
        slope[valid[-1] + 1 :] = slope[valid[-1]]
    return slope


def _slope_plateaus(slope: np.ndarray, log_f: np.ndarray) -> list[dict[str, float]]:
    """Segment the magnitude-slope curve into sustained asymptotic plateaus.

    A plateau is a run of samples that are locally flat AND near an integer
    multiple of 20 dB/dec. Everything between two plateaus is one breakpoint.
    This is robust to a high-Q resonance, which is an overshoot of the running
    slope WITHIN one transition band — its net step is (right - left plateau),
    so the peak never spawns a spurious extra zero+pole.
    """
    dslope = np.abs(np.diff(slope, prepend=slope[0]))

    def is_plateau(k: int) -> bool:
        lvl = slope[k]
        near_int = abs(lvl - _DB_PER_ORDER * round(lvl / _DB_PER_ORDER)) <= _PLATEAU_TOL_DB
        return bool(near_int and dslope[k] <= _FLAT_DSLOPE)

    n = slope.size
    plat = np.array([is_plateau(k) for k in range(n)])

    plateaus: list[dict[str, float]] = []
    k = 0
    while k < n:
        if not plat[k]:
            k += 1
            continue
        j = k
        while j < n and plat[j]:
            j += 1
        span_dec = log_f[j - 1] - log_f[k]
        # Always keep the first/last plateau (DC and HF asymptotes anchor the
        # staircase even if the sweep clips them short); reject short interior
        # plateaus, which are resonance-overshoot apexes.
        is_edge = k == 0 or j >= n
        if span_dec >= _PLATEAU_MIN_DEC or is_edge:
            plateaus.append(
                {"lo": float(k), "hi": float(j - 1), "level": float(np.median(slope[k:j]))}
            )
        k = j
    return plateaus


def _read_breakpoints(freqs: np.ndarray, mag_db: np.ndarray) -> tuple[int, list[_Breakpoint]]:
    """Net high-frequency order and the located breakpoints from the magnitude
    staircase. Returns ``(net_order, breakpoints)``.
    """
    log_f = np.log10(freqs)
    slope = _running_slope(log_f, mag_db, _SLOPE_WINDOW_DEC / 2.0)

    hf_slope = float(np.median(slope[-max(3, slope.size // 20) :]))
    net_order = round(-hf_slope / _DB_PER_ORDER)

    plateaus = _slope_plateaus(slope, log_f)

    # Each consecutive pair of plateaus brackets one breakpoint.
    steps: list[dict[str, float | bool]] = []
    for a, b in pairwise(plateaus):
        height = b["level"] - a["level"]
        if abs(height) < _STEP_MIN_DB:
            continue
        steps.append(
            {
                "f_lo": float(freqs[int(a["hi"])]),
                "f_hi": float(freqs[int(b["lo"])]),
                "height": height,
                "sign": 1.0 if height > 0 else -1.0,
                "merged": False,
            }
        )

    # Merge closely-spaced same-sign transitions (doublet / cluster blur).
    merged: list[dict[str, float | bool]] = []
    for st in steps:
        if merged:
            prev = merged[-1]
            same_sign = prev["sign"] == st["sign"]
            gap = np.log10(float(st["f_lo"])) - np.log10(float(prev["f_hi"]))
            if same_sign and gap < _MERGE_GAP_DEC:
                prev["f_hi"] = st["f_hi"]
                prev["height"] = float(prev["height"]) + float(st["height"])
                prev["merged"] = True
                continue
        merged.append(dict(st))

    breakpoints: list[_Breakpoint] = []
    for st in merged:
        height = float(st["height"])
        order = max(round(abs(height) / _DB_PER_ORDER), 1)
        is_pole = float(st["sign"]) < 0
        if order >= 2:
            kind: CornerKind = "complex_pair"
        elif is_pole:
            kind = "real_pole"
        else:
            kind = "real_zero"
        f_lo = float(st["f_lo"])
        f_hi = float(st["f_hi"])
        i0 = int(np.argmin(np.abs(freqs - f_lo)))
        i1 = int(np.argmin(np.abs(freqs - f_hi)))
        band_peak = float(mag_db[i0 : i1 + 1].max()) if i1 > i0 else float(mag_db[i0])
        resonant = bool(is_pole and band_peak > max(mag_db[i0], mag_db[i1]) + 1.0)
        breakpoints.append(
            {
                "kind": kind,
                "f_lo": f_lo,
                "f_hi": f_hi,
                "order": order,
                "merged": bool(st["merged"]),
                "resonant": resonant,
            }
        )
    return net_order, breakpoints


# ---------------------------------------------------------------------------
# Joint gain-phase corner reader (Q from the phase slope)
# ---------------------------------------------------------------------------


def _q_from_phase_slope(slope_deg_per_log10f: float) -> float:
    """Q of a 2nd-order section from its peak phase slope at f0.

    For a 2nd-order section the phase slope at resonance is
    ``dphi/d(ln w)|_w0 = -2 Q`` (radians per natural-log frequency unit). The
    measured slope is in deg per decade (deg per ``log10 f``); convert and halve.
    """
    rad_per_lnw = abs(slope_deg_per_log10f) * (np.pi / 180.0) / np.log(10.0)
    return float(rad_per_lnw / 2.0)


def _phase_lobes(x: np.ndarray, dphi: np.ndarray) -> list[tuple[int, int, int, float]]:
    """Segment the phase-slope curve into sign-consistent lobes.

    Returns ``(i_start, i_end, sign, total_phase_deg)`` per lobe. ``total`` is
    the integral of ``dphi`` over the lobe = the phase contributed by that
    section (signed). Near-zero slope is treated as flat passband belonging to
    no lobe, so well-separated corners do not merge across it.
    """
    sign = np.sign(dphi)
    flat = np.abs(dphi) < (np.max(np.abs(dphi)) * 0.02 + 1e-12)
    sign[flat] = 0

    lobes: list[tuple[int, int, int, float]] = []
    i = 0
    n = len(dphi)
    while i < n:
        if sign[i] == 0:
            i += 1
            continue
        s = sign[i]
        j = i
        while j + 1 < n and sign[j + 1] == s:
            j += 1
        seg_x = x[i : j + 1]
        total = float(np.trapezoid(dphi[i : j + 1], seg_x)) if len(seg_x) > 1 else 0.0
        lobes.append((i, j, int(s), total))
        i = j + 1
    return lobes


def _f0_at_half_swing(x: np.ndarray, phi: np.ndarray, i0: int, i1: int, total_deg: float) -> float:
    """Frequency where the local phase reaches half its lobe swing.

    Within a lobe the phase moves monotonically by ``total_deg``; f0 is where
    it crosses the midpoint, found by log-interpolated crossing detection so the
    answer matches what is read off a Bode plot. Robust at any Q (a pair's f0
    is its -90° point even when Q ≤ 0.5 leaves no magnitude peak).
    """
    target = phi[i0] + total_deg / 2.0
    seg = phi[i0 : i1 + 1]
    xs = x[i0 : i1 + 1]
    d = seg - target
    crossings = np.where(np.diff(np.sign(d)) != 0)[0]
    if crossings.size == 0:
        k = int(np.argmin(np.abs(d)))
        return float(10.0 ** xs[k])
    k = int(crossings[0])
    y0, y1 = seg[k], seg[k + 1]
    xc = xs[k] if y1 == y0 else xs[k] + (target - y0) / (y1 - y0) * (xs[k + 1] - xs[k])
    return float(10.0**xc)


def _read_corner_q(freqs: np.ndarray, H: np.ndarray) -> list[tuple[float, float]]:
    """Joint gain-phase reading of complex-pair corners: ``(f0_hz, Q)`` per pair.

    Uses the unwrapped phase as the primary evidence. A complex pole pair shows
    a -180° phase swing whose -90° point is f0 and whose steepness gives Q. This
    sees a pair even when Q is low enough that the magnitude has no peak.
    """
    mag = safe_magnitude_db(H)
    phi, _ = unwrap_phase_safe(H)
    x = np.log10(freqs)
    dphi = np.gradient(phi, x)
    dmag = np.gradient(mag, x)

    pairs: list[tuple[float, float]] = []
    for i0, i1, sgn, total in _phase_lobes(x, dphi):
        swing = abs(total)
        if swing < _MIN_LOBE_DEG or swing < _PAIR_PHASE_DEG:
            continue
        f0 = _f0_at_half_swing(x, phi, i0, i1, total)
        seg_slope = dphi[i0 : i1 + 1]
        peak_slope = float(seg_slope[np.argmax(np.abs(seg_slope))])
        q = _q_from_phase_slope(peak_slope)
        if sgn < 0:
            # Falling phase => pole pair.
            pairs.append((f0, q))
        else:
            # Rising phase => zero-pair candidate; confirm via a +20 dB/dec
            # local magnitude-slope step (an RHP zero drops phase and never
            # reaches this branch).
            w = max(2, (i1 - i0) // 2 or 2)
            pre = dmag[max(0, i0 - w) : i0 + 1]
            post = dmag[i1 : min(len(dmag), i1 + w + 1)]
            step = float(np.median(post) - np.median(pre)) if pre.size and post.size else 0.0
            if step >= _ZERO_MAG_RISE_DB and total >= _ZERO_PHASE_RISE_DEG:
                pairs.append((f0, q))
    return pairs


# ---------------------------------------------------------------------------
# Group delay (resonances + transport-delay floor)
# ---------------------------------------------------------------------------


class _GroupDelayPeak(TypedDict):
    f_lo: float
    f_hi: float
    f0: float
    q: float


def _group_delay(freqs: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Group delay ``tau_g(f) = -d(phase)/d(omega)`` in seconds.

    Phase comes from :func:`unwrap_phase_safe` (mandatory); differentiation is
    against the non-uniform ``omega = 2*pi*f`` axis.
    """
    phase_rad = np.deg2rad(unwrap_phase_safe(H)[0])
    omega = 2.0 * np.pi * freqs
    return -np.gradient(phase_rad, omega)


def _delay_floor(freqs: np.ndarray, tau_g: np.ndarray) -> float | None:
    """The flat, nonzero HF tail of ``tau_g`` (a pure transport delay), or None.

    After every finite pole/zero contribution dies out, a constant ``tau_g`` is
    a transport delay ``exp(-s*tau)``. Read it from the top of the sweep,
    trimming the gradient edge artifact, and require it to be genuinely flat.
    """
    n = len(tau_g)
    hf_n = max(5, round(_GD_HF_FRAC * n))
    end = n - _GD_EDGE_TRIM if n - _GD_EDGE_TRIM > hf_n else n
    hf = tau_g[end - hf_n : end]
    if hf.size < 3:
        return None
    floor = float(np.median(hf))
    if floor <= _GD_FLOOR_MIN_TAU_S:
        return None
    within = float(np.mean(np.abs(hf - floor) <= _GD_FLOOR_BAND * abs(floor)))
    if within < _GD_FLOOR_FLAT_FRAC:
        return None
    return floor


def _group_delay_peaks(
    freqs: np.ndarray, tau_g: np.ndarray, floor: float | None
) -> list[_GroupDelayPeak]:
    """Interior group-delay peaks (resonances), each with an inferred Q.

    A peak in ``tau_g`` marks where the dynamics concentrate; at a 2nd-order
    resonance ``tau_g(f0) ≈ 2Q/omega0``. This sees a high-Q pair even when its
    magnitude bump is small, and reads an all-pass (flat magnitude, all the
    action in the phase). The delay floor is subtracted first.
    """
    tau_dyn = tau_g - floor if floor is not None else tau_g
    baseline = float(np.median(tau_dyn))
    scale = float(np.median(np.abs(tau_dyn - baseline))) or float(np.std(tau_dyn)) or 1.0
    prominence = _GD_PEAK_PROMINENCE_FRAC * max(
        scale, abs(baseline) * _GD_PEAK_PROMINENCE_FRAC, 1e-12
    )
    peak_idx, _ = find_peaks(tau_dyn, prominence=prominence)

    peaks: list[_GroupDelayPeak] = []
    for idx in peak_idx:
        f0 = float(freqs[idx])
        tau_above = float(tau_dyn[idx])
        q = tau_above * (2.0 * np.pi * f0) / 2.0
        peak = tau_dyn[idx]
        level = baseline + _GD_WIDTH_REL_HEIGHT * (peak - baseline)
        lo = int(idx)
        while lo > 0 and tau_dyn[lo] > level:
            lo -= 1
        hi = int(idx)
        while hi < len(tau_dyn) - 1 and tau_dyn[hi] > level:
            hi += 1
        peaks.append({"f_lo": float(freqs[lo]), "f_hi": float(freqs[hi]), "f0": f0, "q": q})
    return peaks


# ---------------------------------------------------------------------------
# Gain-phase consistency (Bode) residual: non-minimum-phase + transport delay
# ---------------------------------------------------------------------------


def _min_phase_deg(freqs: np.ndarray, H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict the minimum-phase phase (deg) from ``|H|`` on a trimmed mid-band.

    A minimum-phase system's phase is fixed by its log-magnitude through the
    Bode gain-phase relation: a convolution of the magnitude slope
    ``d(ln|H|)/d(ln w)`` with the weighting kernel ``W(u) = ln|coth(u/2)|``.
    (The plain Hilbert kernel is only the high-frequency approximation and is
    quantitatively wrong even on a single pole, so it is not used.) Returns the
    swept frequencies and predicted phase, both trimmed to the artifact-free
    interior where the convolution does not ring.
    """
    # Floor zeros exactly as safe_magnitude_db does (ln|H| = dB/20 · ln10) so an
    # exact null — a notch sampled at its null, or an all-zero degenerate trace —
    # can't drive log to -inf and poison the residual with NaN/inf.
    ln_mag = safe_magnitude_db(H) * (np.log(10.0) / 20.0)
    x = np.log(freqs)

    x_uni = np.linspace(x[0], x[-1], _RESID_UNIFORM_N)
    du = x_uni[1] - x_uni[0]
    lm_uni = np.interp(x_uni, x, ln_mag)
    slope = np.gradient(lm_uni, du)

    t = du * np.arange(-_RESID_UNIFORM_N, _RESID_UNIFORM_N + 1)
    with np.errstate(divide="ignore"):
        kernel = np.log(np.abs(1.0 / np.tanh(t / 2.0)))
    kernel[~np.isfinite(kernel)] = 0.0

    conv = np.convolve(slope, kernel)
    start = (len(conv) - _RESID_UNIFORM_N) // 2
    min_phase_uni = conv[start : start + _RESID_UNIFORM_N] * du / np.pi
    min_phase_deg = np.rad2deg(np.interp(x, x_uni, min_phase_uni))

    n = len(freqs)
    lo = int(n * _RESID_TRIM_FRAC)
    hi = n - lo
    return freqs[lo:hi], min_phase_deg[lo:hi]


def _gain_phase_residual(freqs: np.ndarray, H: np.ndarray) -> tuple[float, bool, float | None]:
    """Non-minimum-phase reading from the gain-phase consistency residual.

    Returns ``(peak_residual_deg, is_nmp, transport_delay_s)``. The residual is
    the measured (unwrapped) phase minus the minimum-phase prediction, de-meaned
    (the Bode relation pins phase only up to a constant). A large residual means
    excess phase lag the magnitude cannot account for — a right-half-plane zero,
    an all-pass, or a transport delay. A residual growing linearly with
    frequency is the transport-delay signature; its slope gives ``tau``.
    """
    meas_phase = unwrap_phase_safe(H)[0]
    f_mid, min_phase_mid = _min_phase_deg(freqs, H)
    n = len(freqs)
    lo = int(n * _RESID_TRIM_FRAC)
    hi = n - lo
    residual = meas_phase[lo:hi] - min_phase_mid
    residual = residual - np.mean(residual)

    peak_residual = float(np.max(np.abs(residual)))
    if not np.isfinite(peak_residual):  # belt-and-suspenders: never emit NaN/inf
        return 0.0, False, None
    is_nmp = peak_residual > _RESID_NMP_THRESHOLD_DEG

    tau_s: float | None = None
    if is_nmp and len(f_mid) >= 3:
        a, b = np.polyfit(f_mid, residual, 1)
        fit = a * f_mid + b
        ss_res = float(np.sum((residual - fit) ** 2))
        ss_tot = float(np.sum((residual - np.mean(residual)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        tau = -a / 360.0
        if r2 >= _RESID_DELAY_R2_MIN and tau > 0:
            tau_s = float(tau)
    return peak_residual, is_nmp, tau_s


# ---------------------------------------------------------------------------
# Rational fit (Levy least-squares + Sanathanan-Koerner reweighting)
# ---------------------------------------------------------------------------


class _RationalFit(TypedDict):
    net_order: int
    corners: list[Corner]
    non_minimum_phase: bool
    fit_rel_err: float


def _fit_levy_sk(
    s: np.ndarray, H: np.ndarray, nb: int, na: int, niter: int = 5
) -> tuple[np.ndarray, np.ndarray]:
    """Levy least-squares fit of ``B(s)/A(s)`` (degrees ``nb``/``na``, ``A``
    normalized at DC) with Sanathanan-Koerner reweighting.

    Returns ``(b, a)`` highest-degree-first (numpy.roots convention). ``s`` is
    assumed pre-scaled so the normal equations stay well-conditioned.
    """
    Sb = np.vander(s, nb + 1, increasing=True)
    Sa = np.vander(s, na + 1, increasing=True)[:, 1:]
    w = np.ones_like(s, dtype=float)
    b = np.zeros(nb + 1, dtype=complex)
    a_full = np.zeros(na + 1, dtype=complex)
    for _ in range(max(1, niter)):
        m = np.hstack([Sb, -(H[:, None] * Sa)]) * w[:, None]
        rhs = H * w
        a_re = np.vstack([np.real(m), np.imag(m)])
        y_re = np.concatenate([np.real(rhs), np.imag(rhs)])
        coef, *_ = np.linalg.lstsq(a_re, y_re, rcond=None)
        b = coef[: nb + 1]
        a_full = np.concatenate([[1.0], coef[nb + 1 :]])
        a_eval = np.polyval(a_full[::-1], s)
        w = 1.0 / np.maximum(np.abs(a_eval), 1e-12)
    return b[::-1], a_full[::-1]


def _rel_err(s: np.ndarray, H: np.ndarray, b: np.ndarray, a: np.ndarray) -> float:
    """Relative L2 error of the fit ``B(s)/A(s)`` against ``H``."""
    h_fit = np.polyval(b, s) / np.polyval(a, s)
    return float(np.linalg.norm(h_fit - H) / max(np.linalg.norm(H), 1e-30))


def _roots_to_corners(
    roots: np.ndarray, w0: float, kind_complex: CornerKind, kind_real: CornerKind
) -> tuple[list[Corner], bool]:
    """Convert fitted roots (in scaled ``s``) to located corners in Hz.

    A conjugate pair de-duplicates to one ``complex_pair`` corner. Returns the
    corners plus whether any root lies in the right half-plane (non-minimum-phase
    for a zero; an unstable pole otherwise).
    """
    corners: list[Corner] = []
    seen: list[complex] = []
    any_rhp = False
    for raw in roots:
        r = complex(raw)
        if any(abs(r - c) < 1e-6 * max(abs(r), 1.0) for c in seen):
            continue
        is_pair = abs(r.imag) > _ROOT_COMPLEX_FRAC * abs(r)
        seen.append(r.conjugate())
        mag = abs(r) * w0
        if mag < 1e-9:
            continue
        if r.real > _ROOT_COMPLEX_FRAC * abs(r):
            any_rhp = True
        f_hz = mag / (2.0 * np.pi)
        kind = kind_complex if is_pair else kind_real
        q = (mag / (2.0 * abs(r.real) * w0)) if is_pair and abs(r.real) > 0 else None
        corners.append({"kind": kind, "f_lo": f_hz, "f_hi": f_hz, "q": q, "merged": False})
    return corners, any_rhp


def _rational_fit(freqs: np.ndarray, H: np.ndarray) -> _RationalFit | None:
    """Auto-order rational fit of ``H``; ``None`` if no fit could be formed.

    Frequency is scaled by the geometric-mean ``omega`` before fitting (a high
    power of ``s`` at MHz overflows the normal equations otherwise) and the
    roots are rescaled back. Order is chosen by a relative-error scan with a
    small per-order penalty, so the fitter is not told the order in advance.
    """
    w = 2.0 * np.pi * freqs
    w0 = float(np.exp(np.mean(np.log(w))))
    s = 1j * w / w0

    best: tuple[float, float, int, int, np.ndarray, np.ndarray] | None = None
    for na in range(1, 7):
        for nb in range(0, na + 1):
            try:
                b, a = _fit_levy_sk(s, H, nb, na)
            except np.linalg.LinAlgError:
                continue
            err = _rel_err(s, H, b, a)
            score = err + 0.01 * (na + nb)
            if best is None or score < best[0]:
                best = (score, err, nb, na, b, a)
    if best is None:
        return None
    _, err, nb, na, b, a = best

    poles_raw = np.roots(a)
    zeros_raw = np.roots(b) if len(b) > 1 else np.array([])
    net_order = len(poles_raw) - len(zeros_raw)

    pole_corners, pole_rhp = _roots_to_corners(poles_raw, w0, "complex_pair", "real_pole")
    # Complex ZEROS must not share the pole kind — a fitted complex zero pair
    # (e.g. a notch) is structurally distinct and must not read as a pole pair.
    zero_corners, zero_rhp = _roots_to_corners(zeros_raw, w0, "complex_zero_pair", "real_zero")
    corners = sorted(pole_corners + zero_corners, key=lambda c: c["f_lo"])
    return {
        "net_order": net_order,
        "corners": corners,
        "non_minimum_phase": bool(pole_rhp or zero_rhp),
        "fit_rel_err": err,
    }


# ---------------------------------------------------------------------------
# Asymptotic-reading fusion
# ---------------------------------------------------------------------------


def _fuse_asymptotic(
    freqs: np.ndarray, H: np.ndarray, mag_db: np.ndarray
) -> tuple[int, list[Corner]]:
    """Fuse the asymptotic readers into ``(net_order, corners)``.

    The breakpoint reader gives the order and the corner skeleton; the joint
    gain-phase reader and group delay sharpen each complex pair's f0 and Q; a
    high-Q group-delay peak the breakpoint reader missed (e.g. two merged pairs)
    is added as an extra corner.

    A resonant Q is never attached to an under-resolved cluster: when the net
    order exceeds twice the number of located pairs, more poles than pairs are
    blurred together and a single Q would misrepresent the cluster.
    """
    net_order, breakpoints = _read_breakpoints(freqs, mag_db)
    corner_q = _read_corner_q(freqs, H)
    tau_g = _group_delay(freqs, H)
    gd_floor = _delay_floor(freqs, tau_g)
    gd_peaks = _group_delay_peaks(freqs, tau_g, gd_floor)

    located_pairs = sum(1 for bp in breakpoints if bp["kind"] == "complex_pair")
    cluster = located_pairs > 0 and net_order > 2 * located_pairs

    corners: list[Corner] = []
    for bp in breakpoints:
        f_center = float(np.sqrt(bp["f_lo"] * bp["f_hi"]))
        q: float | None = None
        if bp["kind"] == "complex_pair" and not cluster:
            for f0, q_src in corner_q:
                if abs(np.log10(f0 / f_center)) < _MERGE_DEC:
                    q = q_src
                    break
            if q is None:
                for peak in gd_peaks:
                    pc = float(np.sqrt(peak["f_lo"] * peak["f_hi"]))
                    if abs(np.log10(pc / f_center)) < _MERGE_DEC:
                        q = peak["q"]
                        break
        corners.append(
            {
                "kind": bp["kind"],
                "f_lo": bp["f_lo"],
                "f_hi": bp["f_hi"],
                "q": q,
                "merged": bp["merged"],
            }
        )

    # Add genuine resonances the breakpoint reader missed (e.g. merged pairs).
    for peak in gd_peaks:
        if peak["q"] < _GD_RESONANCE_MIN_Q:
            continue
        pc = float(np.sqrt(peak["f_lo"] * peak["f_hi"]))
        if any(
            abs(np.log10(pc / float(np.sqrt(c["f_lo"] * c["f_hi"])))) < _MERGE_DEC for c in corners
        ):
            continue
        corners.append(
            {
                "kind": "complex_pair",
                "f_lo": peak["f_lo"],
                "f_hi": peak["f_hi"],
                "q": None if cluster else peak["q"],
                "merged": False,
            }
        )

    corners.sort(key=lambda c: np.sqrt(c["f_lo"] * c["f_hi"]))
    return net_order, corners


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyze_ac_structure(freqs: np.ndarray, H: np.ndarray) -> AcStructureResult:
    """Read the structural facts of a Bode response.

    Args:
        freqs: Real frequency axis (Hz), strictly increasing, log-spaced. (As
            produced by an ``.AC DEC`` sweep.)
        H: Complex transfer function ``H(jw)`` at each frequency.

    Returns:
        An :class:`AcStructureResult` of surfaced facts: net high-frequency
        order, located corners (real poles/zeros, complex pairs with Q),
        whether the low-frequency asymptote is an integrator, the
        low-frequency magnitude slope, the non-minimum-phase flag with its
        gain-phase residual, any transport delay, which method produced the
        corners (``"rational_fit"`` or ``"asymptotic_reading"``), the rational
        fit's relative error when it was used, and ``observations`` (facts about
        how the structure was read — never verdicts).

    A rational fit is attempted first; if it reproduces the sweep tightly its
    roots are used directly. Otherwise the structure is read off the Bode
    asymptotes. The non-minimum-phase and transport-delay facts always come
    from the gain-phase residual and group delay, independent of which branch
    located the corners.
    """
    freqs, H = prepare_ac_arrays(freqs, H)
    mag_db = safe_magnitude_db(H)
    phase_deg, phase_warn = unwrap_phase_safe(H)

    observations: list[Observation] = []

    # Low-frequency asymptote (integrator / pole at the origin).
    lf_slope, integrator = _lf_asymptote(freqs, mag_db, phase_deg)
    if integrator:
        observations.append(
            {
                "code": "lf_integrator",
                "detail": (
                    f"Low-frequency magnitude slope {lf_slope:.0f} dB/dec with phase near "
                    "-90° — a pole at the origin (integrator), as in a type-2/type-3 "
                    "compensator. Finite corners are read above this asymptote."
                ),
            }
        )

    # Non-minimum-phase + transport delay come from phase, both branches.
    residual_deg, non_minimum_phase, transport_delay_s = _gain_phase_residual(freqs, H)
    if non_minimum_phase:
        observations.append(
            {
                "code": "non_minimum_phase",
                "detail": (
                    f"Phase lags what the magnitude alone predicts by up to {residual_deg:.0f}° "
                    "(excess lag the magnitude cannot account for) — an out-of-phase zero "
                    "or a transport delay. Achievable bandwidth is capped below where this "
                    "excess lag dominates."
                ),
            }
        )
    if transport_delay_s is not None:
        observations.append(
            {
                "code": "transport_delay",
                "detail": (
                    f"Phase residual grows linearly with frequency — a transport delay of "
                    f"~{transport_delay_s * 1e6:.1f} us (group delay floor)."
                ),
            }
        )

    # Corner structure: rational fit first, else asymptotic reading.
    fit = _rational_fit(freqs, H)
    fit_rel_err: float | None
    if fit is not None and fit["fit_rel_err"] < _FIT_TRUST_REL_ERR:
        net_order = fit["net_order"]
        corners = fit["corners"]
        method = "rational_fit"
        fit_rel_err = fit["fit_rel_err"]
        observations.append(
            {
                "code": "rational_fit",
                "detail": (
                    f"A rational B(s)/A(s) fit reproduced the sweep to a relative error of "
                    f"{fit_rel_err:.1e}; its roots give the poles and zeros directly."
                ),
            }
        )
    else:
        net_order, corners = _fuse_asymptotic(freqs, H, mag_db)
        method = "asymptotic_reading"
        # A degenerate fit (e.g. an all-zero trace) can return a non-finite
        # rel_err; surface None rather than a NaN that breaks JSON/schema.
        fit_rel_err = (
            fit["fit_rel_err"] if fit is not None and np.isfinite(fit["fit_rel_err"]) else None
        )
        observations.append(
            {
                "code": "asymptotic_reading",
                "detail": (
                    "No low-order rational fit was faithful; structure was read off the "
                    "Bode asymptotes (magnitude-slope breakpoints, joint gain-phase corner "
                    "reading, group delay, and the gain-phase consistency residual)."
                ),
            }
        )

    if any(c["merged"] for c in corners):
        observations.append(
            {
                "code": "merged_corners",
                "detail": (
                    "Closely-spaced corners could not be separated at this sweep density and "
                    "are reported as one bracketing range. Re-sweep denser around the cluster "
                    "to resolve the individual frequencies."
                ),
            }
        )

    for w in phase_warn:
        observations.append({"code": "unwrap_warning", "detail": w})

    observations.append(
        {
            "code": "review_against_plot",
            "detail": (
                "These facts are read from a finite sweep; closely-spaced features can merge "
                "and a sparse sweep can blur a corner. Review against the Bode plot."
            ),
        }
    )

    return {
        "net_order": net_order,
        "corners": corners,
        "integrator": integrator,
        "lf_slope_db_per_decade": float(lf_slope),
        "non_minimum_phase": non_minimum_phase,
        "phase_residual_deg": residual_deg,
        "transport_delay_s": transport_delay_s,
        "method": method,
        "fit_rel_err": fit_rel_err,
        "observations": observations,
    }
