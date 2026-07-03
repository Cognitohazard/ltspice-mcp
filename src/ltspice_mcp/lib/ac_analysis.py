"""Pure-function AC-analysis primitives for frequency-domain .raw data.

Takes numpy arrays of a frequency axis plus a complex transfer function H(f),
returns dicts of Python floats. No I/O, no spicelib dependencies. Raises
``ValueError`` with user-facing messages on domain errors — the tool layer
re-raises these as ``ResultError``.

Depends on numpy and scipy.signal.find_peaks; no other third-party code.

Convention: throughout this module

    freqs : np.ndarray      — real frequency axis, strictly increasing, Hz
    H     : np.ndarray      — complex transfer function at each freq
    mag_db: np.ndarray      — 20*log10(|H|), clamped away from -inf
    phase : np.ndarray      — UNWRAPPED phase in degrees (np.unwrap applied)

LTspice's .AC DEC/OCT sweeps are log-spaced; every interpolator in this module
operates in log10(f) space so the answers line up with what you'd read off a
Bode plot. Callers that need linear-frequency behavior should interpolate
themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from scipy.signal import find_peaks

# Re-exported as ``magnitude_db`` for readability in the AC domain; the
# implementation lives in raw_parser so AC and transient tools share the
# same clamped-log10 behavior.
from ltspice_mcp.lib.raw_parser import safe_magnitude_db as magnitude_db

# Phase sample step above which we warn about possible unwrap errors — if the
# raw wrapped phase jumps by more than this between adjacent points, unwrap
# can snap the wrong way and invert the sign of the gain margin.
_UNWRAP_WARN_STEP_DEG = 90.0

# Amplitude deadband (dB) for the unity-gain (0 dB) crossing search in
# compute_stability_metrics. A loop whose magnitude hovers at unity across the
# sweep (an allpass, or a flat-gain loop grazing 0 dB) would otherwise register
# float-epsilon sign flips as dozens of unity-gain crossings, producing a bogus
# multi-element phase-margin list and a wrong "conditional" label. The magnitude
# must swing at least this far past 0 dB between crossings for one to count; the
# value is small enough not to miss any real crossover, where the slope carries
# the magnitude well past a few tenths of a dB within one sample.
_UNITY_GAIN_DEADBAND_DB = 0.3

# True half-power point: 20·log10(1/√2) = -3.0103 dB. Use this (not a rounded
# -3.0) for "−3 dB" cutoff/bandwidth so bode_metrics(mode='filter') and the AC
# summary agree with bode_metrics(mode='crossing'/'point'), which already use the
# exact half-power level. The 0.0103 dB difference shifts a 1 kHz RC
# corner by ~0.24%.
HALF_POWER_DB = -3.010299956639812

# Shared type aliases — used by both lib function signatures and the
# pydantic input models in tools/analysis.py, so the string literals live
# in exactly one place.
Quantity = Literal["magnitude_db", "magnitude_linear", "phase_deg"]
SearchDirection = Literal["any", "rising", "falling"]
CrossingDirection = Literal["rising", "falling"]

# Minimum gain drop (dB) required to assign a definite filter class.
# Anything smaller looks like noise / a shallow shoulder and leaves the
# classifier unsure — we'd rather return "unknown" than guess.
_CLEAR_DROP_DB = 10.0

# Ratio test: how close to the peak the "passband-side" endpoint must be
# before we accept an LPF/HPF classification. Measured in multiples of
# ``flatness_db`` so tightening flatness tightens the whole classifier.
_PASSBAND_SIDE_TOL = 2.0

# When classifying a notch, if either sample adjacent to the minimum is
# more than this many dB above the sampled null, the true null likely
# falls between samples and the reported stopband rejection is a lower
# bound only — emit a warning so users know to re-sweep with denser spacing.
_NOTCH_UNDERSAMPLED_GAP_DB = 3.0

# "Interior" guard band for band-pass/band-stop classification: the peak/notch
# must sit in the middle (1 - 2·_INTERIOR_FRAC) of the swept span — i.e. between
# _INTERIOR_FRAC and 1 - _INTERIOR_FRAC of the index range. A peak hugging the
# first/last bin is far more likely a monotonic roll-off caught mid-slope
# (LPF/HPF) than a true BPF resonance, so the 10% edge guards keep BPF/BSF off
# the sweep boundaries.
_INTERIOR_FRAC = 0.1


# ---------------------------------------------------------------------------
# Loading / sanitization
# ---------------------------------------------------------------------------


def prepare_ac_arrays(freqs_raw: np.ndarray, H_raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coerce a raw-file axis and trace into clean (freqs, H) arrays.

    - Frequency axis is made real-valued (LTspice stores it as complex with
      zero imaginary part) and must be strictly increasing — the whole
      module assumes that.
    - Trace must be complex (transient traces are rejected — that's the
      caller's job on the happy path, but we also check here so unit tests
      on the pure lib fail fast).
    """
    if len(freqs_raw) != len(H_raw):
        raise ValueError(f"Axis and wave have different lengths: {len(freqs_raw)} vs {len(H_raw)}")
    if len(freqs_raw) < 3:
        raise ValueError(f"AC sweep has only {len(freqs_raw)} points; need at least 3")
    if not np.iscomplexobj(H_raw):
        raise ValueError(
            "AC analysis tools require complex transfer-function data; "
            "got real-valued samples — is this a transient result?"
        )

    freqs = np.asarray(freqs_raw)
    if np.iscomplexobj(freqs):
        freqs = np.real(freqs)
    freqs = np.asarray(freqs, dtype=float)
    H = np.asarray(H_raw, dtype=np.complex128)

    if not np.all(np.isfinite(freqs)):
        raise ValueError("Frequency axis contains non-finite values")
    if np.any(freqs <= 0):
        raise ValueError(
            "AC sweep contains zero or negative frequencies; log-scale "
            "interpolation requires f > 0"
        )
    if not np.all(np.diff(freqs) > 0):
        raise ValueError("Frequency axis is not strictly increasing")

    finite_mask = np.isfinite(H.real) & np.isfinite(H.imag)
    if not finite_mask.all():
        freqs = freqs[finite_mask]
        H = H[finite_mask]
        if len(freqs) < 3:
            raise ValueError(
                "After dropping non-finite samples, AC sweep has fewer than "
                "3 points — nothing to analyze"
            )

    return freqs, H


def unwrap_phase_safe(H: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return (phase_deg_unwrapped, warnings).

    Detects sparse sweeps where ``np.unwrap`` may have snapped the wrong way:
    if the wrapped phase jumps by more than ~90° between adjacent samples,
    the true crossing could be missed. We still return the unwrapped phase
    (caller decisions that depend on it are common and usually right), just
    flag the risk.
    """
    warnings: list[str] = []
    phase_rad = np.angle(H)
    # Check BEFORE unwrapping: if adjacent samples already differ by >90° in
    # wrapped space the sweep is too sparse for trustworthy unwrap. ±360°
    # jumps (the expected wrap discontinuity) don't count.
    wrapped_step_deg = np.abs(np.rad2deg(np.diff(phase_rad)))
    wrapped_step_deg = np.minimum(wrapped_step_deg, np.abs(wrapped_step_deg - 360.0))
    if wrapped_step_deg.size and np.max(wrapped_step_deg) > _UNWRAP_WARN_STEP_DEG:
        max_step = float(np.max(wrapped_step_deg))
        warnings.append(
            f"Phase changes by up to {max_step:.1f}° between adjacent sweep "
            "points; unwrap may be incorrect. Increase sweep density "
            "(e.g. .ac dec 50 ...) for reliable phase margins."
        )
    return np.rad2deg(np.unwrap(phase_rad)), warnings


# ---------------------------------------------------------------------------
# Log-axis interpolation + crossings
# ---------------------------------------------------------------------------


def log_interp(freqs: np.ndarray, values: np.ndarray, f_query: float) -> float:
    """Linear interpolation of ``values`` vs ``log10(freqs)`` at ``f_query``.

    Real-valued inputs only. For complex H, interpolate magnitude and phase
    separately via ``log_interp_complex``. Extrapolation uses the endpoint
    value (clamped).
    """
    if f_query <= 0:
        raise ValueError(f"f_query must be positive, got {f_query}")
    if f_query <= freqs[0]:
        return float(values[0])
    if f_query >= freqs[-1]:
        return float(values[-1])
    return float(np.interp(np.log10(f_query), np.log10(freqs), values))


def log_interp_complex(freqs: np.ndarray, H: np.ndarray, f_query: float) -> complex:
    """Interpolate complex H at ``f_query`` via (log10 f, magnitude_db, unwrapped phase).

    Interpolating real/imag directly is wrong for log-spaced sweeps — it
    under-estimates gain between sample points whenever the magnitude is
    changing on a dB-per-decade slope. Interpolating mag_db + unwrapped
    phase matches what a human reads off a Bode plot.
    """
    mag_db = magnitude_db(H)
    phase_rad = np.unwrap(np.angle(H))
    m_db = log_interp(freqs, mag_db, f_query)
    p_rad = log_interp(freqs, phase_rad, f_query)
    magnitude = 10.0 ** (m_db / 20.0)
    return complex(magnitude * np.cos(p_rad), magnitude * np.sin(p_rad))


def detect_crossings(
    freqs: np.ndarray,
    values: np.ndarray,
    level: float,
    *,
    direction: SearchDirection = "any",
    min_separation_decades: float = 0.0,
    min_amplitude: float = 0.0,
) -> list[Crossover]:
    """Find all frequencies where ``values`` crosses ``level``.

    Interpolation is log-axis (matches .AC DEC sweep spacing). Returns
    crossings ordered by frequency. ``min_separation_decades`` suppresses
    crossings that would duplicate a jittery near-zero-slope region — two
    crossings within that many decades collapse to the first.

    ``min_amplitude`` is an amplitude deadband (in the same units as
    ``values``) for signals that hover at ``level``: a crossing only counts
    once ``values`` has moved at least this far past ``level`` since the
    previous accepted crossing. With the default of 0.0 every sign flip is
    reported, so existing callers are unaffected. Set it to suppress the
    float-epsilon chatter a flat-at-level loop (e.g. an allpass grazing 0 dB)
    would otherwise register as dozens of crossings.
    """
    if direction not in ("any", "rising", "falling"):
        raise ValueError(f"direction must be 'any', 'rising', or 'falling', got {direction!r}")
    if len(freqs) != len(values):
        raise ValueError(f"freqs ({len(freqs)}) and values ({len(values)}) length mismatch")
    if min_separation_decades < 0:
        raise ValueError(f"min_separation_decades must be >= 0, got {min_separation_decades}")
    if min_amplitude < 0:
        raise ValueError(f"min_amplitude must be >= 0, got {min_amplitude}")

    d = np.asarray(values, dtype=float) - float(level)
    sign_change = (d[:-1] * d[1:]) < 0
    touch_exit = (d[:-1] == 0) & (d[1:] != 0)
    idx = np.where(sign_change | touch_exit)[0]
    if len(idx) == 0:
        return []

    log_f = np.log10(freqs)
    y0 = values[idx]
    y1 = values[idx + 1]
    x0 = log_f[idx]
    x1 = log_f[idx + 1]
    denom = y1 - y0
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(denom != 0, (level - y0) / denom, 0.0)
    xc = x0 + frac * (x1 - x0)
    fc = np.power(10.0, xc)

    rising_mask = y1 > y0
    directions: list[str]
    if direction == "rising":
        fc = fc[rising_mask]
        seg_idx = idx[rising_mask]
        directions = ["rising"] * len(fc)
    elif direction == "falling":
        fc = fc[~rising_mask]
        seg_idx = idx[~rising_mask]
        directions = ["falling"] * len(fc)
    else:
        seg_idx = idx
        directions = ["rising" if r else "falling" for r in rising_mask]

    order = np.argsort(fc)
    fc = fc[order]
    seg_idx = seg_idx[order]
    directions = [directions[i] for i in order]

    crossings: list[Crossover] = []
    last_log = -np.inf
    # Amplitude-deadband hysteresis, applied via the signal's confirmed side of
    # ``level``. The signal is "confirmed high" once d >= +min_amplitude and
    # "confirmed low" once d <= -min_amplitude; in between it is unconfirmed. A
    # candidate crossing counts only when the confirmed side flips across it —
    # i.e. the signal genuinely departed ``level`` by at least min_amplitude on
    # both sides. A signal that only grazes ``level`` (an allpass at 0 dB, or
    # float-epsilon wobble) never confirms either side, so nothing is emitted.
    if min_amplitude > 0:
        confirmed_at: list[int] = []  # sample indices where the side flips
        # Seed the confirmed side from the signal's sign BEFORE any crossing,
        # not 0. A loop that starts inside the deadband but on the high side
        # (e.g. +0.1 dB) and then drops past -min_amplitude genuinely crosses
        # ``level``; seeding with 0 would treat that 0 -> -1 transition as "not
        # yet confirmed on both sides" and silently drop a real unity crossing
        # (misclassifying the loop as never reaching unity). A pure graze still
        # never reaches +/-min_amplitude, so it records no flips regardless of
        # this seed.
        confirmed_side = int(np.sign(d[0])) if len(d) else 0
        for j in range(len(d)):
            val = float(d[j])
            if val >= min_amplitude:
                new_side = 1
            elif val <= -min_amplitude:
                new_side = -1
            else:
                continue
            if new_side != confirmed_side:
                if confirmed_side != 0:
                    confirmed_at.append(j)
                confirmed_side = new_side
        # Each confirmed-side flip brackets exactly one real crossing of
        # ``level``; map it to the candidate crossing whose segment lies in the
        # bracket. ``prev`` walks the last sample of the previous confirmed run.
        kept: list[tuple[float, str]] = []
        prev = 0
        cand = sorted(
            zip(seg_idx.tolist(), fc.tolist(), directions, strict=True),
            key=lambda t: t[0],
        )
        ci = 0
        for j in confirmed_at:
            # Advance to the first candidate crossing at/after the previous
            # bracket end, then take the one inside [prev, j).
            while ci < len(cand) and cand[ci][0] < prev:
                ci += 1
            if ci < len(cand) and cand[ci][0] < j:
                _si, fcross, dirn = cand[ci]
                kept.append((float(fcross), dirn))
                ci += 1
            prev = j
        kept.sort(key=lambda t: t[0])
        # Re-point fc/directions at the deadband-confirmed crossings and fall
        # through to the shared min_separation filter + emit loop below, instead
        # of duplicating it here.
        fc = np.array([f for f, _ in kept], dtype=float)
        directions = [dirn for _, dirn in kept]

    for f, dirn in zip(fc, directions, strict=True):
        lf = float(np.log10(f))
        if min_separation_decades > 0 and (lf - last_log) < min_separation_decades:
            continue
        assert dirn in ("rising", "falling")  # narrow the Literal for pyright
        crossings.append({"frequency_hz": float(f), "direction": dirn})
        last_log = lf
    return crossings


class CrossingWithQuantity(TypedDict):
    """One crossing emitted by :func:`find_crossings_any_quantity`.

    Enriched vs the bare :class:`Crossover` with the quantity+level+units
    that produced the crossing, so downstream consumers can mix multiple
    crossing searches in one result stream and tell them apart.
    """

    frequency_hz: float
    direction: CrossingDirection
    quantity: Quantity
    level: float
    units: str


def find_crossings_any_quantity(
    freqs: np.ndarray,
    H: np.ndarray,
    *,
    quantity: Quantity,
    level: float,
    direction: SearchDirection = "any",
    f_start: float | None = None,
    f_end: float | None = None,
    max_results: int = 10,
    min_separation_decades: float = 0.0,
) -> tuple[list[CrossingWithQuantity], list[str]]:
    """Crossing primitive that backs ``find_crossing``.

    Slices to ``[f_start, f_end]`` (inclusive, log-interpolation at the
    endpoints), selects the quantity to cross, and returns at most
    ``max_results`` crossings. Warnings (e.g. unwrap instability on phase
    queries) come out in the second return value.
    """
    if max_results < 1:
        raise ValueError(f"max_results must be >= 1, got {max_results}")

    warnings: list[str] = []

    lo = freqs[0] if f_start is None else max(float(f_start), float(freqs[0]))
    hi = freqs[-1] if f_end is None else min(float(f_end), float(freqs[-1]))
    if lo >= hi:
        raise ValueError(
            f"f_start ({lo:.6g}) must be less than f_end ({hi:.6g}) and "
            "within the sweep range "
            f"[{freqs[0]:.6g}, {freqs[-1]:.6g}]"
        )

    mask = (freqs >= lo) & (freqs <= hi)
    f_win = freqs[mask]
    H_win = H[mask]
    if len(f_win) < 2:
        raise ValueError(f"Window [{lo:.6g}, {hi:.6g}] contains fewer than 2 sample points")

    if quantity == "magnitude_db":
        values = magnitude_db(H_win)
        units = "dB"
    elif quantity == "magnitude_linear":
        values = np.abs(H_win).astype(float)
        units = ""
    elif quantity == "phase_deg":
        phase_deg, phase_warn = unwrap_phase_safe(H_win)
        warnings.extend(phase_warn)
        values = phase_deg
        units = "deg"
    else:
        raise ValueError(
            f"quantity must be 'magnitude_db', 'magnitude_linear', or "
            f"'phase_deg', got {quantity!r}"
        )

    crossings = detect_crossings(
        f_win,
        values,
        level,
        direction=direction,
        min_separation_decades=min_separation_decades,
    )

    enriched: list[CrossingWithQuantity] = []
    for c in crossings[:max_results]:
        enriched.append(
            {
                "frequency_hz": c["frequency_hz"],
                "direction": c["direction"],
                "quantity": quantity,
                "level": float(level),
                "units": units,
            }
        )

    if len(crossings) > max_results:
        warnings.append(
            f"Found {len(crossings)} crossings; truncated to max_results={max_results}"
        )

    return enriched, warnings


# ---------------------------------------------------------------------------
# Batch gain/phase queries
# ---------------------------------------------------------------------------


class _GainAtPointOptional(TypedDict, total=False):
    phase_deg_unwrapped: float


class GainAtPoint(_GainAtPointOptional):
    """One gain/phase sample emitted by :func:`gain_at_frequencies`.

    ``phase_deg_unwrapped`` is only present when the caller passes
    ``include_unwrapped_phase=True`` — otherwise the key is omitted entirely
    rather than carrying a redundant ``null``.
    """

    frequency_hz: float
    magnitude_db: float
    magnitude_linear: float
    phase_deg: float


def gain_at_frequencies(
    freqs: np.ndarray,
    H: np.ndarray,
    query_freqs: Sequence[float],
    *,
    include_unwrapped_phase: bool = False,
) -> tuple[list[GainAtPoint], list[str]]:
    """Return magnitude_db, magnitude_linear, phase_deg at each query freq.

    All interpolation is log-axis. Query frequencies outside the sweep
    range are clamped to the endpoint value with a warning per out-of-range
    point.
    """
    if len(query_freqs) == 0:
        raise ValueError("query_freqs is empty")
    for f in query_freqs:
        if not np.isfinite(f) or f <= 0:
            raise ValueError(f"Query frequency must be positive and finite, got {f}")

    mag_db = magnitude_db(H)
    phase_deg_unwrapped, phase_warn = unwrap_phase_safe(H)
    warnings: list[str] = list(phase_warn)

    points: list[GainAtPoint] = []
    for f in query_freqs:
        if f < freqs[0] or f > freqs[-1]:
            warnings.append(
                f"Query frequency {f:g} Hz is outside sweep range "
                f"[{freqs[0]:g}, {freqs[-1]:g}] Hz; clamped to nearest endpoint"
            )
        m_db = log_interp(freqs, mag_db, f)
        # Interpolate on the UNWRAPPED phase: interpolating wrapped phase
        # across the ±180° seam (e.g. samples at +179° and -179°) averages
        # straight through 0° — up to ~180° of silent error at the exact
        # frequencies where phase matters most (stability margins).
        p_deg = log_interp(freqs, phase_deg_unwrapped, f)
        # Wrap to (-180, 180] for the reported phase (matches what a user
        # reads off a Bode plot); unwrapped phase is available as an opt-in.
        p_deg_wrapped = ((p_deg + 180.0) % 360.0) - 180.0
        if p_deg_wrapped == -180.0:
            p_deg_wrapped = 180.0
        entry: GainAtPoint = {
            "frequency_hz": float(f),
            "magnitude_db": float(m_db),
            "magnitude_linear": float(10.0 ** (m_db / 20.0)),
            "phase_deg": float(p_deg_wrapped),
        }
        if include_unwrapped_phase:
            entry["phase_deg_unwrapped"] = float(p_deg)
        points.append(entry)

    return points, warnings


# ---------------------------------------------------------------------------
# Filter characterization
# ---------------------------------------------------------------------------


FilterType = Literal["lowpass", "highpass", "bandpass", "bandstop", "allpass", "unknown"]


def classify_filter(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    *,
    flatness_db: float = 1.0,
) -> FilterType:
    """Heuristic filter-type classification based on endpoint vs peak gain.

    Returns a single label. The classifier is deliberately conservative:
    ambiguous sweeps (shallow roll-off, lopsided endpoints, under-sampled
    notches) return ``"unknown"`` rather than a best guess. Callers get a
    clear signal to inspect the plot instead of a plausible-looking wrong
    answer.

    Logic:
      - LPF: both drops agree with LPF shape AND high-side drop ≥
        ``_CLEAR_DROP_DB``.
      - HPF: mirror.
      - BPF: peak interior AND both drops ≥ ``_CLEAR_DROP_DB``.
      - BSF: notch interior AND both rises ≥ ``_CLEAR_DROP_DB`` AND
        endpoints near the peak.
      - allpass: gain variation < ``flatness_db`` across the whole band.
      - unknown: anything else.

    ``flatness_db`` is the tolerance for considering two gain levels "the
    same" — default 1 dB matches what most datasheets mean by "passband
    flatness".
    """
    if len(freqs) != len(mag_db):
        raise ValueError("freqs and mag_db must have the same length")
    if len(freqs) < 3:
        raise ValueError("Need at least 3 frequency points for classification")

    g_lo = float(mag_db[0])
    g_hi = float(mag_db[-1])
    g_max = float(np.max(mag_db))
    g_min = float(np.min(mag_db))
    idx_peak = int(np.argmax(mag_db))
    idx_notch = int(np.argmin(mag_db))

    # Allpass: essentially flat across the whole sweep.
    if (g_max - g_min) < flatness_db:
        return "allpass"

    n = len(freqs)
    frac_peak = idx_peak / (n - 1)
    frac_notch = idx_notch / (n - 1)

    drop_lo = g_max - g_lo
    drop_hi = g_max - g_hi
    rise_lo = g_lo - g_min
    rise_hi = g_hi - g_min

    passband_tol = flatness_db * _PASSBAND_SIDE_TOL

    # BPF: peak interior, both endpoints well below peak.
    if (
        _INTERIOR_FRAC < frac_peak < 1 - _INTERIOR_FRAC
        and drop_lo >= _CLEAR_DROP_DB
        and drop_hi >= _CLEAR_DROP_DB
    ):
        return "bandpass"

    # BSF: notch interior, both endpoints near the peak (not just above min).
    if (
        _INTERIOR_FRAC < frac_notch < 1 - _INTERIOR_FRAC
        and rise_lo >= _CLEAR_DROP_DB
        and rise_hi >= _CLEAR_DROP_DB
        and (g_max - g_lo) <= passband_tol
        and (g_max - g_hi) <= passband_tol
    ):
        return "bandstop"

    # LPF: high-side drops clearly, low-side stays near peak.
    if drop_hi >= _CLEAR_DROP_DB and drop_lo <= passband_tol:
        return "lowpass"

    # HPF: mirror.
    if drop_lo >= _CLEAR_DROP_DB and drop_hi <= passband_tol:
        return "highpass"

    # Anything else is ambiguous — return "unknown" rather than guess.
    return "unknown"


def _find_passband_range(
    freqs: np.ndarray,
    mag_db: np.ndarray,
    filter_type: FilterType,
    *,
    flatness_db: float,
) -> tuple[float, float, float]:
    """Return (f_lo, f_hi, passband_gain_db) for the auto-detected passband.

    The passband is the contiguous region within ``flatness_db`` of the peak
    (or for BSF, the two lobes flanking the notch — we report the broader
    lobe as "the passband" for ripple and the notch itself for rejection).

    ``passband_gain_db`` is the flat-plateau gain — anchored at the DC-side
    edge for a lowpass, the high-frequency edge for a highpass, and the peak
    for a bandpass. It is deliberately NOT the band median: the auto band runs
    from the peak up into the roll-off knee, so the median sits below the true
    plateau and biases the -3 dB cutoff outward (worse the narrower the in-band
    span). A short median at the flat edge stays robust to a noisy edge sample.
    """
    g_max = float(np.max(mag_db))
    in_band = mag_db >= (g_max - flatness_db)

    if filter_type == "bandstop":
        # Use the whole band excluding the notch region; passband gain is
        # the median of the two flat lobes.
        idx_min = int(np.argmin(mag_db))
        flat_mask = in_band.copy()
        flat_mask[idx_min] = False
        if not flat_mask.any():
            return float(freqs[0]), float(freqs[-1]), g_max
        return (
            float(freqs[int(np.argmax(flat_mask))]),
            float(freqs[len(freqs) - 1 - int(np.argmax(flat_mask[::-1]))]),
            float(np.median(mag_db[flat_mask])),
        )

    # For LPF/HPF/BPF, find the largest contiguous run where in_band is True.
    if not in_band.any():
        return float(freqs[0]), float(freqs[-1]), g_max
    # Run-length search.
    idx_peak = int(np.argmax(mag_db))
    lo = idx_peak
    hi = idx_peak
    while lo > 0 and in_band[lo - 1]:
        lo -= 1
    while hi < len(freqs) - 1 and in_band[hi + 1]:
        hi += 1
    if filter_type == "lowpass":
        # Plateau is the DC-side edge; a 3-sample median tolerates a noisy
        # low-frequency point without re-admitting the roll-off knee.
        pb_gain = float(np.median(mag_db[lo : min(lo + 3, hi + 1)]))
    elif filter_type == "highpass":
        pb_gain = float(np.median(mag_db[max(lo, hi - 2) : hi + 1]))
    elif filter_type == "bandpass":
        # The bandpass plateau IS the peak; -3 dB bandwidth is referenced to it.
        pb_gain = g_max
    else:
        # Ambiguous/unknown: no well-defined plateau, keep the band median.
        pb_gain = float(np.median(mag_db[lo : hi + 1]))
    return float(freqs[lo]), float(freqs[hi]), pb_gain


def _estimate_order_from_slope(slope_db_per_dec: float) -> int | None:
    """Round a roll-off slope to an integer filter order, or ``None`` if far
    from any integer multiple of 20 dB/dec.

    Tolerance of ±3 dB/dec around the integer target — keeps the
    heuristic noise-tolerant on real-world slopes (a 2-stage opamp loop
    legitimately measures -17.97 dB/dec near unity) without rounding a
    2nd-order roll-off (-40 dB/dec) up to 3.
    """
    if not np.isfinite(slope_db_per_dec):
        return None
    order_f = abs(slope_db_per_dec) / 20.0
    order_i = round(order_f)
    if order_i < 1:
        return None
    if abs(order_f - order_i) * 20.0 > 3.0:
        return None
    return order_i


class FilterMetricsOutput(TypedDict):
    """Return shape of :func:`compute_filter_metrics`."""

    filter_type: FilterType
    passband_gain_db: float
    passband_low_hz: float
    passband_high_hz: float
    passband_ripple_db: float
    cutoff_low_hz: float | None
    cutoff_high_hz: float | None
    ref_db: float
    cutoff_level_db: float
    stopband_rejection_db: float | None
    transition_bandwidth_hz: float | None
    rolloff_slope_db_per_decade: float | None
    estimated_order: int | None
    warnings: list[str]


def compute_filter_metrics(
    freqs: np.ndarray,
    H: np.ndarray,
    *,
    ref_db: float = HALF_POWER_DB,
    flatness_db: float = 1.0,
    passband_range: tuple[float, float] | None = None,
    stopband_range: tuple[float, float] | None = None,
) -> FilterMetricsOutput:
    """Classify a filter and compute its defining metrics.

    Reported cutoffs are ``ref_db`` BELOW the passband reference gain — so
    ``ref_db=-3`` gives the -3 dB point relative to passband, not an
    absolute -3 dB gain. This matches standard filter datasheet language.

    Stopband rejection: if ``stopband_range`` is given, returns the minimum
    attenuation (dB) inside that range; otherwise reports rejection at the
    furthest endpoint outside the passband.

    Order estimate: taken from the local slope one decade past the cutoff.
    Returned as ``None`` when the slope isn't near an integer multiple of
    20 dB/dec (likely because the sweep didn't reach the asymptote).
    """
    if ref_db >= 0:
        raise ValueError(f"ref_db must be negative (dB below passband), got {ref_db}")

    mag_db = magnitude_db(H)
    filter_type = classify_filter(freqs, mag_db, flatness_db=flatness_db)

    if passband_range is not None:
        f_pb_lo, f_pb_hi = passband_range
        if f_pb_lo >= f_pb_hi:
            raise ValueError("passband_range: low must be < high")
        pb_mask = (freqs >= f_pb_lo) & (freqs <= f_pb_hi)
        if not pb_mask.any():
            raise ValueError(f"passband_range [{f_pb_lo}, {f_pb_hi}] has no sweep samples")
        pb_gain = float(np.median(mag_db[pb_mask]))
    else:
        f_pb_lo, f_pb_hi, pb_gain = _find_passband_range(
            freqs, mag_db, filter_type, flatness_db=flatness_db
        )

    pb_mask_final = (freqs >= f_pb_lo) & (freqs <= f_pb_hi)
    if pb_mask_final.any():
        pb_mag = mag_db[pb_mask_final]
        pb_span = float(np.max(pb_mag) - np.min(pb_mag))
        # A monotonic response (e.g. textbook LPF passband) has no actual
        # ripple — the auto-detected passband just clips the roll-off at
        # ``flatness_db``. Distinguish "ripple" (oscillation) from
        # "monotonic passband variation" by checking sign changes in the
        # first difference.
        if pb_mag.size >= 3:
            diffs = np.diff(pb_mag)
            sign_changes = int(np.sum(np.diff(np.sign(diffs)) != 0))
            is_monotonic = sign_changes <= 1
        else:
            is_monotonic = True
        passband_ripple = 0.0 if is_monotonic else pb_span
    else:
        passband_ripple = 0.0

    cutoff_level = pb_gain + ref_db  # ref_db is negative → cutoff is below passband

    cutoff_low: float | None = None
    cutoff_high: float | None = None
    transition_bw: float | None = None
    warnings: list[str] = []

    if filter_type in ("lowpass", "highpass", "bandpass"):
        crossings = detect_crossings(freqs, mag_db, cutoff_level, direction="any")
        # Cutoffs are the crossings OUTSIDE the passband, ordered by how
        # close they are to the passband edge they flank.
        below = [c for c in crossings if c["frequency_hz"] < f_pb_lo]
        above = [c for c in crossings if c["frequency_hz"] > f_pb_hi]
        if filter_type == "lowpass":
            # Cutoff is the first falling crossing above the passband.
            fall = [c for c in above if c["direction"] == "falling"]
            cutoff_high = fall[0]["frequency_hz"] if fall else None
        elif filter_type == "highpass":
            rise = [c for c in below if c["direction"] == "rising"]
            cutoff_low = rise[-1]["frequency_hz"] if rise else None
        else:  # bandpass
            rise = [c for c in below if c["direction"] == "rising"]
            fall = [c for c in above if c["direction"] == "falling"]
            cutoff_low = rise[-1]["frequency_hz"] if rise else None
            cutoff_high = fall[0]["frequency_hz"] if fall else None

    stopband_rejection: float | None = None
    if stopband_range is not None:
        f_sb_lo, f_sb_hi = stopband_range
        if f_sb_lo >= f_sb_hi:
            raise ValueError("stopband_range: low must be < high")
        sb_mask = (freqs >= f_sb_lo) & (freqs <= f_sb_hi)
        if not sb_mask.any():
            raise ValueError(f"stopband_range [{f_sb_lo}, {f_sb_hi}] has no sweep samples")
        # Rejection = how far below passband the WORST stopband sample is;
        # worst means smallest attenuation (largest mag).
        worst_sb = float(np.max(mag_db[sb_mask]))
        stopband_rejection = pb_gain - worst_sb
    else:
        # Auto: report attenuation at the sweep endpoint farthest from passband.
        if filter_type == "lowpass":
            stopband_rejection = pb_gain - float(mag_db[-1])
        elif filter_type == "highpass":
            stopband_rejection = pb_gain - float(mag_db[0])
        elif filter_type == "bandpass":
            stopband_rejection = pb_gain - float(min(mag_db[0], mag_db[-1]))
        if stopband_rejection is not None:
            # For an unbounded roll-off (e.g. a 1st-order LPF) there is no
            # intrinsic stopband edge, so this number is just |H| at the sweep
            # endpoint — it grows with the sweep range, it is not a filter
            # property. Flag it so it isn't over-interpreted. Pass
            # ``stopband_range`` for a defined-band rejection.
            warnings.append(
                "stopband_rejection/transition_bandwidth are measured at the "
                "sweep endpoint (no stopband_range given); they depend on the "
                "sweep range, not just the filter. Pass stopband_range for a "
                "band-defined figure."
            )
        elif filter_type == "bandstop":
            stopband_rejection = pb_gain - float(np.min(mag_db))

    # Transition bandwidth: distance between cutoff and the frequency where
    # the response drops another ``stopband_ref`` below cutoff. Use
    # 40 dB below passband as the stopband reference.
    if cutoff_high is not None or cutoff_low is not None:
        stopband_level = pb_gain - max(40.0, abs(ref_db) + 20.0)
        sb_crossings = detect_crossings(freqs, mag_db, stopband_level, direction="any")
        if cutoff_high is not None:
            sb_above = [c for c in sb_crossings if c["frequency_hz"] > cutoff_high]
            if sb_above:
                transition_bw = float(sb_above[0]["frequency_hz"] - cutoff_high)
        if cutoff_low is not None and transition_bw is None:
            sb_below = [c for c in sb_crossings if c["frequency_hz"] < cutoff_low]
            if sb_below:
                transition_bw = float(cutoff_low - sb_below[-1]["frequency_hz"])

    # Order estimate from roll-off slope in the ASYMPTOTIC region (one
    # decade past cutoff to two decades past cutoff). Measuring from the
    # cutoff itself underestimates the slope because we're still in the
    # knee of the response.
    estimated_order: int | None = None
    slope: float | None = None
    if filter_type == "lowpass" and cutoff_high is not None:
        f_lo = min(cutoff_high * 10.0, freqs[-1])
        f_hi = min(cutoff_high * 100.0, freqs[-1])
        if f_hi > f_lo * 1.5:  # need at least ~0.2 decades to get a clean slope
            slope = _slope_db_per_decade(freqs, mag_db, f_lo, f_hi)
            estimated_order = _estimate_order_from_slope(slope)
    elif filter_type == "highpass" and cutoff_low is not None:
        f_hi = max(cutoff_low / 10.0, freqs[0])
        f_lo = max(cutoff_low / 100.0, freqs[0])
        if f_hi > f_lo * 1.5:
            slope = _slope_db_per_decade(freqs, mag_db, f_lo, f_hi)
            estimated_order = _estimate_order_from_slope(slope)

    if filter_type == "unknown":
        warnings.append(
            "Could not confidently classify filter type — the response "
            "doesn't fit LPF/HPF/BPF/BSF heuristics. Inspect the gain "
            "curve directly or pass passband_range/stopband_range to force "
            "metric extraction on a region of interest."
        )

    # Notch under-sampling check: a sharp null can fall BETWEEN samples,
    # leaving stopband_rejection_db as a lower bound only. If either
    # sample flanking the minimum is noticeably above it, the true null
    # is likely deeper than reported.
    idx_min = int(np.argmin(mag_db))
    if 0 < idx_min < len(mag_db) - 1:
        neighbor_gap = min(
            float(mag_db[idx_min - 1] - mag_db[idx_min]),
            float(mag_db[idx_min + 1] - mag_db[idx_min]),
        )
        if neighbor_gap > _NOTCH_UNDERSAMPLED_GAP_DB and filter_type in ("bandstop", "unknown"):
            warnings.append(
                f"Minimum at {float(freqs[idx_min]):.6g} Hz is flanked by "
                f"samples ≥{neighbor_gap:.1f} dB higher — the true null likely "
                "falls between samples and stopband_rejection_db is a lower "
                "bound only. Re-sweep with denser spacing around this frequency "
                "(e.g. add a linear sweep band around the notch)."
            )

    return {
        "filter_type": filter_type,
        "passband_gain_db": float(pb_gain),
        "passband_low_hz": float(f_pb_lo),
        "passband_high_hz": float(f_pb_hi),
        "passband_ripple_db": float(passband_ripple),
        "cutoff_low_hz": cutoff_low,
        "cutoff_high_hz": cutoff_high,
        "ref_db": float(ref_db),
        "cutoff_level_db": float(cutoff_level),
        "stopband_rejection_db": (
            None if stopband_rejection is None else float(stopband_rejection)
        ),
        "transition_bandwidth_hz": transition_bw,
        "rolloff_slope_db_per_decade": None if slope is None else float(slope),
        "estimated_order": estimated_order,
        "warnings": warnings,
    }


def _slope_db_per_decade(freqs: np.ndarray, mag_db: np.ndarray, f_lo: float, f_hi: float) -> float:
    """Gain slope between two frequencies, in dB per decade. Log-axis."""
    if f_lo <= 0 or f_hi <= 0 or f_hi <= f_lo:
        raise ValueError(f"Need 0 < f_lo ({f_lo}) < f_hi ({f_hi}) for slope calculation")
    g_lo = log_interp(freqs, mag_db, f_lo)
    g_hi = log_interp(freqs, mag_db, f_hi)
    decades = np.log10(f_hi / f_lo)
    return float((g_hi - g_lo) / decades)


# ---------------------------------------------------------------------------
# Stability: gain/phase margins, all crossovers
# ---------------------------------------------------------------------------


StabilityLabel = Literal[
    "stable",
    "unstable",
    "conditional",
    "unconditional",
    "always_below_unity",
    "flat_at_unity",
]


class Crossover(TypedDict):
    """One unity-gain or -180° phase crossing in the sweep."""

    frequency_hz: float
    direction: CrossingDirection


class PhaseMargin(TypedDict):
    """Phase margin measured at one unity-gain crossing."""

    frequency_hz: float
    margin_deg: float
    direction: CrossingDirection


class GainMargin(TypedDict):
    """Gain margin measured at one -180° phase crossing."""

    frequency_hz: float
    margin_db: float
    direction: CrossingDirection


class StabilityMetricsOutput(TypedDict):
    """Return shape of :func:`compute_stability_metrics`."""

    dc_gain_db: float
    high_freq_gain_db: float
    stability: StabilityLabel
    unity_gain_crossovers: list[Crossover]
    phase_180_crossovers: list[Crossover]
    phase_margins: list[PhaseMargin]
    gain_margins: list[GainMargin]
    phase_margin_worst_deg: float | None
    gain_margin_worst_db: float | None
    warnings: list[str]


def compute_stability_metrics(
    freqs: np.ndarray,
    H: np.ndarray,
    *,
    min_separation_decades: float = 0.1,
) -> StabilityMetricsOutput:
    """Unity-gain and -180° crossovers with per-crossover margins.

    Unlike a single-crossover heuristic, this reports EVERY unity-gain
    crossing and every -180° phase crossing in the sweep, so
    conditionally-stable loops (where phase dips below -180° and comes
    back) are characterized correctly.

    The worst-case margin is the most negative (least stable) margin across
    crossovers — that's the one that determines stability, and a negative
    margin at any crossover must not be masked by a smaller positive one. If
    no unity-gain crossing exists,
    the loop has |H|<1 everywhere and phase margin is meaningless; if no
    -180° crossing exists (and phase stays above -180°), gain margin is
    "infinite" (reported as ``None`` + ``stability: "unconditional"``).

    ``min_separation_decades`` merges near-duplicate crossings from a
    jittery near-zero-slope region (the magnitude graze near unity of a
    heavily compensated amp is a classic source of spurious triplets).
    """
    mag_db = magnitude_db(H)
    phase_deg, phase_warnings = unwrap_phase_safe(H)
    warnings = list(phase_warnings)

    dc_gain_db = float(mag_db[0])
    hf_gain_db = float(mag_db[-1])

    # Loop probes start near 0° at DC; CS-amp / inverting outputs start
    # near ±180°. Margins are computed against -180° and presume the
    # loop-probe convention, so warn when the input doesn't look like
    # one — otherwise a closed-loop AC sweep produces a negative phase
    # margin labelled "unconditional", which is contradictory.
    dc_phase = float(phase_deg[0])
    if abs(abs(dc_phase) - 180.0) < 10.0:
        warnings.append(
            f"DC phase is {dc_phase:.0f}° — the input doesn't look like a "
            "loop-gain probe (those start near 0°). Phase margin is computed "
            "against -180° and will be misleading on a closed-loop or "
            "inverting-amplifier output. Wire a Middlebrook probe and run AC "
            "on the loop signal instead."
        )

    unity_crossings = detect_crossings(
        freqs,
        mag_db,
        0.0,
        direction="any",
        min_separation_decades=min_separation_decades,
        min_amplitude=_UNITY_GAIN_DEADBAND_DB,
    )
    phase_crossings = detect_crossings(
        freqs,
        phase_deg,
        -180.0,
        direction="any",
        min_separation_decades=min_separation_decades,
    )

    phase_margins: list[PhaseMargin] = []
    for c in unity_crossings:
        f = c["frequency_hz"]
        p = log_interp(freqs, phase_deg, f)
        pm = 180.0 + p
        # Normalize phase margin to (-180, 180] for readability — a system
        # with phase at -190° at unity has pm = -10° (unstable), not +350°.
        while pm > 180.0:
            pm -= 360.0
        while pm <= -180.0:
            pm += 360.0
        phase_margins.append(
            {
                "frequency_hz": float(f),
                "margin_deg": float(pm),
                "direction": c["direction"],
            }
        )

    gain_margins: list[GainMargin] = []
    for c in phase_crossings:
        f = c["frequency_hz"]
        g = log_interp(freqs, mag_db, f)
        # Gain margin = how much gain can increase before instability at
        # the -180° crossing, measured in dB below unity. Negative means
        # gain is already above unity at that crossing (definitely
        # unstable at that frequency).
        gm = -g
        gain_margins.append(
            {
                "frequency_hz": float(f),
                "margin_db": float(gm),
                "direction": c["direction"],
            }
        )

    # Worst-case: the signed minimum (most negative, least stable) margin. A
    # negative margin means instability at that crossover, so it must dominate
    # any smaller positive margin — selecting by magnitude would let a +5°
    # margin mask a -170° one and hide the unstable crossing.
    phase_margin_worst: float | None = (
        float(min(m["margin_deg"] for m in phase_margins)) if phase_margins else None
    )
    gain_margin_worst: float | None = (
        float(min(m["margin_db"] for m in gain_margins)) if gain_margins else None
    )

    # Stability classification.
    # Magnitude that hovers within the unity-gain deadband across the entire
    # sweep (an allpass, or a flat 0 dB loop) produces no real crossing — its
    # sign flips are float-epsilon graze. That is degenerate, not "below unity":
    # phase margin is ill-defined and the usual stability verdict does not apply.
    grazes_unity = bool(np.all(np.abs(mag_db) < _UNITY_GAIN_DEADBAND_DB))
    if not unity_crossings and grazes_unity:
        stability = "flat_at_unity"
        warnings.append(
            "Loop magnitude sits at unity (0 dB) across the whole sweep — the "
            "gain never genuinely crosses 0 dB, so phase margin is ill-defined. "
            "This looks like an allpass or a flat-gain loop, not a normal "
            "single-crossover response; the stability verdict does not apply."
        )
    elif not unity_crossings:
        stability = "always_below_unity"
        warnings.append(
            "Loop gain never reaches unity in the sweep range — phase "
            "margin is undefined. The system is stable if the loop is "
            "designed to have <0 dB gain everywhere, but verify the "
            "sweep covers the actual crossover frequency."
        )
    elif not phase_crossings:
        # Phase never touches -180° in the sweep → no gain-margin limit, hence
        # "unconditional". But that label says nothing about phase margin: a
        # 2-pole loop asymptotes toward -180° without crossing it, so a low
        # (e.g. ~20°) phase margin still classifies as "unconditional". Flag the
        # ringing risk so the label isn't read as "comfortably stable".
        stability = "unconditional"
        if phase_margin_worst is not None and phase_margin_worst < 45.0:
            warnings.append(
                f"Unconditionally stable (loop-gain phase never reaches -180°), "
                f"but the worst phase margin is only {phase_margin_worst:.1f}° — "
                "the loop is marginal and will ring/peak. 'unconditional' means "
                "'no gain-margin limit', not 'comfortable phase margin'."
            )
    elif len(unity_crossings) > 1 or len(phase_crossings) > 1:
        stability = "conditional"
        warnings.append(
            f"Multiple crossovers detected ({len(unity_crossings)} unity-gain, "
            f"{len(phase_crossings)} -180° phase). System is conditionally "
            "stable — worst-case margins govern stability but each "
            "crossover deserves individual inspection."
        )
    else:
        # Single unity-gain + single -180° crossing. Stable if both
        # margins are positive.
        pm = phase_margins[0]["margin_deg"] if phase_margins else 0.0
        gm = gain_margins[0]["margin_db"] if gain_margins else 0.0
        stability = "stable" if pm > 0 and gm > 0 else "unstable"

    return {
        "dc_gain_db": dc_gain_db,
        "high_freq_gain_db": hf_gain_db,
        "stability": stability,
        "unity_gain_crossovers": unity_crossings,
        "phase_180_crossovers": phase_crossings,
        "phase_margins": phase_margins,
        "gain_margins": gain_margins,
        "phase_margin_worst_deg": phase_margin_worst,
        "gain_margin_worst_db": gain_margin_worst,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Slope / roll-off
# ---------------------------------------------------------------------------


class RollOffOutput(TypedDict):
    """Return shape of :func:`compute_roll_off`."""

    f_low_hz: float
    f_high_hz: float
    gain_low_db: float
    gain_high_db: float
    delta_db: float
    span_decades: float
    slope_db_per_decade: float
    slope_db_per_octave: float
    nearest_pole_order_estimate: int | None
    warnings: list[str]


def compute_roll_off(
    freqs: np.ndarray,
    H: np.ndarray,
    *,
    f_low: float,
    f_high: float,
) -> RollOffOutput:
    """Magnitude slope between two frequencies, plus pole-order estimate.

    Slope is reported in both dB/decade and dB/octave — these are
    redundant but filter designers expect to see the one they prefer.
    ``nearest_pole_order_estimate`` is the slope rounded to the closest
    multiple of 20 dB/dec, returned only when the roll-off is within
    ±2 dB/dec of that multiple (i.e. we're in the asymptotic region).
    """
    if f_low <= 0 or f_high <= 0:
        raise ValueError(f"f_low and f_high must be positive, got {f_low}, {f_high}")
    if f_low >= f_high:
        raise ValueError(f"f_low ({f_low}) must be less than f_high ({f_high})")
    if f_low < freqs[0] or f_high > freqs[-1]:
        raise ValueError(
            f"Range [{f_low}, {f_high}] is outside sweep [{freqs[0]:.6g}, {freqs[-1]:.6g}]"
        )

    mag_db = magnitude_db(H)
    slope_per_dec = _slope_db_per_decade(freqs, mag_db, f_low, f_high)
    slope_per_oct = slope_per_dec * np.log10(2.0)
    order = _estimate_order_from_slope(slope_per_dec)
    gain_low = log_interp(freqs, mag_db, f_low)
    gain_high = log_interp(freqs, mag_db, f_high)

    warnings: list[str] = []
    decades = float(np.log10(f_high / f_low))
    if decades < 0.3:
        warnings.append(
            f"Slope window spans only {decades:.2f} decades; results are "
            "noisy. Widen to at least 1 decade for a reliable slope."
        )

    return {
        "f_low_hz": float(f_low),
        "f_high_hz": float(f_high),
        "gain_low_db": float(gain_low),
        "gain_high_db": float(gain_high),
        "delta_db": float(gain_high - gain_low),
        "span_decades": decades,
        "slope_db_per_decade": float(slope_per_dec),
        "slope_db_per_octave": float(slope_per_oct),
        "nearest_pole_order_estimate": order,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Resonance / peak detection
# ---------------------------------------------------------------------------


class ResonancePeak(TypedDict):
    """One detected resonant peak from :func:`compute_resonances`."""

    frequency_hz: float
    magnitude_db: float
    phase_deg: float
    q_factor: float | None
    bandwidth_3db_hz: float | None


class ResonancesOutput(TypedDict):
    """Return shape of :func:`compute_resonances`."""

    peaks: list[ResonancePeak]
    num_peaks_detected: int
    warnings: list[str]


def compute_resonances(
    freqs: np.ndarray,
    H: np.ndarray,
    *,
    min_prominence_db: float = 3.0,
    min_separation_decades: float = 0.2,
    max_peaks: int = 20,
) -> ResonancesOutput:
    """Detect magnitude peaks and compute Q factor + -3 dB bandwidth per peak.

    Uses scipy.signal.find_peaks on the dB magnitude. Q is ``f_peak /
    bw_3db`` where bw_3db is the width of the -3 dB flanks (measured
    relative to the PEAK gain, not an absolute 0 dB reference). Peaks
    without two flanking -3 dB crossings within the sweep get Q=None.

    ``min_separation_decades`` merges near-duplicate peaks that
    ``find_peaks`` sometimes emits on a shoulder. ``min_prominence_db``
    excludes the gentle rise of a filter passband — a real resonance
    has a narrow hump, not just a broad peak.
    """
    if min_prominence_db <= 0:
        raise ValueError(f"min_prominence_db must be positive, got {min_prominence_db}")
    if max_peaks < 1:
        raise ValueError(f"max_peaks must be >= 1, got {max_peaks}")

    mag_db = magnitude_db(H)
    phase_deg = np.angle(H, deg=True)

    peak_indices, _props = find_peaks(mag_db, prominence=min_prominence_db)

    log_f = np.log10(freqs)
    selected: list[int] = []
    for idx in peak_indices:
        if selected:
            last_log = log_f[selected[-1]]
            if log_f[idx] - last_log < min_separation_decades:
                # Keep the taller of the two.
                if mag_db[idx] > mag_db[selected[-1]]:
                    selected[-1] = int(idx)
                continue
        selected.append(int(idx))

    peaks: list[ResonancePeak] = []
    warnings: list[str] = []
    for idx in selected[:max_peaks]:
        f_peak = float(freqs[idx])
        peak_db = float(mag_db[idx])
        level = peak_db - 3.0
        crossings = detect_crossings(freqs, mag_db, level, direction="any")
        below = [c for c in crossings if c["frequency_hz"] < f_peak]
        above = [c for c in crossings if c["frequency_hz"] > f_peak]
        q: float | None = None
        bw: float | None = None
        if below and above:
            f_lo = below[-1]["frequency_hz"]
            f_hi = above[0]["frequency_hz"]
            bw = float(f_hi - f_lo)
            if bw > 0:
                q = float(f_peak / bw)
        else:
            warnings.append(
                f"Peak at {f_peak:g} Hz lacks flanking -3 dB crossings "
                "inside the sweep; Q undefined. Widen the sweep."
            )
        peaks.append(
            {
                "frequency_hz": f_peak,
                "magnitude_db": peak_db,
                "phase_deg": float(phase_deg[idx]),
                "q_factor": q,
                "bandwidth_3db_hz": bw,
            }
        )

    if len(selected) > max_peaks:
        warnings.append(f"Detected {len(selected)} peaks; truncated to max_peaks={max_peaks}")

    return {
        "peaks": peaks,
        "num_peaks_detected": len(selected),
        "warnings": warnings,
    }


class NoiseIntegralOutput(TypedDict):
    """Return shape of :func:`integrate_noise`."""

    total_rms: float
    f_start_used: float
    f_end_used: float
    n_points: int
    warnings: list[str]


def integrate_noise(
    freqs: np.ndarray,
    density: np.ndarray,
    f_start: float | None,
    f_end: float | None,
) -> NoiseIntegralOutput:
    """Integrate a noise spectral density to a total RMS over ``[f_start, f_end]``.

    SPICE noise raws store *amplitude* spectral density (V/√Hz or A/√Hz) — this
    holds for both LTspice (``V(onoise)``/``V(inoise)``) and ngspice
    (``onoise_spectrum``/``inoise_spectrum``). The total RMS noise in a band is
    therefore ``sqrt(∫ density² df)``: square the density, integrate over
    frequency, take the root. (Same operation LTspice runs when you Ctrl-click a
    ``V(onoise)`` trace label.)

    A ``.noise`` sweep may be stored high→low (the codebase treats noise axes as
    possibly descending); the axis is flipped to ascending first so the integral
    is over increasing frequency. Returns the band actually integrated (clipped
    to the data) and the sample count, so a band that fell partly outside the
    sweep is visible rather than silently truncated. Noise figure / SNR are
    deliberately not computed — they need the source resistance and a reference
    level the caller supplies.
    """
    if freqs.shape != density.shape:
        raise ValueError(f"freq/density length mismatch: {freqs.size} vs {density.size}")
    if freqs.size < 2:
        raise ValueError(f"Need at least 2 frequency points to integrate; got {freqs.size}")
    # Density is a real, non-negative magnitude; the integral squares it so a
    # real array needs no |.|. Only a complex-stored trace must be reduced to
    # its magnitude first (avoids copying the common real case).
    if np.iscomplexobj(density):
        density = np.abs(density)
    # A descending sweep (e.g. high→low .noise) is legitimate; flip axis + density
    # together so searchsorted/trapezoid see ascending frequency. Reject only a
    # genuinely non-monotonic axis (corruption), mirroring window_and_clean.
    if float(freqs[0]) > float(freqs[-1]):
        freqs = freqs[::-1]
        density = density[::-1]
    if np.any(np.diff(freqs) < 0):
        raise ValueError("Frequency axis is not monotonic; cannot integrate noise.")
    warnings: list[str] = []

    lo = float(freqs[0]) if f_start is None else f_start
    hi = float(freqs[-1]) if f_end is None else f_end
    if lo >= hi:
        raise ValueError(f"f_start ({lo:g}) must be < f_end ({hi:g})")
    if f_start is not None and f_start < freqs[0]:
        warnings.append(f"f_start {f_start:g} Hz is below the sweep; clipped to {freqs[0]:g} Hz.")
    if f_end is not None and f_end > freqs[-1]:
        warnings.append(f"f_end {f_end:g} Hz is above the sweep; clipped to {freqs[-1]:g} Hz.")

    i0 = int(np.searchsorted(freqs, lo, side="left"))
    i1 = int(np.searchsorted(freqs, hi, side="right"))
    fb = freqs[i0:i1]
    db = density[i0:i1]
    # A non-finite density sample (NaN/Inf at a singular point) would NaN-poison
    # the whole integral; drop those samples and flag the approximation rather
    # than return a silent NaN total.
    finite = np.isfinite(db)
    if not finite.all():
        n_bad = int((~finite).sum())
        warnings.append(
            f"Dropped {n_bad} non-finite noise-density sample(s) in band before integrating."
        )
        fb = fb[finite]
        db = db[finite]
    if fb.size < 2:
        raise ValueError(
            f"Band [{lo:g}, {hi:g}] Hz selects {fb.size} of {freqs.size} points; "
            f"need at least 2. Sweep spans [{freqs[0]:g}, {freqs[-1]:g}] Hz."
        )

    total_rms = float(np.sqrt(np.trapezoid(db * db, fb)))
    return {
        "total_rms": total_rms,
        "f_start_used": float(fb[0]),
        "f_end_used": float(fb[-1]),
        "n_points": int(fb.size),
        "warnings": warnings,
    }
