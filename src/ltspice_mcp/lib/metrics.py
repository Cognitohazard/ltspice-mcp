"""The numeric core behind every ``analyze_results`` recipe.

One function per recipe discriminant, each with the same shape::

    async def <metric>(source, recipe, step, state) -> MetricValue

``source`` is an explicit :class:`~ltspice_mcp.lib.services.AnalysisSource` —
the raw, its log, its deck, and the identity of the run they came from. It is
passed, never inherited: an earlier design injected the source through a
task-local and handed the reader a decoy path argument it was expected to
ignore, which made "which file did this number come from?" unanswerable from
the call site. :data:`METRICS` maps each recipe class to its function, and
``tools/analyze.py`` looks the recipe up there rather than branching on type.

The functions return plain dicts because the key set of a metric depends on
the data it read — ``signal_stats`` answers with different keys for a
transient, a DC sweep, an AC sweep and a noise run — so the wire shape is
declared by each tool's output contract rather than by a return annotation
that would have to be optional in every field. Non-finite scrubbing belongs to
the response edge, not here: a caller reading a metric in-process sees the
value the maths produced.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import re
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, Literal, NoReturn, TypeAlias

import numpy as np

from ltspice_mcp.errors import NoAxisError, ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.ac_analysis import (
    Quantity,
    SearchDirection,
    compute_filter_metrics,
    compute_resonances,
    compute_return_loss,
    compute_roll_off,
    compute_stability_metrics,
    find_crossings_any_quantity,
    gain_at_frequencies,
    integrate_noise,
    prepare_ac_arrays,
)
from ltspice_mcp.lib.ac_structure import analyze_ac_structure
from ltspice_mcp.lib.format import format_spice_value, parse_spice_value
from ltspice_mcp.lib.log_parser import (
    extract_log_diagnostics,
    parse_measurements,
    read_device_op_points,
    scan_op_step_log,
)
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    compute_ac_bandwidth_metrics,
    dc_axis_name,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    is_ac_analysis,
    is_dc_analysis,
    is_noise_analysis,
    nearest_index,
    query_point_value,
    real_axis,
    safe_magnitude_db,
    trace_unit,
)
from ltspice_mcp.lib.recipes import (
    AcStructureRecipe,
    BodeCrossingRecipe,
    BodeFilterRecipe,
    BodePointRecipe,
    BodeSlopeRecipe,
    EdgesRecipe,
    MeasurementsRecipe,
    NoiseIntegralRecipe,
    OperatingPointRecipe,
    PeriodicRecipe,
    Recipe,
    ResonanceRecipe,
    ReturnLossRecipe,
    SignalStatsRecipe,
    StabilityRecipe,
    SummaryRecipe,
    ThdRecipe,
    TimingRecipe,
    TransientResponseRecipe,
    ValueRecipe,
    Window,
)
from ltspice_mcp.lib.result_observations import (
    deck_observation_inputs,
    relay_observations,
)
from ltspice_mcp.lib.signal_analysis import (
    analyze_disturbance_response,
    analyze_edge,
    analyze_periodic,
    analyze_pulse_response,
    analyze_thd,
    analyze_timing_between,
    compute_measurement_stats,
    compute_signal_stats,
    window_and_clean,
)
from ltspice_mcp.state import SessionState

#: One metric's answer. See the module docstring for why this is not a
#: TypedDict: the key set is chosen by the data, not by the recipe.
MetricValue: TypeAlias = dict[str, Any]

AggregatedField = Literal["value", "at"]

# What an empty ``device_op_points`` bucket means and how to fill it. One
# string because two channels carry it — the operating-point value's own
# ``warnings`` and the consolidated analyze path's ``observations`` — and a
# caller who reaches the same dead end by either route must be given the same
# way out.
NO_DEVICE_OP_POINTS_NOTE = (
    "No small-signal device params (gm/gds/vth/vdsat) in this run. "
    "On LTspice add '.options logopinfo' to the deck (run_experiments adds it "
    "automatically for .op runs); on "
    "ngspice .save them, e.g. '.save all @m1[gm] @m1[gds] @m1[id]'."
)


# ---------------------------------------------------------------------------
# Shared reading helpers
# ---------------------------------------------------------------------------


def parse_time(s: str | None, name: str) -> float | None:
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


def parse_freq(s: str, name: str = "frequency") -> float:
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


def parse_freq_pair(pair: list[str] | None, name: str) -> tuple[float, float] | None:
    if pair is None:
        return None
    if len(pair) != 2:
        raise ResultError(f"{name} must have exactly 2 elements, got {len(pair)}")
    lo = parse_freq(pair[0], f"{name}[0]")
    hi = parse_freq(pair[1], f"{name}[1]")
    if lo >= hi:
        raise ResultError(f"{name}: low ({lo}) must be less than high ({hi})")
    return lo, hi


def spice_text(value: float | str | None) -> str | None:
    """A recipe's SPICE-notation field as the text the parsers accept."""
    if value is None:
        return None
    return format_spice_value(value)


def window_bounds(window: Window | None) -> tuple[str | None, str | None]:
    """A recipe window as the ``(start, end)`` SPICE-notation pair."""
    if window is None:
        return None, None
    return spice_text(window.start), spice_text(window.end)


def reject_non_transient(raw) -> None:
    """Reject AC / DC / noise raws from a transient-only metric.

    edge/pulse/periodic/timing read rise-times, periods, and delays off a time
    axis; an AC sweep (frequency axis, complex data), a .DC sweep (voltage
    axis), or a noise spectrum (frequency axis) all produce meaningless numbers
    here, so refuse them with a pointer to the right recipe. A .op raw has no
    axis and is not a sweep; the caller's get_axis call is guarded to surface a
    clean ResultError pointing at operating_point.
    """
    sim_type = detect_sim_type(raw)
    if is_ac_analysis(sim_type) or is_dc_analysis(sim_type) or is_noise_analysis(sim_type):
        # show_hint=False: the redirect above is the complete guidance — the
        # generic verify-with-check_job hint would misdirect (errors.py contract).
        raise ResultError(
            f"This tool requires transient analysis (.tran) data; got {sim_type!r}. "
            "For a .DC sweep use the value or signal_stats recipe; for frequency-domain "
            "(.AC / .noise) use the bode_* or signal_stats recipes.",
            show_hint=False,
        )


def guarded_axis(raw, step: int, raw_path: Path | None = None) -> np.ndarray:
    """Real-valued sweep axis for ``step``, or a clean error for a no-axis raw.

    spicelib's ``get_axis`` raises when a result has no axis (e.g. an Operating
    Point run); convert that into a friendly ResultError pointing at
    ``operating_point`` rather than letting it surface as a generic internal
    error. AC frequency axes come back complex — strip to the real part.

    When ``raw_path`` is given, a no-axis raw that is really a stepped ``.op``
    collapsed to step 0 (the log shows >1 bias iteration) gets the same
    ``.dc``-conversion pointer ``build_simulation_summary`` emits — so the
    confused caller learns the fix where they hit the wall, not just that the
    axis is missing.
    """
    try:
        axis = np.asarray(raw.get_axis(step=step))
    except Exception as e:
        hint = "Use operating_point for a .op result."
        if raw_path is not None:
            log_path = services.AnalysisSource.for_raw(raw_path).log
            if log_path is not None and log_path.exists():
                log_steps, op_iters = scan_op_step_log(log_path)
                if max(len(log_steps), op_iters) > 1:
                    param = next(iter(log_steps[0].keys()), "param") if log_steps else "<param>"
                    hint = (
                        "This is a stepped .op whose .raw carries only step 0. Convert "
                        f"to '.dc {param} START STOP STEP' to get an axis over every bias "
                        "point, or use operating_point for a single bias point."
                    )
        raise ResultError(f"This result has no data axis ({e}). {hint}") from e
    if np.iscomplexobj(axis):
        axis = np.real(axis)
    return axis


def classify_analysis(raw) -> tuple[str, str, str, bool]:
    """``(plotname, analysis_type, axis_unit, x_is_log)`` for a result raw.

    ``analysis_type`` is one of ``transient``/``ac``/``dc``/``noise``;
    ``axis_unit`` is the x unit (``s``/``Hz``/``""``); ``x_is_log`` marks a
    log-frequency x (ac/noise). Shared by the waveform, CSV export and plot
    paths so the classification lives in one place.
    """
    sim_type = detect_sim_type(raw)
    if is_noise_analysis(sim_type):
        return sim_type, "noise", "Hz", True
    if is_ac_analysis(sim_type):
        return sim_type, "ac", "Hz", True
    if is_dc_analysis(sim_type):
        return sim_type, "dc", "", False
    return sim_type, "transient", "s", False


def axis_may_descend(sim_type: str) -> bool:
    """Whether this analysis can legitimately produce a descending axis.

    Sweep analyses (DC, noise) may run high→low (e.g. ``.dc V1 5 0 -0.1``), so
    their callers opt into the ``window_and_clean`` flip. Time/frequency axes
    cannot descend, so a non-monotonic one there is corruption, not a sweep.
    """
    return is_dc_analysis(sim_type) or is_noise_analysis(sim_type)


def window_indices(axis: np.ndarray, ts: float | None, te: float | None) -> tuple[int, int]:
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


def apply_window(
    axis: np.ndarray,
    wave: np.ndarray,
    t_start: str | None,
    t_end: str | None,
    *,
    allow_descending: bool = False,
) -> tuple[np.ndarray, np.ndarray, int]:
    ts = parse_time(t_start, "t_start")
    te = parse_time(t_end, "t_end")
    try:
        return window_and_clean(axis, wave, ts, te, allow_descending=allow_descending)
    except ValueError as e:
        raise ResultError(str(e)) from e


def run_compute(compute, *args, **kwargs) -> dict:
    """Invoke a pure-compute function, re-raising ValueError as ResultError."""
    try:
        return compute(*args, **kwargs)
    except ValueError as e:
        raise ResultError(str(e)) from e


def read_log_warnings(raw_path: Path, log_path: Path | None = None) -> tuple[list[str], list[str]]:
    """``(unrecognized-variable warnings, run-level solve-failure lines)`` from
    the source's ``.log``.

    The first is per-signal: a typo'd or unsupported ``.save``'d ``@dev[param]``
    is written to the raw as a real-looking ``0.0`` trace, and the only tell
    that it's bogus is the simulator's unrecognized-variable warning. The second
    is run-wide: a singular/non-converged solve taints every value, so a read
    relays it whatever trace was asked for.
    """
    log_path = log_path or raw_path.with_suffix(".log")
    if not log_path.exists():
        return [], []
    diags = extract_log_diagnostics(log_path)
    unrecognized = [
        w
        for w in diags["warnings"]
        if "unrecognized" in w.lower() or "can't find" in w.lower() or "@" in w
    ]
    return unrecognized, services.solve_failure_lines(diags)


def unrecognized_matches(warning: str, signal: str) -> bool:
    """True when an unrecognized-variable ``warning`` concerns ``signal`` — by
    the signal's name or its ``@dev[param]`` token appearing in the warning. The
    only tell a ``.save``'d variable is bogus is its mention in the log, so this
    is how a read decides whether a 0.0 trace it touched is the real result."""
    sig_l = signal.lower()
    at_match = re.search(r"@[\w.]+\[[^\]]*\]", sig_l)
    at_token = at_match.group(0) if at_match else None
    return sig_l in warning.lower() or (at_token is not None and at_token in warning.lower())


async def solve_failures(source: services.AnalysisSource) -> list[str]:
    """Run-level solve-failure log lines (singular matrix / non-convergence).

    A failed-but-completed solve taints EVERY value in the raw, not one trace,
    so any read relays these regardless of the signal asked for. Empty when the
    solve finished clean or there is no ``.log``.
    """
    return (await asyncio.to_thread(read_log_warnings, source.raw, source.log))[1]


async def signal_log_warnings(source: services.AnalysisSource, signal: str) -> list[str]:
    """Warnings for a single-value read: the signal-filtered unrecognized-variable
    message (only when the queried trace IS the bogus one, matched by its
    ``@dev[param]`` token or the resolved name) plus any run-level solve
    failures."""
    unrecognized, failures = await asyncio.to_thread(read_log_warnings, source.raw, source.log)
    matched = [w for w in unrecognized if unrecognized_matches(w, signal)]
    warnings: list[str] = []
    if matched:
        warnings.append(
            f"Queried {signal!r}, but the simulator did not recognize it (see log) "
            "— this value is not a real result: " + "; ".join(matched)
        )
    warnings.extend(failures)
    return warnings


async def reraise_with_solve_failure(e: ResultError, source: services.AnalysisSource) -> NoReturn:
    """Re-raise ``e``, enriched with any terminal solve failure from the run's
    ``.log``. A failed-but-completed solve often leaves the data degenerate
    enough that the metric itself raises (e.g. no detectable edge), so the
    generic measurement error alone hides the solve failure that explains it."""
    failures = await solve_failures(source)
    if failures:
        raise ResultError(
            f"{e} — the simulator log reports a solve failure that likely explains "
            f"this: {'; '.join(failures)}",
            suggestions=e.suggestions or None,
            show_hint=False,
        ) from e
    raise e


async def run_metric(source: services.AnalysisSource, compute, *args, **kwargs) -> dict:
    """``run_compute`` for a raw-backed metric: on failure, name any terminal
    solve failure from the run's ``.log`` in the error so a degenerate-data
    raise points at the bad solve instead of just the missing feature."""
    try:
        return run_compute(compute, *args, **kwargs)
    except ResultError as e:
        await reraise_with_solve_failure(e, source)


async def relay_solve_failures(source: services.AnalysisSource, data: dict) -> dict:
    """Standard tail for a raw-backed metric: relay any run-level solve failure
    into ``data["warnings"]``, the channel these metrics surface. Routing every
    metric through here keeps the relay from being forgotten when one is added.
    """
    data.setdefault("warnings", []).extend(await solve_failures(source))
    return data


async def load_real_signal(
    source: services.AnalysisSource,
    signal: str,
    step: int,
    state: SessionState,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(axis, wave)`` for a transient signal, rejecting AC/complex data."""
    raw = await services.load_raw(source.raw, state)
    reject_non_transient(raw)
    signal = services.validate_signal(raw, signal)
    services.validate_step(raw, step)
    axis = guarded_axis(raw, step, source.raw)
    wave = np.asarray(raw.get_wave(signal, step=step))
    if np.iscomplexobj(wave):
        raise ResultError(
            f"Signal {signal!r} contains complex values; this tool requires "
            "real-valued transient data."
        )
    return axis, wave


def split_ratio(signal: str) -> tuple[str, str] | None:
    """Split a transfer-function ratio ``A/B`` into ``(A, B)``, or None if it
    isn't a ratio.

    Only a single ``/`` is supported — a two-signal quotient such as
    ``V(out)/V(mid)``. SPICE node names don't contain ``/``, so the operator is
    unambiguous. A ``/`` that doesn't yield exactly two non-empty operands is a
    malformed ratio, not a plain signal, so it raises.
    """
    if "/" not in signal:
        return None
    parts = [p.strip() for p in signal.split("/")]
    if len(parts) != 2 or not all(parts):
        raise ResultError(f"Ratio signal must be exactly 'A/B' (two signals); got {signal!r}.")
    return parts[0], parts[1]


async def load_ac_signal(
    source: services.AnalysisSource,
    signal: str,
    step: int,
    state: SessionState,
) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(freqs, H)`` for an AC signal. Rejects transient data.

    ``signal`` may be a single trace (``V(out)``) or a transfer-function ratio
    of two traces (``V(out)/V(mid)``), in which case the two complex AC waves
    are divided element-wise — the way to express an inter-stage gain, loop
    gain, or PSRR that the simulator never stores as its own trace.

    A single leading ``-`` negates the whole expression (``-V(out)/V(vsense)``
    is −(V(out)/V(vsense))): a 180° phase flip, so a loop gain probed with an
    inverting sense or a reversed impedance probe reads in its natural
    convention without a behavioral inverter node in the deck.
    """
    signal = signal.strip()
    negate = signal.startswith("-")
    if negate:
        signal = signal[1:].lstrip()
        if not signal:
            raise ResultError("Signal is just a '-'; expected '-V(node)' or '-A/B'.")
    raw = await services.load_raw(source.raw, state)
    sim_type = detect_sim_type(raw)
    if not is_ac_analysis(sim_type):
        # show_hint=False: precise redirect; the generic hint would misdirect.
        raise ResultError(
            f"This tool requires AC analysis data; got {sim_type!r}. "
            "Use signal_stats (transient) or run a .AC sweep first.",
            show_hint=False,
        )
    services.validate_step(raw, step)
    axis = np.asarray(raw.get_axis(step=step))

    ratio = split_ratio(signal)
    if ratio is not None:
        num_name = services.validate_signal(raw, ratio[0])
        den_name = services.validate_signal(raw, ratio[1])
        num = np.asarray(raw.get_wave(num_name, step=step))
        den = np.asarray(raw.get_wave(den_name, step=step))
        with np.errstate(divide="ignore", invalid="ignore"):
            wave = num / den
        # A null (or non-finite) denominator makes the ratio singular at those
        # bins — a genuine pole of the requested transfer function.
        # prepare_ac_arrays would silently drop them and analyze the rest,
        # hiding the pole and skewing every metric. Surface the singularity
        # (which frequencies, how many) instead of fabricating clean numbers.
        singular = ~np.isfinite(wave)
        if singular.any():
            axis_real = np.real(axis)
            example = float(axis_real[int(np.argmax(singular))])
            raise ResultError(
                f"Ratio {ratio[0]}/{ratio[1]} is singular at {int(singular.sum())} of "
                f"{singular.size} frequencies — the denominator {den_name} is ~0 there "
                f"(e.g. {example:.6g} Hz), so the transfer function has a pole. Narrow "
                f"the frequency window to exclude the null, or pick a denominator that "
                f"does not cross zero."
            )
    else:
        wave = np.asarray(raw.get_wave(services.validate_signal(raw, signal), step=step))
    if negate:
        wave = -wave
    try:
        return prepare_ac_arrays(axis, wave)
    except ValueError as e:
        raise ResultError(str(e)) from e


# ---------------------------------------------------------------------------
# Operating-point helpers
# ---------------------------------------------------------------------------


def has_active_device(currents: dict[str, float]) -> bool:
    """True if any branch-current name belongs to an M/Q/J/D device — the ones
    with a small-signal operating point (gm/gds/vth/...) — e.g. ``Id(M1)``,
    ``Ic(Q2)``.

    Public because it gates two channels — the operating-point value's own
    empty-op-point warning and the consolidated path's
    ``device_op_points_absent`` observation — and a second copy of the rule is
    how one of them speaks while the other stays silent about the same run.
    """
    for name in currents:
        lp = name.find("(")
        if lp != -1 and lp + 1 < len(name) and name[lp + 1].lower() in "mqjd":
            return True
    return False


_OP_INTERNAL_DEV_RE = re.compile(r"@([\w.:]+)\[")
_OP_TERMINAL_CUR_RE = re.compile(r"^i[a-z]?\(([^)]+)\)$")


def trace_device(name: str) -> str | None:
    """The device a .op trace belongs to: the ``@<dev>[...]`` op-point owner, or
    the ``X(<dev>)`` terminal-current owner. None for plain node voltages."""
    low = name.lower()
    m = _OP_INTERNAL_DEV_RE.search(low)
    if m:
        return m.group(1)
    m = _OP_TERMINAL_CUR_RE.match(low)
    return m.group(1) if m else None


def _dev_core_segments(owner: str) -> list[str]:
    """Identity segments of a .op trace owner: split on ``.``/``:``, drop a
    leading single-letter device-class token, and drop trailing region/
    multiplicity index segments. ngspice uses dots (``m.x1.mn`` -> x1, mn);
    LTspice subcircuit semiconductors use colons with a class prefix and trailing
    indices (``q:q2:1:2`` -> q2). Both callers want exactly this normalized form."""
    segs = [s for s in re.split(r"[.:]", owner) if s]
    if segs and len(segs[0]) == 1 and segs[0].isalpha():
        segs = segs[1:]
    while len(segs) > 1 and segs[-1].isdigit():
        segs.pop()
    return segs


def dev_instance(owner: str) -> str:
    """Reference token for the 'devices present' list: strips LTspice's class
    prefix and trailing region/multiplicity indices from colon-form subcircuit
    names (q:q2:1:2 -> q2). Dotted ngspice paths (m.x1.mn) and top-level refs
    (m1) pass through unchanged."""
    if ":" not in owner:
        return owner
    segs = _dev_core_segments(owner)
    return ":".join(segs) if segs else owner


def device_matches(owner: str, want: str) -> bool:
    """Whether a .op trace owner belongs to the requested device. Top-level and
    ngspice-dotted owners match by exact name or hierarchical path suffix
    (``m.x1.mn`` matches ``mn`` or ``x1.mn``). LTspice colon-form subcircuit
    owners (``q:q2:1:2``) match by instance segment, ignoring the class prefix
    and the trailing region/multiplicity indices."""
    if owner == want:
        return True
    if ":" in owner:
        segs = _dev_core_segments(owner)
        want_parts = [p for p in re.split(r"[.:]", want) if p]
        if not want_parts:
            return False
        if len(want_parts) == 1:
            return want_parts[0] in segs
        return segs[-len(want_parts) :] == want_parts
    return owner.endswith("." + want)


def filter_operating_point(op_data: dict, device: str) -> bool:
    """Narrow ``op_data`` (in place) to one device's operating-point params +
    terminal currents. Returns whether anything matched. Handles top-level
    devices (``m1``), ngspice subcircuit-qualified traces (``@m.x1.mn[gm]``
    matches bare ``mn`` via path suffix), and LTspice colon-form subcircuit
    semiconductors (``q:q2:1:2`` matches ``q2``)."""
    want = device.strip().lower()
    matched = False
    for bucket in ("voltages", "currents", "device_op_points"):
        kept = {}
        for name, value in op_data.get(bucket, {}).items():
            owner = trace_device(name)
            if owner is not None and device_matches(owner, want):
                kept[name] = value
                matched = True
        op_data[bucket] = kept
    return matched


def operating_point_units(raw, op_data: dict) -> dict[str, str]:
    """SI unit per returned trace name, only where the simulator typed it."""
    units: dict[str, str] = {}
    for bucket in ("voltages", "currents", "device_op_points"):
        for name in op_data.get(bucket, {}):
            unit = trace_unit(raw, name)
            if unit:
                units[name] = unit
    return units


def operating_point_flat(value: dict[str, Any]) -> dict[str, Any]:
    """Every named number an operating-point value carries, in one map."""
    flat: dict[str, Any] = {}
    for bucket in ("voltages", "currents", "device_op_points"):
        if isinstance(value.get(bucket), dict):
            flat.update(value[bucket])
    return flat


# ---------------------------------------------------------------------------
# .MEAS aggregation helpers
# ---------------------------------------------------------------------------

_RE_MEAS_WHEN = re.compile(r"\bwhen\b", re.IGNORECASE)
_RE_MEAS_FIND = re.compile(r"\bfind\b", re.IGNORECASE)
_RE_MEAS_HEAD = re.compile(
    r"^\.meas(?:ure)?(?:\s+(?:tran|ac|dc|op|sp|fft|noise))?\s+(?P<name>[A-Za-z0-9_$]+)",
    re.IGNORECASE,
)


def diagnostics_block(diags: list[str], empty_note: str) -> str:
    """Indent diagnostic lines for an error payload, or fall back to
    ``empty_note`` when there is nothing to relay."""
    if not diags:
        return f"  {empty_note}"
    return "\n".join(f"  {d}" for d in diags)


def meas_kinds_from_netlist(netlist: Path | None) -> dict[str, str]:
    """Operator kind per lowercase .MEAS name, read from the deck itself.

    The log alone cannot distinguish a WHEN crossing search from a FIND...AT
    probe — both print ``name: expr=value at X`` — so the shape heuristic in
    :func:`apply_when_axis_swap` has a documented degenerate misread. The deck
    is ground truth: relay the operator when the netlist is readable. Kinds:
    ``"when"`` (bare WHEN — value is the constant trigger level, the crossing
    in ``at`` is the payload) vs ``"value"`` (everything else, including
    FIND...WHEN, whose FIND result is the payload). Names defined inside
    includes stay absent and fall back to the heuristic.
    """
    if netlist is None:
        return {}
    try:
        from ltspice_mcp.lib.spice_lex import cards_from_path

        cards = cards_from_path(netlist).cards
    except Exception:
        return {}
    kinds: dict[str, str] = {}
    for card in cards:
        body = card.body
        m = _RE_MEAS_HEAD.match(body)
        if not m:
            continue
        rest = body[m.end() :]
        is_bare_when = bool(_RE_MEAS_WHEN.search(rest)) and not _RE_MEAS_FIND.search(rest)
        kinds[m.group("name").lower()] = "when" if is_bare_when else "value"
    return kinds


def apply_when_axis_swap(
    flat_values: dict[str, list[float | None]],
    at_map: dict[str, list[float | None]],
    kinds: dict[str, str] | None = None,
) -> dict[str, AggregatedField]:
    """Pick each .MEAS name's aggregation axis, swapping WHEN-style ones to ``at``.

    For a ``WHEN``/``AT`` .MEAS the per-sample ``values`` are the trigger level
    (constant by construction) while the interesting axis is the crossing time in
    ``at``. When the levels are constant (or all-None) and the ``at`` values vary,
    swap ``flat_values[name]`` to the ``at`` list and mark the axis ``"at"``;
    otherwise keep ``"value"``. Mutates ``flat_values`` in place for swapped
    names; returns the axis map.

    ``kinds`` (from :func:`meas_kinds_from_netlist`) relays the directive's
    real operator and overrides the shape inference: a known non-WHEN never
    swaps (kills the degenerate misread where a FIND probe constant to 12
    decimals across runs would aggregate the probe axis), and a known bare
    WHEN swaps whenever crossings are present. Names not in ``kinds`` fall back
    to the numeric-shape heuristic. The chosen axis is always reported via
    ``aggregated_field`` so the consumer can tell.
    """
    axis_map: dict[str, AggregatedField] = {}
    for name, vals in flat_values.items():
        ats = at_map.get(name) or []
        valid_vals = [v for v in vals if v is not None]
        valid_ats = [a for a in ats if a is not None]
        levels_constant = len({round(v, 12) for v in valid_vals}) <= 1
        ats_vary = len({round(a, 12) for a in valid_ats}) > 1
        kind = (kinds or {}).get(name.lower())
        if kind == "when":
            # A deck-confirmed bare WHEN aggregates crossing times whenever the
            # logs carry any — constant crossings (a deterministic batch) are
            # still the requested quantity, not a reason to fall back to the
            # constant trigger level.
            swap = bool(valid_ats)
        elif kind == "value":
            swap = False
        else:
            swap = levels_constant and ats_vary
        if swap:
            flat_values[name] = ats
            axis_map[name] = "at"
        else:
            axis_map[name] = "value"
    return axis_map


def aggregate_log_measurements(
    log_path: Path,
    netlist: Path | None = None,
) -> tuple[
    dict[str, list[float | None]],
    dict[str, AggregatedField],
    str,
    dict[str, list[float | None]],
]:
    """Aggregate the .MEAS results of ONE log file.

    The interesting case is a ``.step`` log, where each .MEAS name carries
    one value per step; a plain single-run log yields one value per name
    (honest n=1 stats).

    Returns ``(flat_values, axis_map, steps_label, at_map)`` — ``at_map`` is the
    per-step ``at`` list per name (unswapped), so the caller can echo the
    reported point for a single-sample read the swap can't classify.
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
        err_block = diagnostics_block(diags, "(log contained no .MEAS results and no diagnostics)")
        raise ResultError(f"No .MEAS results in log:\n{err_block}")

    flat_values = {name: list(entry.get("values", [])) for name, entry in measurements.items()}
    # Per-step ``at`` (crossing time) list per name, so a stepped WHEN .MEAS swaps
    # to the ``at`` axis exactly as the batch path does.
    at_map: dict[str, list[float | None]] = {}
    for name, entry in measurements.items():
        at_field = entry.get("at")
        if isinstance(at_field, list):
            at_map[name] = list(at_field)
        elif isinstance(at_field, int | float):
            at_map[name] = [float(at_field)]
        else:
            at_map[name] = []
    axis_map = apply_when_axis_swap(flat_values, at_map, meas_kinds_from_netlist(netlist))
    steps_label = f"{meas_data.get('step_count', 1)} step(s)"
    return flat_values, axis_map, steps_label, at_map


_RE_NOISE_HEAD = re.compile(r"^[ \t]*\.noise\s+(?P<out>\S+)\s+(?P<src>\S+)", re.IGNORECASE)


def noise_input_source_unit(netlist: Path | None) -> str | None:
    """SI unit implied by a ``.NOISE`` directive's input-source refdes.

    ``.NOISE <output> <src> ...`` names the source inoise is referred to, but
    the trace name alone doesn't say whether that source is voltage or
    current: LTspice always spells the trace ``V(inoise)`` even when
    ``<src>`` is a current source, and ngspice's ``inoise_spectrum`` carries
    no prefix at all. Reads the deck's own ``.NOISE`` line as ground truth —
    a source name starting with V is a voltage source ("V"), I a current
    source ("A"), case-insensitively. Returns None when the deck is
    unavailable, has no ``.NOISE`` directive, or the answer is ambiguous —
    more than one ``.NOISE`` directive that don't agree on the source type, or
    an unrecognized source prefix: the caller then falls back to the existing
    trace-derived unit rather than guessing which directive produced this raw.
    """
    if netlist is None:
        return None
    try:
        from ltspice_mcp.lib.spice_lex import cards_from_path

        cards = cards_from_path(netlist).cards
    except Exception:
        return None
    units: set[str | None] = set()
    for card in cards:
        m = _RE_NOISE_HEAD.match(card.body)
        if not m:
            continue
        prefix = m.group("src")[:1].upper()
        units.add("V" if prefix == "V" else "A" if prefix == "I" else None)
    # Resolve only when every .NOISE directive agrees on one recognized type;
    # otherwise it's ambiguous and we let the trace-derived unit stand.
    if units == {"V"}:
        return "V"
    if units == {"A"}:
        return "A"
    return None


def query_x_label(raw, sim_type: str) -> str:
    """Axis label for a point read: ``f`` for AC, ``t`` for transient, and the
    swept variable's own name for a .dc sweep (not a misleading ``t``)."""
    if is_ac_analysis(sim_type):
        return "f"
    if is_dc_analysis(sim_type):
        name, _ = dc_axis_name(raw)
        return name or "x"
    return "t"


def snap_match(requested: float, actual: float, *, rtol: float = 1e-3) -> bool:
    """True iff ``actual`` is within ``rtol`` (relative) of ``requested``.

    A query snaps to the nearest available sample (a discrete step value or the
    nearest point on a sweep axis); a legitimate lookup lands on (or extremely
    near) one. A large gap means the request fell outside the range and was
    silently clamped to the nearest endpoint — worth flagging rather than
    presenting the clamp as an exact answer. Shared by the step-axis lookup and
    the ``value`` recipe's direct ``at`` path so their snap flags can't drift.
    """
    scale = max(abs(actual), abs(requested), 1e-30)
    return abs(requested - actual) <= rtol * scale


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------


async def summary(
    source: services.AnalysisSource,
    recipe: SummaryRecipe,
    step: int,
    state: SessionState,
    *,
    signal: str | None = None,
) -> MetricValue:
    """Sim type, axis range, trace list, .MEAS results and run diagnostics."""
    raw = await services.load_raw(source.raw, state)
    # Honor ``step`` for the summary itself (range/point_count), not just for
    # ac_bandwidth_metrics. Validate up front so an out-of-range step errors
    # clearly instead of being silently ignored.
    services.validate_step(raw, step)

    # Re-inspection parity with the run-completion summary: a run's own netlist
    # is known, so parse the same requested-outputs and source-amplitude facts
    # the observation surfacer is fed at completion — the same raw must yield
    # the same observations here. A bare raw path has no netlist to trust, so
    # those observations stay unarmed on that path.
    requested = None
    source_amplitudes = None
    if source.netlist is not None:
        requested, source_amplitudes = await asyncio.to_thread(
            deck_observation_inputs, source.netlist
        )

    log_path = source.log if source.log is not None and source.log.exists() else None
    try:
        # ``raw`` here is fully loaded (services.load_raw reads all traces), so
        # the value scan is affordable and surfaces NaN/extreme-value facts.
        facts = await services.bounded_parse(
            source.raw,
            lambda: build_simulation_summary(
                raw,
                log_path,
                None,
                step=step,
                value_scan="scan",
                requested=requested,
                source_amplitudes=source_amplitudes,
            ),
        )
    except ResultError:
        raise
    except Exception as e:
        # Suppress the generic ResultError hint — it points at the summary
        # recipe, which is the thing that just failed (self-referential).
        raise ResultError(f"Failed to build summary: {e}", show_hint=False) from e

    suggestions = services.suggestions_from_errors(facts.get("errors"), state.libraries)
    if suggestions:
        facts["suggestions"] = suggestions

    # Compute AC bandwidth metrics on AC raws. When ``signal`` is omitted,
    # auto-pick the first V(...) trace and warn — silently dropping
    # ac_bandwidth_metrics leaves the caller wondering why their AC summary had
    # no metrics.
    ac_metrics = None
    ac_signal_used: str | None = None
    if is_ac_analysis(facts["sim_type"]):
        ac_signal_used = signal
        if ac_signal_used is None:
            ac_signal_used = next(
                (t for t in facts["signals"] if t.upper().startswith("V(")), None
            )
            if ac_signal_used is not None:
                facts.setdefault("warnings", []).append(
                    f"AC summary built without an explicit ``signal``; "
                    f"defaulted to {ac_signal_used!r} for ac_bandwidth_metrics. "
                    "Pass ``signal=`` to choose a different trace."
                )
        if ac_signal_used:
            with contextlib.suppress(Exception):
                ac_metrics = compute_ac_bandwidth_metrics(raw, ac_signal_used, step)
            # On a multi-step run the metric is for one step only; the bare
            # number next to step_count=N otherwise reads as the whole-run
            # answer (it isn't — it's wrong for the other N-1 steps).
            if ac_metrics is not None and facts.get("step_count", 1) > 1:
                ac_metrics["step"] = step
                facts.setdefault("warnings", []).append(
                    f"ac_bandwidth_metrics is for step {step} of "
                    f"{facts['step_count']}; pass step=N for other steps."
                )

    data = dict(facts)
    if ac_metrics:
        data["ac_bandwidth_metrics"] = ac_metrics
    if ac_signal_used and ac_signal_used != signal:
        data["ac_signal_used"] = ac_signal_used
    return data


async def measurements(
    source: services.AnalysisSource,
    recipe: MeasurementsRecipe,
    step: int,
    state: SessionState,
    *,
    measurement: str | None = None,
) -> MetricValue:
    """Aggregated .MEAS results from the run's log, keyed by measurement name."""
    log_path = source.log
    if log_path is None or not log_path.is_file():
        raise ResultError("source has no log artifact for .MEAS results")
    flat_values, axis_map, _steps_label, at_map = await services.bounded_parse(
        log_path,
        lambda: aggregate_log_measurements(log_path, source.netlist),
    )
    stats = run_compute(
        compute_measurement_stats,
        flat_values,
        histogram_bins=recipe.histogram_bins,
        measurement=measurement,
    )
    # Surface which field each stat block was computed from so a downstream
    # consumer can tell "this is the level (constant)" from "this is the
    # WHEN-clause crossing frequency".
    for name, entry in stats.items():
        entry["aggregated_field"] = axis_map.get(name, "value")
        # n=1 can't trigger the variation-based WHEN swap, so a single-run .meas
        # with an AT/crossing would report only its value. Echo the reported
        # ``at`` too — callers use this to "read my .meas", and the
        # time/frequency is often the answer they want.
        if entry["aggregated_field"] == "value" and entry.get("total_count") == 1:
            crossings = [a for a in at_map.get(name, []) if a is not None]
            if crossings:
                entry["at"] = crossings[0]
    if recipe.names is not None:
        stats = {name: entry for name, entry in stats.items() if name in recipe.names}
    return {"stats": stats}


async def value(
    source: services.AnalysisSource,
    recipe: ValueRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """One signal's value at one point on the run's primary axis.

    With no ``at``, a run whose axis holds a single sample answers from that
    sample; a run with no axis at all falls back to its bias point, where the
    expression is addressed by the name the operating point carries.
    """
    at = recipe.at
    if at is None:
        raw = await services.load_raw(source.raw, state)
        services.validate_step(raw, step)
        try:
            axis = guarded_axis(raw, step, source.raw)
        except ResultError:
            return await _value_from_operating_point(source, recipe, step, state)
        if len(axis) != 1:
            raise ResultError(
                "value.at is required when the selected run has more than one "
                "sample on its primary axis"
            )
        at = float(np.real(axis[0]))
    return await point_value(source, recipe.expr, spice_text(at), step, state)


async def _value_from_operating_point(
    source: services.AnalysisSource,
    recipe: ValueRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """A no-axis run's answer for ``recipe.expr``, read off its bias point."""
    op = await operating_point(
        source, OperatingPointRecipe(key=recipe.key, metric="operating_point"), step, state
    )
    flat = operating_point_flat(op)
    # The 'm1.gm' shorthand the guide and read_device_op_points both promise
    # resolves here too, not only against a raw's trace list: an op-point value
    # read through this path is the one case where the params come from the
    # .log and never appear as a trace, so refusing the documented spelling
    # here refuses it everywhere it is the only route.
    by_lower = {name.lower(): (name, item) for name, item in flat.items()}
    match = next(
        (
            by_lower[form]
            for form in (recipe.expr.lower(), *services.device_param_forms(recipe.expr))
            if form in by_lower
        ),
        None,
    )
    if match is None:
        present = ", ".join(sorted(flat)[:8])
        if len(flat) > 8:
            present += f", ... ({len(flat)} total)"
        raise ResultError(
            f"{recipe.expr!r} is not present in this operating-point "
            "result. Address a value by the name it carries (e.g. "
            "'@m1[gm]', 'V(out)'); a TOP-LEVEL device also accepts the "
            "'m1.gm' shorthand, but a subcircuit device keeps LTspice's "
            "colon-qualified name ('@q:q2:1:2[gm]') and must be named "
            f"literally. Present here: {present}"
        ) from None
    name, item = match
    return {
        "signal": name,
        "value": item,
        "unit": None,
        "warnings": op.get("warnings", []),
    }


async def point_value(
    source: services.AnalysisSource,
    signal: str,
    at: str | None,
    step: int,
    state: SessionState,
) -> MetricValue:
    """The nearest-sample read behind the ``value`` recipe's ``at`` form."""
    if at is None:
        raise ResultError(
            "value recipe: 'at' is required (or use step_axis + step_value). "
            "If this is a .op (operating point) result, it has no time/frequency "
            "axis — use operating_point instead.",
            show_hint=False,
        )
    try:
        target_x = parse_spice_value(at)
    except ValueError as e:
        raise ResultError(f"Invalid 'at' value: {e}", show_hint=False) from e

    # np.searchsorted treats NaN as greater than everything and returns the
    # last index, which looks like a valid result but isn't.
    if not math.isfinite(target_x):
        raise ResultError(
            f"'at' value must be finite, got {at!r} (parsed as {target_x})", show_hint=False
        )

    raw = await services.load_raw(source.raw, state)
    resolved = services.validate_signal(raw, signal)
    services.validate_step(raw, step)

    try:
        result_data = query_point_value(raw, resolved, target_x, step)
    except NoAxisError as e:
        # Operating-point raws have no time/frequency axis. Give a precise,
        # actionable message instead of the generic failure.
        raise ResultError(
            "This is an Operating Point result (no time/frequency axis to "
            "query). Use operating_point to read node voltages and branch "
            "currents.",
            show_hint=False,
        ) from e
    except Exception as e:
        raise ResultError(f"Failed to query value: {e}") from e

    sim_type = detect_sim_type(raw)
    value_unit = trace_unit(raw, resolved)
    if value_unit and is_noise_analysis(sim_type):
        # .noise traces are amplitude spectral density (V/√Hz, A/√Hz), not the
        # plain V/A the trace's whattype declares — match noise_integral and the
        # raw's own "Noise Spectral Density" plotname.
        value_unit = f"{value_unit}/√Hz"

    # The query snaps to the nearest sample; flag when that snap moved the
    # requested point. On a coarse sweep this matters — e.g. a .dc temp sweep
    # snapping 27 → 25 °C silently biases a tempco measurement.
    exact_match = snap_match(float(result_data["requested_x"]), float(result_data["actual_x"]))

    data: MetricValue = {"signal": resolved, **result_data, "exact_match": exact_match}
    if value_unit:
        data["unit"] = value_unit
    data.setdefault("warnings", []).extend(await signal_log_warnings(source, resolved))
    return data


async def signal_stats(
    source: services.AnalysisSource,
    recipe: SignalStatsRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Min/max/mean/RMS and window facts for one trace over one window."""
    t_start, t_end = window_bounds(recipe.window)
    raw = await services.load_raw(source.raw, state)
    signal = services.validate_signal(raw, recipe.signal)
    services.validate_step(raw, step)
    # A failed-but-completed solve makes every stat below garbage; relay it.
    failures = await solve_failures(source)

    try:
        wave = raw.get_wave(signal, step=step)
    except Exception as e:
        raise ResultError(f"Failed to read signal {signal!r}: {e}") from e
    if len(wave) == 0:
        raise ResultError(
            f"Signal {signal!r} has no data points at step {step}; cannot compute statistics."
        )

    if np.iscomplexobj(wave):
        if t_start is not None or t_end is not None:
            raise ResultError(
                "t_start/t_end windowing is not supported for AC analysis. "
                "Use the bode_point recipe to look up a specific frequency.",
                show_hint=False,
            )
        magnitude_db = safe_magnitude_db(wave)
        phase_deg = np.angle(wave, deg=True)
        data: MetricValue = {
            "signal": signal,
            "analysis_type": "ac",
            "min_db": float(np.min(magnitude_db)),
            "max_db": float(np.max(magnitude_db)),
            "mean_db": float(np.mean(magnitude_db)),
            "min_phase": float(np.min(phase_deg)),
            "max_phase": float(np.max(phase_deg)),
            "point_count": len(wave),
        }
        if failures:
            data["warnings"] = failures
        return data

    axis = guarded_axis(raw, step, source.raw)
    wave_real = np.asarray(wave)

    # Distinguish DC sweep (axis = sweep variable, e.g. ``temp``)
    # from transient (axis = time). Trapezoidal mean/RMS over a sweep axis
    # is mathematically meaningless; the t_start/t_end labels are misleading
    # too since the units aren't seconds.
    sim_type_raw = detect_sim_type(raw)
    is_dc_sweep = is_dc_analysis(sim_type_raw)
    is_noise = is_noise_analysis(sim_type_raw)

    if is_noise and (t_start is not None or t_end is not None):
        raise ResultError(
            "t_start/t_end windowing is not supported for Noise analysis (axis is "
            "frequency, not time). Use the value recipe to look up a specific "
            "frequency.",
            show_hint=False,
        )

    ts = parse_time(t_start, "t_start")
    te = parse_time(t_end, "t_end")
    try:
        t_win, y_win, _ = window_and_clean(
            axis, wave_real, ts, te, allow_descending=axis_may_descend(sim_type_raw)
        )
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
        stats: dict[str, Any] = {
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
            "t_at_min": core["t_at_min"],
            "t_at_max": core["t_at_max"],
        }

    # Surface a FACT (not a verdict) when the signal never moves across the
    # window. min == max is the tell of a coerced/latched solve (e.g. a
    # self-biased stage stuck at its trivial DC state) that still completes
    # cleanly — the model, not this code, decides whether that is expected.
    observations: list[dict[str, Any]] = []
    if core["min"] == core["max"]:
        observations.append(
            {
                "code": "constant_window",
                "kind": "value",
                "detail": (
                    f"Signal is constant across the analyzed window "
                    f"(min == max == {core['min']:.6g}); peak-to-peak is 0."
                ),
            }
        )

    data = {"signal": signal, **stats}
    if observations:
        data["observations"] = observations
    if failures:
        data["warnings"] = failures
    return data


async def edges(
    source: services.AnalysisSource,
    recipe: EdgesRecipe,
    step: int,
    state: SessionState,
    *,
    edge_index: int = 0,
    low_pct: float = 10.0,
    high_pct: float = 90.0,
) -> MetricValue:
    """Rise/fall time and slew rate of one edge in the window."""
    t_start, t_end = window_bounds(recipe.window)
    axis, wave = await load_real_signal(source, recipe.signal, step, state)
    t, y, _ = apply_window(axis, wave, t_start, t_end)
    levels = recipe.levels
    data = await run_metric(
        source,
        analyze_edge,
        t,
        y,
        edge=recipe.edge,
        edge_index=edge_index,
        low_pct=low_pct,
        high_pct=high_pct,
        low_level=levels.low if levels else None,
        high_level=levels.high if levels else None,
    )
    data["signal"] = recipe.signal
    return await relay_solve_failures(source, data)


async def pulse_response(
    source: services.AnalysisSource,
    signal: str,
    t_start: str | None,
    t_end: str | None,
    step: int,
    state: SessionState,
    *,
    initial_value: float | None = None,
    final_value: float | None = None,
    settling_tolerance_pct: float = 2.0,
) -> MetricValue:
    axis, wave = await load_real_signal(source, signal, step, state)
    t, y, _ = apply_window(axis, wave, t_start, t_end)
    data = await run_metric(
        source,
        analyze_pulse_response,
        t,
        y,
        initial_value=initial_value,
        final_value=final_value,
        settling_tolerance_pct=settling_tolerance_pct,
    )
    data["signal"] = signal
    return await relay_solve_failures(source, data)


async def disturbance_response(
    source: services.AnalysisSource,
    signal: str,
    t_start: str | None,
    t_end: str | None,
    step: int,
    state: SessionState,
    *,
    baseline: float | None = None,
    settle_band: float | None = None,
    settle_band_pct: float = 2.0,
) -> MetricValue:
    axis, wave = await load_real_signal(source, signal, step, state)
    t, y, _ = apply_window(axis, wave, t_start, t_end)
    data = await run_metric(
        source,
        analyze_disturbance_response,
        t,
        y,
        baseline=baseline,
        settle_band=settle_band,
        settle_band_pct=settle_band_pct,
    )
    data["signal"] = signal
    return await relay_solve_failures(source, data)


async def transient_response(
    source: services.AnalysisSource,
    recipe: TransientResponseRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Step-settling or disturbance-recovery metrics off a transient trace.

    In ``disturbance`` mode with no window start, the window opens at the
    largest transition of the named reference input — the load step itself —
    and the value says so, because a recovery time measured from the start of
    the run answers a different question than the one that was asked.
    """
    t_start, t_end = window_bounds(recipe.window)
    reference_observation: str | None = None
    if recipe.mode == "disturbance":
        assert recipe.input is not None
        raw = await services.load_raw(source.raw, state)
        reference = services.validate_signal(raw, recipe.input)
        if t_start is None:
            axis = guarded_axis(raw, step, source.raw)
            wave = np.asarray(raw.get_wave(reference, step=step))
            if np.iscomplexobj(wave) or len(wave) < 2:
                raise ResultError(
                    "The disturbance reference input must be a real trace "
                    "with at least two samples."
                )
            edge_index = int(np.argmax(np.abs(np.diff(wave)))) + 1
            t_start = spice_text(float(axis[edge_index]))
            reference_observation = (
                f"Disturbance window starts at {t_start}s, the largest transition "
                f"in reference input {reference!r}."
            )
    if recipe.mode == "step":
        data = await pulse_response(source, recipe.signal, t_start, t_end, step, state)
    else:
        data = await disturbance_response(source, recipe.signal, t_start, t_end, step, state)
    if reference_observation is not None:
        data.setdefault("warnings", []).append(reference_observation)
    return data


async def timing(
    source: services.AnalysisSource,
    recipe: TimingRecipe,
    step: int,
    state: SessionState,
    *,
    threshold_pct: float = 50.0,
) -> MetricValue:
    """Signed delay between threshold crossings of two transient traces."""
    t_start, t_end = window_bounds(recipe.window)
    raw = await services.load_raw(source.raw, state)
    reject_non_transient(raw)
    sig_a = services.validate_signal(raw, recipe.from_.signal)
    sig_b = services.validate_signal(raw, recipe.to.signal)
    services.validate_step(raw, step)

    axis = guarded_axis(raw, step, source.raw)
    ya_full = np.asarray(raw.get_wave(sig_a, step=step))
    yb_full = np.asarray(raw.get_wave(sig_b, step=step))
    if np.iscomplexobj(ya_full) or np.iscomplexobj(yb_full):
        raise ResultError(
            "Signals contain complex values; this tool requires real-valued transient data."
        )

    ts = parse_time(t_start, "t_start")
    te = parse_time(t_end, "t_end")
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

    data = await run_metric(
        source,
        analyze_timing_between,
        t_a_arr,
        ya,
        yb,
        threshold_a=recipe.from_.level,
        threshold_b=recipe.to.level,
        threshold_pct=threshold_pct,
        direction_a=recipe.from_.edge,
        direction_b=recipe.to.edge,
        nth=recipe.nth,
    )
    data["signal_a"] = recipe.from_.signal
    data["signal_b"] = recipe.to.signal
    return await relay_solve_failures(source, data)


async def periodic(
    source: services.AnalysisSource,
    recipe: PeriodicRecipe,
    step: int,
    state: SessionState,
    *,
    threshold: float | None = None,
    min_periods: int = 2,
) -> MetricValue:
    """Period, frequency, duty cycle and jitter of a repetitive trace."""
    t_start, t_end = window_bounds(recipe.window)
    axis, wave = await load_real_signal(source, recipe.signal, step, state)
    t, y, _ = apply_window(axis, wave, t_start, t_end)
    data = await run_metric(
        source,
        analyze_periodic,
        t,
        y,
        threshold=threshold,
        min_periods=min_periods,
    )
    data["signal"] = recipe.signal
    return await relay_solve_failures(source, data)


async def thd(
    source: services.AnalysisSource,
    recipe: ThdRecipe,
    step: int,
    state: SessionState,
    *,
    window: Literal["coherent", "hann"] = "coherent",
) -> MetricValue:
    """Total harmonic distortion of a transient trace over its window."""
    t_start, t_end = window_bounds(recipe.window)
    axis, wave = await load_real_signal(source, recipe.signal, step, state)
    t, y, _ = apply_window(axis, wave, t_start, t_end)
    f0 = parse_time(spice_text(recipe.fundamental_hz), "fundamental")
    data = await run_metric(
        source,
        analyze_thd,
        t,
        y,
        fundamental=f0,
        n_harmonics=recipe.harmonics,
        window=window,
    )
    data["signal"] = recipe.signal
    # Label the per-harmonic magnitudes with the signal's native unit (load_raw
    # is cached — load_real_signal already read this raw).
    unit = trace_unit(await services.load_raw(source.raw, state), recipe.signal)
    if unit:
        data["unit"] = unit
    return await relay_solve_failures(source, data)


async def filter_metrics(
    source: services.AnalysisSource,
    signal: str,
    step: int,
    state: SessionState,
    *,
    ref_db: float = -3.0,
    flatness_db: float = 1.0,
    passband_range: list[str] | None = None,
    stopband_range: list[str] | None = None,
) -> MetricValue:
    freqs, h = await load_ac_signal(source, signal, step, state)
    data = run_compute(
        compute_filter_metrics,
        freqs,
        h,
        ref_db=ref_db,
        flatness_db=flatness_db,
        passband_range=parse_freq_pair(passband_range, "passband_range"),
        stopband_range=parse_freq_pair(stopband_range, "stopband_range"),
    )
    data["signal"] = signal
    return data


async def gain_at(
    source: services.AnalysisSource,
    signal: str,
    frequencies: list[str],
    step: int,
    state: SessionState,
    *,
    include_unwrapped_phase: bool = False,
) -> MetricValue:
    if not frequencies:
        raise ResultError("frequencies list is empty")
    if len(frequencies) > 1000:
        raise ResultError(f"Too many frequencies ({len(frequencies)}); cap is 1000")
    freqs_q = [parse_freq(f, "frequency") for f in frequencies]
    freqs, h = await load_ac_signal(source, signal, step, state)
    points, warnings = run_compute(
        gain_at_frequencies,
        freqs,
        h,
        freqs_q,
        include_unwrapped_phase=include_unwrapped_phase,
    )
    return {"signal": signal, "points": points, "warnings": warnings}


async def find_crossing(
    source: services.AnalysisSource,
    signal: str,
    quantity: Quantity,
    level: float,
    step: int,
    state: SessionState,
    *,
    direction: SearchDirection = "any",
    f_start: str | None = None,
    f_end: str | None = None,
    max_results: int = 10,
    min_separation_decades: float = 0.0,
) -> MetricValue:
    freqs, h = await load_ac_signal(source, signal, step, state)
    if max_results < 1 or max_results > 1000:
        raise ResultError(f"max_results must be in [1, 1000], got {max_results}")
    try:
        crossings, warnings = find_crossings_any_quantity(
            freqs,
            h,
            quantity=quantity,
            level=level,
            direction=direction,
            f_start=parse_freq(f_start, "f_start") if f_start else None,
            f_end=parse_freq(f_end, "f_end") if f_end else None,
            max_results=max_results,
            min_separation_decades=min_separation_decades,
        )
    except ValueError as e:
        raise ResultError(str(e)) from e
    return {
        "signal": signal,
        "quantity": quantity,
        "level": level,
        "direction": direction,
        "crossings": crossings,
        "warnings": warnings,
    }


async def roll_off(
    source: services.AnalysisSource,
    signal: str,
    f_low: str,
    f_high: str,
    step: int,
    state: SessionState,
) -> MetricValue:
    f_lo = parse_freq(f_low, "f_low")
    f_hi = parse_freq(f_high, "f_high")
    freqs, h = await load_ac_signal(source, signal, step, state)
    data = run_compute(compute_roll_off, freqs, h, f_low=f_lo, f_high=f_hi)
    data["signal"] = signal
    return data


async def _bode(
    source: services.AnalysisSource,
    compute: Coroutine[Any, Any, MetricValue],
) -> MetricValue:
    """Run one bode-mode read, enriching a failure with any solve failure and
    relaying a run-level one into the value's warnings."""
    try:
        data = await compute
    except ResultError as e:
        await reraise_with_solve_failure(e, source)
    return await relay_solve_failures(source, data)


async def bode_filter(
    source: services.AnalysisSource,
    recipe: BodeFilterRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Passband, cutoffs, rejection and roll-off of an AC magnitude response."""
    return await _bode(source, filter_metrics(source, recipe.signal, step, state))


async def bode_point(
    source: services.AnalysisSource,
    recipe: BodePointRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Gain and phase at one frequency of an AC response."""
    return await _bode(
        source,
        gain_at(source, recipe.signal, [format_spice_value(recipe.at_hz)], step, state),
    )


async def bode_crossing(
    source: services.AnalysisSource,
    recipe: BodeCrossingRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Every frequency where an AC response crosses a magnitude or phase level."""
    quantity: Quantity = "phase_deg" if recipe.level_deg is not None else "magnitude_db"
    level = recipe.level_deg if recipe.level_deg is not None else recipe.level_db
    assert level is not None
    return await _bode(
        source,
        find_crossing(source, recipe.signal, quantity, level, step, state),
    )


async def bode_slope(
    source: services.AnalysisSource,
    recipe: BodeSlopeRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Magnitude slope in dB/decade between two frequencies."""
    return await _bode(
        source,
        roll_off(
            source,
            recipe.signal,
            format_spice_value(recipe.from_hz),
            format_spice_value(recipe.to_hz),
            step,
            state,
        ),
    )


async def stability(
    source: services.AnalysisSource,
    recipe: StabilityRecipe,
    step: int,
    state: SessionState,
    *,
    min_separation_decades: float = 0.1,
) -> MetricValue:
    """Loop-gain margins: unity-gain crossovers, phase margin, gain margin."""
    freqs, h = await load_ac_signal(source, recipe.signal, step, state)
    data = await run_metric(
        source,
        compute_stability_metrics,
        freqs,
        h,
        min_separation_decades=min_separation_decades,
    )
    data["signal"] = recipe.signal
    return await relay_solve_failures(source, data)


async def ac_structure(
    source: services.AnalysisSource,
    recipe: AcStructureRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Pole/zero structure read off an AC response: order, corners, excess phase."""
    freqs, h = await load_ac_signal(source, recipe.signal, step, state)
    data = await run_metric(source, analyze_ac_structure, freqs, h)
    data["signal"] = recipe.signal
    data.setdefault("observations", []).extend(
        relay_observations({"errors": await solve_failures(source)})
    )
    return data


async def resonance(
    source: services.AnalysisSource,
    recipe: ResonanceRecipe,
    step: int,
    state: SessionState,
    *,
    min_prominence_db: float = 3.0,
    min_separation_decades: float = 0.2,
    max_peaks: int = 20,
) -> MetricValue:
    """Resonant peaks of an AC magnitude response, with Q and -3 dB bandwidth."""
    if max_peaks < 1 or max_peaks > 1000:
        raise ResultError(f"max_peaks must be in [1, 1000], got {max_peaks}")
    freqs, h = await load_ac_signal(source, recipe.signal, step, state)
    data = await run_metric(
        source,
        compute_resonances,
        freqs,
        h,
        min_prominence_db=min_prominence_db,
        min_separation_decades=min_separation_decades,
        max_peaks=max_peaks,
    )
    data["signal"] = recipe.signal
    return await relay_solve_failures(source, data)


async def return_loss(
    source: services.AnalysisSource,
    recipe: ReturnLossRecipe,
    step: int,
    state: SessionState,
    *,
    at: str | None = None,
) -> MetricValue:
    """Return loss, VSWR and Γ from an AC input-impedance trace."""
    freqs, h = await load_ac_signal(source, recipe.signal, step, state)
    data = await run_metric(
        source,
        compute_return_loss,
        freqs,
        h,
        z0=recipe.z0,
        at_hz=parse_freq(at, "at") if at else None,
    )
    data["signal"] = recipe.signal
    data["z0_ohm"] = recipe.z0
    return await relay_solve_failures(source, data)


async def noise_integral(
    source: services.AnalysisSource,
    recipe: NoiseIntegralRecipe,
    step: int,
    state: SessionState,
) -> MetricValue:
    """Total RMS noise integrated over a band of a .noise spectral density."""
    raw = await services.load_raw(source.raw, state)
    sim_type = detect_sim_type(raw)
    if not is_noise_analysis(sim_type):
        raise ResultError(
            f"noise_integral needs a .noise result; got {sim_type!r}. Run a noise "
            "analysis (LTspice .noise / ngspice noise) first.",
            show_hint=False,
        )
    services.validate_step(raw, step)
    signal = services.validate_signal(raw, recipe.signal or "onoise")
    freqs = real_axis(np.asarray(raw.get_axis(step=step)))
    density = np.asarray(raw.get_wave(signal, step=step))
    data = await run_metric(
        source,
        integrate_noise,
        freqs,
        density,
        parse_time(spice_text(recipe.from_hz), "f_start"),
        parse_time(spice_text(recipe.to_hz), "f_end"),
    )

    unit = trace_unit(raw, signal)
    if "inoise" in signal.lower():
        # trace_unit() alone can't distinguish a voltage- from a
        # current-referred inoise trace (see noise_input_source_unit); check
        # the deck's .NOISE line when the run carries one.
        resolved = noise_input_source_unit(source.netlist)
        if resolved is not None:
            unit = resolved
        else:
            data.setdefault("warnings", []).append(
                "Could not verify the input-referred noise unit against the "
                f"deck's .NOISE source; assuming {unit or 'V'!r}. Analyze a run "
                "whose deck is known so the .NOISE line can be checked "
                "(V-source -> V, I-source -> A)."
            )
    data["signal"] = signal
    data["unit"] = unit or ""
    data["density_unit"] = f"{unit}/√Hz" if unit else "amplitude/√Hz"
    return await relay_solve_failures(source, data)


async def operating_point(
    source: services.AnalysisSource,
    recipe: OperatingPointRecipe,
    step: int,
    state: SessionState,
    *,
    at: str | None = None,
) -> MetricValue:
    """DC bias point: node voltages, branch currents, per-device small-signal params."""
    raw = await services.load_raw(source.raw, state)

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

    services.validate_step(raw, step)
    op_step_count = get_step_count(raw)

    is_dc = "transfer" in sim_lower or "dc" in sim_lower.split()

    # For a .dc sweep, at=<value> reads the full bias snapshot at a chosen sweep
    # point (nearest) instead of the sweep's first point.
    point_index = 0
    sweep_value: float | None = None
    if at is not None:
        try:
            at_value = parse_spice_value(at)
        except Exception as e:
            raise ResultError(f"Invalid 'at' value {at!r}: {e}", show_hint=False) from e
        axis = real_axis(np.asarray(raw.get_axis(step=step)))
        if axis.size > 1:
            point_index = nearest_index(axis, at_value)
            sweep_value = float(axis[point_index])

    try:
        op_data: dict[str, Any] = dict(
            extract_operating_point(raw, step=step, point_index=point_index)
        )
    except Exception as e:
        raise ResultError(f"Failed to extract operating point: {e}") from e

    op_data["step"] = step
    op_data["step_count"] = op_step_count
    # Always present so a clean run reads as "no warnings", not a missing key.
    op_data["warnings"] = []

    # LTspice exposes per-device small-signal params (gm/gds/vth/vdsat/caps) in
    # the .log under '.options logopinfo', not as @dev[param] raw traces the way
    # ngspice does. Fold the log block into device_op_points — keyed in the same
    # @dev[param] form — so they read back by name (m1.gm) on either simulator.
    # ngspice logs never carry the block, so skip the read entirely there.
    # Dialect resolved per raw: a per-run simulator override can differ from
    # the session default. Don't clobber a value the raw gave.
    raw_dialect = source.dialect or services.raw_dialect_for(source.raw, state)
    if raw_dialect != "ngspice":
        log_op_points = (
            await asyncio.to_thread(read_device_op_points, source.log)
            if source.log is not None
            else {}
        )
        if log_op_points:
            di = op_data.setdefault("device_op_points", {})
            for key, item in log_op_points.items():
                di.setdefault(key, item)

    # device= narrows to one device's op-point params + terminal currents.
    # Refuse with the list of devices present rather than returning a
    # misleading empty result when nothing matches. ``available_devs`` is
    # gathered before the filter, which empties the buckets in place.
    if recipe.device is not None:
        available_devs = sorted(
            {
                dev_instance(d)
                for bucket in ("currents", "device_op_points")
                for name in op_data.get(bucket, {})
                if (d := trace_device(name)) is not None
            }
        )
        if not filter_operating_point(op_data, recipe.device):
            dev_list = ", ".join(available_devs) if available_devs else "none found"
            raise ResultError(
                f"No operating-point params or terminal currents for device "
                f"{recipe.device!r} in this result. Devices present: {dev_list}.",
                show_hint=False,
            )
        op_data["device"] = recipe.device

    # Unit per trace, only where the simulator declared the type.
    op_data["units"] = operating_point_units(raw, op_data)

    # Carry the simulator's own diagnostics. An unrecognized .save'd @-param
    # (a typo, or one that device class lacks) is written as a real-looking 0.0
    # trace — indistinguishable from a true 0 without the log warning that says
    # it's bogus. A solve failure (singular/non-converged) taints every value
    # here, so it's relayed too. The bias point returns every trace, so the
    # full unrecognized list is relevant.
    unrecognized, failures = await asyncio.to_thread(read_log_warnings, source.raw, source.log)
    op_data["warnings"].extend([*unrecognized, *failures])

    # device_op_points is empty because the deck didn't request the per-device
    # small-signal params — say how to get them, otherwise the empty bucket
    # reads as "this circuit has none". Both simulators can produce them; the
    # remedy differs by simulator, so name both rather than risk misattributing
    # the producer on a cross-simulator raw read. Gate so passive circuits stay
    # note-free: fire when an M/Q/J/D terminal current proves a device is present
    # (LTspice exposes those), or when the run is ngspice — whose bare .op shows
    # no device traces, so a saved-nothing run is indistinguishable from a
    # passive one. raw_dialect only gates here, never picks the wording.
    if not op_data.get("device_op_points") and (
        has_active_device(op_data.get("currents", {})) or raw_dialect == "ngspice"
    ):
        op_data["warnings"].append(NO_DEVICE_OP_POINTS_NOTE)

    # A DC sweep raw has no single "operating point". With at=, report which
    # sweep point was read; otherwise flag that point 0 is just the start bias.
    if is_dc and sweep_value is not None:
        op_data["sweep_value"] = sweep_value
        op_data["warnings"].append(
            f"DC sweep bias at sweep value {sweep_value:.6g} (nearest requested point)."
        )
    elif is_dc:
        op_data["warnings"].append(
            "This is a DC sweep raw; the values are sweep point "
            f"{step} (the sweep's starting bias), not a chosen operating "
            "point. Pass at=<sweep value> to read the bias at a specific point, "
            "or run a .OP."
        )
    return op_data


# ---------------------------------------------------------------------------
# The evaluator seam
# ---------------------------------------------------------------------------

MetricFn: TypeAlias = Callable[
    [services.AnalysisSource, Any, int, SessionState],
    Coroutine[Any, Any, MetricValue],
]

#: Which function answers each scalar/keyed recipe. ``waveform`` and ``plot``
#: are absent on purpose: they write artifacts and page series rather than
#: returning one value, so the evaluator owns them.
METRICS: dict[type[Recipe], MetricFn] = {
    SummaryRecipe: summary,
    MeasurementsRecipe: measurements,
    ValueRecipe: value,
    SignalStatsRecipe: signal_stats,
    EdgesRecipe: edges,
    TimingRecipe: timing,
    PeriodicRecipe: periodic,
    TransientResponseRecipe: transient_response,
    ThdRecipe: thd,
    BodeFilterRecipe: bode_filter,
    BodePointRecipe: bode_point,
    BodeCrossingRecipe: bode_crossing,
    BodeSlopeRecipe: bode_slope,
    StabilityRecipe: stability,
    AcStructureRecipe: ac_structure,
    ResonanceRecipe: resonance,
    ReturnLossRecipe: return_loss,
    NoiseIntegralRecipe: noise_integral,
    OperatingPointRecipe: operating_point,
}
