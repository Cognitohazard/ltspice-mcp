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
    is_op_stepping_failure,
    parse_fourier_data,
    parse_measurements,
)
from ltspice_mcp.lib.result_observations import surface_observations


class _MultiPlotAsciiGuard:
    """Break spicelib's trailing-empty-line skip when it meets the next plot.

    spicelib's ``PlotData._read_ascii_vector`` ends each ASCII plot by skipping
    trailing blank lines: it reads a line, and if it is non-empty seeks back to
    re-read it, else breaks. On a multi-plot ASCII raw with no blank line
    between plots — ngspice writes ``.noise`` as two plots (spectral density,
    then integrated noise) exactly this way — the "non-empty" line is the next
    plot's ``Title:`` header, so it seeks back and re-reads the same line
    forever: a CPU-bound infinite loop that runs synchronously and hangs the
    whole server. This wrapper watches for that one pathological move — a
    seek back to a line just read as non-empty — and returns a one-shot empty
    read so the skip loop breaks with the cursor left at the next plot's
    header, which ``RawRead``'s outer loop then reads as plot 2. The data-read
    loop never seeks, so it is untouched. Version-independent: it keys on the
    read/seek pattern, not on spicelib internals, and is harmless on a spicelib
    that already breaks correctly.
    """

    # Hard backstop: a forward-only reader always advances the high-water byte
    # offset, so a run of reads that never passes it is a loop. Catches ANY
    # pathological ASCII shape the specific break above doesn't, at a few dozen
    # microseconds' cost. A legitimately huge raw advances every read, so this
    # counts NON-advancing reads only — it can't false-positive on size.
    _STALL_LIMIT = 128

    def __init__(self, fobj: object) -> None:
        self._f = fobj
        self._last_read_start: int | None = None
        self._last_nonempty = False
        self._break_at: int | None = None
        self._max_pos = -1
        self._stall_reads = 0

    def readline(self, *args: object) -> bytes:
        pos = self._f.tell()  # type: ignore[attr-defined]
        if self._break_at is not None and pos == self._break_at:
            # One-shot: break the skip loop, leaving the cursor on the next
            # plot's header (do not consume it) so the outer reader continues.
            self._break_at = None
            return b""
        self._last_read_start = pos
        line = self._f.readline(*args)  # type: ignore[attr-defined]
        self._last_nonempty = bool(line.strip())
        new_pos = self._f.tell()  # type: ignore[attr-defined]
        if new_pos > self._max_pos:
            self._max_pos = new_pos
            self._stall_reads = 0
        else:
            self._stall_reads += 1
            if self._stall_reads > self._STALL_LIMIT:
                # Fail THIS parse (surfaced as a parse error) rather than let a
                # not-yet-understood loop hang the whole server.
                raise RuntimeError(
                    "ASCII raw parse made no forward progress for "
                    f"{self._STALL_LIMIT} reads at byte {new_pos} — aborting a "
                    "suspected parser loop (malformed or unsupported raw layout)."
                )
        return line

    def seek(self, pos: int, *args: object) -> object:
        # A seek back to a line just read as non-empty is the trailing-skip
        # loop rewinding onto the next plot's header — arm the one-shot break.
        if pos == self._last_read_start and self._last_nonempty:
            self._break_at = pos
        return self._f.seek(pos, *args)  # type: ignore[attr-defined]

    def __getattr__(self, name: str) -> object:
        return getattr(self._f, name)


def _install_multiplot_ascii_guard() -> None:
    """Wrap ``PlotData._read_ascii_vector`` so a multi-plot ASCII raw can't hang.

    Idempotent. Applied at import because every server raw read constructs a
    ``RawRead`` (via ``OffsetAwareRawRead``), and a single ngspice ``.noise``
    run would otherwise wedge the process. Report/patch upstream separately;
    this guard no-ops once spicelib breaks the loop itself.
    """
    from spicelib.raw.plot_data import PlotData

    original = PlotData._read_ascii_vector  # pyright: ignore[reportPrivateUsage]
    if getattr(original, "_multiplot_guarded", False):
        return

    def guarded(self: object, raw_file: object) -> object:
        return original(self, _MultiPlotAsciiGuard(raw_file))  # type: ignore[arg-type]

    guarded._multiplot_guarded = True  # type: ignore[attr-defined]
    PlotData._read_ascii_vector = guarded  # type: ignore[assignment,method-assign]


_install_multiplot_ascii_guard()


class OffsetAwareRawRead(RawRead):
    """RawRead that rebases a windowed-transient time axis to deck time.

    LTspice stores ``.tran 0 <tstop> <tstart>`` output with the time axis
    rebased to 0 and the true start in the header's ``Offset:`` field;
    spicelib parses the field but never applies it, so every axis consumer
    (analysis windows, measurements, exports) silently works in the offset
    frame — a ``.tran 0 202u 196u`` run reads as 0..6 µs. Applying the offset
    once here puts every downstream tool in deck coordinates. Only transient
    plots with a nonzero offset are affected: other analyses' axes are not
    time, and LTspice writes ``Offset: 0`` for unwindowed runs.

    Server-side raw loads construct this class, not bare RawRead.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_offset = 0.0
        try:
            offset = float(str(self.raw_params.get("Offset", 0) or 0).strip())
        except (TypeError, ValueError):
            offset = 0.0
        plotname = str(self.raw_params.get("Plotname", "")).lower()
        if offset != 0.0 and "transient" in plotname:
            self.time_offset = offset

    def get_axis(self, step: int = 0):
        axis = super().get_axis(step)
        if self.time_offset:
            # Rebase AFTER the parent's abs() (negative stored axis entries
            # encode compression points), never on the stored data.
            return np.asarray(axis) + self.time_offset
        return axis


class _OperatingPointStepMeta(TypedDict, total=False):
    """Optional metadata populated by the tool layer."""

    step: int
    step_count: int
    # SI unit per trace name (only entries the simulator typed); see ``trace_unit``.
    units: dict[str, str]
    # Echoed when a ``device=`` filter narrowed the result to one device.
    device: str
    # Nearest sweep value read when ``at=`` selects a .dc point.
    sweep_value: float
    warnings: list[str]


class OperatingPointOutput(_OperatingPointStepMeta):
    """Return shape of :func:`extract_operating_point`.

    ``step`` / ``step_count`` are present only when the result is built by
    the tool layer for a stepped .OP run.
    """

    voltages: dict[str, float]
    currents: dict[str, float]
    device_op_points: dict[str, float]


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


def is_dc_analysis(sim_type: str) -> bool:
    """Check if the sim type is a .DC sweep (transfer characteristic).

    Word-boundary match on "DC" so LTspice's "DC transfer characteristic"
    matches while substrings inside unrelated words do not.
    """
    return bool(_RE_DC.search(sim_type))


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


def _nearest_ascending(axis: np.ndarray, target: float) -> int:
    """Index in an ascending axis nearest ``target`` (binary search + closer-of-pair)."""
    ins = int(np.searchsorted(axis, target))
    if ins == 0:
        return 0
    if ins >= len(axis):
        return len(axis) - 1
    return ins - 1 if abs(axis[ins - 1] - target) < abs(axis[ins] - target) else ins


def nearest_index(axis: np.ndarray, target: float) -> int:
    """Return the axis index nearest to ``target`` (binary search, O(log N)).

    SPICE sweep axes are monotonic but may run high->low (e.g. ``.dc Vg 1.8 0
    -0.01``). ``np.searchsorted`` assumes ascending order, so on a descending
    axis the bracket is found on the reversed view and the index mapped back —
    otherwise every lookup lands at an endpoint and silently returns the wrong
    sample. This is the one place all three readers (query_value in raw and
    job-run modes, step_get) resolve a sweep point, so the direction handling
    lives here, not in each caller.
    """
    n = len(axis)
    if n == 0:
        return 0
    if n > 1 and axis[0] > axis[-1]:
        return n - 1 - _nearest_ascending(axis[::-1], target)
    return _nearest_ascending(axis, target)


# whattype is spicelib's verbatim per-trace ``var_type`` from the raw header's
# variable list (the simulator's own declaration). Map the known SPICE types to
# an SI unit; an unknown type yields None rather than a wrong guess. The raw
# trace name is always shown regardless — this only *adds* a unit when the
# simulator stated the type, so it never invents one from a parameter name.
_WHATTYPE_UNIT = {
    "voltage": "V",
    "current": "A",
    "device_current": "A",
    "time": "s",
    "frequency": "Hz",
    "hertz": "Hz",
    "admittance": "S",
    "capacitance": "F",
}


def whattype_unit(whattype: str | None) -> str | None:
    """SI unit for a spicelib ``whattype`` string, or None if it is not a
    known SPICE type. Pure string lookup — no raw access."""
    if not whattype:
        return None
    return _WHATTYPE_UNIT.get(whattype.strip().lower())


def trace_unit(raw: RawRead, name: str) -> str | None:
    """SI unit for a trace: the simulator's declared ``whattype`` if it maps to
    a known SPICE type, else the ``V(``/``I(`` name prefix, else None.

    Deliberately never guesses a unit from a device operating-point parameter name
    (e.g. it won't claim ``@m1[gm]`` is siemens unless the simulator typed the
    trace as ``admittance``) — that would be a vendor catalog, not a relay.
    """
    with contextlib.suppress(Exception):
        unit = whattype_unit(getattr(raw.get_trace(name), "whattype", None))
        if unit:
            return unit
    low = name.lstrip().lower()
    if low.startswith("v("):
        return "V"
    if low.startswith("i") and "(" in low:
        return "A"
    return None


def dc_axis_name(raw: RawRead) -> tuple[str | None, str | None]:
    """``(name, SI unit)`` of a .dc sweep's swept-variable axis (trace 0).

    Returns ``(None, None)`` if the axis name is unavailable. The unit comes from
    the axis's declared ``whattype``. Lets the readers label a .dc sweep by its
    swept variable (e.g. ``Vin`` / ``Vin_V``) instead of a generic ``t``/``sweep``
    tag — the one place that introspection lives, shared by the text and CSV paths.
    """
    with contextlib.suppress(Exception):
        ax = raw.get_trace(0)
        name = getattr(ax, "name", None)
        if name:
            return str(name), whattype_unit(getattr(ax, "whattype", None))
    return None, None


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


# Branch-current trace names carry an optional terminal letter between ``I`` and
# ``(`` for multi-terminal devices: e.g. ``Ic(Q1)``/``Ib(Q1)`` (BJT),
# ``Id(M1)``/``Ig(M1)`` (MOSFET/JFET). A bare ``I(...)`` covers two-terminal
# element currents like ``I(RC)``/``I(VCC)``. A plain ``startswith("I(")`` test
# drops the terminal-letter forms, so match the optional letter explicitly.
_OP_CURRENT_RE = re.compile(r"^I[A-Z]?\(", re.IGNORECASE)


def extract_operating_point(
    raw: RawRead, step: int = 0, point_index: int = 0
) -> OperatingPointOutput:
    """Extract DC operating point data (node voltages, branch currents, device operating point).

    Works best with Operating Point (.OP) simulations, but can extract
    first-point values from any simulation type. ``step`` selects which
    iteration of a stepped .OP run to return (0 by default).

    Args:
        raw: Loaded RawRead instance
        step: Step index for stepped .OP / .DC runs.

    Returns:
        Dictionary with 'voltages', 'currents', and 'device_op_points' dicts
        mapping trace names to values. All values are Python float.
    """
    trace_names = raw.get_trace_names()

    voltages = {}
    currents = {}
    device_op_points = {}

    for trace in trace_names:
        wave = raw.get_wave(trace, step=step)
        if len(wave) == 0:
            continue
        # point_index selects a sweep point for a .dc raw (all traces share the
        # axis); clamp so a stray index can't IndexError. Default 0 = .op bias.
        value = float(wave[min(point_index, len(wave) - 1)])

        # ngspice writes device small-signal / model parameters as @dev[param]
        # — bare (@m1[gm]), or wrapped as v(@m1[vth]) / i(@m1[id]) depending on
        # the quantity. These are model state, not a node voltage or branch
        # current, so the '@' marker takes precedence over the V(/I( wrapping:
        # otherwise v(@m1[vth]) is mislabeled a node voltage and bare @m1[gm]
        # falls through both buckets and is dropped entirely.
        if "@" in trace:
            device_op_points[trace] = value
            continue

        # SPICE node names are case-insensitive; spicelib may return either case.
        trace_upper = trace.upper()
        if trace_upper.startswith("V("):
            voltages[trace] = value
        elif _OP_CURRENT_RE.match(trace):
            currents[trace] = value

    return {
        "voltages": voltages,
        "currents": currents,
        "device_op_points": device_op_points,
    }


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
            # Worst-case unity-gain crossover is the one with the most negative
            # (least stable) phase margin — a negative margin must not be masked
            # by a smaller positive one (matches compute_stability_metrics).
            worst_pm = min(pm_entries, key=lambda m: m["margin_deg"])
            metrics["unity_gain_freq"] = float(worst_pm["frequency_hz"])
    except Exception:
        pass

    return metrics


def _raw_node_data_is_finite(
    raw: RawRead, trace_names: list[str], has_axis: bool, step: int
) -> bool:
    """Whether the raw's node traces hold real, finite data (a solved bias point).

    ngspice can print an OP ``<method> stepping failed`` line, recover via a
    later method it does not announce in wording this parser recognizes, and
    still write a valid raw — but it ALSO writes a rail-pinned/NaN raw on a
    genuine floating-node failure and exits 0. The log alone can't tell recovery
    from failure; the data can. Returns True only when at least one non-axis
    trace was checked and every checked trace is finite and off the ~1e30 rail.
    """
    axis = trace_names[0] if (has_axis and trace_names) else None
    checked = 0
    for name in trace_names:
        if name == axis:
            continue
        try:
            arr = np.asarray(raw.get_wave(name, step=step))
        except Exception:
            continue
        if arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)) or float(np.abs(arr).max()) > 1e29:
            return False
        checked += 1
    return checked > 0


def build_simulation_summary(
    raw: RawRead,
    log_path: Path | None,
    duration: float | None = None,
    *,
    step: int = 0,
    requested: dict[str, list[str]] | None = None,
    value_scan: str = "off",
    source_amplitudes: dict[str, float] | None = None,
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
        source_amplitudes: Parsed independent voltage-source amplitudes from
            the deck (``parse_source_amplitudes``); arms the source-relative
            extreme-value observation. None when the caller has no netlist.

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
        from ltspice_mcp.lib.log_parser import (
            make_log_reader,
            parse_temperatures,
            read_log_text,
            scan_op_step_log,
        )

        # Read the log buffer once and hand the text to every text parser in
        # this block (read_log_text exists for exactly this) instead of one
        # syscall + full decode per parser.
        log_text = read_log_text(log_path)

        # Ambient/nominal temperature is a provenance fact the simulator prints
        # by default — surface it so temp-sensitive tasks (noise, leakage,
        # tempco) can confirm it instead of assuming 27 °C.
        temp_c, tnom_c = parse_temperatures(text=log_text)
        if temp_c is not None:
            summary["temp_c"] = temp_c
        if tnom_c is not None:
            summary["tnom_c"] = tnom_c

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

        # Raw-validity gate for OP "stepping failed" errors. The log-only
        # converged-check keys on LTspice's success wording, so an ngspice run
        # that recovered via an unannounced fallback leaves a false hard error.
        # When the raw actually holds finite, off-rail node data, the solve DID
        # recover — demote those stepping-failure errors to warnings. A genuine
        # no-data run (NaN/±1e30 raw) fails the check and keeps the error, and
        # always-terminal failures (iteration limit) aren't candidates. Only
        # loads waves when such an error exists, so the cost is paid rarely.
        errs = summary.get("errors")
        if errs:
            demoted = [e for e in errs if is_op_stepping_failure(e)]
            if demoted and _raw_node_data_is_finite(raw, trace_names, has_axis, step):
                kept = [e for e in errs if e not in demoted]
                if kept:
                    summary["errors"] = kept
                else:
                    summary.pop("errors", None)
                warnings.extend(
                    f"{d} — run produced finite node data (OP solve recovered via an "
                    "unlabeled fallback); surfaced as a warning, not an error."
                    for d in demoted
                )

        # Stepped ``.op`` runs the bias point per step, but LTspice only
        # writes step 0 to the .raw — leaving the user thinking it's the
        # only step. Two signals: ``.step name=value`` lines (only emitted
        # for ``.tran``/``.ac`` steps) and the Newton-iteration counter
        # (the only signal for stepped ``.op``). Single log walk picks up
        # both.
        if step_count <= 1 and "operating" in sim_type.lower():
            log_steps, op_iters = scan_op_step_log(text=log_text)
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
        summary,
        requested=requested,
        value_traces=value_traces,
        value_scan=value_scan,
        source_amplitudes=source_amplitudes,
    )

    return summary
