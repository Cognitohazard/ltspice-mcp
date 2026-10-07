""".raw file parsing and waveform analysis.

Provides core functions for parsing .raw files, extracting trace names,
computing statistics, and querying data points. Numerical helpers consume the
RawData protocol; decoded plots supply physical metadata for their selected
analysis. Explicit dependency readers remain usable by offline tests.

For .log file parsing (measurements, Fourier data), see log_parser.py.

Functions are synchronous — callers invoke them directly (see concurrency contract in tools/_base.py).
"""

from __future__ import annotations

import codecs
import contextlib
import math
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from spicelib.raw.raw_classes import SpiceReadException
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import NoAxisError
from ltspice_mcp.lib.decoded_log import DecodedLog, LogSectionName
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw, PlotDescriptor, RawData
from ltspice_mcp.lib.format import cap_list
from ltspice_mcp.lib.log_parser import is_op_stepping_failure
from ltspice_mcp.lib.result_observations import surface_observations

# What a raw accessor raises when the thing being asked for is not there: a
# missing property (ValueError), a missing trace (IndexError from spicelib,
# KeyError from a name-keyed lookup), a raw with no axis (RuntimeError), a file
# it cannot read that far (SpiceReadException), or an alias form it does not
# implement (NotImplementedError). Catching this tuple instead of every
# Exception keeps a bug on OUR side (AttributeError, TypeError) from being
# swallowed as "the trace isn't there" and answered with a fallback value.
_RAW_ACCESS_ERRORS = (
    ValueError,
    IndexError,
    KeyError,
    RuntimeError,
    NotImplementedError,
    SpiceReadException,
)


def _parse_failure(what: str, exc: BaseException) -> str:
    """One line naming a summary field whose parser raised.

    A parse fault used to leave the field simply absent, which on the wire is
    indistinguishable from "the deck asked for nothing" — a log this build
    could not read looked exactly like a run with no ``.meas`` in it. Naming
    the field and the exception makes the difference visible without judging
    the result.
    """
    return f"{what} unavailable: {type(exc).__name__}: {exc}"


# Header prefixes shared by restart recovery and dialect sniffing.
_RAW_HEADER_ASCII = b"Title:"
_RAW_TITLE_UTF16 = "Title:".encode("utf-16-le")
_RAW_HEADER_UTF16 = (b"\xff\xfe" + _RAW_TITLE_UTF16, _RAW_TITLE_UTF16)


def has_valid_raw_header(path: Path | None) -> bool:
    """True if ``path`` looks like a real ``.raw`` file, by header magic.

    The one answer to "did this run actually write results?" wherever a restart
    promotes an interrupted job from the filesystem. A truncated or unrelated
    file at the expected path must not be mistaken for a result, and two copies
    of that check are how one of them comes to accept what the other rejects.
    """
    if path is None:
        return False
    try:
        with path.open("rb") as handle:
            header = handle.read(max(map(len, _RAW_HEADER_UTF16)))
    except OSError:
        return False
    return header.startswith(_RAW_HEADER_ASCII) or header.startswith(_RAW_HEADER_UTF16)


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

    def tell(self) -> int:
        return self._f.tell()  # type: ignore[attr-defined,no-any-return]

    def __getattr__(self, name: str) -> object:
        return getattr(self._f, name)


def _install_multiplot_ascii_guard() -> None:
    """Wrap ``PlotData._read_ascii_vector`` so a multi-plot ASCII raw can't hang.

    Idempotent. Retained for explicit dependency readers while their callers
    migrate to decoded results; a single ngspice ``.noise`` recording would
    otherwise wedge those reads. Report/patch upstream separately;
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


def _transient_offset(plotname: object, offset: object) -> float:
    """Where a windowed transient's stored time axis starts, in deck time.

    LTspice stores ``.tran 0 <tstop> <tstart>`` output from 0 with the true
    start in the header's ``Offset:`` field. Other analyses' axes are not time,
    and an unwindowed run writes ``Offset: 0``, so both read as 0 here.
    """
    if "transient" not in str(plotname or "").lower():
        return 0.0
    try:
        return float(str(offset or 0).strip())
    except ValueError:
        return 0.0


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

    Retained for existing dependency-reader imports and regression fixtures.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.time_offset = _transient_offset(
            self.raw_params.get("Plotname"), self.raw_params.get("Offset")
        )

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
    #: Traces the raw carries that are neither. Present so a reader can tell
    #: "this run holds nothing" from "we did not recognise these names": an
    #: LTspice ``.tf`` result is typed ``transfer``/``impedance`` and named
    #: without a ``V(``/``I(`` prefix, so it used to be dropped on the floor.
    other: dict[str, float]


# Smallest positive normal float — floor for magnitude before log10 to avoid -inf
_FLOAT_TINY = np.finfo(float).tiny

# Word-boundary simulation-type matchers. Substring matching would false-positive
# on phrases like "DC transfer characteristic" (contains "AC") or "backup" (also
# contains "AC"), so detection is anchored to whole words.
_RE_TRANSIENT = re.compile(r"\bTRANSIENT\b", re.IGNORECASE)
_RE_AC = re.compile(r"\bAC\b", re.IGNORECASE)
_RE_DC = re.compile(r"\bDC\b", re.IGNORECASE)
_RE_NOISE = re.compile(r"\bNOISE\b", re.IGNORECASE)
_RE_OP = re.compile(r"\bOPERATING\s+POINT\b", re.IGNORECASE)


def safe_magnitude_db(wave: np.ndarray) -> np.ndarray:
    """Convert complex waveform to magnitude in dB, clamping zeros to avoid -inf."""
    magnitude = np.abs(wave)
    magnitude = np.where(magnitude > 0, magnitude, _FLOAT_TINY)
    return 20 * np.log10(magnitude)


def _selected_descriptor(raw: RawData) -> PlotDescriptor | None:
    """Physical metadata only for actual decoded readers, never guessed from a stub."""
    return raw.descriptor if isinstance(raw, (DecodedRaw, DecodedPlot)) else None


def detect_sim_type(raw: RawData) -> str:
    """Detect simulation type from raw file metadata.

    Args:
        raw: Selected decoded plot or structural reader

    Returns:
        Simulation type string (e.g., "Transient Analysis", "AC Analysis")
        or "Unknown" if detection fails
    """
    if descriptor := _selected_descriptor(raw):
        return descriptor.original_plot_name if descriptor.analysis != "unknown" else "Unknown"
    try:
        plot_name = raw.get_raw_property("Plotname")
        if plot_name:
            return str(plot_name)
    except _RAW_ACCESS_ERRORS:
        # A raw with no Plotname property is a real shape, not a fault: the
        # caller gets the "Unknown" it documents.
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


def is_operating_point(sim_type: str) -> bool:
    """Check if the sim type is a bias-point solve (``.op``).

    Both simulators name it "Operating Point". Word-boundary matched like its
    siblings so a composite name cannot false-positive on a substring.
    """
    return bool(_RE_OP.search(sim_type))


def is_dc_analysis(sim_type: str) -> bool:
    """Check if the sim type is a .DC sweep (transfer characteristic).

    Word-boundary match on "DC" so LTspice's "DC transfer characteristic"
    matches while substrings inside unrelated words do not.
    """
    return bool(_RE_DC.search(sim_type))


def get_step_count(raw: RawData) -> int:
    """Get number of simulation steps (for .step directives).

    Args:
        raw: Selected decoded plot or structural reader

    Returns:
        Number of validated step slices. Legacy lookup misses default to one;
        decoded plots with unresolved boundaries refuse numeric selection.
    """
    if _selected_descriptor(raw) is not None:
        return len(raw.get_steps())
    try:
        return len(raw.get_steps())
    except _RAW_ACCESS_ERRORS:
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
    "impedance": "Ω",
    "capacitance": "F",
}


def whattype_unit(whattype: str | None) -> str | None:
    """SI unit for a spicelib ``whattype`` string, or None if it is not a
    known SPICE type. Pure string lookup — no raw access."""
    if not whattype:
        return None
    return _WHATTYPE_UNIT.get(whattype.strip().lower())


def declared_type(raw: RawData, name: str) -> str | None:
    """The simulator's own word for what a trace holds, lowercased, or None.

    This relays metadata, not physical meaning: some ngspice analyses label
    derived quantities as voltage even when they represent an impedance.

    A trace the raw does not carry, or one the reader cannot type, answers
    None so the caller falls back rather than reporting a fault it can do
    nothing about.
    """
    with contextlib.suppress(*_RAW_ACCESS_ERRORS):
        whattype = getattr(raw.get_trace(name), "whattype", None)
        if whattype:
            return str(whattype).strip().lower()
    return None


def trace_unit(raw: RawData, name: str) -> str | None:
    """Physical unit from the selected decoded descriptor, including unknown units.

    Structural readers use declared ``whattype`` and then a ``V(``/``I(``
    prefix. A decoded descriptor's absent unit never falls back to that guess:
    native transfer, root and sensitivity quantities may be declared voltage.

    Deliberately never guesses a unit from a device operating-point parameter name
    (e.g. it won't claim ``@m1[gm]`` is siemens unless the simulator typed the
    trace as ``admittance``) — that would be a vendor catalog, not a relay.
    """
    if descriptor := _selected_descriptor(raw):
        for trace in descriptor.traces:
            if trace.name.lower() == name.lower():
                return trace.unit
        return None
    unit = whattype_unit(declared_type(raw, name))
    if unit:
        return unit
    low = name.lstrip().lower()
    if low.startswith("v("):
        return "V"
    if low.startswith("i") and "(" in low:
        return "A"
    return None


def dc_axis_name(raw: RawData) -> tuple[str | None, str | None]:
    """``(name, SI unit)`` of a .dc sweep's swept-variable axis (trace 0).

    Decoded plots use their sampled axis descriptor; a table has no axis even
    when trace zero is declared voltage or frequency. Structural readers use
    trace zero and its declared ``whattype``. Lets readers label a .dc sweep by its
    swept variable (e.g. ``Vin`` / ``Vin_V``) instead of a generic ``t``/``sweep``
    tag — the one place that introspection lives, shared by the text and CSV paths.
    """
    if descriptor := _selected_descriptor(raw):
        return (descriptor.axis.name, descriptor.axis.unit) if descriptor.axis else (None, None)
    # Legacy readers may not expose a trace zero for an axis-less plot.
    with contextlib.suppress(*_RAW_ACCESS_ERRORS):
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


def query_point_value(
    raw: RawData,
    trace_name: str,
    target_x: float,
    step: int = 0,
    *,
    read_wave: Callable[[], np.ndarray] | None = None,
) -> dict:
    """Query signal value at a specific time/frequency (nearest neighbor).

    Uses binary search for O(log n) lookup. No interpolation - returns
    the nearest data point to the requested value.

    Args:
        raw: Selected decoded plot or structural reader
        trace_name: Name of trace to query
        target_x: Time or frequency value to query
        step: Step index (default 0)
        read_wave: Reads the step's samples, for a signal that is not a
            stored trace (a node-pair difference); called only once the axis
            is known to exist. ``trace_name`` is read when omitted

    Returns:
        Dictionary with trace name, requested/actual x values, and signal value.
        For AC data, includes magnitude_db and phase_deg.
        All values are Python float (not numpy scalars).

    Raises:
        ValueError: If the trace contains no data points.
    """
    try:
        raw_axis = raw.get_axis(step=step)
    except (RuntimeError, TypeError) as exc:
        # Two spellings of the same fact: spicelib raises RuntimeError ("This
        # RAW file does not have an axis.") for an operating-point raw, and
        # returns an unsized empty array for a file that held no plots at all,
        # so len() inside it raises TypeError.
        raise NoAxisError(f"Result has no sweep axis to query at step {step}.") from exc
    axis = real_axis(np.asarray(raw_axis))
    wave = read_wave() if read_wave is not None else raw.get_wave(trace_name, step=step)

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

#: Declared trace types that place a trace in a bias-point bucket. A type
#: outside this map belongs in ``other``. Only an absent type falls back to
#: the name, so a declared impedance cannot become a voltage from its spelling.
_OP_BUCKET_BY_TYPE = {
    "voltage": "voltages",
    "current": "currents",
    "device_current": "currents",
}


def extract_operating_point(
    raw: RawData, step: int = 0, point_index: int = 0
) -> OperatingPointOutput:
    """Extract DC operating point data (node voltages, branch currents, device operating point).

    Works best with Operating Point (.OP) simulations, but can extract
    first-point values from any simulation type. ``step`` selects which
    iteration of a stepped .OP run to return (0 by default).

    Args:
        raw: Selected decoded plot or structural reader
        step: Step index for stepped .OP / .DC runs.

    Returns:
        Dictionary with 'voltages', 'currents', 'device_op_points' and 'other' dicts
        mapping trace names to values. All values are Python float.
    """
    trace_names = raw.get_trace_names()

    buckets: OperatingPointOutput = {
        "voltages": {},
        "currents": {},
        "device_op_points": {},
        "other": {},
    }

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
            buckets["device_op_points"][trace] = value
            continue

        whattype = declared_type(raw, trace)
        bucket = _OP_BUCKET_BY_TYPE.get(whattype or "")
        if not whattype:
            # SPICE node names are case-insensitive; spicelib may return either.
            if trace.upper().startswith("V("):
                bucket = "voltages"
            elif _OP_CURRENT_RE.match(trace):
                bucket = "currents"
        buckets[bucket or "other"][trace] = value

    return buckets


def compute_ac_bandwidth_metrics(raw: RawData, trace_name: str, step: int = 0) -> dict:
    """Compute -3 dB bandwidth and unity-gain frequency for AC simulations.

    Returns a dict with ``bandwidth_3db`` and ``unity_gain_freq`` (each a
    Python float or None). The bandwidth is the first −3 dB crossing
    relative to DC gain (low cutoff for LPFs, low edge for BPFs). The
    unity-gain frequency is the worst-case 0 dB crossover from the full
    stability sweep — meaningful for amplifier-shaped responses.

    A ``warnings`` list is added naming any metric whose computation raised:
    a None meaning "this response has no such crossing" and a None meaning
    "computing it failed" are otherwise the same value on the wire. It is
    absent when nothing raised.

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

    metrics: dict[str, Any] = {
        "bandwidth_3db": None,
        "unity_gain_freq": None,
    }
    failures: list[str] = []

    try:
        axis_raw = raw.get_axis(step=step)
        wave_raw = raw.get_wave(trace_name, step=step)
        freqs, H = prepare_ac_arrays(np.asarray(axis_raw), np.asarray(wave_raw))
    except _RAW_ACCESS_ERRORS as exc:
        metrics["warnings"] = [_parse_failure(f"AC data for {trace_name!r}", exc)]
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
    except Exception as exc:
        failures.append(_parse_failure("bandwidth_3db", exc))

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
    except Exception as exc:
        failures.append(_parse_failure("unity_gain_freq", exc))

    if failures:
        metrics["warnings"] = failures
    return metrics


def _raw_node_data_is_finite(raw: RawData, trace_names: list[str], step: int) -> bool:
    """Whether the raw holds a real, finite NODE VOLTAGE (a solved bias point).

    ngspice can print an OP ``<method> stepping failed`` line, recover via a
    later method it does not announce in wording this parser recognizes, and
    still write a valid raw — but it ALSO writes a rail-pinned/NaN raw on a
    genuine floating-node failure and exits 0. The log alone can't tell recovery
    from failure; the data can. Only node-voltage traces (``V(...)``) count: a
    failing ``.op`` can still write a finite branch current or device-parameter
    trace (``.save i(V1)``, ``@m1[gm]``) while the node voltage sits at NaN/rail,
    so a non-voltage trace must not vouch for a bias point that didn't converge.
    Returns True only when at least one voltage trace was checked and every
    checked voltage is finite and off the ~1e30 rail.
    """
    checked = 0
    for name in trace_names:
        if not name.lower().startswith("v("):
            continue
        try:
            arr = np.asarray(raw.get_wave(name, step=step))
        except _RAW_ACCESS_ERRORS:
            # A trace this raw can't produce vouches for nothing; ``checked``
            # stays where it was, so an all-unreadable raw still answers False.
            continue
        if arr.size == 0:
            continue
        if not np.all(np.isfinite(arr)) or float(np.abs(arr).max()) > 1e29:
            return False
        checked += 1
    return checked > 0


# Structured-channel cap for the summary's signal-name list (see the
# truncation note at the attachment site in ``build_simulation_summary``).
_SIGNALS_STRUCTURED_CAP = 100


def build_simulation_summary(
    raw: RawData,
    logs: DecodedLog | None,
    duration: float | None = None,
    *,
    step: int = 0,
    requested: dict[str, list[str]] | None = None,
    value_scan: bool = False,
    source_amplitudes: dict[str, float] | None = None,
) -> dict:
    """Build comprehensive, type-aware simulation summary.

    Args:
        raw: Selected decoded plot or structural reader
        logs: Optional resident facts from the captured log/console decode.
        duration: Optional simulation duration in seconds
        step: Which .step iteration to summarize (axis/range/point_count).
            Defaults to 0. Callers exposing a step (simulation_summary) thread
            it through so the range reflects the chosen step, not always step 0.
        requested: Parsed ``.meas``/``.four`` names from the deck, for the
            requested-vs-produced reconciliation in the observation surfacer.
            None when the caller has no netlist (skips reconciliation).
        value_scan: Whether to load this ``raw``'s traces and scan them for
            non-finite and extreme values. False where value surfacing does
            not apply to the caller.
        source_amplitudes: Parsed independent voltage-source amplitudes from
            the deck (``parse_source_amplitudes``); arms the source-relative
            extreme-value observation. None when the caller has no netlist.

    Returns:
        Dictionary with sim_type, range info, signals, point_count, step_count,
        optional measurements, warnings, Fourier data, duration, and an
        always-present ``observations`` list (see ``result_observations``).
        All numpy types converted to Python float.

        A field whose parser raised is reported in ``warnings``, naming the
        field and the exception, rather than being left out: an absent
        ``measurements`` key is otherwise the same wire shape whether the deck
        had no ``.meas`` or the log could not be read at all.
    """
    sim_type = detect_sim_type(raw)
    trace_names = raw.get_trace_names()
    step_count = get_step_count(raw)
    descriptor = _selected_descriptor(raw)

    # Collected here rather than in the log block below so a fault anywhere in
    # this function has somewhere to land; attached to the summary once, at the
    # end, so a warning raised after the attachment point is not lost.
    warnings: list[str] = []

    # Stepped ``.op`` raw files have no axis — spicelib raises RuntimeError
    # "This RAW file does not have an axis." Treat that as a valid degenerate
    # case (no range, no point_count beyond step_count) instead of aborting
    # the whole summary.
    try:
        axis = raw.get_axis(step=step)
        point_count = len(axis)
        has_axis = True
    except (NoAxisError, RuntimeError, TypeError):
        # TypeError is the same shape by another route: for a file that held no
        # plots at all, spicelib's get_axis returns ``np.ndarray([])`` — an
        # UNSIZED 0-d array — so ``len()`` raises instead of answering 0.
        axis = None  # type: ignore[assignment]
        point_count = (
            len(raw.get_wave(trace_names[0], step=step))
            if descriptor is not None and trace_names
            else step_count
        )
        has_axis = False
    except _RAW_ACCESS_ERRORS as exc:
        # Any OTHER read fault is not the axis-less shape above: the range and
        # point count are missing because the raw could not be read, and that
        # is a different fact from "this analysis has no axis".
        axis = None  # type: ignore[assignment]
        point_count = step_count
        has_axis = False
        warnings.append(_parse_failure("axis (range, point_count)", exc))

    range_info: dict = {}
    if has_axis and point_count > 0 and axis is not None:
        if _RE_TRANSIENT.search(sim_type):
            range_info = {"time_start": float(axis[0]), "time_end": float(axis[-1])}
        elif (
            is_ac_analysis(sim_type)
            or is_noise_analysis(sim_type)
            or (
                descriptor is not None
                and descriptor.axis is not None
                and descriptor.axis.unit == "Hz"
                and descriptor.axis.real_coordinates
            )
        ):
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
    }
    # Cap the structured signal list: a device-heavy raw (.save all @m*[*],
    # PDK-level node dumps) can carry hundreds-to-thousands of trace names,
    # and this summary is re-sent on every run_simulation completion and
    # check_job poll. The full list stays addressable via the
    # spice://results/{job_id}/signals resource.
    cap_list(summary, "signals", trace_names, _SIGNALS_STRUCTURED_CAP)

    if logs is not None:

        def section_value(name: LogSectionName, label: str) -> Any:
            section = logs.section(name)
            if section["status"] == "error":
                error = section["error"]
                warnings.append(f"{label} unavailable: {error['type']}: {error['message']}")
            if section["nonfinite_count"]:
                warnings.append(
                    f"{label}: {section['nonfinite_count']} non-finite numeric log "
                    "value(s) retained as null."
                )
            return section["value"]

        temperatures = section_value("temperatures", "temperatures")
        if temperatures is not None:
            for name in ("temp_c", "tnom_c"):
                if temperatures[name] is not None:
                    summary[name] = temperatures[name]

        meas_data = section_value("measurements", "measurements")
        if meas_data is not None:
            if meas_data["measurements"]:
                summary["measurements"] = meas_data["measurements"]
            # A failed measurement is distinct from an absent request or a
            # section the worker could not decode.
            if meas_data["failed_measurements"]:
                summary["failed_measurements"] = meas_data["failed_measurements"]

        diagnostics = section_value("diagnostics", "log diagnostics (errors, warnings)")
        if diagnostics is not None:
            warnings.extend(diagnostics["warnings"])
            if diagnostics["errors"]:
                summary["errors"] = diagnostics["errors"]
            if diagnostics.get("meas_errors"):
                summary["meas_errors"] = diagnostics["meas_errors"]

        # How many bias-point solves the log records — each OP-solve block opens
        # with a "Direct Newton iteration" line (whether it converges or fails).
        # A stepped LTspice ``.op`` stores one point per step in the .raw, with
        # the stepped parameter as its first variable, and the decoder reads it
        # as that many steps. The count warns the user when a raw holds fewer
        # steps than the log solved (operating-point runs), and gates the
        # OP-error demote below (one step's point can't vouch for another's).
        op_log_steps = section_value("steps", "step rows")
        op_iterations = section_value("op_iterations", "OP iterations")
        op_coverage_known = op_log_steps is not None and op_iterations is not None
        op_solve_count = max(
            len(op_log_steps or []),
            op_iterations["attempts"] if op_iterations is not None else 0,
            1,
        )

        # Raw-validity gate for OP "stepping failed" errors. The log-only
        # converged-check keys on LTspice's success wording, so an ngspice run
        # that recovered via an unannounced fallback leaves a false hard error.
        # When the raw actually holds finite, off-rail node data, the solve DID
        # recover — demote those stepping-failure errors to warnings. A genuine
        # no-data run (NaN/±1e30 raw) fails the check and keeps the error, and
        # always-terminal failures (iteration limit) aren't candidates. Demote
        # only when the raw covers the WHOLE run: a single solve block
        # (op_solve_count <= 1) written to a single-step raw (step_count <= 1). A
        # stepped .op (only its first point read, log shows >1 solve) or a
        # multi-step raw (later steps not checked here) keeps the error — the
        # first step's finite data can't clear a failure that belongs to another.
        errs = summary.get("errors")
        if errs:
            demoted = [e for e in errs if is_op_stepping_failure(e)]
            if (
                demoted
                and step_count <= 1
                and op_coverage_known
                and op_solve_count <= 1
                and _raw_node_data_is_finite(raw, trace_names, step)
            ):
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

        if op_solve_count > 1 and step_count <= 1 and "operating" in sim_type.lower():
            if op_log_steps:
                param_name = next(iter(op_log_steps[0].keys()), "param")
                suggestion = (
                    f"Convert to '.dc {param_name} START STOP STEP' to access every bias point."
                )
            else:
                suggestion = (
                    "Convert the parametric .op to '.dc <param> START STOP "
                    "STEP' or wrap the .op inside a .tran to capture every "
                    "bias point."
                )
            warnings.append(
                f"Stepped .op detected: log shows {op_solve_count} bias-"
                "point iterations, and the .raw exposes one; only that one is read "
                "here. " + suggestion
            )

        fourier_data = section_value("fourier", "fourier")
        if fourier_data:
            # Preserve the existing omission of blocks without THD/harmonics;
            # sanitized nonfinite facts still leave an explicit warning above.
            fourier_data = [
                f for f in fourier_data if f.get("thd") is not None or f.get("harmonics")
            ]
            if fourier_data:
                summary["fourier"] = fourier_data

    if duration is not None:
        summary["duration"] = float(duration)

    # Surface observations (a "surfacer", not a "judger" — see
    # ``result_observations``). Always present, possibly empty. Value traces are
    # extracted here only when the caller signalled they're loaded.
    value_traces: dict | None = None
    if value_scan:
        # The sweep axis (time / frequency / DC source) is trace 0 and isn't a
        # signal worth scanning. Skip it only when the raw actually HAS an axis:
        # an operating-point raw has none, so ITS trace 0 is a real node, and
        # skipping it there hid a degenerate first-sorted value (e.g. a floating
        # node at ~1e30) — exactly the case this scan exists to catch.
        axis_name = trace_names[0] if (has_axis and trace_names) else None
        value_traces = {}
        unreadable: list[str] = []
        for name in trace_names:
            if name == axis_name:
                continue
            try:
                value_traces[name] = np.asarray(raw.get_wave(name, step=step))
            except _RAW_ACCESS_ERRORS:
                unreadable.append(name)
        if unreadable:
            # The scan reports on what it read. A trace it could not read is a
            # hole in that coverage, and silently narrowing the scan makes the
            # remaining traces look like the whole picture.
            shown = ", ".join(unreadable[:10])
            more = f" (+{len(unreadable) - 10} more)" if len(unreadable) > 10 else ""
            warnings.append(
                f"{len(unreadable)} trace(s) not read during the value scan: {shown}{more}"
            )

    if warnings:
        summary["warnings"] = warnings

    summary["observations"] = surface_observations(
        summary,
        requested=requested,
        value_traces=value_traces,
        source_amplitudes=source_amplitudes,
    )

    return summary


#: How much of a raw header to read for its ``Command:`` field. Every header
#: field that matters (``Command`` last among them) precedes the variables block.
_SNIFF_BYTES = 8192


def _read_head(path: Path) -> bytes | None:
    """The first ``_SNIFF_BYTES`` of a file, or None when it cannot be read."""
    try:
        with path.open("rb") as handle:
            return handle.read(_SNIFF_BYTES)
    except OSError:
        return None


def raw_writer_command(path: Path) -> str | None:
    """The first plot header's ``Command:`` value: the writer naming itself.

    ``Linear Technology Corporation LTspice XVII`` from LTspice XVII, or
    ``ngspice-46, Build Mar 29 2026 15:02:07`` from ngspice 44 and later. None
    when the file is not a raw, has no such field (ngspice before 44), or
    cannot be read. Only the header text before the variables block is
    searched, and only the first ``_SNIFF_BYTES`` of the file are read.
    """
    head = _read_head(path)
    if head is None:
        return None
    if head.startswith(_RAW_HEADER_UTF16):
        width = 2
    elif head.startswith(_RAW_HEADER_ASCII):
        width = 1
    else:
        return None
    fields, _ = _parse_plot_header(head.decode(_RAW_CODECS[width], errors="replace"))
    return fields.get("command") or None


# ---------------------------------------------------------------------------
# How far a stopped run got
# ---------------------------------------------------------------------------

#: Plots whose first variable is a value, not a swept axis (spicelib's list).
_AXISLESS_PLOTS = frozenset({"operating point", "transfer function", "integrated noise"})
#: Header text by character width: every raw header is ASCII or UTF-16LE.
#: ``surrogatepass`` round-trips any UTF-16 code unit, and latin-1 any byte, so
#: re-encoding a decoded prefix gives its byte length exactly.
_RAW_CODECS = {1: "latin-1", 2: "utf-16-le"}
#: Bytes one plot header may take before the reader stops looking for its end.
#: A header is a few lines plus one per variable, so this is far past any deck.
#: Read in small chunks: a running job's cases are read on every status call,
#: and a typical header ends inside the first one.
_PARTIAL_HEADER_CAP = 16 * 1024 * 1024
_PARTIAL_READ_CHUNK = 8 * 1024
#: ASCII data has no fixed record size, so its last point is found by reading
#: the end of the file: a window sized for two points of the plot's variables,
#: grown until it holds a complete one.
_ASCII_TAIL_START = 4 * 1024
_ASCII_TAIL_PER_VARIABLE = 128
_ASCII_TAIL_CAP = 16 * 1024 * 1024
#: Bytes of a finished ASCII plot searched, a block at a time, for the plot
#: after it; past this the count is reported as unknown.
_ASCII_SKIP_BLOCK = 1024 * 1024
_ASCII_SKIP_CAP = 8 * 1024 * 1024
_RE_DATA_START = re.compile(r"(?im)^(binary|values):[ \t]*\r?\n")


@dataclass(frozen=True)
class PartialRawProgress:
    """How far a raw file got, counted from the bytes on disk.

    ``plot`` and ``axis`` name the plot that was being written when the file
    ended (a file can hold finished plots before it) and its swept variable;
    ``axis`` is None for a plot with no sweep, such as an operating point.
    ``points`` counts complete points of that plot, or is None when the
    layout does not allow counting them. ``last_axis_value`` is the axis value
    of the last complete point in deck coordinates, None when there is no
    complete point or no axis. ``header_complete`` is False when the file ends
    before the plot's data section begins. ``stepped`` is the header's own
    flag: the axis restarts at every ``.step``, so the last value is a position
    within the current step.
    """

    raw_bytes: int
    header_complete: bool
    plot: str | None
    axis: str | None
    points: int | None
    last_axis_value: float | None
    stepped: bool


@dataclass(frozen=True)
class _PlotHeader:
    fields: dict[str, str]
    variables: list[str]
    data_start: int | None
    data_kind: str | None


def read_partial_raw_progress(path: Path, dialect: str | None = None) -> PartialRawProgress | None:
    """Count what a raw file holds without trusting its declared point count.

    A simulator writes the header first, points as it solves them, and the
    ``No. Points`` count only when a plot ends. A run killed part way leaves
    the count at 0 (ngspice pads it with spaces to patch in place), and
    spicelib rejects that file outright, or fails a short read when the count
    runs past the data. So the count here comes from the data itself: complete
    fixed-size records for a ``Binary:`` plot, and the last complete point in
    the tail of a ``Values:`` plot.

    Records follow spicelib's ``PlotData`` layout: every value is a double
    except LTspice, which stores values as 4-byte floats unless the header
    flags ``double``, keeps the axis a double, and writes the whole AC record,
    axis included, as complex. A complete plot followed by another (ngspice
    writes one per analysis, so ``.op`` then ``.tran`` is two) is stepped over
    by its declared count, and the plot reported is the one the file ends in.

    ``dialect`` names the simulator when the header does not: ngspice before
    version 44 writes no ``Command:`` line. A UTF-16 header is LTspice's.

    Raises FileNotFoundError when there is no file; returns None when the file
    is unreadable or not a raw.
    """
    try:
        with path.open("rb") as handle:
            # Sized through the open handle rather than by path, so the count
            # and every read come from one view of a file the simulator may
            # still be writing.
            size = handle.seek(0, os.SEEK_END)
            handle.seek(0)
            magic = handle.read(len(_RAW_HEADER_UTF16[0]))
            if magic.startswith(_RAW_HEADER_UTF16):
                width = 2
            elif magic.startswith(_RAW_HEADER_ASCII):
                width = 1
            else:
                return None
            offset = 0
            while True:
                header = _read_plot_header(handle, offset, width)
                following = _progress_of_plot(handle, header, size, width, dialect)
                if isinstance(following, int):
                    offset = following
                    continue
                return following
    except FileNotFoundError:
        raise
    except OSError:
        return None


def _read_plot_header(handle: Any, offset: int, width: int) -> _PlotHeader:
    """Read one plot header from ``offset``, up to the line that starts its data."""
    codec = _RAW_CODECS[width]
    decoder = codecs.getincrementaldecoder(codec)(errors="surrogatepass")
    handle.seek(offset)
    text = ""
    consumed = 0
    while consumed < _PARTIAL_HEADER_CAP:
        chunk = handle.read(_PARTIAL_READ_CHUNK)
        if not chunk:
            break
        consumed += len(chunk)
        searched = len(text)
        text += decoder.decode(chunk)
        # Back up by the marker's length: it may straddle two chunks.
        match = _RE_DATA_START.search(text, max(0, searched - 16))
        if match is not None:
            prefix = text[: match.end()].encode(codec, errors="surrogatepass")
            fields, variables = _parse_plot_header(text[: match.start()])
            return _PlotHeader(fields, variables, offset + len(prefix), match.group(1).lower())
    fields, variables = _parse_plot_header(text)
    return _PlotHeader(fields, variables, None, None)


def _parse_plot_header(text: str) -> tuple[dict[str, str], list[str]]:
    """Header fields keyed in lower case, and the variable names in order."""
    fields: dict[str, str] = {}
    variables: list[str] = []
    in_variables = False
    for line in text.split("\n"):
        line = line.rstrip("\r")
        if in_variables:
            parts = line.lstrip().split("\t")
            if len(parts) >= 2:
                variables.append(parts[1])
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key = key.strip().lstrip("\ufeff").lower()
        if key == "variables":
            in_variables = True
            continue
        fields.setdefault(key, value.strip())
    return fields, variables


def _int_field(fields: dict[str, str], key: str) -> int:
    try:
        return int(fields.get(key, "").strip() or 0)
    except ValueError:
        return 0


def _raw_writer(fields: dict[str, str], width: int, dialect: str | None) -> str:
    """The simulator that wrote a plot, from its header before any hint."""
    command = fields.get("command", "").lower()
    writer = None
    # spicelib's order, where a later match wins.
    for name in ("ltspice", "qspice", "ngspice", "xyce"):
        if name in command:
            writer = name
    if writer is not None:
        return writer
    if width == 2:
        return "ltspice"
    return dialect or "ngspice"


def _progress_of_plot(
    handle: Any,
    header: _PlotHeader,
    size: int,
    width: int,
    dialect: str | None,
) -> PartialRawProgress | int | None:
    """This plot's progress, or the offset of the plot that follows it."""
    fields = header.fields
    plot = fields.get("plotname") or None
    flags = fields.get("flags", "").lower().split()
    has_axis = bool(header.variables) and (plot or "").lower() not in _AXISLESS_PLOTS
    axis = header.variables[0] if has_axis else None

    def progress(points: int | None, last: float | None) -> PartialRawProgress:
        return PartialRawProgress(
            raw_bytes=size,
            header_complete=header.data_start is not None,
            plot=plot,
            axis=axis,
            points=points,
            last_axis_value=_deck_axis_value(last, axis, plot, fields) if axis else None,
            stepped="stepped" in flags,
        )

    if header.data_start is None:
        return progress(0, None)
    n_vars = _int_field(fields, "no. variables") or len(header.variables)
    if n_vars < 1:
        return None
    declared = _int_field(fields, "no. points")

    if header.data_kind == "values":
        if declared > 0 and width == 1:
            end = _ascii_plot_end(handle, header.data_start, size)
            if end is None:
                # The plot this file ends in is out of reach; its lines would
                # be read with this plot's variable count.
                return progress(None, None)
            if end < size:
                return end
        return progress(*_ascii_last_point(handle, header.data_start, size, width, n_vars))

    writer = _raw_writer(fields, width, dialect)
    complex_values = "complex" in flags or (plot or "").lower() == "ac analysis"
    if complex_values:
        value_size = 16
    elif writer != "ltspice" or "double" in flags:
        value_size = 8
    else:
        value_size = 4
    axis_size = 8 if value_size == 4 or (complex_values and writer == "qspice") else value_size
    record = axis_size + (n_vars - 1) * value_size
    on_disk = (size - header.data_start) // record
    if 0 < declared < on_disk:
        following = header.data_start + declared * record
        if _plot_starts_at(handle, following, width):
            return following
    # Every complete record counts, even past a nonzero declared count: LTspice
    # rewrites that count only now and then while it runs, so a killed run's
    # header lags the records behind it.
    points = on_disk
    if points == 0:
        return progress(0, None)
    if "fastaccess" in flags:
        # Axis first, as one contiguous block; only a finished plot is converted.
        at = header.data_start + (points - 1) * axis_size
    else:
        at = header.data_start + (points - 1) * record
    handle.seek(at)
    value = handle.read(8)
    if len(value) < 8:
        return progress(None, None)
    # A complex axis stores its real part first.
    return progress(points, struct.unpack("<d", value)[0])


def _plot_starts_at(handle: Any, offset: int, width: int) -> bool:
    handle.seek(offset)
    head = handle.read(len(_RAW_HEADER_UTF16[0]))
    return head.startswith(_RAW_HEADER_UTF16 if width == 2 else _RAW_HEADER_ASCII)


def _ascii_plot_end(handle: Any, data_start: int, size: int) -> int | None:
    """Where a finished ASCII plot's data ends: the next plot's offset, or ``size``.

    None when the search cap comes first. Searched a block at a time for the
    next ``Title:`` line, so the scan runs in C rather than line by line.
    """
    marker = b"\n" + _RAW_HEADER_ASCII
    handle.seek(data_start)
    position = data_start
    carry = b""
    while position - data_start < _ASCII_SKIP_CAP:
        block = handle.read(min(_ASCII_SKIP_BLOCK, _ASCII_SKIP_CAP - (position - data_start)))
        if not block:
            return size
        found = (carry + block).find(marker)
        if found >= 0:
            return position - len(carry) + found + 1
        # Keep enough of this block to find a marker that straddles the next.
        carry = block[-(len(marker) - 1) :]
        position += len(block)
    return None


def _ascii_last_point(
    handle: Any,
    data_start: int,
    size: int,
    width: int,
    n_vars: int,
) -> tuple[int | None, float | None]:
    """The point count and axis value of the last complete ASCII point.

    ``(0, None)`` when the data holds no complete point; ``(None, None)`` when
    the last one could not be found within the read cap or the data does not
    have the shape of raw values.
    """
    codec = _RAW_CODECS[width]
    window = _ASCII_TAIL_START + _ASCII_TAIL_PER_VARIABLE * n_vars
    while True:
        start = max(data_start, size - window)
        start += (start - data_start) % width
        handle.seek(start)
        text = handle.read(size - start).decode(codec, errors="replace")
        lines = text.split("\n")
        # The last element is the unterminated remainder of a line the
        # simulator was writing, or empty after the final newline.
        lines.pop()
        if start > data_start and lines:
            lines.pop(0)
        try:
            found = _last_complete_ascii_point(lines, n_vars)
        except ValueError:
            return None, None
        if found is not None:
            return found
        if start == data_start:
            return 0, None
        if window >= _ASCII_TAIL_CAP:
            return None, None
        window *= 4


def _last_complete_ascii_point(lines: list[str], n_vars: int) -> tuple[int, float | None] | None:
    """Walk back to the last point with all its values; None when not in these lines.

    A point is an index line (``<n>`` then the axis value) followed by one line
    per remaining variable. Raises ValueError on a line that fits neither shape.
    """
    values_after = 0
    for line in reversed(lines):
        tokens = line.split()
        if not tokens:
            continue
        if len(tokens) >= 2 and tokens[0].isdigit():
            if values_after == n_vars - 1:
                try:
                    axis_value: float | None = float(tokens[1].split(",")[0])
                except ValueError:
                    axis_value = None
                return int(tokens[0]) + 1, axis_value
            values_after = 0
            continue
        if len(tokens) != 1:
            raise ValueError(f"not a raw value line: {line[:80]!r}")
        values_after += 1
    return None


def _deck_axis_value(
    value: float | None,
    axis: str | None,
    plot: str | None,
    fields: dict[str, str],
) -> float | None:
    """A stored axis value in the coordinates the deck and every reader use.

    The same two corrections a full load makes: LTspice may store a time value
    negated (spicelib's ``Axis.get_wave`` takes the absolute value of a
    ``time`` axis), and a windowed transient's stored time starts from 0.
    """
    if value is None or not math.isfinite(value):
        return None
    if (axis or "").lower() == "time":
        value = abs(value)
    return value + _transient_offset(plot, fields.get("offset"))
