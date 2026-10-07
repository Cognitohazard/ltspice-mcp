"""Simulation-result analysis tools.

All tools in this module consume simulation output files (.raw, .log) and
return derived metrics. Organized by what the tool answers:

    Scalar summaries:
        signal_stats        — mean/RMS/pk-pk/etc for one signal
        query_value         — value at a specific time/frequency
        operating_point     — DC node voltages, branch currents, per-device
                              operating point (gm/gds/vth/…); device= scopes to one

    Waveform metrics (transient only, reject AC):
        edge_metrics        — rise/fall time + slew rate
        transient_response  — step settling or disturbance recovery
        timing_between      — signed delay between two signals
        periodic_metrics    — period/frequency/duty/jitter

    .MEAS extraction:
        measurement_stats   — aggregate .MEAS across sweep/MC
                                       (single-run .MEAS values are folded
                                        into simulation_summary)

    High-level overview:
        simulation_summary  — sim type, signals, warnings, key metrics
"""

import asyncio
import csv
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired

import numpy as np
from pydantic import Field

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError
from ltspice_mcp.lib import atomic_write, atomic_write_bytes, desktop, plot_settings, services
from ltspice_mcp.lib.ac_analysis import (
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.ac_structure import AcStructureResult, analyze_ac_structure
from ltspice_mcp.lib.format import si_prefix
from ltspice_mcp.lib.ltspice_bridge import BridgeError
from ltspice_mcp.lib.metrics import (
    guarded_axis,
    parse_time,
    window_indices,
)
from ltspice_mcp.lib.plot_html import (
    WIDGET_RESOURCE_URI,
    WIDGET_SPEC_META_KEY,
    build_plot_html,
)
from ltspice_mcp.lib.plot_svg import render_plot_svg
from ltspice_mcp.lib.raster import RasterSupport, RenderedImage, raster_support, render_image
from ltspice_mcp.lib.raw_parser import (
    get_step_count,
    safe_magnitude_db,
)
from ltspice_mcp.lib.signal_analysis import (
    TraceStats,
    downsample_minmax,
    summarize_trace,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    NEW_WORK_ANNOTATIONS,
    OBSERVATIONS_SCHEMA,
    RawSelectionFields,
    ToolInput,
    format_observations,
    format_response,
    image_content,
    registry,
    safe_path,
)
from ltspice_mcp.tools._schema import schema_from_typeddict

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _direct_source(
    raw_file: str | None,
    job_id: str | None,
    state: SessionState,
    *,
    plot_index: int = 0,
    dialect: str | None = None,
) -> services.AnalysisSource:
    """The source a direct call names, refusing an ambiguous or empty pair.

    A user-supplied ``raw_file`` is untrusted input → validated via
    ``safe_path``. Truthiness, not identity: an empty/whitespace raw_file
    (StrictModel strips to "") must count as absent, else it slips past and
    safe_path("") resolves to the working dir → a confusing "not a valid .raw"
    error downstream.
    """
    if bool(raw_file) == bool(job_id):
        # Complete redirect, so no generic hint: the analysis tools read an
        # existing result — a caller holding only a netlist runs it first.
        raise ResultError(
            "Pass exactly one of 'raw_file' or 'job_id'. Analysis tools read "
            "an existing result — if you only have a netlist, run_experiments "
            "produces the job_id/raw to analyze.",
            show_hint=False,
        )
    # Only ``raw_file`` is forwarded: an experiment's runs are case-addressed,
    # so ``_experiment_case`` has already answered every call that named a job.
    return services.resolve_analysis_source(
        state, raw_file=raw_file, plot_index=plot_index, dialect=dialect
    )


async def _experiment_case(
    raw_file: str | None,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    state: SessionState,
) -> services.RunContext | None:
    """The case a ``job_id`` names when it is a run_experiments job, else ``None``.

    Experiment runs are addressed by case, never through ``resolve_run`` — the
    same split ``Api.load_raw`` makes. ``case_id`` only means something here, so
    it is refused beside a raw_file. raw_file together with job_id is left to
    ``_direct_source``'s exclusivity error.
    """
    job = await services.resolve_job_async(job_id, state) if job_id and not raw_file else None
    if job is not None:
        return services.experiment_run_context(job, state, run_index=run_index, case_id=case_id)
    if case_id is not None:
        raise ResultError(
            "case_id selects a run_experiments case; pass it with that job's job_id."
        )
    return None


# ---------------------------------------------------------------------------
# export_waveform — full-fidelity CSV egress to disk
# ---------------------------------------------------------------------------

# Generous backstop against a pathological export exhausting memory/disk. Full
# fidelity is the contract, so this is high and RAISES with guidance to window —
# never silently truncates (no silent caps).
_EXPORT_MAX_ROWS = 20_000_000


def _csv_x_header(raw, analysis_type: str) -> str:
    """Name the selected descriptor's axis and its declared unit."""
    axis = raw.descriptor.axis
    if axis is None:
        raise ResultError("This plot has no sampled axis.")
    base = re.sub(r"[^0-9A-Za-z]+", "_", axis.name).strip("_") or "axis"
    if analysis_type in {"ac", "noise"} and axis.quantity == "frequency":
        base = "freq"
    return f"{base}_{axis.unit}" if axis.unit else base


def _step_label(values: Mapping[str, Any]) -> str:
    return ";".join(
        f"{key}={value:g}" if isinstance(value, (int, float)) else f"{key}={value}"
        for key, value in values.items()
    )


def _complex_columns(
    name: str, wave: np.ndarray, complex_format: str
) -> tuple[list[str], list[np.ndarray]]:
    """Expand one complex AC trace into (column names, real-valued arrays).

    Phase is the WRAPPED ``np.angle`` in degrees — the lossless primitive
    matching query_value/bode_metrics; a consumer who wants a continuous curve
    runs ``np.unwrap`` themselves. Magnitude uses the shared ``safe_magnitude_db``
    (floored to avoid -inf at exact zeros).
    """
    if complex_format == "re_im":
        return [f"{name}_re", f"{name}_im"], [np.real(wave), np.imag(wave)]
    if complex_format == "both":
        return (
            [f"{name}_mag_dB", f"{name}_phase_deg", f"{name}_re", f"{name}_im"],
            [safe_magnitude_db(wave), np.degrees(np.angle(wave)), np.real(wave), np.imag(wave)],
        )
    return (
        [f"{name}_mag_dB", f"{name}_phase_deg"],
        [safe_magnitude_db(wave), np.degrees(np.angle(wave))],
    )


def build_waveform_csv(
    raw,
    raw_path: Path,
    cols: list[services.Signal],
    n_steps: int,
    analysis_type: str,
    ts: float | None,
    te: float | None,
    complex_format: str,
    out_path: Path,
    should_abort: Callable[[], bool] | None = None,
    steps_to_export: list[int] | None = None,
    step_values: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Assemble tidy/long rows for every step, render CSV, write it atomically.

    Runs entirely in a worker thread: heavy numpy reads, O(N) row assembly, and
    file I/O. Returns only FACTS — the handler turns them into observations and
    builds the response on the event loop (the concurrency contract keeps
    response building off worker threads).
    """
    stepped = n_steps > 1
    if not cols:
        raise ResultError("Pass at least one signal to export.")
    has_axis = raw.descriptor.axis is not None
    if not has_axis and (ts is not None or te is not None):
        raise ResultError("A table plot has no axis to window.")
    x_header = _csv_x_header(raw, analysis_type) if has_axis else None
    step_dicts = step_values or []

    header: list[str] | None = None
    row_count = 0
    non_finite = 0
    had_complex = False
    empty_steps: list[int] = []
    win_lo: float | None = None
    win_hi: float | None = None

    # Stream rows straight into the atomic temp file: one step's data is the most
    # held in memory at once (no global rows list, no whole-CSV StringIO copy), so
    # a huge full-fidelity export does not balloon RAM. Any exception here unlinks
    # the temp and leaves the destination untouched.
    with atomic_write(out_path) as f:
        csv_writer = csv.writer(f)
        selected_steps = steps_to_export if steps_to_export is not None else list(range(n_steps))
        for step in selected_steps:
            axis = guarded_axis(raw, step) if has_axis else np.arange(len(cols[0].wave(raw, step)))
            lo, hi = window_indices(axis, ts, te)
            if lo >= hi:
                # This step's axis does not intersect the window (a step may end
                # earlier than its siblings). Skip it; surfaced as a fact.
                empty_steps.append(step)
                continue
            axis_w = axis[lo:hi]
            col_names: list[str] = []
            col_arrays: list[np.ndarray] = []
            for sig in cols:
                name = sig.name
                wave = sig.wave(raw, step)
                if wave.size == 0:
                    raise ResultError(f"Signal {name!r} has no data points at step {step}.")
                wave_w = wave[lo:hi]
                non_finite += int(np.count_nonzero(~np.isfinite(wave_w)))
                # Key on the trace's own dtype, not the run type: an AC raw can hold
                # a real trace, and a stray complex trace must not become one column.
                if np.iscomplexobj(wave_w):
                    had_complex = True
                    names, arrays = _complex_columns(
                        name, wave_w, complex_format if analysis_type == "ac" else "re_im"
                    )
                else:
                    names, arrays = [name], [wave_w]
                col_names.extend(names)
                col_arrays.extend(arrays)

            if header is None:
                prefix = ["step_index", "step_value"] if stepped else []
                header = [*prefix, *([x_header] if x_header else []), *col_names]
                csv_writer.writerow(header)

            if has_axis:
                lo0, hi0 = float(axis_w[0]), float(axis_w[-1])
                win_lo = lo0 if win_lo is None else min(win_lo, lo0)
                win_hi = hi0 if win_hi is None else max(win_hi, hi0)

            # .tolist() converts numpy -> python floats (full round-trippable repr)
            # at C speed; zip transposes columns into tidy/long rows.
            columns = [*([axis_w.tolist()] if has_axis else []), *(a.tolist() for a in col_arrays)]
            if stepped:
                label = _step_label(step_dicts[step]) if step < len(step_dicts) else ""
                rows = ([step, label, *values] for values in zip(*columns, strict=True))
            else:
                rows = zip(*columns, strict=True)
            chunk: list = []
            for row in rows:
                chunk.append(row)
                if len(chunk) >= 4096:
                    if should_abort is not None and should_abort():
                        raise AnalysisDeadlineExceeded(
                            "CSV artifact exceeded its analysis item deadline; "
                            "narrow the window or export fewer signals."
                        )
                    csv_writer.writerows(chunk)
                    chunk.clear()
            if chunk:
                if should_abort is not None and should_abort():
                    raise AnalysisDeadlineExceeded(
                        "CSV artifact exceeded its analysis item deadline; "
                        "narrow the window or export fewer signals."
                    )
                csv_writer.writerows(chunk)
            row_count += len(axis_w)
            if row_count > _EXPORT_MAX_ROWS:
                raise ResultError(
                    f"Export exceeds the {_EXPORT_MAX_ROWS:,}-row safety cap "
                    f"({row_count:,}+ rows). Narrow [t_start, t_end] or export fewer signals."
                )

        if header is None:
            # Every step was skipped — the window selected no samples anywhere.
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {n_steps} steps." if stepped else ".")
            )

    return {
        "row_count": row_count,
        "column_count": len(header),
        "columns": header,
        "n_steps": len(selected_steps),
        "window_used": [win_lo, win_hi] if win_lo is not None else [],
        "non_finite": non_finite,
        "had_complex": had_complex,
        "empty_steps": empty_steps,
        "step_values_available": any(bool(values) for values in step_dicts) if stepped else None,
    }


# ---------------------------------------------------------------------------
# Response TypedDicts — compose lib output + per-tool metadata (signal names)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# AC analysis tools
# ---------------------------------------------------------------------------


# ---- Input models ---------------------------------------------------------


# ---- Output schemas -------------------------------------------------------


# ---- Handlers -------------------------------------------------------------


# ---------------------------------------------------------------------------
# plot_waveform — interactive HTML chart opened on the local desktop
# ---------------------------------------------------------------------------

# Per-series point budget for the in-chat widget's chart spec (MCP Apps): the
# spec rides in the tool result (hidden in _meta), so it is decimated harder than
# the on-disk file (full fidelity stays in the file) — an overview for the eye.
# Bounded by the caller's max_points too (min of the two).
_WIDGET_SPEC_MAX_POINTS = 4_000
# Total serialized-spec ceiling for the widget. Over this the widget is skipped
# and delivery falls back to opening the full-fidelity file locally (surfaced as a
# fact) — keeps a pathological many-series/step overlay from shipping a huge _meta
# payload. Not a model-context limit (the spec is hidden from the model).
_WIDGET_MAX_BYTES = 4_000_000
_DEFAULT_PLOT_MAX_POINTS = 100_000
PLOT_MAX_POINTS_CEILING = 2_000_000
# Global backstop on a rendered panel. ``max_points`` caps each series, but the
# chart aligns a panel's series onto one x (uPlot.join), and a .step run whose
# steps have distinct axes multiplies series-count by the union length — so a
# many-step distinct-axis run could blow far past the per-series cap in the
# browser. Refuse with guidance before building it (no silent truncation).
_PLOT_MAX_CELLS = 10_000_000

# Most per-trace summaries one reply carries. A stepped or Monte Carlo overlay
# can plot hundreds of traces; past this many the reply lists the first ones and
# says how many there were, and analyze_results' signal_stats reads the rest.
_TRACE_SUMMARY_MAX = 32
# The attached image is budgeted by the points it draws in all. Its plot area is
# under 700 px wide, so past about two points per pixel column a series gains
# bytes, not detail (the min/max decimation keeps every spike), and a many-step
# overlay drawn at full per-series density takes seconds to rasterize.
_IMAGE_TOTAL_POINTS = 40_000
_IMAGE_SERIES_POINTS = (200, 1_400)
# Most panels a caller may lay out by hand.
_MAX_PANELS = 8


def _x_axis(raw, analysis_type: str) -> tuple[str, str | None]:
    """Name the selected descriptor's axis and its declared unit."""
    axis = raw.descriptor.axis
    if axis is None:
        raise ResultError("This plot has no sampled axis; inspect its table instead.")
    return (f"{axis.name} ({axis.unit})" if axis.unit else axis.name), axis.unit


def _trace_units(
    raw,
    cols: list[services.Signal],
) -> tuple[dict[str, str | None], list[str]]:
    """Selected descriptor units, with unknown dimensions explicitly reported."""
    units: dict[str, str | None] = {}
    unverified: list[str] = []
    descriptors = {trace.name: trace for trace in raw.descriptor.traces}
    for sig in cols:
        unit = descriptors[sig.trace].unit
        if sig.minus is not None and descriptors[sig.minus].unit != unit:
            unit = None
        units[sig.name] = unit
        if unit is None:
            unverified.append(sig.name)
    return units, unverified


def _unit_groups(
    cols: list[services.Signal], units: dict[str, str | None]
) -> list[list[services.Signal]]:
    """One panel per unit, in the order each unit first appears; traces with no
    known unit share a panel of their own. Scale is never guessed at: two
    same-unit traces of very different size share a panel unless the caller
    lays the panels out."""
    groups: dict[str | None, list[services.Signal]] = {}
    for sig in cols:
        groups.setdefault(units[sig.name], []).append(sig)
    return list(groups.values())


def _group_title(group: list[services.Signal], units: dict[str, str | None]) -> str:
    names = ", ".join(s.name for s in group) if len(group) <= 3 else f"{len(group)} signals"
    group_units = list(dict.fromkeys(u for s in group if (u := units[s.name])))
    return f"{names} ({', '.join(group_units)})" if group_units else names


@dataclass(frozen=True)
class PlotPlan:
    """What a plot draws: which traces share a panel, what the panels and axes
    are called, and which steps and window it covers. Resolved once per call;
    everything a plot renders is built from one plan."""

    groups: list[list[services.Signal]]
    units: dict[str, str | None]
    steps: list[int]
    step_dicts: list[dict[str, Any]]
    analysis_type: str
    x_is_log: bool
    x_label: str
    x_unit: str | None
    ts: float | None
    te: float | None
    annotate: bool = False
    #: The run is a .step sweep, so each summary names its step even when one
    #: step was selected.
    stepped: bool = False
    #: Signals whose descriptor does not establish a unit.
    unverified_units: tuple[str, ...] = ()

    @property
    def signals(self) -> list[services.Signal]:
        return [sig for group in self.groups for sig in group]


def plan_plot(
    raw,
    groups: list[list[services.Signal]],
    *,
    split_by_unit: bool,
    netlist: Path | None,
    steps: list[int],
    step_dicts: list[dict[str, Any]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    annotate: bool = False,
) -> PlotPlan:
    """Resolve what a plot draws. ``groups`` are the requested signals, one list
    per panel; with ``split_by_unit`` they are regrouped into one panel per unit.

    Uses resident descriptors without reopening a source deck or log.
    """
    cols = [sig for group in groups for sig in group]
    units, unverified = _trace_units(raw, cols)
    x_label, x_unit = _x_axis(raw, analysis_type)
    return PlotPlan(
        groups=_unit_groups(cols, units) if split_by_unit else groups,
        units=units,
        steps=steps,
        step_dicts=step_dicts,
        analysis_type=analysis_type,
        x_is_log=x_is_log,
        x_label=x_label,
        x_unit=x_unit,
        ts=ts,
        te=te,
        annotate=annotate,
        stepped=get_step_count(raw) > 1,
        unverified_units=tuple(unverified),
    )


def _plot_filename(
    raw_path: Path, analysis_type: str, job_id: str | None, run_index: int, *, plot_index: int = 0
) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    run = f"_run{run_index}" if job_id else ""
    return f"{raw_path.stem}_{analysis_type}{run}_plot{plot_index}_{stamp}.html"


def _to_json_floats(arr: np.ndarray) -> list[float | None]:
    """Array -> JSON-safe list; non-finite samples become ``None`` (a uPlot gap).

    Pairs with ``json.dumps(allow_nan=False)`` in the renderer: a literal NaN/Inf
    token would make the browser's ``JSON.parse`` reject the whole blob, silently
    blanking the chart — so non-finite values are converted to null here.
    """
    return [v if math.isfinite(v) else None for v in np.asarray(arr, dtype=float).tolist()]


def _plot_cells_exceeded(n_rows: int, x_len: int) -> ResultError:
    return ResultError(
        f"This plot would render ~{n_rows * x_len:,} cells ({n_rows - 1} series x "
        f"{x_len:,} x-points), over the {_PLOT_MAX_CELLS:,} cap. Stepped runs with "
        "distinct per-step axes inflate the shared x-axis — plot fewer signals, a "
        "single step (step=N), or narrow [t_start, t_end]."
    )


def _panel(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    x_scale: str,
    x_label: str,
    y_label: str,
) -> dict:
    """One chart panel from per-series ``(x, y, label)``.

    Series that share an axis share a table ``[x, y...]``; a series on an axis
    of its own — each step of a transient ``.step`` run has its own adaptive
    time vector — starts a new table. The renderers align tables themselves
    (the chart with ``uPlot.join``, which joins a line across another table's
    samples and breaks it only at its own nulls), so no series is padded.

    Guards the size the chart will build in two cheap stages before anything is
    converted: the longest series is a lower bound on the aligned axis (stage
    1), and the union of the tables' axes is its exact length (stage 2, only
    when there is more than one table).
    """
    n_rows = len(series) + 1  # the x row plus one row per series
    longest = max(len(s[0]) for s in series)
    if n_rows * longest > _PLOT_MAX_CELLS:
        raise _plot_cells_exceeded(n_rows, longest)
    # Consecutive series on the same axis share a table. Compared by value, not
    # merged with np.unique: solver restarts emit duplicate timepoints, which a
    # shared table must keep.
    groups: list[tuple[np.ndarray, list[np.ndarray]]] = []
    for x, y, _ in series:
        if groups and len(groups[-1][0]) == len(x) and np.array_equal(groups[-1][0], x):
            groups[-1][1].append(y)
        else:
            groups.append((x, [y]))
    if len(groups) > 1:
        union = len(np.unique(np.concatenate([x for x, _ in groups])))
        if n_rows * union > _PLOT_MAX_CELLS:
            raise _plot_cells_exceeded(n_rows, union)
    return {
        "x_scale": x_scale,
        "x_label": x_label,
        "y_label": y_label,
        "series": [{"label": label} for _, _, label in series],
        "tables": [[_to_json_floats(x), *(_to_json_floats(y) for y in ys)] for x, ys in groups],
    }


def _compact_hz(f: float) -> str:
    """Render a frequency (Hz) compactly with an engineering suffix.

    e.g. 1200 -> "1.2k", 20000 -> "20k", 3.4e6 -> "3.4M". Used for the on-plot
    corner-marker labels, which must stay short.
    """
    if not np.isfinite(f) or f <= 0:
        return "?"
    scale, suffix = si_prefix(f)
    return f"{f / scale:.1f}".rstrip("0").rstrip(".") + suffix


# Structure reading is density-robust (validated at 10 and 50 points/decade), so a
# huge AC sweep is decimated to this many evenly-spaced samples before the
# per-sample analysis — bounding annotation cost without changing the result (the
# plot itself is downsampled separately by max_points).
_ANNOTATION_MAX_POINTS = 4000

# Corner kinds whose marker is a zero (circle); everything else is a pole (cross).
_ZERO_KINDS = ("real_zero", "complex_zero_pair")


def _ac_annotations(freq: np.ndarray, h: np.ndarray) -> tuple[list[dict], bool]:
    """Corner markers + non-minimum-phase flag for a single AC trace.

    Runs :func:`analyze_ac_structure` (on a bounded, evenly-spaced subset of large
    sweeps) and projects each located corner to one annotation: an x at the
    corner's geometric center (``sqrt(f_lo*f_hi)``), a compact label naming the
    kind/frequency (Q appended for a complex pair, ``" (merged)"`` for an
    under-resolved cluster), and a ``marker`` of ``"pole"`` (drawn as a cross) or
    ``"zero"`` (a circle). Returns ``(annotations, non_minimum_phase)``. All x
    values are finite (json.dumps(allow_nan=False) is used for the widget).
    """
    if len(freq) > _ANNOTATION_MAX_POINTS:
        idx = np.linspace(0, len(freq) - 1, _ANNOTATION_MAX_POINTS).astype(int)
        freq, h = freq[idx], h[idx]
    result: AcStructureResult = analyze_ac_structure(freq, h)
    annotations: list[dict] = []
    for corner in result["corners"]:
        f_lo = float(corner["f_lo"])
        f_hi = float(corner["f_hi"])
        center = float(np.sqrt(f_lo * f_hi)) if f_lo > 0 and f_hi > 0 else max(f_lo, f_hi)
        if not np.isfinite(center) or center <= 0:
            continue
        kind = corner["kind"]
        if kind == "real_pole":
            label = f"pole ~{_compact_hz(center)}"
        elif kind == "real_zero":
            label = f"zero ~{_compact_hz(center)}"
        else:
            noun = "zero pair" if kind == "complex_zero_pair" else "pole pair"
            q = corner["q"]
            label = f"{noun} ~{_compact_hz(center)}"
            if q is not None and np.isfinite(q):
                label += f" Q{q:.1f}".rstrip("0").rstrip(".")
        if corner["merged"]:
            label += " (merged)"
        marker = "zero" if kind in _ZERO_KINDS else "pole"
        annotations.append({"x": center, "label": label, "kind": kind, "marker": marker})
    return annotations, bool(result["non_minimum_phase"])


class TraceSummary(TraceStats):
    """One plotted trace's summary, as the reply carries it."""

    signal: str
    step: NotRequired[int]
    label: NotRequired[str]
    panel: int
    unit: str | None
    phase_initial_deg: NotRequired[float | None]
    phase_final_deg: NotRequired[float | None]


@dataclass(frozen=True)
class _Trace:
    """One plotted trace at full resolution: a signal at one step, windowed.

    ``ys`` holds one array per panel row the trace is drawn in: the values, or
    an AC trace's magnitude (dB) and unwrapped phase (deg).
    """

    label: str
    x: np.ndarray
    ys: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class PlotData:
    """Every trace a plan draws, read, windowed and measured once.

    The HTML file, the in-chat widget spec and the attached image are each a
    projection of this at its own point budget (:func:`plot_spec`), so the raw
    is read once however many are built, and the summaries, the AC annotations
    and the coverage facts all come from the full-resolution data.
    """

    plan: PlotPlan
    #: Per requested panel group: its row titles (one per panel it becomes) and
    #: its traces.
    groups: list[tuple[list[str], list[_Trace]]]
    annotations: list[dict] | None
    nmp: bool | None
    facts: dict[str, Any]


def extract_plot(raw, plan: PlotPlan, *, summary_limit: int = 0) -> PlotData:
    """Read, window and measure every trace ``plan`` draws (no I/O but the raw).

    Runs in a worker thread (heavy numpy). Each step's axis is fetched and
    windowed once, however many signals are drawn over it. The first
    ``summary_limit`` traces are summarized (:func:`summarize_trace`) from every
    sample in the window; ``facts["traces_total"]`` counts them all.
    """
    is_ac = plan.analysis_type == "ac"
    multi = len(plan.steps) > 1

    def _label(col: str, step: int) -> str:
        if not multi:
            return col
        sv = _step_label(plan.step_dicts[step]) if step < len(plan.step_dicts) else ""
        return f"{col} [{sv}]" if sv else f"{col} [step {step}]"

    windows: dict[int, tuple[np.ndarray, int, int]] = {}
    empty_steps: list[int] = []
    for step in plan.steps:
        axis = guarded_axis(raw, step)
        lo, hi = window_indices(axis, plan.ts, plan.te)
        if lo < hi:
            windows[step] = (axis[lo:hi], lo, hi)
        else:
            # This step's axis does not intersect the window (a step may end
            # earlier than its siblings). Skip it; surfaced as a fact.
            empty_steps.append(step)
    if not windows:
        raise ResultError(
            "The [t_start, t_end] window selects no samples"
            + (f" in any of the {len(plan.steps)} steps." if multi else ".")
        )

    # Single-trace annotation reads the full-resolution complex response.
    annotate_one = is_ac and plan.annotate and len(plan.signals) == 1 and len(windows) == 1
    annotations: list[dict] | None = None
    nmp: bool | None = None
    non_finite = 0
    phase_warnings: list[str] = []
    summaries: list[TraceSummary] = []
    total = 0
    groups: list[tuple[list[str], list[_Trace]]] = []
    suffix_titles = len(plan.groups) > 1
    for group in plan.groups:
        title = _group_title(group, plan.units)
        if is_ac:
            suffix = f" — {title}" if suffix_titles else ""
            rows = [f"Magnitude (dB){suffix}", f"Phase (deg){suffix}"]
        else:
            rows = [title]
        panel = sum(len(r) for r, _ in groups)
        traces: list[_Trace] = []
        for col in group:
            for step, (axis_w, lo, hi) in windows.items():
                wave = col.wave(raw, step)[lo:hi]
                if is_ac:
                    x, h = prepare_ac_arrays(axis_w, wave)
                    phase, warns = unwrap_phase_safe(h)
                    phase_warnings.extend(warns)
                    ys: tuple[np.ndarray, ...] = (safe_magnitude_db(h), phase)
                    if annotate_one:
                        annotations, nmp = _ac_annotations(x, h)
                else:
                    x, ys = axis_w, (np.real(wave) if np.iscomplexobj(wave) else wave,)
                    if np.iscomplexobj(wave):
                        traces.append(
                            _Trace(f"{_label(col.name, step)} (imag)", x, (np.imag(wave),))
                        )
                        non_finite += int(np.count_nonzero(~np.isfinite(np.imag(wave))))
                non_finite += sum(int(np.count_nonzero(~np.isfinite(y))) for y in ys)
                label = _label(col.name, step)
                if not is_ac and np.iscomplexobj(wave):
                    label += " (real)"
                if total < summary_limit:
                    summaries.append(_trace_summary(plan, col, step, label, panel, x, ys))
                total += 1
                traces.append(_Trace(label, x, ys))
        groups.append((rows, traces))

    everything = [t for _, traces in groups for t in traces]
    facts = {
        "empty_steps": empty_steps,
        "non_finite": non_finite,
        "phase_warnings": phase_warnings,
        "window_used": [
            min(float(t.x[0]) for t in everything),
            max(float(t.x[-1]) for t in everything),
        ],
        "step_values_available": (
            any(bool(values) for values in plan.step_dicts) if multi else None
        ),
        "traces": summaries,
        "traces_total": total,
    }
    return PlotData(plan, groups, annotations, nmp, facts)


def _trace_summary(
    plan: PlotPlan,
    col: services.Signal,
    step: int,
    label: str,
    panel: int,
    x: np.ndarray,
    ys: tuple[np.ndarray, ...],
) -> TraceSummary:
    """One trace's summary. An AC trace is summarized in dB, with its unwrapped
    phase at both ends; only a transient reports a (time-weighted) mean."""
    is_ac = len(ys) == 2
    entry = TraceSummary(
        signal=col.name,
        panel=panel,
        unit="dB" if is_ac else plan.units[col.name],
        **summarize_trace(x, ys[0], time_weighted_mean=plan.analysis_type == "transient"),
    )
    if plan.stepped:
        entry["step"] = step
        if label != col.name:
            entry["label"] = label
    if is_ac:
        ends = summarize_trace(x, ys[1], time_weighted_mean=False)
        entry["phase_initial_deg"] = ends["initial"]
        entry["phase_final_deg"] = ends["final"]
    return entry


def plot_spec(plot: PlotData, max_points: int) -> tuple[dict, dict]:
    """Project extracted traces into a renderer-ready spec at ``max_points`` per series.

    Returns ``(spec, facts)``: ``spec`` is what :func:`build_plot_html`, the
    widget and :func:`render_plot_svg` draw; ``facts`` are this projection's
    own (panel and series counts, points per series, whether decimation
    engaged). A series over the budget is reduced by min/max-preserving
    decimation, whose output axis depends only on the input axis, so series
    that shared an axis still share one.
    """
    plan = plot.plan
    x_scale = "log" if plan.analysis_type == "ac" or plan.x_is_log else "linear"
    panels: list[dict] = []
    points_per_series: list[int] = []
    downsampled = False
    for rows, traces in plot.groups:
        reduced: list[tuple[str, np.ndarray, list[np.ndarray]]] = []
        for t in traces:
            x, ys = t.x, list(t.ys)
            if len(x) > max_points:
                downsampled = True
                cut = [downsample_minmax(t.x, y, max_points) for y in t.ys]
                x, ys = cut[0][0], [y for _, y in cut]
            points_per_series.append(len(x))
            reduced.append((t.label, x, ys))
        for r, title in enumerate(rows):
            series = [(x, ys[r], label) for label, x, ys in reduced]
            panels.append(_panel(series, x_scale, plan.x_label, title))
    spec: dict[str, Any] = {"analysis_type": plan.analysis_type, "panels": panels}
    if plot.annotations is not None:
        spec["annotations"] = plot.annotations
        spec["nmp"] = plot.nmp
    facts = {
        "panels": len(panels),
        "series_count": sum(len(p["series"]) for p in panels),
        "points_per_series": points_per_series,
        "downsampled": downsampled,
    }
    return spec, facts


def write_plot_file(
    plot: PlotData, raw_path: Path, max_points: int, out_path: Path, title: str
) -> dict:
    """Assemble the offline HTML at ``max_points`` and write it atomically; return
    the projection's facts. Runs in a worker thread (HTML build + file I/O)."""
    spec, facts = plot_spec(plot, max_points)
    summary = f"{raw_path.stem} — {plot.plan.analysis_type}: {facts['series_count']} series"
    html_str = build_plot_html(spec, title=title, summary=summary)
    with atomic_write(out_path) as f:
        f.write(html_str)
    return facts


def build_plot_file(
    raw, raw_path: Path, plan: PlotPlan, max_points: int, out_path: Path, title: str
) -> dict:
    """Extract, project and write one HTML chart; return its facts.

    For a caller that needs only the file (the ``analyze_results`` plot recipe):
    no trace is summarized. Runs in a worker thread.
    """
    plot = extract_plot(raw, plan)
    return {**plot.facts, **write_plot_file(plot, raw_path, max_points, out_path, title)}


def _widget_spec_json(plot: PlotData, max_points: int) -> str:
    """Build the compact widget chart spec and serialize it — all in the worker.

    Returns the spec as a JSON string for the result ``_meta`` (read by the
    widget in ``app.ontoolresult``); raises ``ResultError`` like
    :func:`plot_spec` (e.g. the cell cap), which the handler catches to fall
    back to local-open delivery.
    """
    spec, _ = plot_spec(plot, max_points)
    return json.dumps(spec, ensure_ascii=True, allow_nan=False)


def build_plot_image(
    plot: PlotData, max_points: int, title: str, png_path: Path
) -> RenderedImage | RasterSupport:
    """Render the plot as a PNG and write it, or say why this host cannot.

    Runs in a worker thread (SVG build, rasterization, file I/O). Draws the same
    panels as the chart, with each series' points set by the total the image
    draws (``_IMAGE_TOTAL_POINTS``) and never above ``max_points``. Without a
    rasterizer it returns the :class:`RasterSupport` naming what is missing and
    how to install it, checked first so nothing is drawn only to be discarded.
    """
    support = raster_support()
    if not support.png:
        return support
    n_series = sum(len(t.ys) for _, traces in plot.groups for t in traces)
    lo, hi = _IMAGE_SERIES_POINTS
    per_series = min(max_points, max(lo, min(hi, _IMAGE_TOTAL_POINTS // n_series)))
    spec, _ = plot_spec(plot, per_series)
    image = render_image(render_plot_svg(spec, title=title), image_format="png", scale=1.0)
    if image.png_unavailable is not None:
        return image.png_unavailable
    atomic_write_bytes(png_path, image.data, durable=False)
    return image


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _summary_lines(traces: list[TraceSummary], x_unit: str | None) -> list[str]:
    """One text line per summarized trace (the text mirror of ``traces``)."""
    at = f" {x_unit}" if x_unit else ""
    lines = []
    for t in traces:
        name = t.get("label") or (
            f"{t['signal']} [step {t['step']}]" if "step" in t else t["signal"]
        )
        unit = f" ({t['unit']})" if t.get("unit") else ""
        parts = [
            f"min {_fmt(t['min'])} at {_fmt(t['x_at_min'])}{at}",
            f"max {_fmt(t['max'])} at {_fmt(t['x_at_max'])}{at}",
            f"initial {_fmt(t['initial'])}",
            f"final {_fmt(t['final'])}",
        ]
        if "mean" in t:
            parts.append(f"mean {_fmt(t['mean'])}")
        if "phase_initial_deg" in t and "phase_final_deg" in t:
            parts.append(
                f"phase {_fmt(t['phase_initial_deg'])} to {_fmt(t['phase_final_deg'])} deg"
            )
        if non_finite := t.get("non_finite"):
            parts.append(f"{non_finite} non-finite samples left out")
        lines.append(f"  {name}{unit}, panel {t['panel']}: " + ", ".join(parts))
    return lines


class PlotWaveformInput(RawSelectionFields, ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this or ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Plot one run of a finished job instead of a raw_file path; pick "
            "the run with ``run_index`` (or ``case_id`` for an experiment)."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    case_id: str | None = Field(
        default=None,
        description=(
            "For a run_experiments job_id: the case to plot, as named in its receipt "
            "and analysis identities (an alternative to ``run_index``)."
        ),
    )
    signals: list[str] | Literal["all"] = Field(
        default="all",
        description=(
            "Trace names or node pairs to plot (e.g. ['V(out)', 'V(inp,inn)']), or "
            "'all' for every non-axis trace."
        ),
    )
    panels: list[Annotated[list[str], Field(min_length=1)]] | None = Field(
        default=None,
        min_length=1,
        max_length=_MAX_PANELS,
        description=(
            "One list of signals per panel, e.g. [['V(out)'], ['V(sense)']] to part "
            "traces of very different size. Replaces signals."
        ),
    )
    step: int | None = Field(
        default=None,
        description=(
            "For a .step run: omit to overlay all steps as separate traces, or give "
            "a 0-based step index to plot just that one."
        ),
    )
    t_start: str | None = Field(
        default=None,
        description="Window start in SPICE notation (e.g. '1m', '1k'); bounds the plotted range.",
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation.",
    )
    max_points: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Per-series point budget before a min/max-preserving downsample engages "
            f"(default {_DEFAULT_PLOT_MAX_POINTS}). Full fidelity below this; spikes "
            "are preserved when it engages."
        ),
    )
    open: bool | None = Field(
        default=None,
        description=(
            "Open the HTML in a local browser (default [analysis] open_plot, on); "
            "ignored for an in-chat widget."
        ),
    )
    attach_plot: bool | None = Field(
        default=None,
        description=(
            "Also return a PNG of the chart for a vision model, ~1k tokens "
            "(default [analysis] attach_plot, off)."
        ),
    )
    in_ltspice: bool = Field(
        default=False,
        description=(
            "Also open the run in the user's open LTspice window (26.1+) with "
            "these traces drawn; writes a .plt beside the results file."
        ),
    )
    annotate: bool = Field(
        default=True,
        description=(
            "Annotate an AC/Bode plot with detected corner markers and an "
            "out-of-phase-zero / delay flag. AC only; ignored for transient/DC."
        ),
    )
    out_dir: str | None = Field(
        default=None,
        description=(
            "Directory for the HTML (under an allowed path; created if needed). "
            "Default: the server's store, '.ltspice-mcp/plots/' in the working "
            "directory unless LTSPICE_MCP_STORE_DIR moves it."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_format": {"type": "string"},
        "mime_type": {"type": "string"},
        "scale": {"type": ["number", "null"]},
        "width": {"type": ["integer", "null"]},
        "height": {"type": ["integer", "null"]},
        "bytes": {"type": "integer"},
        "estimated_tokens": {"type": ["integer", "null"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["image_format", "width", "height", "bytes"],
}


_ALREADY_OPEN_NOTE = (
    "A results file LTspice already had open keeps the traces it was showing; "
    "close it there and ask again to change them."
)


def _show_in_ltspice(
    state: SessionState, results: Path, analysis: str, panes: list[list[str]]
) -> dict[str, Any]:
    """Open ``results`` in an LTspice window with ``panes`` drawn. Blocking.

    The traces are written first, as the plot settings file LTspice loads when
    it opens the results (``lib/plot_settings.py``), so they are there for a
    person who opens the file by hand when no window could be reached. Nothing
    here fails the plot: the chart and its numbers are already made, and what
    happened in LTspice is a fact beside them.
    """
    report: dict[str, Any] = {
        "shown": False,
        "results": str(results),
        "plot_settings": None,
        "panes": panes,
    }
    windows = state.open_windows
    if not windows.available:
        report["reason"] = f"LTspice windows cannot be reached here: {windows.unavailable}"
        return report
    try:
        left_alone = plot_settings.write_beside(results, analysis, panes)
    except (plot_settings.PlotSettingsError, OSError) as error:
        left_alone = f"the plot settings could not be written ({error})"
    if left_alone is None:
        report["plot_settings"] = str(plot_settings.settings_path(results))
    try:
        report["pid"], report["version"] = windows.show_results(results)
    except BridgeError as error:
        report["reason"] = str(error)
        if left_alone is None:
            report["note"] = (
                "Opened by hand in LTspice, the results file shows these traces: the "
                "plot settings beside it name them."
            )
        return report
    report["shown"] = True
    report["note"] = _ALREADY_OPEN_NOTE if left_alone is None else f"{left_alone}."
    return report


def _ltspice_line(report: Mapping[str, Any]) -> str:
    if report["shown"]:
        return f"Opened in LTspice {report['version']} (process {report['pid']}). {report['note']}"
    return f"Not shown in LTspice: {report['reason']}. " + str(report.get("note", ""))


@registry.tool(
    name="plot_waveform",
    title="Plot Waveforms",
    description=(
        "Render an interactive chart (zoom/pan/hover) of one or more signals for "
        "a person to look at. The chart type follows the run (transient, DC sweep, "
        "AC Bode, noise), a .step or Monte Carlo run overlays every step, and each "
        "unit gets its own panel.\n\n"
        "Writes a self-contained HTML file and returns its path — into "
        "``out_dir`` if given, else the server's store. On a host that "
        "supports MCP Apps the chart is also "
        "embedded as an in-chat widget; otherwise it opens in your local "
        "browser. The reply summarizes each trace (min and max and where, first "
        "and final value, mean on a transient); attach_plot adds a PNG; "
        "in_ltspice also opens the run in the user's LTspice window.\n\n"
        "For more numbers use analyze_results: the waveform recipe returns a "
        "table (or CSV on disk), and signal_stats and the bode_* recipes return "
        "scalars."
    ),
    input_model=PlotWaveformInput,
    annotations=NEW_WORK_ANNOTATIONS,
    # MCP Apps (SEP-1865): declare the in-chat renderer so an apps-capable host
    # fetches it via resources/read and pipes the chart spec into it.
    meta={"ui": {"resourceUri": WIDGET_RESOURCE_URI}},
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "analysis_type": {"type": "string"},
            "plot_index": {"type": "integer", "minimum": 0},
            "descriptor": {"type": "object"},
            "signals": {"type": "array", "items": {"type": "string"}},
            "n_steps": {"type": "integer"},
            "steps_plotted": {"type": "integer"},
            "panels": {"type": "integer"},
            "series_count": {"type": "integer"},
            "points_per_series": {"type": "array", "items": {"type": "integer"}},
            "max_points": {"type": "integer"},
            "downsampled": {"type": "boolean"},
            "window_used": {"type": "array", "items": {"type": "number"}},
            "x_unit": {"type": ["string", "null"]},
            "traces": {"type": "array", "items": schema_from_typeddict(TraceSummary)},
            "traces_truncated": {"type": "integer"},
            "delivery": {"type": "string", "enum": ["terminal", "ui"]},
            "opened": {"type": "boolean"},
            "opener": {"type": ["string", "null"]},
            "ltspice": {
                "type": "object",
                "description": "Present with in_ltspice: what happened in the LTspice window.",
                "properties": {
                    "shown": {"type": "boolean"},
                    "pid": {"type": "integer", "description": "The LTspice process."},
                    "version": {"type": "string"},
                    "results": {"type": "string", "description": "The file opened."},
                    "plot_settings": {
                        "type": ["string", "null"],
                        "description": "The .plt written beside it; null when none was.",
                    },
                    "panes": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "The traces asked for, per pane.",
                    },
                    "reason": {"type": "string", "description": "Why it was not shown."},
                    "note": {"type": "string"},
                },
                "required": ["shown", "results", "plot_settings", "panes"],
            },
            "image": _IMAGE_SCHEMA,
            "image_path": {"type": "string"},
            "observations": OBSERVATIONS_SCHEMA,
            # Carried only when the server adds its one read-the-guide
            # reminder to a session's first reply (server.call_tool).
            "hint": {"type": "string"},
        },
    },
)
async def handle_plot_waveform(args: PlotWaveformInput, state: SessionState):
    case = await _experiment_case(args.raw_file, args.job_id, args.run_index, args.case_id, state)
    if case is not None:
        source = services.source_for_run(case, plot_index=args.plot_index, dialect=args.dialect)
        run_index = case.identity["run_index"]
        netlist: Path | None = case.netlist
    else:
        source = _direct_source(
            args.raw_file, args.job_id, state, plot_index=args.plot_index, dialect=args.dialect
        )
        run_index = args.run_index
        netlist = source.netlist
    raw_path = source.raw
    fmt = args.format
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")
    if args.panels is not None and args.signals != "all":
        raise ResultError("Pass signals or panels, not both: panels names the signals it plots.")

    raw = await services.load_raw(source, state)
    assert raw_path is not None
    if raw.descriptor.axis is None:
        raise ResultError("This plot has no sampled axis; use inspect results table instead.")
    # Refuse unsupported coordinates before building a plot.
    guarded_axis(raw, 0, raw_path)

    analysis_type = raw.descriptor.analysis
    x_is_log = raw.descriptor.axis is not None and raw.descriptor.axis.quantity == "frequency"

    trace_names = raw.get_trace_names()
    assert raw.descriptor.axis is not None
    axis_name = raw.descriptor.axis.name
    # A signal list is one panel's worth; resolved once, whatever the spelling.
    requested = (
        args.panels
        if args.panels is not None
        else None
        if args.signals == "all"
        else [args.signals]
    )
    groups: list[list[services.Signal]]
    if requested is None:
        groups = [[services.Signal(name, name) for name in trace_names if name != axis_name]]
    else:
        groups = []
        panel_of: dict[str, int] = {}
        for index, names in enumerate(requested):
            group: list[services.Signal] = []
            for name in names:
                sig = services.resolve_signal(raw, name)
                if sig.name == axis_name:
                    raise ResultError(f"{name!r} is the sweep axis, not a signal column.")
                if sig.name in panel_of:
                    if panel_of[sig.name] != index:
                        raise ResultError(
                            f"{sig.name!r} is named in more than one panel; give each "
                            "signal one panel."
                        )
                    continue
                panel_of[sig.name] = index
                group.append(sig)
            groups.append(group)
    cols = [sig for group in groups for sig in group]
    if not cols:
        raise ResultError("No signal traces to plot (the result has only an axis).")

    n_steps = get_step_count(raw)
    if args.step is not None:
        services.validate_step(raw, args.step)
        steps_to_plot = [args.step]
    else:
        steps_to_plot = list(range(n_steps))

    ts = parse_time(args.t_start, "t_start")
    te = parse_time(args.t_end, "t_end")

    step_dicts = [dict(step.parameters) for step in raw.descriptor.steps]

    max_points = min(args.max_points or _DEFAULT_PLOT_MAX_POINTS, PLOT_MAX_POINTS_CEILING)
    # Asked for in LTspice, the chart is not also opened in a browser unless
    # that was asked for too.
    open_default = state.config.open_plot and not args.in_ltspice
    open_locally = open_default if args.open is None else args.open
    attach = state.config.attach_plot if args.attach_plot is None else args.attach_plot

    plan = await asyncio.to_thread(
        plan_plot,
        raw,
        groups,
        split_by_unit=args.panels is None,
        netlist=netlist,
        steps=steps_to_plot,
        step_dicts=step_dicts,
        analysis_type=analysis_type,
        x_is_log=x_is_log,
        ts=ts,
        te=te,
        annotate=args.annotate,
    )

    # Into ``out_dir`` when named, else the store: never beside a raw the
    # caller named, which on WSL can sit in a Windows temp the client cannot read.
    dest_dir = safe_path(args.out_dir, state) if args.out_dir else state.store.plots_dir
    out_path = dest_dir / _plot_filename(
        raw_path, analysis_type, args.job_id, run_index, plot_index=raw.plot_index
    )

    title = f"{raw_path.stem} — {analysis_type}"
    try:
        plot = await asyncio.to_thread(extract_plot, raw, plan, summary_limit=_TRACE_SUMMARY_MAX)
        written = await asyncio.to_thread(
            write_plot_file, plot, raw_path, max_points, out_path, title
        )
    except ValueError as e:
        raise ResultError(f"Failed to build the plot (corrupt or truncated .raw?): {e}") from e
    facts = {**plot.facts, **written}

    # Is this an MCP Apps host? Resolved ON THE LOOP (the request context is a
    # ContextVar not propagated into to_thread workers). If so, build a COMPACT spec
    # to ride in the result _meta (the host pipes it into the widget via
    # ontoolresult; the tool declares ui.resourceUri and the host fetched the
    # renderer via resources/read). Decimated harder than the on-disk file (bounded
    # by the caller's max_points too); the spec build AND its JSON serialization run
    # in the worker so nothing heavy hits the loop. If it can't be built within
    # budget, widget_spec_json stays None and we open the full-fidelity file locally.
    from ltspice_mcp.server import get_client_capabilities

    is_ui = desktop.resolve_delivery_channel(get_client_capabilities()) == "ui"
    widget_spec_json: str | None = None
    widget_skipped: str | None = None
    if is_ui:
        budget = min(max_points, _WIDGET_SPEC_MAX_POINTS)
        try:
            widget_spec_json = await asyncio.to_thread(_widget_spec_json, plot, budget)
        except ResultError as e:
            widget_skipped = str(e)
        if widget_spec_json is not None and len(widget_spec_json) > _WIDGET_MAX_BYTES:
            widget_skipped = (
                f"The widget chart ({len(widget_spec_json):,} bytes) exceeds the "
                f"{_WIDGET_MAX_BYTES:,}-byte in-result budget — plot fewer "
                "signals/steps or a single step (step=N) for an in-chat widget."
            )
            widget_spec_json = None

    # No widget (terminal host, or a UI build that fell back) → open the file.
    # Opened before the image renders: it needs only the file.
    opened, opener = False, None
    if widget_spec_json is None and open_locally:
        opened, opener = await asyncio.to_thread(desktop.open_in_desktop, out_path)

    shown_in_ltspice: dict[str, Any] | None = None
    if args.in_ltspice:
        panes = [[sig.name for sig in group] for group in plan.groups if group]
        shown_in_ltspice = await asyncio.to_thread(
            _show_in_ltspice, state, raw_path, raw.descriptor.original_plot_name, panes
        )

    # The model's own frame: a static PNG of the same panels, on request.
    image: RenderedImage | None = None
    image_problem: str | None = None
    png_path = out_path.with_suffix(".png")
    if attach:
        try:
            rendered = await asyncio.to_thread(build_plot_image, plot, max_points, title, png_path)
        # An optional add-on: the chart and its numbers are already built, so a
        # render failure is reported beside them rather than failing the call.
        except Exception as e:
            image_problem = f"The image could not be rendered ({type(e).__name__}: {e})."
        else:
            if isinstance(rendered, RasterSupport):
                # SVG path data as text is no use to a model and costs far more
                # than the picture would, so the fallback is reported, not sent.
                image_problem = f"No image attached: {rendered.reason}. To fix: {rendered.remedy}."
            else:
                image = rendered

    traces: list[TraceSummary] = facts["traces"]
    traces_total: int = facts["traces_total"]

    # Surface FACTS, not verdicts (result-trust doctrine). What the reply's own
    # fields already say (the panels, series and steps drawn, `opened`,
    # `delivery`) is not restated here.
    observations: list[dict] = []
    if traces_total > len(traces):
        observations.append(
            {
                "code": "trace_summary_truncated",
                "kind": "coverage",
                "detail": (
                    f"Summarized {len(traces)} of {traces_total} plotted traces; "
                    "analyze_results' signal_stats reads any of the rest."
                ),
            }
        )
    if plan.unverified_units:
        observations.append(
            {
                "code": "trace_unit_unknown",
                "kind": "value",
                "detail": (
                    "The selected descriptor does not establish a unit for "
                    f"{', '.join(plan.unverified_units)}."
                ),
            }
        )
    if facts["downsampled"]:
        observations.append(
            {
                "code": "downsampled",
                "kind": "coverage",
                "detail": (
                    f"At least one series exceeded {max_points} points and was reduced by "
                    "min/max-preserving decimation (spikes preserved; sub-bucket detail not "
                    "shown). Pass a larger max_points or narrow [t_start, t_end] for more. "
                    "The trace summaries are read from every sample."
                ),
            }
        )
    # A reported phase outside (-180, 180] is the unwrapped curve's, and differs
    # from the wrapped angle the waveform recipe's CSV keeps; inside it the two
    # agree, so there is nothing to say.
    unwrapped = [
        trace["signal"]
        for trace in traces
        if any(
            isinstance(phase, float) and not -180.0 < phase <= 180.0
            for phase in (trace.get("phase_initial_deg"), trace.get("phase_final_deg"))
        )
    ]
    if unwrapped:
        observations.append(
            {
                "code": "phase_unwrapped",
                "kind": "value",
                "detail": (
                    f"The phase of {', '.join(dict.fromkeys(unwrapped))} is unwrapped: a "
                    "summary value outside ±180 deg differs by a multiple of 360 from the "
                    "wrapped angle the waveform recipe's CSV keeps."
                ),
            }
        )
    for warn in facts["phase_warnings"]:
        observations.append({"code": "sparse_sweep", "kind": "value", "detail": warn})
    if facts["empty_steps"]:
        observations.append(
            {
                "code": "window_empty_steps",
                "kind": "coverage",
                "detail": (
                    f"{len(facts['empty_steps'])} step(s) had no samples in the window and "
                    f"were omitted: {facts['empty_steps']}."
                ),
            }
        )
    if facts["non_finite"]:
        observations.append(
            {
                "code": "non_finite",
                "kind": "value",
                "detail": (
                    f"{facts['non_finite']} non-finite sample(s) are present; they render as "
                    "gaps in the plot and are left out of the trace summaries."
                ),
            }
        )
    if facts["step_values_available"] is False:
        observations.append(
            {
                "code": "step_value_unavailable",
                "kind": "value",
                "detail": (
                    "Step legend labels left blank: no .step parameter map found in the "
                    "sibling .log."
                ),
            }
        )
    if image_problem is not None:
        observations.append(
            {"code": "image_unavailable", "kind": "coverage", "detail": image_problem}
        )
    if widget_spec_json is None:
        if widget_skipped is not None:
            observations.append(
                {
                    "code": "widget_unavailable",
                    "kind": "coverage",
                    "detail": (
                        widget_skipped + " Wrote the full-fidelity file and opened it "
                        "locally instead."
                    ),
                }
            )
        if open_locally and not opened:
            observations.append(
                {
                    "code": "open_failed",
                    "kind": "coverage",
                    "detail": (
                        "Could not launch a local opener (headless or none found); open the "
                        "returned path manually."
                    ),
                }
            )

    data: dict[str, Any] = {
        "path": str(out_path),
        "analysis_type": analysis_type,
        "plot_index": raw.plot_index,
        "descriptor": asdict(raw.descriptor),
        "signals": [sig.name for sig in cols],
        "n_steps": n_steps,
        "steps_plotted": len(steps_to_plot),
        "panels": facts["panels"],
        "series_count": facts["series_count"],
        "points_per_series": facts["points_per_series"],
        "max_points": max_points,
        "downsampled": facts["downsampled"],
        "window_used": facts["window_used"],
        "x_unit": plan.x_unit,
        "traces": traces,
        "delivery": "ui" if widget_spec_json is not None else "terminal",
        "opened": opened,
        "opener": opener,
        "observations": observations,
    }
    # format.cap_list's convention: a cut list carries its total beside it. The
    # summaries past the cap were never computed, so the total is counted, not
    # sliced from a list.
    if traces_total > len(traces):
        data["traces_truncated"] = traces_total
    if image is not None:
        data["image"] = image.to_dict()
        data["image_path"] = str(png_path)
    if shown_in_ltspice is not None:
        data["ltspice"] = shown_in_ltspice
    if widget_spec_json is not None:
        head = (
            f"Rendered an interactive {analysis_type} plot widget in-chat (also wrote {out_path})"
        )
    else:
        head = f"Wrote interactive {analysis_type} plot to {out_path}"
        if opened:
            head += f" (opened with {opener})"
    lines = [head]
    if shown_in_ltspice is not None:
        lines.append(_ltspice_line(shown_in_ltspice))
    if image is not None:
        lines.append(f"Attached a PNG of the chart ({image.width}x{image.height}, {png_path}).")
    lines.append(f"Traces ({len(traces)} of {traces_total}):")
    lines.extend(_summary_lines(traces, plan.x_unit))
    lines.extend(format_observations(observations))
    result = format_response("\n".join(lines), data, fmt)

    if image is not None:
        result.content.append(image_content(image))
    if widget_spec_json is not None:
        # Pipe the compact chart spec through the result _meta (a non-model-visible
        # channel) as a JSON string. The MCP Apps host forwards the full result to
        # the widget (declared via the tool's ui.resourceUri), where
        # ``app.ontoolresult`` reads _meta and renders it — the model reads the
        # trace summaries and the path, not the chart data.
        result.meta = {WIDGET_SPEC_META_KEY: widget_spec_json}
    return result
