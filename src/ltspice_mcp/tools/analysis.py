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
import concurrent.futures
import contextlib
import contextvars
import csv
import functools
import json
import math
import os
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, NotRequired, TypeVar

import numpy as np
from pydantic import Field, model_validator

from ltspice_mcp.errors import (
    AnalysisDeadlineExceeded,
    NetlistError,
    NoAxisError,
    ResultError,
)
from ltspice_mcp.lib import atomic_write, atomic_write_bytes, desktop, plot_settings, services
from ltspice_mcp.lib.ac_analysis import (
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.ac_structure import AcStructureResult, analyze_ac_structure
from ltspice_mcp.lib.asc_document import is_asc
from ltspice_mcp.lib.decoded_raw import DecodedRaw, PlotDescriptor
from ltspice_mcp.lib.experiment_types import ExperimentCase, ExperimentJob
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.format import cap_note, si_prefix
from ltspice_mcp.lib.ltspice_bridge import BridgeError
from ltspice_mcp.lib.ltspice_window import (
    OpenWindows,
    Shown,
    WindowsUnavailable,
)
from ltspice_mcp.lib.metrics import (
    guarded_axis,
    parse_time,
    signal_stats_defined,
    window_indices,
)
from ltspice_mcp.lib.plot_html import (
    WIDGET_RESOURCE_URI,
    WIDGET_SPEC_META_KEY,
    build_plot_html,
)
from ltspice_mcp.lib.plot_svg import render_plot_svg
from ltspice_mcp.lib.points import label_assignments, point_label, values_text
from ltspice_mcp.lib.raster import RasterSupport, RenderedImage, raster_support, render_image
from ltspice_mcp.lib.raw_parser import (
    OtherRawKind,
    get_step_count,
    safe_magnitude_db,
)
from ltspice_mcp.lib.recipes import WAVEFORM_MAX_POINTS_CAP, StepSelector
from ltspice_mcp.lib.reply_cuts import CUT, ITEMS, NEVER
from ltspice_mcp.lib.result_kinds import AC_SHAPED, kind_named
from ltspice_mcp.lib.sheet_results import (
    NotPlaced,
    Placement,
    check_placement,
    files_beside,
    place_checked,
)
from ltspice_mcp.lib.signal_analysis import (
    TraceStats,
    downsample_minmax,
    summarize_trace,
)
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    LTSPICE_SHOWN_PROPERTIES,
    NEW_WORK_ANNOTATIONS,
    OBSERVATIONS_SCHEMA,
    CaseSelection,
    NotedModel,
    OptionalRawSelectionFields,
    OtherRawFields,
    RunsSelection,
    ToolInput,
    check_runs,
    difference_in_full,
    format_response,
    image_content,
    open_window_call,
    registry,
    safe_path,
    sandboxed,
    sha256_from_open_in_ltspice,
    shown_in_window,
    window_difference,
)
from ltspice_mcp.tools._schema import schema_from_typeddict

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _direct_source(
    path: str | None,
    job_id: str | None,
    state: SessionState,
    *,
    plot_index: int = 0,
    dialect: str | None = None,
) -> services.AnalysisSource:
    """The source a direct call names, refusing an ambiguous or empty pair.

    A user-supplied ``path`` is untrusted input → validated via
    ``safe_path``. The input model refuses an empty path or job_id, so each
    is either a value or absent.
    """
    if (path is None) == (job_id is None):
        # Complete redirect, so no generic hint: the analysis tools read an
        # existing result — a caller holding only a netlist runs it first.
        raise ResultError(
            "Pass exactly one of 'path' or 'job_id'. Analysis tools read "
            "an existing result — if you only have a netlist, run_experiments "
            "produces the job_id/raw to analyze.",
            show_hint=False,
        )
    # Only ``path`` is forwarded: an experiment's runs are case-addressed,
    # so ``_experiment_case`` has already answered every call that named a job.
    return services.resolve_analysis_source(
        state, raw_file=path, plot_index=plot_index, dialect=dialect
    )


async def _experiment_case(
    path: str | None,
    job_id: str | None,
    run_index: int,
    case_id: str | None,
    state: SessionState,
    *,
    other_raw: OtherRawKind | None = None,
) -> tuple[ExperimentJob, ExperimentCase, services.RunContext] | None:
    """The job a ``job_id`` names, its case, and what a read of the case takes,
    when it is a run_experiments job, else ``None``.

    Experiment runs are addressed by case, never through ``resolve_run`` — the
    same split ``Api.load_raw`` makes. ``run_index``, ``case_id`` and
    ``other_raw`` only mean something here, so beside a path they are refused
    as every reader of a path refuses them
    (``services.run_selectors_beside_path``). A path together with job_id, or
    neither, is left to ``_direct_source``'s exclusivity error.
    """
    job = (
        await services.resolve_job_async(job_id, state)
        if job_id is not None and path is None
        else None
    )
    if job is not None:
        case, context = await services.readable_case(
            job, state, run_index=run_index, case_id=case_id, other_raw=other_raw
        )
        return job, case, context
    if (
        path is not None
        and job_id is None
        and (
            refusal := services.run_selectors_beside_path(
                run_index=run_index, case_id=case_id, other_raw=other_raw
            )
        )
    ):
        # An argument mistake: the family's hint (check the run's state) does
        # not apply, as for _direct_source's refusal of path with job_id.
        raise ResultError(refusal, show_hint=False)
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
    if (analysis_type in AC_SHAPED or analysis_type == "noise") and axis.quantity == "frequency":
        base = "freq"
    return f"{base}_{axis.unit}" if axis.unit else base


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
                        name, wave_w, complex_format if analysis_type in AC_SHAPED else "re_im"
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
                label = values_text(step_dicts[step]) if step < len(step_dicts) else ""
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
# Global backstop on a rendered panel. ``max_points`` caps each series, but the
# chart aligns a panel's series onto one x (uPlot.join), and a .step run whose
# steps have distinct axes multiplies series-count by the union length — so a
# many-step distinct-axis run could blow far past the per-series cap in the
# browser. Refuse with guidance before building it (no silent truncation).
_PLOT_MAX_CELLS = 10_000_000

# Most per-trace summaries one reply carries. A stepped or Monte Carlo overlay
# can plot hundreds of traces; past this many the reply lists the first ones,
# says how many there were, and names the call that reads the rest.
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
    units_read: tuple[dict[str, str | None], list[str]] | None = None,
) -> PlotPlan:
    """Resolve what a plot draws. ``groups`` are the requested signals, one list
    per panel; with ``split_by_unit`` they are regrouped into one panel per unit.
    ``raw`` names the axis; ``units_read`` gives each signal's unit and those not
    established, read from ``raw`` when not given (a chart of several cases
    reads them across the cases, ``_units_across``).

    Uses resident descriptors without reopening a source deck or log.
    """
    cols = [sig for group in groups for sig in group]
    units, unverified = _trace_units(raw, cols) if units_read is None else units_read
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
    raw_path: Path,
    analysis_type: str,
    job_id: str | None,
    run_index: int,
    *,
    plot_index: int = 0,
    cases: int | None = None,
) -> str:
    """The chart's file name: the results file's stem, the analysis, the run
    of a job (``run2``) or the number of its cases drawn (``cases5``), the
    plot and the time."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    run = f"_cases{cases}" if cases is not None else f"_run{run_index}" if job_id else ""
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
    case_id: NotRequired[str]
    run_index: NotRequired[int]
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


@dataclass(frozen=True)
class PlotCase:
    """One case's results a plot draws: its raw, the steps of it drawn, and
    what its traces' labels and summaries name of its point.

    A plot of one case's results (a path, or one case of a job) draws one,
    with no ``case_id``: its traces are labelled by step alone, as they always
    were. A plot of several cases of a job draws one each.
    """

    raw: Any
    steps: list[int]
    step_dicts: list[dict[str, Any]]
    #: The results are a .step sweep, so each summary names its step even
    #: when one step is drawn.
    stepped: bool
    #: The plan's signals as this raw resolves them, by the name the plan
    #: gives them; one it lacks is absent. None: the plan's own signals.
    signals: Mapping[str, services.Signal] | None = None
    #: With several cases drawn: the case, which each trace summary names.
    case_id: str | None = None
    run_index: int | None = None
    #: What its labels name of the case: the assignments that tell it from
    #: the other cases drawn, else (None) its case id.
    label_assignments: Mapping[str, Any] | None = None

    @classmethod
    def of_plan(cls, raw: Any, plan: PlotPlan) -> "PlotCase":
        """The one case a plan of one case's results draws."""
        return cls(raw, plan.steps, plan.step_dicts, plan.stepped)

    def signal(self, sig: services.Signal) -> services.Signal | None:
        """``sig`` as this case's raw holds it, or None when it holds none."""
        return sig if self.signals is None else self.signals.get(sig.name)

    def label(self, signal: str, step: int) -> str:
        """The legend label of ``signal`` at ``step`` of this case: its step
        where it draws several, and, in a chart of several cases, wherever its
        results are stepped, so each trace says which case and which step it is."""
        several_steps = len(self.steps) > 1 or (self.case_id is not None and self.stepped)
        values = self.step_dicts[step] if several_steps and step < len(self.step_dicts) else None
        return point_label(
            signal,
            case_id=(
                self.case_id
                if self.case_id is not None and self.label_assignments is None
                else None
            ),
            assignments=self.label_assignments,
            step_values=values,
            step_index=step if several_steps else None,
        )


def _by_response(
    plan: PlotPlan, is_complex: Callable[[services.Signal], bool]
) -> list[list[services.Signal]]:
    """``plan``'s panel groups for an AC plot, each parted into the signals
    that hold a complex response, drawn as a magnitude and a phase, and those
    that hold real values, drawn as they are: a device parameter ngspice saves
    in an AC run, which its ASCII raw file holds as real numbers
    (``raw_header._Scanner.text_payload``)."""
    parted: list[list[services.Signal]] = []
    for group in plan.groups:
        kinds = [is_complex(sig) for sig in group]
        for response in (True, False):
            part = [sig for sig, kind in zip(group, kinds, strict=True) if kind == response]
            if part:
                parted.append(part)
    return parted


def extract_plot(raw, plan: PlotPlan, *, summary_limit: int = 0) -> PlotData:
    """Read, window and measure every trace ``plan`` draws of one case's
    results (no I/O but the raw); :func:`extract_cases` for several."""
    return extract_cases([PlotCase.of_plan(raw, plan)], plan, summary_limit=summary_limit)


def extract_cases(
    cases: Sequence[PlotCase], plan: PlotPlan, *, summary_limit: int = 0
) -> PlotData:
    """Read, window and measure every trace ``plan`` draws of each of
    ``cases`` (no I/O but the raws).

    Runs in a worker thread (heavy numpy). Each step's axis is fetched and
    windowed once, however many signals are drawn over it. A signal's traces
    are drawn case by case, each case's step by step. The first
    ``summary_limit`` traces are summarized (:func:`summarize_trace`) from every
    sample in the window; ``facts["traces_total"]`` counts them all, across
    cases and steps.
    """
    is_ac = plan.analysis_type in AC_SHAPED
    several = len(cases) > 1
    multi = several or any(len(case.steps) > 1 for case in cases)
    #: The cases are cases of a job drawn by ``runs``, whose traces name them.
    named = any(case.case_id is not None for case in cases)

    windows: list[dict[int, tuple[np.ndarray, int, int]]] = []
    empty_steps: list[int | str] = []
    empty_cases: list[str] = []
    for case in cases:
        held: dict[int, tuple[np.ndarray, int, int]] = {}
        outside: list[int] = []
        for step in case.steps:
            axis = guarded_axis(case.raw, step)
            lo, hi = window_indices(axis, plan.ts, plan.te)
            if lo < hi:
                held[step] = (axis[lo:hi], lo, hi)
            else:
                # This step's axis does not intersect the window (a step may end
                # earlier than its siblings). Skip it; surfaced as a fact.
                outside.append(step)
        if case.case_id is None:
            empty_steps.extend(outside)
        elif not held:
            # A case of several the window leaves nothing of is not drawn.
            empty_cases.append(case.case_id)
        else:
            empty_steps.extend(f"{case.case_id} step {step}" for step in outside)
        windows.append(held)
    drawn_steps = sum(len(held) for held in windows)
    if not drawn_steps:
        raise ResultError(
            "The [t_start, t_end] window selects no samples"
            + (
                f" in any of the {len(cases)} cases."
                if several
                else f" in any of the {len(plan.steps)} steps."
                if multi
                else "."
            )
        )

    def first_wave(sig: services.Signal) -> np.ndarray:
        """``sig`` at the first windowed step of the first case that holds it."""
        for case, held in zip(cases, windows, strict=True):
            own = case.signal(sig)
            if own is not None and held:
                return own.wave(case.raw, next(iter(held)))
        raise ResultError(f"No case drawn holds {sig.name!r}.")

    # Single-trace annotation reads the full-resolution complex response.
    annotate_one = is_ac and plan.annotate and len(plan.signals) == 1 and drawn_steps == 1
    annotations: list[dict] | None = None
    nmp: bool | None = None
    non_finite = 0
    phase_warnings: list[str] = []
    summaries: list[TraceSummary] = []
    summarized_cases: set[str] = set()
    total = 0
    lacking: dict[str, list[str]] = {}
    groups: list[tuple[list[str], list[_Trace]]] = []
    drawn = (
        _by_response(plan, lambda sig: np.iscomplexobj(first_wave(sig))) if is_ac else plan.groups
    )
    suffix_titles = len(drawn) > 1
    for group in drawn:
        title = _group_title(group, plan.units)
        # A trace of an AC plot that holds real values is drawn as its values.
        response = is_ac and np.iscomplexobj(first_wave(group[0]))
        if response:
            suffix = f" — {title}" if suffix_titles else ""
            rows = [f"Magnitude (dB){suffix}", f"Phase (deg){suffix}"]
        else:
            rows = [title]
        panel = sum(len(r) for r, _ in groups)
        traces: list[_Trace] = []
        for col in group:
            for case, held in zip(cases, windows, strict=True):
                own = case.signal(col)
                if own is None:
                    if case.case_id is not None:
                        lacking.setdefault(col.name, []).append(case.case_id)
                    continue
                for step, (axis_w, lo, hi) in held.items():
                    wave = own.wave(case.raw, step)[lo:hi]
                    if response:
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
                                _Trace(f"{case.label(col.name, step)} (imag)", x, (np.imag(wave),))
                            )
                            non_finite += int(np.count_nonzero(~np.isfinite(np.imag(wave))))
                    non_finite += sum(int(np.count_nonzero(~np.isfinite(y))) for y in ys)
                    label = case.label(col.name, step)
                    if not is_ac and np.iscomplexobj(wave):
                        label += " (real)"
                    if total < summary_limit:
                        summaries.append(
                            _trace_summary(plan, case, col, step, label, panel, x, ys)
                        )
                        if case.case_id is not None:
                            summarized_cases.add(case.case_id)
                    total += 1
                    traces.append(_Trace(label, x, ys))
        groups.append((rows, traces))

    everything = [t for _, traces in groups for t in traces]
    facts: dict[str, Any] = {
        "empty_steps": empty_steps,
        "non_finite": non_finite,
        "phase_warnings": phase_warnings,
        "window_used": [
            min(float(t.x[0]) for t in everything),
            max(float(t.x[-1]) for t in everything),
        ],
        "step_values_available": (
            any(bool(values) for values in plan.step_dicts) if multi and not named else None
        ),
        "traces": summaries,
        "traces_total": total,
    }
    if named:
        # Each stepped case's labels name its steps (PlotCase.label), by
        # their values where its .log gave them.
        facts["cases_without_step_values"] = [
            case.case_id
            for case in cases
            if (len(case.steps) > 1 or case.stepped)
            and not any(bool(values) for values in case.step_dicts)
        ]
        facts["cases_window_empty"] = empty_cases
        facts["signals_lacking"] = lacking
        facts["cases_unsummarized"] = sum(
            1
            for case, held in zip(cases, windows, strict=True)
            if held and case.case_id not in summarized_cases
        )
    return PlotData(plan, groups, annotations, nmp, facts)


def _trace_summary(
    plan: PlotPlan,
    case: PlotCase,
    col: services.Signal,
    step: int,
    label: str,
    panel: int,
    x: np.ndarray,
    ys: tuple[np.ndarray, ...],
) -> TraceSummary:
    """One trace's summary. An AC trace is summarized in dB, with its unwrapped
    phase at both ends; only a transient reports a (time-weighted) mean. With
    several cases drawn it names its case by the fields an analyze_results
    row names it with (``points.RowIdentity``)."""
    is_ac = len(ys) == 2
    entry = TraceSummary(
        signal=col.name,
        panel=panel,
        unit="dB" if is_ac else plan.units[col.name],
        **summarize_trace(x, ys[0], time_weighted_mean=plan.analysis_type == "transient"),
    )
    if case.case_id is not None and case.run_index is not None:
        entry["case_id"] = case.case_id
        entry["run_index"] = case.run_index
    if case.stepped:
        entry["step"] = step
    # One case's summary carries its label only where it names a step.
    if (case.stepped or case.case_id is not None) and label != col.name:
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
    x_scale = "log" if plan.analysis_type in AC_SHAPED or plan.x_is_log else "linear"
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
    plot: PlotData, name: str, max_points: int, out_path: Path, title: str
) -> dict:
    """Assemble the offline HTML at ``max_points`` and write it atomically; return
    the projection's facts. ``name`` names what is drawn in the page's summary
    line (a results file's stem, or a job). Runs in a worker thread (HTML
    build + file I/O)."""
    spec, facts = plot_spec(plot, max_points)
    summary = f"{name} — {plot.plan.analysis_type}: {facts['series_count']} series"
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
    return {**plot.facts, **write_plot_file(plot, raw_path.stem, max_points, out_path, title)}


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


class PlotWaveformInput(OptionalRawSelectionFields, OtherRawFields, ToolInput, NotedModel):
    path: str | None = Field(
        default=None,
        min_length=1,
        description="Path to a .raw result file. Pass this or job_id (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Plot one run of a finished job instead of a path; pick "
            "the run with run_index or case_id, or several with runs."
        ),
    )
    run_index: int = Field(
        default=0,
        strict=True,
        ge=0,
        description="0-based run to read when job_id is given (default 0).",
    )
    case_id: str | None = Field(
        default=None,
        description=(
            "For a run_experiments job_id: the case to plot, as named in its receipt "
            "and analysis identities (an alternative to run_index)."
        ),
    )
    runs: RunsSelection | None = Field(
        default=None,
        description=(
            "With job_id: cases to draw in one chart, as an analyze_results source "
            "takes them. Not with run_index or case_id."
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
    step: int | StepSelector | None = Field(
        default=None,
        description=(
            "For a .step run: omit to overlay every step, or plot one by its 0-based "
            "index or its axis value ({axis:'temp', value:27}), which a step must have."
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
            f"(default {_DEFAULT_PLOT_MAX_POINTS}, held to {WAVEFORM_MAX_POINTS_CAP}). "
            "Full fidelity below it; spikes survive."
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
            "Also open the run in the user's LTspice window (26.1+), started if "
            "none is open, with these traces drawn; writes a .plt beside the results file. A sheet's "
            "run is opened from the sheet, so a click on a net plots it: its "
            "results replace those beside the sheet, reopening what is open."
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
            "Directory for the HTML (created if needed). "
            "Default: the server's store, '.ltspice-mcp/plots/' in the working "
            "directory unless LTSPICE_MCP_STORE_DIR moves it."
        ),
    )

    @model_validator(mode="after")
    def _hold_max_points(self) -> "PlotWaveformInput":
        self.hold_to_cap("max_points", WAVEFORM_MAX_POINTS_CAP)
        return self

    @model_validator(mode="after")
    def _one_way_to_name_cases(self) -> "PlotWaveformInput":
        if self.runs is None:
            return self
        if self.job_id is None:
            raise ValueError("runs selects cases of a job: pass it with job_id, not a path")
        named = [
            name
            for name, given in (
                ("run_index", "run_index" in self.model_fields_set),
                ("case_id", self.case_id is not None),
            )
            if given
        ]
        if named:
            raise ValueError(
                f"runs names the cases to draw, and so does {' and '.join(named)}: "
                "pass one of them"
            )
        check_runs(self)
        return self


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


_TIED_NOTE = "The plot is tied to the sheet: a click on a net there plots it."
_ALONE_NOTE = (
    "The plot is not tied to the sheet, so clicking a net there does not plot it, "
    "because LTspice opened the sheet before it had results and has unsaved changes "
    "to it. Ask the user whether to keep those changes, then call "
    + open_window_call("this sheet", sha256_from_open_in_ltspice("this sheet"), "'save'|'discard'")
    + ". After 'discard', showing this run again ties the plot; after 'save', the "
    "sheet holds their changes, so run it again and show that run."
)
_BY_HAND_NOTE = (
    "The results are beside the sheet: View > Visible Traces on it in LTspice opens "
    "them, and a click on a net then plots it."
)
_REPLACE_NOTE = (
    "saved_panes lists them as the set_plot_panes op of edit_schematic takes panes. To "
    "draw others, replace them with that op on the sheet; the axis ranges saved with "
    "them go too."
)
_REOPENED_NOTE = (
    "LTspice already had a plot of these results open; it was closed and reopened to "
    "show the current results, and traces added to it by hand were lost."
)
_CLOSED_FOR_NOTHING_NOTE = (
    "LTspice had a plot of the results beside {sheet} open, which was closed to make "
    "way for these before their placement was refused: traces added to it by hand "
    "were lost, and opened from the sheet again it shows the results still there."
)
_SHEET_REOPENED_NOTE = (
    "LTspice opened the sheet before it had results and only looks for results when "
    "it opens a sheet, so the sheet was closed and reopened."
)


_T = TypeVar("_T")

#: The threads that talk to LTspice's window for ``plot_waveform``. Not the
#: default pool: a placement from one of them (``_JobRun.place``) waits on the
#: loop for work that runs on the default pool, and a pool whose every thread
#: waited so would never run it. A call past this many waits for a thread.
_WINDOW_THREADS = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="ltspice-window"
)


async def _in_window_thread(work: Callable[..., _T], *args: Any) -> _T:
    """``work(*args)`` on one of ``_WINDOW_THREADS``, awaited."""
    loop = asyncio.get_running_loop()
    # As ``asyncio.to_thread`` does, so the work sees this request's context.
    context = contextvars.copy_context()
    return await loop.run_in_executor(_WINDOW_THREADS, functools.partial(context.run, work, *args))


@dataclass(frozen=True)
class _JobRun:
    """The job and case whose results a plot shows, and the loop that
    placement beside the case's circuit runs on (``sheet_results.place_checked``)."""

    job: ExperimentJob
    case: ExperimentCase
    loop: asyncio.AbstractEventLoop

    def place(self, sheet: Path) -> Placement:
        """Put the case's results beside ``sheet`` by the one checked
        placement, from a worker thread. Blocking; raises ``NotPlaced``."""
        placing = place_checked(self.job, self.case, sheet)
        try:
            future = asyncio.run_coroutine_threadsafe(placing, self.loop)
        except RuntimeError:  # the loop is closed: the server is stopping
            placing.close()
            raise
        # Never on the default pool's threads, which the placement needs
        # (``_WINDOW_THREADS``), so this wait cannot starve it.
        return future.result()


def _show_in_ltspice(
    state: SessionState,
    results: Path,
    run: _JobRun | None,
    plot_name: str,
    panes: list[list[str]],
) -> dict[str, Any]:
    """Open a run in an LTspice window with ``panes`` drawn, and say what happened. Blocking.

    A job's run of a sheet is opened from the sheet, which ties the plot to
    it, once its results may stand beside the sheet as its own: they are put
    there by the placement every caller takes (``sheet_results.place_checked``).
    Results named by path are opened from the sheet whose results file they
    already are. Any other run, and one whose placement is refused, is opened
    on its own, and the reply says why it was not placed (``not_placed``).
    Nothing here fails the plot: the chart and its numbers are already made,
    and what happened in LTspice is a fact beside them.
    """
    report: dict[str, Any] = {
        "shown": False,
        "results": str(results),
        "plot_settings": None,
        "panes": panes,
    }
    windows = state.open_windows
    if run is None:
        sheet = _sheet_of_results(results, state)
        if sheet is None:
            _show_alone(windows, report, results, plot_name, panes)
        else:
            _show_from_sheet(windows, report, sheet, results, None, plot_name, panes)
        return report
    sheet = _sheet_to_tie(run.case.circuit_path, state)
    if sheet is None:
        _show_alone(windows, report, results, plot_name, panes)
        return report
    try:
        # Asked before the window is: a refusal leaves the window as it was.
        # The placement asks again under the locks.
        check_placement(run.job, run.case, sheet)
    except NotPlaced as refusal:
        _say_not_placed(report, refusal.code, str(refusal))
        _show_alone(windows, report, results, plot_name, panes)
        return report
    _show_from_sheet(windows, report, sheet, results, run, plot_name, panes)
    return report


def _sheet_to_tie(circuit: Path, state: SessionState) -> Path | None:
    """The sheet a plot of a case that ran ``circuit`` can be tied to in
    LTspice, or None: the circuit, when it is a sheet the sandbox admits,
    since its results and plot settings are written beside it. Whether it is
    still there is the placement's to say (``sheet_missing``)."""
    if not is_asc(circuit):
        return None
    return sandboxed(str(circuit), state)


def _sheet_of_results(results: Path, state: SessionState) -> Path | None:
    """The sheet whose own results file ``results``, named by path, is, or
    None. Nothing is copied for it: the results are already where LTspice
    keeps that sheet's."""
    sheet = _sheet_to_tie(results.with_suffix(".asc"), state)
    if sheet is None or not sheet.is_file():
        return None
    beside = files_beside(sheet).raw
    try:
        return sheet if beside.is_file() and os.path.samefile(beside, results) else None
    except OSError:
        return None


def _say_not_placed(report: dict[str, Any], code: str, message: str) -> None:
    """Why a run's results were not put beside its sheet: the placement's
    refusal, with its code and the fix it names."""
    report["not_placed"] = {"code": code, "message": message}


def _write_panes(
    report: dict[str, Any],
    results: Path,
    plot_name: str,
    panes: list[list[str]],
    *,
    of_a_sheet: bool = False,
) -> str | None:
    """Write the plot settings that name ``panes`` beside ``results``.

    Returns None once written, with the file named in ``report``, and
    otherwise what the reply's note says of it: why they were not. Where the
    file there was saved from LTspice and is left as it is, ``report`` gets
    the panes it holds (``saved_panes``), since those are what the window
    draws, and for a sheet's the note names the edit that replaces them. A
    sheet's plot settings are also the ``set_plot_panes`` op's to write, so
    they are written under the lock an edit takes for them, held for this
    write alone.
    """
    if not panes:
        # Asked for only by a plot with no sampled axis to draw traces against.
        return (
            "No traces are named for it: how LTspice reads the plot settings of a "
            "plot with no sampled axis is not recorded."
        )
    target = plot_settings.plot_settings_path(results)
    try:
        with file_lock(Store.circuit_lock(target)) if of_a_sheet else contextlib.nullcontext():
            kept = plot_settings.write_beside(results, plot_name, panes)
    except (NetlistError, OSError) as error:
        return f"The plot settings could not be written ({error})."
    if kept is None:
        report["plot_settings"] = str(target)
        return None
    if kept.saved is None:
        return f"{kept.reason}."
    report["saved_panes"] = plot_settings.pane_specs(kept.saved)
    return f"{kept.reason}. {_REPLACE_NOTE}" if of_a_sheet else f"{kept.reason}."


def _show_alone(
    windows: OpenWindows,
    report: dict[str, Any],
    results: Path,
    plot_name: str,
    panes: list[list[str]],
) -> None:
    """Open ``results`` in an LTspice window on its own, and fill in ``report``.

    The traces are written first, as the plot settings file LTspice loads when
    it opens the results, so they are there for a person who opens the file by
    hand when no window could be reached.
    """
    try:
        windows.check()
    except WindowsUnavailable as error:
        report["reason"] = str(error)
        return
    left_alone = _write_panes(report, results, plot_name, panes)
    try:
        shown = windows.show_results(results)
    except BridgeError as error:
        report["reason"] = str(error)
        if left_alone is None:
            report["note"] = (
                "Opened by hand in LTspice, the results file shows these traces: the "
                "plot settings beside it name them."
            )
        return
    _say_shown(report, shown, left_alone)


def _say_shown(report: dict[str, Any], shown: Shown, *said: str | None) -> None:
    """Fill in ``report`` for results that were shown: the window, what was
    closed and opened again to show them, and a note of ``said`` and of that."""
    report.update(shown_in_window(shown))
    if shown.reopened:
        report["reopened"] = True
    if shown.sheet_reopened:
        report["sheet_reopened"] = True
    note = " ".join(
        filter(
            None,
            (
                *said,
                _REOPENED_NOTE if shown.reopened else None,
                _SHEET_REOPENED_NOTE if shown.sheet_reopened else None,
            ),
        )
    )
    if note:
        report["note"] = note


def _show_from_sheet(
    windows: OpenWindows,
    report: dict[str, Any],
    sheet: Path,
    results: Path,
    run: _JobRun | None,
    plot_name: str,
    panes: list[list[str]],
) -> None:
    """Open a run in an LTspice window from ``sheet``, tied to it, and fill in ``report``.

    A job's results (``run``) are put beside the sheet under its name by the
    checked placement, with the plot settings that name ``panes``, and the
    sheet's own command opens them (``OpenWindows.results_from_sheet``);
    results named by path (no ``run``) are already there. Nothing is written
    until a window is known that can show them, and until then the reply
    names the run's own results and no sheet. A placement refused then (the
    sheet changed since it was checked) leaves the sheet's results as they
    were, and the run is opened on its own, as a refused one is. ``tied`` in
    the reply says whether the plot is tied to the sheet: it is not where the
    window holds a sheet with changes that are not in the file and knows of
    no results for it, and the results beside the sheet are then opened on
    their own.
    """
    beside = files_beside(sheet).raw
    left_alone: str | None = None

    closed = False

    def place(reopened: bool) -> None:
        nonlocal left_alone, closed
        closed = reopened
        if run is not None:
            placement = run.place(sheet)
            report.update(not_checked=placement.not_checked, left_in_place=placement.left_in_place)
        report.update(results=str(beside), sheet=str(sheet))
        left_alone = _write_panes(report, beside, plot_name, panes, of_a_sheet=True)

    try:
        shown = windows.results_from_sheet(sheet, place)
    except BridgeError as error:
        report["reason"] = str(error)
        # The reply names the sheet once the results are beside it.
        if "sheet" in report:
            report["note"] = _BY_HAND_NOTE
        return
    except (NotPlaced, NetlistError) as refusal:
        # Refused under the locks (the sheet changed since the check before
        # the window was asked), or another server process held the sheet or
        # its results file past the wait, as the run option reports it.
        code = refusal.code if isinstance(refusal, NotPlaced) else "sheet_locked"
        _say_not_placed(report, code, str(refusal))
        _show_alone(windows, report, results, plot_name, panes)
        if closed:
            lost = _CLOSED_FOR_NOTHING_NOTE.format(sheet=sheet.name)
            report["note"] = f"{report['note']} {lost}" if report.get("note") else lost
        return
    except OSError as error:
        report["reason"] = f"the run's results could not be put beside {sheet.name} ({error})"
        return
    report["tied"] = shown.tied
    _say_shown(report, shown, _TIED_NOTE if shown.tied else _ALONE_NOTE, left_alone)
    if shown.held is not None:
        report.update(window_difference(sheet.read_bytes(), shown.held))
    if report.get("differs_from_file"):
        # ``difference`` names the entries in part; the note names the call for all of them.
        whole = difference_in_full(sheet="this sheet", window="this pid")
        report["note"] = f"{report['note']} {whole}" if report.get("note") else whole


def _why_no_axis(descriptor: PlotDescriptor) -> str:
    """What a plot with no sampled axis holds instead, for a reply to say."""
    kind = kind_named(descriptor.stepped_kind)
    if kind is not None and kind.stepping == "point_a_step":
        return (
            f"it holds one point a step, the stepped {', '.join(descriptor.step_variables)} first"
        )
    if kind is not None:
        return "it holds one point, and nothing in it says which step that is"
    return f"its {descriptor.analysis} results are one table of values"


async def _show_without_chart(
    args: PlotWaveformInput,
    state: SessionState,
    source: services.AnalysisSource,
    raw: DecodedRaw,
    placing: _JobRun | None,
):
    """Open a plot with no sampled axis in LTspice, with no chart of it here.

    The chart needs an axis to draw against; LTspice's window takes the run as
    it is, from its sheet where it has one, as any other run is shown. No
    traces are named for it (``_write_panes``), and what the window draws of
    it is not recorded, which the reply says.
    """
    assert source.raw is not None
    shown = await _in_window_thread(
        _show_in_ltspice,
        state,
        source.raw,
        placing,
        raw.descriptor.original_plot_name,
        [],
    )
    observations = list(await asyncio.to_thread(services.read_observations, source, raw))
    observations.append(
        {
            "code": "no_chart",
            "kind": "coverage",
            "detail": (
                f"No chart was drawn: this plot has no sampled axis "
                f"({_why_no_axis(raw.descriptor)}). What LTspice's window draws of it is "
                "not recorded."
            ),
        }
    )
    data: dict[str, Any] = {
        "analysis_type": raw.descriptor.analysis,
        "plot_index": raw.plot_index,
        "descriptor": asdict(raw.descriptor),
        "ltspice": shown,
        "observations": observations,
    }
    return format_response(data)


def _rest_of_summary(
    args: PlotWaveformInput,
    analysis: str,
    *,
    n_steps: int,
    run_index: int,
) -> str:
    """The call that reads the traces a summary cut at ``_TRACE_SUMMARY_MAX``
    leaves out, as the reply's ``hint`` names it, reading what this call
    plotted: the same run, window and steps.

    analyze_results' signal_stats recipe reads them where it defines
    statistics for the plot over its window (``signal_stats_defined``), else
    its waveform recipe returns their samples, as it does for any plot with a
    sampled axis. An analyze_results source takes this tool's path, job_id,
    plot_index, other_raw and dialect, but picks a job's run with ``runs``.
    """
    windowed = args.t_start is not None or args.t_end is not None
    if signal_stats_defined(analysis, windowed=windowed):
        recipes = "a signal_stats recipe for each signal, at most 256 a call"
    else:
        recipes = "waveform recipes naming their signals, at most 32 a recipe; they return samples"
    if windowed:
        window = {
            edge: value
            for edge, value in (("start", args.t_start), ("end", args.t_end))
            if value is not None
        }
        recipes += f"; each with window={json.dumps(window)}"
    source = (
        "the path or job_id this call named, with its plot_index, other_raw and dialect if any"
    )
    if args.job_id is not None:
        # The runs this call named, as it named them: a chart of several
        # cases passes its selection on whole, never a list of every case.
        runs = (
            _runs_as_given(args.runs)
            if args.runs is not None
            else {"case_ids": [args.case_id]}
            if args.case_id is not None
            else [run_index]
        )
        source += f", and runs={json.dumps(runs)}"
    said = [
        "analyze_results reads the traces this summary leaves out.",
        f"Source: {source}.",
        f"Recipes: {recipes}.",
    ]
    if n_steps > 1:
        if isinstance(args.step, StepSelector):
            said.append(f"Steps: step={json.dumps(args.step.model_dump(mode='json'))}.")
        elif args.step is not None:
            said.append(
                f"Steps: all_steps=true; this plot's step is the rows with step_index {args.step}."
            )
        else:
            said.append("Steps: all_steps=true, one row per step.")
    return " ".join(said)


def _signal_groups(
    args: PlotWaveformInput,
    resolve: Callable[[str], services.Signal],
    is_axis: Callable[[str], bool],
    every: Callable[[], list[str]],
) -> list[list[services.Signal]]:
    """The signals a call asks to draw, one list per panel: ``panels``, else
    ``signals``, else (``'all'``) every trace ``every`` lists. ``resolve``
    reads a requested name against the results drawn, and ``is_axis`` says
    whether a resolved name is their sweep axis."""
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")
    # A signal list is one panel's worth; resolved once, whatever the spelling.
    requested = (
        args.panels
        if args.panels is not None
        else None
        if args.signals == "all"
        else [args.signals]
    )
    groups: list[list[services.Signal]]
    panel_of: dict[str, int] = {}
    if requested is None:
        groups = [[services.Signal(name, name) for name in every()]]
    else:
        groups = []
        for index, names in enumerate(requested):
            group: list[services.Signal] = []
            for name in names:
                sig = resolve(name)
                if is_axis(sig.name):
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
    if args.panels is not None and args.signals != "all":
        # The panels name what is plotted; a signal list beside them says it
        # again, and is refused only where it names something they do not.
        unplotted = [name for name in args.signals if resolve(name).name not in panel_of]
        if unplotted:
            raise ResultError(
                f"signals names {', '.join(map(repr, unplotted))}, which no panel "
                "plots: panels alone names what is plotted, so add it to a panel or "
                "leave signals out."
            )
    if not any(groups):
        raise ResultError("No signal traces to plot (the result has only an axis).")
    return groups


def _steps_named(args: PlotWaveformInput, raw: Any, step_dicts: list[dict[str, Any]]) -> list[int]:
    """The steps of ``raw`` that ``step`` names: the one it names, by index or
    by value, or every step. Raises ``ResultError`` for a step ``raw`` lacks."""
    if isinstance(args.step, StepSelector):
        return [args.step.pick(step_dicts)]
    if args.step is not None:
        services.validate_step(raw, args.step)
        return [args.step]
    return list(range(get_step_count(raw)))


@registry.tool(
    name="plot_waveform",
    title="Plot Waveforms",
    description=(
        "Render an interactive chart (zoom/pan/hover) of one or more signals for "
        "a person to look at. The chart type follows the run (transient, DC sweep, "
        "AC Bode, noise), a .step or Monte Carlo run overlays every step, and each "
        "unit gets its own panel; runs draws several cases of a job in one chart.\n\n"
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
            "cases": {
                "type": "array",
                CUT: ITEMS,
                "description": (
                    "With runs naming several cases: each case drawn, with its "
                    "assignments and its number of steps."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "run_index": {"type": "integer"},
                        "assignments": {"type": "object"},
                        "n_steps": {"type": "integer"},
                    },
                    "required": ["case_id", "run_index", "n_steps"],
                },
            },
            "cases_not_drawn": {
                "type": "array",
                # Each names a case the chart leaves out; leaving one out of
                # this list would pass it off as drawn.
                CUT: NEVER,
                "description": (
                    "With runs: each case selected and not drawn, with why: "
                    "case_not_found, run_not_found, raw_not_produced (with its status), "
                    "run_not_finished, other_raw_not_written, result_unreadable or "
                    "analysis_deadline for a results file that cannot be parsed, "
                    "step_not_found for a case without the step asked for, or "
                    "window_empty."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "case_id": {"type": "string"},
                        "run_index": {"type": "integer"},
                        "status": {"type": "string"},
                        "code": {"type": "string"},
                        "detail": {"type": "string"},
                    },
                    "required": ["code"],
                },
            },
            "signals_not_in_cases": {
                "type": "array",
                CUT: NEVER,
                "description": (
                    "With runs: each signal drawn that some cases do not hold, and "
                    "those cases; it is drawn for the others."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "signal": {"type": "string"},
                        "case_ids": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
                    },
                    "required": ["signal", "case_ids"],
                },
            },
            "signals": {"type": "array", "items": {"type": "string"}, CUT: ITEMS},
            "n_steps": {"type": "integer"},
            "steps_plotted": {"type": "integer"},
            "panels": {"type": "integer"},
            "series_count": {"type": "integer"},
            "points_per_series": {
                "type": "array",
                "items": {"type": "integer"},
                CUT: ITEMS,
            },
            "max_points": {"type": "integer"},
            "downsampled": {"type": "boolean"},
            "window_used": {"type": "array", "items": {"type": "number"}, CUT: NEVER},
            "x_unit": {"type": ["string", "null"]},
            "traces": {
                "type": "array",
                "items": schema_from_typeddict(TraceSummary),
                CUT: ITEMS,
            },
            "traces_truncated": {
                "type": "integer",
                "description": (
                    "Present when traces summarizes only the first plotted traces: "
                    "how many were plotted. hint names the call that reads the rest."
                ),
            },
            "cases_unsummarized": {
                "type": "integer",
                "description": (
                    "With traces_truncated and several cases: how many cases drawn "
                    "have no trace in traces."
                ),
            },
            "delivery": {"type": "string", "enum": ["terminal", "ui"]},
            "opened": {"type": "boolean"},
            "opener": {"type": ["string", "null"]},
            "ltspice": {
                "type": "object",
                "description": "Present with in_ltspice: what happened in the LTspice window.",
                "properties": {
                    **LTSPICE_SHOWN_PROPERTIES,
                    "results": {"type": "string", "description": "The file opened."},
                    "reopened": {
                        "type": "boolean",
                        "description": (
                            "LTspice already had a plot of these results open; it was "
                            "closed and reopened, and traces added to it by hand were lost."
                        ),
                    },
                    "sheet_reopened": {
                        "type": "boolean",
                        "description": (
                            "LTspice opened the sheet before it had results, so the "
                            "sheet was closed and reopened to tie the plot to it."
                        ),
                    },
                    "sheet": {
                        "type": "string",
                        "description": (
                            "The sheet the results were put beside, when the run is a sheet's."
                        ),
                    },
                    "tied": {
                        "type": "boolean",
                        "description": (
                            "With sheet: whether the plot is tied to the sheet, so "
                            "clicking a net there plots it. False when the sheet has "
                            "unsaved changes in LTspice and could not be reopened; "
                            "note says more."
                        ),
                    },
                    "differs_from_file": {
                        "type": "boolean",
                        "description": (
                            "The window already had the sheet open and its copy is "
                            "not the file the run was made from."
                        ),
                    },
                    "difference": {"type": "string"},
                    "plot_settings": {
                        "type": ["string", "null"],
                        "description": "The .plt written beside it; absent when none was.",
                    },
                    "saved_panes": {
                        "type": "array",
                        CUT: NEVER,
                        "items": {
                            "type": "object",
                            "properties": {
                                "traces": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    CUT: NEVER,
                                },
                                "x_scale": {"type": "string"},
                                "y_scale": {"type": "string"},
                            },
                            "required": ["traces"],
                        },
                        "description": (
                            "The panes, top first, of a .plt saved from LTspice that "
                            "was left as it is: what the window draws in place of "
                            "panes, in the form set_plot_panes takes. Absent when it "
                            "holds none for this run."
                        ),
                    },
                    "panes": {
                        "type": "array",
                        CUT: NEVER,
                        "items": {"type": "array", "items": {"type": "string"}, CUT: NEVER},
                        "description": "The traces asked for, per pane.",
                    },
                    "reason": {"type": "string", "description": "Why it was not shown."},
                    "note": {"type": "string"},
                    "not_placed": {
                        "type": "object",
                        "description": (
                            "Why a job's results were not put beside the sheet it ran "
                            "(the code and message jobs(action='place_beside_sheet') "
                            "refuses with); the run was opened on its own."
                        ),
                        "properties": {
                            "code": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["code", "message"],
                    },
                    "not_checked": {
                        "type": "array",
                        # Each is a check the reply would otherwise imply was made.
                        CUT: NEVER,
                        "items": {"type": "string"},
                        "description": (
                            "With sheet, for a job's results put beside it: what the "
                            "placement did not check, each said in full."
                        ),
                    },
                    "left_in_place": {
                        "type": "array",
                        # What a write left: leaving one out would pass it off as this run's.
                        CUT: NEVER,
                        "items": {"type": "string"},
                        "description": (
                            "Each of the sheet's result files beside it that this run did "
                            "not write: another run's, left where it was."
                        ),
                    },
                },
                "required": ["shown", "results", "plot_settings", "panes"],
            },
            "image": _IMAGE_SCHEMA,
            "image_path": {"type": "string"},
            "observations": OBSERVATIONS_SCHEMA,
            # Also carries the read-the-guide reminder the server adds to a
            # session's first reply (server.call_tool).
            "hint": {
                "type": "string",
                "description": "With traces_truncated: the call that reads the traces left out.",
            },
        },
    },
)
async def handle_plot_waveform(args: PlotWaveformInput, state: SessionState):
    if args.runs is not None:
        chosen = await _cases_chosen(args, state)
        if isinstance(chosen, _SeveralCases):
            return await _plot_several(args, state, chosen)
        ran: tuple[ExperimentJob, ExperimentCase, services.RunContext] | None = chosen
    else:
        ran = await _experiment_case(
            args.path, args.job_id, args.run_index, args.case_id, state, other_raw=args.other_raw
        )
    case = None if ran is None else ran[2]
    if case is not None:
        source = services.source_for_run(
            case, plot_index=args.plot_index or 0, dialect=args.dialect
        )
        run_index = case.identity["run_index"]
        netlist: Path | None = case.netlist
    else:
        source = _direct_source(
            args.path,
            args.job_id,
            state,
            plot_index=args.plot_index or 0,
            dialect=args.dialect,
        )
        run_index = args.run_index
        netlist = source.netlist
    raw_path = source.raw
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")

    raw = await services.load_raw(source, state)
    assert raw_path is not None
    # An other raw is not the case's results: shown on its own, as its path
    # would be, never as the results beside the case's sheet.
    placing = (
        _JobRun(ran[0], ran[1], asyncio.get_running_loop())
        if ran is not None and args.other_raw is None
        else None
    )
    sampled = raw.descriptor.axis
    if sampled is None:
        if not args.in_ltspice:
            raise NoAxisError(
                f"This plot has no sampled axis to chart: {_why_no_axis(raw.descriptor)}. "
                "analyze_results' value metric or inspect(kind='results', view='table') "
                "reads its values, and in_ltspice=true opens it in LTspice without a chart.",
                show_hint=False,
            )
        return await _show_without_chart(args, state, source, raw, placing)
    # Refuse unsupported coordinates before building a plot.
    guarded_axis(raw, 0)

    analysis_type = raw.descriptor.analysis
    x_is_log = sampled.quantity == "frequency"

    trace_names = raw.get_trace_names()
    axis_name = sampled.name
    groups = _signal_groups(
        args,
        functools.partial(services.resolve_signal, raw),
        lambda name: name == axis_name,
        lambda: [name for name in trace_names if name != axis_name],
    )
    cols = [sig for group in groups for sig in group]

    n_steps = get_step_count(raw)
    step_dicts = [dict(step.parameters) for step in raw.descriptor.steps]
    steps_to_plot = _steps_named(args, raw, step_dicts)

    ts = parse_time(args.t_start, "t_start")
    te = parse_time(args.t_end, "t_end")

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

    async def show() -> dict[str, Any]:
        return await _in_window_thread(
            _show_in_ltspice,
            state,
            raw_path,
            placing,
            raw.descriptor.original_plot_name,
            [[sig.name for sig in group] for group in plan.groups if group],
        )

    return await _deliver(
        args,
        state,
        _Chart(
            plan=plan,
            cases=[PlotCase.of_plan(raw, plan)],
            file_name=_plot_filename(
                raw_path, analysis_type, args.job_id, run_index, plot_index=raw.plot_index
            ),
            name=raw_path.stem,
            title=f"{raw_path.stem} — {analysis_type}",
            read=[(source, raw)],
            head={
                "analysis_type": analysis_type,
                "plot_index": raw.plot_index,
                "descriptor": asdict(raw.descriptor),
                "signals": [sig.name for sig in cols],
                "n_steps": n_steps,
                "steps_plotted": len(steps_to_plot),
            },
            show=show if args.in_ltspice else None,
            rest=lambda: _rest_of_summary(
                args, analysis_type, n_steps=n_steps, run_index=run_index
            ),
        ),
    )


@dataclass(frozen=True)
class _SeveralCases:
    """The cases of a job a ``runs`` selection names, when it names more
    than one: each that can be read, with what a read of it takes, and each
    it names that cannot, with why."""

    job: ExperimentJob
    readable: list[tuple[ExperimentCase, services.RunContext]]
    not_drawn: list[dict[str, Any]]


#: Why ``in_ltspice`` is refused for a chart of several cases.
_IN_LTSPICE_ONE_CASE = (
    "in_ltspice takes one case, and runs names more than one run: an LTspice window "
    "shows one case's results, from its sheet. Pass run_index or case_id with "
    "in_ltspice for the case to open there, or leave in_ltspice out to chart the "
    "cases here."
)


async def _cases_chosen(
    args: PlotWaveformInput, state: SessionState
) -> tuple[ExperimentJob, ExperimentCase, services.RunContext] | _SeveralCases:
    """The cases of the job that ``args.runs`` names (``services.select_cases``,
    as analyze_results reads them). A selection of one case of the job is read
    as that case named by ``case_id`` is; of several, each case is read as
    analyze_results reads it (``services.read_selected_case``), and each that
    cannot be drawn is named with why."""
    assert args.job_id is not None and args.runs is not None
    job = await services.resolve_job_async(args.job_id, state)
    selected, unmatched = services.select_cases(job, args.runs)
    if len(selected) == 1 and not unmatched:
        (case,) = selected
        _, context = await services.readable_case(
            job, state, case_id=case.case_id, other_raw=args.other_raw
        )
        return job, case, context
    if args.in_ltspice:
        raise ResultError(_IN_LTSPICE_ONE_CASE, show_hint=False)
    not_drawn: list[dict[str, Any]] = [
        {"case_id": run.case_id, "run_index": run.run_index, "code": run.code} for run in unmatched
    ]
    readable: list[tuple[ExperimentCase, services.RunContext]] = []
    for case in selected:
        ctx = await services.read_selected_case(
            job, case, state, require_raw=True, other_raw=args.other_raw
        )
        if isinstance(ctx, services.CaseUnread):
            not_drawn.append(
                {
                    **_case_named(case),
                    "code": ctx.code,
                    "detail": ctx.detail,
                    **({"evidence": ctx.evidence} if ctx.evidence is not None else {}),
                }
            )
            continue
        readable.append((case, ctx))
    return _SeveralCases(job, readable, not_drawn)


def _case_named(case: ExperimentCase) -> dict[str, Any]:
    """A case not drawn, as ``cases_not_drawn`` names it: its id, its run
    index and its status. The status is said, never decided on: whether the
    case can be read is ``services.case_refusal``'s answer."""
    return {"case_id": case.case_id, "run_index": case.run_index, "status": case.status}


def _runs_as_given(runs: RunsSelection) -> Any:
    """A ``runs`` value as a call names it, for a reply to name the same runs."""
    return runs.model_dump() if isinstance(runs, CaseSelection) else runs


#: Most cases a refusal of a selection with nothing to draw names with why.
_NOTHING_TO_DRAW_NAMED = 5


def _nothing_to_draw(
    not_drawn: list[dict[str, Any]], job_id: str, runs: RunsSelection
) -> ResultError:
    """The refusal of a selection none of whose cases can be drawn: why for
    the first few, and for more, how many there are and the call that names
    every one."""
    shown = not_drawn[:_NOTHING_TO_DRAW_NAMED]
    said = "; ".join(
        f"{item['case_id'] if item.get('case_id') else 'run ' + str(item['run_index'])}: "
        f"{item.get('detail') or item['code']}"
        for item in shown
    )
    if len(not_drawn) > len(shown):
        source = json.dumps({"job_id": job_id, "runs": _runs_as_given(runs)})
        said += "; " + cap_note(
            len(shown),
            len(not_drawn),
            "cases",
            f"analyze_results with the source {source} names every one under "
            "coverage.missing_cases",
        )
    return ResultError(f"No case of the selection can be drawn. {said}.", show_hint=False)


def _units_across(
    cases: list[PlotCase], cols: list[services.Signal]
) -> tuple[dict[str, str | None], list[str]]:
    """Each signal's unit as the descriptor of each case that holds it
    declares it (``_trace_units``); a unit the cases do not agree on is
    reported unknown."""
    units: dict[str, str | None] = {}
    unverified: list[str] = []
    for sig in cols:
        seen: list[str | None] = []
        for case in cases:
            own = case.signal(sig)
            if own is not None:
                case_units, _ = _trace_units(case.raw, [own])
                seen.append(case_units[own.name])
        unit = seen[0] if seen and all(u == seen[0] for u in seen) else None
        units[sig.name] = unit
        if unit is None:
            unverified.append(sig.name)
    return units, unverified


async def _plot_several(args: PlotWaveformInput, state: SessionState, chosen: _SeveralCases):
    """Draw several cases of one job in one chart: each case's traces, each
    labelled by what tells its point from the others drawn, with every case
    selected and not drawn named."""
    assert args.job_id is not None and args.runs is not None
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")
    not_drawn = list(chosen.not_drawn)
    loaded: list[tuple[ExperimentCase, services.AnalysisSource, DecodedRaw]] = []
    for case, ctx in chosen.readable:
        source = services.source_for_run(
            ctx, plot_index=args.plot_index or 0, dialect=args.dialect
        )
        try:
            raw = await services.load_raw(source, state)
        except ResultError as exc:
            # A results file that cannot be parsed, or whose parse ran past its
            # deadline, costs only its case: named, and the others drawn.
            not_drawn.append({**_case_named(case), "code": exc.code, "detail": str(exc)})
            continue
        loaded.append((case, source, raw))
    if not loaded:
        raise _nothing_to_draw(not_drawn, args.job_id, args.runs)

    no_axis = [case.case_id for case, _, raw in loaded if raw.descriptor.axis is None]
    if no_axis:
        raise NoAxisError(
            f"The plots of {', '.join(no_axis)} have no sampled axis to chart. "
            "analyze_results' value metric reads their values, and plot_waveform with "
            "one case_id and in_ltspice=true opens one in LTspice.",
            show_hint=False,
        )
    by_analysis: dict[str, list[str]] = {}
    for case, _, raw in loaded:
        by_analysis.setdefault(raw.descriptor.analysis, []).append(case.case_id)
    if len(by_analysis) > 1:
        kinds = "; ".join(f"{kind}: {', '.join(ids)}" for kind, ids in by_analysis.items())
        raise ResultError(
            f"These cases' plots differ in analysis ({kinds}), and one chart has one kind "
            "of axis. Plot them in separate calls, each with runs naming cases of one "
            "analysis, or pick another plot_index.",
            show_hint=False,
        )
    for _, _, raw in loaded:
        # Refuse unsupported coordinates before building a plot.
        guarded_axis(raw, 0)

    stepped: list[tuple[ExperimentCase, services.AnalysisSource, DecodedRaw, list[int]]] = []
    for case, source, raw in loaded:
        try:
            steps = _steps_named(
                args, raw, [dict(step.parameters) for step in raw.descriptor.steps]
            )
        except ResultError as exc:
            not_drawn.append(
                {
                    **_case_named(case),
                    "code": "step_not_found",
                    "detail": str(exc),
                }
            )
            continue
        stepped.append((case, source, raw, steps))
    if not stepped:
        raise _nothing_to_draw(not_drawn, args.job_id, args.runs)

    first_raw = stepped[0][2]
    first_path = stepped[0][1].raw
    assert first_path is not None
    analysis_type = first_raw.descriptor.analysis
    axes = {raw.descriptor.axis.name for _, _, raw, _ in stepped if raw.descriptor.axis}
    # The spelling each signal was asked by, to read it in every case.
    asked: dict[str, str] = {}

    def resolve(name: str) -> services.Signal:
        first_refusal: ResultError | None = None
        for _, _, raw, _ in stepped:
            try:
                sig = services.resolve_signal(raw, name)
            except ResultError as exc:
                first_refusal = first_refusal or exc
                continue
            asked.setdefault(sig.name, name)
            return sig
        assert first_refusal is not None
        raise first_refusal

    def every() -> list[str]:
        names: dict[str, None] = {}
        for _, _, raw, _ in stepped:
            axis = raw.descriptor.axis.name if raw.descriptor.axis else None
            names.update(dict.fromkeys(n for n in raw.get_trace_names() if n != axis))
        return list(names)

    groups = _signal_groups(args, resolve, lambda name: name in axes, every)
    cols = [sig for group in groups for sig in group]

    def own_signals(raw: DecodedRaw) -> dict[str, services.Signal]:
        held: dict[str, services.Signal] = {}
        for sig in cols:
            try:
                held[sig.name] = services.resolve_signal(raw, asked.get(sig.name, sig.name))
            except ResultError:
                continue
        return held

    labels = label_assignments([dict(case.assignments) for case, _, _, _ in stepped])
    plot_cases = [
        PlotCase(
            raw,
            steps,
            [dict(step.parameters) for step in raw.descriptor.steps],
            get_step_count(raw) > 1,
            signals=own_signals(raw),
            case_id=case.case_id,
            run_index=case.run_index,
            label_assignments=label,
        )
        for (case, _, raw, steps), label in zip(stepped, labels, strict=True)
    ]
    sampled = first_raw.descriptor.axis
    assert sampled is not None
    # Each case carries its own steps (PlotCase); the plan says what the
    # chart draws and how its axis and panels are named, as for one case.
    plan = plan_plot(
        first_raw,
        groups,
        split_by_unit=args.panels is None,
        netlist=None,
        steps=[],
        step_dicts=[],
        analysis_type=analysis_type,
        x_is_log=sampled.quantity == "frequency",
        ts=parse_time(args.t_start, "t_start"),
        te=parse_time(args.t_end, "t_end"),
        annotate=args.annotate,
        units_read=_units_across(plot_cases, cols),
    )
    n_cases = len(stepped)
    return await _deliver(
        args,
        state,
        _Chart(
            plan=plan,
            cases=plot_cases,
            file_name=_plot_filename(
                first_path,
                analysis_type,
                args.job_id,
                0,
                plot_index=first_raw.plot_index,
                cases=n_cases,
            ),
            name=args.job_id,
            title=f"{args.job_id} — {analysis_type}: {n_cases} cases",
            read=[(source, raw) for _, source, raw, _ in stepped],
            head={
                "analysis_type": analysis_type,
                "plot_index": first_raw.plot_index,
                "cases": [
                    {
                        "case_id": case.case_id,
                        "run_index": case.run_index,
                        "assignments": dict(case.assignments),
                        "n_steps": get_step_count(raw),
                    }
                    for case, _, raw, _ in stepped
                ],
                "signals": [sig.name for sig in cols],
            },
            not_drawn=not_drawn,
            show=None,
            rest=lambda: _rest_of_summary(
                args,
                analysis_type,
                n_steps=max(get_step_count(raw) for _, _, raw, _ in stepped),
                run_index=0,
            ),
        ),
    )


@dataclass(frozen=True)
class _Chart:
    """What one chart draws and how its reply names it, for :func:`_deliver`."""

    plan: PlotPlan
    cases: list[PlotCase]
    #: The HTML file's name, in the plots folder or ``out_dir``.
    file_name: str
    #: What is drawn, for the page's summary line.
    name: str
    title: str
    #: Each source drawn with its raw, whose read observations the reply carries.
    read: list[tuple[services.AnalysisSource, DecodedRaw]]
    #: The reply's fields that say what was drawn, after its path.
    head: dict[str, Any]
    #: Opens the run in LTspice's window, for ``in_ltspice``.
    show: Callable[[], Awaitable[dict[str, Any]]] | None
    #: The call that reads the traces a cut summary leaves out.
    rest: Callable[[], str]
    #: With several cases: each selected case not drawn, with why.
    not_drawn: list[dict[str, Any]] | None = None


async def _deliver(args: PlotWaveformInput, state: SessionState, chart: _Chart):
    """Write the chart, deliver it (widget, browser, image, LTspice window) and
    build the reply: what was drawn, each trace's summary, and what was not."""
    plan = chart.plan
    # Held to WAVEFORM_MAX_POINTS_CAP by the input model, which notes it.
    max_points = args.max_points or _DEFAULT_PLOT_MAX_POINTS
    # Asked for in LTspice, the chart is not also opened in a browser unless
    # that was asked for too.
    open_default = state.config.open_plot and not args.in_ltspice
    open_locally = open_default if args.open is None else args.open
    attach = state.config.attach_plot if args.attach_plot is None else args.attach_plot

    # Into ``out_dir`` when named, else the store: never beside a raw the
    # caller named, which on WSL can sit in a Windows temp the client cannot read.
    dest_dir = safe_path(args.out_dir, state) if args.out_dir else state.store.plots_dir
    out_path = dest_dir / chart.file_name

    title = chart.title
    try:
        plot = await asyncio.to_thread(
            extract_cases, chart.cases, plan, summary_limit=_TRACE_SUMMARY_MAX
        )
        written = await asyncio.to_thread(
            write_plot_file, plot, chart.name, max_points, out_path, title
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
    if chart.show is not None:
        shown_in_ltspice = await chart.show()

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
    # `delivery`) is not restated here. Each case drawn says what its read
    # says; one thing several cases say is said once.
    observations: list[dict[str, Any]] = []
    for source, raw in chart.read:
        for item in await asyncio.to_thread(services.read_observations, source, raw):
            if item not in observations:
                observations.append(item)
    if plan.unverified_units:
        observations.append(
            {
                "code": "trace_unit_unknown",
                "kind": "value",
                "detail": (
                    "The selected descriptor does not establish a unit for "
                    f"{', '.join(plan.unverified_units)}."
                    + (
                        " Where the cases drawn declare different units for one, "
                        "it is reported unknown."
                        if len(chart.cases) > 1
                        else ""
                    )
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
    if facts.get("cases_without_step_values"):
        observations.append(
            {
                "code": "step_value_unavailable",
                "kind": "value",
                "detail": (
                    "Step legend labels name the step's index, not its values, for "
                    f"{', '.join(facts['cases_without_step_values'])}: no .step parameter "
                    "map found in their .log."
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
        **chart.head,
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
    if chart.not_drawn is not None:
        # A case the window leaves no sample of is not drawn: said, not listed.
        empty = set(facts["cases_window_empty"])
        data["cases"] = [case for case in data["cases"] if case["case_id"] not in empty]
        data["cases_not_drawn"] = [
            *chart.not_drawn,
            *(
                {
                    "case_id": case["case_id"],
                    "run_index": case["run_index"],
                    "code": "window_empty",
                    "detail": "The [t_start, t_end] window holds no sample of its results.",
                }
                for case in chart.head["cases"]
                if case["case_id"] in empty
            ),
        ]
        data["signals_not_in_cases"] = [
            {"signal": signal, "case_ids": case_ids}
            for signal, case_ids in facts["signals_lacking"].items()
        ]
    # format.cap_list's convention: a cut list carries its total beside it. The
    # summaries past the cap were never computed, so the total is counted, not
    # sliced from a list. The hint beside it names the call that reads the rest.
    if traces_total > len(traces):
        data["traces_truncated"] = traces_total
        if chart.not_drawn is not None:
            data["cases_unsummarized"] = facts["cases_unsummarized"]
        data["hint"] = chart.rest()
    held = args.argument_notes()
    if held:
        data["warnings"] = held
    if image is not None:
        data["image"] = image.to_dict()
        data["image_path"] = str(png_path)
    if shown_in_ltspice is not None:
        data["ltspice"] = shown_in_ltspice
    result = format_response(data)

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
