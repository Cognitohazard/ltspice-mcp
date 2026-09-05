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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
from mcp import types
from pydantic import Field

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError
from ltspice_mcp.lib import atomic_write, desktop, services
from ltspice_mcp.lib.ac_analysis import (
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.ac_structure import AcStructureResult, analyze_ac_structure
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.metrics import (
    classify_analysis,
    guarded_axis,
    parse_time,
    window_indices,
)
from ltspice_mcp.lib.plot_html import (
    WIDGET_RESOURCE_URI,
    WIDGET_SPEC_META_KEY,
    build_plot_html,
)
from ltspice_mcp.lib.raw_parser import (
    dc_axis_name,
    get_step_count,
    safe_magnitude_db,
)
from ltspice_mcp.lib.signal_analysis import (
    downsample_minmax,
)
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import ExperimentJob, SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    OBSERVATIONS_SCHEMA,
    ToolInput,
    format_observations,
    format_response,
    registry,
    safe_path,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _direct_source(
    raw_file: str | None, job_id: str | None, state: SessionState
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
    return services.resolve_analysis_source(state, raw_file=raw_file, job_id=job_id)


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
    it is refused beside a raw_file or a legacy job. raw_file together with
    job_id is left to ``_direct_source``'s exclusivity error.
    """
    job = await services.resolve_job_async(job_id, state) if job_id and not raw_file else None
    if isinstance(job, ExperimentJob):
        return services.experiment_run_context(job, state, run_index=run_index, case_id=case_id)
    if case_id is not None:
        raise ResultError(
            "case_id selects a run_experiments case; pass it with that job's job_id."
        )
    return None


async def _resolve_artifact_dest(
    *,
    out_dir: str | None,
    job_id: str | None,
    raw_file: str | None,
    filename: str,
    artifact: str,
    state: SessionState,
    circuit_dir: Path | None = None,
) -> Path:
    """Resolve where a generated artifact (the plot HTML) is written.

    An explicit ``out_dir`` (validated via ``safe_path``) wins; otherwise the
    destination is ``Store.circuit_plots`` of a Linux-side anchor — the CIRCUIT
    for a job_id, the raw's own directory for a raw_file, because a job-run raw
    can live in a Windows temp under /mnt/c the client cannot Read. A caller
    that already resolved the circuit (an experiment case, whose job has no
    single netlist) passes it as ``circuit_dir``. Server-artifact paths skip
    ``safe_path`` except the out_dir override; the resolved path must stay under
    its anchor (a symlinked sidecar would otherwise redirect the write out).
    """
    if out_dir:
        dest_anchor = safe_path(out_dir, state)
        out_path = (dest_anchor / filename).resolve()
    else:
        if circuit_dir is not None:
            dest_anchor = circuit_dir
        elif job_id:
            # Only a caller that resolved the run itself can name a circuit
            # directory (``circuit_dir`` above); a bare job_id cannot, because
            # an experiment spans several decks.
            raise ResultError(
                "Pass out_dir, or resolve the run first — a job id alone does not "
                "name one circuit directory to write beside."
            )
        else:
            dest_anchor = safe_path(raw_file, state).parent  # type: ignore[arg-type]
        out_path = (Store.circuit_plots(dest_anchor) / filename).resolve()
    if not out_path.is_relative_to(dest_anchor.resolve()):
        raise ResultError(
            f"Refusing to write the {artifact} outside the destination directory "
            "(a symlinked .ltspice-mcp/ sidecar would redirect it)."
        )
    return out_path


# ---------------------------------------------------------------------------
# export_waveform — full-fidelity CSV egress to disk
# ---------------------------------------------------------------------------

# Generous backstop against a pathological export exhausting memory/disk. Full
# fidelity is the contract, so this is high and RAISES with guidance to window —
# never silently truncates (no silent caps).
_EXPORT_MAX_ROWS = 20_000_000

# x-column header per analysis type (unit-tagged so the CSV is self-describing).
_X_HEADER = {
    "transient": "time_s",
    "ac": "freq_Hz",
    "noise": "freq_Hz",
    "dc": "sweep",
}


def _csv_x_header(raw, analysis_type: str) -> str:
    """X-column header. Transient/AC/noise are fixed (time_s/freq_Hz); a .dc
    sweep names the swept variable from the raw (e.g. ``Vin_V``) instead of a
    bare ``sweep`` so the column is self-describing."""
    if analysis_type != "dc":
        return _X_HEADER[analysis_type]
    name, unit = dc_axis_name(raw)
    if name:
        base = re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_") or "sweep"
        return f"{base}_{unit}" if unit else base
    return "sweep"


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
    cols: list[str],
    n_steps: int,
    analysis_type: str,
    ts: float | None,
    te: float | None,
    complex_format: str,
    out_path: Path,
    should_abort: Callable[[], bool] | None = None,
    steps_to_export: list[int] | None = None,
    step_log_path: Path | None = None,
) -> dict:
    """Assemble tidy/long rows for every step, render CSV, write it atomically.

    Runs entirely in a worker thread: heavy numpy reads, O(N) row assembly, and
    file I/O. Returns only FACTS — the handler turns them into observations and
    builds the response on the event loop (the concurrency contract keeps
    response building off worker threads).
    """
    stepped = n_steps > 1
    x_header = _csv_x_header(raw, analysis_type)

    # Per-step .step parameter values for the step_value column. spicelib's
    # get_steps() carries only integers, not the name=value map, so recover it
    # from the sibling .log (same source query_value(step_axis=) uses).
    # parse_step_iterations returns [] for a missing/unreadable log.
    step_dicts: list[dict[str, float]] = (
        parse_step_iterations(step_log_path or raw_path.with_suffix(".log")) if stepped else []
    )

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
            axis = guarded_axis(raw, step)
            lo, hi = window_indices(axis, ts, te)
            if lo >= hi:
                # This step's axis does not intersect the window (a step may end
                # earlier than its siblings). Skip it; surfaced as a fact.
                empty_steps.append(step)
                continue
            axis_w = axis[lo:hi]
            col_names: list[str] = []
            col_arrays: list[np.ndarray] = []
            for name in cols:
                wave = np.asarray(raw.get_wave(name, step=step))
                if wave.size == 0:
                    raise ResultError(f"Signal {name!r} has no data points at step {step}.")
                wave_w = wave[lo:hi]
                non_finite += int(np.count_nonzero(~np.isfinite(wave_w)))
                # Key on the trace's own dtype, not the run type: an AC raw can hold
                # a real trace, and a stray complex trace must not become one column.
                if np.iscomplexobj(wave_w):
                    had_complex = True
                    names, arrays = _complex_columns(name, wave_w, complex_format)
                else:
                    names, arrays = [name], [wave_w]
                col_names.extend(names)
                col_arrays.extend(arrays)

            if header is None:
                prefix = ["step_index", "step_value"] if stepped else []
                header = [*prefix, x_header, *col_names]
                csv_writer.writerow(header)

            lo0, hi0 = float(axis_w[0]), float(axis_w[-1])
            win_lo = lo0 if win_lo is None else min(win_lo, lo0)
            win_hi = hi0 if win_hi is None else max(win_hi, hi0)

            # .tolist() converts numpy -> python floats (full round-trippable repr)
            # at C speed; zip transposes columns into tidy/long rows.
            columns = [axis_w.tolist(), *(a.tolist() for a in col_arrays)]
            if stepped:
                label = (
                    ";".join(f"{k}={v:g}" for k, v in step_dicts[step].items())
                    if step < len(step_dicts)
                    else ""
                )
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
        "step_values_available": bool(step_dicts) if stepped else None,
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
# Global backstop on the rendered panel size. ``max_points`` caps each series,
# but null-padding every series onto a union x (distinct per-step axes) multiplies
# series-count by union-length — so a many-step distinct-axis run could blow far
# past the per-series cap. Refuse with guidance before materializing, rather than
# allocate a giant payload / write an unopenable HTML (no silent truncation).
_PLOT_MAX_CELLS = 10_000_000

_X_LABEL = {
    "transient": "Time (s)",
    "ac": "Frequency (Hz)",
    "noise": "Frequency (Hz)",
    "dc": "Sweep",
}


def _plot_filename(raw_path: Path, analysis_type: str, job_id: str | None, run_index: int) -> str:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S_%f")
    run = f"_run{run_index}" if job_id else ""
    return f"{raw_path.stem}_{analysis_type}{run}_{stamp}.html"


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


def _union_panel(
    series: list[tuple[np.ndarray, np.ndarray, str]],
    x_scale: str,
    x_label: str,
    y_label: str,
) -> tuple[dict, bool]:
    """Build a uPlot panel (shared x + N y-series) from per-series ``(x, y, label)``.

    Series whose x differs from the others — e.g. a transient ``.step`` run where
    each step has its own adaptive time vector — are null-padded onto the union x
    so each renders as a clean gap off its own support (uPlot's data model is one
    shared x-row + N y-series). Returns ``(panel, unioned)``.

    Guards the rendered size in two cheap stages so neither the concat nor the pad
    can blow up: the longest single series is a lower bound on the union (stage 1,
    before concatenating — also bounds the concat to <= the cap), and the actual
    union length is the exact size (stage 2, before padding).
    """
    n_rows = len(series) + 1  # the x row plus one row per series
    longest = max(len(s[0]) for s in series)
    if n_rows * longest > _PLOT_MAX_CELLS:
        raise _plot_cells_exceeded(n_rows, longest)
    # When every series already shares one x vector (the common case: several
    # signals from a single run), use it directly. np.unique would collapse
    # legitimately-repeated timepoints (solver restarts emit duplicate x), which
    # then makes each series look mismatched and wrongly flags the panel as
    # step-axis-unioned even though there is no .step sweep.
    first_x = series[0][0]
    all_same = all(len(s[0]) == len(first_x) and np.array_equal(s[0], first_x) for s in series)
    union = first_x if all_same else np.unique(np.concatenate([s[0] for s in series]))
    if n_rows * len(union) > _PLOT_MAX_CELLS:
        raise _plot_cells_exceeded(n_rows, len(union))
    data: list[list[float | None]] = [_to_json_floats(union)]
    labels: list[dict[str, str]] = []
    unioned = False
    for x, y, label in series:
        if len(x) == len(union) and np.array_equal(x, union):
            data.append(_to_json_floats(y))
        else:
            unioned = True
            col = np.full(len(union), np.nan)
            col[np.searchsorted(union, x)] = y
            data.append(_to_json_floats(col))
        labels.append({"label": label})
    panel = {
        "x_scale": x_scale,
        "x_label": x_label,
        "y_label": y_label,
        "series": labels,
        "data": data,
    }
    return panel, unioned


def _compact_hz(f: float) -> str:
    """Render a frequency (Hz) compactly with an engineering suffix.

    e.g. 1200 -> "1.2k", 20000 -> "20k", 3.4e6 -> "3.4M". Used for the on-plot
    corner-marker labels, which must stay short.
    """
    if not np.isfinite(f) or f <= 0:
        return "?"
    for div, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m")):
        if f >= div:
            v = f / div
            s = f"{v:.1f}".rstrip("0").rstrip(".")
            return f"{s}{suffix}"
    return f"{f:.2g}"


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


def _compute_plot_spec(
    raw,
    cols: list[str],
    steps_to_plot: list[int],
    step_dicts: list[dict[str, float]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    max_points: int,
    annotate: bool = False,
) -> tuple[dict, dict]:
    """Build the renderer-ready plot spec + coverage facts (no I/O).

    Runs in a worker thread (heavy numpy). Returns ``(spec, facts)``: ``spec`` is
    the PlotSpec (panels/bode/analysis_type) consumed by both the offline HTML
    file and the in-chat widget (called at different point budgets); ``facts`` are
    the coverage facts the handler turns into observations on the event loop.
    """
    is_ac = analysis_type == "ac"
    x_label = _X_LABEL[analysis_type]
    multi = len(steps_to_plot) > 1

    def _label(col: str, step: int) -> str:
        if not multi:
            return col
        sv = (
            ";".join(f"{k}={v:g}" for k, v in step_dicts[step].items())
            if step < len(step_dicts)
            else ""
        )
        return f"{col} [{sv}]" if sv else f"{col} [step {step}]"

    empty_steps: list[int] = []
    non_finite = 0
    downsampled = False
    points_per_series: list[int] = []
    phase_warnings: list[str] = []
    win_lo: float | None = None
    win_hi: float | None = None

    def _track_window(x: np.ndarray) -> None:
        nonlocal win_lo, win_hi
        lo0, hi0 = float(x[0]), float(x[-1])
        win_lo = lo0 if win_lo is None else min(win_lo, lo0)
        win_hi = hi0 if win_hi is None else max(win_hi, hi0)

    if is_ac:
        mag_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        phase_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        # Single-trace annotation: capture the full-resolution complex response
        # BEFORE any downsampling so the corner reading runs on every sample.
        single_trace = annotate and len(cols) == 1 and len(steps_to_plot) == 1
        annotate_freq: np.ndarray | None = None
        annotate_h: np.ndarray | None = None
        for step in steps_to_plot:
            axis = guarded_axis(raw, step)
            lo, hi = window_indices(axis, ts, te)
            if lo >= hi:
                empty_steps.append(step)
                continue
            for col in cols:
                wave = np.asarray(raw.get_wave(col, step=step))[lo:hi]
                freq, h = prepare_ac_arrays(axis[lo:hi], wave)
                if single_trace:
                    annotate_freq, annotate_h = freq, h
                mag = safe_magnitude_db(h)
                phase, warns = unwrap_phase_safe(h)
                phase_warnings.extend(warns)
                non_finite += int(np.count_nonzero(~np.isfinite(mag)))
                non_finite += int(np.count_nonzero(~np.isfinite(phase)))
                if len(freq) > max_points:
                    downsampled = True
                    f_ds, mag = downsample_minmax(freq, mag, max_points)
                    _, phase = downsample_minmax(freq, phase, max_points)
                    freq = f_ds
                _track_window(freq)
                points_per_series.append(len(freq))
                label = _label(col, step)
                mag_series.append((freq, mag, label))
                phase_series.append((freq, phase, label))
        if not mag_series:
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {len(steps_to_plot)} steps." if multi else ".")
            )
        mag_panel, u1 = _union_panel(mag_series, "log", x_label, "Magnitude (dB)")
        phase_panel, u2 = _union_panel(phase_series, "log", x_label, "Phase (deg)")
        unioned = u1 or u2
        spec = {"analysis_type": analysis_type, "bode": True, "panels": [mag_panel, phase_panel]}
        if single_trace and annotate_freq is not None and annotate_h is not None:
            annotations, nmp = _ac_annotations(annotate_freq, annotate_h)
            spec["annotations"] = annotations
            spec["nmp"] = nmp
    else:
        plot_series: list[tuple[np.ndarray, np.ndarray, str]] = []
        for col in cols:
            for step in steps_to_plot:
                axis = guarded_axis(raw, step)
                lo, hi = window_indices(axis, ts, te)
                if lo >= hi:
                    if step not in empty_steps:
                        empty_steps.append(step)
                    continue
                axis_w = axis[lo:hi]
                wave = np.asarray(raw.get_wave(col, step=step))[lo:hi]
                if np.iscomplexobj(wave):
                    # Defensive: a stray complex trace in a non-AC raw.
                    wave = np.real(wave)
                non_finite += int(np.count_nonzero(~np.isfinite(wave)))
                x_arr, y_arr = axis_w, wave
                if len(y_arr) > max_points:
                    downsampled = True
                    x_arr, y_arr = downsample_minmax(axis_w, wave, max_points)
                _track_window(x_arr)
                points_per_series.append(len(y_arr))
                plot_series.append((x_arr, y_arr, _label(col, step)))
        if not plot_series:
            raise ResultError(
                "The [t_start, t_end] window selects no samples"
                + (f" in any of the {len(steps_to_plot)} steps." if multi else ".")
            )
        y_label = ", ".join(cols) if len(cols) <= 3 else f"{len(cols)} signals"
        panel, unioned = _union_panel(
            plot_series, "log" if x_is_log else "linear", x_label, y_label
        )
        spec = {"analysis_type": analysis_type, "bode": False, "panels": [panel]}

    series_count = sum(len(p["series"]) for p in spec["panels"])
    facts = {
        "panels": len(spec["panels"]),
        "series_count": series_count,
        "points_per_series": points_per_series,
        "downsampled": downsampled,
        "unioned": unioned,
        "empty_steps": sorted(set(empty_steps)),
        "non_finite": non_finite,
        "phase_unwrapped": is_ac,
        "phase_warnings": phase_warnings,
        "window_used": [win_lo, win_hi] if win_lo is not None else [],
        "step_values_available": (bool(step_dicts) if multi else None),
    }
    return spec, facts


def build_plot_file(
    raw,
    raw_path: Path,
    cols: list[str],
    steps_to_plot: list[int],
    step_dicts: list[dict[str, float]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    max_points: int,
    out_path: Path,
    title: str,
    annotate: bool = False,
) -> dict:
    """Compute the spec, assemble the offline HTML, write it atomically; return facts.

    Runs in a worker thread (heavy numpy + HTML build + file I/O). The handler
    turns the returned facts into observations on the event loop (the concurrency
    contract keeps response building off worker threads).
    """
    spec, facts = _compute_plot_spec(
        raw, cols, steps_to_plot, step_dicts, analysis_type, x_is_log, ts, te, max_points, annotate
    )
    summary = f"{raw_path.stem} — {analysis_type}: {facts['series_count']} series"
    html_str = build_plot_html(spec, title=title, summary=summary)
    with atomic_write(out_path) as f:
        f.write(html_str)
    return facts


def _compute_widget_spec_json(
    raw,
    cols: list[str],
    steps_to_plot: list[int],
    step_dicts: list[dict[str, float]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    max_points: int,
    annotate: bool = False,
) -> str:
    """Build the compact widget chart spec and serialize it — all in the worker.

    Both the numpy spec build AND the (potentially large) JSON serialization run
    off the event loop. Returns the spec as a JSON string for the result ``_meta``
    (read by the widget in ``app.ontoolresult``); raises ``ResultError`` like
    :func:`_compute_plot_spec` (e.g. the cell cap), which the handler catches to
    fall back to local-open delivery.
    """
    spec, _ = _compute_plot_spec(
        raw, cols, steps_to_plot, step_dicts, analysis_type, x_is_log, ts, te, max_points, annotate
    )
    return json.dumps(spec, ensure_ascii=True, allow_nan=False)


class PlotWaveformInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this or ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Plot one run of a finished job — a run_experiments job or a legacy "
            "single/sweep/MC job — instead of a raw_file path; pick the run with "
            "``run_index`` (or ``case_id`` for an experiment)."
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
        description="Trace names to plot (e.g. ['V(out)', 'I(R1)']) or 'all' for every non-axis trace.",
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
    open: bool = Field(
        default=True,
        description=(
            "Open the written HTML in the local browser. Applies to terminal clients "
            "only — ignored when the chart is delivered as an in-chat widget (MCP Apps "
            "host). Set false to only write the file."
        ),
    )
    annotate: bool = Field(
        default=True,
        description=(
            "Annotate an AC/Bode plot with detected corner markers (vertical lines) + "
            "an out-of-phase-zero / delay flag, from ac_structure. AC plots only; ignored "
            "for transient/DC."
        ),
    )
    out_dir: str | None = Field(
        default=None,
        description=(
            "Directory to write the HTML into (resolved under an allowed path; "
            "created if needed). Default: a '.ltspice-mcp/plots/' sidecar next to "
            "the circuit for a job_id, or next to the raw for a raw_file."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@registry.tool(
    name="plot_waveform",
    description=(
        "Render an interactive chart of one or more signals for a person to look at "
        "(zoom/pan/hover). It returns no data values to the model; it produces a "
        "picture.\n\n"
        "Picks the chart from the run type: transient (V/I vs time), DC sweep, AC "
        "Bode (stacked magnitude-dB + phase-deg vs log frequency), noise (vs log "
        "frequency); a .step / Monte-Carlo run overlays every step as a labelled "
        "trace (or pass ``step`` for one). Full fidelity by default, with a "
        "min/max-preserving downsample above ``max_points`` that keeps spikes and "
        "reports that it downsampled. Writes a self-contained HTML file and returns "
        "its path "
        "— into ``out_dir`` if given, else a '.ltspice-mcp/plots/' sidecar next to "
        "the circuit (for a job_id) or next to the raw (for a raw_file); on a host "
        "that supports MCP Apps the chart is "
        "also embedded as an interactive in-chat widget, otherwise it opens in your "
        "local browser.\n\n"
        "For numbers use analyze_results instead: the waveform recipe returns a "
        "decimated table in context (or every sample as CSV on disk), and "
        "signal_stats and the bode_* recipes return scalars."
    ),
    input_model=PlotWaveformInput,
    annotations=types.ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=True,
    ),
    # MCP Apps (SEP-1865): declare the in-chat renderer so an apps-capable host
    # fetches it via resources/read and pipes the chart spec into it.
    meta={"ui": {"resourceUri": WIDGET_RESOURCE_URI}},
    output_schema={
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "analysis_type": {"type": "string", "enum": ["transient", "ac", "dc", "noise"]},
            "signals": {"type": "array", "items": {"type": "string"}},
            "n_steps": {"type": "integer"},
            "steps_plotted": {"type": "integer"},
            "panels": {"type": "integer"},
            "series_count": {"type": "integer"},
            "points_per_series": {"type": "array", "items": {"type": "integer"}},
            "max_points": {"type": "integer"},
            "downsampled": {"type": "boolean"},
            "window_used": {"type": "array", "items": {"type": "number"}},
            "delivery": {"type": "string", "enum": ["terminal", "ui"]},
            "opened": {"type": "boolean"},
            "opener": {"type": ["string", "null"]},
            "observations": OBSERVATIONS_SCHEMA,
        },
    },
)
async def handle_plot_waveform(args: PlotWaveformInput, state: SessionState):
    case = await _experiment_case(args.raw_file, args.job_id, args.run_index, args.case_id, state)
    if case is not None:
        raw_path = case.raw
        run_index = case.identity["run_index"]
    else:
        raw_path = _direct_source(args.raw_file, args.job_id, state).raw
        run_index = args.run_index
    fmt = args.format
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")

    raw = await services.load_raw(raw_path, state)
    # A .op raw has no sweep axis to plot — refuse early with the clean pointer.
    guarded_axis(raw, 0, raw_path)

    _, analysis_type, _, x_is_log = classify_analysis(raw)

    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]
    if args.signals == "all":
        cols = list(trace_names[1:])
    else:
        seen: set[str] = set()
        cols = []
        for s in args.signals:
            canon = services.validate_signal(raw, s)
            if canon == axis_name:
                raise ResultError(f"{s!r} is the sweep axis, not a signal column.")
            if canon not in seen:
                seen.add(canon)
                cols.append(canon)
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

    # parse_step_iterations returns [] for a missing/unreadable log.
    step_dicts: list[dict[str, float]] = (
        parse_step_iterations(raw_path.with_suffix(".log")) if len(steps_to_plot) > 1 else []
    )

    max_points = min(args.max_points or _DEFAULT_PLOT_MAX_POINTS, PLOT_MAX_POINTS_CEILING)

    out_path = await _resolve_artifact_dest(
        out_dir=args.out_dir,
        job_id=args.job_id,
        raw_file=args.raw_file,
        filename=_plot_filename(raw_path, analysis_type, args.job_id, run_index),
        artifact="plot",
        state=state,
        circuit_dir=case.circuit_path.parent if case is not None else None,
    )

    title = f"{raw_path.stem} — {analysis_type}"
    try:
        facts = await asyncio.to_thread(
            build_plot_file,
            raw,
            raw_path,
            cols,
            steps_to_plot,
            step_dicts,
            analysis_type,
            x_is_log,
            ts,
            te,
            max_points,
            out_path,
            title,
            args.annotate,
        )
    except ValueError as e:
        raise ResultError(f"Failed to build the plot (corrupt or truncated .raw?): {e}") from e

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
            widget_spec_json = await asyncio.to_thread(
                _compute_widget_spec_json,
                raw,
                cols,
                steps_to_plot,
                step_dicts,
                analysis_type,
                x_is_log,
                ts,
                te,
                budget,
                args.annotate,
            )
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
    opened, opener = False, None
    if widget_spec_json is None and args.open:
        opened, opener = await asyncio.to_thread(desktop.open_in_desktop, out_path)

    # Surface FACTS, not verdicts (result-trust doctrine).
    observations: list[dict] = [
        {
            "code": "plot_written",
            "kind": "coverage",
            "detail": (
                f"Wrote an interactive {analysis_type} plot: {facts['panels']} panel(s), "
                f"{facts['series_count']} series ({len(steps_to_plot)} of {n_steps} step(s))."
            ),
        }
    ]
    if facts["downsampled"]:
        observations.append(
            {
                "code": "downsampled",
                "kind": "coverage",
                "detail": (
                    f"At least one series exceeded {max_points} points and was reduced by "
                    "min/max-preserving decimation (spikes preserved; sub-bucket detail not "
                    "shown). Pass a larger max_points or narrow [t_start, t_end] for more."
                ),
            }
        )
    if facts["phase_unwrapped"]:
        observations.append(
            {
                "code": "phase_unwrapped",
                "kind": "value",
                "detail": (
                    "Bode phase is UNWRAPPED for a readable continuous curve — this differs "
                    "from the waveform recipe's CSV, which keeps the wrapped np.angle as its lossless "
                    "primitive."
                ),
            }
        )
    for warn in facts["phase_warnings"]:
        observations.append({"code": "sparse_sweep", "kind": "value", "detail": warn})
    if facts["unioned"]:
        observations.append(
            {
                "code": "step_axis_unioned",
                "kind": "coverage",
                "detail": (
                    "Steps have different per-step x vectors; series were aligned onto a "
                    "union x (each renders as a gap off its own support)."
                ),
            }
        )
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
                    "gaps in the plot."
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
    if widget_spec_json is not None:
        observations.append(
            {
                "code": "widget_delivered",
                "kind": "coverage",
                "detail": (
                    "Client advertises MCP Apps (ui://) support; the chart spec rides in "
                    "this result's _meta for the host to render in-chat (not shown to the "
                    "model; local open skipped). The full-fidelity HTML was still written "
                    "to the returned path."
                ),
            }
        )
    else:
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
        if not args.open:
            observations.append(
                {
                    "code": "open_skipped",
                    "kind": "coverage",
                    "detail": "Local open skipped (open=false); open the returned path manually.",
                }
            )
        elif not opened:
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

    data = {
        "path": str(out_path),
        "analysis_type": analysis_type,
        "signals": cols,
        "n_steps": n_steps,
        "steps_plotted": len(steps_to_plot),
        "panels": facts["panels"],
        "series_count": facts["series_count"],
        "points_per_series": facts["points_per_series"],
        "max_points": max_points,
        "downsampled": facts["downsampled"],
        "window_used": facts["window_used"],
        "delivery": "ui" if widget_spec_json is not None else "terminal",
        "opened": opened,
        "opener": opener,
        "observations": observations,
    }
    if widget_spec_json is not None:
        head = (
            f"Rendered an interactive {analysis_type} plot widget in-chat (also wrote {out_path})"
        )
    else:
        head = f"Wrote interactive {analysis_type} plot to {out_path}"
        if opened:
            head += f" (opened with {opener})"
    lines = [head, *format_observations(observations)]
    result = format_response("\n".join(lines), data, fmt)

    if widget_spec_json is not None:
        # Pipe the compact chart spec through the result _meta (a non-model-visible
        # channel) as a JSON string. The MCP Apps host forwards the full result to
        # the widget (declared via the tool's ui.resourceUri), where
        # ``app.ontoolresult`` reads _meta and renders it — so the model sees the
        # summary/path, never the numbers.
        result.meta = {WIDGET_SPEC_META_KEY: widget_spec_json}
    return result
