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
import base64
import csv
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from mcp import types
from pydantic import Field

from ltspice_mcp.errors import AnalysisDeadlineExceeded, ResultError
from ltspice_mcp.lib import atomic_write, atomic_write_bytes, desktop, services
from ltspice_mcp.lib.ac_analysis import (
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.ac_structure import AcStructureResult, analyze_ac_structure
from ltspice_mcp.lib.log_parser import parse_step_iterations
from ltspice_mcp.lib.metrics import (
    classify_analysis,
    guarded_axis,
    noise_input_source_unit,
    parse_time,
    window_indices,
)
from ltspice_mcp.lib.plot_html import (
    WIDGET_RESOURCE_URI,
    WIDGET_SPEC_META_KEY,
    build_plot_html,
)
from ltspice_mcp.lib.plot_svg import render_plot_svg
from ltspice_mcp.lib.raster import RenderedImage, render_image
from ltspice_mcp.lib.raw_parser import (
    dc_axis_name,
    get_step_count,
    safe_magnitude_db,
    trace_unit,
)
from ltspice_mcp.lib.signal_analysis import (
    downsample_minmax,
    summarize_trace,
)
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    NEW_WORK_ANNOTATIONS,
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
    # Only ``raw_file`` is forwarded: an experiment's runs are case-addressed,
    # so ``_experiment_case`` has already answered every call that named a job.
    return services.resolve_analysis_source(state, raw_file=raw_file)


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
    cols: list[services.Signal],
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

# Most per-trace summaries one reply carries. A stepped or Monte Carlo overlay
# can plot hundreds of traces; past this many the reply lists the first ones and
# says how many it left out, and analyze_results' signal_stats reads the rest.
_TRACE_SUMMARY_MAX = 32
# Per-series point budget for the attached image. The image is under a thousand
# pixels wide, so more than about two points per pixel column adds bytes, not
# detail; the min/max decimation keeps every spike.
_IMAGE_MAX_POINTS = 2_000
# Most panels a caller may lay out by hand.
_MAX_PANELS = 8

_X_LABEL = {
    "transient": "Time (s)",
    "ac": "Frequency (Hz)",
    "noise": "Frequency (Hz)",
    "dc": "Sweep",
}
_X_UNIT = {"transient": "s", "ac": "Hz", "noise": "Hz"}


def _x_axis(raw, analysis_type: str) -> tuple[str, str | None]:
    """The x-axis label and unit. A .dc sweep is labelled with its swept
    variable (``V1 (V)``) when the raw names it."""
    if analysis_type != "dc":
        return _X_LABEL[analysis_type], _X_UNIT[analysis_type]
    name, unit = dc_axis_name(raw)
    if not name:
        return _X_LABEL["dc"], None
    return (f"{name} ({unit})" if unit else name), unit


def _is_input_noise(sig: services.Signal) -> bool:
    return "inoise" in sig.trace.lower()


async def resolve_input_noise_unit(
    analysis_type: str, cols: list[services.Signal], netlist: Path | None
) -> str | None:
    """The unit input-referred noise is referred to, from the deck's .NOISE line.

    Only read when a noise run plots an ``inoise`` trace; the deck read is a
    file parse, so it runs off the loop. ``None`` when there is no deck or it
    does not say.
    """
    if analysis_type != "noise" or not any(_is_input_noise(sig) for sig in cols):
        return None
    return await asyncio.to_thread(noise_input_source_unit, netlist)


def _trace_units(
    raw,
    cols: list[services.Signal],
    analysis_type: str,
    input_noise_unit: str | None,
) -> tuple[dict[str, str | None], list[str]]:
    """The unit of each signal's plotted values, and the input-noise traces whose
    unit could not be checked against the deck.

    The unit is the one the simulator declared (``trace_unit``), never one
    guessed from a name. A noise run plots spectral densities, so a declared V
    or A becomes V/√Hz or A/√Hz. LTspice declares input-referred noise as a
    voltage even when the .NOISE source is a current source, so an ``inoise``
    trace takes ``input_noise_unit`` when the deck gave one, and is listed as
    unverified when it did not.
    """
    units: dict[str, str | None] = {}
    unverified: list[str] = []
    for sig in cols:
        unit = trace_unit(raw, sig.trace)
        if analysis_type == "noise" and unit is not None:
            if _is_input_noise(sig):
                if input_noise_unit is None:
                    unverified.append(sig.name)
                else:
                    unit = input_noise_unit
            unit = f"{unit}/√Hz"
        units[sig.name] = unit
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
    """What a plot draws, resolved once and shared by every rendering of it.

    The HTML file, the in-chat widget spec and the attached image are each
    built from one plan, at their own point budgets, so they cannot disagree
    about which traces share a panel, what the panels are called, or which
    steps and window they cover.
    """

    groups: list[list[services.Signal]]
    units: dict[str, str | None]
    steps: list[int]
    step_dicts: list[dict[str, float]]
    analysis_type: str
    x_is_log: bool
    x_label: str
    ts: float | None
    te: float | None
    annotate: bool = False
    #: The run is a .step sweep, so each summary names its step even when one
    #: step was selected.
    stepped: bool = False

    @property
    def signals(self) -> list[services.Signal]:
        return [sig for group in self.groups for sig in group]


def plan_plot(
    raw,
    cols: list[services.Signal],
    *,
    steps: list[int],
    step_dicts: list[dict[str, float]],
    analysis_type: str,
    x_is_log: bool,
    ts: float | None,
    te: float | None,
    panels: list[list[services.Signal]] | None = None,
    annotate: bool = False,
    input_noise_unit: str | None = None,
) -> tuple[PlotPlan, list[str]]:
    """Lay ``cols`` out into panels: the caller's ``panels`` as given, else one
    panel per unit. Returns the plan and the input-noise traces whose unit the
    deck did not confirm (see :func:`_trace_units`)."""
    units, unverified = _trace_units(raw, cols, analysis_type, input_noise_unit)
    x_label, _ = _x_axis(raw, analysis_type)
    plan = PlotPlan(
        groups=panels if panels is not None else _unit_groups(cols, units),
        units=units,
        steps=steps,
        step_dicts=step_dicts,
        analysis_type=analysis_type,
        x_is_log=x_is_log,
        x_label=x_label,
        ts=ts,
        te=te,
        annotate=annotate,
        stepped=get_step_count(raw) > 1,
    )
    return plan, unverified


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
    labels: list[dict[str, Any]] = []
    unioned = False
    for x, y, label in series:
        if len(x) == len(union) and np.array_equal(x, union):
            data.append(_to_json_floats(y))
            labels.append({"label": label})
            continue
        unioned = True
        col = np.full(len(union), np.nan)
        at = np.searchsorted(union, x)
        col[at] = y
        data.append(_to_json_floats(col))
        # Padding and the series' own non-finite samples are both null in the
        # column, but a renderer must join the line across the first and break
        # it at the second — else a step whose samples interleave with the
        # others' is drawn as isolated points. ``gaps`` names the second kind.
        own_gaps = at[~np.isfinite(np.asarray(y, dtype=float))]
        labels.append({"label": label, "padded": True, "gaps": sorted(set(own_gaps.tolist()))})
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


def _summary_entry(
    plan: PlotPlan, sig: services.Signal, step: int, label: str, panel: int
) -> dict[str, Any]:
    """The identifying half of one trace summary; the numbers are added after."""
    entry: dict[str, Any] = {"signal": sig.name}
    if plan.stepped:
        entry["step"] = step
        if label != sig.name:
            entry["label"] = label
    entry["panel"] = panel
    return entry


def _compute_plot_spec(
    raw,
    plan: PlotPlan,
    max_points: int,
    *,
    summarize: bool = False,
) -> tuple[dict, dict]:
    """Build the renderer-ready plot spec + coverage facts (no I/O).

    Runs in a worker thread (heavy numpy). Returns ``(spec, facts)``: ``spec`` is
    the PlotSpec (panels/bode/analysis_type) consumed by the offline HTML file,
    the in-chat widget and the attached image (called at different point
    budgets); ``facts`` are the coverage facts the handler turns into
    observations on the event loop. With ``summarize``, ``facts["traces"]``
    holds one summary per plotted trace, read from the full-resolution window
    before any decimation.
    """
    analysis_type = plan.analysis_type
    is_ac = analysis_type == "ac"
    steps_to_plot = plan.steps
    step_dicts = plan.step_dicts
    ts, te = plan.ts, plan.te
    x_label = plan.x_label
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

    empty_steps: set[int] = set()
    non_finite = 0
    downsampled = False
    points_per_series: list[int] = []
    phase_warnings: list[str] = []
    traces: list[dict[str, Any]] = []
    panels: list[dict] = []
    unioned = False
    win_lo: float | None = None
    win_hi: float | None = None

    def _track_window(x: np.ndarray) -> None:
        nonlocal win_lo, win_hi
        lo0, hi0 = float(x[0]), float(x[-1])
        win_lo = lo0 if win_lo is None else min(win_lo, lo0)
        win_hi = hi0 if win_hi is None else max(win_hi, hi0)

    def _no_samples() -> ResultError:
        return ResultError(
            "The [t_start, t_end] window selects no samples"
            + (f" in any of the {len(steps_to_plot)} steps." if multi else ".")
        )

    annotate_freq: np.ndarray | None = None
    annotate_h: np.ndarray | None = None
    # Single-trace annotation: capture the full-resolution complex response
    # BEFORE any downsampling so the corner reading runs on every sample.
    single_trace = is_ac and plan.annotate and len(plan.signals) == 1 and len(steps_to_plot) == 1
    one_group = len(plan.groups) == 1

    for group in plan.groups:
        title = _group_title(group, plan.units)
        panel_index = len(panels)
        if is_ac:
            mag_series: list[tuple[np.ndarray, np.ndarray, str]] = []
            phase_series: list[tuple[np.ndarray, np.ndarray, str]] = []
            for col in group:
                for step in steps_to_plot:
                    axis = guarded_axis(raw, step)
                    lo, hi = window_indices(axis, ts, te)
                    if lo >= hi:
                        empty_steps.add(step)
                        continue
                    wave = col.wave(raw, step)[lo:hi]
                    freq, h = prepare_ac_arrays(axis[lo:hi], wave)
                    if single_trace:
                        annotate_freq, annotate_h = freq, h
                    mag = safe_magnitude_db(h)
                    phase, warns = unwrap_phase_safe(h)
                    phase_warnings.extend(warns)
                    non_finite += int(np.count_nonzero(~np.isfinite(mag)))
                    non_finite += int(np.count_nonzero(~np.isfinite(phase)))
                    label = _label(col.name, step)
                    if summarize:
                        entry = _summary_entry(plan, col, step, label, panel_index)
                        entry["unit"] = "dB"
                        entry.update(summarize_trace(freq, mag, time_weighted_mean=False))
                        ends = [float(phase[i]) for i in (0, -1)]
                        entry["phase_initial_deg"] = ends[0] if math.isfinite(ends[0]) else None
                        entry["phase_final_deg"] = ends[1] if math.isfinite(ends[1]) else None
                        traces.append(entry)
                    if len(freq) > max_points:
                        downsampled = True
                        f_ds, mag = downsample_minmax(freq, mag, max_points)
                        _, phase = downsample_minmax(freq, phase, max_points)
                        freq = f_ds
                    _track_window(freq)
                    points_per_series.append(len(freq))
                    mag_series.append((freq, mag, label))
                    phase_series.append((freq, phase, label))
            if not mag_series:
                raise _no_samples()
            suffix = "" if one_group else f" — {title}"
            mag_panel, u1 = _union_panel(mag_series, "log", x_label, f"Magnitude (dB){suffix}")
            phase_panel, u2 = _union_panel(phase_series, "log", x_label, f"Phase (deg){suffix}")
            unioned = unioned or u1 or u2
            panels.extend([mag_panel, phase_panel])
        else:
            plot_series: list[tuple[np.ndarray, np.ndarray, str]] = []
            for col in group:
                for step in steps_to_plot:
                    axis = guarded_axis(raw, step)
                    lo, hi = window_indices(axis, ts, te)
                    if lo >= hi:
                        empty_steps.add(step)
                        continue
                    axis_w = axis[lo:hi]
                    wave = col.wave(raw, step)[lo:hi]
                    if np.iscomplexobj(wave):
                        # Defensive: a stray complex trace in a non-AC raw.
                        wave = np.real(wave)
                    non_finite += int(np.count_nonzero(~np.isfinite(wave)))
                    label = _label(col.name, step)
                    if summarize:
                        entry = _summary_entry(plan, col, step, label, panel_index)
                        entry["unit"] = plan.units[col.name]
                        entry.update(
                            summarize_trace(
                                axis_w, wave, time_weighted_mean=analysis_type == "transient"
                            )
                        )
                        traces.append(entry)
                    x_arr, y_arr = axis_w, wave
                    if len(y_arr) > max_points:
                        downsampled = True
                        x_arr, y_arr = downsample_minmax(axis_w, wave, max_points)
                    _track_window(x_arr)
                    points_per_series.append(len(y_arr))
                    plot_series.append((x_arr, y_arr, label))
            if not plot_series:
                raise _no_samples()
            panel, u = _union_panel(
                plot_series, "log" if plan.x_is_log else "linear", x_label, title
            )
            unioned = unioned or u
            panels.append(panel)

    spec: dict[str, Any] = {"analysis_type": analysis_type, "bode": is_ac, "panels": panels}
    if single_trace and annotate_freq is not None and annotate_h is not None:
        annotations, nmp = _ac_annotations(annotate_freq, annotate_h)
        spec["annotations"] = annotations
        spec["nmp"] = nmp

    series_count = sum(len(p["series"]) for p in spec["panels"])
    facts = {
        "panels": len(spec["panels"]),
        "series_count": series_count,
        "points_per_series": points_per_series,
        "downsampled": downsampled,
        "unioned": unioned,
        "empty_steps": sorted(empty_steps),
        "non_finite": non_finite,
        "phase_unwrapped": is_ac,
        "phase_warnings": phase_warnings,
        "window_used": [win_lo, win_hi] if win_lo is not None else [],
        "step_values_available": (bool(step_dicts) if multi else None),
        "traces": traces,
    }
    return spec, facts


def build_plot_file(
    raw,
    raw_path: Path,
    plan: PlotPlan,
    max_points: int,
    out_path: Path,
    title: str,
) -> dict:
    """Compute the spec, assemble the offline HTML, write it atomically; return facts.

    Runs in a worker thread (heavy numpy + HTML build + file I/O). The handler
    turns the returned facts into observations on the event loop (the concurrency
    contract keeps response building off worker threads). The facts carry the
    per-trace summaries.
    """
    spec, facts = _compute_plot_spec(raw, plan, max_points, summarize=True)
    summary = f"{raw_path.stem} — {plan.analysis_type}: {facts['series_count']} series"
    html_str = build_plot_html(spec, title=title, summary=summary)
    with atomic_write(out_path) as f:
        f.write(html_str)
    return facts


def _compute_widget_spec_json(raw, plan: PlotPlan, max_points: int) -> str:
    """Build the compact widget chart spec and serialize it — all in the worker.

    Both the numpy spec build AND the (potentially large) JSON serialization run
    off the event loop. Returns the spec as a JSON string for the result ``_meta``
    (read by the widget in ``app.ontoolresult``); raises ``ResultError`` like
    :func:`_compute_plot_spec` (e.g. the cell cap), which the handler catches to
    fall back to local-open delivery.
    """
    spec, _ = _compute_plot_spec(raw, plan, max_points)
    return json.dumps(spec, ensure_ascii=True, allow_nan=False)


def build_plot_image(
    raw, plan: PlotPlan, max_points: int, title: str, png_path: Path
) -> RenderedImage:
    """Render the plot as a static image and, when it rasterized, write the PNG.

    Runs in a worker thread (numpy, SVG build, rasterization, file I/O). Draws
    the same panels as the chart, decimated to ``max_points`` per series. Without
    the raster extra the returned image is the SVG with a ``note`` and nothing is
    written; the handler reports that instead of passing markup to the model.
    """
    spec, _ = _compute_plot_spec(raw, plan, max_points)
    image = render_image(render_plot_svg(spec, title=title), image_format="png", scale=1.0)
    if image.is_raster:
        atomic_write_bytes(png_path, image.data, durable=False)
    return image


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _summary_lines(traces: list[dict[str, Any]], x_unit: str | None) -> list[str]:
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
        if "phase_final_deg" in t:
            parts.append(
                f"phase {_fmt(t['phase_initial_deg'])} to {_fmt(t['phase_final_deg'])} deg"
            )
        if t.get("non_finite"):
            parts.append(f"{t['non_finite']} non-finite samples left out")
        lines.append(f"  {name}{unit}, panel {t['panel']}: " + ", ".join(parts))
    return lines


class PlotWaveformInput(ToolInput):
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
            "Default: a '.ltspice-mcp/plots/' sidecar next to the circuit or "
            "the raw file."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


_NULLABLE_NUMBER: dict[str, Any] = {"type": ["number", "null"]}

_TRACE_SUMMARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "signal": {"type": "string"},
        "step": {"type": "integer"},
        "label": {"type": "string"},
        "panel": {"type": "integer"},
        "unit": {"type": ["string", "null"]},
        "min": _NULLABLE_NUMBER,
        "max": _NULLABLE_NUMBER,
        "x_at_min": _NULLABLE_NUMBER,
        "x_at_max": _NULLABLE_NUMBER,
        "initial": _NULLABLE_NUMBER,
        "final": _NULLABLE_NUMBER,
        "mean": _NULLABLE_NUMBER,
        "phase_initial_deg": _NULLABLE_NUMBER,
        "phase_final_deg": _NULLABLE_NUMBER,
        "non_finite": {"type": "integer"},
    },
    "required": [
        "signal",
        "panel",
        "unit",
        "min",
        "max",
        "x_at_min",
        "x_at_max",
        "initial",
        "final",
    ],
    "additionalProperties": False,
}

_IMAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "image_format": {"type": "string"},
        "mime_type": {"type": "string"},
        "scale": _NULLABLE_NUMBER,
        "width": {"type": ["integer", "null"]},
        "height": {"type": ["integer", "null"]},
        "bytes": {"type": "integer"},
        "estimated_tokens": {"type": ["integer", "null"]},
        "note": {"type": ["string", "null"]},
    },
    "required": ["image_format", "width", "height", "bytes"],
}


@registry.tool(
    name="plot_waveform",
    title="Plot Waveforms",
    description=(
        "Render an interactive chart (zoom/pan/hover) of one or more signals for "
        "a person to look at. The chart type follows the run (transient, DC sweep, "
        "AC Bode, noise), a .step or Monte Carlo run overlays every step, and each "
        "unit gets its own panel.\n\n"
        "Writes a self-contained HTML file and returns its path — into "
        "``out_dir`` if given, else a '.ltspice-mcp/plots/' sidecar next to the "
        "circuit or the raw. On a host that supports MCP Apps the chart is also "
        "embedded as an in-chat widget; otherwise it opens in your local "
        "browser. The reply summarizes each trace (min and max and where, first "
        "and final value, mean on a transient); attach_plot adds a PNG.\n\n"
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
            "x_unit": {"type": ["string", "null"]},
            "traces": {"type": "array", "items": _TRACE_SUMMARY_SCHEMA},
            "traces_total": {"type": "integer"},
            "delivery": {"type": "string", "enum": ["terminal", "ui"]},
            "opened": {"type": "boolean"},
            "opener": {"type": ["string", "null"]},
            "image": _IMAGE_SCHEMA,
            "image_path": {"type": "string"},
            "observations": OBSERVATIONS_SCHEMA,
        },
    },
)
async def handle_plot_waveform(args: PlotWaveformInput, state: SessionState):
    case = await _experiment_case(args.raw_file, args.job_id, args.run_index, args.case_id, state)
    if case is not None:
        raw_path = case.raw
        run_index = case.identity["run_index"]
        netlist: Path | None = case.netlist
    else:
        source = _direct_source(args.raw_file, args.job_id, state)
        raw_path = source.raw
        run_index = args.run_index
        netlist = source.netlist
    fmt = args.format
    if isinstance(args.signals, list) and not args.signals:
        raise ResultError("Pass at least one signal, or 'all'.")
    if args.panels is not None and args.signals != "all":
        raise ResultError("Pass signals or panels, not both: panels names the signals it plots.")

    raw = await services.load_raw(raw_path, state)
    # A .op raw has no sweep axis to plot — refuse early with the clean pointer.
    guarded_axis(raw, 0, raw_path)

    _, analysis_type, _, x_is_log = classify_analysis(raw)

    trace_names = raw.get_trace_names()
    axis_name = trace_names[0]

    def _resolve(name: str) -> services.Signal:
        sig = services.resolve_signal(raw, name)
        if sig.name == axis_name:
            raise ResultError(f"{name!r} is the sweep axis, not a signal column.")
        return sig

    cols: list[services.Signal]
    explicit: list[list[services.Signal]] | None = None
    if args.panels is not None:
        explicit = []
        panel_of: dict[str, int] = {}
        for index, names in enumerate(args.panels):
            group: list[services.Signal] = []
            for name in names:
                sig = _resolve(name)
                if sig.name in panel_of:
                    if panel_of[sig.name] != index:
                        raise ResultError(
                            f"{sig.name!r} is named in more than one panel; give each "
                            "signal one panel."
                        )
                    continue
                panel_of[sig.name] = index
                group.append(sig)
            explicit.append(group)
        cols = [sig for group in explicit for sig in group]
    elif args.signals == "all":
        cols = [services.Signal(name, name) for name in trace_names[1:]]
    else:
        seen: set[str] = set()
        cols = []
        for s in args.signals:
            sig = _resolve(s)
            if sig.name not in seen:
                seen.add(sig.name)
                cols.append(sig)
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
    open_locally = state.config.open_plot if args.open is None else args.open
    attach = state.config.attach_plot if args.attach_plot is None else args.attach_plot

    input_noise_unit = await resolve_input_noise_unit(analysis_type, cols, netlist)
    plan, unverified_noise = plan_plot(
        raw,
        cols,
        steps=steps_to_plot,
        step_dicts=step_dicts,
        analysis_type=analysis_type,
        x_is_log=x_is_log,
        ts=ts,
        te=te,
        panels=explicit,
        annotate=args.annotate,
        input_noise_unit=input_noise_unit,
    )
    _, x_unit = _x_axis(raw, analysis_type)

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
            build_plot_file, raw, raw_path, plan, max_points, out_path, title
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
                _compute_widget_spec_json, raw, plan, budget
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

    # The model's own frame: a static PNG of the same panels, on request.
    image: RenderedImage | None = None
    image_path: Path | None = None
    image_problem: str | None = None
    if attach:
        png_path = out_path.with_suffix(".png")
        try:
            rendered = await asyncio.to_thread(
                build_plot_image,
                raw,
                plan,
                min(max_points, _IMAGE_MAX_POINTS),
                title,
                png_path,
            )
        # An optional add-on: the chart and its numbers are already built, so a
        # render failure is reported beside them rather than failing the call.
        except Exception as e:
            image_problem = f"The image could not be rendered ({type(e).__name__}: {e})."
        else:
            if rendered.is_raster:
                image, image_path = rendered, png_path
            else:
                # SVG path data as text is no use to a model and costs far more
                # than the picture would, so the fallback is reported, not sent.
                image_problem = (
                    "No image attached: PNG rendering needs the optional raster "
                    "extra (pip install 'ltspice-mcp[raster]'). " + (rendered.note or "")
                ).strip()

    # No widget (terminal host, or a UI build that fell back) → open the file.
    opened, opener = False, None
    if widget_spec_json is None and open_locally:
        opened, opener = await asyncio.to_thread(desktop.open_in_desktop, out_path)

    traces_all: list[dict[str, Any]] = facts["traces"]
    traces = traces_all[:_TRACE_SUMMARY_MAX]

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
    if len(traces_all) > len(traces):
        observations.append(
            {
                "code": "trace_summary_truncated",
                "kind": "coverage",
                "detail": (
                    f"Summarized {len(traces)} of {len(traces_all)} plotted traces; "
                    "analyze_results' signal_stats reads any of the rest."
                ),
            }
        )
    if unverified_noise:
        observations.append(
            {
                "code": "noise_input_unit_unverified",
                "kind": "value",
                "detail": (
                    f"{', '.join(unverified_noise)} carries the simulator's declared "
                    "unit; no .NOISE line was found to check it against, and LTspice "
                    "declares input-referred noise as a voltage even when the input "
                    "source is a current source (then it is A/√Hz)."
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
    if facts["phase_unwrapped"]:
        observations.append(
            {
                "code": "phase_unwrapped",
                "kind": "value",
                "detail": (
                    "Bode phase is unwrapped for a readable continuous curve — this differs "
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
                    "union x, and each is drawn through its own samples only."
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
    if widget_spec_json is not None:
        observations.append(
            {
                "code": "widget_delivered",
                "kind": "coverage",
                "detail": (
                    "Client advertises MCP Apps (ui://) support; the chart spec is in "
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
        if not open_locally:
            why = "open=false" if args.open is False else "[analysis] open_plot = false"
            observations.append(
                {
                    "code": "open_skipped",
                    "kind": "coverage",
                    "detail": f"Local open skipped ({why}); open the returned path manually.",
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

    data: dict[str, Any] = {
        "path": str(out_path),
        "analysis_type": analysis_type,
        "signals": [sig.name for sig in cols],
        "n_steps": n_steps,
        "steps_plotted": len(steps_to_plot),
        "panels": facts["panels"],
        "series_count": facts["series_count"],
        "points_per_series": facts["points_per_series"],
        "max_points": max_points,
        "downsampled": facts["downsampled"],
        "window_used": facts["window_used"],
        "x_unit": x_unit,
        "traces": traces,
        "traces_total": len(traces_all),
        "delivery": "ui" if widget_spec_json is not None else "terminal",
        "opened": opened,
        "opener": opener,
        "observations": observations,
    }
    if image is not None and image_path is not None:
        data["image"] = image.to_dict()
        data["image_path"] = str(image_path)
    if widget_spec_json is not None:
        head = (
            f"Rendered an interactive {analysis_type} plot widget in-chat (also wrote {out_path})"
        )
    else:
        head = f"Wrote interactive {analysis_type} plot to {out_path}"
        if opened:
            head += f" (opened with {opener})"
    lines = [head]
    if image_path is not None and image is not None:
        lines.append(f"Attached a PNG of the chart ({image.width}x{image.height}, {image_path}).")
    lines.append(f"Traces ({len(traces)} of {len(traces_all)}):")
    lines.extend(_summary_lines(traces, x_unit))
    lines.extend(format_observations(observations))
    result = format_response("\n".join(lines), data, fmt)

    if image is not None:
        result.content.append(
            types.ImageContent(
                type="image",
                data=base64.b64encode(image.data).decode("ascii"),
                mime_type=image.mime_type,
            )
        )
    if widget_spec_json is not None:
        # Pipe the compact chart spec through the result _meta (a non-model-visible
        # channel) as a JSON string. The MCP Apps host forwards the full result to
        # the widget (declared via the tool's ui.resourceUri), where
        # ``app.ontoolresult`` reads _meta and renders it — the model reads the
        # trace summaries and the path, not the chart data.
        result.meta = {WIDGET_SPEC_META_KEY: widget_spec_json}
    return result
