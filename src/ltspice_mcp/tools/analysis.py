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
import bisect
import csv
import json
import math
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np
from mcp import types
from pydantic import Field
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import AnalysisDeadlineExceeded, NetlistError, ResultError
from ltspice_mcp.lib import atomic_write, desktop, metrics, services
from ltspice_mcp.lib.ac_analysis import (
    CrossingWithQuantity,
    FilterMetricsOutput,
    GainAtPoint,
    Quantity,
    ResonancesOutput,
    ReturnLossOutput,
    RollOffOutput,
    SearchDirection,
    StabilityMetricsOutput,
    prepare_ac_arrays,
    unwrap_phase_safe,
)
from ltspice_mcp.lib.ac_structure import AcStructureResult, analyze_ac_structure
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.job_store import SIDECAR_DIRNAME
from ltspice_mcp.lib.log_parser import parse_measurements, parse_step_iterations
from ltspice_mcp.lib.metrics import (
    AggregatedField,
    classify_analysis,
    guarded_axis,
    parse_time,
    snap_match,
    window_indices,
)
from ltspice_mcp.lib.plot_html import (
    WIDGET_RESOURCE_URI,
    WIDGET_SPEC_META_KEY,
    build_plot_html,
)
from ltspice_mcp.lib.raw_parser import (
    OperatingPointOutput,
    dc_axis_name,
    detect_sim_type,
    get_step_count,
    nearest_index,
    real_axis,
    safe_magnitude_db,
    sample_to_dict,
)
from ltspice_mcp.lib.recipes import (
    AcStructureRecipe,
    EdgesRecipe,
    Levels,
    MeasurementsRecipe,
    NoiseIntegralRecipe,
    OperatingPointRecipe,
    PeriodicRecipe,
    ResonanceRecipe,
    ReturnLossRecipe,
    SignalStatsRecipe,
    StabilityRecipe,
    SummaryRecipe,
    ThdRecipe,
    TimingEndpoint,
    TimingRecipe,
    Window,
)
from ltspice_mcp.lib.signal_analysis import (
    EdgeMetricsOutput,
    MeasurementStatsEntry,
    PeriodicMetricsOutput,
    ThdOutput,
    TimingBetweenOutput,
    downsample_minmax,
)
from ltspice_mcp.state import ExperimentJob, SessionState, legacy_record_message
from ltspice_mcp.tools._base import (
    FORMAT_DESCRIPTION,
    MEAS_ERRORS_SCHEMA,
    MEASUREMENTS_SCHEMA,
    OBSERVATIONS_SCHEMA,
    SUGGESTIONS_SCHEMA,
    WARNINGS_SCHEMA,
    ToolInput,
    declare_output_schema,
    format_meas_errors,
    format_observations,
    format_response,
    registry,
    result_text,
    safe_path,
    schema_from_typeddict,
)

FormatField = Literal["json", "text"] | None

# ``signal`` field description shared by the transient/point analysis tools whose
# signal argument also accepts a device operating-point shorthand for an ngspice
# ``.save``'d parameter. Kept in one place so the three tools stay consistent.
_OP_SIGNAL_FIELD_DESC = (
    "Signal/trace name (e.g., 'V(out)', 'I(R1)'), or a device operating-point "
    "shorthand for an ngspice .save'd parameter: 'm1.gm' / 'm1.vth' "
    "(resolves to '@m1[gm]', incl. subcircuit paths like 'x1.m1.gm')."
)

# Shared note for AC-tool `signal` fields whose expression support is generic
# (all AC tools share _load_ac_signal); tools where the '-' has tool-specific
# meaning (loop-gain probe sense, reversed impedance probe) word it bespoke.
_AC_SIGNAL_EXPR_NOTE = (
    "Ratios ('V(out)/V(in)') and a leading '-' (180° phase flip) are accepted "
    "like the other AC tools."
)

# Fourier harmonics shown in the simulation_summary text before it truncates to a
# "... (N total)" line. The full list always remains in structuredContent.
_MAX_SUMMARY_HARMONICS = 10


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


# Header line for a rendered warnings block. Shared so _strip_warning_block
# (which splits rendered text on it) can't drift from what _warning_lines emits.
_WARNINGS_HEADER = "Warnings:"


def _warning_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["", _WARNINGS_HEADER, *(f"  - {w}" for w in warnings)]


def _window_note(data: dict, windowed: bool) -> str:
    """The ``(window [a, b] s)`` suffix a windowed transient stat line carries."""
    lo = data.get("t_start_used")
    hi = data.get("t_end_used")
    if not windowed or lo is None or hi is None:
        return ""
    return f" (window [{lo:.6g}, {hi:.6g}] s)"


class SignalStatsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``. Lets you summarize a sweep "
            "run the same way you'd summarize a standalone raw."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description=_OP_SIGNAL_FIELD_DESC)
    step: int = Field(default=0, description="Step index for .step directives")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start in SPICE notation (e.g. '1m', '100u'). Transient only. "
            "Strongly recommended when computing RMS or average — the startup "
            "transient otherwise biases the result. Rejected for AC analysis (time-windowing a frequency sweep is an error)."
        ),
    )
    t_end: str | None = Field(
        default=None,
        description="Window end in SPICE notation. Transient only; rejected for AC.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


def _window_arg(t_start: str | None, t_end: str | None) -> Window | None:
    """A pair of SPICE-notation bounds as the recipe window the metrics take."""
    if t_start is None and t_end is None:
        return None
    return Window(start=t_start, end=t_end)


def _source_for(args, state: SessionState) -> services.AnalysisSource:
    """The resolved source behind a direct ``raw_file``/``job_id`` call."""
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    return services.AnalysisSource.for_raw(raw_path)


def _ac_source_for(raw_file: str | Path, state: SessionState) -> services.AnalysisSource:
    """``_source_for`` for the AC adapters, whose raw is a path or a trusted Path.

    A ``Path`` is an already-resolved server artifact (a run's raw, resolved
    through the read model) and is used directly; a ``str`` is untrusted caller
    input and is validated through ``safe_path``.
    """
    raw_path = raw_file if isinstance(raw_file, Path) else safe_path(raw_file, state)
    return services.AnalysisSource.for_raw(raw_path)


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


def _effective_raw_path(
    raw_file: str | None, job_id: str | None, run_index: int, state: SessionState
) -> Path:
    """The .raw a direct call reads, from EITHER a user ``raw_file`` OR a job run."""
    del run_index  # a job's runs are case-addressed; this route resolves neither
    return _direct_source(raw_file, job_id, state).raw


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
    job_id is left to ``_effective_raw_path``'s exclusivity error.
    """
    job = await services.resolve_job_async(job_id, state) if job_id and not raw_file else None
    if isinstance(job, ExperimentJob):
        return services.experiment_run_context(job, state, run_index=run_index, case_id=case_id)
    if case_id is not None:
        raise ResultError(
            "case_id selects a run_experiments case; pass it with that job's job_id."
        )
    return None


def _run_meta(job_id: str | None, run_index: int, state: SessionState) -> dict | None:
    """Identify which job run an analysis addressed: ``{run_index, params}``.

    ``None`` for a direct ``raw_file`` — there is no run to name. A consolidated
    caller resolves the run itself and injects the source, so the assignments
    travel on the injected identity rather than being looked up again here.
    """
    if not job_id:
        return None
    return {"run_index": run_index, "params": {}}


async def _resolve_artifact_dest(
    *,
    out_dir: str | None,
    job_id: str | None,
    raw_file: str | None,
    subdir: str,
    filename: str,
    artifact: str,
    state: SessionState,
    circuit_dir: Path | None = None,
) -> Path:
    """Resolve where a generated artifact (CSV / HTML) is written.

    An explicit ``out_dir`` (validated via ``safe_path``) wins; otherwise a
    Linux-side ``.ltspice-mcp/<subdir>/`` sidecar next to the CIRCUIT for a
    job_id, or next to the raw for a raw_file — a job-run raw can live in a
    Windows temp under /mnt/c the client cannot Read, so the job path anchors on
    the circuit. A caller that already resolved the circuit (an experiment case,
    whose job has no single netlist) passes it as ``circuit_dir``.
    Server-artifact paths skip ``safe_path`` except the out_dir
    override; the resolved path must stay under its anchor (a symlinked sidecar
    would otherwise redirect the write out).
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
        # Sidecar next to the anchor — but if the anchor is already inside a
        # .ltspice-mcp/ tree (e.g. a job-run raw passed by path), write the
        # subdir there directly rather than nesting another sidecar
        # (…/.ltspice-mcp/runs/.ltspice-mcp/waveforms/…).
        rel = subdir if SIDECAR_DIRNAME in dest_anchor.parts else f"{SIDECAR_DIRNAME}/{subdir}"
        out_path = (dest_anchor / rel / filename).resolve()
    if not out_path.is_relative_to(dest_anchor.resolve()):
        raise ResultError(
            f"Refusing to write the {artifact} outside the destination directory "
            "(a symlinked .ltspice-mcp/ sidecar would redirect it)."
        )
    return out_path


class QueryValueInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``. Lets you query a sweep run "
            "the same way you'd query a standalone raw."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description=_OP_SIGNAL_FIELD_DESC)
    at: str | None = Field(
        default=None,
        description=(
            "Value on the run's primary sweep axis in SPICE notation (e.g., '1m', "
            "'100u', '1G', '2.5k'): time for .tran, frequency for .AC, or the .dc "
            "sweep variable (e.g. '27' for a `.dc temp` sweep). The nearest data "
            "point is returned without interpolation. Required unless ``step_axis`` "
            "is given (then it picks the inner-axis point within the chosen step; "
            "optional)."
        ),
    )
    step: int = Field(
        default=0,
        description="Step index for .step directives (ignored when ``step_axis`` is used).",
    )
    step_axis: str | None = Field(
        default=None,
        description=(
            "Select a run of a stepped (.step) sweep by its parameter VALUE instead "
            "of an index: the parameter name (e.g. 'temp', 'Rval'). Pair with "
            "``step_value``. The nearest step is chosen and flagged with "
            "``exact_match``. NOT for a bare non-stepped .dc/.ac sweep, where the "
            "swept variable is the run's primary axis — query it directly with "
            "``at`` instead (e.g. ``at='27'`` on a `.dc temp` sweep)."
        ),
    )
    step_value: str | None = Field(
        default=None,
        description="Target value of ``step_axis`` in SPICE notation (e.g. '27', '1k'). "
        "Required when ``step_axis`` is given.",
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


class OperatingPointInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Read the operating point of a specific run of a completed sweep/MC "
            "(or single) job instead of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    step: int = Field(
        default=0,
        description=(
            "Step index for stepped .OP runs (e.g. ``.step temp ...`` + ``.op``). "
            "Default 0 returns the first step. Out-of-range values raise a "
            "structured error rather than silently returning the wrong step."
        ),
    )
    at: str | None = Field(
        default=None,
        description=(
            "For a .dc sweep raw: the sweep-axis VALUE to read the full bias "
            "snapshot at (SPICE notation, e.g. '2.5', '1.2'). Nearest point is "
            "used. Default reads the sweep's first point; ignored for plain .op "
            "runs (no sweep axis)."
        ),
    )
    device: str | None = Field(
        default=None,
        description=(
            "Narrow the result to one device: its operating-point params (@dev[param]) "
            "and its terminal currents (e.g. Id/Ig/Is(M1)), each typed with its "
            "unit. Pass the device reference (e.g. 'M1', 'Q2', or a subcircuit "
            "path 'x1.mn'); LTspice subcircuit semiconductors are matched by "
            "instance regardless of the log's colon-qualified name. Default "
            "returns the whole circuit."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


class SimulationSummaryInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Summarize a specific run of a completed sweep/MC (or single) job "
            "instead of a raw_file path; pair with ``run_index``. The .log is "
            "taken from beside the run's raw unless ``log_file`` is given."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to summarize when ``job_id`` is given (default 0).",
    )
    log_file: str | None = Field(
        default=None,
        description=(
            "Optional path to .log file. Defaults to the resolved raw with the "
            "extension swapped to ``.log`` — pass an explicit value only if "
            "the log lives somewhere unusual."
        ),
    )
    signal: str | None = Field(
        default=None,
        description="Signal for AC bandwidth metrics (e.g., 'V(outp)'). Required for AC analysis.",
    )
    step: int = Field(
        default=0,
        description=(
            "Step index for ac_bandwidth_metrics on a stepped (.step) run. "
            "Default 0 (first step). On a multi-step run the metric is computed "
            "for this step only — a warning notes it."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            "analysis_type": {"type": "string"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "mean": {"type": "number"},
            "rms": {"type": "number"},
            "std": {"type": "number"},
            "abs_mean": {"type": "number"},
            "peak_to_peak": {"type": "number"},
            "point_count": {"type": "integer"},
            # Transient-only window metadata
            "t_start_used": {"type": ["number", "null"]},
            "t_end_used": {"type": ["number", "null"]},
            "duration": {"type": ["number", "null"]},
            # Time OF the min/max sample (transient only) — not window bounds,
            # which are t_start_used/t_end_used above
            "t_at_min": {"type": "number"},
            "t_at_max": {"type": "number"},
            # DC-sweep window metadata (axis is the swept variable, not time)
            "sweep_start_used": {"type": ["number", "null"]},
            "sweep_end_used": {"type": ["number", "null"]},
            "sweep_span": {"type": ["number", "null"]},
            # Noise-only window metadata (axis is frequency)
            "freq_start_used": {"type": ["number", "null"]},
            "freq_end_used": {"type": ["number", "null"]},
            # AC-only fields
            "min_db": {"type": "number"},
            "max_db": {"type": "number"},
            "mean_db": {"type": "number"},
            "min_phase": {"type": "number"},
            "max_phase": {"type": "number"},
            "observations": OBSERVATIONS_SCHEMA,
            "warnings": WARNINGS_SCHEMA,
        },
    }
)
async def handle_signal_stats(args: SignalStatsInput, state: SessionState):
    data = await metrics.signal_stats(
        _source_for(args, state),
        SignalStatsRecipe(
            key="signal_stats",
            metric="signal_stats",
            signal=args.signal,
            window=_window_arg(args.t_start, args.t_end),
        ),
        args.step,
        state,
    )
    signal = data["signal"]
    warnings = data.get("warnings", [])
    if data["analysis_type"] == "ac":
        lines = [
            f"Signal: {signal} (AC Analysis)",
            "",
            "Magnitude (dB):",
            f"  Min: {data['min_db']:.2f} dB",
            f"  Max: {data['max_db']:.2f} dB",
            f"  Mean: {data['mean_db']:.2f} dB",
            "",
            "Phase:",
            f"  Min: {data['min_phase']:.2f} deg",
            f"  Max: {data['max_phase']:.2f} deg",
            "",
            f"Data Points: {data['point_count']}",
        ]
        lines += [f"\u26a0 {w}" for w in warnings]
        return format_response("\n".join(lines), data, args.format)

    windowed = args.t_start is not None or args.t_end is not None
    lines = [
        f"Signal: {signal}{_window_note(data, windowed)}",
        f"Min:          {data['min']:.6g}",
        f"Max:          {data['max']:.6g}",
        f"Peak-to-Peak: {data['peak_to_peak']:.6g}",
    ]
    # Mean / abs-mean are present for transient and DC sweeps; the Noise branch
    # omits them on purpose (a plain mean of spectral density is dominated by
    # sample clustering and the sweep span, not the circuit), so guard them.
    if "mean" in data:
        lines.append(f"Mean:         {data['mean']:.6g}")
    if "rms" in data:
        lines.append(f"RMS:          {data['rms']:.6g}")
    if "std" in data:
        lines.append(f"Std:          {data['std']:.6g}")
    if "abs_mean" in data:
        lines.append(f"Abs mean:     {data['abs_mean']:.6g}")
    if "duration" in data:
        lines.append(f"Duration:     {data['duration']:.6g} s  ({data['point_count']} samples)")
    elif "sweep_span" in data:
        lines.append(f"Sweep span:   {data['sweep_span']:.6g}  ({data['point_count']} samples)")
    elif "freq_start_used" in data:
        lines.append(
            f"Frequency:    {data['freq_start_used']:.6g}..{data['freq_end_used']:.6g} Hz"
            f"  ({data['point_count']} samples)"
        )
    if data.get("observations"):
        lines += ["", *format_observations(data["observations"])]
    lines += [f"\u26a0 {w}" for w in warnings]
    return format_response("\n".join(lines), data, args.format)


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


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            # direct (at) path
            "requested_x": {"type": "number"},
            "actual_x": {"type": "number"},
            "value": {"type": "number"},
            "unit": {"type": "string"},
            "magnitude_db": {"type": "number"},
            "magnitude_linear": {"type": "number"},
            "phase_deg": {"type": "number"},
            # step_axis path (delegated to the step lookup); keys are optional
            "axis": {"type": "string"},
            "requested_value": {"type": "number"},
            "actual_value": {"type": "number"},
            "exact_match": {"type": "boolean"},
            "step_index": {"type": "integer"},
            "requested_at": {"type": "number"},
            "actual_at": {"type": "number"},
            "warnings": WARNINGS_SCHEMA,
        },
    }
)
async def handle_query_value(args: QueryValueInput, state: SessionState):
    """Query signal value at a specific time/frequency, or at a chosen sweep step."""
    # Step-by-axis-value mode folds in the former step_get tool. It selects a
    # step WITHIN a single .step raw, so it is raw_file-only — ``job_id`` already
    # selects the run, so the two selection mechanisms are mutually exclusive.
    if args.step_axis is not None:
        if args.job_id is not None:
            raise ResultError(
                "value recipe: 'step_axis' selects a step of a .step raw and can't be "
                "combined with 'job_id' (the run is already selected — pass 'at').",
                show_hint=False,
            )
        step_raw = args.raw_file
        if step_raw is None:
            raise ResultError("value recipe: 'step_axis' requires 'raw_file'.", show_hint=False)
        if args.step_value is None:
            raise ResultError(
                "value recipe: 'step_value' is required when 'step_axis' is given.",
                show_hint=False,
            )
        result = await handle_step_get(
            StepGetInput(
                raw_file=step_raw,
                axis=args.step_axis,
                value=args.step_value,
                signal=args.signal,
                at=args.at,
                format=args.format,
            ),
            state,
        )
        # The stepped read returns a fake 0.0 for an unrecognized @-param and a
        # real-looking value from a failed solve just like the direct path, so it
        # gets the same diagnostic relay. The handler resolved the signal name
        # into structuredContent; reuse it for the per-signal filter.
        step_source = services.AnalysisSource.for_raw(safe_path(step_raw, state))
        resolved = (result.structuredContent or {}).get("signal") or args.signal
        return _append_warnings_to_result(
            result, await metrics.signal_log_warnings(step_source, resolved), args.format
        )

    source = _source_for(args, state)
    data = await metrics.point_value(source, args.signal, args.at, args.step, state)
    signal = data["signal"]
    raw = await services.load_raw(source.raw, state)
    x_unit = metrics.query_x_label(raw, detect_sim_type(raw))
    unit_suffix = f" {data['unit']}" if data.get("unit") else ""

    req_x = float(data["requested_x"])
    act_x = float(data["actual_x"])
    snap_note = "" if data["exact_match"] else f"  (requested {req_x:.6g}, snapped)"
    if "magnitude_db" in data:
        lines = [
            f"Signal: {signal} at {x_unit}={req_x:.6g}",
            f"Requested: {req_x:.6g}",
            f"Nearest point: {act_x:.6g}{snap_note}",
            f"Magnitude: {data['magnitude_db']:.2f} dB ({data['magnitude_linear']:.6g})",
            f"Phase: {data['phase_deg']:.2f} deg",
        ]
    else:
        lines = [
            f"Signal: {signal} at {x_unit}={req_x:.6g}",
            f"Requested: {req_x:.6g}",
            f"Nearest point: {act_x:.6g}{snap_note}",
            f"Value: {data['value']:.6g}{unit_suffix}",
        ]
    lines += [f"\u26a0 {w}" for w in data.get("warnings", [])]
    return format_response("\n".join(lines), data, args.format)


def _format_measurements(
    measurements: dict, step_count: int, errors: list[str] | None = None
) -> str:
    """Format .MEAS results for display. Shared between handlers.

    Accepts the new structured shape (``{name: {"values": [...], ...}}``)
    where each entry may carry ``range_from`` / ``range_to`` / ``at`` metadata.
    """
    if not measurements:
        if errors:
            lines = ["No .MEAS results — errors in log:", ""]
            for err in errors:
                lines.append(f"  {err}")
            return "\n".join(lines)
        return "No .MEAS results found in log file"

    def _fmt_meta(value: object) -> str:
        # Per-step lists are summarised as ``[lo..hi]`` rather than
        # echoed in full — the per-step values already accompany them in
        # the entry's ``values`` field.
        if isinstance(value, list):
            nums = [v for v in value if isinstance(v, int | float)]
            if not nums:
                return "[…]"
            return f"[{min(nums):g}..{max(nums):g}]"
        if isinstance(value, int | float):
            return f"{value:g}"
        return str(value)

    def _meta_suffix(entry: dict) -> str:
        bits: list[str] = []
        if entry.get("range_from") is not None or entry.get("range_to") is not None:
            lo = entry.get("range_from")
            hi = entry.get("range_to")
            if lo is not None and hi is not None:
                bits.append(f"FROM={_fmt_meta(lo)} TO={_fmt_meta(hi)}")
            elif lo is not None:
                bits.append(f"FROM={_fmt_meta(lo)}")
            elif hi is not None:
                bits.append(f"TO={_fmt_meta(hi)}")
        if entry.get("at") is not None:
            bits.append(f"AT={_fmt_meta(entry['at'])}")
        return f"  ({', '.join(bits)})" if bits else ""

    if step_count <= 1:
        lines = [".MEAS Results:", ""]
        for name, entry in measurements.items():
            values = entry.get("values", [])
            value = values[0] if values else None
            suffix = _meta_suffix(entry)
            if value is None:
                lines.append(f"  {name} = FAILED{suffix}")
            else:
                lines.append(f"  {name} = {value:.6g}{suffix}")
    else:
        lines = [f".MEAS Results ({step_count} steps):", ""]
        for name, entry in measurements.items():
            values = entry.get("values", [])
            value_strs: list[str] = []
            for val in values:
                if val is None:
                    value_strs.append("FAILED")
                else:
                    value_strs.append(f"{val:.6g}")
            suffix = _meta_suffix(entry)
            lines.append(f"  {name}: [{', '.join(value_strs)}]{suffix}")

    return "\n".join(lines)


def _append_warnings_to_result(
    result: types.CallToolResult, extra: list[str], fmt: str | None
) -> types.CallToolResult:
    """Append warning strings to an already-built result, keeping the
    structuredContent ``warnings`` list and the text in sync: for ``fmt="json"``
    the text is a JSON snapshot of structuredContent, so it is re-dumped;
    otherwise the warnings are appended as ``⚠`` lines."""
    if not extra:
        return result
    sc = result.structuredContent
    if isinstance(sc, dict):
        existing = sc.get("warnings")
        sc["warnings"] = [*existing, *extra] if isinstance(existing, list) else list(extra)
    if fmt == "json":
        if isinstance(sc, dict) and result.content:
            block = result.content[0]
            if isinstance(block, types.TextContent):
                block.text = json.dumps(sc, indent=2)
    else:
        for block in result.content:
            if isinstance(block, types.TextContent):
                block.text += "\n" + "\n".join(f"⚠ {w}" for w in extra)
                break
    return result


def _rendered(lines: list[str], data: dict, fmt: str | None) -> types.CallToolResult:
    """Standard tail for a metric adapter: render the value's warnings block
    under the metric's own lines. The relay that fills ``warnings`` belongs to
    the metric function, so an adapter cannot forget it."""
    return format_response("\n".join(lines + _warning_lines(data.get("warnings", []))), data, fmt)


@declare_output_schema(output_model=OperatingPointOutput)
async def handle_operating_point(args: OperatingPointInput, state: SessionState):
    """Read DC operating point data (node voltages, branch currents, device operating point)."""
    op_data = await metrics.operating_point(
        _source_for(args, state),
        OperatingPointRecipe(key="operating_point", metric="operating_point", device=args.device),
        args.step,
        state,
        at=args.at,
    )
    units = op_data["units"]

    def _u(name: str) -> str:
        u = units.get(name)
        return f" {u}" if u else ""

    # The warnings the value carries in the order the metric appended them:
    # the simulator's own relayed lines first, then the empty-device-params
    # note, then the DC-sweep note. The text block below renders each where it
    # belongs rather than as one undifferentiated list.
    warnings = list(op_data["warnings"])
    dc_sweep_note = next(
        (w for w in warnings if w.startswith("DC sweep bias") or w.startswith("This is a DC")),
        None,
    )
    op_point_note = (
        metrics.NO_DEVICE_OP_POINTS_NOTE if metrics.NO_DEVICE_OP_POINTS_NOTE in warnings else None
    )
    relayed = [w for w in warnings if w is not dc_sweep_note and w != op_point_note]

    title = f"Operating Point — device {args.device}" if args.device else "DC Operating Point"
    lines = [title, ""]
    if dc_sweep_note:
        lines.append(f"\u26a0 {dc_sweep_note}")
        lines.append("")
    if op_data["step_count"] > 1:
        lines.append(
            f"Step {args.step} of {op_data['step_count']} (use step=N to read other "
            "iterations of stepped .OP runs)"
        )
        lines.append("")

    if op_data["voltages"]:
        lines.append("Node Voltages:")
        for name, value in op_data["voltages"].items():
            lines.append(f"  {name} = {value:.6g}{_u(name)}")
        lines.append("")

    if op_data["currents"]:
        lines.append("Branch Currents:")
        for name, value in op_data["currents"].items():
            lines.append(f"  {name} = {value:.6g}{_u(name)}")

    if op_data.get("device_op_points"):
        if op_data["voltages"] or op_data["currents"]:
            lines.append("")
        lines.append("Device Operating Point (@dev[param], e.g. gm/gds/vth/id):")
        for name, value in op_data["device_op_points"].items():
            lines.append(f"  {name} = {value:.6g}{_u(name)}")
    elif op_point_note:
        if op_data["voltages"] or op_data["currents"]:
            lines.append("")
        lines.append(f"\u26a0 {op_point_note}")

    for w in relayed:
        lines.append(f"\u26a0 {w}")

    return format_response("\n".join(lines), op_data, args.format)


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "sim_type": {"type": "string"},
            "range": {"type": "object"},
            "point_count": {"type": "integer"},
            "step_count": {"type": "integer"},
            "signals": {"type": "array", "items": {"type": "string"}},
            # Present only when the trace list was capped for the structured
            # channel; carries the TOTAL trace count. Full list stays
            # addressable via the spice://results/{job_id}/signals resource.
            "signals_truncated": {"type": "integer"},
            # Ambient / nominal temperature the simulator ran at (°C), when the
            # log records it — a provenance fact for temp-sensitive tasks.
            "temp_c": {"type": "number"},
            "tnom_c": {"type": "number"},
            "measurements": MEASUREMENTS_SCHEMA,
            "fourier": {"type": "array", "items": {"type": "object"}},
            "ac_bandwidth_metrics": {
                "type": "object",
                "properties": {
                    "bandwidth_3db": {"type": ["number", "null"]},
                    "unity_gain_freq": {"type": ["number", "null"]},
                    # Present only on a multi-step run: the step index these
                    # metrics were computed for (they are one step's answer).
                    "step": {"type": "integer"},
                },
            },
            "warnings": WARNINGS_SCHEMA,
            "errors": {"type": "array", "items": {"type": "string"}},
            "meas_errors": MEAS_ERRORS_SCHEMA,
            "failed_measurements": {"type": "array", "items": {"type": "string"}},
            "observations": OBSERVATIONS_SCHEMA,
            # Model-resolution help keyed by the missing model/subcircuit ref;
            # present only when the run's errors named unresolved refs.
            "suggestions": SUGGESTIONS_SCHEMA,
            # The trace ac_bandwidth_metrics was computed on when ``signal`` was
            # omitted and one was auto-picked; absent when the caller passed one.
            "ac_signal_used": {"type": "string"},
        },
    }
)
async def handle_simulation_summary(args: SimulationSummaryInput, state: SessionState):
    """Get comprehensive simulation summary."""
    source = _direct_source(args.raw_file, args.job_id, state)
    if args.log_file is not None:
        source = replace(source, log=safe_path(args.log_file, state))
    elif source.log is None:
        # Callers shouldn't have to pass both ``raw_file`` and the adjacent
        # ``.log``; derive the log path from the raw path when it's not given.
        source = replace(source, log=services.AnalysisSource.for_raw(source.raw).log)

    data = await metrics.summary(
        source,
        SummaryRecipe(key="summary", metric="summary"),
        args.step,
        state,
        signal=args.signal,
    )
    if args.format == "json":
        return format_response("", data, args.format)
    ac_metrics = data.get("ac_bandwidth_metrics")
    return format_response(_format_summary_text(data, ac_metrics), data, args.format)


def _format_summary_text(summary: dict, ac_metrics: dict | None) -> str:
    """Render the human-readable simulation summary from the computed facts.

    Presentation only: everything here is already carried in the structured
    ``json_data`` the handler returns alongside this text.
    """
    lines = [f"Simulation Summary: {summary['sim_type']}", ""]

    if "time_start" in summary["range"]:
        lines.append(
            f"Time span: {summary['range']['time_start']:.6g} to {summary['range']['time_end']:.6g}"
        )
    elif "freq_start" in summary["range"]:
        lines.append(
            f"Frequency range: {summary['range']['freq_start']:.6g} to {summary['range']['freq_end']:.6g}"
        )
    elif "sweep_start" in summary["range"]:
        lines.append(
            f"DC sweep: {summary['range']['sweep_start']:.6g} to {summary['range']['sweep_end']:.6g}"
        )

    lines.append(
        f"Data points: {summary['point_count']} per signal, {summary['step_count']} step(s)"
    )
    if "temp_c" in summary:
        tnom = summary.get("tnom_c")
        tnom_note = (
            f" (tnom {tnom:g} °C)" if tnom is not None and tnom != summary["temp_c"] else ""
        )
        lines.append(f"Temperature: {summary['temp_c']:g} °C{tnom_note}")
    lines.append("")

    total_signals = summary.get("signals_truncated", len(summary["signals"]))
    lines.append(f"Signals ({total_signals}):")
    for signal in summary["signals"]:
        lines.append(f"  - {signal}")
    if total_signals > len(summary["signals"]):
        lines.append(
            f"  ... and {total_signals - len(summary['signals'])} more "
            "(full list: the spice://results/{job_id}/signals resource)"
        )
    lines.append("")

    if "measurements" in summary:
        lines.append(
            _format_measurements(
                summary["measurements"],
                summary.get("step_count", 1),
            )
        )
        lines.append("")

    if summary.get("failed_measurements"):
        lines.append("FAIL'ed measurements (logged but did not trigger):")
        for name in summary["failed_measurements"]:
            lines.append(f"  {name}")
        lines.append("")

    if "fourier" in summary:
        lines.append("Fourier Analysis:")
        for fourier in summary["fourier"]:
            lines.append(f"  Signal: {fourier['signal']}")
            if fourier["thd"] is not None:
                lines.append(f"  THD: {fourier['thd']:.2f}%")
            if fourier["fundamental_frequency"] is not None:
                lines.append(f"  Fundamental: {fourier['fundamental_frequency']:.6g} Hz")
            if fourier["harmonics"]:
                lines.append("  Harmonics:")
                for harm in fourier["harmonics"][:_MAX_SUMMARY_HARMONICS]:
                    lines.append(
                        f"    {harm['number']}: {harm['frequency']:.6g} Hz, "
                        f"{harm['magnitude']:.6g}, {harm['phase']:.2f} deg"
                    )
                if len(fourier["harmonics"]) > _MAX_SUMMARY_HARMONICS:
                    lines.append(f"    ... ({len(fourier['harmonics'])} total)")
        lines.append("")

    if ac_metrics:
        lines.append("AC Bandwidth Metrics:")
        if ac_metrics["bandwidth_3db"] is not None:
            lines.append(f"  -3dB point: {ac_metrics['bandwidth_3db']:.6g} Hz")
        if ac_metrics["unity_gain_freq"] is not None:
            lines.append(f"  Unity-gain frequency: {ac_metrics['unity_gain_freq']:.6g} Hz")
        lines.append("")

    if "errors" in summary:
        lines.append(f"Errors ({len(summary['errors'])}):")
        for error in summary["errors"]:
            lines.append(f"  {error}")
        lines.append("")

    suggestion_block = services.format_suggestion_block(summary.get("suggestions"))
    if suggestion_block:
        lines.append(suggestion_block)
        lines.append("")

    if "warnings" in summary:
        lines.append(f"Warnings ({len(summary['warnings'])}):")
        for warning in summary["warnings"]:
            lines.append(f"  {warning}")
        lines.append("")

    meas_lines = format_meas_errors(summary.get("meas_errors", []))
    if meas_lines:
        lines.extend(meas_lines)
        lines.append("")

    obs_lines = format_observations(summary.get("observations", []))
    if obs_lines:
        lines.extend(obs_lines)
        lines.append("")

    return "\n".join(lines)


class EdgeMetricsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start time in SPICE notation (e.g. '1m', '100u'). Strongly "
            "recommended when the transient contains startup transients or "
            "multiple edges — otherwise the first edge in the full waveform is "
            "measured (often the power-up glitch)."
        ),
    )
    t_end: str | None = Field(default=None, description="Window end time in SPICE notation")
    edge: Literal["rising", "falling", "auto"] = Field(
        default="auto",
        description="Edge direction. 'auto' infers from window endpoints.",
    )
    edge_index: int = Field(
        default=0,
        description="Which matching edge in the window (0 = first). Use with tight t_start/t_end for determinism.",
    )
    low_pct: float = Field(default=10.0, description="Low threshold percent (default 10%)")
    high_pct: float = Field(default=90.0, description="High threshold percent (default 90%)")
    low_level: float | None = Field(
        default=None,
        description=(
            "Absolute low rail level, overriding auto-detection. Use when the "
            "auto estimate (mean of first/last 10%) is biased — e.g. a "
            "rise-from-rail where early samples cluster in the fast ramp, or a "
            "step that starts at t=0 from rest (no flat low rail to average). "
            "Pass low_level/high_level explicitly there."
        ),
    )
    high_level: float | None = Field(
        default=None,
        description="Absolute high rail level, overriding auto-detection.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class PulseResponseInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description="Window start — ideally the stimulus edge. Defaults to full transient.",
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    initial_value: float | None = Field(
        default=None,
        description="Pre-step steady value. Auto = mean of first 10% of window. Set explicitly if the start is contaminated by ringing.",
    )
    final_value: float | None = Field(
        default=None,
        description="Post-step steady value. Auto = mean of last 10% of window.",
    )
    settling_tolerance_pct: float = Field(
        default=2.0,
        description="Settling band as percent of |final - initial|. 2% is standard; 1% or 5% also common.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class DisturbanceResponseInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Regulated output node, e.g. 'V(vout)'")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description=(
            "Window start — set it at or just before the load-step edge. recovery_time "
            "is measured from t_start, and the baseline defaults to the mean of the "
            "leading 10% of the window (the pre-disturbance level). Defaults to full "
            "transient."
        ),
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    baseline: float | None = Field(
        default=None,
        description=(
            "Pre-disturbance output level. Auto = mean of the leading 10% of the "
            "window; set explicitly if the window doesn't start in steady state."
        ),
    )
    settle_band: float | None = Field(
        default=None,
        description=(
            "Recovery band as an absolute ± tolerance in signal units (e.g. 0.005 for "
            "±5 mV). Overrides settle_band_pct."
        ),
    )
    settle_band_pct: float = Field(
        default=2.0,
        description="Recovery band as percent of |baseline| when settle_band is not given (default 2%).",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class TimingBetweenInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal_a: str = Field(description="Reference signal (e.g. 'V(in)')")
    signal_b: str = Field(description="Delayed signal (e.g. 'V(out)'). delay = t_b - t_a.")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(default=None, description="Window start in SPICE notation")
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    threshold_a: float | None = Field(
        default=None,
        description="Absolute threshold for signal_a. If omitted, threshold_pct of signal_a's range is used.",
    )
    threshold_b: float | None = Field(
        default=None,
        description="Absolute threshold for signal_b. If omitted, threshold_pct of signal_b's range is used.",
    )
    threshold_pct: float = Field(
        default=50.0,
        description="Threshold percent applied PER SIGNAL (not shared) — asymmetric for CMOS with different rails.",
    )
    direction_a: Literal["rising", "falling"] = Field(default="rising")
    direction_b: Literal["rising", "falling"] = Field(default="rising")
    nth: int = Field(
        default=1,
        ge=1,
        description="1-based same-index threshold crossing to use for t_a/t_b/delay.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class PeriodicMetricsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal name (e.g. 'V(clk)')")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description="Window start — recommended to skip the startup transient.",
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation")
    threshold: float | None = Field(
        default=None,
        description="Absolute threshold level. Auto = midpoint of window min/max. For drifting signals, set explicitly.",
    )
    min_periods: int = Field(
        default=2,
        description="Minimum complete periods required; error if window has fewer.",
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


class MeasurementStatsInput(ToolInput):
    log_file: str | None = Field(
        default=None,
        description=(
            "Path to .log file from a single ``.step`` run that already "
            "concatenates every step's .MEAS results. For Monte Carlo / "
            "multi-run sweep jobs that emit one log per run, pass ``job_id`` "
            "instead and the aggregator walks every run's log."
        ),
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Job ID. For a sweep or Monte Carlo batch job the "
            "tool loads each completed run's log, concatenates the .MEAS "
            "results (one row per run), and aggregates. For a completed "
            "single-simulation job it aggregates that run's log (per-step "
            "values for a .step run). Mutually exclusive with ``log_file``."
        ),
    )
    measurement: str | None = Field(
        default=None,
        description="If given, stats for only this .MEAS; otherwise all measurements.",
    )
    histogram_bins: int = Field(
        default=10,
        description="Histogram bin count. Set to 0 to skip histogram computation.",
    )
    include_per_run: bool | None = Field(
        default=None,
        description=(
            "Whether to include the per-run {run_index, params} table for batch "
            "jobs. Default: included up to 100 runs, then truncated with "
            "per_run_truncated set to the full count. true = always the full "
            "table; false = omit it."
        ),
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


# ---------------------------------------------------------------------------
# Response TypedDicts — compose lib output + per-tool metadata (signal names)
# ---------------------------------------------------------------------------


class EdgeMetricsResponse(EdgeMetricsOutput):
    signal: str


class TimingBetweenResponse(TimingBetweenOutput):
    signal_a: str
    signal_b: str


class PeriodicMetricsResponse(PeriodicMetricsOutput):
    signal: str


class MeasurementStatsResponseEntry(MeasurementStatsEntry):
    """Per-measurement stats entry as returned by the MCP layer.

    Adds ``aggregated_field`` to the lib-level :class:`MeasurementStatsEntry`
    so callers can tell which per-run scalar (the trigger level or the
    WHEN-clause crossing point) the stats describe.
    """

    aggregated_field: AggregatedField
    # Present only on an n=1 read whose axis stayed on the level: the x
    # (time/frequency) the log reported this measurement at, echoed so a
    # single-run .meas with an AT/crossing still shows it (the variation-based
    # swap can't fire on one sample). Neutral fact — a WHEN crossing and a
    # FIND...AT probe point are indistinguishable at n=1, so it claims neither.
    at: NotRequired[float]


class MeasurementStatsResponse(TypedDict):
    stats: dict[str, MeasurementStatsResponseEntry]
    # Present only for a multi-run batch (job_id): per-run {run_index, params} in
    # the order the value lists use, so min_step_index/max_step_index name a corner.
    per_run: NotRequired[list[dict[str, Any]]]
    # Present when per_run was truncated to the row cap: the FULL run count.
    # Pass include_per_run=true for the whole table.
    per_run_truncated: NotRequired[int]
    # Measurement-level caveats, e.g. "batch still running: stats aggregate a
    # partial run set".
    warnings: NotRequired[list[str]]


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


@declare_output_schema(output_model=EdgeMetricsResponse)
async def handle_edge_metrics(args: EdgeMetricsInput, state: SessionState):
    data = await metrics.edges(
        _source_for(args, state),
        EdgesRecipe(
            key="edges",
            metric="edges",
            signal=args.signal,
            levels=Levels(low=args.low_level, high=args.high_level),
            edge=args.edge,
            window=_window_arg(args.t_start, args.t_end),
        ),
        args.step,
        state,
        edge_index=args.edge_index,
        low_pct=args.low_pct,
        high_pct=args.high_pct,
    )

    label = "Rise time" if data["is_rise_time"] else "Fall time"
    lines = [
        f"Edge Metrics: {args.signal} ({data['edge_direction']} edge "
        f"{args.edge_index} of {data['num_edges_in_window']})",
        "",
        f"{label} ({data['low_pct']:.0f}%-{data['high_pct']:.0f}%): "
        f"{data['transition_time']:.6g} s",
        f"Slew rate: {data['slew_rate']:.6g} (units/s)",
        f"Low level: {data['low_level']:.6g}",
        f"High level: {data['high_level']:.6g}",
        f"t(low): {data['t_low_crossing']:.6g} s",
        f"t(high): {data['t_high_crossing']:.6g} s",
        f"t(mid): {data['t_mid_crossing']:.6g} s",
    ]
    return _rendered(lines, data, args.format)


# Internal compute adapter — exposed publicly via transient_response(mode="step").
async def handle_pulse_response(args: PulseResponseInput, state: SessionState):
    data = await metrics.pulse_response(
        _source_for(args, state),
        args.signal,
        args.t_start,
        args.t_end,
        args.step,
        state,
        initial_value=args.initial_value,
        final_value=args.final_value,
        settling_tolerance_pct=args.settling_tolerance_pct,
    )

    # settling_time None has FOUR distinct meanings; keep them separate so an
    # unknown/unreliable state never reads as a definitive design failure:
    #   - full-pulse window     → metrics undefined (net step ~0 baseline)
    #   - trusted, never crossed → genuinely "never settled within the window"
    #   - untrusted final value  → UNKNOWN (trailing window too noisy to anchor a band)
    #   - in-band only near the end → UNKNOWN (final value from that same short tail)
    quality = data.get("quality", [])
    undefined = "net_step_small_vs_swing" in quality
    noisy_tail = "settling_final_value_from_noisy_tail" in quality
    short_dwell = "settling_dwell_near_window_end" in quality

    def _pct(value: float | None) -> str:
        return "undefined (full-pulse window)" if value is None else f"{value:.3f} %"

    if data["settling_time"] is not None:
        settle = f"{data['settling_time']:.6g} s"
    elif undefined:
        settle = "undefined (full-pulse window)"
    elif noisy_tail:
        settle = "unknown (final value from a still-ringing tail; pass final_value)"
    elif short_dwell:
        settle = (
            "unknown (in-band only near the window end; pass final_value, "
            "tighten t_start, or extend the window)"
        )
    else:
        settle = "never (within window)"
    lines = [
        f"Pulse Response: {args.signal} ({data['direction']} step)",
        "",
        f"Initial: {data['initial_value']:.6g}",
        f"Final:   {data['steady_state_value']:.6g}",
        f"Peak:    {data['peak_value']:.6g} at t={data['peak_time']:.6g} s",
        f"Overshoot:  {_pct(data['overshoot_pct'])}",
        f"Undershoot: {_pct(data['undershoot_pct'])}",
        f"Settling time (\u00b1{data['settling_tolerance_pct']:.2f}%): {settle}",
    ]
    if data.get("quality"):
        lines.append(f"Quality flags: {', '.join(data['quality'])}")
    return _rendered(lines, data, args.format)


# Internal compute adapter — exposed publicly via
# transient_response(mode="disturbance").
async def handle_disturbance_response(args: DisturbanceResponseInput, state: SessionState):
    data = await metrics.disturbance_response(
        _source_for(args, state),
        args.signal,
        args.t_start,
        args.t_end,
        args.step,
        state,
        baseline=args.baseline,
        settle_band=args.settle_band,
        settle_band_pct=args.settle_band_pct,
    )

    rec = data["recovery_time"]
    recovery = f"{rec:.6g} s" if rec is not None else "unavailable (see warnings)"
    lines = [
        f"Disturbance Response: {args.signal}",
        "",
        f"Baseline: {data['baseline']:.6g} ({data['baseline_source']})",
        f"Min: {data['min_value']:.6g} at t={data['min_time']:.6g} s",
        f"Max: {data['max_value']:.6g} at t={data['max_time']:.6g} s",
        f"Max droop:     {data['max_droop']:.6g}",
        f"Max overshoot: {data['max_overshoot']:.6g}",
        f"Recovery time (\u00b1{data['settle_band']:.3g}): {recovery}",
    ]
    if data.get("quality"):
        lines.append(f"Quality flags: {', '.join(data['quality'])}")
    return _rendered(lines, data, args.format)


@declare_output_schema(output_model=TimingBetweenResponse)
async def handle_timing_between(args: TimingBetweenInput, state: SessionState):
    data = await metrics.timing(
        _source_for(args, state),
        TimingRecipe.model_validate(
            {
                "key": "timing",
                "metric": "timing",
                "from": TimingEndpoint(
                    signal=args.signal_a, edge=args.direction_a, level=args.threshold_a
                ),
                "to": TimingEndpoint(
                    signal=args.signal_b, edge=args.direction_b, level=args.threshold_b
                ),
                "nth": args.nth,
                "window": _window_arg(args.t_start, args.t_end),
            }
        ),
        args.step,
        state,
        threshold_pct=args.threshold_pct,
    )

    lines = [
        f"Timing: {args.signal_a} ({data['direction_a']}) \u2192 "
        f"{args.signal_b} ({data['direction_b']})",
        "",
        f"t({args.signal_a}) = {data['t_a']:.6g} s @ threshold={data['threshold_a_used']:.6g}",
        f"t({args.signal_b}) = {data['t_b']:.6g} s @ threshold={data['threshold_b_used']:.6g}",
        f"Delay (t_b - t_a): {data['delay']:.6g} s",
    ]
    if data["pair_count"] > 1:
        lines.append(
            f"All {data['pair_count']} edge pairs: min {data['delay_min']:.6g} s "
            f"(at t={data['delay_min_at']:.6g}), max {data['delay_max']:.6g} s "
            f"(at t={data['delay_max_at']:.6g}), mean {data['delay_mean']:.6g} s"
        )
    return _rendered(lines, data, args.format)


@declare_output_schema(output_model=PeriodicMetricsResponse)
async def handle_periodic_metrics(args: PeriodicMetricsInput, state: SessionState):
    data = await metrics.periodic(
        _source_for(args, state),
        PeriodicRecipe(
            key="periodic",
            metric="periodic",
            signal=args.signal,
            window=_window_arg(args.t_start, args.t_end),
        ),
        args.step,
        state,
        threshold=args.threshold,
        min_periods=args.min_periods,
    )

    duty = f"{data['duty_cycle_pct']:.3f} %" if data["duty_cycle_pct"] is not None else "n/a"
    high_w = f"{data['pulse_width_high']:.6g} s" if data["pulse_width_high"] is not None else "n/a"
    low_w = f"{data['pulse_width_low']:.6g} s" if data["pulse_width_low"] is not None else "n/a"
    lines = [
        f"Periodic Metrics: {args.signal}",
        "",
        f"Period:      {data['period']:.6g} s",
        f"Frequency:   {data['frequency']:.6g} Hz",
        f"Duty cycle:  {duty}",
        f"High width:  {high_w}",
        f"Low width:   {low_w}",
        f"Jitter RMS:  {data['jitter_rms']:.6g} s",
        f"Threshold:   {data['threshold_used']:.6g}",
        f"Edges: {data['num_rising_edges']} rising / "
        f"{data['num_falling_edges']} falling "
        f"({data['num_periods_measured']} period(s))",
    ]
    return _rendered(lines, data, args.format)


class ThdInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw transient result. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a run of a completed sweep/MC (or single) job instead of a "
            "raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description="Signal to analyze (e.g. 'V(out)').")
    step: int = Field(default=0, description="Step index for .step sweeps")
    t_start: str | None = Field(
        default=None,
        description="Window start (SPICE notation) — skip the startup transient before measuring.",
    )
    t_end: str | None = Field(default=None, description="Window end in SPICE notation.")
    fundamental: str | None = Field(
        default=None,
        description=(
            "Fundamental frequency in SPICE notation (e.g. '1k'). Omit to "
            "auto-detect (largest FFT bin); pass it for an exact coherent result."
        ),
    )
    n_harmonics: int = Field(
        default=7, description="Harmonics 2..n folded into THD (1..50, default 7)."
    )
    window: Literal["coherent", "hann"] = Field(
        default="coherent",
        description=(
            "'coherent' trims to whole fundamental cycles + rectangular window "
            "(exact, no leakage); 'hann' analyzes the full window with a Hann "
            "taper (approximate — use when cycles can't be made integer)."
        ),
    )
    format: FormatField = Field(default=None, description="'json' or 'text'")


@declare_output_schema(output_model=ThdOutput)
async def handle_thd(args: ThdInput, state: SessionState):
    data = await metrics.thd(
        _source_for(args, state),
        ThdRecipe(
            key="thd",
            metric="thd",
            signal=args.signal,
            fundamental_hz=args.fundamental,
            harmonics=args.n_harmonics,
            window=_window_arg(args.t_start, args.t_end),
        ),
        args.step,
        state,
        window=args.window,
    )

    lines = [
        f"THD: {args.signal}",
        "",
        f"Fundamental: {data['fundamental_hz']:.6g} Hz ({data['fundamental_source']})",
        f"THD:    {data['thd_pct']:.4g} %  ({data['thd_db']:.2f} dB, ratio {data['thd_ratio']:.4g})",
        f"THD+N:  {data['thd_n_pct']:.4g} %  (ratio {data['thd_n_ratio']:.4g})",
        f"Window: {data['window']}, {data['n_cycles']:.4g} cycle(s), "
        f"{data['n_fft']}-pt FFT @ {data['fs_hz']:.6g} Hz",
        "",
        "Harmonics (relative to fundamental):",
    ]
    for h in data["harmonics"]:
        lines.append(f"  {h['n']}x ({h['frequency']:.6g} Hz): {h['db_rel']:.1f} dB")
    return _rendered(lines, data, args.format)


class NoiseIntegralInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to .raw .noise result. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Integrate a run of a completed sweep/MC (or single) job instead of a "
            "raw_file path; pair with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    signal: str | None = Field(
        default=None,
        description=(
            "Noise-density trace to integrate: 'V(onoise)'/'V(inoise)' (LTspice) "
            "or 'onoise_spectrum'/'inoise_spectrum' (ngspice). Default integrates "
            "the output noise (onoise)."
        ),
    )
    f_start: str | None = Field(
        default=None,
        description="Band start in SPICE notation (e.g. '20'); default = sweep start.",
    )
    f_end: str | None = Field(
        default=None, description="Band end (e.g. '20k'); default = sweep end."
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None, description="'json' or 'text'")


@declare_output_schema(
    {
        "type": "object",
        "properties": {
            "signal": {"type": "string"},
            "unit": {"type": "string"},
            "density_unit": {"type": "string"},
            "total_rms": {"type": "number"},
            "f_start_used": {"type": "number"},
            "f_end_used": {"type": "number"},
            "n_points": {"type": "integer"},
            "warnings": WARNINGS_SCHEMA,
        },
    }
)
async def handle_noise_integral(args: NoiseIntegralInput, state: SessionState):
    data = await metrics.noise_integral(
        _source_for(args, state),
        NoiseIntegralRecipe(
            key="noise_integral",
            metric="noise_integral",
            signal=args.signal,
            from_hz=args.f_start,
            to_hz=args.f_end,
        ),
        args.step,
        state,
    )
    signal = data["signal"]
    unit = data["unit"]
    total_str = f"{data['total_rms']:.6g}" + (f" {unit}" if unit else "")
    lines = [
        f"Integrated noise: {signal}",
        "",
        f"Band: [{data['f_start_used']:.6g}, {data['f_end_used']:.6g}] Hz, "
        f"{data['n_points']} points",
        f"Total RMS: {total_str}",
        f"Density unit: {data['density_unit']}; total = sqrt(\u222b density\u00b2 df).",
        "Noise figure / SNR are not computed \u2014 they need the source resistance and",
        "reference level only you have.",
    ]
    return _rendered(lines, data, args.format)


# Row cap for the per-run corner table (see include_per_run); large Monte
# Carlo batches otherwise append hundreds of rows to every stats response.
_MAX_PER_RUN_ROWS = 100


@declare_output_schema(output_model=MeasurementStatsResponse)
async def handle_measurement_stats(args: MeasurementStatsInput, state: SessionState):
    if args.log_file is not None and args.job_id is not None:
        raise ResultError(
            "Pass either ``log_file`` (single .step log) or ``job_id`` "
            "(walk every run's log of a Monte Carlo / sweep batch), not both."
        )
    if args.log_file is None and args.job_id is None:
        raise ResultError("Provide either ``log_file`` or ``job_id``.")

    if args.job_id is not None:
        # Both shapes this branch served — a batch's per-run log walk and a
        # single simulation's one log — belonged to job types earlier releases
        # wrote. An experiment's .MEAS results are read per case through
        # analyze_results, which resolves the source itself.
        job = await services.resolve_job_async(args.job_id, state)
        if isinstance(job, ExperimentJob):
            raise ResultError(
                f"Job {args.job_id!r} is an experiment; its .MEAS results are read per "
                "case. Use analyze_results with the measurements recipe."
            )
        raise ResultError(legacy_record_message(args.job_id))
    source = services.resolve_analysis_source(state, log_file=args.log_file)
    if source.log is None:
        raise ResultError("This source has no log artifact for measurement results.")

    steps_label = f"{await _step_label(source, state)} step(s)"
    data = await metrics.measurements(
        source,
        MeasurementsRecipe(
            key="measurements", metric="measurements", histogram_bins=args.histogram_bins
        ),
        0,
        state,
        measurement=args.measurement,
    )
    stats = data["stats"]

    lines = [
        f"Measurement Stats ({steps_label})",
        "",
    ]
    for name, entry in stats.items():
        lines.append(f"{name}:")
        field = entry.get("aggregated_field", "value")
        lines.append(
            f"  valid {entry['valid_count']}/{entry['total_count']} "
            f"(failed {entry['failure_count']})  field={field}"
        )
        if "at" in entry:
            # Neutral label — must not claim WHEN-crossing vs FIND-probe (see ``at`` field).
            lines.append(f"  at={entry['at']:.6g}  (time/freq the value was reported at)")
        if entry["valid_count"] > 0:
            lines.append(
                f"  min={entry['min']:.6g}  max={entry['max']:.6g}  "
                f"mean={entry['mean']:.6g}  median={entry['median']:.6g}  "
                f"std={entry['std']:.6g}"
            )
            lines.append(
                f"  p10={entry['p10']:.6g}  p90={entry['p90']:.6g}  "
                f"argmin step={entry['min_step_index']}  "
                f"argmax step={entry['max_step_index']}"
            )
        lines.append("")

    return format_response("\n".join(lines).rstrip(), data, args.format)


async def _step_label(source: services.AnalysisSource, state: SessionState) -> int:
    """How many .step points the log this read aggregated carries."""
    assert source.log is not None
    log_path = source.log
    return (
        await services.bounded_parse(
            log_path, lambda: parse_measurements(log_path).get("step_count", 1)
        )
    ) or 1


# ---------------------------------------------------------------------------
# AC analysis tools
# ---------------------------------------------------------------------------


# ---- Input models ---------------------------------------------------------


class FindCrossingInput(ToolInput):
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    quantity: Quantity = Field(
        description=(
            "What to cross: 'magnitude_db' (dB), 'magnitude_linear' (absolute |H|), "
            "or 'phase_deg' (UNWRAPPED phase in degrees)."
        ),
    )
    level: float = Field(
        description="Level to cross at, in the units of `quantity`. e.g. 0 for 0 dB, -180 for phase margin.",
    )
    direction: SearchDirection = Field(default="any")
    f_start: str | None = Field(
        default=None,
        description="Lower frequency bound in SPICE notation (e.g. '10k'). Defaults to sweep start.",
    )
    f_end: str | None = Field(
        default=None,
        description="Upper frequency bound in SPICE notation. Defaults to sweep end.",
    )
    max_results: int = Field(default=10, description="Cap on returned crossings (1..100).")
    min_separation_decades: float = Field(
        default=0.0,
        description="Merge crossings within this many decades; useful when gain grazes the level.",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class GainAtInput(ToolInput):
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    frequencies: list[str] = Field(
        description=(
            "Frequencies to query, each in SPICE notation (e.g. ['100', '1k', '10k']). "
            "Log-axis interpolation is used — queries between sample points are exact "
            "under a log-scale linear assumption, which matches .AC DEC spacing."
        ),
    )
    include_unwrapped_phase: bool = Field(
        default=False,
        description="Also return cumulative unwrapped phase (handy for delay / margin prep).",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class FilterMetricsInput(ToolInput):
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    ref_db: float = Field(
        default=-3.0,
        description=(
            "Cutoff reference BELOW passband in dB (must be negative). "
            "Standard -3 for half-power; use -1 for tighter passband specs "
            "or -6 for voltage-half."
        ),
    )
    flatness_db: float = Field(
        default=1.0,
        description="Passband flatness tolerance in dB used for auto-detecting the passband range.",
    )
    passband_range: list[str] | None = Field(
        default=None,
        description=(
            "Optional [f_lo, f_hi] SPICE-notation override for the passband. "
            "If omitted, auto-detected from the flat region near the peak."
        ),
    )
    stopband_range: list[str] | None = Field(
        default=None,
        description=(
            "Optional [f_lo, f_hi] SPICE-notation stopband region. If given, "
            "stopband_rejection is the worst-case attenuation in that range."
        ),
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class StabilityMetricsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to loop-gain AC analysis .raw file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a completed job run by id instead of a raw_file path; pair "
            "with ``run_index``. Lets you read a sweep / Monte-Carlo run's margins."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(
        description=(
            "Loop-gain signal: a trace (e.g. 'V(loop)'), a ratio ('V(ret)/V(inj)'), "
            "or either with a leading '-' for a sign-inverting probe "
            "('-V(vout)/V(vsense)') — the 180° flip is applied here, no behavioral "
            "inverter node needed."
        )
    )
    min_separation_decades: float = Field(
        default=0.1,
        description="Merge near-duplicate crossovers closer than this many decades.",
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class RollOffInput(ToolInput):
    raw_file: str | Path = Field(description="Path to AC analysis .raw result file")
    signal: str = Field(description="Signal name (e.g. 'V(out)')")
    f_low: str = Field(description="Low frequency bound (SPICE notation)")
    f_high: str = Field(description="High frequency bound (SPICE notation)")
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class ResonanceInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to AC analysis .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a completed job run by id instead of a raw_file path; pair "
            "with ``run_index``. Lets you read a sweep / Monte-Carlo run's peaks."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(description=f"Signal name (e.g. 'V(out)'). {_AC_SIGNAL_EXPR_NOTE}")
    min_prominence_db: float = Field(
        default=3.0,
        description=(
            "Minimum peak prominence in dB. Smaller = more sensitive but also "
            "catches gentle humps. 3 dB rejects filter-passband shoulders."
        ),
    )
    min_separation_decades: float = Field(
        default=0.2,
        description="Merge peaks closer than this many decades (find_peaks can emit duplicates on shoulders).",
    )
    max_peaks: int = Field(default=20, description="Maximum peaks returned (1..1000)")
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


# ---- Output schemas -------------------------------------------------------


class FilterMetricsResponse(FilterMetricsOutput):
    """Tool-layer response = lib output + the signal name the user asked about."""

    signal: str


class StabilityMetricsResponse(StabilityMetricsOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


class FindCrossingResponse(TypedDict):
    """Tool-layer response for :func:`handle_find_crossing`."""

    signal: str
    quantity: Quantity
    level: float
    direction: SearchDirection
    crossings: list[CrossingWithQuantity]
    warnings: list[str]


class GainAtResponse(TypedDict):
    """Tool-layer response for :func:`handle_gain_at`."""

    signal: str
    points: list[GainAtPoint]
    warnings: list[str]


class RollOffResponse(RollOffOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


class ResonancesResponse(ResonancesOutput):
    """Tool-layer response = lib output + the signal name."""

    signal: str


# ---- Handlers -------------------------------------------------------------


# Internal compute adapter — exposed publicly via bode_metrics(mode="crossing").
async def handle_find_crossing(args: FindCrossingInput, state: SessionState):
    data = await metrics.find_crossing(
        _ac_source_for(args.raw_file, state),
        args.signal,
        args.quantity,
        args.level,
        args.step,
        state,
        direction=args.direction,
        f_start=args.f_start,
        f_end=args.f_end,
        max_results=args.max_results,
        min_separation_decades=args.min_separation_decades,
    )
    crossings = data["crossings"]
    unit = crossings[0]["units"] if crossings else ""
    lines = [
        f"Crossings of {args.signal}.{args.quantity} at {args.level:g}{unit}:",
        "",
    ]
    if not crossings:
        lines.append("  (none found in window)")
    else:
        for c in crossings:
            lines.append(f"  {c['frequency_hz']:.6g} Hz ({c['direction']})")
    return _rendered(lines, data, args.format)


# Internal compute adapter — exposed publicly via bode_metrics(mode="point").
async def handle_gain_at(args: GainAtInput, state: SessionState):
    data = await metrics.gain_at(
        _ac_source_for(args.raw_file, state),
        args.signal,
        args.frequencies,
        args.step,
        state,
        include_unwrapped_phase=args.include_unwrapped_phase,
    )
    lines = [f"Gain/phase of {args.signal}:", ""]
    header = "  {:>14s}  {:>10s}  {:>10s}".format("Frequency (Hz)", "Mag (dB)", "Phase (\u00b0)")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for point in data["points"]:
        lines.append(
            f"  {point['frequency_hz']:>14.6g}  {point['magnitude_db']:>10.3f}  "
            f"{point['phase_deg']:>10.2f}"
        )
    return _rendered(lines, data, args.format)


# Internal compute adapter — exposed publicly via bode_metrics(mode="filter").
async def handle_filter_metrics(args: FilterMetricsInput, state: SessionState):
    data = await metrics.filter_metrics(
        _ac_source_for(args.raw_file, state),
        args.signal,
        args.step,
        state,
        ref_db=args.ref_db,
        flatness_db=args.flatness_db,
        passband_range=args.passband_range,
        stopband_range=args.stopband_range,
    )

    fc_lo = "-" if data["cutoff_low_hz"] is None else f"{data['cutoff_low_hz']:.6g} Hz"
    fc_hi = "-" if data["cutoff_high_hz"] is None else f"{data['cutoff_high_hz']:.6g} Hz"
    rej = (
        "-" if data["stopband_rejection_db"] is None else f"{data['stopband_rejection_db']:.2f} dB"
    )
    slope = (
        "-"
        if data["rolloff_slope_db_per_decade"] is None
        else f"{data['rolloff_slope_db_per_decade']:.2f} dB/dec"
    )
    order = "-" if data["estimated_order"] is None else f"{data['estimated_order']}"
    tbw = (
        "-"
        if data["transition_bandwidth_hz"] is None
        else f"{data['transition_bandwidth_hz']:.6g} Hz"
    )
    lines = [
        f"Filter Metrics: {args.signal}",
        "",
        f"Type:                {data['filter_type']}",
        f"Passband:            "
        f"[{data['passband_low_hz']:.6g}, {data['passband_high_hz']:.6g}] Hz "
        f"@ {data['passband_gain_db']:.2f} dB",
        f"Passband ripple:     {data['passband_ripple_db']:.3f} dB",
        f"Cutoff (ref {args.ref_db:+.1f} dB): low={fc_lo}  high={fc_hi}",
        f"Stopband rejection:  {rej}",
        f"Transition BW:       {tbw}",
        f"Roll-off slope:      {slope}",
        f"Estimated order:     {order}",
    ]
    return _rendered(lines, data, args.format)


@declare_output_schema(output_model=StabilityMetricsResponse)
async def handle_stability_metrics(args: StabilityMetricsInput, state: SessionState):
    data = await metrics.stability(
        _source_for(args, state),
        StabilityRecipe(key="stability", metric="stability", signal=args.signal),
        args.step,
        state,
        min_separation_decades=args.min_separation_decades,
    )

    pm_worst = (
        "-"
        if data["phase_margin_worst_deg"] is None
        else f"{data['phase_margin_worst_deg']:.2f}\u00b0"
    )
    gm_worst = (
        "-" if data["gain_margin_worst_db"] is None else f"{data['gain_margin_worst_db']:.2f} dB"
    )
    lines = [
        f"Stability: {args.signal}",
        "",
        f"DC gain:          {data['dc_gain_db']:.2f} dB",
        f"High-freq gain:   {data['high_freq_gain_db']:.2f} dB",
        f"Classification:   {data['stability']}",
        f"PM (worst):       {pm_worst}",
        f"GM (worst):       {gm_worst}",
        "",
        f"Unity-gain crossings ({len(data['unity_gain_crossovers'])}):",
    ]
    for c, m in zip(data["unity_gain_crossovers"], data["phase_margins"], strict=True):
        lines.append(
            f"  {c['frequency_hz']:.6g} Hz ({c['direction']})  PM={m['margin_deg']:+.2f}\u00b0"
        )
    lines.append(f"Phase -180\u00b0 crossings ({len(data['phase_180_crossovers'])}):")
    for c, m in zip(data["phase_180_crossovers"], data["gain_margins"], strict=True):
        lines.append(
            f"  {c['frequency_hz']:.6g} Hz ({c['direction']})  GM={m['margin_db']:+.2f} dB"
        )
    return _rendered(lines, data, args.format)


# Internal compute adapter — exposed publicly via bode_metrics(mode="slope").
async def handle_roll_off(args: RollOffInput, state: SessionState):
    data = await metrics.roll_off(
        _ac_source_for(args.raw_file, state),
        args.signal,
        args.f_low,
        args.f_high,
        args.step,
        state,
    )

    order = (
        "-"
        if data["nearest_pole_order_estimate"] is None
        else str(data["nearest_pole_order_estimate"])
    )
    lines = [
        f"Roll-off: {args.signal}",
        "",
        f"Span: [{data['f_low_hz']:.6g}, {data['f_high_hz']:.6g}] Hz "
        f"({data['span_decades']:.2f} decades)",
        f"Gain:  {data['gain_low_db']:.2f} \u2192 {data['gain_high_db']:.2f} dB "
        f"(\u0394 = {data['delta_db']:+.2f} dB)",
        f"Slope: {data['slope_db_per_decade']:.2f} dB/decade "
        f"({data['slope_db_per_octave']:.2f} dB/octave)",
        f"Estimated nearest pole order: {order}",
    ]
    return _rendered(lines, data, args.format)


BodeMode = Literal["filter", "slope", "point", "crossing"]


def _bode_output_schema() -> dict:
    """Union object schema across the four mode shapes for client introspection.

    Merges the per-mode response TypedDicts so the published ``outputSchema``
    accurately documents every key a mode can return (each is optional — the
    actual key set depends on ``mode``). Reusing the TypedDicts keeps the
    mode shapes single-sourced with the internal compute adapters.
    """
    merged: dict = {}
    for td in (FilterMetricsResponse, RollOffResponse, GainAtResponse, FindCrossingResponse):
        merged.update(schema_from_typeddict(td).get("properties", {}))
    # all_steps mode wraps per-step results under ``steps``; a step entry is a
    # mode result plus its ``step`` index (or an ``error`` if that step failed).
    # ``step_params`` maps the index to the .step name=value point — LTspice
    # runs a ``.step ... list`` ascending-sorted, not in declared order, so the
    # bare index is not safe to correlate with the deck's list order.
    step_item = {
        "type": "object",
        "properties": {
            **merged,
            "step": {"type": "integer"},
            "step_params": {"type": "object", "additionalProperties": {"type": "number"}},
            "error": {"type": "string"},
        },
    }
    return {
        "type": "object",
        "properties": {
            **merged,
            "mode": {"type": "string"},
            "signal": {"type": "string"},
            "all_steps": {"type": "boolean"},
            "step_count": {"type": "integer"},
            "steps": {"type": "array", "items": step_item},
            "warnings": WARNINGS_SCHEMA,
            "run_index": {"type": "integer"},
            "params": {"type": "object"},
        },
    }


class BodeMetricsInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to AC analysis .raw result file. Pass this OR ``job_id``, not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a specific run of a completed sweep/MC (or single) job instead "
            "of a raw_file path; pair with ``run_index``. The analyzed run's swept "
            "parameter values are echoed back under ``params`` (with ``run_index``), "
            "so you can tell which sweep point this is without a separate "
            "lookup. Combine with ``all_steps`` to sweep the .step axis "
            "WITHIN that run (a value-list/param sweep stores each run as its own "
            "raw — address those by run_index, not all_steps)."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(
        description=(
            "Signal to analyze: a single trace (e.g. 'V(out)') or a "
            "transfer-function ratio of two traces (e.g. 'V(out)/V(mid)'), which "
            "divides the two complex AC waves — the way to express an inter-stage "
            "gain, loop gain, or PSRR the simulator doesn't store as its own trace. "
            "A leading '-' negates the whole expression (180° phase flip), e.g. "
            "'-V(out)/V(vsense)' for a loop gain probed through an inverting sense — "
            "no behavioral inverter node needed."
        )
    )
    mode: BodeMode = Field(
        description=(
            "Which view of the AC response to compute:\n"
            "  'filter'   — LPF/HPF/BPF/BSF type, cutoffs, ripple, rejection, "
            "and an auto-estimated asymptotic roll-off slope (dB/decade) — so "
            "one call gives both cutoff AND slope "
            "(args: ref_db, flatness_db, passband_range, stopband_range)\n"
            "  'slope'    — magnitude slope between two explicit frequencies; "
            "use when you need a custom window or dB/octave ('filter' already "
            "reports an asymptotic dB/decade slope) "
            "(args: f_low, f_high — both required)\n"
            "  'point'    — magnitude (dB + linear) and phase at specific "
            "frequencies (args: frequencies — required; include_unwrapped_phase)\n"
            "  'crossing' — every frequency where magnitude/phase crosses a "
            "level (args: quantity + level — required; direction, f_start, "
            "f_end, max_results, min_separation_decades)"
        )
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    all_steps: bool = Field(
        default=False,
        description=(
            "Compute the metric for EVERY step of a stepped (.step) sweep in one "
            "call, instead of the single `step`. Returns `steps`: a list of "
            "per-step results (each tagged with its `step` index). A step whose "
            "computation fails is returned with an `error` field rather than "
            "aborting the whole call. On a non-stepped raw this returns a single "
            "entry. Use this for 'give me the cutoff/slope/gain at every step'."
        ),
    )
    # mode="crossing"
    quantity: Quantity | None = Field(
        default=None,
        description="crossing: 'magnitude_db' | 'magnitude_linear' | 'phase_deg'.",
    )
    level: float | None = Field(
        default=None, description="crossing: level to cross, in the units of `quantity`."
    )
    direction: SearchDirection = Field(default="any", description="crossing: edge direction.")
    f_start: str | None = Field(default=None, description="crossing: lower frequency bound.")
    f_end: str | None = Field(default=None, description="crossing: upper frequency bound.")
    max_results: int = Field(default=10, description="crossing: cap on returned crossings.")
    min_separation_decades: float = Field(
        default=0.0, description="crossing: merge crossings within this many decades."
    )
    # mode="point"
    frequencies: list[str] | None = Field(
        default=None, description="point: frequencies to query (SPICE notation)."
    )
    include_unwrapped_phase: bool = Field(
        default=False, description="point: also return cumulative unwrapped phase."
    )
    # mode="filter"
    ref_db: float = Field(
        default=-3.0, description="filter: cutoff reference below passband (dB)."
    )
    flatness_db: float = Field(
        default=1.0, description="filter: passband flatness tolerance (dB)."
    )
    passband_range: list[str] | None = Field(
        default=None, description="filter: optional [f_lo, f_hi] passband override."
    )
    stopband_range: list[str] | None = Field(
        default=None, description="filter: optional [f_lo, f_hi] stopband region."
    )
    # mode="slope"
    f_low: str | None = Field(default=None, description="slope: low frequency bound (required).")
    f_high: str | None = Field(default=None, description="slope: high frequency bound (required).")
    format: FormatField = Field(default=None)


@declare_output_schema(_bode_output_schema())
async def handle_bode_metrics(args: BodeMetricsInput, state: SessionState):
    """Dispatch to the per-mode AC compute adapters (one shared AC load each)."""
    _validate_bode_mode_args(args)
    raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
    run_meta = _run_meta(args.job_id, args.run_index, state)
    source = services.AnalysisSource.for_raw(raw_path)
    if not args.all_steps:
        try:
            res = await _bode_dispatch(args, args.step, state, raw_path)
        except ResultError as e:
            await metrics.reraise_with_solve_failure(e, source)
        if run_meta and res.structuredContent is not None:
            res.structuredContent.update(run_meta)
        return _append_warnings_to_result(res, await metrics.solve_failures(source), args.format)

    # all_steps: compute the metric for every step of the sweep.
    raw = await services.load_raw(raw_path, state)
    step_count = get_step_count(raw)

    # Per-step .step name=value points from the sibling .log (same source
    # export_waveform's step_value column uses). LTspice runs a ``.step ...
    # list`` ascending-sorted, not in declared order, so labeling entries with
    # only the bare index invites mis-attribution of curves to list positions.
    # parse_step_iterations returns [] for a missing/unreadable log.
    step_params: list[dict[str, float]] = []
    if step_count > 1:
        step_params = await asyncio.to_thread(parse_step_iterations, raw_path.with_suffix(".log"))

    steps_out: list[dict] = []
    step_texts: list[str] = []
    # Distinct per-step warning -> the step indices that emitted it. A warning
    # that fires identically on every step (e.g. the no-stopband_range sweep-
    # endpoint note) is surfaced ONCE at the top level with its step coverage,
    # not repeated per step in both the structured 'steps' and the text.
    warning_steps: dict[str, list[int]] = {}
    first_error: ResultError | None = None
    for i in range(step_count):
        params_i = step_params[i] if i < len(step_params) else None
        label = f"step {i}"
        entry: dict = {"step": i}
        if params_i:
            label += " (" + ", ".join(f"{k}={v:g}" for k, v in params_i.items()) + ")"
            entry["step_params"] = params_i
        try:
            res = await _bode_dispatch(args, i, state, raw_path)
            sc = _structured(res)
            for w in sc.pop("warnings", None) or []:
                warning_steps.setdefault(w, []).append(i)
            entry.update(sc)
            step_texts.append(f"── {label} ──\n{_strip_warning_block(result_text(res))}")
        except ResultError as e:
            if first_error is None:
                first_error = e
            entry["error"] = str(e)
            step_texts.append(f"── {label} ── error: {e}")
        steps_out.append(entry)

    errored = [s for s in steps_out if "error" in s]
    if step_count and len(errored) == step_count and first_error is not None:
        # Every step failed (e.g. a non-AC raw fed to all_steps) — re-raise the
        # ORIGINAL error (its show_hint/suggestions survive), enriched with any
        # solve failure, instead of a "success" full of buried per-step errors.
        await metrics.reraise_with_solve_failure(first_error, source)

    data: dict = {
        "mode": args.mode,
        "signal": args.signal,
        "all_steps": True,
        "step_count": step_count,
        "steps": steps_out,
    }
    if run_meta:
        data.update(run_meta)
    warnings: list[str] = []
    if step_count == 1:
        warnings.append("Raw is not stepped (step_count=1); 'steps' has a single entry.")
    for w, idxs in warning_steps.items():
        warnings.append(f"{w} ({_warning_coverage(idxs, step_count)})")
    if errored:
        warnings.append(f"{len(errored)} of {step_count} steps failed (see per-step 'error').")
    # Run-level solve failure taints every step; surface it once at the top.
    warnings.extend(await metrics.solve_failures(source))
    if warnings:
        data["warnings"] = warnings

    header = [
        f"bode {args.mode} (all_steps) — {args.signal}",
        f"Steps: {step_count}",
        *(f"⚠ {w}" for w in warnings),
        "",
    ]
    return format_response("\n".join(header) + "\n".join(step_texts), data, args.format)


def _validate_bode_mode_args(args: BodeMetricsInput) -> None:
    """Raise for missing per-mode required args. Called once up front so a
    caller mistake surfaces immediately instead of being swallowed per-step in
    ``all_steps`` mode."""
    if args.mode == "crossing" and (args.quantity is None or args.level is None):
        raise ResultError("bode_crossing requires 'quantity' and 'level'.")
    if args.mode == "point" and not args.frequencies:
        raise ResultError("bode_point requires 'frequencies'.")
    if args.mode == "slope" and (args.f_low is None or args.f_high is None):
        raise ResultError("bode_slope requires 'f_low' and 'f_high'.")


async def _bode_dispatch(
    args: BodeMetricsInput, step: int, state: SessionState, raw_path: Path
) -> types.CallToolResult:
    """Build the per-mode input for ``step`` and dispatch to its compute adapter.

    ``raw_path`` is the already-resolved .raw (from raw_file or a job run) — the
    adapters load it directly (trusted Path), so a sweep run is analyzed by the
    same AC machinery as a standalone raw. Assumes ``_validate_bode_mode_args``
    has already validated required args.
    """
    if args.mode == "crossing":
        assert args.quantity is not None and args.level is not None
        return await handle_find_crossing(
            FindCrossingInput(
                raw_file=raw_path,
                signal=args.signal,
                quantity=args.quantity,
                level=args.level,
                direction=args.direction,
                f_start=args.f_start,
                f_end=args.f_end,
                max_results=args.max_results,
                min_separation_decades=args.min_separation_decades,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "point":
        assert args.frequencies
        return await handle_gain_at(
            GainAtInput(
                raw_file=raw_path,
                signal=args.signal,
                frequencies=args.frequencies,
                include_unwrapped_phase=args.include_unwrapped_phase,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "filter":
        return await handle_filter_metrics(
            FilterMetricsInput(
                raw_file=raw_path,
                signal=args.signal,
                ref_db=args.ref_db,
                flatness_db=args.flatness_db,
                passband_range=args.passband_range,
                stopband_range=args.stopband_range,
                step=step,
                format=args.format,
            ),
            state,
        )
    if args.mode == "slope":
        assert args.f_low is not None and args.f_high is not None
        return await handle_roll_off(
            RollOffInput(
                raw_file=raw_path,
                signal=args.signal,
                f_low=args.f_low,
                f_high=args.f_high,
                step=step,
                format=args.format,
            ),
            state,
        )
    raise ResultError(f"Unknown bode mode {args.mode!r}")


def _structured(result: types.CallToolResult) -> dict:
    """structuredContent of an adapter result as a dict (``{}`` if absent)."""
    return dict(result.structuredContent) if result.structuredContent else {}


def _strip_warning_block(text: str) -> str:
    """Drop the trailing ``_warning_lines`` block from an adapter's rendered
    text. In all_steps mode the per-step warnings are hoisted (deduped) to the
    top level, so repeating them inside every step's text is noise."""
    return text.split(f"\n\n{_WARNINGS_HEADER}\n", 1)[0]


def _warning_coverage(step_indices: list[int], step_count: int) -> str:
    """Describe which steps a hoisted all_steps warning applies to. Lists the
    actual step indices for ANY partial subset — never a bare count — so a
    consumer can still identify exactly which sweep cases emitted it; only the
    every-step case collapses to a compact label."""
    if len(step_indices) == step_count:
        return f"all {step_count} steps"
    return "steps " + ",".join(str(i) for i in step_indices)


@declare_output_schema(output_model=ResonancesResponse)
async def handle_resonance(args: ResonanceInput, state: SessionState):
    data = await metrics.resonance(
        _source_for(args, state),
        ResonanceRecipe(key="resonance", metric="resonance", signal=args.signal),
        args.step,
        state,
        min_prominence_db=args.min_prominence_db,
        min_separation_decades=args.min_separation_decades,
        max_peaks=args.max_peaks,
    )

    lines = [f"Resonances: {args.signal}", "", f"Peaks detected: {data['num_peaks_detected']}"]
    for peak in data["peaks"]:
        q = "-" if peak["q_factor"] is None else f"{peak['q_factor']:.2f}"
        bw = "-" if peak["bandwidth_3db_hz"] is None else f"{peak['bandwidth_3db_hz']:.6g} Hz"
        lines.append(
            f"  f={peak['frequency_hz']:.6g} Hz  gain={peak['magnitude_db']:.2f} dB "
            f"(|{peak['magnitude_linear']:.6g}|)  "
            f"Q={q}  BW-3dB={bw}  phase={peak['phase_deg']:+.2f}\u00b0"
        )
    return _rendered(lines, data, args.format)


class ReturnLossInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to AC analysis .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a completed job run by id instead of a raw_file path; pair "
            "with ``run_index``."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to analyze when ``job_id`` is given (default 0).",
    )
    signal: str = Field(
        description=(
            "Input-impedance trace: the node voltage under a 1 A AC probe, where "
            "V(node) = Zin (e.g. 'V(in)'). A leading '-' negates the trace — the "
            "in-place fix for a probe wired backwards. See spice://guide for the "
            "probe idiom and sign convention."
        )
    )
    z0: float = Field(default=50.0, description="Reference impedance in ohms (default 50).")
    at: str | None = Field(
        default=None,
        description=(
            "Frequency in SPICE notation (e.g. '100meg') to evaluate. Omit to report "
            "the worst-match point (maximum |Γ| / minimum return loss) across the sweep."
        ),
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None, description="'json' or 'text'")


class ReturnLossResponse(ReturnLossOutput):
    signal: str
    z0_ohm: float


@declare_output_schema(output_model=ReturnLossResponse)
async def handle_return_loss(args: ReturnLossInput, state: SessionState):
    data = await metrics.return_loss(
        _source_for(args, state),
        ReturnLossRecipe(key="return_loss", metric="return_loss", signal=args.signal, z0=args.z0),
        args.step,
        state,
        at=args.at,
    )

    rl = data["return_loss_db"]
    rl_str = "\u221e (perfect match)" if rl is None else f"{rl:.2f} dB"
    vswr = data["vswr"]
    vswr_str = "\u221e (open/short)" if vswr is None else f"{vswr:.3f}"
    where = "worst match" if data["worst_match"] else "at requested frequency"
    lines = [
        f"Return Loss: {args.signal} (z0={args.z0:g} \u03a9, {where})",
        "",
        f"Frequency: {data['frequency_hz']:.6g} Hz",
        f"Zin: {data['zin_real_ohm']:.4g} {data['zin_imag_ohm']:+.4g}j \u03a9 "
        f"(|Zin|={data['zin_mag_ohm']:.4g} \u03a9)",
        f"|\u0393|: {data['gamma_mag']:.4g}  \u2220{data['gamma_phase_deg']:.2f}\u00b0",
        f"Return loss: {rl_str}",
        f"VSWR: {vswr_str}",
    ]
    if "zin_min_mag_ohm" in data:
        lines.append(
            f"|Zin| range: {data['zin_min_mag_ohm']:.4g} \u03a9 at "
            f"{data['zin_min_freq_hz']:.6g} Hz \u2026 {data['zin_max_mag_ohm']:.4g} \u03a9 at "
            f"{data['zin_max_freq_hz']:.6g} Hz"
        )
    return _rendered(lines, data, args.format)


class AcStructureInput(ToolInput):
    raw_file: str | None = Field(
        default=None,
        description="Path to AC analysis .raw result file. Pass this OR ``job_id`` (a job run), not both.",
    )
    job_id: str | None = Field(
        default=None,
        description=(
            "Analyze a completed job run by id instead of a raw_file path; pair "
            "with ``run_index``. Lets you read a sweep / Monte-Carlo run's structure."
        ),
    )
    run_index: int = Field(
        default=0,
        description="0-based run to read when ``job_id`` is given (default 0).",
    )
    signal: str = Field(
        description=(
            "Signal name (e.g. 'V(out)') — the transfer function H(jω) to analyze. "
            f"{_AC_SIGNAL_EXPR_NOTE}"
        )
    )
    step: int = Field(default=0, description="Step index for .step sweeps")
    format: FormatField = Field(default=None)


class AcStructureResponse(AcStructureResult):
    """Tool-layer response = lib structure facts + the signal name."""

    signal: str


def _fmt_hz_range(f_lo: float, f_hi: float) -> str:
    """Compact display of a corner's frequency range (a point sets lo == hi)."""
    if abs(f_hi - f_lo) <= 1e-9 * max(abs(f_hi), 1.0):
        return f"{f_lo:.3g} Hz"
    return f"{f_lo:.3g} to {f_hi:.3g} Hz"


@declare_output_schema(output_model=AcStructureResponse)
async def handle_ac_structure(args: AcStructureInput, state: SessionState):
    data = await metrics.ac_structure(
        _source_for(args, state),
        AcStructureRecipe(key="ac_structure", metric="ac_structure", signal=args.signal),
        args.step,
        state,
    )

    order = data["net_order"]
    lines = [
        f"AC structure: {args.signal}",
        "",
        f"Net high-frequency order (pole-zero excess): "
        f"{order if order is not None else 'unknown'}",
    ]
    if data["integrator"]:
        lines.append(
            f"Low-frequency asymptote: integrator / pole at origin "
            f"({data['lf_slope_db_per_decade']:.0f} dB/dec)"
        )
    if data["corners"]:
        lines.append("Corners:")
        kinds = {
            "real_pole": "real pole",
            "real_zero": "real zero",
            "complex_pair": "complex pole pair",
            "complex_zero_pair": "complex zero pair",
        }
        for corner in data["corners"]:
            q = f", Q~{corner['q']:.1f}" if corner["q"] is not None else ""
            merged = (
                " (closely-spaced cluster — range only, not individually resolved)"
                if corner["merged"]
                else ""
            )
            rng = _fmt_hz_range(corner["f_lo"], corner["f_hi"])
            lines.append(f"  - {kinds[corner['kind']]} near {rng}{q}{merged}")
    else:
        lines.append("Corners: none located (flat magnitude or near-cancellation)")
    if data["non_minimum_phase"]:
        resid = data["phase_residual_deg"]
        extra = f" (phase residual ~{resid:.0f}\u00b0)" if resid is not None else ""
        lines.append(
            f"Out-of-phase zero / delay: YES{extra} — excess phase lag the "
            "magnitude alone understates; caps achievable bandwidth"
        )
    else:
        lines.append("No excess phase: phase tracks the magnitude (no out-of-phase zero or delay)")
    if data["transport_delay_s"]:
        lines.append(f"Transport delay: ~{data['transport_delay_s'] * 1e6:.1f} \u00b5s")
    fit = data["fit_rel_err"]
    lines.append(
        f"Method: {data['method']}" + (f" (fit error {fit:.1e})" if fit is not None else "")
    )
    if data["observations"]:
        lines.append("")
        lines.append("Observations (facts to weigh, not verdicts):")
        lines += [f"  - {o['detail']}" for o in data["observations"]]
    return format_response("\n".join(lines), data, args.format)


# ---------------------------------------------------------------------------
# plot_waveform — interactive HTML chart opened on the local desktop
# ---------------------------------------------------------------------------

PLOTS_SUBDIR = "plots"
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
        description="Path to .raw result file. Pass this OR ``job_id`` (a job run), not both.",
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
            "For a .step run: omit to overlay ALL steps as separate traces, or give "
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
        "Render an INTERACTIVE chart of one or more signals FOR A HUMAN to look at "
        "(zoom/pan/hover) — the co-design complement to the numeric tools. It returns "
        "NO data values to the model; it produces a picture.\n\n"
        "Picks the chart from the run type: transient (V/I vs time), DC sweep, AC "
        "Bode (stacked magnitude-dB + phase-deg vs log frequency), noise (vs log "
        "frequency); a .step / Monte-Carlo run overlays every step as a labelled "
        "trace (or pass ``step`` for one). Full fidelity by default, with a "
        "min/max-preserving downsample above ``max_points`` (spikes survive; "
        "surfaced as a fact). Writes a self-contained HTML file and returns its path "
        "— into ``out_dir`` if given, else a '.ltspice-mcp/plots/' sidecar next to "
        "the circuit (for a job_id) or next to the raw (for a raw_file); on a host "
        "that supports MCP Apps the chart is "
        "also embedded as an interactive in-chat widget, otherwise it opens in your "
        "local browser.\n\n"
        "Sibling egress, don't confuse: for numbers use analyze_results — the "
        "waveform recipe for a decimated table in context (or every sample as CSV "
        "on disk), signal_stats / the bode_* recipes for scalars. This tool is for "
        "looking, not measuring."
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
        raw_path = _effective_raw_path(args.raw_file, args.job_id, args.run_index, state)
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
        subdir=PLOTS_SUBDIR,
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


class StepGetInput(ToolInput):
    raw_file: str = Field(description="Path to a stepped .raw result")
    axis: str = Field(
        description=(
            "Step parameter name to query (e.g. ``temp``, ``RS``). For .DC "
            "sweeps the axis is the swept variable; for .step parametric "
            "runs it's the parameter that was stepped."
        ),
    )
    value: str = Field(
        description="SPICE-notation target value (e.g. ``27``, ``1k``, ``100u``).",
    )
    signal: str = Field(description="Signal to read at the chosen step (e.g. ``V(out)``).")
    at: str | None = Field(
        default=None,
        description=(
            "Optional inner-axis position to query within the chosen step "
            "(time for .tran, frequency for .ac). Defaults to the first "
            "sample, which is the only useful answer for stepped .op runs "
            "but rarely the right one for .ac/.tran. SPICE notation."
        ),
    )
    format: Literal["json", "text"] | None = Field(
        default=None,
        description=FORMAT_DESCRIPTION,
    )


def _step_get_native_axis(
    raw: RawRead, args: StepGetInput, signal: str, target: float
) -> types.CallToolResult:
    """Query on the .raw's native axis (DC sweep variable / AC frequency).

    The queried axis IS the inner axis, so this is a nearest-neighbour lookup
    on the axis values; a request beyond the axis ends is a clamp worth flagging.
    """
    # On the native-axis branch the queried axis IS the inner axis, so
    # there is no second position for ``at`` to select. Silently
    # ignoring it would return a value at ``value`` while the caller
    # believes the ``at`` slice was applied — refuse loudly instead.
    if args.at is not None:
        raise NetlistError(
            f"'at' does not apply here: {args.axis!r} is the raw file's "
            "native axis, so the query position is 'value' itself. "
            "'at' selects the inner-axis point only when 'axis' names a "
            ".step parameter."
        )
    try:
        axis_vals = real_axis(np.asarray(raw.get_axis(step=0))).tolist()
    except Exception as e:
        raise NetlistError(
            f"Cannot read axis values: {e}. Use the analyze_results value "
            "recipe if the raw doesn't have an explicit axis."
        ) from e
    if not axis_vals:
        raise NetlistError(f"Axis {args.axis!r} has no samples in this raw file.")
    # nearest neighbour
    ins = bisect.bisect_left(axis_vals, target)
    if ins == 0:
        idx = 0
    elif ins == len(axis_vals):
        idx = len(axis_vals) - 1
    else:
        idx = ins - 1 if abs(axis_vals[ins - 1] - target) <= abs(axis_vals[ins] - target) else ins
    wave = raw.get_wave(signal, step=0)
    actual = float(axis_vals[idx])
    # This is a continuous native axis (DC sweep variable / AC frequency),
    # not a discrete step list: an off-grid interior request is a normal
    # nearest-neighbour lookup, and only a request beyond the axis ends is
    # genuinely clamped. sample_to_dict keeps complex AC samples intact
    # (magnitude/phase) instead of float() silently dropping the imag part.
    sample_dict = sample_to_dict(wave[idx])
    exact = snap_match(target, actual)
    lo, hi = min(axis_vals[0], axis_vals[-1]), max(axis_vals[0], axis_vals[-1])
    out_of_range = target < lo or target > hi
    data = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": actual,
        "exact_match": exact,
        **sample_dict,
    }
    sample_str = (
        f"{sample_dict['value']:g}"
        if "value" in sample_dict
        else f"{sample_dict['magnitude_db']:.3f} dB / {sample_dict['phase_deg']:.2f}°"
    )
    summary = f"{signal} at {args.axis}={actual:g}: {sample_str}"
    if out_of_range:
        warning = (
            f"Requested {args.axis}={target:g} is outside the swept range "
            f"[{lo:g}, {hi:g}]; clamped to the nearest end {actual:g}."
        )
        data["warnings"] = [warning]
        summary += f"\nWarning: {warning}"
    return format_response(summary, data, args.format)


def _step_get_param_lookup(
    raw: RawRead,
    raw_path: Path,
    args: StepGetInput,
    signal: str,
    target: float,
    axis_lower: str,
) -> types.CallToolResult:
    """Query by .step parameter value, using the nearest stepped run.

    Falls back to .log parsing when spicelib's ``get_steps`` returns nothing
    (which it does for ``.step param NAME`` runs — the parameter map lives in
    the log, not the .raw header).
    """
    try:
        steps = list(raw.get_steps() or [])
    except Exception:
        steps = []

    if not any(isinstance(s, dict) and s for s in steps):
        # parse_step_iterations swallows OSError, so no .exists() guard.
        steps = list(parse_step_iterations(raw_path.with_suffix(".log")))

    best_idx = None
    best_actual: float | None = None
    for i, step_record in enumerate(steps):
        if not isinstance(step_record, dict):
            continue
        v = step_record.get(args.axis)
        if v is None:
            # try case-insensitive match
            for k, val in step_record.items():
                if k.lower() == axis_lower:
                    v = val
                    break
        if v is None:
            continue
        try:
            v_f = float(v)
        except (TypeError, ValueError):
            continue
        if best_actual is None or abs(v_f - target) < abs(best_actual - target):
            best_actual = v_f
            best_idx = i

    if best_idx is None:
        # Build the axis listing only on the error path.
        available_axes: list[str] = []
        for step_record in steps:
            if isinstance(step_record, dict):
                for k in step_record:
                    if k not in available_axes:
                        available_axes.append(k)
        if available_axes:
            raise NetlistError(
                f"Step axis {args.axis!r} not found in this raw file. "
                "Available axes: " + ", ".join(available_axes)
            )
        # No .step parameters at all — the caller likely meant the primary sweep
        # axis of a bare .dc/.ac sweep, which isn't a step. Point at the direct
        # route instead of a bare "not found".
        raise NetlistError(
            f"This raw file has no .step parameters, so {args.axis!r} is not a step "
            f"axis. If {args.axis!r} is the primary sweep variable of a bare .dc/.ac "
            f"sweep, query it directly with the analyze_results value recipe "
            f"at={args.value!r}."
        )

    assert best_actual is not None  # set in lockstep with best_idx above
    wave = raw.get_wave(signal, step=best_idx)
    if len(wave) == 0:
        raise NetlistError(
            f"Step {best_idx} of {signal!r} contains no samples; "
            "verify the simulation completed and the signal exists in this step."
        )

    # Pick the inner-axis sample. Default is index 0 (correct for .op
    # results); when ``at=`` is given, find the nearest neighbour on the
    # per-step axis (frequency for .AC, time for .TRAN).
    inner_idx = 0
    target_at: float | None = None
    actual_at: float | None = None
    warnings: list[str] = []
    if args.at is not None:
        try:
            target_at = parse_spice_value(args.at)
        except ValueError as e:
            raise NetlistError(f"Invalid at {args.at!r}: {e}") from e
        try:
            inner_axis = real_axis(np.asarray(raw.get_axis(step=best_idx)))
        except Exception as e:
            raise NetlistError(
                f"Cannot read inner axis for at={args.at!r}: {e}. "
                "Drop the ``at`` argument for .op-style raws."
            ) from e
        if inner_axis.size == 0:
            raise NetlistError(f"Step {best_idx} has an empty axis; ``at`` cannot be applied.")
        inner_idx = nearest_index(inner_axis, target_at)
        actual_at = float(inner_axis[inner_idx])
    else:
        # No inner coordinate requested. For .op raws index 0 is the only
        # sample; for .ac/.tran it's the first (passband / t=0) bin, whose
        # value is uninterpretable without knowing the coordinate. Surface
        # the implied coordinate when there is a real inner axis.
        try:
            inner_axis = real_axis(np.asarray(raw.get_axis(step=best_idx)))
        except Exception:
            inner_axis = np.asarray([])
        if inner_axis.size > 1:
            actual_at = float(inner_axis[0])
            warnings.append(
                f"No 'at' given: returning the first inner sample at {actual_at:g}. "
                "Pass 'at' (frequency for .ac, time for .tran) to pick a point."
            )

    if not snap_match(target, best_actual):
        warnings.append(
            f"Requested {args.axis}={target:g} but no step matches; using the "
            f"nearest step {best_actual:g}."
        )

    sample_dict = sample_to_dict(wave[inner_idx])
    data: dict = {
        "signal": signal,
        "axis": args.axis,
        "requested_value": target,
        "actual_value": best_actual,
        "exact_match": snap_match(target, best_actual),
        "step_index": best_idx,
        **sample_dict,
    }
    if target_at is not None:
        data["requested_at"] = target_at
    if actual_at is not None:
        data["actual_at"] = actual_at
    if warnings:
        data["warnings"] = warnings

    sample_str = (
        f"{sample_dict['value']:g}"
        if "value" in sample_dict
        else f"{sample_dict['magnitude_db']:.3f} dB / {sample_dict['phase_deg']:.2f}°"
    )
    at_str = f", at={actual_at:g}" if actual_at is not None else ""
    summary = f"{signal} at {args.axis}={best_actual:g} (step {best_idx}){at_str}: {sample_str}"
    for warning in warnings:
        summary += f"\nWarning: {warning}"
    return format_response(summary, data, args.format)


# Internal compute adapter — exposed publicly via query_value(step_axis=, step_value=).
# Operates on a SINGLE multi-step .raw (as produced by .step/.dc). An external
# sweep job (configure_sweep/run_sweep) emits N single-point raws with no step
# axis instead — use batch_results for those.
async def handle_step_get(args: StepGetInput, state: SessionState) -> types.CallToolResult:
    """Query a signal at a specific axis value of a stepped .raw result."""
    raw_path = safe_path(args.raw_file, state)
    raw = await services.load_raw(raw_path, state)

    try:
        target = parse_spice_value(args.value)
    except ValueError as e:
        raise NetlistError(f"Invalid value {args.value!r}: {e}") from e

    signal = services.validate_signal(raw, args.signal)

    # Strategy: if ``axis`` matches the .raw's axis name (case-insensitive),
    # use the axis values directly. Otherwise fall back to .step parameter
    # lookup via spicelib's ``get_steps``.
    raw_axis_name = ""
    try:
        plot = raw.get_raw_property("Plotname")
        if plot:
            # Plotname doesn't carry the axis name; pull from trace 0.
            raw_axis_name = raw.get_trace_names()[0]
    except Exception:
        pass

    axis_lower = args.axis.lower()
    if raw_axis_name and axis_lower == raw_axis_name.lower():
        return _step_get_native_axis(raw, args, signal, target)
    return _step_get_param_lookup(raw, raw_path, args, signal, target, axis_lower)
