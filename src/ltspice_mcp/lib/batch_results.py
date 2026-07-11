"""Batch result statistics and parameter filtering for sweep and Monte Carlo jobs.

Provides functions for computing aggregate statistics across batch simulation
runs, filtering runs by parameter values, and querying job progress.

All functions are synchronous — callers invoke them directly
(see concurrency contract in tools/_base.py).
"""

import math
import time
from pathlib import Path

import numpy as np

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
from ltspice_mcp.state import BatchJob

_STAT_KEYS = (
    "max_across_runs",
    "min_across_runs",
    "mean_across_runs",
    "std_across_runs",
    "median_across_runs",
)


def _aggregate_peaks(
    peak_values: list[float], per_run_summaries: list[dict]
) -> tuple[dict, int | None, int | None]:
    """Aggregate across-run stats over the FINITE peaks only.

    A diverged-but-completed run (a normal outcome when Monte-Carlo / sweep
    perturbation pushes a solve to a NaN/Inf sample) would otherwise NaN-poison
    every statistic — np.max/mean over an array holding one NaN returns NaN —
    and both np.argmax/argmin would point at that same diverged run, mislabeling
    it as the worst AND best case. Excluding non-finite peaks keeps the aggregate
    and the worst/best-case pointers meaningful; the per-run rows still carry the
    non-finite values, so divergence stays visible to the caller.
    """
    none_stats: dict = {k: None for k in _STAT_KEYS}
    if not peak_values:
        return none_stats, None, None
    peaks = np.asarray(peak_values, dtype=float)
    finite_idx = np.flatnonzero(np.isfinite(peaks))
    if finite_idx.size == 0:
        return none_stats, None, None
    fp = peaks[finite_idx]
    stats = {
        "max_across_runs": float(np.max(fp)),
        "min_across_runs": float(np.min(fp)),
        "mean_across_runs": float(np.mean(fp)),
        "std_across_runs": float(np.std(fp)),
        "median_across_runs": float(np.median(fp)),
    }
    max_case_run = per_run_summaries[int(finite_idx[np.argmax(fp)])]["run_index"]
    min_case_run = per_run_summaries[int(finite_idx[np.argmin(fp)])]["run_index"]
    return stats, max_case_run, min_case_run


def compute_batch_stats(
    run_results: dict[int, dict],
    signal: str,
    *,
    at: float | None = None,
    dialect: str | None = None,
) -> dict:
    """Compute aggregate statistics for a signal across all batch runs.

    Loads each run's raw file, extracts the requested signal waveform, and
    computes per-run scalars (max/min/mean of absolute values). Aggregates
    those scalars across all runs to produce min/max/mean/std/median and
    identifies the worst-case (highest peak) and best-case (lowest peak) run.

    With ``at`` (a target time/frequency), each run is sliced to a single
    point first — useful for AC sweeps where the per-run peak across the
    full frequency range conflates startup roll-off with run-to-run
    variation. ``peak``/``mean``/``min`` collapse to the same value at that
    point, and ``stats.*_across_runs`` answer "what's the spread of the
    magnitude at this frequency across runs?".

    Runs with missing raw files are skipped gracefully — useful for cancelled
    jobs that produced only partial results.

    All numpy scalars are converted to Python float for JSON serialization.

    Args:
        run_results: Dict mapping run_index -> {raw_file, log_file, params}
        signal: Signal name to extract (e.g. "V(out)", "I(R1)")
        at: Optional time (transient) or frequency (AC) point. When given,
            collapses each run to a single sample using nearest-neighbour
            lookup on the run's axis. Without it, the per-run peak across
            the full waveform is used (legacy behaviour).

    Returns:
        Dict with:
            signal: str — the queried signal name
            run_count: int — number of runs with results
            runs: list[dict] — per-run summary with run_index, params, and scalars
            stats: dict — aggregate min/max/mean/std/median across runs
            max_case_run: int | None — run with highest peak absolute value
            min_case_run: int | None — run with lowest peak absolute value
            at: float | None — echo of the slicing point (for downstream display)
    """
    per_run_summaries = []
    peak_values: list[float] = []
    # Runs whose raw carries an inner .step sweep: we read only step 0, so the
    # other steps are dropped. Surface them rather than silently collapse.
    step_collapsed: list[int] = []
    # Runs whose step metadata couldn't be read at all: step 0 is still returned,
    # but dropped steps can't be ruled out — surface separately, never swallow.
    step_unknown: list[int] = []

    for run_index in sorted(run_results.keys()):
        run = run_results[run_index]
        raw_path = run.get("raw_file", "")

        # Skip runs with missing raw files (partial results from cancelled jobs)
        if not raw_path or not Path(raw_path).exists():
            continue

        try:
            raw = OffsetAwareRawRead(raw_path, traces_to_read=signal, dialect=dialect)
            try:
                n_inner_steps = len(raw.get_steps())
            except Exception:
                # Step metadata unreadable: we still read step 0 below, but can't
                # tell whether other steps were dropped. Surface that rather than
                # silently assuming a single step — the failure mode this guard
                # exists to remove.
                step_unknown.append(run_index)
            else:
                if n_inner_steps > 1:
                    step_collapsed.append(run_index)
            wave = raw.get_wave(signal, step=0)

            # AC (complex): use magnitude; transient: use raw values
            if np.iscomplexobj(wave):
                wave = np.abs(wave)

            point: float | None = None
            try:
                axis = np.asarray(raw.get_axis(step=0))
                axis_size = axis.size
            except Exception:
                # ``.op`` raws have no axis; the wave is a single scalar, so
                # ``peak``/``mean``/``min`` collapse to one value.
                axis = None  # type: ignore[assignment]
                axis_size = 0

            # ``collapsed`` marks a genuine point query (``at=`` or a .op
            # single-sample raw), where peak/mean/min are the same sample by
            # construction. It is NOT inferred from data equality: a flat
            # full-waveform run also has peak==mean==min, but it is still a
            # waveform and must keep the trio so the row shape stays uniform
            # across a sweep.
            if axis is None or axis_size == 0:
                if wave.size == 0:
                    continue
                point = float(wave[0])
                peak = mean_val = min_val = point
                collapsed = True
            elif at is not None:
                if np.iscomplexobj(axis):
                    axis = np.real(axis)
                if axis.size == 0:
                    continue
                # A DC/param sweep axis may run high->low (e.g. ``.dc V1 5 0
                # -0.1``). searchsorted assumes ascending order, so flip axis
                # and wave together (both views, no copy) when descending —
                # otherwise the slice silently returns the wrong sample.
                if axis.size > 1 and axis[0] > axis[-1]:
                    axis = axis[::-1]
                    wave = wave[::-1]
                # SPICE sweep axes are monotonic; binary-search beats
                # ``argmin(abs(axis - at))`` which materializes a full
                # diff array per run (multi-MB for long transients).
                ins = int(np.searchsorted(axis, at))
                if ins == 0:
                    idx = 0
                elif ins == axis.size:
                    idx = axis.size - 1
                else:
                    idx = ins - 1 if abs(axis[ins - 1] - at) <= abs(axis[ins] - at) else ins
                point = float(wave[idx])
                peak = mean_val = min_val = point
                collapsed = True
            else:
                peak = float(np.max(wave))
                mean_val = float(np.mean(wave))
                min_val = float(np.min(wave))
                collapsed = False

            entry: dict = {
                "run_index": run_index,
                "params": run.get("params", {}),
            }
            # Point query -> surface just ``value``; full-waveform aggregation
            # -> always keep peak/mean/min, even when they happen to be equal,
            # so every run in a full-waveform sweep has the same row shape.
            if collapsed:
                entry["value"] = peak
            else:
                entry["peak"] = peak
                entry["mean"] = mean_val
                entry["min"] = min_val
            per_run_summaries.append(entry)
            peak_values.append(peak)

        except Exception:
            # Skip runs where signal can't be read (wrong signal name, corrupt file)
            continue

    # Aggregate stats across runs (non-finite peaks excluded — see _aggregate_peaks).
    stats, max_case_run, min_case_run = _aggregate_peaks(peak_values, per_run_summaries)

    return {
        "signal": signal,
        "at": at,
        "run_count": len(per_run_summaries),
        "runs": per_run_summaries,
        "step_collapsed_runs": step_collapsed,
        "step_unknown_runs": step_unknown,
        "stats": stats,
        # Neutral naming: "worst"/"best" would assume larger-peak = worse,
        # which has no inherent meaning for an arbitrary signal (e.g. a
        # passband magnitude).
        "max_case_run": max_case_run,
        "min_case_run": min_case_run,
    }


def filter_runs_by_params(
    run_results: dict[int, dict],
    filters: dict[str, str],
) -> list[int]:
    """Filter batch run indices by parameter values.

    For each filter key-value pair, checks if the run's params match.

    Filter value formats:
        - Exact:  "1k"      — parse via parse_spice_value(), compare with 1e-6 rel. tol.
        - Range:  "1k..5k"  — split on "..", parse both bounds, check lo <= val <= hi
        - String: if parse_spice_value() fails, fall back to str equality

    Args:
        run_results: Dict mapping run_index -> {raw_file, log_file, params}
        filters: Dict mapping param name -> filter expression string

    Returns:
        Sorted list of run indices matching ALL filters
    """
    matching = []

    for run_index in sorted(run_results.keys()):
        run = run_results[run_index]
        params = run.get("params", {})
        # Case-insensitive name match: run params carry the netlist's casing
        # ("R1"), user filters often arrive lowercase ("r1") — an exact-key
        # lookup silently matched zero runs (same failure shape as the
        # case-sensitive W/L lookup fixed in the Monte Carlo engine).
        params_ci = {str(k).lower(): v for k, v in params.items()}
        all_match = True

        for param_name, filter_expr in filters.items():
            if param_name.lower() not in params_ci:
                all_match = False
                break

            run_value = params_ci[param_name.lower()]

            if ".." in filter_expr:
                # Range filter
                parts = filter_expr.split("..", 1)
                lo_str, hi_str = parts[0].strip(), parts[1].strip()
                try:
                    lo = parse_spice_value(lo_str)
                    hi = parse_spice_value(hi_str)
                    try:
                        run_numeric = float(run_value)
                    except (TypeError, ValueError):
                        all_match = False
                        break
                    # NaN bounds or NaN values never match a range
                    if (
                        math.isnan(lo)
                        or math.isnan(hi)
                        or math.isnan(run_numeric)
                        or not (lo <= run_numeric <= hi)
                    ):
                        all_match = False
                        break
                except ValueError:
                    # Non-numeric range filter — fall back to string equality
                    if str(run_value) != filter_expr:
                        all_match = False
                        break
            else:
                # Exact filter
                try:
                    target = parse_spice_value(filter_expr)
                    try:
                        run_numeric = float(run_value)
                    except (TypeError, ValueError):
                        all_match = False
                        break
                    # NaN never matches anything (including itself)
                    if math.isnan(target) or math.isnan(run_numeric):
                        all_match = False
                        break
                    # Compare with relative tolerance of 1e-6
                    if target == 0.0:
                        if run_numeric != 0.0:
                            all_match = False
                            break
                    else:
                        if abs(run_numeric - target) / abs(target) > 1e-6:
                            all_match = False
                            break
                except ValueError:
                    # Non-numeric filter — string equality
                    if str(run_value) != filter_expr:
                        all_match = False
                        break

        if all_match:
            matching.append(run_index)

    return matching


def get_progress_snapshot(batch_job: BatchJob, start_time: float) -> dict:
    """Return a progress snapshot for a running batch job.

    Computes elapsed time and estimates remaining time (ETA) based on
    the current completion rate.

    Args:
        batch_job: The BatchJob to snapshot
        start_time: The batch job wall-clock start time (time.time())

    Returns:
        Dict with:
            completed: int — number of completed runs
            total: int — total runs in batch
            failed: int — number of failed runs
            elapsed_s: float — seconds since start_time
            eta_s: float | None — estimated seconds remaining (None if no runs done yet)
    """
    # Clamp elapsed to >= 0 in case the wall clock moved backwards or
    # start_time was set in the future for some reason.
    elapsed = max(0.0, time.time() - start_time)
    completed = batch_job.completed_runs
    total = batch_job.total_runs
    failed = batch_job.failed_runs

    if completed > 0 and elapsed > 0:
        rate = completed / elapsed  # runs per second
        remaining = max(0, total - completed)  # don't go negative on overshoot
        eta_s = remaining / rate if rate > 0 else None
    else:
        eta_s = None

    return {
        "completed": completed,
        "total": total,
        "failed": failed,
        "elapsed_s": float(elapsed),
        "eta_s": float(eta_s) if eta_s is not None else None,
    }
