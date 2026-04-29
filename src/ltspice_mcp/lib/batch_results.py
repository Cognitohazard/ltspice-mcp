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
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.state import BatchJob


def compute_batch_stats(
    run_results: dict[int, dict],
    signal: str,
    *,
    at: float | None = None,
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
            worst_case_run: int | None — run with highest peak absolute value
            best_case_run: int | None — run with lowest peak absolute value
            at: float | None — echo of the slicing point (for downstream display)
    """
    per_run_summaries = []
    peak_values: list[float] = []

    for run_index in sorted(run_results.keys()):
        run = run_results[run_index]
        raw_path = run.get("raw_file", "")

        # Skip runs with missing raw files (partial results from cancelled jobs)
        if not raw_path or not Path(raw_path).exists():
            continue

        try:
            raw = RawRead(raw_path, traces_to_read=signal)
            wave = raw.get_wave(signal, step=0)

            # AC (complex): use magnitude; transient: use raw values
            if np.iscomplexobj(wave):
                wave = np.abs(wave)

            point: float | None = None
            try:
                axis = np.asarray(raw.get_axis(step=0))
                axis_size = axis.size
            except Exception:
                # Friction J: ``.op`` raws have no axis; the wave is a single
                # scalar. ``peak``/``mean``/``min`` collapse to one value.
                axis = None  # type: ignore[assignment]
                axis_size = 0

            if axis is None or axis_size == 0:
                if wave.size == 0:
                    continue
                point = float(wave[0])
                peak = mean_val = min_val = point
            elif at is not None:
                if np.iscomplexobj(axis):
                    axis = np.real(axis)
                if axis.size == 0:
                    continue
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
            else:
                peak = float(np.max(wave))
                mean_val = float(np.mean(wave))
                min_val = float(np.min(wave))

            entry: dict = {
                "run_index": run_index,
                "params": run.get("params", {}),
            }
            # When peak/mean/min collapse (point query via ``at=`` or .op
            # single-sample raw), drop the redundant trio and surface just
            # ``value``. Otherwise keep peak/mean/min so callers can see
            # the per-run waveform shape.
            if peak == mean_val == min_val:
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

    # Aggregate stats across runs
    if peak_values:
        peaks_arr = np.array(peak_values)
        stats = {
            "max_across_runs": float(np.max(peaks_arr)),
            "min_across_runs": float(np.min(peaks_arr)),
            "mean_across_runs": float(np.mean(peaks_arr)),
            "std_across_runs": float(np.std(peaks_arr)),
            "median_across_runs": float(np.median(peaks_arr)),
        }
        worst_case_run = per_run_summaries[int(np.argmax(peaks_arr))]["run_index"]
        best_case_run = per_run_summaries[int(np.argmin(peaks_arr))]["run_index"]
    else:
        stats = {
            "max_across_runs": None,
            "min_across_runs": None,
            "mean_across_runs": None,
            "std_across_runs": None,
            "median_across_runs": None,
        }
        worst_case_run = None
        best_case_run = None

    return {
        "signal": signal,
        "at": at,
        "run_count": len(per_run_summaries),
        "runs": per_run_summaries,
        "stats": stats,
        "worst_case_run": worst_case_run,
        "best_case_run": best_case_run,
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
        all_match = True

        for param_name, filter_expr in filters.items():
            if param_name not in params:
                all_match = False
                break

            run_value = params[param_name]

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
