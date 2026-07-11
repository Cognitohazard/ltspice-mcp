"""Tests for batch_results: filtering, progress, and stats."""

import math
import time
from pathlib import Path

from ltspice_mcp.lib.batch_results import (
    _aggregate_peaks,
    compute_batch_stats,
    filter_runs_by_params,
    get_progress_snapshot,
)
from ltspice_mcp.state import BatchJob
from tests.conftest import FIXTURES_DIR


def _make_run(params: dict, raw_file: str = "") -> dict:
    return {"raw_file": raw_file, "log_file": "", "params": params}


class TestAggregatePeaks:
    """Across-run aggregation must not be NaN-poisoned by a diverged run."""

    def test_non_finite_peak_excluded(self) -> None:
        # A diverged-but-completed run (NaN peak) must not poison the aggregate,
        # nor make both case pointers aim at the diverged run.
        summaries = [{"run_index": 0}, {"run_index": 1}, {"run_index": 2}]
        stats, max_run, min_run = _aggregate_peaks([5.0, float("nan"), 9.0], summaries)
        assert stats["max_across_runs"] == 9.0
        assert stats["min_across_runs"] == 5.0
        assert math.isfinite(stats["mean_across_runs"])
        assert max_run == 2  # finite max, not the NaN run
        assert min_run == 0

    def test_inf_peak_excluded(self) -> None:
        summaries = [{"run_index": 0}, {"run_index": 1}]
        stats, max_run, _ = _aggregate_peaks([float("inf"), 3.0], summaries)
        assert stats["max_across_runs"] == 3.0
        assert max_run == 1

    def test_all_non_finite_yields_none(self) -> None:
        summaries = [{"run_index": 0}, {"run_index": 1}]
        stats, max_run, min_run = _aggregate_peaks([float("nan"), float("inf")], summaries)
        assert stats["max_across_runs"] is None
        assert max_run is None and min_run is None

    def test_empty_yields_none(self) -> None:
        stats, max_run, min_run = _aggregate_peaks([], [])
        assert stats["mean_across_runs"] is None
        assert max_run is None and min_run is None


class TestFilterRunsByParams:
    def test_filter_exact_spice_value(self):
        runs = {0: _make_run({"R": 1000.0})}
        result = filter_runs_by_params(runs, {"R": "1k"})
        assert result == [0]

    def test_filter_param_name_case_insensitive(self):
        # Run params carry the netlist's casing ("R1"); user filters often
        # arrive lowercase. An exact-key match silently returned zero runs.
        runs = {0: _make_run({"R1": 1000.0}), 1: _make_run({"R1": 2000.0})}
        assert filter_runs_by_params(runs, {"r1": "1k"}) == [0]
        assert filter_runs_by_params(runs, {"R1": "2k"}) == [1]

    def test_filter_range(self):
        runs = {
            0: _make_run({"R": 1000.0}),
            1: _make_run({"R": 3000.0}),
            2: _make_run({"R": 5000.0}),
        }
        result = filter_runs_by_params(runs, {"R": "1k..5k"})
        assert result == [0, 1, 2]

    def test_filter_range_excludes(self):
        runs = {0: _make_run({"R": 6000.0})}
        result = filter_runs_by_params(runs, {"R": "1k..5k"})
        assert result == []

    def test_filter_string_fallback(self):
        runs = {
            0: _make_run({"model": "NPN"}),
            1: _make_run({"model": "PNP"}),
        }
        result = filter_runs_by_params(runs, {"model": "NPN"})
        assert result == [0]

    def test_filter_multiple_params(self):
        runs = {
            0: _make_run({"R": 1000.0, "C": 1e-9}),
            1: _make_run({"R": 1000.0, "C": 1e-6}),
        }
        result = filter_runs_by_params(runs, {"R": "1k", "C": "1n"})
        assert result == [0]

    def test_filter_missing_param(self):
        runs = {0: _make_run({"R": 1000.0})}
        result = filter_runs_by_params(runs, {"C": "100n"})
        assert result == []

    def test_filter_empty_results(self):
        result = filter_runs_by_params({}, {"R": "1k"})
        assert result == []


class TestGetProgressSnapshot:
    def _make_batch_job(self, total: int, completed: int, failed: int = 0) -> BatchJob:
        return BatchJob(
            job_id="test",
            job_type="sweep",
            netlist=Path("/tmp/test.cir"),
            total_runs=total,
            completed_runs=completed,
            failed_runs=failed,
        )

    def test_progress_no_runs_done(self):
        job = self._make_batch_job(total=10, completed=0)
        snap = get_progress_snapshot(job, time.time() - 5.0)
        assert snap["eta_s"] is None
        assert snap["completed"] == 0

    def test_progress_with_runs(self):
        job = self._make_batch_job(total=10, completed=5)
        start = time.time() - 10.0  # 10s elapsed, 5 done → 2s/run → 5 remaining → ~10s ETA
        snap = get_progress_snapshot(job, start)
        assert snap["eta_s"] is not None
        assert 8.0 < snap["eta_s"] < 12.0  # some tolerance for timing

    def test_progress_all_done(self):
        job = self._make_batch_job(total=10, completed=10)
        start = time.time() - 5.0
        snap = get_progress_snapshot(job, start)
        assert snap["eta_s"] is not None
        assert snap["eta_s"] < 0.1  # remaining ≈ 0


class TestComputeBatchStats:
    def test_stats_empty_runs(self):
        result = compute_batch_stats({}, "V(out)")
        assert result["run_count"] == 0
        assert result["stats"]["max_across_runs"] is None
        assert result["max_case_run"] is None

    def test_stats_missing_raw_files(self):
        runs = {
            0: _make_run({"R": 1000.0}, raw_file="/nonexistent/run0.raw"),
            1: _make_run({"R": 2000.0}, raw_file=""),
        }
        result = compute_batch_stats(runs, "V(out)")
        assert result["run_count"] == 0  # all skipped


class TestComputeBatchStatsAt:
    """``at`` slices each run to a single point on the axis before
    aggregating. For AC sweeps the per-run peak across all frequencies
    conflated low-frequency roll-off (Cin high-pass corner) with run-to-run
    variation — ``at`` lets the caller ask "what's the spread of |H| at
    this specific frequency across runs?"."""

    def _write_transient_raw(
        self, path: Path, axis: list[float], waves: dict[str, list[float]]
    ) -> None:
        from spicelib import RawWrite, Trace

        rw = RawWrite()
        rw.add_trace(Trace("time", axis))
        for name, vals in waves.items():
            rw.add_trace(Trace(name, vals))
        rw.save(str(path))

    def test_at_slices_to_single_point(self, tmp_path: Path):
        # Each run has a different value at t=2.0; ``at`` picks that point.
        run0 = tmp_path / "r0.raw"
        run1 = tmp_path / "r1.raw"
        self._write_transient_raw(run0, [0.0, 1.0, 2.0, 3.0], {"V(out)": [0.0, 5.0, 10.0, 5.0]})
        self._write_transient_raw(run1, [0.0, 1.0, 2.0, 3.0], {"V(out)": [0.0, 3.0, 6.0, 3.0]})

        runs = {
            0: _make_run({"R": 1000.0}, raw_file=str(run0)),
            1: _make_run({"R": 2000.0}, raw_file=str(run1)),
        }

        # Without ``at``: per-run peak across all time → 10.0 vs 6.0
        no_at = compute_batch_stats(runs, "V(out)")
        assert no_at["stats"]["max_across_runs"] == 10.0
        assert no_at["stats"]["min_across_runs"] == 6.0

        # With ``at=2.0``: each run reduced to the value at that time. When
        # peak/mean/min would collapse, the row drops them and
        # surfaces just ``value``.
        sliced = compute_batch_stats(runs, "V(out)", at=2.0)
        assert sliced["at"] == 2.0
        assert sliced["runs"][0]["value"] == 10.0
        assert "peak" not in sliced["runs"][0]
        assert "mean" not in sliced["runs"][0]
        assert "min" not in sliced["runs"][0]
        assert sliced["runs"][1]["value"] == 6.0
        assert sliced["stats"]["max_across_runs"] == 10.0
        assert sliced["stats"]["min_across_runs"] == 6.0

    def test_at_handles_descending_sweep_axis(self, tmp_path: Path):
        # A DC/param sweep can run high->low (.dc V1 5 0 -0.1). The ``at=``
        # slice must pick the sample at that axis value, not silently
        # mis-index because searchsorted assumes ascending order. Axis
        # [3,2,1,0]: a naive searchsorted lands past the end and returns the
        # value at 0.0 (=0.0) instead of the value at 2.0 (=20.0).
        run0 = tmp_path / "desc.raw"
        self._write_transient_raw(run0, [3.0, 2.0, 1.0, 0.0], {"V(out)": [30.0, 20.0, 10.0, 0.0]})
        runs = {0: _make_run({"R": 1000.0}, raw_file=str(run0))}
        sliced = compute_batch_stats(runs, "V(out)", at=2.0)
        assert sliced["runs"][0]["value"] == 20.0

    def test_inner_step_sweep_is_surfaced_not_silently_collapsed(self):
        # A run whose raw carries its own .step sweep is read at step 0 only.
        # Dropping the other steps silently would be invisible wrong data, so
        # the run index is surfaced for the caller to handle per-step.
        stepped = FIXTURES_DIR / "ltspice_step_tran.raw"
        runs = {0: _make_run({"R": 1000.0}, raw_file=str(stepped))}
        result = compute_batch_stats(runs, "V(out)")
        assert result["step_collapsed_runs"] == [0]

    def test_step_detection_failure_is_surfaced_not_swallowed(self, tmp_path: Path, monkeypatch):
        # If step metadata can't be read, step 0 is still returned — but the run
        # must be flagged, not silently treated as single-step (which would
        # recreate the silent step-0 reduction this guard exists to remove).
        import ltspice_mcp.lib.batch_results as br

        run0 = tmp_path / "ok.raw"
        self._write_transient_raw(run0, [0.0, 1.0, 2.0, 3.0], {"V(out)": [0.0, 5.0, 10.0, 5.0]})

        def boom(self):
            raise RuntimeError("unreadable step metadata")

        monkeypatch.setattr(br.OffsetAwareRawRead, "get_steps", boom)
        runs = {0: _make_run({"R": 1000.0}, raw_file=str(run0))}
        result = compute_batch_stats(runs, "V(out)")
        assert result["step_unknown_runs"] == [0]
        assert result["step_collapsed_runs"] == []
        # The data we COULD read (step 0) is still returned, not dropped.
        assert result["run_count"] == 1

    def test_constant_waveform_run_keeps_peak_mean_min_shape(self, tmp_path: Path):
        # Row shape is decided by an explicit point-query flag, NOT by whether
        # peak/mean/min happen to be equal. A flat full-waveform run (at=None)
        # has peak==mean==min by value, but it is still a waveform and must
        # keep the {peak,mean,min} trio so every row in a sweep has the same
        # shape — it must NOT collapse to {value} like a genuine point query.
        const_run = tmp_path / "const.raw"
        vary_run = tmp_path / "vary.raw"
        self._write_transient_raw(
            const_run, [0.0, 1.0, 2.0, 3.0], {"V(out)": [5.0, 5.0, 5.0, 5.0]}
        )
        self._write_transient_raw(vary_run, [0.0, 1.0, 2.0, 3.0], {"V(out)": [1.0, 4.0, 8.0, 2.0]})

        runs = {
            0: _make_run({"R": 1000.0}, raw_file=str(const_run)),
            1: _make_run({"R": 2000.0}, raw_file=str(vary_run)),
        }

        # No ``at`` → full-waveform aggregation. The constant run keeps the
        # same row shape as the varying run.
        result = compute_batch_stats(runs, "V(out)")
        assert result["at"] is None
        assert result["run_count"] == 2

        const_row = result["runs"][0]
        vary_row = result["runs"][1]

        # Constant run: trio present, all equal to the flat value; no ``value``.
        assert const_row["peak"] == 5.0
        assert const_row["mean"] == 5.0
        assert const_row["min"] == 5.0
        assert "value" not in const_row

        # Varying run: same shape (trio present, no ``value``).
        assert "peak" in vary_row
        assert "mean" in vary_row
        assert "min" in vary_row
        assert "value" not in vary_row

        # Both rows expose exactly the same keys → uniform sweep row shape.
        assert const_row.keys() == vary_row.keys()
