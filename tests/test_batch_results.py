"""Tests for batch_results: filtering, progress, and stats."""

import time
from pathlib import Path

from ltspice_mcp.lib.batch_results import (
    compute_batch_stats,
    filter_runs_by_params,
    get_progress_snapshot,
)
from ltspice_mcp.state import BatchJob


def _make_run(params: dict, raw_file: str = "") -> dict:
    return {"raw_file": raw_file, "log_file": "", "params": params}


class TestFilterRunsByParams:
    def test_filter_exact_spice_value(self):
        runs = {0: _make_run({"R": 1000.0})}
        result = filter_runs_by_params(runs, {"R": "1k"})
        assert result == [0]

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
        assert result["worst_case_run"] is None

    def test_stats_missing_raw_files(self):
        runs = {
            0: _make_run({"R": 1000.0}, raw_file="/nonexistent/run0.raw"),
            1: _make_run({"R": 2000.0}, raw_file=""),
        }
        result = compute_batch_stats(runs, "V(out)")
        assert result["run_count"] == 0  # all skipped
