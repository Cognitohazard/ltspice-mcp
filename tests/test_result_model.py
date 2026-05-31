"""Tests for the unified single-run/batch result read-model (runs_of/RunRef).

The read-model treats a single-run job as a batch-of-one so every result
extraction routine can be written once against ``RunRef``. Covers:
- ``runs_of`` over both job shapes + empty-path normalization,
- ``resolve_run`` index bounds,
- ``resolve_raw_file``/``resolve_log_file`` reaching an arbitrary run index.
"""

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import now, services
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob
from ltspice_mcp.tools.analysis import (
    BodeMetricsInput,
    QueryValueInput,
    SimulationSummaryInput,
    handle_bode_metrics,
    handle_query_value,
    handle_simulation_summary,
)


def _inject_raw(state: SessionState, path: Path, raw: MagicMock) -> None:
    path.write_bytes(b"placeholder")
    state.results.set(path, raw)


def _sim(state: SessionState, *, raw=None, log=None, status="completed") -> SimulationJob:
    job = SimulationJob(
        job_id="j1",
        netlist=Path("/tmp/t.cir"),
        simulator="FakeSim",
        status=status,  # type: ignore[arg-type]
        started_at=now(),
        completed_at=now() + timedelta(seconds=1),
        raw_file=raw,
        log_file=log,
    )
    state.jobs["j1"] = job
    return job


def _batch(state: SessionState, run_results: dict, *, status="completed") -> BatchJob:
    bj = BatchJob(
        job_id="b1",
        job_type="sweep",
        netlist=Path("/tmp/x.cir"),
        total_runs=len(run_results),
        completed_runs=len(run_results),
        status=status,  # type: ignore[arg-type]
    )
    bj.run_results = run_results
    if status == "completed":
        bj.completed_at = bj.started_at + timedelta(seconds=5)
    state.batch_jobs["b1"] = bj
    return bj


class TestRunsOf:
    def test_single_run_is_batch_of_one(self, state_no_sim: SessionState):
        job = _sim(state_no_sim, raw=Path("/tmp/a.raw"), log=Path("/tmp/a.log"))
        runs = services.runs_of(job)
        assert len(runs) == 1
        assert runs[0].index == 0
        assert runs[0].raw_file == Path("/tmp/a.raw")
        assert runs[0].log_file == Path("/tmp/a.log")
        assert runs[0].params == {}

    def test_single_run_empty_paths_normalize_to_none(self, state_no_sim: SessionState):
        # An empty Path coerces to "." — runs_of must surface that as absent.
        job = _sim(state_no_sim, raw=Path(""), log=None)
        runs = services.runs_of(job)
        assert runs[0].raw_file is None
        assert runs[0].log_file is None

    def test_batch_projects_each_run_sorted(self, state_no_sim: SessionState):
        bj = _batch(
            state_no_sim,
            {
                1: {"raw_file": "/tmp/r1.raw", "log_file": "/tmp/r1.log", "params": {"R": "2k"}},
                0: {"raw_file": Path("/tmp/r0.raw"), "log_file": None, "params": {"R": "1k"}},
            },
        )
        runs = services.runs_of(bj)
        assert [r.index for r in runs] == [0, 1]  # sorted by index
        assert runs[0].raw_file == Path("/tmp/r0.raw")
        assert runs[0].params == {"R": "1k"}
        assert runs[1].raw_file == Path("/tmp/r1.raw")  # str coerced to Path
        assert runs[1].log_file == Path("/tmp/r1.log")

    def test_batch_empty_string_path_is_none(self, state_no_sim: SessionState):
        bj = _batch(state_no_sim, {0: {"raw_file": "", "log_file": "", "params": {}}})
        runs = services.runs_of(bj)
        assert runs[0].raw_file is None


class TestResolveRun:
    def test_default_index_zero(self, state_no_sim: SessionState):
        _batch(state_no_sim, {0: {"raw_file": "/tmp/r0.raw", "params": {}}})
        run = services.resolve_run("b1", state_no_sim)
        assert run.index == 0

    def test_reach_run_n(self, state_no_sim: SessionState):
        _batch(
            state_no_sim,
            {
                0: {"raw_file": "/tmp/r0.raw", "params": {}},
                1: {"raw_file": "/tmp/r1.raw", "params": {}},
            },
        )
        run = services.resolve_run("b1", state_no_sim, 1)
        assert run.raw_file == Path("/tmp/r1.raw")

    def test_out_of_range_raises(self, state_no_sim: SessionState):
        _batch(state_no_sim, {0: {"raw_file": "/tmp/r0.raw", "params": {}}})
        with pytest.raises(ResultError, match="out of range"):
            services.resolve_run("b1", state_no_sim, 5)

    def test_single_run_index_above_zero_out_of_range(self, state_no_sim: SessionState):
        _sim(state_no_sim, raw=Path("/tmp/a.raw"))
        with pytest.raises(ResultError, match="out of range"):
            services.resolve_run("j1", state_no_sim, 1)


class TestResolveResultFileByRun:
    def test_resolve_raw_file_run_index(self, state_no_sim: SessionState, tmp_path: Path):
        r0, r1 = tmp_path / "r0.raw", tmp_path / "r1.raw"
        r0.write_text("d")
        r1.write_text("d")
        _batch(
            state_no_sim,
            {
                0: {"raw_file": r0, "log_file": r0, "params": {}},
                1: {"raw_file": r1, "log_file": r1, "params": {}},
            },
        )
        assert services.resolve_raw_file("b1", state_no_sim, 1) == r1
        assert services.resolve_raw_file("b1", state_no_sim) == r0  # default 0

    def test_resolve_raw_file_out_of_range(self, state_no_sim: SessionState, tmp_path: Path):
        r0 = tmp_path / "r0.raw"
        r0.write_text("d")
        _batch(state_no_sim, {0: {"raw_file": r0, "log_file": r0, "params": {}}})
        with pytest.raises(ResultError, match="out of range"):
            services.resolve_raw_file("b1", state_no_sim, 3)

    def test_single_run_run_index_above_zero_rejected(
        self, state_no_sim: SessionState, tmp_path: Path
    ):
        raw = tmp_path / "a.raw"
        raw.write_text("d")
        _sim(state_no_sim, raw=raw, log=raw)
        with pytest.raises(ResultError, match="out of range"):
            services.resolve_raw_file("j1", state_no_sim, 1)


def _stepped_tran_raw() -> MagicMock:
    """A 2-step transient raw whose step 0 and step 1 have different time spans."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "Transient Analysis"
    raw.get_trace_names.return_value = ["time", "V(out)"]
    raw.get_steps.return_value = [0, 1]
    raw.get_axis.side_effect = lambda step=0: (
        np.array([0.0, 1.0]) if step == 0 else np.array([0.0, 5.0])
    )
    return raw


@pytest.mark.asyncio
class TestSimulationSummaryStepAware:
    async def test_summary_reflects_chosen_step(self, state_no_sim: SessionState, work_dir: Path):
        # Seam 3: build_simulation_summary used to hardcode step 0, so the range
        # was always step 0's. It must now reflect args.step.
        path = work_dir / "stepped.raw"
        _inject_raw(state_no_sim, path, _stepped_tran_raw())
        res0 = await handle_simulation_summary(
            SimulationSummaryInput(raw_file="stepped.raw", step=0), state_no_sim
        )
        res1 = await handle_simulation_summary(
            SimulationSummaryInput(raw_file="stepped.raw", step=1), state_no_sim
        )
        assert res0.structuredContent is not None and res1.structuredContent is not None
        assert res0.structuredContent["range"]["time_end"] == 1.0
        assert res1.structuredContent["range"]["time_end"] == 5.0  # step 1, not step 0

    async def test_summary_out_of_range_step_rejected(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = work_dir / "stepped2.raw"
        _inject_raw(state_no_sim, path, _stepped_tran_raw())
        with pytest.raises(ResultError, match="out of range"):
            await handle_simulation_summary(
                SimulationSummaryInput(raw_file="stepped2.raw", step=9), state_no_sim
            )


# ---------------------------------------------------------------------------
# Phase 2 — query_value / bode_metrics address a batch run (job_id + run_index)
# ---------------------------------------------------------------------------


def _tran_raw() -> MagicMock:
    raw = MagicMock()
    raw.get_raw_property.return_value = "Transient Analysis"
    raw.get_trace_names.return_value = ["time", "V(out)"]
    raw.get_steps.return_value = [0]
    raw.get_axis.return_value = np.array([0.0, 1.0])
    raw.get_wave.return_value = np.array([1.0, 2.0])
    return raw


def _ac_raw_lpf(fc: float) -> MagicMock:
    raw = MagicMock()
    raw.get_raw_property.return_value = "AC Analysis"
    raw.get_trace_names.return_value = ["frequency", "V(out)"]
    freq = np.logspace(0, 5, 200)
    H = 1.0 / (1.0 + 1j * (freq / fc))
    raw.get_axis.return_value = freq
    raw.get_steps.return_value = [0]
    raw.get_wave = lambda name, step=0: H
    return raw


@pytest.mark.asyncio
class TestQueryValueJobRun:
    async def test_query_run_by_index(self, state_no_sim: SessionState, work_dir: Path):
        p0, p1 = work_dir / "run0.raw", work_dir / "run1.raw"
        _inject_raw(state_no_sim, p0, _tran_raw())
        _inject_raw(state_no_sim, p1, _tran_raw())
        _batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        res = await handle_query_value(
            QueryValueInput(job_id="b1", run_index=1, signal="V(out)", at="0.5"),
            state_no_sim,
        )
        assert res.structuredContent is not None

    async def test_raw_file_and_job_id_mutually_exclusive(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_query_value(
                QueryValueInput(raw_file="x.raw", job_id="b1", signal="V(out)", at="1"),
                state_no_sim,
            )

    async def test_neither_raw_nor_job(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_query_value(QueryValueInput(signal="V(out)", at="1"), state_no_sim)

    async def test_step_axis_with_job_id_rejected(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="can't be combined with 'job_id'"):
            await handle_query_value(
                QueryValueInput(job_id="b1", step_axis="R", step_value="1k", signal="V(out)"),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestBodeMetricsJobRun:
    async def test_bode_run_by_index_tracks_that_run(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        p0, p1 = work_dir / "b_run0.raw", work_dir / "b_run1.raw"
        _inject_raw(state_no_sim, p0, _ac_raw_lpf(500.0))
        _inject_raw(state_no_sim, p1, _ac_raw_lpf(5000.0))
        _batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        res = await handle_bode_metrics(
            BodeMetricsInput(
                job_id="b1",
                run_index=1,
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        cs = res.structuredContent["crossings"]
        assert cs and abs(cs[0]["frequency_hz"] - 5000.0) / 5000.0 < 0.05  # run 1's fc

    async def test_bode_raw_and_job_mutually_exclusive(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="x.raw", job_id="b1", signal="V(out)", mode="filter"),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# Review fixes: status gate, non-contiguous range message, empty-string guard
# ---------------------------------------------------------------------------


class TestResolveRunStatusGate:
    def test_running_job_rejected(self, state_no_sim: SessionState):
        _batch(state_no_sim, {0: {"raw_file": "/tmp/r0.raw", "params": {}}}, status="running")
        with pytest.raises(ResultError, match="not completed"):
            services.resolve_run("b1", state_no_sim, 0)

    def test_failed_job_with_partial_runs_rejected(self, state_no_sim: SessionState):
        # A batch that failed mid-run still has valid raws for completed sub-runs;
        # reading them via job_id must NOT silently succeed — same gate as
        # resolve_raw_file, so all job_id-addressed reads behave identically.
        _batch(state_no_sim, {0: {"raw_file": "/tmp/r0.raw", "params": {}}}, status="failed")
        with pytest.raises(ResultError, match="not completed"):
            services.resolve_run("b1", state_no_sim, 0)

    def test_noncontiguous_range_message_lists_actual_indices(self, state_no_sim: SessionState):
        # After a mid-batch failure run_results can be non-contiguous (gap at 2).
        # The error must list the indices actually present, not "0..3".
        _batch(
            state_no_sim,
            {
                0: {"raw_file": "/tmp/r0.raw", "params": {}},
                1: {"raw_file": "/tmp/r1.raw", "params": {}},
                3: {"raw_file": "/tmp/r3.raw", "params": {}},
            },
        )
        with pytest.raises(ResultError, match=r"valid indices: \[0, 1, 3\]"):
            services.resolve_run("b1", state_no_sim, 2)


@pytest.mark.asyncio
class TestEmptyRawFileGuard:
    async def test_query_value_empty_raw_file(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_query_value(
                QueryValueInput(raw_file="", signal="V(out)", at="1"), state_no_sim
            )

    async def test_query_value_whitespace_raw_file(self, state_no_sim: SessionState):
        # StrictModel strips to "" — must still be treated as absent.
        with pytest.raises(ResultError, match="exactly one"):
            await handle_query_value(
                QueryValueInput(raw_file="  ", signal="V(out)", at="1"), state_no_sim
            )

    async def test_bode_metrics_empty_raw_file(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="", signal="V(out)", mode="filter"), state_no_sim
            )

    async def test_query_value_job_id_on_running_batch_rejected(self, state_no_sim: SessionState):
        _batch(state_no_sim, {0: {"raw_file": "/tmp/r0.raw", "params": {}}}, status="running")
        with pytest.raises(ResultError, match="not completed"):
            await handle_query_value(
                QueryValueInput(job_id="b1", signal="V(out)", at="1"), state_no_sim
            )
