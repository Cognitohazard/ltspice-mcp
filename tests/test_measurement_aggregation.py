"""measurement_stats job-id aggregation over real per-run LTspice logs.

The ``job_id`` branch of measurement_stats resolves any job: a batch job's
``run_results`` are walked (one .log per run, as recorded by the sweep/MC
runners) and the per-run .MEAS scalars aggregated; a completed single-sim
job aggregates its one log. The fixture logs were recorded from a
real 3-run LTspice parameter sweep of an RC low-pass (R1 = 1k / 2.2k / 4.7k,
C = 100n, 1 V step input) with two measurements in the deck:

    .meas tran vfinal FIND V(out) AT=0.9m     (value varies per run)
    .meas tran tcross WHEN V(out)=0.5         (level constant, time varies)

The WHEN-style measurement exercises the axis swap: its per-run scalar is the
constant trigger level 0.5, so the aggregator must switch to the folded ``at``
field (the crossing time) instead.
"""

from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import BatchJob, SessionState, SimulationJob
from ltspice_mcp.tools.analysis import MeasurementStatsInput, handle_measurement_stats
from tests.conftest import (
    FIXTURES_DIR,
    LTSPICE_SWEEP_RUN_LOGS,
    LTSPICE_TRAN_RC_LOG,
    LTSPICE_TRAN_RC_VFINAL,
    make_batch_job,
    make_sim_job,
)

# R1 of sweep runs 0 / 1 / 2 (one LTSPICE_SWEEP_RUN_LOGS entry per run).
RUN_PARAMS = [1000.0, 2200.0, 4700.0]

# Values printed by LTspice in each per-run log (vfinal: FIND V(out) AT=0.9m).
VFINAL = [0.999876166042, 0.98323999039, 0.852486569628]
# Crossing times printed by LTspice (tcross: WHEN V(out)=0.5 -> "... AT <t>").
TCROSS_AT = [6.98285618328e-05, 0.000152998664495, 0.000326276827772]


def _make_sweep_batch(
    state: SessionState,
    *,
    job_id: str = "sweep1",
    log_files: list[Path | None] | None = None,
) -> BatchJob:
    """Completed 3-run sweep whose run_results mirror the runner's key shape.

    Each entry carries ``raw_file``/``log_file``/``params`` exactly as
    ``_record_run_completion`` stores them (string paths, 0-based int keys).
    """
    logs = LTSPICE_SWEEP_RUN_LOGS if log_files is None else log_files
    bj = make_batch_job(
        job_id,
        netlist=Path("/tmp/sweep_meas_rc.cir"),
        total_runs=len(logs),
        completed_runs=len(logs),
    )
    for i, log in enumerate(logs):
        entry: dict = {
            "raw_file": str(FIXTURES_DIR / f"unused_run{i}.raw"),
            "params": {"R1": RUN_PARAMS[i]},
        }
        if log is not None:
            entry["log_file"] = str(log)
        bj.run_results[i] = entry
    state.batch_jobs[job_id] = bj
    return bj


async def _stats(state: SessionState, job_id: str = "sweep1") -> dict:
    result = await handle_measurement_stats(MeasurementStatsInput(job_id=job_id), state)
    assert result.structuredContent is not None
    return result.structuredContent["stats"]


def _text(result) -> str:
    """Concatenated text of every content item in a CallToolResult."""
    return "".join(getattr(c, "text", "") for c in (result.content or []))


@pytest.mark.asyncio
class TestJobAggregation:
    async def test_find_style_aggregates_on_value_axis(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        stats = await _stats(state_no_sim)

        entry = stats["vfinal"]
        assert entry["aggregated_field"] == "value"
        assert entry["total_count"] == 3
        assert entry["valid_count"] == 3
        assert entry["failure_count"] == 0
        assert entry["min"] == pytest.approx(min(VFINAL))
        assert entry["max"] == pytest.approx(max(VFINAL))
        assert entry["mean"] == pytest.approx(sum(VFINAL) / 3)
        assert entry["median"] == pytest.approx(VFINAL[1])
        # Largest R -> slowest charge -> smallest V(out) at 0.9 ms.
        assert entry["best_step_index"] == 2
        assert entry["worst_step_index"] == 0

    async def test_when_style_swaps_to_at_axis(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        stats = await _stats(state_no_sim)

        entry = stats["tcross"]
        # The per-run scalar is the constant trigger level (0.5 in every
        # log); the aggregator must report the crossing TIMES instead.
        assert entry["aggregated_field"] == "at"
        assert entry["valid_count"] == 3
        assert entry["min"] == pytest.approx(TCROSS_AT[0])
        assert entry["max"] == pytest.approx(TCROSS_AT[2])
        assert entry["mean"] == pytest.approx(sum(TCROSS_AT) / 3)
        assert entry["median"] == pytest.approx(TCROSS_AT[1])
        assert entry["best_step_index"] == 0
        assert entry["worst_step_index"] == 2
        # Physics cross-check: crossing time tracks R*C*ln(2) (within the
        # 1 us source rise time + solver step).
        for t, r in zip(TCROSS_AT, RUN_PARAMS, strict=True):
            assert t == pytest.approx(r * 100e-9 * 0.6931, rel=0.02, abs=1.5e-6)

    async def test_run_count_in_text_output(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        assert "3 run(s)" in _text(result)

    async def test_missing_run_log_is_skipped(self, state_no_sim: SessionState):
        # Run 1's log file does not exist on disk: the aggregator skips that
        # run silently and computes stats over the remaining two.
        _make_sweep_batch(
            state_no_sim,
            log_files=[
                LTSPICE_SWEEP_RUN_LOGS[0],
                Path("/nonexistent/run1.log"),
                LTSPICE_SWEEP_RUN_LOGS[2],
            ],
        )
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="sweep1"), state_no_sim
        )
        assert result.structuredContent is not None
        stats = result.structuredContent["stats"]

        entry = stats["vfinal"]
        assert entry["total_count"] == 2
        assert entry["valid_count"] == 2
        assert entry["min"] == pytest.approx(VFINAL[2])
        assert entry["max"] == pytest.approx(VFINAL[0])
        assert entry["mean"] == pytest.approx((VFINAL[0] + VFINAL[2]) / 2)
        # Level still constant across the surviving runs, times still vary:
        # the at-axis swap holds for the partial aggregate too.
        tcross = stats["tcross"]
        assert tcross["aggregated_field"] == "at"
        assert tcross["min"] == pytest.approx(TCROSS_AT[0])
        assert tcross["max"] == pytest.approx(TCROSS_AT[2])

        assert "2 run(s)" in _text(result)

    async def test_entry_without_log_file_key_is_skipped(self, state_no_sim: SessionState):
        _make_sweep_batch(
            state_no_sim,
            log_files=[LTSPICE_SWEEP_RUN_LOGS[0], None, LTSPICE_SWEEP_RUN_LOGS[2]],
        )
        stats = await _stats(state_no_sim)
        assert stats["vfinal"]["valid_count"] == 2
        assert stats["vfinal"]["mean"] == pytest.approx((VFINAL[0] + VFINAL[2]) / 2)

    async def test_all_logs_missing_errors(self, state_no_sim: SessionState):
        _make_sweep_batch(
            state_no_sim,
            log_files=[Path("/nonexistent/a.log"), Path("/nonexistent/b.log")],
        )
        with pytest.raises(ResultError, match=r"No \.MEAS results found across the runs"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)

    async def test_no_completed_runs_errors(self, state_no_sim: SessionState):
        bj = _make_sweep_batch(state_no_sim, log_files=[])
        bj.run_results = {}
        with pytest.raises(ResultError, match="no completed runs"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="sweep1"), state_no_sim)


def _make_single_sim(
    state: SessionState,
    *,
    job_id: str = "single1",
    status: str = "completed",
    log_file: Path | None = LTSPICE_TRAN_RC_LOG,
) -> SimulationJob:
    job = make_sim_job(job_id, status=status, log_file=log_file)
    state.jobs[job_id] = job
    return job


@pytest.mark.asyncio
class TestSingleSimJobAggregation:
    """A completed single-simulation job id aggregates its own log — the same
    physical shape as the ``log_file`` input. A plain (non-.step) run yields
    honest n=1 stats."""

    async def test_completed_single_sim_yields_n1_stats(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim)
        result = await handle_measurement_stats(
            MeasurementStatsInput(job_id="single1"), state_no_sim
        )
        assert result.structuredContent is not None
        entry = result.structuredContent["stats"]["vfinal"]
        assert entry["aggregated_field"] == "value"
        assert entry["total_count"] == 1
        assert entry["valid_count"] == 1
        assert entry["failure_count"] == 0
        assert entry["min"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        assert entry["max"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        assert entry["mean"] == pytest.approx(LTSPICE_TRAN_RC_VFINAL)
        # The single log reads as one step, matching the log_file branch.
        assert "1 step(s)" in _text(result)

    async def test_running_single_sim_is_gated_on_completion(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim, status="running")
        with pytest.raises(ResultError, match=r"Job 'single1' is not completed"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="single1"), state_no_sim)

    async def test_completed_single_sim_without_log_errors(self, state_no_sim: SessionState):
        _make_single_sim(state_no_sim, log_file=None)
        with pytest.raises(ResultError, match="has no log file"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="single1"), state_no_sim)


@pytest.mark.asyncio
class TestJobResolutionErrors:
    async def test_unknown_job_id(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="Job not found: nope"):
            await handle_measurement_stats(MeasurementStatsInput(job_id="nope"), state_no_sim)

    async def test_log_file_and_job_id_mutually_exclusive(self, state_no_sim: SessionState):
        _make_sweep_batch(state_no_sim)
        with pytest.raises(ResultError, match="not both"):
            await handle_measurement_stats(
                MeasurementStatsInput(job_id="sweep1", log_file=str(LTSPICE_SWEEP_RUN_LOGS[0])),
                state_no_sim,
            )
