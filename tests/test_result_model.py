"""Tests for the unified single-run/batch result read-model (runs_of/RunRef).

The read-model treats a single-run job as a batch-of-one so every result
extraction routine can be written once against ``RunRef``. Covers:
- ``runs_of`` over both job shapes + empty-path normalization,
- ``resolve_run`` index bounds,
- ``resolve_raw_file``/``resolve_log_file`` reaching an arbitrary run index.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import SessionState
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
    async def test_bode_raw_and_job_mutually_exclusive(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="x.raw", job_id="b1", signal="V(out)", mode="filter"),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# Review fixes: status gate, non-contiguous range message, empty-string guard
# ---------------------------------------------------------------------------


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
