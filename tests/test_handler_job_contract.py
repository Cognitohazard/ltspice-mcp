"""Cross-handler job CONTRACT tests — the parametrized variant matrix that
closes the ``single_variant`` / ``bypass_wiring`` class the seam audit flagged.

Two invariants every job-aware tool handler must satisfy, asserted through the
REAL handlers (not internals), so a fix that lands in a leaf without wiring the
entry path fails here:

1. STATUS COMPLETENESS — a terminal batch status (incl. ``interrupted``, assigned
   on restart recovery) must format without raising "unexpected status". This is
   the exact shape of the B1 bug (a hardcoded status allowlist that omitted
   ``interrupted``); parametrizing over every terminal status catches the whole
   class, not the one instance.

2. DUAL-STORE RESOLUTION — a handler that takes a ``job_id`` must resolve BOTH a
   single-sim job (``state.jobs``) and a batch job (``state.batch_jobs``), never
   reject one store's ids as "not found". This is the shape of the cancel_job
   bug (it resolved only the single-sim store).
"""

from datetime import timedelta
from pathlib import Path

import pytest
from mcp import types

from ltspice_mcp.lib import now
from ltspice_mcp.state import TERMINAL_STATUSES, BatchJob, SessionState, SimulationJob
from ltspice_mcp.tools.advanced import GetBatchResultsInput, handle_batch_results
from ltspice_mcp.tools.simulation import CheckJobInput, handle_check_job

pytestmark = pytest.mark.asyncio

# Terminal statuses a BatchJob can actually hold: TERMINAL_STATUSES minus
# 'timeout' (single-sim only, not in BatchJob.status' Literal). Derived so a new
# terminal status auto-propagates into this contract instead of escaping it.
BATCH_TERMINAL_STATUSES = sorted(TERMINAL_STATUSES - {"timeout"})

# Single-sim terminal statuses check_job can format WITHOUT result files on disk.
# 'completed' needs a real raw/log (→ covered end-to-end by the ngspice e2e tier),
# so it's excluded here. Derived so a new terminal status is caught automatically.
SINGLE_TERMINAL_STATUSES_NO_FILES = sorted(TERMINAL_STATUSES - {"completed"})


def _text(result) -> str:
    item = result.content[0]
    assert isinstance(item, types.TextContent)
    return item.text


def _make_batch(state: SessionState, *, status: str, job_id: str = "b1") -> BatchJob:
    bj = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=Path("/tmp/x.cir"),
        total_runs=4,
        completed_runs=2,
        failed_runs=0,
        status=status,  # type: ignore[arg-type]
    )
    if status == "completed":
        bj.completed_at = bj.started_at + timedelta(seconds=3)
    state.add_batch_job(bj)
    return bj


def _make_sim(state: SessionState, *, status: str, job_id: str = "j1") -> SimulationJob:
    job = SimulationJob(
        job_id=job_id,
        netlist=Path("/tmp/x.cir"),
        simulator="Sim",
        status=status,  # type: ignore[arg-type]
        started_at=now(),
    )
    state.jobs[job_id] = job
    return job


class TestBatchStatusCompleteness:
    """batch_results must format EVERY terminal batch status without raising
    'unexpected status'. Regression class: B1 (interrupted) — the formatter's
    status allowlist omitted a real terminal status."""

    @pytest.mark.parametrize("status", BATCH_TERMINAL_STATUSES)
    async def test_batch_results_handles_terminal_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_batch(state_no_sim, status=status)
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        text = _text(result).lower()
        assert "unexpected status" not in text
        # the status is surfaced to the caller (not swallowed)
        assert status in text


class TestSingleSimStatusCompleteness:
    """check_job must format EVERY terminal single-sim status without falling
    through to 'unexpected status'. Same class as the batch B1 bug, one store
    over: the single-sim formatter omitted 'interrupted' (assigned on restart
    recovery, e.g. a job whose raw didn't survive to be promoted to completed)."""

    @pytest.mark.parametrize("status", SINGLE_TERMINAL_STATUSES_NO_FILES)
    async def test_check_job_handles_terminal_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_sim(state_no_sim, status=status)
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "unexpected status" not in _text(result).lower()


class TestJobHandlerDualStore:
    """A job_id-taking handler must resolve BOTH job stores. Regression class:
    cancel_job resolved only the single-sim store and rejected batch ids."""

    async def test_check_job_resolves_single_sim_job(self, state_no_sim: SessionState):
        _make_sim(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(job_id="j1"), state_no_sim)
        assert "not found" not in _text(result).lower()

    async def test_check_job_resolves_batch_job(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="running")
        result = await handle_check_job(CheckJobInput(job_id="b1"), state_no_sim)
        assert "not found" not in _text(result).lower()
