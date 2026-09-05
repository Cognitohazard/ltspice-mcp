"""Cross-handler CONTRACT tests — a parametrized variant matrix that closes
the bug class where a fix lands for one variant (one status, one job store,
one dispatch fork, one file extension) while sibling variants stay broken.

Invariants asserted through the REAL handlers (not internals), so a fix that
lands in a leaf without wiring the entry path fails here:

1. STATUS COMPLETENESS — a terminal batch status (incl. ``interrupted``, assigned
   on restart recovery) must format without raising "unexpected status". This is
   the exact shape of the interrupted-status formatter bug (a hardcoded status
   allowlist that omitted ``interrupted``); parametrizing over every terminal
   status catches the whole class, not the one instance.

2. DUAL-STORE RESOLUTION — a handler that takes a ``job_id`` must resolve BOTH a
   single-sim job (``state.jobs``) and a batch job (``state.batch_jobs``), never
   reject one store's ids as "not found". This is the shape of the cancel_job
   bug (it resolved only the single-sim store).

3. ROUTING-FORK COVERAGE — a handler that forks on job type must reach EVERY
   fork: cancel_job's batch fork has runner-routing tests elsewhere, but the
   single-sim fork was reachable only past guards no unit test crossed.

4. DUAL-DISPATCH (.cir vs .asc) — a circuit tool that accepts both extensions
   must work through BOTH dispatch branches (spice_lex pipeline vs AscEditor),
   and a write must persist to disk, not just to a cached editor.
"""

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp import types

from ltspice_mcp.state import (
    NON_TERMINAL_LIVE_STATUSES,
    TERMINAL_STATUSES,
    BatchJob,
    SessionState,
    SimulationJob,
)
from ltspice_mcp.tools.experiments import JobsInput, handle_jobs
from ltspice_mcp.tools.simulation import (
    CancelJobInput,
    handle_cancel_job,
)
from tests.conftest import (
    FIXTURES_DIR,
    LTSPICE_TRAN_RC_LOG,
    make_batch_job,
    make_sim_job,
)

pytestmark = pytest.mark.asyncio

_FIXTURE_DRAFT = FIXTURES_DIR / "Draft1.asc"

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


def _read_bytes(p: Path) -> bytes:
    return p.read_bytes()


def _make_batch(state: SessionState, *, status: str, job_id: str = "b1") -> BatchJob:
    bj = make_batch_job(job_id, status=status, total_runs=4, completed_runs=2)
    state.add_batch_job(bj)
    return bj


def _make_sim(
    state: SessionState,
    *,
    status: str,
    job_id: str = "j1",
    log_file: Path | None = None,
) -> SimulationJob:
    job = make_sim_job(job_id, status=status, log_file=log_file)
    state.jobs[job_id] = job
    return job


class TestBatchStatusCompleteness:
    """The jobs status reader must format EVERY terminal batch status without
    raising 'unexpected status'. Regression class: the interrupted-status
    formatter bug — a hardcoded status allowlist omitted a real terminal
    status."""

    @pytest.mark.parametrize("status", BATCH_TERMINAL_STATUSES)
    async def test_jobs_status_handles_terminal_batch_status(
        self, status: str, state_no_sim: SessionState
    ):
        _make_batch(state_no_sim, status=status)
        result = await handle_jobs(
            JobsInput.model_validate({"action": "status", "job_id": "b1"}), state_no_sim
        )
        data = result.structuredContent
        assert data is not None
        assert "unexpected status" not in _text(result).lower()
        # the status is surfaced to the caller (not swallowed)
        assert data["status"] == status


class TestCrossTypeResolution:
    """The consolidated jobs reader is type-agnostic: an id from EITHER store
    (single-sim or batch) must resolve through the same status action — never
    "not found" for a job that exists, never a wrong-type rejection. This is
    the dual-store invariant that replaced the old per-type tools' redirect
    contract when those tools merged into one."""

    async def test_jobs_status_resolves_both_stores(self, state_no_sim: SessionState):
        _make_sim(state_no_sim, status="completed", log_file=LTSPICE_TRAN_RC_LOG)
        _make_batch(state_no_sim, status="completed")
        for job_id in ("j1", "b1"):
            result = await handle_jobs(
                JobsInput.model_validate({"action": "status", "job_id": job_id}),
                state_no_sim,
            )
            data = result.structuredContent
            assert data is not None
            assert data["job_id"] == job_id
            assert data["status"] == "completed"


class TestCancelJobRoutingFork:
    """cancel_job forks on job type (BatchJob → batch runner, SimulationJob →
    single-sim runner). The batch fork has runner-routing tests elsewhere; the
    single-sim fork sat behind guards (unknown id, not running, no simulator)
    that every prior unit test stopped at, so it was never exercised. These
    tests cross the guards and pin the routing: the live job object itself must
    be handed to the single-sim runner's cancel."""

    @pytest.mark.parametrize("status", sorted(NON_TERMINAL_LIVE_STATUSES))
    async def test_live_single_sim_routes_to_sim_runner(
        self, status: str, state_with_sim: SessionState
    ):
        job = _make_sim(state_with_sim, status=status)
        fake_runner = MagicMock(cancel=AsyncMock())
        with patch(
            "ltspice_mcp.tools.simulation._get_or_create_runner", return_value=fake_runner
        ) as get_runner:
            result = await handle_cancel_job(CancelJobInput(job_id="j1"), state_with_sim)
        assert "cancelled" in _text(result).lower()
        # Resolved via the job's own netlist so the runner's output folder matches
        # the one the job launched with. simulator_class is None here: the
        # job's recorded name ("ltspice") matches no detected class, so the
        # runner falls back to the session default.
        get_runner.assert_called_once_with(state_with_sim, job.netlist, simulator_class=None)
        fake_runner.cancel.assert_awaited_once()
        # The exact job resolved from state.jobs reaches the runner —
        # not a re-looked-up copy, not a batch-runner detour.
        assert fake_runner.cancel.await_args is not None
        assert fake_runner.cancel.await_args.args[0] is job

    async def test_single_sim_fork_does_not_touch_batch_runners(
        self, state_with_sim: SessionState
    ):
        _make_sim(state_with_sim, status="running")
        fake_runner = MagicMock(cancel=AsyncMock())
        with (
            patch(
                "ltspice_mcp.tools.simulation._get_or_create_runner",
                return_value=fake_runner,
            ),
            patch.object(state_with_sim.runners, "get_batch_runner_for") as batch,
        ):
            await handle_cancel_job(CancelJobInput(job_id="j1"), state_with_sim)
        batch.assert_not_called()
        fake_runner.cancel.assert_awaited_once()


# --- DUAL-DISPATCH (.cir vs .asc) -----------------------------------------

# Both files contain a resistor R1 with value 1k, so the same assertions run
# against both dispatch branches (spice_lex pipeline vs AscEditor).
_CIR_NETLIST = "* RC filter\nR1 in out 1k\nC1 out 0 100n\nV1 in 0 1\n.op\n.end\n"


@pytest.fixture
def circuit_file(request: pytest.FixtureRequest, work_dir: Path) -> Path:
    """Circuit file of the parametrized extension inside the allowed dir.

    ``.asc`` copies the Draft1 fixture (R1=1k) and pulls in the session-scoped
    symbol cache so AscEditor can resolve its symbols; ``.cir`` writes an
    equivalent netlist with the same R1=1k.
    """
    ext = request.param
    if ext == "asc":
        request.getfixturevalue("asc_symbols")
        dest = work_dir / "Draft1.asc"
        shutil.copy(_FIXTURE_DRAFT, dest)
        return dest
    path = work_dir / "rc_filter.cir"
    path.write_text(_CIR_NETLIST)
    return path
