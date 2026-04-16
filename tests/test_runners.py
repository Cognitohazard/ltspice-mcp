"""Direct unit tests for SimulationRunner / SweepRunner / MonteCarloRunner internals.

These tests bypass the spicelib SimRunner machinery and exercise the
event-loop callback handlers (_handle_completion, _handle_run_completion,
_handle_sweep_completion, etc.) and cancel() methods, all of which are
pure logic operating on BatchJob/SimulationJob state.
"""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltspice_mcp.lib import now
from ltspice_mcp.lib.montecarlo_runner import MonteCarloRunner
from ltspice_mcp.lib.sim_runner import SimulationRunner, generate_job_id
from ltspice_mcp.lib.sweep_runner import SweepRunner
from ltspice_mcp.state import BatchJob, MonteCarloConfig, SessionState, SimulationJob, SweepConfig


class FakeSim:
    """Minimal simulator stub."""


@pytest.fixture
def loop():
    return asyncio.new_event_loop()


@pytest.fixture
def sim_runner(loop, work_dir: Path) -> SimulationRunner:
    return SimulationRunner(
        loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=1
    )


@pytest.fixture
def sweep_runner(loop, work_dir: Path) -> SweepRunner:
    return SweepRunner(
        loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=1
    )


@pytest.fixture
def mc_runner(loop, work_dir: Path) -> MonteCarloRunner:
    return MonteCarloRunner(
        loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=1
    )


def _make_job(
    state: SessionState, work_dir: Path, status: str = "running"
) -> SimulationJob:
    job = SimulationJob(
        job_id="sim_test_1",
        netlist=work_dir / "n.cir",
        simulator="FakeSim",
        status=status,  # type: ignore[arg-type]
        started_at=now(),
    )
    state.jobs[job.job_id] = job
    return job


class TestGenerateJobId:
    def test_format(self):
        jid = generate_job_id()
        assert jid.startswith("sim_")
        assert len(jid.split("_")) == 3


class TestSimulationRunnerHandleCompletion:
    def test_completion_success(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        job = _make_job(state_no_sim, work_dir)
        raw = work_dir / "out.raw"
        raw.write_text("non-empty")
        log = work_dir / "out.log"
        log.write_text("ok")
        sim_runner._handle_completion(job.job_id, str(raw), str(log), state_no_sim)
        assert job.status == "completed"
        assert job.done_event.is_set()

    def test_completion_empty_raw_marks_failed(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        job = _make_job(state_no_sim, work_dir)
        raw = work_dir / "empty.raw"
        raw.write_bytes(b"")  # zero size
        log = work_dir / "empty.log"
        log.write_text("Error: convergence failed\n")
        sim_runner._handle_completion(job.job_id, str(raw), str(log), state_no_sim)
        assert job.status == "failed"
        assert job.error is not None
        assert "no output" in job.error
        assert job.done_event.is_set()

    def test_completion_unknown_job(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        # Should silently warn, not raise
        sim_runner._handle_completion("missing", "/x.raw", "/x.log", state_no_sim)

    def test_completion_terminal_state_skipped(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        job = _make_job(state_no_sim, work_dir, status="cancelled")
        raw = work_dir / "out.raw"
        raw.write_text("data")
        log = work_dir / "out.log"
        log.write_text("ok")
        sim_runner._handle_completion(job.job_id, str(raw), str(log), state_no_sim)
        # Status should not change from cancelled
        assert job.status == "cancelled"



class TestSimulationRunnerCancel:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        job = _make_job(state_no_sim, work_dir)
        # Runner not registered for this job
        await sim_runner.cancel(job)


def _make_batch(
    state: SessionState, work_dir: Path, *, job_type: str = "sweep"
) -> BatchJob:
    bj = BatchJob(
        job_id=f"{job_type}_test",
        job_type=job_type,  # type: ignore[arg-type]
        netlist=work_dir / "n.cir",
        total_runs=3,
    )
    if job_type == "sweep":
        bj.sweep_config = SweepConfig(netlist=work_dir / "n.cir", dimensions=[])
    else:
        bj.mc_config = MonteCarloConfig(netlist=work_dir / "n.cir")
    state.batch_jobs[bj.job_id] = bj
    return bj


class TestSweepRunnerHandlers:
    def test_handle_run_completion(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir)
        raw = work_dir / "r0.raw"
        raw.write_text("d")
        log = work_dir / "r0.log"
        log.write_text("l")
        sweep_runner._handle_run_completion(bj.job_id, raw, log, state_no_sim)
        assert bj.completed_runs == 1
        assert 0 in bj.run_results

    def test_handle_run_completion_unknown(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        sweep_runner._handle_run_completion(
            "missing", work_dir / "x.raw", work_dir / "x.log", state_no_sim
        )

    def test_handle_run_completion_terminal_state(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir)
        bj.status = "cancelled"
        sweep_runner._handle_run_completion(
            bj.job_id, work_dir / "x.raw", work_dir / "x.log", state_no_sim
        )
        assert bj.completed_runs == 0

    def test_handle_sweep_completion(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir)
        bj.run_results = {0: {"raw_file": "x", "log_file": "y", "params": {}}}
        stepper = MagicMock()
        stepper.sim_info = {0: {"R1": "1k", "netlist": "n.cir"}}
        sweep_runner._handle_sweep_completion(bj.job_id, stepper, state_no_sim)
        assert bj.status == "completed"
        assert bj.done_event.is_set()
        assert bj.run_results[0]["params"]["R1"] == 1000.0

    def test_handle_sweep_completion_cancelled(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir)
        bj.status = "cancelled"
        stepper = MagicMock()
        stepper.sim_info = {}
        sweep_runner._handle_sweep_completion(bj.job_id, stepper, state_no_sim)
        # Status remains cancelled
        assert bj.status == "cancelled"

    def test_handle_sweep_completion_unknown(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState
    ):
        stepper = MagicMock()
        sweep_runner._handle_sweep_completion("missing", stepper, state_no_sim)

    @pytest.mark.asyncio
    async def test_cancel(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir)
        await sweep_runner.cancel(bj)
        assert bj.status == "cancelled"
        assert bj.done_event.is_set()


class TestMonteCarloRunnerHandlers:
    def test_handle_run_completion(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        raw = work_dir / "r0.raw"
        raw.write_text("d")
        log = work_dir / "r0.log"
        log.write_text("l")
        mc_runner._handle_run_completion(bj.job_id, raw, log, state_no_sim)
        assert bj.completed_runs == 1

    def test_handle_run_completion_unknown(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState, work_dir: Path
    ):
        mc_runner._handle_run_completion(
            "missing", work_dir / "x.raw", work_dir / "x.log", state_no_sim
        )

    def test_handle_mc_completion(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        bj.run_results = {0: {"raw_file": "x", "log_file": "y", "params": {}}}
        mc_runner._handle_mc_completion(bj.job_id, state_no_sim)
        assert bj.status == "completed"
        assert bj.done_event.is_set()

    def test_handle_mc_completion_cancelled(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        bj.status = "cancelled"
        mc_runner._handle_mc_completion(bj.job_id, state_no_sim)
        assert bj.status == "cancelled"

    def test_handle_mc_completion_unknown(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState
    ):
        mc_runner._handle_mc_completion("missing", state_no_sim)

    @pytest.mark.asyncio
    async def test_cancel(
        self, mc_runner: MonteCarloRunner, state_no_sim: SessionState, work_dir: Path
    ):
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        await mc_runner.cancel(bj)
        assert bj.status == "cancelled"
