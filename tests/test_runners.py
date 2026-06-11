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
    return SweepRunner(loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=1)


@pytest.fixture
def mc_runner(loop, work_dir: Path) -> MonteCarloRunner:
    return MonteCarloRunner(
        loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=1
    )


def _make_job(
    state: SessionState,
    work_dir: Path,
    status: str = "running",
    job_id: str = "sim_test_1",
) -> SimulationJob:
    job = SimulationJob(
        job_id=job_id,
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

    def test_completion_dot_placeholder_marks_failed(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        """Bug I: spicelib signals failure by passing ``raw_file="."`` and a
        ``.fail`` log file. Treating ``Path(".")`` as a real raw file
        would let ``stat()`` succeed (directory size is non-zero) and
        leak ``status="completed"`` + ``raw_file="."`` to clients."""
        job = _make_job(state_no_sim, work_dir)
        log = work_dir / "out.fail"
        log.write_text("Error on line 2 : Q1 c b e mystery — undefined model\n")
        sim_runner._handle_completion(job.job_id, ".", str(log), state_no_sim)
        assert job.status == "failed"
        assert job.raw_file is None
        assert job.error is not None and "no output" in job.error


class TestSimulationRunnerCancel:
    @pytest.mark.asyncio
    async def test_cancel_unknown_job(
        self, sim_runner: SimulationRunner, state_no_sim: SessionState, work_dir: Path
    ):
        job = _make_job(state_no_sim, work_dir)
        # Runner not registered for this job
        await sim_runner.cancel(job)


class TestSimulationRunnerKillWsl:
    """Regression: kill()/cancel() must terminate the real (Windows) sim.

    On WSL spicelib's ``kill_all_spice`` can't see the Windows process, so the
    runner additionally taskkills it by job_id. cancel() must also mark the job
    terminal BEFORE killing, so the killed sim's late completion callback can't
    record a partial raw as success.
    """

    @pytest.mark.asyncio
    async def test_kill_invokes_windows_taskkill_and_native(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        runner = SimulationRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=FakeSim,
            output_folder=work_dir,
            max_parallel=2,
        )
        job = _make_job(state_no_sim, work_dir)
        fake_spice = MagicMock()
        runner._runners[job.job_id] = fake_spice
        tokens: list[str] = []
        monkeypatch.setattr(
            "ltspice_mcp.lib.sim_runner.kill_windows_ltspice_by_token",
            lambda tok: tokens.append(tok) or 1,
        )
        await runner.kill(job.job_id)
        assert tokens == [job.job_id]  # Windows kill targeted this specific job
        fake_spice.kill_all_spice.assert_called_once()  # native/Wine path still runs
        assert job.job_id not in runner._runners  # tracked runner dropped

    @pytest.mark.asyncio
    async def test_submit_passes_exe_log_and_job_token(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        # run_simulation must hand spicelib exe_log=True (ngspice stdout capture)
        # and a job_id-named run_filename (the cancel/kill token) at submit time. A
        # silent revert of either line would regress the feature with no other
        # test catching it.
        runner = SimulationRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=FakeSim,
            output_folder=work_dir,
            max_parallel=2,
        )
        captured: list[dict] = []
        monkeypatch.setattr(
            runner,
            "_build_sim_runner",
            lambda: MagicMock(run=MagicMock(side_effect=lambda *a, **k: captured.append(k))),
        )
        job = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_xlog")
        task = asyncio.get_running_loop().create_task(
            runner.start_simulation(job.netlist, job, state_no_sim)
        )
        await asyncio.sleep(0.05)
        assert len(captured) == 1
        assert captured[0].get("exe_log") is True
        assert str(captured[0].get("run_filename", "")).startswith("sim_xlog")
        if not task.done():
            task.cancel()

    @pytest.mark.asyncio
    async def test_cancel_marks_terminal_before_kill(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        runner = SimulationRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=FakeSim,
            output_folder=work_dir,
            max_parallel=2,
        )
        job = _make_job(state_no_sim, work_dir)  # status="running"
        status_at_kill: dict[str, str] = {}
        monkeypatch.setattr(
            "ltspice_mcp.lib.sim_runner.kill_windows_ltspice_by_token",
            lambda tok: status_at_kill.setdefault("status", job.status) or 0,
        )
        await runner.cancel(job, state_no_sim)
        assert status_at_kill["status"] == "cancelled"  # terminal set before the kill ran
        assert job.status == "cancelled"


class TestSimulationRunnerConcurrencyGate:
    """Regression: independent run_simulation jobs honor max_parallel.

    Each job builds its own spicelib SimRunner (one task), so the session-level
    semaphore is the only thing bounding concurrency across jobs.
    """

    @pytest.mark.asyncio
    async def test_caps_concurrent_jobs_and_admits_queued_on_completion(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        loop = asyncio.get_running_loop()
        runner = SimulationRunner(
            loop=loop, simulator_class=FakeSim, output_folder=work_dir, max_parallel=2
        )
        # submit_sim must return quickly WITHOUT firing the completion callback,
        # so each launched job holds its slot until we release it explicitly.
        monkeypatch.setattr(runner, "_build_sim_runner", lambda: MagicMock())

        jobs = [
            _make_job(state_no_sim, work_dir, status="queued", job_id=f"sim_gate_{i}")
            for i in range(3)
        ]

        tasks = [
            loop.create_task(runner.start_simulation(j.netlist, j, state_no_sim)) for j in jobs
        ]
        await asyncio.sleep(0.05)
        # max_parallel=2 -> exactly two launched, the third still queued.
        assert sum(j.status == "running" for j in jobs) == 2, [j.status for j in jobs]
        assert sum(j.status == "queued" for j in jobs) == 1

        # Complete one running job -> frees a slot -> the queued job launches.
        running = next(j for j in jobs if j.status == "running")
        raw = work_dir / "g.raw"
        raw.write_text("data")
        log = work_dir / "g.log"
        log.write_text("ok")
        runner._handle_completion(running.job_id, str(raw), str(log), state_no_sim)
        await asyncio.sleep(0.05)

        assert running.status == "completed"
        assert sum(j.status == "queued" for j in jobs) == 0  # the waiter got admitted
        assert sum(j.status == "running" for j in jobs) == 2

        for t in tasks:
            if not t.done():
                t.cancel()

    @staticmethod
    def _gate_runner(work_dir: Path, launched: list, monkeypatch) -> SimulationRunner:
        """A max_parallel=1 runner whose submit records each launched run_filename
        (so a test can assert a job did / did NOT actually launch)."""
        runner = SimulationRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=FakeSim,
            output_folder=work_dir,
            max_parallel=1,
        )

        def build():
            m = MagicMock()
            m.run.side_effect = lambda *a, **k: launched.append(k.get("run_filename"))
            return m

        monkeypatch.setattr(runner, "_build_sim_runner", build)
        return runner

    @pytest.mark.asyncio
    async def test_timeout_while_queued_does_not_launch_orphan(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        """A job timed out while still QUEUED on the gate must end terminal and
        must NOT launch when a slot later frees."""
        from ltspice_mcp.lib.job_lifecycle import transition

        launched: list = []
        runner = self._gate_runner(work_dir, launched, monkeypatch)
        loop = asyncio.get_running_loop()
        a = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_to_a")
        b = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_to_b")
        ta = loop.create_task(runner.start_simulation(a.netlist, a, state_no_sim))
        tb = loop.create_task(runner.start_simulation(b.netlist, b, state_no_sim))
        await asyncio.sleep(0.05)
        assert a.status == "running" and b.status == "queued"
        launched_before = list(launched)

        # The timeout handler marks the still-queued job terminal (queued->timeout).
        transition(b, "timeout", state=state_no_sim)
        # Free the only slot; b's task wakes and must self-heal, not launch.
        raw = work_dir / "to.raw"
        raw.write_text("data")
        log = work_dir / "to.log"
        log.write_text("ok")
        runner._handle_completion(a.job_id, str(raw), str(log), state_no_sim)
        await asyncio.sleep(0.05)

        assert b.status == "timeout"  # stayed terminal
        assert b.job_id not in runner._runners  # never registered as running
        assert launched == launched_before  # b's sim was never submitted (no orphan)
        for t in (ta, tb):
            if not t.done():
                t.cancel()

    @pytest.mark.asyncio
    async def test_cancel_while_queued_self_heals_and_frees_slot(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        """Review finding: cancelling a queued job must not launch it nor raise an
        illegal transition; the freed slot must admit the next job."""
        launched: list = []
        runner = self._gate_runner(work_dir, launched, monkeypatch)
        loop = asyncio.get_running_loop()
        a = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_cq_a")
        b = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_cq_b")
        ta = loop.create_task(runner.start_simulation(a.netlist, a, state_no_sim))
        tb = loop.create_task(runner.start_simulation(b.netlist, b, state_no_sim))
        await asyncio.sleep(0.05)
        assert a.status == "running" and b.status == "queued"

        await runner.cancel(b, state_no_sim)
        assert b.status == "cancelled"
        launched_before = list(launched)

        raw = work_dir / "cq.raw"
        raw.write_text("data")
        log = work_dir / "cq.log"
        log.write_text("ok")
        runner._handle_completion(a.job_id, str(raw), str(log), state_no_sim)
        await asyncio.sleep(0.05)
        assert b.status == "cancelled"  # woken task did not flip it to running
        assert launched == launched_before  # no orphan launch

        # The slot freed by completing 'a' (and not re-taken by cancelled 'b')
        # admits a fresh job.
        c = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_cq_c")
        tc = loop.create_task(runner.start_simulation(c.netlist, c, state_no_sim))
        await asyncio.sleep(0.05)
        assert c.status == "running"
        for t in (ta, tb, tc):
            if not t.done():
                t.cancel()

    @pytest.mark.asyncio
    async def test_submission_failure_releases_slot(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        """A submit that raises must free the slot so later jobs aren't wedged."""
        runner = SimulationRunner(
            loop=asyncio.get_running_loop(),
            simulator_class=FakeSim,
            output_folder=work_dir,
            max_parallel=1,
        )

        def boom():
            raise RuntimeError("submit boom")

        monkeypatch.setattr(runner, "_build_sim_runner", boom)
        a = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_sf_a")
        await runner.start_simulation(a.netlist, a, state_no_sim)
        assert a.status == "failed"

        # Slot released despite the failure: a working job runs.
        monkeypatch.setattr(runner, "_build_sim_runner", lambda: MagicMock())
        b = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_sf_b")
        tb = asyncio.get_running_loop().create_task(
            runner.start_simulation(b.netlist, b, state_no_sim)
        )
        await asyncio.sleep(0.05)
        assert b.status == "running"
        if not tb.done():
            tb.cancel()

    @pytest.mark.asyncio
    async def test_unverified_kill_keeps_slot_reserved_until_finalized(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch
    ):
        """Codex review (high): a cancel whose process termination cannot be
        verified (WSL taskkill finds nothing / native kill_all_spice raises) must
        NOT free the concurrency slot — otherwise a queued job launches alongside
        a still-running orphan and exceeds max_parallel. The slot stays reserved
        until the process is actually finalized (completion callback fires)."""
        launched: list = []
        runner = self._gate_runner(work_dir, launched, monkeypatch)  # max_parallel=1
        loop = asyncio.get_running_loop()
        # Both best-effort termination paths "fail": WSL taskkill confirms nothing,
        # and kill_all_spice raises (and is swallowed).
        monkeypatch.setattr(
            "ltspice_mcp.lib.sim_runner.kill_windows_ltspice_by_token", lambda tok: 0
        )
        a = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_fk_a")
        b = _make_job(state_no_sim, work_dir, status="queued", job_id="sim_fk_b")
        ta = loop.create_task(runner.start_simulation(a.netlist, a, state_no_sim))
        tb = loop.create_task(runner.start_simulation(b.netlist, b, state_no_sim))
        await asyncio.sleep(0.05)
        assert a.status == "running" and b.status == "queued"
        mock_runner = runner._runners[a.job_id]
        assert isinstance(mock_runner, MagicMock)  # _gate_runner builds MagicMock runners
        mock_runner.kill_all_spice.side_effect = RuntimeError("kill boom")
        launched_before = list(launched)

        await runner.cancel(a, state_no_sim)
        await asyncio.sleep(0.05)
        assert a.status == "cancelled"
        # Unverified kill -> slot stays reserved -> b must NOT have started.
        assert b.status == "queued", (
            "queued job must not start while a possibly-live orphan holds the slot"
        )
        assert launched == launched_before

        # The orphan finally ends -> completion callback finalizes a -> slot freed -> b runs.
        raw = work_dir / "fk.raw"
        raw.write_text("data")
        log = work_dir / "fk.log"
        log.write_text("ok")
        runner._handle_completion(a.job_id, str(raw), str(log), state_no_sim)
        await asyncio.sleep(0.05)
        assert b.status == "running"
        for t in (ta, tb):
            if not t.done():
                t.cancel()


def _make_batch(state: SessionState, work_dir: Path, *, job_type: str = "sweep") -> BatchJob:
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
        # spicelib runnos are 1-based; run_results keys are 0-based runno.
        bj.run_results = {0: {"raw_file": "x", "log_file": "y", "params": {}}}
        stepper = MagicMock()
        stepper.sim_info = {1: {"R1": "1k", "netlist": "n.cir"}}
        sweep_runner._handle_sweep_completion(bj.job_id, stepper, state_no_sim)
        assert bj.status == "completed"
        assert bj.done_event.is_set()
        assert bj.run_results[0]["params"]["R1"] == 1000.0

    def test_parallel_completion_pairs_params_correctly(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        """Regression: under max_parallel>1, runs complete out of runno order.

        Before this fix, run_results was keyed by completion-order index but
        sim_info was zipped via runno-sorted enumerate, so under parallel
        execution params got attached to the WRONG .raw. Here we record three
        completions in reverse runno order and verify each .raw still ends
        up paired with its own runno's params.
        """
        bj = _make_batch(state_no_sim, work_dir)
        bj.total_runs = 3
        # Three runs complete in reverse runno order (3, 2, 1) — what
        # max_parallel>1 would produce when later runs happen to finish first.
        for runno in (3, 2, 1):
            raw = work_dir / f"sweep_{runno}.raw"
            raw.write_text("d")
            log = work_dir / f"sweep_{runno}.log"
            log.write_text("l")
            sweep_runner._handle_run_completion(bj.job_id, raw, log, state_no_sim)
        assert set(bj.run_results.keys()) == {0, 1, 2}
        # Each run_result should reference its own runno's raw file.
        assert bj.run_results[0]["raw_file"].endswith("sweep_1.raw")
        assert bj.run_results[1]["raw_file"].endswith("sweep_2.raw")
        assert bj.run_results[2]["raw_file"].endswith("sweep_3.raw")

        stepper = MagicMock()
        stepper.sim_info = {
            1: {"Rd": "0.5", "netlist": "n.cir"},
            2: {"Rd": "5", "netlist": "n.cir"},
            3: {"Rd": "50", "netlist": "n.cir"},
        }
        sweep_runner._handle_sweep_completion(bj.job_id, stepper, state_no_sim)
        # Each runno's params must pair with the matching raw file.
        assert bj.run_results[0]["params"]["Rd"] == 0.5
        assert bj.run_results[1]["params"]["Rd"] == 5.0
        assert bj.run_results[2]["params"]["Rd"] == 50.0

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

    @pytest.mark.asyncio
    async def test_cancel_taskkills_windows_by_job_token(
        self,
        sweep_runner: SweepRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # Batch process-kill: sub-runs are named "{job_id}_<index>", so cancel()
        # must taskkill the Windows processes by job_id token on WSL (where
        # kill_all_spice is a no-op), in addition to the native kill_all_spice.
        # After a pass that killed something, cancel re-scans until a clean
        # pass confirms nothing matched (see TestBatchCancelSpawnRace).
        bj = _make_batch(state_no_sim, work_dir)
        fake_runner = MagicMock()
        sweep_runner._register_runner(bj.job_id, fake_runner)
        monkeypatch.setattr("ltspice_mcp.lib.runner_base._CANCEL_KILL_RESCAN_DELAY", 0.001)
        tokens: list[str] = []
        kill_returns = iter([1, 0, 0])
        monkeypatch.setattr(
            "ltspice_mcp.lib.runner_base.kill_windows_ltspice_by_token",
            lambda tok: tokens.append(tok) or next(kill_returns),
        )
        await sweep_runner.cancel(bj, state_no_sim)
        # Windows kill targeted this batch's token, then re-scanned to clean.
        assert tokens == [bj.job_id] * 3
        fake_runner.kill_all_spice.assert_called_once()  # native/Wine path still runs
        assert bj.status == "cancelled"

    @pytest.mark.asyncio
    async def test_sweep_passes_job_token_filenamer_and_exe_log(
        self,
        sweep_runner: SweepRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # execute_sweep must hand run_all a filenamer that embeds the job_id (so
        # cancel's WSL taskkill can target the batch) and exe_log=True (so
        # ngspice's stdout-only diagnostics are captured). spicelib's SimStepper
        # / SimRunner are mocked — we assert OUR wiring, not spicelib iteration.
        # SpiceEditor requires a leading "*" title line for encoding detection.
        (work_dir / "n.cir").write_text("* sweep test\nV1 in 0 1\nR1 in 0 1k\n.tran 1m\n.end\n")
        bj = _make_batch(state_no_sim, work_dir)

        captured: dict = {}

        class FakeStepper:
            def add_value_sweep(self, *a, **k):
                pass

            def add_param_sweep(self, *a, **k):
                pass

            def total_number_of_simulations(self):
                return 0

            def run_all(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "ltspice_mcp.lib.sweep_runner._create_stepper",
            lambda editor, runner: FakeStepper(),
        )
        monkeypatch.setattr(sweep_runner, "_build_sim_runner", lambda: MagicMock(_runno=0))
        monkeypatch.setattr(sweep_runner, "_handle_sweep_completion", lambda *a, **k: None)

        await sweep_runner.start_sweep(bj, state_no_sim)

        assert captured.get("exe_log") is True
        namer = captured.get("filenamer")
        assert callable(namer)
        # spicelib calls filenamer(**current_values); here current_values is {}.
        n1 = namer()
        n2 = namer()
        assert isinstance(n1, str) and isinstance(n2, str)
        assert n1.startswith(f"{bj.job_id}_") and n2.startswith(f"{bj.job_id}_")
        assert n1 != n2  # unique per run


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

    @pytest.mark.asyncio
    async def test_mc_runs_named_by_job_token_and_capture_stdout(
        self,
        mc_runner: MonteCarloRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # Process-kill token + ngspice capture for the MC path. MC uses a
        # per-call run_filename STRING (distinct from sweep's run_all filenamer),
        # so it needs its own coverage: each sub-run must embed the job_id
        # (cancel's WSL taskkill token) and pass exe_log=True (ngspice stdout capture).
        (work_dir / "n.cir").write_text("* mc test\nV1 in 0 1\nR1 in 0 1k\n.tran 1m\n.end\n")
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        bj.total_runs = 2
        bj.mc_config = MonteCarloConfig(
            netlist=work_dir / "n.cir",
            type_tolerances={"R": (0.05, "uniform")},
            num_runs=2,
            seed=1,
        )

        runs: list = []

        def build():
            m = MagicMock(_runno=0)
            m.run.side_effect = lambda *a, **k: runs.append(
                (k.get("run_filename"), k.get("exe_log"))
            )
            return m

        monkeypatch.setattr(mc_runner, "_build_sim_runner", build)
        monkeypatch.setattr(mc_runner, "_handle_mc_completion", lambda *a, **k: None)

        await mc_runner.start_montecarlo(bj, state_no_sim)

        assert len(runs) == 2
        names = [n for n, _ in runs]
        assert all(isinstance(n, str) and n.startswith(f"{bj.job_id}_") for n in names)
        assert len(set(names)) == 2  # unique per run
        assert all(exe is True for _, exe in runs)  # ngspice stdout capture on


class TestParseRunno:
    """``_parse_runno`` extracts spicelib's 1-based runno from raw filenames."""

    def test_simple_runno(self):
        from ltspice_mcp.lib.runner_base import _parse_runno

        assert _parse_runno(Path("rlc_sweep_1.raw")) == 1
        assert _parse_runno(Path("rlc_sweep_42.raw")) == 42

    def test_stem_with_internal_underscores(self):
        from ltspice_mcp.lib.runner_base import _parse_runno

        # Trailing _<digits> is what counts; earlier underscores are stem.
        assert _parse_runno(Path("circuit_v2_5.raw")) == 5
        assert _parse_runno(Path("my_test_circuit_99.raw")) == 99

    def test_no_trailing_runno(self):
        from ltspice_mcp.lib.runner_base import _parse_runno

        # One-shot sims (job-id stems) don't follow the spicelib pattern.
        assert _parse_runno(Path("sim_1234abc.raw")) is None
        assert _parse_runno(Path("plain.raw")) is None


class TestWrapRunnerForRunnoCallbacks:
    """The runner wrapper injects task.runno into the user's callback,
    sidestepping spicelib's filename-parsing fallback path entirely."""

    def test_callback_receives_runno_kwarg(self):
        from unittest.mock import MagicMock

        from ltspice_mcp.lib.runner_base import wrap_runner_for_runno_callbacks

        runner = MagicMock()
        runner._runno = 6  # wrap predicts _runno + 1 = 7

        def fake_run(*args, callback=None, callback_args=None, **kwargs):
            task = MagicMock()
            task.runno = 7
            task.callback = callback
            return task

        runner.run = fake_run
        wrapped = wrap_runner_for_runno_callbacks(runner)

        captured = {}

        def user_cb(rf, lf, runno):
            captured["rf"] = rf
            captured["lf"] = lf
            captured["runno"] = runno

        # The wrapper passes runno_bound as the callback to original_run;
        # fake_run stashes it on task.callback so we can invoke it here.
        task = wrapped.run("netlist.cir", callback=user_cb)  # type: ignore[arg-type]
        assert task is not None
        runno_bound_cb = task.callback
        assert runno_bound_cb is not None
        runno_bound_cb(Path("netlist_7.raw"), Path("netlist_7.log"))
        assert captured == {
            "rf": Path("netlist_7.raw"),
            "lf": Path("netlist_7.log"),
            "runno": 7,
        }

    def test_idempotent(self):
        from unittest.mock import MagicMock

        from ltspice_mcp.lib.runner_base import wrap_runner_for_runno_callbacks

        runner = MagicMock()

        def fake_run(*args, callback=None, **kwargs):
            task = MagicMock()
            task.runno = 1
            task.callback = callback
            return task

        runner.run = fake_run
        first = wrap_runner_for_runno_callbacks(runner)
        first_run = first.run
        second = wrap_runner_for_runno_callbacks(runner)
        # Wrapping twice should not double-wrap.
        assert second.run is first_run

    def test_no_callback_passes_through(self):
        from unittest.mock import MagicMock

        from ltspice_mcp.lib.runner_base import wrap_runner_for_runno_callbacks

        runner = MagicMock()
        seen_callbacks = []

        def fake_run(*args, callback=None, **kwargs):
            seen_callbacks.append(callback)
            task = MagicMock()
            task.runno = 1
            return task

        runner.run = fake_run
        wrap_runner_for_runno_callbacks(runner)
        runner.run("netlist.cir")
        # Original is invoked with callback=None when user passes no cb.
        assert seen_callbacks == [None]


class TestMCSampler:
    """Our own MC perturbation engine. Replaces spicelib's Montecarlo class."""

    def test_normal_distribution_is_multiplicative(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=42)
        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        nominal = 1e-3  # 1 mH

        samples = [sampler.sample(nominal, spec) for _ in range(2000)]
        mean = statistics.fmean(samples)
        stdev = statistics.stdev(samples)
        # Mean within 3σ/√n of nominal.
        assert abs(mean - nominal) < 3 * (nominal * 0.05 / 3) / (len(samples) ** 0.5)
        # Stddev within 20% of theoretical σ = value * tol / 3.
        expected_sigma = nominal * 0.05 / 3
        assert 0.8 * expected_sigma < stdev < 1.2 * expected_sigma
        # No nonsense negatives or off-by-orders-of-magnitude values.
        assert all(s > 0 for s in samples)
        assert all(0.7 * nominal < s < 1.3 * nominal for s in samples)

    def test_uniform_distribution_within_tolerance(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, distribution="uniform")
        nominal = 25e-6  # 25 µF
        samples = [sampler.sample(nominal, spec) for _ in range(500)]
        # Every sample within ±10% of nominal.
        assert all(nominal * 0.9 <= s <= nominal * 1.1 for s in samples)

    def test_seed_reproducibility(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        s1 = MCSampler(seed=12345)
        s2 = MCSampler(seed=12345)
        seq1 = [s1.sample(1e-3, spec) for _ in range(20)]
        seq2 = [s2.sample(1e-3, spec) for _ in range(20)]
        assert seq1 == seq2

    def test_different_seeds_diverge(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        spec = ToleranceSpec(tolerance=0.05, distribution="normal")
        s1 = MCSampler(seed=1).sample(1e-3, spec)
        s2 = MCSampler(seed=2).sample(1e-3, spec)
        assert s1 != s2

    def test_unknown_distribution_raises(self):
        import pytest

        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=0)
        # Deliberately bypass the Literal type to exercise the runtime
        # error path — the engine validates the distribution name even
        # though the static type system already constrains it.
        bad_spec = ToleranceSpec(tolerance=0.1, distribution="weibull")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="Unknown distribution"):
            sampler.sample(1.0, bad_spec)


class TestExpandTolerances:
    def test_per_ref_override_wins_over_type(self):
        from ltspice_mcp.lib.montecarlo import expand_tolerances

        out = expand_tolerances(
            ["R1", "R2", "C1"],
            type_tolerances={"R": (0.05, "normal")},
            component_overrides={"R1": (0.01, "uniform")},
        )
        assert out["R1"].tolerance == 0.01
        assert out["R1"].distribution == "uniform"
        # R2 falls back to the type rule.
        assert out["R2"].tolerance == 0.05
        # C1 has no rule, so it's not in the map.
        assert "C1" not in out

    def test_unperturbable_prefixes_skipped(self):
        from ltspice_mcp.lib.montecarlo import expand_tolerances

        # Voltage sources, switches, etc. are excluded even if rules try.
        out = expand_tolerances(
            ["V1", "S1", "R1"],
            type_tolerances={"V": (0.05, "normal"), "S": (0.05, "normal"), "R": (0.05, "normal")},
            component_overrides={},
        )
        assert "R1" in out
        assert "V1" not in out
        assert "S1" not in out


class TestParseValue:
    def test_engineering_suffixes(self):
        import pytest

        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("1k") == pytest.approx(1e3)
        assert parse_value("100u") == pytest.approx(1e-4)
        assert parse_value("2.2n") == pytest.approx(2.2e-9)
        assert parse_value("10Meg") == pytest.approx(10e6)
        assert parse_value("1m") == pytest.approx(1e-3)
        assert parse_value("1") == pytest.approx(1.0)
        assert parse_value("1.5e-6") == pytest.approx(1.5e-6)

    def test_parametric_returns_none(self):
        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("{Rd}") is None
        assert parse_value("R*2") is None  # operator
        assert parse_value("table(...)") is None

    def test_invalid_returns_none(self):
        from ltspice_mcp.lib.montecarlo import parse_value

        assert parse_value("") is None
        assert parse_value("abc") is None


class TestSampleOffset:
    """``sample_offset`` returns additive deltas — the call site composes
    them with the nominal. Relative kind scales by |nominal|; absolute
    kind uses the raw tolerance as σ (or half-range)."""

    def test_relative_zero_nominal_yields_zero_delta(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, kind="relative")
        # σ = |nominal| * tol / 3 = 0 → all samples are 0.
        samples = [sampler.sample_offset(0.0, spec) for _ in range(20)]
        assert all(s == 0.0 for s in samples)

    def test_absolute_kind_uses_raw_tolerance_as_3sigma(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=42)
        # 30 mV ± 3σ → σ = 10 mV. Sample many; check std ≈ 10 mV.
        spec = ToleranceSpec(tolerance=0.030, kind="absolute")
        samples = [sampler.sample_offset(0.7, spec) for _ in range(5000)]
        sigma_estimate = statistics.stdev(samples)
        assert 0.0085 < sigma_estimate < 0.0115  # within ~15% of σ=10mV

    def test_relative_kind_scales_by_nominal(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=7)
        spec = ToleranceSpec(tolerance=0.10, kind="relative")
        # σ = |1k| * 0.10 / 3 ≈ 33.3
        samples = [sampler.sample_offset(1000.0, spec) for _ in range(5000)]
        sigma = statistics.stdev(samples)
        assert 28.0 < sigma < 38.0


class TestModelPerturbationMath:
    def test_sample_model_perturbation_skips_missing_nominals(self):
        from ltspice_mcp.lib.montecarlo import (
            MCSampler,
            ToleranceSpec,
            sample_model_perturbation,
        )

        sampler = MCSampler(seed=0)
        out = sample_model_perturbation(
            sampler,
            "NMOS1",
            nominals={"VTO": 0.7},  # KP is missing
            tolerances={
                "VTO": ToleranceSpec(tolerance=0.05, kind="relative"),
                "KP": ToleranceSpec(tolerance=0.10, kind="relative"),  # not in nominals
            },
        )
        # Only VTO comes back; KP is skipped without raising.
        assert "VTO" in out
        assert "KP" not in out

    def test_sample_model_perturbation_adds_delta(self):
        from ltspice_mcp.lib.montecarlo import (
            MCSampler,
            ToleranceSpec,
            sample_model_perturbation,
        )

        sampler = MCSampler(seed=42)
        # Absolute σ_VTH = 12 mV → ±36 mV at 3σ tolerance
        out = sample_model_perturbation(
            sampler,
            "NMOS1",
            nominals={"VTO": 0.7, "KP": 100e-6},
            tolerances={
                "VTO": ToleranceSpec(tolerance=0.036, kind="absolute"),
                "KP": ToleranceSpec(tolerance=0.10, kind="relative"),
            },
        )
        # VTO perturbation should be in vicinity of 0.7 ± several σ
        assert 0.6 < out["VTO"] < 0.8
        assert 80e-6 < out["KP"] < 120e-6


class TestPerturbModelInText:
    def test_replaces_existing_param(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = (
            "* test\n"
            ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u LAMBDA=0.02)\n"
            "M1 d g 0 0 NMOS1 W=10u L=1u\n"
            ".END\n"
        )
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.715, "KP": 0.000105})
        # Old values must be gone
        assert "VTO=0.7\b" not in out
        assert "KP=100u" not in out
        # New values present
        assert "VTO=0.715" in out
        assert "KP=0.000105" in out

    def test_appends_missing_param_inside_paren(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n"
        out = perturb_model_in_text(text, "NMOS1", {"LAMBDA": 0.025})
        assert "LAMBDA=0.025" in out
        # Closing paren still present and balanced.
        assert out.count("(") == out.count(")")

    def test_case_insensitive_match(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".model nmos1 nmos(vto=0.7)\n"
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.65})
        assert "0.65" in out

    def test_continuation_lines_merged(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        text = ".MODEL NMOS1 NMOS(VTO=0.7\n+ KP=100u LAMBDA=0.02)\n.END\n"
        out = perturb_model_in_text(text, "NMOS1", {"VTO": 0.715})
        assert "0.715" in out

    def test_missing_model_raises(self):
        from ltspice_mcp.lib.montecarlo import perturb_model_in_text

        with pytest.raises(ValueError, match="not found"):
            perturb_model_in_text(".MODEL OTHER NPN(BF=200)\n", "NMOS1", {"VTO": 0.7})


class TestPelgromMismatch:
    def test_smaller_devices_have_larger_sigma(self):
        import statistics

        from ltspice_mcp.lib.montecarlo import (
            InstanceGeometry,
            MCSampler,
            MismatchRule,
            sample_instance_mismatch,
        )

        rule = MismatchRule(prefix="M", avt=3e-3, ak=0.0)
        # Big device: W=L=10 µm → W·L=100 µm² → σ_VTH = 3mV/√100 = 300 µV
        big = InstanceGeometry("M1", "NMOS1", width_m=10e-6, length_m=10e-6)
        # Small device: W=L=0.5 µm → W·L=0.25 µm² → σ_VTH = 6 mV
        small = InstanceGeometry("M2", "NMOS1", width_m=0.5e-6, length_m=0.5e-6)

        sampler_big = MCSampler(seed=1)
        sampler_small = MCSampler(seed=2)
        big_samples = [
            sample_instance_mismatch(sampler_big, big, rule)["dvth"] for _ in range(2000)
        ]
        small_samples = [
            sample_instance_mismatch(sampler_small, small, rule)["dvth"] for _ in range(2000)
        ]
        sigma_big = statistics.stdev(big_samples)
        sigma_small = statistics.stdev(small_samples)
        # Theoretical ratio: σ_small/σ_big = √(WL_big / WL_small) = √(100/0.25) = 20
        ratio = sigma_small / sigma_big
        assert 15 < ratio < 25  # within ~25% of the analytical 20

    def test_disabled_when_coefficients_zero(self):
        from ltspice_mcp.lib.montecarlo import (
            InstanceGeometry,
            MCSampler,
            MismatchRule,
            sample_instance_mismatch,
        )

        rule = MismatchRule(prefix="M", avt=0.0, ak=0.0)
        inst = InstanceGeometry("M1", "NMOS1", width_m=1e-6, length_m=1e-6)
        sampler = MCSampler(seed=0)
        out = sample_instance_mismatch(sampler, inst, rule)
        assert out["dvth"] == 0.0
        assert out["dk_over_k"] == 0.0


class TestVariantModelGeneration:
    def test_render_variant_renames_and_overrides(self):
        from ltspice_mcp.lib.montecarlo import render_variant_model_card

        base = ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u LAMBDA=0.02)\n"
        variant = render_variant_model_card(base, "NMOS1__M1", {"VTO": 0.714, "KP": 0.000098})
        assert ".MODEL NMOS1__M1" in variant
        # Make sure the original NMOS1 token isn't left behind in the card
        assert ".MODEL NMOS1 " not in variant
        assert "VTO=0.714" in variant
        assert "KP=9.8e-05" in variant or "KP=0.0000980" in variant or "KP=9.8e-5" in variant

    def test_inject_card_before_end(self):
        from ltspice_mcp.lib.montecarlo import inject_card_before_end

        text = ".MODEL NMOS1 NMOS(VTO=0.7)\nM1 d g 0 0 NMOS1\n.END\n"
        out = inject_card_before_end(text, ".MODEL NMOS1__M1 NMOS(VTO=0.715)\n")
        assert ".MODEL NMOS1__M1" in out
        # Variant card must appear before .END
        end_idx = out.lower().rindex(".end")
        variant_idx = out.index("NMOS1__M1")
        assert variant_idx < end_idx

    def test_rewrite_instance_model_preserves_params(self):
        from ltspice_mcp.lib.montecarlo import rewrite_instance_model

        text = "M1 d g 0 0 NMOS1 W=10u L=1u m=2\n"
        out = rewrite_instance_model(text, "M1", "NMOS1__M1")
        assert "NMOS1__M1" in out
        # W= and L= preserved; the original model token is replaced not
        # duplicated.
        assert "W=10u" in out
        assert "L=1u" in out
        assert " NMOS1 " not in out

    def test_rewrite_instance_model_no_params(self):
        from ltspice_mcp.lib.montecarlo import rewrite_instance_model

        text = "Q1 c b e MYNPN\n"
        out = rewrite_instance_model(text, "Q1", "MYNPN__Q1")
        assert "Q1 c b e MYNPN__Q1" in out


class TestExtractMosfetInstances:
    def test_finds_W_L_geometry(self):
        from ltspice_mcp.lib.montecarlo import extract_mosfet_instances

        text = (
            "* test\n"
            ".MODEL NMOS1 NMOS(VTO=0.7)\n"
            "M1 d g 0 0 NMOS1 W=10u L=180n\n"
            "M2 d g 0 0 NMOS1 W=2u L=180n\n"
            ".END\n"
        )
        instances = extract_mosfet_instances(text)
        refs = {i.ref: i for i in instances}
        assert "M1" in refs and "M2" in refs
        assert refs["M1"].width_m == pytest.approx(10e-6)
        assert refs["M1"].length_m == pytest.approx(180e-9)
        assert refs["M2"].width_m == pytest.approx(2e-6)
        assert refs["M1"].model_name == "NMOS1"

    def test_skips_instances_without_geometry(self):
        from ltspice_mcp.lib.montecarlo import extract_mosfet_instances

        # No W= / L= → can't compute Pelgrom σ; skipped.
        text = "M1 d g 0 0 NMOS1\n"
        instances = extract_mosfet_instances(text)
        assert instances == []


class TestParamPerturbation:
    def test_perturb_param_replaces_value(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        text = "* test\n.PARAM vto_n=0.7\n.PARAM kp_n=100u\n.END\n"
        out = perturb_param_in_text(text, "vto_n", 0.715)
        assert ".PARAM vto_n=0.715" in out
        assert ".PARAM kp_n=100u" in out  # untouched

    def test_perturb_param_case_insensitive(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        text = ".param vto_n = 0.7\n"
        out = perturb_param_in_text(text, "VTO_N", 0.715)
        assert "0.715" in out

    def test_perturb_param_missing_raises(self):
        from ltspice_mcp.lib.montecarlo import perturb_param_in_text

        with pytest.raises(ValueError, match="not found"):
            perturb_param_in_text(".PARAM rd=1k\n", "vto_n", 0.7)

    def test_parse_param_nominal(self):
        from ltspice_mcp.lib.montecarlo import parse_param_nominal

        text = ".PARAM vto_n=0.7\n.PARAM kp_n=100u\n"
        assert parse_param_nominal(text, "vto_n") == pytest.approx(0.7)
        assert parse_param_nominal(text, "kp_n") == pytest.approx(100e-6)
        assert parse_param_nominal(text, "missing") is None


class TestMismatchRuleMatching:
    def test_finds_first_matching_prefix(self):
        from ltspice_mcp.lib.montecarlo import MismatchRule, find_mismatch_rule

        rules = [
            MismatchRule(prefix="M", avt=3e-3, ak=0.02),
            MismatchRule(prefix="Q", avt=2e-3),
        ]
        m_rule = find_mismatch_rule("M1", rules)
        assert m_rule is not None
        assert m_rule.prefix == "M"

        q_rule = find_mismatch_rule("Q5", rules)
        assert q_rule is not None
        assert q_rule.prefix == "Q"

        assert find_mismatch_rule("R7", rules) is None


class TestStreamIsolation:
    """Per-stream RNGs in MCSampler — adding/removing a perturbation
    source mustn't shift other sources' samples. This is the property
    that makes regression-fixed-seed tests stable as the engine evolves."""

    def test_stream_keys_independent(self):
        from ltspice_mcp.lib.montecarlo import MCSampler

        sampler = MCSampler(seed=42)
        a1 = sampler.stream("A").gauss(0.0, 1.0)
        b1 = sampler.stream("B").gauss(0.0, 1.0)

        # Re-create with same seed, draw from B first then A — order doesn't
        # matter because each stream is a self-contained RNG keyed by name.
        sampler2 = MCSampler(seed=42)
        b2 = sampler2.stream("B").gauss(0.0, 1.0)
        a2 = sampler2.stream("A").gauss(0.0, 1.0)

        assert a1 == a2
        assert b1 == b2

    def test_default_stream_compat(self):
        """The default stream still works for legacy single-stream callers."""
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        s1 = MCSampler(seed=7)
        s2 = MCSampler(seed=7)
        spec = ToleranceSpec(tolerance=0.05)
        assert s1.sample(100.0, spec) == s2.sample(100.0, spec)

    def test_derive_yields_independent_child(self):
        """``derive(namespace)`` produces a child sampler whose streams
        are independent of the parent's, but reproducible from the parent
        seed + namespace."""
        from ltspice_mcp.lib.montecarlo import MCSampler

        parent = MCSampler(seed=99)
        child_a = parent.derive("run1")
        child_b = parent.derive("run1")  # same namespace → same samples
        assert child_a.stream("rcl:R1").gauss(0, 1) == child_b.stream("rcl:R1").gauss(0, 1)

        child_c = parent.derive("run2")
        # Different namespace → different stream output (>99% probability;
        # we just check inequality on a single draw, sufficient given seed).
        assert child_a.stream("rcl:R1").gauss(0, 1) != child_c.stream("rcl:R1").gauss(0, 1)

    def test_adding_stream_doesnt_shift_existing(self):
        """If a future engine version adds a new perturbation source, the
        existing sources' sample sequences must be unchanged."""
        from ltspice_mcp.lib.montecarlo import MCSampler

        # Old engine: only one stream "rcl:R1"
        old = MCSampler(seed=123)
        old_samples = [old.stream("rcl:R1").gauss(0, 1) for _ in range(5)]

        # New engine: adds a "model:NMOS1.VTO" stream. Sampling from the
        # new stream first must not shift "rcl:R1"'s subsequent draws.
        new = MCSampler(seed=123)
        _ = [new.stream("model:NMOS1.VTO").gauss(0, 1) for _ in range(3)]
        new_samples = [new.stream("rcl:R1").gauss(0, 1) for _ in range(5)]

        assert old_samples == new_samples


class TestTruncatedGaussian:
    """The ±tolerance bound is the user-promised ±3σ truncation. Without
    truncation, rare-but-real outliers produce nonsensical perturbed
    values (e.g. negative VTO) that don't reflect real silicon."""

    def test_normal_samples_stay_within_bound(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=1)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal")
        # 10000 samples — at least one would fall outside ±3σ in the
        # untruncated distribution (~27 expected). With truncation, all
        # must satisfy |delta/value - 1| <= 0.10.
        for _ in range(10000):
            perturbed = sampler.sample(1.0, spec)
            assert abs(perturbed - 1.0) <= 0.10 + 1e-12  # within ±10% bound

    def test_offset_samples_stay_within_bound_absolute(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=2)
        spec = ToleranceSpec(tolerance=0.030, distribution="normal", kind="absolute")
        for _ in range(10000):
            delta = sampler.sample_offset(0.7, spec)
            assert abs(delta) <= 0.030 + 1e-12

    def test_offset_samples_stay_within_bound_relative(self):
        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=3)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal", kind="relative")
        for _ in range(10000):
            delta = sampler.sample_offset(1000.0, spec)
            assert abs(delta) <= 100.0 + 1e-9  # |nominal| * tolerance

    def test_truncation_preserves_distribution_shape(self):
        """Truncation at ±3σ should leave the central distribution
        approximately Gaussian — std should still be close to the
        nominal σ (a tiny shrinkage from rejection at the tails)."""
        import statistics

        from ltspice_mcp.lib.montecarlo import MCSampler, ToleranceSpec

        sampler = MCSampler(seed=4)
        spec = ToleranceSpec(tolerance=0.10, distribution="normal")
        samples = [sampler.sample(100.0, spec) - 100.0 for _ in range(20000)]
        sigma_est = statistics.stdev(samples)
        # σ = nominal * tol / 3 = 100 * 0.10 / 3 = 3.333
        # Truncation at ±3σ shrinks σ by ~2-3% (analytical) — well within
        # the ±10% bound below.
        assert 3.0 < sigma_est < 3.5


class TestMCRunnerCardFlowIntegration:
    """Integration coverage for the per-run card-mutation hot path.

    ``execute_montecarlo`` is an async closure inside ``MonteCarloRunner``
    that's hard to unit-test directly. These tests exercise the same
    composition (lex → build lookup dicts → Phase 1/2/3 mutations →
    emit) the runner uses, so a regression in any of:

    - lookup-dict construction
    - per-key shifts after sequential setters
    - variant-card injection updating the model dict
    - emit pushing back to the right shape

    is caught here rather than only at simulation time.
    """

    def _baseline_netlist(self) -> str:
        return (
            "* MC integration test\n"
            ".PARAM Vdd=5\n"
            ".MODEL NMOS1 NMOS(VTO=0.7 KP=100u)\n"
            "M1 out gate 0 0 NMOS1 W=10u L=1u\n"
            "M2 out gate 0 0 NMOS1 W=20u L=1u\n"
            "R1 vdd out 1k\n"
            ".TRAN 1m\n"
            ".END\n"
        )

    def test_phase1_model_perturbation_mutates_card_in_place(self):
        from ltspice_mcp.lib.spice_lex import SpiceCard, lex
        from ltspice_mcp.lib.spice_lex_views import ModelCard

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        # Phase 1: perturb VTO and KP on NMOS1.
        view = ModelCard.from_card(model_by_name["nmos1"])
        view.set_param("VTO", 0.715)
        view.set_param("KP", 95e-6)
        # The cached model card now reflects both edits — the second
        # set_param relied on _shift_cached_param_tokens to keep KP's
        # body_offset aligned after the VTO length change.
        from ltspice_mcp.lib.spice_lex import emit

        text = emit(cards)
        assert "VTO=0.715" in text
        assert "KP=9.5e-05" in text
        # The corruption signature (KP glued to VTO's value) must be absent.
        assert "VTO=0.715KP" not in text

    def test_phase2_variant_injection_updates_lookup(self):
        from ltspice_mcp.lib.montecarlo import (
            render_variant_model_card,
            variant_model_name,
        )
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit, lex
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end
        from ltspice_mcp.lib.spice_lex_views import InstanceLine

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        instance_by_ref: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "instance" and c.name
        }
        base = model_by_name["nmos1"]
        variant = variant_model_name("NMOS1", "M1")
        variant_text = render_variant_model_card("".join(base.raw_lines), variant, {"VTO": 0.715})
        new_card = inject_card_before_end(cards, variant_text)
        # The runner registers the new model in the lookup dict so a
        # subsequent Phase-2 instance referencing it could resolve.
        if new_card.name:
            model_by_name[new_card.name.lower()] = new_card
        assert variant.lower() in model_by_name
        # Rewrite M1's model token through the cached instance card.
        InstanceLine.from_card(instance_by_ref["m1"]).set_model(variant)

        out = emit(cards)
        assert variant in out
        # Variant card must land before the .END.
        assert out.index(variant) < out.lower().rindex(".end")
        # M1 line uses the variant; M2 still references the base model.
        for line in out.splitlines():
            if line.startswith("M1 "):
                assert variant in line
            elif line.startswith("M2 "):
                assert "NMOS1" in line and variant not in line

    def test_phase3_param_perturbation_mutates_param_card(self):
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit, lex
        from ltspice_mcp.lib.spice_lex_views import ParamCard

        cards = lex(self._baseline_netlist()).cards
        param_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "param" and c.name
        }
        ParamCard.from_card(param_by_name["vdd"]).set_value(3.3)
        out = emit(cards)
        assert "Vdd=3.3" in out
        assert "Vdd=5" not in out

    def test_full_run_compose_phases_in_order(self):
        # Replicates execute_montecarlo's per-run flow: build lookup
        # dicts once after lex, apply all three phases, emit. Verifies
        # the composition produces a self-consistent netlist with all
        # mutations present.
        from ltspice_mcp.lib.montecarlo import (
            render_variant_model_card,
            variant_model_name,
        )
        from ltspice_mcp.lib.spice_lex import SpiceCard, emit, lex
        from ltspice_mcp.lib.spice_lex_ops import inject_card_before_end
        from ltspice_mcp.lib.spice_lex_views import (
            InstanceLine,
            ModelCard,
            ParamCard,
        )

        cards = lex(self._baseline_netlist()).cards
        model_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "model" and c.name
        }
        instance_by_ref: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "instance" and c.name
        }
        param_by_name: dict[str, SpiceCard] = {
            c.name.lower(): c for c in cards if c.kind == "param" and c.name
        }

        # Phase 1
        ModelCard.from_card(model_by_name["nmos1"]).set_param("VTO", 0.71)

        # Phase 2 — variant for M1 only
        base = model_by_name["nmos1"]
        variant = variant_model_name("NMOS1", "M1")
        variant_text = render_variant_model_card("".join(base.raw_lines), variant, {"VTO": 0.72})
        new_card = inject_card_before_end(cards, variant_text)
        if new_card.name:
            model_by_name[new_card.name.lower()] = new_card
        InstanceLine.from_card(instance_by_ref["m1"]).set_model(variant)

        # Phase 3
        ParamCard.from_card(param_by_name["vdd"]).set_value(3.3)

        out = emit(cards)
        # All three phases visible.
        assert "VTO=0.71" in out  # Phase 1
        assert variant in out  # Phase 2 variant card
        assert "Vdd=3.3" in out  # Phase 3
        # Re-parse to confirm the result is structurally valid.
        re_cards = lex(out).cards
        models = [c.name for c in re_cards if c.kind == "model"]
        assert "NMOS1" in models
        assert variant in models


# Relocated from tests/test_v6_fixes.py (regression).
class TestAN1HierarchicalMcDoesNotJoinSpiceCircuits:
    """A-N1: the MC runner used to do ``"".join(editor.netlist)`` which
    crashed on hierarchical netlists where ``editor.netlist`` contains
    ``SpiceCircuit`` objects for ``.subckt`` blocks. The fix reads the
    netlist via ``read_spice_text`` and lexes once instead.
    """

    def test_hierarchical_netlist_lexes_via_read_spice_text(self, tmp_path: Path) -> None:
        from ltspice_mcp.lib.encoding import read_spice_text
        from ltspice_mcp.lib.spice_lex import lex
        from ltspice_mcp.lib.spice_lex_views import InstanceLine, ModelCard

        cir = tmp_path / "hier.cir"
        cir.write_text(
            "* hierarchical\n"
            ".subckt stage in out vss\n"
            "M1 out in vss vss NM W=10u L=0.5u\n"
            ".model NM NMOS(VTO=0.4 KP=200u)\n"
            ".ends stage\n"
            "X1 in1 out1 0 stage\n"
            "V1 in1 0 1\n"
            ".tran 1u\n"
            ".end\n"
        )

        baseline_text = read_spice_text(cir)
        cards = lex(baseline_text).cards

        # The model inside the subckt is reachable.
        model_cards = [c for c in cards if c.kind == "model"]
        assert any(c.name == "NM" for c in model_cards)
        nm = next(c for c in model_cards if c.name == "NM")
        view = ModelCard.from_card(nm)
        view.set_param("VTO", 0.5)
        assert view.params["VTO"] == "0.5"

        # The X-instance is also reachable as an instance card with model "stage".
        x_cards = [c for c in cards if c.kind == "instance" and c.name == "X1"]
        assert len(x_cards) == 1
        x_view = InstanceLine.from_card(x_cards[0])
        assert x_view.model == "stage"


class TestBatchCancelSpawnRace:
    """Cancelling a batch races spicelib's submission loop: killing the
    in-flight runs frees simulator slots, which resumes a submission blocked
    inside ``runner.run`` — observed live as a Monte-Carlo child created the
    same second as the cancel that survived it and kept simulating. Two
    defenses, both tested here: cancel() re-scans until a clean kill pass,
    and the gated runner refuses submissions once the cancel event is set.
    """

    @pytest.mark.asyncio
    async def test_cancel_rescans_until_late_spawn_killed(
        self,
        sweep_runner: SweepRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        bj = _make_batch(state_no_sim, work_dir)
        monkeypatch.setattr("ltspice_mcp.lib.runner_base._CANCEL_KILL_RESCAN_DELAY", 0.001)
        calls: list[str] = []
        # A late spawn becomes visible only on the third scan; the loop must
        # keep scanning past the first clean pass to catch it.
        kill_returns = iter([2, 0, 1, 0])
        monkeypatch.setattr(
            "ltspice_mcp.lib.runner_base.kill_windows_ltspice_by_token",
            lambda tok: calls.append(tok) or next(kill_returns),
        )
        await sweep_runner.cancel(bj, state_no_sim)
        assert calls == [bj.job_id] * 4
        assert bj.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_single_pass_when_nothing_matched(
        self,
        mc_runner: MonteCarloRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # No process matched (non-WSL no-op, or the batch already drained):
        # one pass, no re-scan delay.
        bj = _make_batch(state_no_sim, work_dir, job_type="montecarlo")
        calls: list[str] = []
        monkeypatch.setattr(
            "ltspice_mcp.lib.runner_base.kill_windows_ltspice_by_token",
            lambda tok: calls.append(tok) or 0,
        )
        await mc_runner.cancel(bj, state_no_sim)
        assert calls == [bj.job_id]
        assert bj.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_kill_passes_are_bounded(
        self,
        sweep_runner: SweepRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # A process that taskkill never manages to remove must not loop
        # cancel() forever.
        from ltspice_mcp.lib import runner_base

        bj = _make_batch(state_no_sim, work_dir)
        monkeypatch.setattr("ltspice_mcp.lib.runner_base._CANCEL_KILL_RESCAN_DELAY", 0.001)
        calls: list[str] = []
        monkeypatch.setattr(
            "ltspice_mcp.lib.runner_base.kill_windows_ltspice_by_token",
            lambda tok: calls.append(tok) or 1,
        )
        await sweep_runner.cancel(bj, state_no_sim)
        assert len(calls) == runner_base._CANCEL_KILL_MAX_PASSES
        assert bj.status == "cancelled"


class TestGateRunnerOnCancel:
    def test_gate_blocks_submissions_once_cancelled(self):
        import threading

        from ltspice_mcp.lib.runner_base import BatchCancelledError, gate_runner_on_cancel

        ev = threading.Event()
        inner = MagicMock()
        launched: list[tuple] = []
        inner.run = lambda *a, **k: launched.append(a)
        gated = gate_runner_on_cancel(inner, ev, "sweep_x")

        gated.run("first")
        assert launched == [("first",)]

        ev.set()
        with pytest.raises(BatchCancelledError):
            gated.run("second")
        assert launched == [("first",)]  # nothing launched after cancel

    def test_mark_batch_failed_ignores_cancel_abort(
        self, sweep_runner: SweepRunner, state_no_sim: SessionState, work_dir: Path
    ):
        # The gate can fire while cancel() is still mid-kill (job not yet
        # transitioned). The abort exception must not mark the job failed —
        # cancel() owns the terminal transition.
        from ltspice_mcp.lib.runner_base import BatchCancelledError

        bj = _make_batch(state_no_sim, work_dir)
        bj.status = "running"
        sweep_runner._mark_batch_failed(
            bj, state_no_sim, BatchCancelledError("cancelled"), kind="sweep"
        )
        assert bj.status == "running"  # untouched, not "failed"
        assert bj.error is None

    @pytest.mark.asyncio
    async def test_cancelled_sweep_stops_launching_queued_runs(
        self,
        sweep_runner: SweepRunner,
        state_no_sim: SessionState,
        work_dir: Path,
        monkeypatch,
    ):
        # End-to-end through start_sweep: once the cancel event fires
        # mid-batch, the submission loop must stop launching the remaining
        # queued runs instead of working through the rest of the queue.
        (work_dir / "n.cir").write_text("* t\nV1 in 0 1\nR1 in 0 1k\n.tran 1m\n.end\n")
        bj = _make_batch(state_no_sim, work_dir)
        bj.status = "running"
        inner = MagicMock(_runno=0)
        launched: list[int] = []

        class FakeStepper:
            def __init__(self, runner):
                self.runner = runner

            def add_value_sweep(self, *a, **k):
                pass

            def add_param_sweep(self, *a, **k):
                pass

            def total_number_of_simulations(self):
                return 5

            def run_all(self, **kwargs):
                for i in range(5):
                    self.runner.run(f"run{i}")
                    launched.append(i)
                    if i == 1:
                        # Cancel lands while the batch is mid-queue.
                        sweep_runner._cancel_events[bj.job_id].set()

        monkeypatch.setattr(
            "ltspice_mcp.lib.sweep_runner._create_stepper",
            lambda editor, runner: FakeStepper(runner),
        )
        monkeypatch.setattr(sweep_runner, "_build_sim_runner", lambda: inner)
        await sweep_runner.start_sweep(bj, state_no_sim)

        assert launched == [0, 1]  # run 2..4 never submitted
        # The abort is not a batch failure: cancel() owns the status.
        assert bj.status == "running"
        assert bj.error is None
