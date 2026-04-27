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

        def fake_run(*args, callback=None, callback_args=None, **kwargs):
            # Stash the post-wrap callback the wrapper sets on the task.
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

        task = wrapped.run("netlist.cir", callback=user_cb)  # type: ignore[arg-type]
        # The wrapper rebinds task.callback; invoking it should now
        # forward runno=task.runno to the user callback.
        assert task is not None
        rebound = task.callback
        assert rebound is not None
        rebound(Path("netlist_7.raw"), Path("netlist_7.log"))
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
        bad_spec = ToleranceSpec(tolerance=0.1, distribution="weibull")
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
