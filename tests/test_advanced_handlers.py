"""Tests for advanced sweep/Monte Carlo handlers (configure + get_batch_results)."""

import asyncio
from datetime import timedelta
from pathlib import Path

import pytest

from ltspice_mcp.errors import BatchJobError, SimulationError
from ltspice_mcp.state import BatchJob, SessionState
from ltspice_mcp.tools import advanced
from ltspice_mcp.tools.advanced import (
    ConfigureMonteCarloInput,
    ConfigureSweepInput,
    GetBatchResultsInput,
    MonteCarloTolerance,
    RunBatchInput,
    SweepParameter,
    handle_batch_results,
    handle_configure_montecarlo,
    handle_configure_sweep,
    handle_run_montecarlo,
    handle_run_sweep,
)


def _make_batch(state: SessionState, **overrides) -> BatchJob:
    bj = BatchJob(
        job_id=overrides.get("job_id", "b1"),
        job_type=overrides.get("job_type", "sweep"),
        netlist=overrides.get("netlist", Path("/tmp/x.cir")),
        total_runs=overrides.get("total_runs", 1),
        completed_runs=overrides.get("completed_runs", 0),
        failed_runs=overrides.get("failed_runs", 0),
        status=overrides.get("status", "running"),
    )
    if "run_results" in overrides:
        bj.run_results = overrides["run_results"]
    if bj.status != "running":
        bj.completed_at = bj.started_at + timedelta(seconds=2)
    state.batch_jobs[bj.job_id] = bj
    return bj


@pytest.mark.asyncio
class TestConfigureSweep:
    async def test_basic_step(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1000, stop=10000, step=1000)
                ],
            ),
            state_no_sim,
        )
        text = result.content[0].text
        assert "Config ID:" in text
        assert "Total simulations:" in text
        assert len(state_no_sim.sweep_configs) == 1

    async def test_rejects_oversized_cross_product(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        # A cross-product over the cap must be refused up front, not spawned as
        # tens of thousands of cold simulator processes. 101 x 101 = 10201 > cap.
        with pytest.raises(BatchJobError, match="over the 10000 cap"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(name="R1", type="component", start=0, stop=100, step=1),
                        SweepParameter(name="R2", type="component", start=0, stop=100, step=1),
                    ],
                ),
                state_no_sim,
            )
        assert len(state_no_sim.sweep_configs) == 0

    async def test_oversized_points_rejected_without_materializing(
        self, state_no_sim: SessionState, sample_netlist: Path, monkeypatch
    ):
        # points=1e9 must be refused by the cap, NOT materialized — np.linspace
        # would allocate ~8 GB and OOM the server. Sentinel: blow up if the
        # allocator is reached, proving the cap fires on count() first.
        import ltspice_mcp.lib.sweep_utils as su

        def boom(*a, **k):
            raise AssertionError("materialized the range before the cap check")

        monkeypatch.setattr(su, "generate_sweep_range", boom)
        with pytest.raises(BatchJobError, match="over the 10000 cap"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(name="R1", type="component", start=0, stop=1, points=10**9)
                    ],
                ),
                state_no_sim,
            )
        assert len(state_no_sim.sweep_configs) == 0

    async def test_values_list(self, state_no_sim: SessionState, sample_netlist: Path):
        # F5: an explicit discrete value list (e.g. E-series) — previously
        # impossible through configure_sweep, which only generated linear/log
        # grids even though the backend's add_value_sweep accepts a list.
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", values=[1000, 2200, 4700])
                ],
            ),
            state_no_sim,
        )
        assert len(state_no_sim.sweep_configs) == 1
        dim = next(iter(state_no_sim.sweep_configs.values())).dimensions[0]
        assert dim.resolved_values() == [1000.0, 2200.0, 4700.0]

    async def test_values_spice_notation_matches_plain_numbers(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        # SPICE-notation strings ('1k', '4.7k', '10k') must resolve to the same
        # sweep values as the equivalent plain numbers — consistency with the
        # rest of the surface (set_component_value, filters), which parse them.
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", values=["1k", "4.7k", "10k"])
                ],
            ),
            state_no_sim,
        )
        spice_dim = next(iter(state_no_sim.sweep_configs.values())).dimensions[0]

        state_no_sim.sweep_configs.clear()
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", values=[1000, 4700, 10000])
                ],
            ),
            state_no_sim,
        )
        plain_dim = next(iter(state_no_sim.sweep_configs.values())).dimensions[0]

        assert spice_dim.resolved_values() == plain_dim.resolved_values()
        assert spice_dim.resolved_values() == [1000.0, 4700.0, 10000.0]

    async def test_range_spice_notation_matches_plain_numbers(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        # start/stop/step accept SPICE notation too.
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start="1k", stop="3k", step="1k")
                ],
            ),
            state_no_sim,
        )
        spice_dim = next(iter(state_no_sim.sweep_configs.values())).dimensions[0]

        state_no_sim.sweep_configs.clear()
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1000, stop=3000, step=1000)
                ],
            ),
            state_no_sim,
        )
        plain_dim = next(iter(state_no_sim.sweep_configs.values())).dimensions[0]

        assert spice_dim.resolved_values() == plain_dim.resolved_values()

    async def test_unparseable_spice_value_rejected(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        with pytest.raises(BatchJobError, match="not a number or valid SPICE notation"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(name="R1", type="component", values=["notanumber"])
                    ],
                ),
                state_no_sim,
            )

    async def test_values_mutually_exclusive_with_range(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        with pytest.raises(BatchJobError, match="mutually exclusive"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(
                            name="R1", type="component", values=[1, 2], start=1, stop=2, step=1
                        )
                    ],
                ),
                state_no_sim,
            )

    async def test_values_empty_rejected(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="non-empty"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[SweepParameter(name="R1", type="component", values=[])],
                ),
                state_no_sim,
            )

    async def test_range_form_still_requires_start_stop(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        # With neither values nor start/stop, the range form must error clearly.
        with pytest.raises(BatchJobError, match="start and stop are required"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[SweepParameter(name="R1", type="component", step=1)],
                ),
                state_no_sim,
            )

    async def test_basic_points(self, state_no_sim: SessionState, sample_netlist: Path):
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1, stop=10, points=10)
                ],
            ),
            state_no_sim,
        )

    async def test_log_scale(self, state_no_sim: SessionState, sample_netlist: Path):
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(
                        name="R1",
                        type="component",
                        start=1,
                        stop=1000,
                        points=4,
                        scale="log",
                    )
                ],
            ),
            state_no_sim,
        )

    async def test_empty_parameters(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="At least one parameter"):
            await handle_configure_sweep(
                ConfigureSweepInput(netlist=sample_netlist.name, parameters=[]),
                state_no_sim,
            )

    async def test_step_and_points_conflict(
        self, state_no_sim: SessionState, sample_netlist: Path
    ):
        with pytest.raises(BatchJobError, match="mutually exclusive"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(
                            name="R1",
                            type="component",
                            start=1,
                            stop=10,
                            step=1,
                            points=5,
                        )
                    ],
                ),
                state_no_sim,
            )

    async def test_neither_step_nor_points(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="one of step or points"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[SweepParameter(name="R1", type="component", start=1, stop=10)],
                ),
                state_no_sim,
            )

    async def test_step_zero(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="step must be > 0"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(name="R1", type="component", start=1, stop=10, step=0)
                    ],
                ),
                state_no_sim,
            )

    async def test_points_too_few(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="points must be >= 2"):
            await handle_configure_sweep(
                ConfigureSweepInput(
                    netlist=sample_netlist.name,
                    parameters=[
                        SweepParameter(name="R1", type="component", start=1, stop=10, points=1)
                    ],
                ),
                state_no_sim,
            )

    async def test_multi_dimension(self, state_no_sim: SessionState, sample_netlist: Path):
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1, stop=10, points=3),
                    SweepParameter(name="C1", type="component", start=1e-9, stop=1e-6, points=4),
                ],
            ),
            state_no_sim,
        )


@pytest.mark.asyncio
class TestRunSweep:
    async def test_unknown_config(self, state_no_sim: SessionState):
        with pytest.raises(BatchJobError, match="not found"):
            await handle_run_sweep(RunBatchInput(config_id="missing"), state_no_sim)

    async def test_no_simulator(self, state_no_sim: SessionState, sample_netlist: Path):
        # First create a config
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1, stop=10, points=3)
                ],
            ),
            state_no_sim,
        )
        config_id = next(iter(state_no_sim.sweep_configs.keys()))
        with pytest.raises(SimulationError, match="No SPICE simulator"):
            await handle_run_sweep(RunBatchInput(config_id=config_id), state_no_sim)

    async def test_cancel_during_output_folder_resolve_leaves_no_job(
        self, state_no_sim: SessionState, sample_netlist: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A request cancelled while resolving the output folder must not
        leave a registered (and persisted) "running" batch job with no task
        to advance it — the job is registered only after the last await."""
        await handle_configure_sweep(
            ConfigureSweepInput(
                netlist=sample_netlist.name,
                parameters=[
                    SweepParameter(name="R1", type="component", start=1, stop=10, points=3)
                ],
            ),
            state_no_sim,
        )
        config_id = next(iter(state_no_sim.sweep_configs.keys()))
        state_no_sim.default_simulator = type("FakeSim", (), {})

        entered = asyncio.Event()

        async def hanging_resolve(state, netlist_path=None):
            entered.set()
            await asyncio.Event().wait()  # suspend until cancelled

        monkeypatch.setattr(advanced, "resolve_output_folder", hanging_resolve)

        task = asyncio.create_task(
            handle_run_sweep(RunBatchInput(config_id=config_id), state_no_sim)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert state_no_sim.batch_jobs == {}


@pytest.mark.asyncio
class TestConfigureMonteCarlo:
    async def test_basic(self, state_no_sim: SessionState, sample_netlist: Path):
        result = await handle_configure_montecarlo(
            ConfigureMonteCarloInput(
                netlist=sample_netlist.name,
                tolerances=[
                    MonteCarloTolerance(ref="R", tolerance=0.05, distribution="uniform"),
                    MonteCarloTolerance(ref="R1", tolerance=0.01, distribution="gaussian"),
                ],
                num_runs=50,
            ),
            state_no_sim,
        )
        text = result.content[0].text
        assert "Config ID:" in text
        assert "50" in text
        assert len(state_no_sim.mc_configs) == 1
        # Ack vocabulary maps to the input schema's ref/type distinction, and
        # never uses the internal word "override".
        assert "Per-component tolerances (refs e.g. R1):" in text
        assert "Type-level tolerances (type names e.g. R/resistors):" in text
        assert "override" not in text.lower()

    async def test_empty_tolerances(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="At least one tolerance"):
            await handle_configure_montecarlo(
                ConfigureMonteCarloInput(netlist=sample_netlist.name, tolerances=[], num_runs=10),
                state_no_sim,
            )

    async def test_num_runs_too_high(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="num_runs must be"):
            await handle_configure_montecarlo(
                ConfigureMonteCarloInput(
                    netlist=sample_netlist.name,
                    tolerances=[MonteCarloTolerance(ref="R", tolerance=0.05)],
                    num_runs=999_999,
                ),
                state_no_sim,
            )

    async def test_num_runs_zero(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(BatchJobError, match="num_runs"):
            await handle_configure_montecarlo(
                ConfigureMonteCarloInput(
                    netlist=sample_netlist.name,
                    tolerances=[MonteCarloTolerance(ref="R", tolerance=0.05)],
                    num_runs=0,
                ),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestRunMonteCarlo:
    async def test_unknown_config(self, state_no_sim: SessionState):
        with pytest.raises(BatchJobError, match="not found"):
            await handle_run_montecarlo(RunBatchInput(config_id="missing"), state_no_sim)

    async def test_no_simulator(self, state_no_sim: SessionState, sample_netlist: Path):
        await handle_configure_montecarlo(
            ConfigureMonteCarloInput(
                netlist=sample_netlist.name,
                tolerances=[MonteCarloTolerance(ref="R", tolerance=0.05)],
                num_runs=10,
            ),
            state_no_sim,
        )
        config_id = next(iter(state_no_sim.mc_configs.keys()))
        with pytest.raises(SimulationError, match="No SPICE simulator"):
            await handle_run_montecarlo(RunBatchInput(config_id=config_id), state_no_sim)


@pytest.mark.asyncio
class TestGetBatchResults:
    async def test_unknown_job(self, state_no_sim: SessionState):
        with pytest.raises(BatchJobError):
            await handle_batch_results(GetBatchResultsInput(job_id="missing"), state_no_sim)

    async def test_status_running(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="running")
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        assert "running" in result.content[0].text.lower()

    async def test_status_completed(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="completed", completed_runs=5, total_runs=5)
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        text = result.content[0].text.lower()
        assert "completed" in text

    async def test_status_failed(self, state_no_sim: SessionState):
        bj = _make_batch(state_no_sim, status="failed")
        bj.error = "test error"
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        assert "failed" in result.content[0].text.lower()

    async def test_signal_no_completed_runs(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="completed", completed_runs=0)
        with pytest.raises(BatchJobError, match="No completed runs"):
            await handle_batch_results(
                GetBatchResultsInput(job_id="b1", signal="V(out)"), state_no_sim
            )

    async def test_status_cancelled(self, state_no_sim: SessionState):
        _make_batch(state_no_sim, status="cancelled", completed_runs=2, total_runs=10)
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        assert "cancelled" in result.content[0].text.lower()

    async def test_status_interrupted_returns_partial(self, state_no_sim: SessionState):
        # 'interrupted' is a terminal status assigned on restart recovery when
        # the owning server stopped mid-batch. The formatter must surface the
        # partial results, not raise "unexpected status: interrupted".
        _make_batch(state_no_sim, status="interrupted", completed_runs=3, total_runs=8)
        result = await handle_batch_results(GetBatchResultsInput(job_id="b1"), state_no_sim)
        text = result.content[0].text
        assert "interrupted" in text.lower()
        assert "3 of 8" in text


class TestFormatBatchTextHelpers:
    def test_status_running_with_eta(self):
        from ltspice_mcp.tools.advanced import _format_batch_status_text

        text = _format_batch_status_text(
            {
                "job_id": "b1",
                "job_type": "sweep",
                "status": "running",
                "completed": 5,
                "total": 10,
                "failed": 0,
                "netlist": "x.cir",
                "eta_s": 65.0,
            }
        )
        assert "running" in text
        assert "1m" in text or "65s" in text

    def test_status_running_short_eta(self):
        from ltspice_mcp.tools.advanced import _format_batch_status_text

        text = _format_batch_status_text(
            {
                "job_id": "b1",
                "job_type": "sweep",
                "status": "running",
                "completed": 5,
                "total": 10,
                "failed": 1,
                "netlist": "x.cir",
                "eta_s": 30.0,
            }
        )
        assert "30s" in text

    def test_status_failed(self):
        from ltspice_mcp.tools.advanced import _format_batch_status_text

        text = _format_batch_status_text(
            {
                "job_id": "b1",
                "job_type": "sweep",
                "status": "failed",
                "netlist": "x.cir",
                "error": "boom",
                "completed_runs": 0,
                "total_runs": 5,
                "failed_runs": 5,
                "successful": 0,
                "duration": None,
            }
        )
        assert "failed" in text
        assert "boom" in text

    def test_status_unknown(self):
        from ltspice_mcp.tools.advanced import _format_batch_status_text

        with pytest.raises(BatchJobError):
            _format_batch_status_text({"job_id": "b1", "status": "weird"})

    def test_aggregate_text(self, state_no_sim: SessionState):
        from ltspice_mcp.tools.advanced import _format_batch_aggregate_text

        bj = BatchJob(
            job_id="b1",
            job_type="sweep",
            netlist=Path("/tmp/x.cir"),
            total_runs=3,
            run_results={
                0: {"params": {"R1": 1000.0}},
                1: {"params": {"R1": 2000.0}},
            },
        )
        data = {
            "signal": "V(out)",
            "job_id": "b1",
            "job_type": "sweep",
            "run_count": 2,
            "filtered": False,
            "total_matching": 2,
            "total_available": 2,
            "stats": {
                "max_across_runs": 5.0,
                "min_across_runs": 1.0,
                "mean_across_runs": 3.0,
                "std_across_runs": 1.0,
                "median_across_runs": 3.0,
            },
            "max_case_run": 0,
            "min_case_run": 1,
        }
        text = _format_batch_aggregate_text(data, bj)
        assert "V(out)" in text
        assert "Highest-peak" in text
        assert "Lowest-peak" in text

    def test_raw_text(self):
        from ltspice_mcp.tools.advanced import _format_batch_raw_text

        data = {
            "signal": "V(out)",
            "job_id": "b1",
            "offset": 0,
            "limit": 50,
            "total_matching": 1,
            "runs": [
                {
                    "run_index": 0,
                    "peak": 5.0,
                    "mean": 3.0,
                    "min": 1.0,
                    "params": {"R1": 1000.0},
                }
            ],
            "pagination": {"has_more": True, "next_offset": 50},
        }
        text = _format_batch_raw_text(data)
        assert "V(out)" in text
        assert "Run" in text
        assert "Next page" in text

    def test_raw_text_renders_collapsed_value_rows(self):
        # A run sliced to a single sample (at=/.op-style, or an exactly
        # constant waveform) collapses to a lone "value" key; the table
        # used to render N/A in all three columns, hiding the very number
        # the at= slice computed.
        from ltspice_mcp.tools.advanced import _format_batch_raw_text

        data = {
            "signal": "V(out)",
            "job_id": "b1",
            "offset": 0,
            "limit": 50,
            "total_matching": 1,
            "runs": [{"run_index": 0, "value": 6.6667, "params": {"R2": 2000.0}}],
            "pagination": {"has_more": False, "next_offset": None},
        }
        text = _format_batch_raw_text(data)
        assert "6.6667" in text
        assert "N/A" not in text
