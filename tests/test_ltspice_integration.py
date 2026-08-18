"""Integration tests requiring a real LTspice binary.

These tests are skipped unless LTspice integration is explicitly enabled and
LTspice is available on the system.
They exercise the full simulation pipeline: create netlist → run sim → parse results.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
from mcp.types import TextContent, TextResourceContents

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from tests.conftest import terminal_experiment

# Path to the test fixture .asc schematic
_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _text(contents) -> str:
    """Extract text from a resource contents entry, asserting it is text."""
    assert isinstance(contents, TextResourceContents)
    return contents.text


def _result_text(result) -> str:
    """Extract text from a tool result's first content block, asserting it is text."""
    item = result.content[0]
    assert isinstance(item, TextContent)
    return item.text


def _make_ltspice_state(work_dir: Path) -> SessionState | None:
    """Create a SessionState with LTspice detected, or return None."""
    config = ServerConfig(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
    )
    # Load real config to get simulator_exe for WSL
    real_config = ServerConfig.load()
    if real_config.simulator_exe:
        config = ServerConfig(
            simulator="ltspice",
            simulator_exe=real_config.simulator_exe,
            working_dir=work_dir,
            allowed_paths=[work_dir],
            log_level="DEBUG",
        )
    available = detect_simulators(config)
    if "ltspice" not in available:
        return None

    # Configure AscEditor library paths (same as server_lifespan does)
    from ltspice_mcp.engine import configure_asc_editor

    configure_asc_editor(config, available)

    return SessionState.create(config, available)


def _ltspice_available() -> bool:
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        return _make_ltspice_state(Path(td)) is not None


def _ltspice_integration_skip_reason() -> str | None:
    if os.environ.get("LTSPICE_MCP_RUN_LTSPICE_INTEGRATION") != "1":
        return "LTspice integration tests are opt-in; set LTSPICE_MCP_RUN_LTSPICE_INTEGRATION=1"
    if not _ltspice_available():
        return "LTspice not available on this system"
    return None


# Skip entire module unless real LTspice tests are explicitly requested.
_skip_reason = _ltspice_integration_skip_reason()
pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


@pytest.fixture
def ltspice_state(work_dir: Path) -> SessionState:
    state = _make_ltspice_state(work_dir)
    assert state is not None
    return state


@pytest.fixture
def rc_netlist(work_dir: Path) -> Path:
    """Simple RC low-pass filter for AC analysis."""
    p = work_dir / "rc_filter.cir"
    p.write_text(
        "* RC Low-Pass Filter\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 AC 1\n"
        ".ac dec 100 1 1Meg\n"
        ".meas AC fc WHEN mag(V(out))=0.707\n"
        ".END\n"
    )
    return p


@pytest.fixture
def tran_netlist(work_dir: Path) -> Path:
    """Simple RC circuit for transient analysis."""
    p = work_dir / "rc_tran.cir"
    p.write_text(
        "* RC Transient\n"
        "R1 in out 1k\n"
        "C1 out 0 100n\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 0.5m 1m)\n"
        ".tran 0 5m 0 1u\n"
        ".meas TRAN vout_max MAX V(out)\n"
        ".meas TRAN rise_time TRIG V(out) VAL=0.1 RISE=1 TARG V(out) VAL=0.9 RISE=1\n"
        ".END\n"
    )
    return p


@pytest.mark.asyncio
class TestEndToEndSimulation:
    """Full pipeline: create → simulate → check results."""

    async def test_ac_simulation_completes(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        result = await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )
        text = _result_text(result)
        assert "completed successfully" in text, f"Sim failed: {text[:300]}"

    async def test_raw_and_log_files_produced(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        result = await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )
        text = _result_text(result)
        assert "completed successfully" in text, f"Sim did not complete: {text[:200]}"

        # Check job state for raw/log files
        assert ltspice_state.jobs, "No jobs recorded"
        job = next(iter(ltspice_state.jobs.values()))
        assert job.raw_file is not None, "No raw file in job state"
        assert job.raw_file.exists(), f"Raw file does not exist: {job.raw_file}"
        assert job.log_file is not None, "No log file in job state"
        assert job.log_file.exists(), f"Log file does not exist: {job.log_file}"


@pytest.mark.asyncio
class TestJobTracking:
    """Async job tracking: start, poll, complete."""

    async def test_async_job_returns_id(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        result = await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=120),
            ltspice_state,
        )
        text = _result_text(result)
        # Either completed inline (fast) or returned a job ID
        assert "Job ID" in text

    async def test_check_job_status(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.simulation import (
            CheckJobInput,
            RunSimulationInput,
            handle_check_job,
            handle_run_simulation,
        )

        # Run with wait=True to ensure completion
        await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )

        # Check that jobs dict has entries
        assert len(ltspice_state.jobs) > 0
        job_id = next(iter(ltspice_state.jobs))

        # Ran with wait=True, so the job must be terminal-completed. Pin that
        # exact status via the structured channel rather than accepting any text
        # that merely contains the word "status".
        result = await handle_check_job(CheckJobInput(job_id=job_id), ltspice_state)
        sc = result.structuredContent
        assert sc is not None
        assert sc["status"] == "completed"


@pytest.mark.asyncio
class TestMeasExtraction:
    """.MEAS result extraction from simulation log."""

    async def _run_and_get_job(self, state, netlist_name):
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        result = await handle_run_simulation(
            RunSimulationInput(netlist=netlist_name, timeout=60, wait=True), state
        )
        assert "completed successfully" in _result_text(result), _result_text(result)[:200]
        job = next(iter(state.jobs.values()))
        return job

    async def test_get_measurements(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.analysis import SimulationSummaryInput, handle_simulation_summary

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.raw_file and job.log_file and job.log_file.exists()

        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=str(job.raw_file), log_file=str(job.log_file)),
            ltspice_state,
        )
        text = _result_text(result)
        # With Windows-native output dir, .MEAS should work and the summary
        # surfaces the result alongside other metadata.
        assert "fc" in text.lower(), f"Expected 'fc' measurement, got: {text[:300]}"

    async def test_transient_measurements(self, ltspice_state: SessionState, tran_netlist: Path):
        from ltspice_mcp.tools.analysis import SimulationSummaryInput, handle_simulation_summary

        job = await self._run_and_get_job(ltspice_state, tran_netlist.name)
        assert job.raw_file and job.log_file and job.log_file.exists()

        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=str(job.raw_file), log_file=str(job.log_file)),
            ltspice_state,
        )
        text = _result_text(result)
        assert "vout_max" in text.lower(), f"Expected 'vout_max' measurement, got: {text[:300]}"


@pytest.mark.asyncio
class TestACAnalysis:
    """AC analysis signal listing and summary."""

    async def _run_and_get_job(self, state, netlist_name):
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        result = await handle_run_simulation(
            RunSimulationInput(netlist=netlist_name, timeout=60, wait=True), state
        )
        assert "completed successfully" in _result_text(result), _result_text(result)[:200]
        job = next(iter(state.jobs.values()))
        return job

    async def test_list_signals(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.analysis import SimulationSummaryInput, handle_simulation_summary

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.raw_file and job.raw_file.exists()

        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=str(job.raw_file)), ltspice_state
        )
        text = _result_text(result)
        assert "V(out)" in text or "v(out)" in text.lower()

    async def test_simulation_summary(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.tools.analysis import SimulationSummaryInput, handle_simulation_summary

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.raw_file and job.raw_file.exists()

        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=str(job.raw_file)), ltspice_state
        )
        text = _result_text(result)
        assert "frequency" in text.lower() or "ac" in text.lower()


@pytest.mark.asyncio
class TestWSLPathConversion:
    """Verify WSL path conversion works in actual simulation."""

    async def test_wsl_path_in_simulation(self, ltspice_state: SessionState, rc_netlist: Path):
        """If we're on WSL and simulation completes, path conversion worked."""
        from ltspice_mcp.lib.wsl import is_wsl
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        if not is_wsl():
            pytest.skip("Not running in WSL")

        result = await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )
        text = _result_text(result)
        # If simulation completed, WSL path conversion worked
        assert "completed successfully" in text, f"WSL sim failed: {text[:300]}"


@pytest.mark.asyncio
class TestResourcesWithResults:
    """Resource browsing after simulation produces results."""

    async def test_netlists_resource_lists_files(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.resources import handle_read_resource

        result = handle_read_resource("spice://netlists/", ltspice_state)
        text = _text(result.contents[0])
        assert "rc_filter.cir" in text

    async def test_results_resource_after_sim(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.resources import handle_read_resource
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )

        result = handle_read_resource("spice://results/", ltspice_state)
        text = _text(result.contents[0])
        # The one simulation just run must be the only listed result.
        data = json.loads(text)
        assert data["count"] == 1

    async def test_signals_resource_for_job(self, ltspice_state: SessionState, rc_netlist: Path):
        from ltspice_mcp.resources import handle_read_resource
        from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

        await handle_run_simulation(
            RunSimulationInput(netlist=rc_netlist.name, timeout=60, wait=True),
            ltspice_state,
        )

        assert ltspice_state.jobs, "No jobs after simulation"
        job_id = next(iter(ltspice_state.jobs))
        job = ltspice_state.jobs[job_id]

        # Ran with wait=True, so the job must have completed with a raw file — a
        # skip here would silently mask a real simulation failure.
        assert job.status == "completed", f"Job not completed: status={job.status}"
        assert job.raw_file is not None

        result = handle_read_resource(f"spice://results/{job_id}/signals", ltspice_state)
        text = _text(result.contents[0])
        assert "signals" in text.lower()


@pytest.mark.asyncio
class TestExportNetlist:
    """Integration test for export_netlist with a real .asc file."""

    @pytest.fixture
    def asc_in_workdir(self, work_dir: Path) -> Path:
        """Copy the fixture .asc into the test work_dir."""
        src = _FIXTURE_DIR / "Draft1.asc"
        dst = work_dir / "Draft1.asc"
        shutil.copy2(src, dst)
        return dst

    async def test_export_asc_to_net(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import ExportNetlistInput, handle_export_netlist

        result = await handle_export_netlist(
            ExportNetlistInput(path=asc_in_workdir.name), ltspice_state
        )
        text = _result_text(result)
        assert "Draft1" in text
        # Should contain SPICE netlist content
        assert ".net" in text or "R1" in text

    async def test_exported_net_file_exists(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import ExportNetlistInput, handle_export_netlist

        await handle_export_netlist(ExportNetlistInput(path=asc_in_workdir.name), ltspice_state)
        # .net file should exist alongside the .asc
        net_file = asc_in_workdir.with_suffix(".net")
        assert net_file.exists(), f"Expected {net_file} to exist after export"


@pytest.mark.asyncio
class TestUnifiedCircuitTools:
    """Test unified circuit tools work on both .cir and .asc files."""

    @pytest.fixture
    def asc_in_workdir(self, work_dir: Path) -> Path:
        src = _FIXTURE_DIR / "Draft1.asc"
        dst = work_dir / "Draft1.asc"
        shutil.copy2(src, dst)
        return dst

    async def test_list_components_on_asc(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import ListComponentsInput, handle_list_components

        result = await handle_list_components(
            ListComponentsInput(path=asc_in_workdir.name), ltspice_state
        )
        text = _result_text(result)
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text

    async def test_get_component_value_on_asc(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import ListComponentsInput, handle_list_components

        result = await handle_list_components(
            ListComponentsInput(path=asc_in_workdir.name, reference="R1"), ltspice_state
        )
        assert "1k" in _result_text(result)

    async def test_set_component_value_on_asc(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            ListComponentsInput,
            SetComponentValueInput,
            handle_list_components,
            handle_set_component_value,
        )

        await handle_set_component_value(
            SetComponentValueInput(path=asc_in_workdir.name, reference="R1", value="4.7k"),
            ltspice_state,
        )

        result = await handle_list_components(
            ListComponentsInput(path=asc_in_workdir.name, reference="R1"), ltspice_state
        )
        assert "4.7k" in _result_text(result)


@pytest.mark.asyncio
class TestSchematicOnlyTools:
    """Tests for schematic-only operations (position, rotation, attributes, export)."""

    @pytest.fixture
    def asc_in_workdir(self, work_dir: Path) -> Path:
        src = _FIXTURE_DIR / "Draft1.asc"
        dst = work_dir / "Draft1.asc"
        shutil.copy2(src, dst)
        return dst

    async def test_get_schematic_info(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import CircuitReadInput, handle_read_circuit

        result = await handle_read_circuit(
            CircuitReadInput(path=asc_in_workdir.name), ltspice_state
        )
        text = _result_text(result)
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text
        assert "pos=" in text  # positions included
        assert "Wires:" in text

    async def test_get_schematic_info_shows_labels(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import CircuitReadInput, handle_read_circuit

        result = await handle_read_circuit(
            CircuitReadInput(path=asc_in_workdir.name), ltspice_state
        )
        text = _result_text(result)
        # Draft1.asc has FLAG "filtered" label
        assert "filtered" in text

    async def test_move_component(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import MoveComponentInput, handle_move_component

        result = await handle_move_component(
            MoveComponentInput(path=asc_in_workdir.name, reference="R1", x=200, y=200),
            ltspice_state,
        )
        text = _result_text(result)
        assert "Moved R1" in text
        assert "(200,200)" in text

    async def test_move_component_with_rotation(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import MoveComponentInput, handle_move_component

        result = await handle_move_component(
            MoveComponentInput(
                path=asc_in_workdir.name, reference="R1", x=300, y=100, rotation="R0"
            ),
            ltspice_state,
        )
        assert "R0" in _result_text(result)

    async def test_set_component_attribute(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            SetComponentAttributeInput,
            handle_set_component_attribute,
        )

        result = await handle_set_component_attribute(
            SetComponentAttributeInput(
                path=asc_in_workdir.name, reference="R1", attribute="Value", value="4.7k"
            ),
            ltspice_state,
        )
        assert "4.7k" in _result_text(result)

    async def test_remove_component(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import RemoveComponentInput, handle_remove_component

        result = await handle_remove_component(
            RemoveComponentInput(path=asc_in_workdir.name, reference="C1"),
            ltspice_state,
        )
        assert "Removed C1" in _result_text(result)

        # Verify C1 is gone
        from ltspice_mcp.tools.circuit import CircuitReadInput, handle_read_circuit

        info = await handle_read_circuit(CircuitReadInput(path=asc_in_workdir.name), ltspice_state)
        assert "C1" not in _result_text(info)

    async def test_remove_nonexistent_component_raises(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import RemoveComponentInput, handle_remove_component

        with pytest.raises(NetlistError, match="not found"):
            await handle_remove_component(
                RemoveComponentInput(path=asc_in_workdir.name, reference="R99"),
                ltspice_state,
            )

    async def test_export_netlist(self, ltspice_state: SessionState, asc_in_workdir: Path):
        from ltspice_mcp.tools.circuit import ExportNetlistInput, handle_export_netlist

        result = await handle_export_netlist(
            ExportNetlistInput(path=asc_in_workdir.name), ltspice_state
        )
        text = _result_text(result)
        # Should contain SPICE netlist content
        assert ".net" in text or "R1" in text

    async def test_rejects_cir_file(self, ltspice_state: SessionState, work_dir: Path):
        """Schematic-only tools reject .cir files with a helpful message."""
        from ltspice_mcp.tools.circuit import MoveComponentInput, handle_move_component

        cir = work_dir / "test.cir"
        cir.write_text("* test\nR1 1 0 1k\n.END\n")

        with pytest.raises(NetlistError, match=r"requires an \.asc"):
            await handle_move_component(
                MoveComponentInput(path="test.cir", reference="R1", x=0, y=0),
                ltspice_state,
            )


# The live-LTspice wait: WSL interop and .asc export make these runs slower
# than the ngspice tier, so the shared helper gets a longer jobs(wait).
async def _run_experiment(state: SessionState, payload: dict) -> dict:
    return await terminal_experiment(state, payload, wait_timeout_s=240)


@pytest.mark.asyncio
class TestSweepIntegration:
    """Full pipeline: run_experiments fan-out → attached analysis → results."""

    async def test_sweep_runs_every_case(self, ltspice_state: SessionState, rc_netlist: Path):
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-sweep-cases",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [{"kind": "assign", "assign": {"R1": ["500", "1k", "1.5k"]}}],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        # The point of this tier: a real LTspice binary produced these runs.
        assert receipt["source"][0]["simulator"].startswith("LTspice")
        counts = receipt["completeness"]
        assert counts["expanded"] == counts["produced"] == 3
        assert counts["failed"] == counts["cancelled"] == counts["skipped"] == 0
        assert {run["assignments"]["R1"] for run in receipt["runs"]["items"]} == {
            "500",
            "1k",
            "1.5k",
        }

    async def test_sweep_results_carry_the_moving_corner(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        """The attached analysis reads every run's real corner frequency, and
        the extremes are attributed to the case that produced them: a bigger R1
        gives a lower fc for a fixed C."""
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-sweep-corner",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [{"kind": "assign", "assign": {"R1": ["800", "1.2k"]}}],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
                "analyze": {
                    "recipes": [
                        {
                            "key": "corner",
                            "metric": "bode_filter",
                            "signal": "V(out)",
                            "reduce": ["min", "max"],
                            "reduce_field": "cutoff_high_hz",
                        }
                    ],
                    "include": {"per_run": {"limit": 10}},
                },
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        stage = receipt["analysis"]
        assert stage["error"] is None, stage["error"]
        entry = stage["result"]["results"]["corner"]

        reduced = {item["stat"]: item for item in entry["reduced"]}
        # 1/(2*pi*R*100n): 800 ohm -> ~1990 Hz, 1.2k -> ~1326 Hz.
        assert reduced["max"]["assignments"]["R1"] == "800"
        assert reduced["min"]["assignments"]["R1"] == "1.2k"
        assert reduced["max"]["value"] == pytest.approx(1990, rel=0.1)
        assert reduced["min"]["value"] == pytest.approx(1326, rel=0.1)
        assert len(entry["per_run"]["items"]) == 2


@pytest.mark.asyncio
class TestMonteCarloIntegration:
    """Full pipeline: random variation → runs → aggregated analysis."""

    async def test_montecarlo_runs_stay_inside_the_tolerance_band(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        receipt = await _run_experiment(
            ltspice_state,
            {
                "request_id": "lt-mc-rc",
                "circuits": [{"path": rc_netlist.name, "id": "rc"}],
                "variations": [
                    {
                        "kind": "random",
                        "runs": 3,
                        "seed": 7,
                        "rules": [
                            {
                                "rule": "component",
                                "target": "R1",
                                "tolerance": 0.05,
                                "distribution": "uniform",
                            }
                        ],
                    }
                ],
                "execution": {"wait_s": 120, "simulator": "ltspice"},
                "analyze": {
                    "recipes": [
                        {
                            "key": "corner",
                            "metric": "bode_filter",
                            "signal": "V(out)",
                            "reduce": ["min", "max"],
                            "reduce_field": "cutoff_high_hz",
                        }
                    ]
                },
            },
        )
        assert receipt["status"] == "completed", receipt.get("failures")
        counts = receipt["completeness"]
        assert counts["expanded"] == counts["produced"] == 3
        assert counts["failed"] == 0

        stage = receipt["analysis"]
        assert stage["error"] is None, stage["error"]
        reduced = {item["stat"]: item for item in stage["result"]["results"]["corner"]["reduced"]}
        # R1 perturbed uniformly by +/-5% around 1k, C1 fixed at 100n: every
        # sampled resistance and every corner frequency stays in its band.
        for item in reduced.values():
            assert 950.0 <= item["assignments"]["random:component:R1"] <= 1050.0
            assert 1516 <= item["value"] <= 1676  # 1/(2*pi*R*100n) over that band
