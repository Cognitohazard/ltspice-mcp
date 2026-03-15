"""Integration tests requiring a real LTspice binary.

These tests are skipped if LTspice is not available on the system.
They exercise the full simulation pipeline: create netlist → run sim → parse results.
"""

import asyncio
import shutil
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import NetlistError
from ltspice_mcp.lib.simulator import detect_simulators, select_default_simulator
from ltspice_mcp.state import SessionState

# Path to the test fixture .asc schematic
_FIXTURE_DIR = Path(__file__).parent / "fixtures"


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
    from ltspice_mcp.server import _configure_asc_editor
    _configure_asc_editor(config, available)

    return SessionState.create(config, available)


# Skip entire module if LTspice is not available
def _ltspice_available() -> bool:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        return _make_ltspice_state(Path(td)) is not None


pytestmark = pytest.mark.skipif(
    not _ltspice_available(),
    reason="LTspice not available on this system",
)


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

    async def test_ac_simulation_completes(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.simulation import handle_run_simulation

        result = await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )
        text = result[0].text
        assert "completed successfully" in text, f"Sim failed: {text[:300]}"

    async def test_raw_and_log_files_produced(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.simulation import handle_run_simulation

        result = await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )
        text = result[0].text
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

    async def test_async_job_returns_id(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.simulation import handle_run_simulation

        result = await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 120},
            ltspice_state,
        )
        text = result[0].text
        # Either completed inline (fast) or returned a job ID
        assert "Job ID" in text

    async def test_check_job_status(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.simulation import handle_run_simulation, handle_check_job

        # Run with wait=True to ensure completion
        await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )

        # Check that jobs dict has entries
        assert len(ltspice_state.jobs) > 0
        job_id = next(iter(ltspice_state.jobs))

        result = await handle_check_job({"job_id": job_id}, ltspice_state)
        text = result[0].text
        assert "completed" in text.lower() or "status" in text.lower()


@pytest.mark.asyncio
class TestMeasExtraction:
    """.MEAS result extraction from simulation log."""

    async def _run_and_get_job(self, state, netlist_name):
        from ltspice_mcp.tools.simulation import handle_run_simulation
        result = await handle_run_simulation(
            {"netlist": netlist_name, "timeout": 60, "wait": True}, state
        )
        assert "completed successfully" in result[0].text, result[0].text[:200]
        job = next(iter(state.jobs.values()))
        return job

    async def test_get_measurements(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.analysis import handle_get_measurements

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.log_file and job.log_file.exists()

        result = await handle_get_measurements(
            {"log_file": str(job.log_file)}, ltspice_state
        )
        text = result[0].text
        # With Windows-native output dir, .MEAS should work
        assert "fc" in text.lower(), f"Expected 'fc' measurement, got: {text[:300]}"

    async def test_transient_measurements(
        self, ltspice_state: SessionState, tran_netlist: Path
    ):
        from ltspice_mcp.tools.analysis import handle_get_measurements

        job = await self._run_and_get_job(ltspice_state, tran_netlist.name)
        assert job.log_file and job.log_file.exists()

        result = await handle_get_measurements(
            {"log_file": str(job.log_file)}, ltspice_state
        )
        text = result[0].text
        assert "vout_max" in text.lower(), f"Expected 'vout_max' measurement, got: {text[:300]}"


@pytest.mark.asyncio
class TestACAnalysis:
    """AC analysis signal listing and summary."""

    async def _run_and_get_job(self, state, netlist_name):
        from ltspice_mcp.tools.simulation import handle_run_simulation
        result = await handle_run_simulation(
            {"netlist": netlist_name, "timeout": 60, "wait": True}, state
        )
        assert "completed successfully" in result[0].text, result[0].text[:200]
        job = next(iter(state.jobs.values()))
        return job

    async def test_list_signals(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.analysis import handle_get_simulation_summary

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.raw_file and job.raw_file.exists()

        result = await handle_get_simulation_summary(
            {"raw_file": str(job.raw_file)}, ltspice_state
        )
        text = result[0].text
        assert "V(out)" in text or "v(out)" in text.lower()

    async def test_simulation_summary(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.analysis import handle_get_simulation_summary

        job = await self._run_and_get_job(ltspice_state, rc_netlist.name)
        assert job.raw_file and job.raw_file.exists()

        result = await handle_get_simulation_summary(
            {"raw_file": str(job.raw_file)}, ltspice_state
        )
        text = result[0].text
        assert "frequency" in text.lower() or "ac" in text.lower()


@pytest.mark.asyncio
class TestWSLPathConversion:
    """Verify WSL path conversion works in actual simulation."""

    async def test_wsl_path_in_simulation(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        """If we're on WSL and simulation completes, path conversion worked."""
        from ltspice_mcp.lib.wsl import is_wsl
        from ltspice_mcp.tools.simulation import handle_run_simulation

        if not is_wsl():
            pytest.skip("Not running in WSL")

        result = await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )
        text = result[0].text
        # If simulation completed, WSL path conversion worked
        assert "completed successfully" in text, f"WSL sim failed: {text[:300]}"


@pytest.mark.asyncio
class TestResourcesWithResults:
    """Resource browsing after simulation produces results."""

    async def test_netlists_resource_lists_files(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.resources import handle_read_resource

        result = await handle_read_resource("ltspice://netlists/", ltspice_state)
        text = result.contents[0].text
        assert "rc_filter.cir" in text

    async def test_results_resource_after_sim(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.resources import handle_read_resource
        from ltspice_mcp.tools.simulation import handle_run_simulation

        await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )

        result = await handle_read_resource("ltspice://results/", ltspice_state)
        text = result.contents[0].text
        assert '"count": 1' in text or "simulation" in text.lower()

    async def test_signals_resource_for_job(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.resources import handle_read_resource
        from ltspice_mcp.tools.simulation import handle_run_simulation

        await handle_run_simulation(
            {"netlist": rc_netlist.name, "timeout": 60, "wait": True},
            ltspice_state,
        )

        assert ltspice_state.jobs, "No jobs after simulation"
        job_id = next(iter(ltspice_state.jobs))
        job = ltspice_state.jobs[job_id]

        if job.status != "completed" or job.raw_file is None:
            pytest.skip(f"Job not completed: status={job.status}")

        result = await handle_read_resource(
            f"ltspice://results/{job_id}/signals", ltspice_state
        )
        text = result.contents[0].text
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

    async def test_export_asc_to_net(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_export_netlist

        result = await handle_export_netlist(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        text = result[0].text
        assert "Draft1" in text
        # Should contain SPICE netlist content
        assert ".net" in text or "R1" in text

    async def test_exported_net_file_exists(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_export_netlist

        await handle_export_netlist(
            {"path": asc_in_workdir.name}, ltspice_state
        )
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

    async def test_list_components_on_asc(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_list_components

        result = await handle_list_components(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        text = result[0].text
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text

    async def test_get_component_value_on_asc(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_list_components

        result = await handle_list_components(
            {"path": asc_in_workdir.name, "reference": "R1"}, ltspice_state
        )
        assert "1k" in result[0].text

    async def test_set_component_value_on_asc(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import (
            handle_list_components,
            handle_set_component_value,
        )

        await handle_set_component_value(
            {"path": asc_in_workdir.name, "reference": "R1", "value": "4.7k"},
            ltspice_state,
        )

        result = await handle_list_components(
            {"path": asc_in_workdir.name, "reference": "R1"}, ltspice_state
        )
        assert "4.7k" in result[0].text


@pytest.mark.asyncio
class TestSchematicOnlyTools:
    """Tests for schematic-only operations (position, rotation, attributes, export)."""

    @pytest.fixture
    def asc_in_workdir(self, work_dir: Path) -> Path:
        src = _FIXTURE_DIR / "Draft1.asc"
        dst = work_dir / "Draft1.asc"
        shutil.copy2(src, dst)
        return dst

    async def test_get_schematic_info(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_read_circuit

        result = await handle_read_circuit(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        text = result[0].text
        assert "R1" in text
        assert "C1" in text
        assert "V1" in text
        assert "pos=" in text  # positions included
        assert "Wires:" in text

    async def test_get_schematic_info_shows_labels(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_read_circuit

        result = await handle_read_circuit(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        text = result[0].text
        # Draft1.asc has FLAG "filtered" label
        assert "filtered" in text

    async def test_move_component(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_move_component

        result = await handle_move_component(
            {"path": asc_in_workdir.name, "reference": "R1", "x": 200, "y": 200},
            ltspice_state,
        )
        text = result[0].text
        assert "Moved R1" in text
        assert "(200,200)" in text

    async def test_move_component_with_rotation(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_move_component

        result = await handle_move_component(
            {
                "path": asc_in_workdir.name,
                "reference": "R1",
                "x": 300,
                "y": 100,
                "rotation": "R0",
            },
            ltspice_state,
        )
        assert "R0" in result[0].text

    async def test_set_component_attribute(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_set_component_attribute

        result = await handle_set_component_attribute(
            {
                "path": asc_in_workdir.name,
                "reference": "R1",
                "attribute": "Value",
                "value": "4.7k",
            },
            ltspice_state,
        )
        assert "4.7k" in result[0].text

    async def test_remove_component(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_remove_component

        result = await handle_remove_component(
            {"path": asc_in_workdir.name, "reference": "C1"},
            ltspice_state,
        )
        assert "Removed C1" in result[0].text

        # Verify C1 is gone
        from ltspice_mcp.tools.circuit import handle_read_circuit

        info = await handle_read_circuit(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        assert "C1" not in info[0].text

    async def test_remove_nonexistent_component_raises(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_remove_component

        with pytest.raises(NetlistError, match="not found"):
            await handle_remove_component(
                {"path": asc_in_workdir.name, "reference": "R99"},
                ltspice_state,
            )

    async def test_export_netlist(
        self, ltspice_state: SessionState, asc_in_workdir: Path
    ):
        from ltspice_mcp.tools.circuit import handle_export_netlist

        result = await handle_export_netlist(
            {"path": asc_in_workdir.name}, ltspice_state
        )
        text = result[0].text
        # Should contain SPICE netlist content
        assert ".net" in text or "R1" in text

    async def test_rejects_cir_file(
        self, ltspice_state: SessionState, work_dir: Path
    ):
        """Schematic-only tools reject .cir files with a helpful message."""
        from ltspice_mcp.tools.circuit import handle_move_component

        cir = work_dir / "test.cir"
        cir.write_text("* test\nR1 1 0 1k\n.END\n")

        with pytest.raises(NetlistError, match="requires an .asc"):
            await handle_move_component(
                {"path": "test.cir", "reference": "R1", "x": 0, "y": 0},
                ltspice_state,
            )


@pytest.mark.asyncio
class TestSweepIntegration:
    """Full pipeline: configure sweep → run → check → get results."""

    async def test_configure_and_run_sweep(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.advanced import (
            handle_get_batch_results,
            handle_configure_sweep,
            handle_run_sweep,
        )

        # Configure a small 3-point sweep on R1
        config_result = await handle_configure_sweep(
            {
                "netlist": rc_netlist.name,
                "parameters": [
                    {
                        "name": "R1",
                        "type": "component",
                        "start": 500,
                        "stop": 1500,
                        "points": 3,
                    }
                ],
            },
            ltspice_state,
        )
        config_text = config_result[0].text
        assert "Config ID" in config_text
        assert "Total simulations: 3" in config_text

        # Extract config_id
        config_id = None
        for line in config_text.splitlines():
            if line.startswith("Config ID:"):
                config_id = line.split(":", 1)[1].strip()
                break
        assert config_id is not None

        # Run the sweep
        run_result = await handle_run_sweep(
            {"config_id": config_id}, ltspice_state
        )
        run_text = run_result[0].text
        assert "Job ID" in run_text

        # Extract job_id
        job_id = None
        for line in run_text.splitlines():
            if line.startswith("Job ID:"):
                job_id = line.split(":", 1)[1].strip()
                break
        assert job_id is not None

        # Wait for completion (poll with timeout)
        batch_job = ltspice_state.batch_jobs[job_id]
        try:
            await asyncio.wait_for(batch_job.done_event.wait(), timeout=120)
        except asyncio.TimeoutError:
            pytest.fail(f"Sweep job {job_id} timed out")

        # Check status
        status_result = await handle_get_batch_results(
            {"job_id": job_id}, ltspice_state
        )
        status_text = status_result[0].text
        assert "completed" in status_text.lower(), f"Sweep not completed: {status_text[:300]}"

    async def test_sweep_results_queryable(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.advanced import (
            handle_configure_sweep,
            handle_get_batch_results,
            handle_run_sweep,
        )

        config_result = await handle_configure_sweep(
            {
                "netlist": rc_netlist.name,
                "parameters": [
                    {
                        "name": "R1",
                        "type": "component",
                        "start": 800,
                        "stop": 1200,
                        "points": 3,
                    }
                ],
            },
            ltspice_state,
        )
        config_id = config_result[0].text.split("Config ID:")[1].split("\n")[0].strip()

        run_result = await handle_run_sweep({"config_id": config_id}, ltspice_state)
        job_id = run_result[0].text.split("Job ID:")[1].split("\n")[0].strip()

        batch_job = ltspice_state.batch_jobs[job_id]
        await asyncio.wait_for(batch_job.done_event.wait(), timeout=120)

        # Query results for V(out)
        results = await handle_get_batch_results(
            {"job_id": job_id, "signal": "V(out)"}, ltspice_state
        )
        text = results[0].text
        assert "Batch Results" in text
        assert "V(out)" in text


@pytest.mark.asyncio
class TestMonteCarloIntegration:
    """Full pipeline: configure MC → run → check → get results."""

    async def test_configure_and_run_montecarlo(
        self, ltspice_state: SessionState, rc_netlist: Path
    ):
        from ltspice_mcp.tools.advanced import (
            handle_get_batch_results,
            handle_configure_montecarlo,
            handle_run_montecarlo,
        )

        config_result = await handle_configure_montecarlo(
            {
                "netlist": rc_netlist.name,
                "tolerances": [
                    {"ref": "resistors", "tolerance": 0.05, "distribution": "uniform"},
                    {"ref": "capacitors", "tolerance": 0.1, "distribution": "gaussian"},
                ],
                "num_runs": 5,
            },
            ltspice_state,
        )
        config_text = config_result[0].text
        assert "Config ID" in config_text
        assert "Runs: 5" in config_text

        config_id = config_text.split("Config ID:")[1].split("\n")[0].strip()

        run_result = await handle_run_montecarlo(
            {"config_id": config_id}, ltspice_state
        )
        run_text = run_result[0].text
        assert "Job ID" in run_text

        job_id = run_text.split("Job ID:")[1].split("\n")[0].strip()

        batch_job = ltspice_state.batch_jobs[job_id]
        try:
            await asyncio.wait_for(batch_job.done_event.wait(), timeout=180)
        except asyncio.TimeoutError:
            pytest.fail(f"Monte Carlo job {job_id} timed out")

        status_result = await handle_get_batch_results(
            {"job_id": job_id}, ltspice_state
        )
        status_text = status_result[0].text
        assert "completed" in status_text.lower(), f"MC not completed: {status_text[:300]}"
