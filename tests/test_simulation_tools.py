"""Integration tests for simulation tools in degraded mode (no simulator)."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import SimulationError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation


@pytest.mark.asyncio
class TestSimulationWithoutSimulator:
    async def test_run_raises_no_simulator(self, state_no_sim: SessionState, sample_netlist: Path):
        with pytest.raises(SimulationError, match="No SPICE simulator"):
            await handle_run_simulation(
                RunSimulationInput(netlist=sample_netlist.name), state_no_sim
            )

    async def test_run_nonexistent_netlist(self, state_no_sim: SessionState):
        with pytest.raises(SimulationError, match="not found"):
            await handle_run_simulation(
                RunSimulationInput(netlist="nonexistent.cir"), state_no_sim
            )

    async def test_run_path_escape(self, state_no_sim: SessionState):
        with pytest.raises(SimulationError):
            await handle_run_simulation(RunSimulationInput(netlist="/etc/passwd"), state_no_sim)

    async def test_asc_without_ltspice_names_exe_knob(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # An .asc needs LTspice to export; with none detected, the error must
        # name the executable knob + restart, not just "configure LTspice".
        from ltspice_mcp.tools._base import resolve_runnable_netlist

        asc = work_dir / "x.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        with pytest.raises(SimulationError) as excinfo:
            resolve_runnable_netlist(asc.name, state_no_sim)
        assert "LTSPICE_MCP_SIMULATOR_EXE" in str(excinfo.value)
