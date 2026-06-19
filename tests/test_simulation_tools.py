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
