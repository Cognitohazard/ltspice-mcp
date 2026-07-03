"""Integration tests for simulation tools in degraded mode (no simulator)."""

import asyncio
import threading
import time
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
            await resolve_runnable_netlist(asc.name, state_no_sim)
        assert "LTSPICE_MCP_SIMULATOR_EXE" in str(excinfo.value)

    async def test_concurrent_same_asc_exports_serialized(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # LTspice always writes the same sidecar .net next to the .asc, so two
        # concurrent exports of one schematic must not run in parallel — the
        # per-.asc lock has to serialize them (a torn .net means running the
        # wrong deck). The offloaded export made this reachable; pin it.
        from ltspice_mcp.tools._base import resolve_runnable_netlist

        asc = work_dir / "race.asc"
        asc.write_text("Version 4\nSHEET 1 880 680\n")
        net = asc.with_suffix(".net")

        gate = threading.Lock()
        active = 0
        max_active = 0

        class FakeLTspice:
            @classmethod
            def create_netlist(cls, path: str) -> str:
                nonlocal active, max_active
                with gate:
                    active += 1
                    max_active = max(max_active, active)
                time.sleep(0.05)  # hold the "export" long enough to overlap
                net.write_text("* exported\n.end\n")
                with gate:
                    active -= 1
                return str(net)

        state_no_sim.available_simulators["ltspice"] = FakeLTspice
        try:
            results = await asyncio.gather(
                resolve_runnable_netlist(asc.name, state_no_sim),
                resolve_runnable_netlist(asc.name, state_no_sim),
            )
        finally:
            del state_no_sim.available_simulators["ltspice"]

        assert [str(p) for p in results] == [str(net)] * 2
        assert max_active == 1
