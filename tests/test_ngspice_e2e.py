"""Real-ngspice end-to-end tests — the CI gate for seam bugs.

Unlike the LTspice integration suite (opt-in + needs a Windows binary, so it
never runs in CI) and the stdio e2e suite (opt-in, no real simulator), this tier
gates ONLY on ``ngspice`` being on PATH — which it is on Linux CI runners. It
drives the REAL tool handlers against REAL ngspice so the full stack runs:
handler dispatch -> SessionState -> runner -> ngspice -> RawRead (ngspice
dialect) -> log/result parsing. That is exactly the seam hermetic unit tests
cannot reach, where the six live-found defects (and the phantom-measurement bug)
lived. Run shape assertions on REAL ngspice output, not hand-built fixtures.
"""

import shutil
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.simulation import RunSimulationInput, handle_run_simulation

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH"),
]


@pytest.fixture
def ngspice_state(work_dir: Path) -> SessionState:
    config = ServerConfig(
        simulator="ngspice",
        working_dir=work_dir,
        allowed_paths=[work_dir],
        log_level="DEBUG",
    )
    available = detect_simulators(config)
    if "ngspice" not in available:
        pytest.skip("ngspice detected on PATH but not usable by detect_simulators")
    return SessionState.create(config, available)


def _write(work_dir: Path, name: str, content: str) -> str:
    (work_dir / name).write_text(content)
    return name


async def test_op_divider_full_stack(ngspice_state: SessionState, work_dir: Path):
    # Whole pipeline on a circuit with a known answer: handler -> ngspice -> parse.
    net = _write(
        work_dir, "div.cir", "* divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"
    assert "Operating Point" in sc["sim_type"]
    # ngspice lowercases node names; the divider node must be present.
    assert any(s.lower() == "v(out)" for s in sc["signals"])
    # A clean divider trips no observation checks.
    assert sc["observations"] == []


async def test_no_phantom_circuit_measurement(ngspice_state: SessionState, work_dir: Path):
    # A deck with NO .meas must not yield a fabricated 'circuit' measurement
    # scraped from ngspice's 'Circuit: <title>' echo line. Locks the fix
    # end-to-end against the real ngspice log format (not a recorded fixture).
    net = _write(
        work_dir,
        "rc.cir",
        "* rc lpf fc=1591 hz\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 100n\n.ac dec 50 1 1Meg\n.end\n",
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"
    assert "AC Analysis" in sc["sim_type"]
    measurements = sc.get("measurements") or {}
    assert not any(k.lower() == "circuit" for k in measurements), (
        f"phantom 'circuit' measurement scraped from the title echo: {measurements}"
    )


async def test_transient_runs_and_parses(ngspice_state: SessionState, work_dir: Path):
    net = _write(
        work_dir,
        "step.cir",
        "* rc step\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.tran 1u 5m\n.end\n",
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"
    assert "Transient" in sc["sim_type"]
