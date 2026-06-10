"""Real-ngspice end-to-end tests — the CI gate for seam bugs.

Unlike the LTspice integration suite (opt-in + needs a Windows binary, so it
never runs in CI) and the stdio e2e suite (no real simulator), this tier
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
from ltspice_mcp.tools.analysis import QueryValueInput, handle_query_value
from ltspice_mcp.tools.simulation import (
    CheckJobInput,
    RunSimulationInput,
    handle_check_job,
    handle_run_simulation,
)

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


async def test_dc_sweep_endpoint_value(ngspice_state: SessionState, work_dir: Path):
    # .dc sweep of a 1k/1k divider: at V1=5 the output must be exactly half.
    # Exercises the DC branch of sim-type detection AND a real numeric value
    # read back from the ngspice-dialect raw via the query_value handler.
    net = _write(
        work_dir,
        "dcdiv.cir",
        "* dc divider\nV1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.dc V1 0 5 0.5\n.end\n",
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"
    assert "dc" in sc["sim_type"].lower()
    assert any(s.lower() == "v(out)" for s in sc["signals"])

    qres = await handle_query_value(
        QueryValueInput(raw_file=sc["raw_file"], signal="v(out)", at="5"),
        ngspice_state,
    )
    qsc = qres.structuredContent
    assert qsc is not None
    assert qsc["actual_x"] == pytest.approx(5.0)
    assert qsc["value"] == pytest.approx(2.5, rel=1e-6)


async def test_check_job_completed_with_result_files(ngspice_state: SessionState, work_dir: Path):
    # After a synchronous run, check_job(job_id) must rebuild the full success
    # summary from the raw/log files that really exist on disk — the dialect-
    # sensitive re-parse path that listing-only tests never touch.
    net = _write(
        work_dir, "chk.cir", "* divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"

    chk = await handle_check_job(CheckJobInput(job_id=sc["job_id"]), ngspice_state)
    csc = chk.structuredContent
    assert csc is not None
    assert csc["job_id"] == sc["job_id"]
    assert csc["status"] == "completed"
    assert "Operating Point" in csc["sim_type"]
    assert any(s.lower() == "v(out)" for s in csc["signals"])
    assert "error" not in csc
    assert Path(csc["raw_file"]).exists()  # noqa: ASYNC240
    assert Path(csc["log_file"]).exists()  # noqa: ASYNC240


async def test_tran_meas_skipped_in_batch_mode_is_surfaced(
    ngspice_state: SessionState, work_dir: Path
):
    # Verified against real ngspice-42: batch mode (-b with -r rawfile) does
    # NOT evaluate .meas at all ("No .measure possible in batch mode"). The
    # contract is therefore not a measurement value — it is that the requested
    # measurement is reconciled as skipped: an unmet_request observation names
    # it, a warning explains the batch-mode skip, and nothing fabricates a
    # 'vfinal' entry in measurements.
    net = _write(
        work_dir,
        "meas.cir",
        "* rc meas\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\n"
        "R1 in out 1k\n"
        "C1 out 0 1u\n"
        ".tran 10u 5m\n"
        ".meas tran vfinal FIND v(out) AT=4m\n"
        ".end\n",
    )
    res = await handle_run_simulation(RunSimulationInput(netlist=net, wait=True), ngspice_state)
    sc = res.structuredContent
    assert sc is not None
    assert sc["status"] == "completed"
    assert "Transient" in sc["sim_type"]
    assert not (sc.get("measurements") or {}), "batch-mode ngspice cannot have produced .meas"
    unmet = [o for o in sc["observations"] if o["code"] == "unmet_request"]
    assert len(unmet) == 1
    assert unmet[0]["evidence"]["name"] == "vfinal"
    assert unmet[0]["evidence"]["reason"] == "skipped_in_batch_mode"
    assert any("vfinal" in w and "batch mode" in w for w in sc["warnings"])
