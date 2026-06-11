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

import asyncio
import re
import shutil
from pathlib import Path

import pytest

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import TERMINAL_STATUSES, SessionState
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
from ltspice_mcp.tools.analysis import (
    MeasurementStatsInput,
    QueryValueInput,
    handle_measurement_stats,
    handle_query_value,
)
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


def _extract_id(pattern: str, result) -> str:
    """Pull a Config/Job ID out of a text_response — the real client contract."""
    text = result.content[0].text
    match = re.search(pattern, text)
    assert match is not None, f"{pattern!r} not found in response:\n{text}"
    return match.group(1)


async def _poll_batch_done(state: SessionState, job_id: str, timeout_s: float = 90.0) -> dict:
    """Poll batch_results (the real monitoring path) until the job is terminal."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        res = await handle_batch_results(GetBatchResultsInput(job_id=job_id), state)
        sc = res.structuredContent
        assert sc is not None
        if sc["status"] in TERMINAL_STATUSES:
            return sc
        if loop.time() > deadline:
            pytest.fail(f"batch job {job_id} not terminal after {timeout_s}s: {sc}")
        await asyncio.sleep(0.1)


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


async def test_sweep_full_stack_analytic_values(ngspice_state: SessionState, work_dir: Path):
    # Real parameter sweep through configure_sweep -> run_sweep -> batch_results:
    # three ngspice .dc runs of a divider with R2 = 1k/2k/3k. Every assertion is
    # a known analytic value (V(out) at V1=10 is 10*R2/(R1+R2)), read back via
    # the real batch extraction path: per-run ngspice-dialect raws aggregated by
    # compute_batch_stats. This is the only CI coverage where the batch seam
    # (per-run files, params labeling, dialect, convergence log walk) sees real
    # simulator artifacts.
    net = _write(
        work_dir,
        "sweepdiv.cir",
        "* dc divider\nV1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.dc V1 0 10 5\n.end\n",
    )
    cfg = await handle_configure_sweep(
        ConfigureSweepInput(
            netlist=net,
            parameters=[SweepParameter(name="R2", type="component", values=[1000, 2000, 3000])],
        ),
        ngspice_state,
    )
    assert "Total simulations: 3" in cfg.content[0].text
    config_id = _extract_id(r"Config ID: (\S+)", cfg)

    run = await handle_run_sweep(RunBatchInput(config_id=config_id), ngspice_state)
    job_id = _extract_id(r"Job ID: (\S+)", run)

    status = await _poll_batch_done(ngspice_state, job_id)
    assert status["status"] == "completed"
    assert status["total_runs"] == 3
    assert status["completed_runs"] == 3
    assert status["failed_runs"] == 0
    assert status["successful"] == 3
    # Clean job: error is omitted, not emitted as null (null broke
    # schema-validating clients on the check_job batch branch).
    assert "error" not in status

    # Convergence contract: the per-run log walk ran over the real ngspice logs
    # and found nothing — get_batch_status omits the key entirely for a clean
    # job, and the scan caches an empty list (not None) on the job afterwards.
    assert "convergence_warnings" not in status
    assert ngspice_state.batch_jobs[job_id].convergence_warnings == []

    def vout(r2: float) -> float:
        return 10.0 * r2 / (1000.0 + r2)

    # Aggregate path: compute_batch_stats over the three real raws, sliced to
    # the V1=10 endpoint of each run's .dc axis.
    agg = await handle_batch_results(
        GetBatchResultsInput(job_id=job_id, signal="v(out)", at="10"), ngspice_state
    )
    asc = agg.structuredContent
    assert asc is not None
    assert asc["mode"] == "aggregate"
    assert asc["run_count"] == 3
    stats = asc["stats"]
    assert stats["max_across_runs"] == pytest.approx(vout(3000), rel=1e-6)  # 7.5
    assert stats["min_across_runs"] == pytest.approx(vout(1000), rel=1e-6)  # 5.0
    assert stats["median_across_runs"] == pytest.approx(vout(2000), rel=1e-6)  # 6.667
    assert stats["mean_across_runs"] == pytest.approx(
        (vout(1000) + vout(2000) + vout(3000)) / 3, rel=1e-6
    )

    # Raw mode: each run's value must match the analytic answer for the R2 the
    # sweep recorded in that run's params (order-independent check).
    rawres = await handle_batch_results(
        GetBatchResultsInput(job_id=job_id, signal="v(out)", at="10", raw=True), ngspice_state
    )
    rsc = rawres.structuredContent
    assert rsc is not None
    assert len(rsc["runs"]) == 3
    seen_r2 = set()
    for entry in rsc["runs"]:
        r2 = entry["params"]["R2"]
        seen_r2.add(r2)
        assert entry["value"] == pytest.approx(vout(r2), rel=1e-6)
    assert seen_r2 == {1000.0, 2000.0, 3000.0}

    # The max-case run reported by the aggregate must be the R2=3k run.
    by_index = {e["run_index"]: e for e in rsc["runs"]}
    assert by_index[asc["max_case_run"]]["params"]["R2"] == 3000.0
    assert by_index[asc["min_case_run"]]["params"]["R2"] == 1000.0

    # query_value on a single run of the job must agree with the batch view.
    r2_3k_index = next(e["run_index"] for e in rsc["runs"] if e["params"]["R2"] == 3000.0)
    qres = await handle_query_value(
        QueryValueInput(job_id=job_id, run_index=r2_3k_index, signal="v(out)", at="10"),
        ngspice_state,
    )
    qsc = qres.structuredContent
    assert qsc is not None
    assert qsc["actual_x"] == pytest.approx(10.0)
    assert qsc["value"] == pytest.approx(vout(3000), rel=1e-6)


async def test_montecarlo_without_meas_aggregates_signal_not_measurements(
    ngspice_state: SessionState, work_dir: Path
):
    # A 3-run Monte Carlo on a measurement-less .op deck. ngspice batch mode
    # never evaluates .meas, so the per-job measurement aggregation must raise
    # the explanatory no-results error (not crash, not fabricate entries from
    # the title echo), while signal aggregation over the same runs still works
    # and stays inside the analytic tolerance band.
    net = _write(
        work_dir, "mcdiv.cir", "* op divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    cfg = await handle_configure_montecarlo(
        ConfigureMonteCarloInput(
            netlist=net,
            tolerances=[MonteCarloTolerance(ref="R2", tolerance=0.05)],
            num_runs=3,
            seed=42,
        ),
        ngspice_state,
    )
    config_id = _extract_id(r"Config ID: (\S+)", cfg)
    run = await handle_run_montecarlo(RunBatchInput(config_id=config_id), ngspice_state)
    job_id = _extract_id(r"Job ID: (\S+)", run)

    status = await _poll_batch_done(ngspice_state, job_id)
    assert status["status"] == "completed"
    assert status["completed_runs"] == status["total_runs"] == 3
    assert status["failed_runs"] == 0

    with pytest.raises(ResultError, match=r"No \.MEAS results found across the runs"):
        await handle_measurement_stats(MeasurementStatsInput(job_id=job_id), ngspice_state)

    # Signal aggregation still works on the same runs. R2 is uniform ±5%, so
    # every run's .op point obeys 10*R2'/(1k+R2') for R2' in [950, 1050].
    lo = 10.0 * 950.0 / (1000.0 + 950.0)
    hi = 10.0 * 1050.0 / (1000.0 + 1050.0)
    agg = await handle_batch_results(
        GetBatchResultsInput(job_id=job_id, signal="v(out)"), ngspice_state
    )
    asc = agg.structuredContent
    assert asc is not None
    assert asc["run_count"] == 3
    assert lo <= asc["stats"]["min_across_runs"] <= asc["stats"]["max_across_runs"] <= hi
