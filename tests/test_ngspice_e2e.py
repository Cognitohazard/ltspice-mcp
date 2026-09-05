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
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results
from ltspice_mcp.tools.experiments import JobsInput, handle_jobs
from tests.conftest import terminal_experiment

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
        # ngspice IS on PATH (the module-level skipif already excused its genuine
        # absence), so a detection miss here is a real contract failure — detecting
        # an on-PATH simulator is part of the stack this tier gates. Fail loudly
        # rather than silently darkening the whole e2e tier with a skip.
        pytest.fail("ngspice is on PATH but detect_simulators did not report it usable")
    return SessionState.create(config, available)


def _write(work_dir: Path, name: str, content: str) -> str:
    (work_dir / name).write_text(content)
    return name


def _analysis(receipt: dict) -> dict:
    """The attached analysis payload from a terminal receipt."""
    stage = receipt["analysis"]
    assert stage["error"] is None, stage["error"]
    assert stage["result"] is not None, stage
    return stage["result"]


async def _run_one(state: SessionState, request_id: str, netlist: str, **analyze) -> dict:
    """Run a single deck to a terminal receipt through the live control plane."""
    payload: dict = {
        "request_id": request_id,
        "circuits": [{"path": netlist, "id": "dut"}],
        "execution": {"wait_s": 90, "simulator": "ngspice"},
    }
    if analyze:
        payload["analyze"] = analyze
    receipt = await terminal_experiment(state, payload)
    assert receipt["status"] == "completed", receipt
    return receipt


async def _analyze(state: SessionState, job_id: str, recipes: list[dict], **kw) -> dict:
    """Run recipes over every run of a completed job."""
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "runs": "all", "label": "dut"}],
                "recipes": recipes,
                **kw,
            }
        ),
        state,
    )
    data = result.structuredContent
    assert data is not None, result.content[0].text
    return data


async def _summary(state: SessionState, job_id: str) -> dict:
    """The per-run summary (sim type, point/step counts, temperatures)."""
    data = await _analyze(state, job_id, [{"key": "sum", "metric": "summary"}])
    return data["results"]["sum"]["values"][0]["value"]


async def _signals(state: SessionState, job_id: str) -> list[str]:
    """Every trace name the job's raw carries, via the discovery include."""
    data = await _analyze(
        state,
        job_id,
        [{"key": "sum", "metric": "summary"}],
        include={"signals_available": True},
    )
    available = data["signals_available"]
    assert available, data
    return [name for names in available.values() for name in names]


async def _op(state: SessionState, job_id: str) -> dict:
    """The operating point of a completed .op run."""
    data = await _analyze(state, job_id, [{"key": "op", "metric": "operating_point"}])
    return data["results"]["op"]["values"][0]["value"]


async def test_sweep_full_stack_analytic_values(ngspice_state: SessionState, work_dir: Path):
    # Real parameter sweep through run_experiments -> attached analysis: three
    # ngspice .dc runs of a divider with R2 = 1k/2k/3k. Every assertion is a
    # known analytic value (V(out) at V1=10 is 10*R2/(R1+R2)), read back over
    # the fan-out seam (per-case files, assignment labelling, ngspice dialect)
    # with real simulator artifacts — what hermetic tests cannot reach.
    net = _write(
        work_dir,
        "sweepdiv.cir",
        "* dc divider\nV1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.dc V1 0 10 5\n.end\n",
    )

    def vout(r2: float) -> float:
        return 10.0 * r2 / (1000.0 + r2)

    receipt = await terminal_experiment(
        ngspice_state,
        {
            "request_id": "ng-sweep-divider",
            "circuits": [{"path": net, "id": "div"}],
            "variations": [{"kind": "assign", "assign": {"R2": ["1k", "2k", "3k"]}}],
            "execution": {"wait_s": 90, "simulator": "ngspice"},
            "analyze": {
                "recipes": [
                    {
                        "key": "vout",
                        "metric": "value",
                        "expr": "v(out)",
                        "at": "10",
                        "reduce": ["min", "max", "mean", "p50"],
                    }
                ],
                "include": {"per_run": {"limit": 10}},
            },
        },
    )

    assert receipt["status"] == "completed"
    counts = receipt["completeness"]
    assert counts["expanded"] == counts["produced"] == 3
    assert counts["failed"] == counts["cancelled"] == counts["skipped"] == 0
    assert receipt["failures"] == []

    # Reductions across the three real runs, attributed to the case that owns
    # each extreme — the aggregate view of the fan-out.
    entry = _analysis(receipt)["results"]["vout"]
    reduced = {item["stat"]: item for item in entry["reduced"]}
    assert reduced["min"]["value"] == pytest.approx(vout(1000), rel=1e-6)  # 5.0
    assert reduced["max"]["value"] == pytest.approx(vout(3000), rel=1e-6)  # 7.5
    assert reduced["p50"]["value"] == pytest.approx(vout(2000), rel=1e-6)  # 6.667
    assert reduced["mean"]["value"] == pytest.approx(
        (vout(1000) + vout(2000) + vout(3000)) / 3, rel=1e-6
    )
    # The extremes name their own case, not just a number.
    assert reduced["max"]["assignments"]["R2"] == "3k"
    assert reduced["min"]["assignments"]["R2"] == "1k"

    # Per-run rows: each run's value matches the analytic answer for the R2 the
    # expansion recorded on that case (order-independent).
    rows = entry["per_run"]["items"]
    assert len(rows) == 3
    seen = set()
    for row in rows:
        r2 = row["assignments"]["R2"]
        seen.add(r2)
        assert row["value"]["value"] == pytest.approx(vout(float(r2.rstrip("k")) * 1000), rel=1e-6)
    assert seen == {"1k", "2k", "3k"}

    # Addressing ONE case of the job agrees with the fan-out view: the receipt
    # names the case, and a case-scoped re-analysis reads the same number back.
    r2_3k = next(run for run in receipt["runs"]["items"] if run["assignments"]["R2"] == "3k")
    single = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [
                    {
                        "job_id": receipt["job_id"],
                        "runs": {"case_ids": [r2_3k["case_id"]]},
                        "label": "r2_3k",
                    }
                ],
                "recipes": [{"key": "vout", "metric": "value", "expr": "v(out)", "at": "10"}],
            }
        ),
        ngspice_state,
    )
    ssc = single.structuredContent
    assert ssc is not None
    assert ssc["coverage"]["runs_analyzed"] == 1
    only = ssc["results"]["vout"]["values"][0]["value"]
    assert only["actual_x"] == pytest.approx(10.0)
    assert only["value"] == pytest.approx(vout(3000), rel=1e-6)


async def test_montecarlo_without_meas_reports_no_measurements(
    ngspice_state: SessionState, work_dir: Path
):
    # A 3-run Monte Carlo on a measurement-less .op deck. ngspice batch mode
    # never evaluates .meas, so the measurements recipe must come back empty
    # with an explanation (not crash, not fabricate entries from the title
    # echo), while signal extraction over the same runs still works and stays
    # inside the analytic tolerance band.
    net = _write(
        work_dir, "mcdiv.cir", "* op divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    receipt = await terminal_experiment(
        ngspice_state,
        {
            "request_id": "ng-mc-divider",
            "circuits": [{"path": net, "id": "div"}],
            "variations": [
                {
                    "kind": "random",
                    "runs": 3,
                    "seed": 42,
                    "rules": [
                        {
                            "rule": "component",
                            "target": "R2",
                            "tolerance": 0.05,
                            "distribution": "uniform",
                        }
                    ],
                }
            ],
            "execution": {"wait_s": 90, "simulator": "ngspice"},
        },
    )
    assert receipt["status"] == "completed"
    assert receipt["completeness"]["produced"] == receipt["completeness"]["expanded"] == 3
    assert receipt["completeness"]["failed"] == 0

    # R2 is uniform +/-5%, so every run's .op point obeys 10*R2'/(1k+R2') for
    # R2' in [950, 1050]. Asking for both metrics in one call also pins that a
    # metric with nothing to report does not take the other one down with it.
    lo = 10.0 * 950.0 / (1000.0 + 950.0)
    hi = 10.0 * 1050.0 / (1000.0 + 1050.0)
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": receipt["job_id"], "runs": "all", "label": "mc"}],
                "recipes": [
                    {"key": "measured", "metric": "measurements"},
                    {
                        "key": "vout",
                        "metric": "value",
                        "expr": "v(out)",
                        "reduce": ["min", "max"],
                    },
                ],
            }
        ),
        ngspice_state,
    )
    data = result.structuredContent
    assert data is not None
    assert data["coverage"]["runs_analyzed"] == 3

    reduced = {item["stat"]: item["value"] for item in data["results"]["vout"]["reduced"]}
    assert lo <= reduced["min"] <= reduced["max"] <= hi

    # Every run's perturbed R2 landed inside the requested +/-5% band.
    for item in data["results"]["vout"]["reduced"]:
        assert 950.0 <= item["assignments"]["random:component:R2"] <= 1050.0

    # Nothing was fabricated for the .meas a batch-mode ngspice never evaluated:
    # no 'measured' result at all, and one explanatory failure per case.
    assert "measured" not in data["results"]
    meas_failures = [f for f in data["failures"] if "No .MEAS results" in f["message"]]
    assert len(meas_failures) == 3, data["failures"]
    assert all(f["code"] == "recipe_failed" for f in meas_failures)


async def test_op_divider_full_stack(ngspice_state: SessionState, work_dir: Path):
    # Whole pipeline on a circuit with a known answer: tool -> ngspice -> parse.
    net = _write(
        work_dir, "div.cir", "* divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    receipt = await _run_one(ngspice_state, "ng-op-divider", net)
    summary = await _summary(ngspice_state, receipt["job_id"])
    assert "Operating Point" in summary["sim_type"]
    # ngspice lowercases node names; the divider node must be present.
    signals = await _signals(ngspice_state, receipt["job_id"])
    assert any(s.lower() == "v(out)" for s in signals)
    # A clean divider trips no observation checks and no per-case failure.
    assert summary.get("observations", []) == []
    assert receipt["failures"] == []


async def test_active_npn_switch_op_full_stack(ngspice_state: SessionState, work_dir: Path):
    # An ACTIVE circuit must simulate end to end through the real stack — the
    # capability the removed netlist->asc converter silently lacked (it dropped
    # every active device, so active circuits could not even start). A heavily
    # base-driven NPN saturates, pulling the collector from the 5 V rail down to
    # Vce(sat); an unconducting or floating device would sit near 5 V. The
    # assertion is physical ground truth (saturation), not a model-fit number.
    net = _write(
        work_dir,
        "npnsw.cir",
        "* npn saturated switch\n"
        "Vcc vcc 0 5\n"
        "Vb vb 0 5\n"
        "Rb vb b 10k\n"
        "Rc vcc c 1k\n"
        "Q1 c b 0 QMOD\n"
        ".model QMOD NPN(BF=100)\n"
        ".op\n"
        ".end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-npn-switch", net)
    voltages = (await _op(ngspice_state, receipt["job_id"]))["voltages"]
    vc = next(v for k, v in voltages.items() if k.lower() == "v(c)")
    # Saturated switch: collector pulled well below the 5 V rail, proving the
    # transistor is actually conducting (active), not floating or cut off.
    assert 0.0 < vc < 1.0, voltages


async def test_subckt_macromodel_op_full_stack(ngspice_state: SessionState, work_dir: Path):
    # A .subckt X-instance must simulate end to end. The macromodel is its own
    # parse/expansion path (X-card, port-order binding, internal nodes) — the
    # form every real opamp/regulator ships in, and exactly the multi-terminal
    # construct a converter that "only understood 2-terminal devices" would
    # mishandle. An ideal-opamp .subckt wired as a unity buffer must drive the
    # output to the input: ground truth is V(out) == Vin, not a model-fit number.
    net = _write(
        work_dir,
        "buffer.cir",
        "* opamp subckt unity buffer\n"
        ".subckt opamp inp inn out\n"
        "Eop out 0 inp inn 100k\n"
        ".ends\n"
        "Vin inp 0 2\n"
        "Xop inp out out opamp\n"
        "Rl out 0 1k\n"
        ".op\n"
        ".end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-subckt-buffer", net)
    voltages = (await _op(ngspice_state, receipt["job_id"]))["voltages"]
    vout = next(v for k, v in voltages.items() if k.lower() == "v(out)")
    # Unity buffer of a 2 V input: the subckt expanded and its feedback closed.
    assert vout == pytest.approx(2.0, abs=1e-2), voltages


async def test_mosfet_op_surfaces_device_op_points(ngspice_state: SessionState, work_dir: Path):
    # The absence test for the dropped/mislabeled device internals. A MOSFET in
    # saturation with .save @m1[...] must surface gm/gds/id/vth in the
    # device_op_points bucket — NOT dropped, and vth NOT mislabeled as a node
    # voltage. Runs the full real stack so it would have caught the original bug.
    net = _write(
        work_dir,
        "mosop.cir",
        "* nmos operating point\n"
        "Vd d 0 1.8\n"
        "Vg g 0 1.2\n"
        "M1 d g 0 0 NM L=1u W=10u\n"
        ".model NM NMOS (LEVEL=1 VTO=0.5 KP=120u)\n"
        ".save @m1[id] @m1[gm] @m1[gds] @m1[vth]\n"
        ".op\n"
        ".end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-mosfet-op", net)
    op = await _op(ngspice_state, receipt["job_id"])
    internals = op["device_op_points"]
    # Internals are present (not dropped) and keyed by their @-name.
    assert any("[gm]" in k.lower() for k in internals), internals
    assert any("[id]" in k.lower() for k in internals), internals
    gm = next(v for k, v in internals.items() if "[gm]" in k.lower())
    idd = next(v for k, v in internals.items() if "[id]" in k.lower())
    # Saturated NMOS: positive transconductance and drain current (ground truth).
    assert gm > 0, internals
    assert idd > 0, internals
    # vth is a parameter, never a node voltage — it must NOT be in voltages.
    assert not any("@" in k for k in op["voltages"]), op["voltages"]


async def test_mosfet_dc_sweep_internal_reachable_by_shorthand(
    ngspice_state: SessionState, work_dir: Path
):
    # The headline idiom, end to end: a .dc sweep with .save @m1[gm] makes gm a
    # trace with an axis, reachable by the 'dev.param' shorthand through the
    # value recipe — the gm/ID-style read the tool supports first-class.
    net = _write(
        work_dir,
        "mosdc.cir",
        "* nmos gm vs vgs\n"
        "Vd d 0 1.8\n"
        "Vg g 0 0\n"
        "M1 d g 0 0 NM L=1u W=10u\n"
        ".model NM NMOS (LEVEL=1 VTO=0.5 KP=120u)\n"
        ".save @m1[gm]\n"
        ".dc Vg 0 1.8 0.01\n"
        ".end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-mosfet-dc", net)
    data = await _analyze(
        ngspice_state,
        receipt["job_id"],
        [{"key": "gm", "metric": "value", "expr": "m1.gm", "at": "1.2"}],
    )
    value = data["results"]["gm"]["values"][0]["value"]
    # Above threshold (VTO=0.5), gm at Vgs=1.2 is strictly positive.
    assert value["value"] > 0, value


async def test_no_phantom_circuit_measurement(ngspice_state: SessionState, work_dir: Path):
    # A deck with NO .meas must not yield a fabricated 'circuit' measurement
    # scraped from ngspice's 'Circuit: <title>' echo line. Locks the fix
    # end-to-end against the real ngspice log format (not a recorded fixture).
    net = _write(
        work_dir,
        "rc.cir",
        "* rc lpf fc=1591 hz\nV1 in 0 AC 1\nR1 in out 1k\nC1 out 0 100n\n.ac dec 50 1 1Meg\n.end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-no-phantom-meas", net)
    summary = await _summary(ngspice_state, receipt["job_id"])
    assert "AC Analysis" in summary["sim_type"]
    measurements = summary.get("measurements") or {}
    assert not any(k.lower() == "circuit" for k in measurements), (
        f"phantom 'circuit' measurement scraped from the title echo: {measurements}"
    )


async def test_transient_runs_and_parses(ngspice_state: SessionState, work_dir: Path):
    net = _write(
        work_dir,
        "step.cir",
        "* rc step\nV1 in 0 PULSE(0 1 0 1n 1n 1 2)\nR1 in out 1k\nC1 out 0 1u\n.tran 1u 5m\n.end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-transient", net)
    summary = await _summary(ngspice_state, receipt["job_id"])
    assert "Transient" in summary["sim_type"]


async def test_control_script_deck_produces_readable_raw(
    ngspice_state: SessionState, work_dir: Path
):
    # A `.control` block replaces ngspice's default raw output; without the
    # server's injected `write` this deck (which never writes its own
    # output) would leave nothing for the analysis tools to read even though
    # the run completes cleanly. See inject_ngspice_control_write.
    net = _write(
        work_dir,
        "ctrl_step.cir",
        "* rc step via control script\n"
        "V1 in 0 PULSE(0 1 0 1n 1n 1 2)\n"
        "R1 in out 1k\n"
        "C1 out 0 1u\n"
        ".tran 1u 5m\n"
        ".control\n"
        "run\n"
        ".endc\n"
        ".end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-control-script", net)
    signals = await _signals(ngspice_state, receipt["job_id"])
    assert any(s.lower() == "v(out)" for s in signals)

    data = await _analyze(
        ngspice_state,
        receipt["job_id"],
        [{"key": "vout", "metric": "value", "expr": "v(out)", "at": "5m"}],
    )
    # 5 RC of a 1k/1uF step response: nearly fully charged.
    assert data["results"]["vout"]["values"][0]["value"]["value"] > 0.9


async def test_dc_sweep_endpoint_value(ngspice_state: SessionState, work_dir: Path):
    # .dc sweep of a 1k/1k divider: at V1=5 the output must be exactly half.
    # Exercises the DC branch of sim-type detection AND a real numeric value
    # read back from the ngspice-dialect raw.
    net = _write(
        work_dir,
        "dcdiv.cir",
        "* dc divider\nV1 in 0 0\nR1 in out 1k\nR2 out 0 1k\n.dc V1 0 5 0.5\n.end\n",
    )
    receipt = await _run_one(ngspice_state, "ng-dc-endpoint", net)
    summary = await _summary(ngspice_state, receipt["job_id"])
    assert "dc" in summary["sim_type"].lower()
    signals = await _signals(ngspice_state, receipt["job_id"])
    assert any(s.lower() == "v(out)" for s in signals)

    data = await _analyze(
        ngspice_state,
        receipt["job_id"],
        [{"key": "vout", "metric": "value", "expr": "v(out)", "at": "5"}],
    )
    value = data["results"]["vout"]["values"][0]["value"]
    assert value["actual_x"] == pytest.approx(5.0)
    assert value["value"] == pytest.approx(2.5, rel=1e-6)


async def test_job_status_reports_result_files_that_exist(
    ngspice_state: SessionState, work_dir: Path
):
    # After a run, jobs(status) must report the run's artifacts from what really
    # exists on disk — the dialect-sensitive re-parse path listing-only tests
    # never touch.
    net = _write(
        work_dir, "chk.cir", "* divider\nV1 in 0 10\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"
    )
    receipt = await _run_one(ngspice_state, "ng-job-status", net)
    status = await handle_jobs(
        JobsInput.model_validate({"action": "status", "job_id": receipt["job_id"]}), ngspice_state
    )
    data = status.structuredContent
    assert data is not None
    assert data["job_id"] == receipt["job_id"]
    assert data["status"] == "completed"
    assert data["completeness"]["produced"] == 1

    row = data["runs"]["items"][0]
    assert row["status"] == "produced"

    # The artifacts the run really wrote, reported through the provenance
    # include — the dialect-sensitive re-parse path a listing never touches.
    analysis = await _analyze(
        ngspice_state,
        receipt["job_id"],
        [{"key": "sum", "metric": "summary"}],
        include={"provenance": True},
    )
    (hashes,) = analysis["source_hashes"]
    assert Path(hashes["raw_path"]).exists()  # noqa: ASYNC240
    assert Path(hashes["log_path"]).exists()  # noqa: ASYNC240
    assert hashes["raw_sha256"]


async def test_tran_meas_is_refused_before_submission_in_batch_mode(
    ngspice_state: SessionState, work_dir: Path
):
    # Verified against real ngspice-42: batch mode (-b with -r rawfile) does
    # NOT evaluate .meas at all ("No .measure possible in batch mode"). The deck
    # lint refuses the case up front rather than running it and reporting an
    # unmet request afterwards, so nothing fabricates a 'vfinal' entry and the
    # caller learns why before paying for a simulation.
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
    receipt = await terminal_experiment(
        ngspice_state,
        {
            "request_id": "ng-meas-batch-skip",
            "circuits": [{"path": net, "id": "dut"}],
            "execution": {"wait_s": 90, "simulator": "ngspice"},
        },
    )
    assert receipt["status"] == "completed_with_failures"
    assert receipt["completeness"]["skipped"] == 1
    assert receipt["completeness"]["produced"] == 0

    (finding,) = receipt["lint"][0]["findings"]
    assert finding["rule_id"] == "meas-ngspice-batch"
    assert finding["subject"] == "vfinal"
    assert "batch mode" in finding["evidence"]["reason"]
    assert [f["code"] for f in receipt["failures"]] == ["lint_blocked"]
