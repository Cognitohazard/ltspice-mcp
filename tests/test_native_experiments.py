"""Public native PDK experiments, including independent sample replay."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.pdk_native import ENTRYPOINT, PROFILE, WRAPPER
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
from ltspice_mcp.lib.simulator import detect_simulators
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import JobsInput, handle_jobs
from tests.conftest import LIVENESS_S, terminal_experiment


@pytest.fixture
def native_state(work_dir):
    if shutil.which("ngspice") is None:
        pytest.skip("ngspice is not on PATH")
    root = Path(os.environ.get("LTSPICE_MCP_TEST_PDK_ROOT", "/nonexistent/sky130A"))
    if not (root / ENTRYPOINT).is_file():
        pytest.skip("set LTSPICE_MCP_TEST_PDK_ROOT to the pinned sky130A installation")
    config = ServerConfig(
        simulator="ngspice",
        working_dir=work_dir,
        allowed_paths=[work_dir, root],
        ngbehavior="kiltpsa",
        max_parallel_sims=2,
    )
    available = detect_simulators(config)
    assert "ngspice" in available
    return SessionState.create(config, available), root


def _deck(root: Path, mode: str) -> str:
    section = {"nominal": "tt", "mismatch": "tt_mm", "process": "mc", "combined": "mc"}[mode]
    controls = ".param mc_mm_switch=1\n" if mode == "combined" else ""
    return (
        f'* repeated native devices\n.lib "{(root / ENTRYPOINT).as_posix()}" {section}\n'
        f"{controls}.subckt block d g params: width=1\n"
        f"Xdev d g 0 0 {WRAPPER} w={{width}} l=0.15\n"
        "R1 d 0 100k\n.ends block\n"
        "VA da 0 0.9\nVB db 0 0.9\nVG g 0 0.9\n"
        "XA da g block\nXB db g block\n"
        f".save i(va) i(vb) @m.xa.xdev.m{WRAPPER}[vth] @m.xb.xdev.m{WRAPPER}[vth]\n"
        ".op\n.end\n"
    )


def _payload(name, mode="nominal", *, runs=1, start=0, parallel=1):
    return {
        "request_id": name,
        "circuits": [{"path": "bench.cir", "id": "bench"}],
        "variations": [
            {
                "kind": "pdk_native",
                "id": "process_samples",
                "profile": PROFILE,
                "mode": mode,
                "seed": 1947,
                "runs": runs,
                "sample_start": start,
            }
        ],
        "execution": {"simulator": "ngspice", "wait_s": 120, "max_parallel": parallel},
    }


def _values(raw_path):
    raw = OffsetAwareRawRead(raw_path, dialect="ngspice")
    return tuple(
        float(raw.get_wave(name)[0])
        for name in (
            "i(va)",
            "i(vb)",
            f"v(@m.xa.xdev.m{WRAPPER}[vth])",
            f"v(@m.xb.xdev.m{WRAPPER}[vth])",
        )
    )


@pytest.mark.parametrize("mode", ["nominal", "mismatch", "process", "combined"])
async def test_native_modes_use_pinned_models_and_match_direct_run(native_state, work_dir, mode):
    state, root = native_state
    bench = work_dir / "bench.cir"
    bench.write_text(_deck(root, mode), encoding="utf-8")
    before = hashlib.sha256(bench.read_bytes()).hexdigest()
    receipt = await terminal_experiment(state, _payload("native-" + mode, mode))
    assert receipt["status"] == "completed", receipt
    job = state.all_jobs[receipt["job_id"]]
    case = job.cases[0]
    record = case.native_statistics
    assert record is not None and record.sample is not None and record.prepared is not None
    values = _values(case.raw_file)
    if mode in {"nominal", "process"}:
        assert values[2] == pytest.approx(values[3], abs=1e-12)
    else:
        assert abs(values[2] - values[3]) > 1e-9
    assert len(record.sample.coverage) == 2
    assert record.simulator is not None and record.simulator.version
    assert hashlib.sha256(bench.read_bytes()).hexdigest() == before
    assert case.submitted_at is not None
    assert case.raw_file == Path(record.prepared.paths.raw)
    original = case.raw_file.read_bytes()
    # Same prepared seed/input in a fresh real process is the independent oracle.
    run = await asyncio.to_thread(
        subprocess.run,
        [
            *state.available_simulators["ngspice"].spice_exe,
            "-n",
            "-b",
            str(record.prepared.paths.prepared_driver),
        ],
        cwd=record.prepared.paths.cwd,
        capture_output=True,
        timeout=30,
    )
    assert run.returncode == 0, run.stderr
    assert _values(case.raw_file) == values
    case.raw_file.write_bytes(original)
    loaded = experiment_store.load_job(job.job_id, work_dir, own_is_alive=True)
    assert loaded is not None
    assert loaded.cases[0].native_statistics == record
    public = await handle_jobs(
        JobsInput.model_validate(
            {
                "action": "runs",
                "job_id": job.job_id,
                "run_fields": ["native_statistics"],
            }
        ),
        state,
    )
    assert public.structured_content is not None
    assert (
        public.structured_content["items"][0]["native_statistics"]["validated"]["effective_seed"]
        == record.sample.effective_seed
    )


async def test_native_sample_replay_is_independent_of_parallelism(
    native_state, work_dir, monkeypatch
):
    from ltspice_mcp.lib import pdk_native

    original_cards = pdk_native._original_cards
    parsed = []

    def counted_originals(captures):
        parsed.append(len(captures))
        return original_cards(captures)

    monkeypatch.setattr(pdk_native, "_original_cards", counted_originals)
    state, root = native_state
    (work_dir / "bench.cir").write_text(_deck(root, "combined"), encoding="utf-8")
    snapshots = []
    for name, runs, start, parallel in [
        ("serial", 2, 4, 1),
        ("parallel", 2, 4, 2),
        ("selected", 1, 5, 1),
    ]:
        receipt = await terminal_experiment(
            state, _payload(name, "combined", runs=runs, start=start, parallel=parallel)
        )
        assert receipt["status"] == "completed", receipt
        rows = {}
        for case in state.all_jobs[receipt["job_id"]].cases:
            assert case.native_statistics is not None
            sample = case.native_statistics.sample
            assert sample is not None
            rows[sample.request.sample_index] = (
                sample.sample_key,
                sample.effective_seed,
                _values(case.raw_file),
            )
        snapshots.append(rows)
        assert len(parsed) == len(snapshots), "each circuit parses its original captures once"
    assert snapshots[0] == snapshots[1]
    assert snapshots[2] == {5: snapshots[0][5]}
    assert snapshots[0][4][1] != snapshots[0][5][1]
    assert snapshots[0][4][2] != snapshots[0][5][2]


async def test_missing_native_input_keeps_requested_provenance(state_with_sim):
    state = state_with_sim
    state.available_simulators["ngspice"] = NGspiceSimulator
    receipt = await terminal_experiment(state, _payload("missing", "process", runs=2, start=9))
    assert receipt["status"] == "completed_with_failures", receipt
    job = state.all_jobs[receipt["job_id"]]
    assert [case.native_statistics.request.sample_index for case in job.cases] == [9, 10]
    for case in job.cases:
        assert case.native_statistics.sample is None
        assert case.native_statistics.prepared is None
        assert case.submitted_at is None
        assert case.native_statistics.public()["unavailable"]["effective_seed"]


async def test_unknown_native_profile_is_request_rejection(state_with_sim):
    payload = _payload("unknown")
    state_with_sim.available_simulators["ngspice"] = NGspiceSimulator
    payload["variations"][0]["profile"] = "unrecognized"
    response = await handle_run_experiments(
        RunExperimentsInput.model_validate(payload), state_with_sim
    )
    assert response.structured_content is not None
    assert response.structured_content["error"]["commit_state"] == "not_started"
    assert not state_with_sim.all_jobs


async def test_native_geometry_failure_preserves_other_cases(native_state, work_dir):
    state, root = native_state
    device = f"Xdev d g 0 0 {WRAPPER} w={{width}} l=0.15"
    (work_dir / "bench.cir").write_text(
        _deck(root, "nominal").replace(device, '.include "device.inc"'), encoding="utf-8"
    )
    (work_dir / "device.inc").write_text(device + "\n", encoding="utf-8")
    payload = _payload("geometry")
    payload["variations"].insert(
        0,
        {
            "kind": "assign",
            "instances": [
                {
                    "instance": ["XA", "Xdev"],
                    "attribute": "parameter",
                    "parameter": "w",
                    "values": [1, -1],
                }
            ],
        },
    )
    receipt = await terminal_experiment(state, payload)
    assert receipt["status"] == "completed_with_failures", receipt
    cases = state.all_jobs[receipt["job_id"]].cases
    assert [case.status for case in cases] == ["produced", "failed"]
    assert cases[1].native_statistics.sample is None
    assert cases[1].failure_evidence == {"reason": "geometry"}
    assert cases[1].submitted_at is None
    assert cases[0].native_statistics.sample.coverage[0].width_m == pytest.approx(1e-6)
    dependencies = cases[0].native_statistics.prepared.dependencies
    clones = [item for item in dependencies if item.original_capture == "bench/device.inc"]
    assert len(clones) == 2
    assert len({item.path for item in clones}) == 2


async def test_native_setup_drift_fails_before_launch_and_preserves_peer(
    native_state, work_dir, monkeypatch
):
    from ltspice_mcp.lib import experiment_runner

    state, root = native_state
    (work_dir / "bench.cir").write_text(_deck(root, "nominal"), encoding="utf-8")
    original = experiment_runner.prepare_native_cases

    def drift_after_preparation(job, working_dir, simulator):
        original(job, working_dir, simulator)
        prepared = job.cases[0].native_statistics.prepared
        assert prepared is not None
        Path(prepared.paths.prepared_driver).write_bytes(b"changed after preparation")

    monkeypatch.setattr(experiment_runner, "prepare_native_cases", drift_after_preparation)
    receipt = await terminal_experiment(state, _payload("drift", runs=2))
    assert receipt["status"] == "completed_with_failures", receipt
    cases = state.all_jobs[receipt["job_id"]].cases
    assert [case.status for case in cases] == ["failed", "produced"]
    assert cases[0].submitted_at is None
    assert cases[1].submitted_at is not None
    assert "bytes" in cases[0].error
    assert cases[0].native_statistics.prepared is not None


async def test_changed_bench_dependency_is_rejected_before_sample_derivation(
    native_state, work_dir, monkeypatch
):
    from ltspice_mcp.tools import experiments

    state, root = native_state
    (work_dir / "bench.cir").write_text(
        _deck(root, "nominal").replace("R1 d 0 100k", '.include "load.inc"'),
        encoding="utf-8",
    )
    (work_dir / "load.inc").write_text("R1 d 0 100k\n", encoding="utf-8")
    materialize = experiments.materialize_variants

    def drift_after_materialization(deck, *args, **kwargs):
        cases = materialize(deck, *args, **kwargs)
        dependency = next(
            item.path for item in deck.includes if item.text.strip() == "R1 d 0 100k"
        )
        dependency.write_text("R1 d 0 200k\n", encoding="utf-8")
        return cases

    monkeypatch.setattr(experiments, "materialize_variants", drift_after_materialization)
    receipt = await terminal_experiment(state, _payload("changed-bench"))
    assert receipt["status"] == "completed_with_failures", receipt
    case = state.all_jobs[receipt["job_id"]].cases[0]
    assert case.failure_evidence == {"reason": "artifact_drift"}
    assert case.native_statistics.sample is None
    assert case.native_statistics.prepared is None
    assert case.submitted_at is None
    assert case.raw_file is None


async def test_cancelled_native_case_keeps_preparation_without_claiming_submission(
    native_state, work_dir, monkeypatch
):
    from ltspice_mcp.lib.experiment_runner import ExperimentRunner

    state, root = native_state
    (work_dir / "bench.cir").write_text(_deck(root, "nominal"), encoding="utf-8")
    waiting = asyncio.Event()
    release = asyncio.Event()
    acquire = ExperimentRunner.acquire_launch_slot

    async def wait_for_capacity(runner):
        waiting.set()
        await release.wait()
        await acquire(runner)

    monkeypatch.setattr(ExperimentRunner, "acquire_launch_slot", wait_for_capacity)
    payload = _payload("cancel-native")
    payload["execution"]["wait_s"] = 0
    response = await handle_run_experiments(RunExperimentsInput.model_validate(payload), state)
    assert response.structured_content is not None
    receipt = response.structured_content
    job = state.all_jobs[receipt["job_id"]]
    await asyncio.wait_for(waiting.wait(), LIVENESS_S)
    try:
        cancelled = await handle_jobs(
            JobsInput.model_validate(
                {"action": "cancel", "job_id": job.job_id, "control_token": job.control_token}
            ),
            state,
        )
        assert not cancelled.is_error, cancelled
    finally:
        release.set()
    await asyncio.wait_for(job.done_event.wait(), LIVENESS_S)
    case = job.cases[0]
    assert job.status == "cancelled"
    assert case.status == "cancelled"
    assert case.native_statistics.sample is not None
    assert case.native_statistics.prepared is not None
    assert case.submitted_at is None
    assert case.raw_file is None
    loaded = experiment_store.load_job(job.job_id, work_dir, own_is_alive=True)
    assert loaded is not None
    assert loaded.cases[0].native_statistics == case.native_statistics
    assert loaded.cases[0].submitted_at is None
