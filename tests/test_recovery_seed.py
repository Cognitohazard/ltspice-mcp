"""Explicit ngspice seeds through frozen inputs and the existing coordinator."""

import asyncio
import json
import math
import shutil
import struct
from dataclasses import replace
from pathlib import Path, PureWindowsPath

import pytest
from pydantic import ValidationError

from ltspice_mcp.api import Api
from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.controlled_ngspice import prepare_seeded_driver
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import capture_case_inputs, verify_case_inputs
from ltspice_mcp.lib.experiment_resume import resume_experiment
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY, ArtifactDigest, NativePaths, driver_bytes
from ltspice_mcp.lib.recovery_journal import new_root_journal, save_journal
from ltspice_mcp.lib.recovery_records import ExecutionRecord, RecoveryError
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.test_recovery_launch import _process, _start
from tests.test_recovery_launch import committed as committed
from tests.test_recovery_records import recovery_job
from tests.test_resume_surface import _state


def _request(seed=None, **execution):
    return RunExperimentsInput.model_validate(
        {
            "request_id": "seeded",
            "circuits": [{"path": "deck.cir"}],
            "execution": {"recoverable": True, "simulator_seed": seed, **execution},
        }
    )


@pytest.mark.parametrize("seed", [1, 17, 2147483646])
def test_seed_is_a_strict_bounded_execution_fact(tmp_path, seed):
    args = _request(seed)
    assert args.execution.simulator_seed == seed
    recovery = recovery_job(tmp_path).recovery
    assert recovery is not None
    execution = replace(recovery.execution, simulator_seed=seed)
    assert ExecutionRecord.from_record(execution.to_record()) == execution
    assert args.strip_presentation()["execution"]["simulator_seed"] == seed


@pytest.mark.parametrize("seed", [True, False, 0, -1, 2147483647, 1.0, "17"])
def test_invalid_seed_refuses_in_requests_and_saved_records(tmp_path, seed):
    with pytest.raises(ValidationError):
        _request(seed)
    recovery = recovery_job(tmp_path).recovery
    assert recovery is not None
    record = recovery.execution.to_record()
    record["simulator_seed"] = seed
    with pytest.raises(RecoveryError):
        ExecutionRecord.from_record(record)


def test_seed_requires_recovery_and_cannot_mix_with_native(tmp_path):
    with pytest.raises(ValidationError):
        _request(17, recoverable=False)
    with pytest.raises(ValidationError):
        _request(17, simulator="ltspice")
    recovery = recovery_job(tmp_path).recovery
    assert recovery is not None
    with pytest.raises(ValueError, match="native statistics"):
        replace(
            recovery.execution,
            simulator_seed=17,
            native_policy=LAUNCH_POLICY,
        )


def test_absent_seed_preserves_old_request_identity_and_record(tmp_path):
    args = _request()
    assert "simulator_seed" not in args.strip_presentation()["execution"]
    recovery = recovery_job(tmp_path).recovery
    assert recovery is not None
    record = recovery.execution.to_record()
    record.pop("simulator_seed", None)
    assert ExecutionRecord.from_record(record).simulator_seed is None


def test_shared_sequence_preserves_the_audited_native_driver_bytes():
    root = PureWindowsPath("C:/jobs/with spaces")
    paths = NativePaths(
        root,
        root / "draw.input.cir",
        root / "draw.setup.cir",
        root / "draw.cir",
        root / "draw.raw",
        root / "draw.log",
    )
    assert driver_bytes(17, paths, "draw") == (
        b"Native PDK initialization\n.control\nset ngbehavior=hsa\n"
        b"set ng_nomodcheck\nsetseed 17\nsource draw.input.cir\n"
        b"run\nwrite draw.raw\nquit\n.endc\n.end\n"
    )


def _capture(tmp_path, body, *, seeded, analysis=b".op"):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    case.staged_deck.write_bytes(b"* seeded\n" + body + b"\n" + analysis + b"\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    return capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder, seeded=seeded)


@pytest.mark.parametrize(
    "expression", ["agauss(10,1,1)", "gauss(10,.1,1)", "aunif(10,1)", "unif(10,.1)", "limit(10,1)"]
)
def test_only_source_verified_static_random_functions_are_admitted(tmp_path, expression):
    captured = _capture(
        tmp_path, f".param draw={expression}\nV1 n 0 {{draw}}".encode(), seeded=True
    )
    verify_case_inputs(captured, seeded=True)
    with pytest.raises(RecoveryError, match="randomness"):
        verify_case_inputs(captured)


@pytest.mark.parametrize(
    "body",
    [
        b".param draw=rand()",
        b".param draw=sgauss(0)",
        b"V1 n 0 trnoise(1 1n)",
        b"V1 n 0 trrandom(1 1n)",
        b"B1 n 0 V=unknown_function(1)",
        b".control\nrun\n.endc",
        b".load ambient.cm",
        b"A1 n 0 external_module",
    ],
)
def test_seeded_admission_keeps_other_recovery_guards(tmp_path, body):
    with pytest.raises(RecoveryError):
        _capture(tmp_path, body, seeded=True)


@pytest.mark.parametrize(
    "analysis",
    [
        b".op\n.ac dec 3 1 10",
        b".op\n.tran 1n 10n",
        b".op\n.op",
        b".noise V(n) V1 dec 3 1 10",
        b"",
        b".op\n.step param resistance list 1 2",
    ],
)
def test_seeded_analysis_without_a_single_plot_contract_refuses(tmp_path, analysis):
    with pytest.raises(RecoveryError, match="exactly one"):
        _capture(tmp_path, b"V1 n 0 DC 1 AC 1\nR1 n 0 1000", seeded=True, analysis=analysis)


@pytest.mark.parametrize("analysis", [b".op", b".ac dec 3 1 10", b".dc V1 1 2 1", b".tran 1n 10n"])
def test_seeded_single_analysis_is_admitted(tmp_path, analysis):
    captured = _capture(tmp_path, b"V1 n 0 DC 1 AC 1\nR1 n 0 1000", seeded=True, analysis=analysis)
    verify_case_inputs(captured, seeded=True)


async def _submit_seeded(state, request_id, seed, *, variations=None):
    args = RunExperimentsInput.model_validate(
        {
            "request_id": request_id,
            "circuits": [{"path": str(state.working_dir / "draw.cir")}],
            "execution": {
                "recoverable": True,
                "simulator_seed": seed,
                "wait_s": 5,
                "run_timeout_s": 10,
                "max_parallel": 1,
            },
            "variations": variations or [],
            "lint": "off",
        }
    )
    result = await asyncio.wait_for(handle_run_experiments(args, state), 15)
    assert not result.is_error, result.structured_content
    job = state.all_jobs[result.structured_content["job_id"]]
    await asyncio.wait_for(job.task, 15)
    await state.job_registry.drain_pending()
    return job


def _voltage(case):
    content = Path(case.raw_file).read_bytes()
    assert b"v(n)" in content
    values = struct.unpack("<dd", content.split(b"Binary:\n", 1)[1])
    return values[0]


def _write_draw(folder):
    # Includes and directories with spaces exercise the source command's path
    # handling rather than relying on a flat, dependency-free test deck.
    (folder / "models with spaces").mkdir()
    (folder / "models with spaces" / "draw.inc").write_bytes(b".param draw=agauss(10,1,1)\n")
    (folder / "draw.cir").write_bytes(
        b'* Seeded parameter evaluation\n.include "models with spaces/draw.inc"\n'
        b"V1 n 0 {draw}\nR1 n 0 1000\n.op\n.end\n"
    )


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
def test_python_api_preserves_seed_execution_field(work_dir):
    _write_draw(work_dir)
    with Api(
        working_dir=work_dir,
        config_path=work_dir / "owned-config.toml",
        allowed_paths=[work_dir],
        enabled_simulators=["ngspice"],
        simulator="ngspice",
        simulator_exe=shutil.which("ngspice"),
        ngbehavior="hsa",
        run_timeout=10,
    ) as api:
        result = api.run_experiments(
            request_id="api-seeded",
            circuits=[{"path": str(work_dir / "draw.cir")}],
            execution={"simulator": "ngspice", "recoverable": True, "simulator_seed": 17},
            lint="off",
        )
        assert result["status"] == "completed"
        recorded = experiment_store.load_job(result["job_id"], work_dir, own_is_alive=True)
        assert recorded is not None
        recovery = recorded.recovery
        assert recovery is not None and recovery.execution.simulator_seed == 17
        case_recovery = recorded.cases[0].recovery
        assert case_recovery is not None
        launch = case_recovery.attempt.launch
        assert launch is not None and launch.adaptation == "seeded_driver"
        assert recorded.cases[0].native_statistics is None


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.asyncio
async def test_real_seeded_coordinator_reproduces_electrical_results(work_dir):
    _write_draw(work_dir)
    state = _state(work_dir)
    jobs = [await _submit_seeded(state, f"seed-{i}", seed) for i, seed in enumerate([17, 17, 19])]
    assert all(job.status == "completed" for job in jobs), [job.failures for job in jobs]
    values = [_voltage(job.cases[0]) for job in jobs]
    print("Seeded coordinator voltages:", json.dumps(values))
    assert values[0] == values[1]
    assert values[0] != values[2]
    for job in jobs:
        case = job.cases[0]
        assert case.native_statistics is None
        assert case.recovery.inputs.electrical.sha256 == case.deck_sha256
        assert case.recovery.attempt.launch.adaptation == "seeded_driver"
        assert (
            case.recovery.attempt.launch.executed.sha256
            == case.recovery.attempt.seeded_driver.sha256
        )
        assert b"setseed" not in job.recovery.execution.startup.spinit.path.read_bytes()


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.asyncio
async def test_real_seed_domain_endpoints(work_dir):
    _write_draw(work_dir)
    state = _state(work_dir)
    seeds = [1, 2147483646, 2147483646]
    jobs = [await _submit_seeded(state, f"endpoint-{i}", seed) for i, seed in enumerate(seeds)]
    assert all(job.status == "completed" for job in jobs), [job.failures for job in jobs]
    values = [_voltage(job.cases[0]) for job in jobs]
    assert all(math.isfinite(value) for value in values)
    assert values[1] == values[2]
    assert values[0] != values[1]
    print("Seed endpoints and voltages:", json.dumps(list(zip(seeds, values, strict=True))))


def _operating_point_values(case):
    header, samples = Path(case.raw_file).read_bytes().split(b"Binary:\n", 1)
    assert b"Plotname: Operating Point\n" in header
    assert b"No. Points: 1" in header
    names = [line.split()[1] for line in header.decode().split("Variables:\n", 1)[1].splitlines()]
    values = struct.unpack("<" + "d" * len(names), samples)
    return dict(zip(names, values, strict=True))


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.asyncio
async def test_real_repeat_for_each_admitted_random_function(work_dir):
    expressions = {
        "agauss": "agauss(10,1,1)",
        "gauss": "gauss(10,.1,1)",
        "aunif": "aunif(10,1)",
        "unif": "unif(10,.1)",
        "limit": "limit(10,1)",
    }
    lines = ["* Static random functions"]
    for name, expression in expressions.items():
        lines.extend(
            [
                f".param d_{name}={expression}",
                f"V_{name} n_{name} 0 {{d_{name}}}",
                f"R_{name} n_{name} 0 1000",
            ]
        )
    (work_dir / "draw.cir").write_bytes(("\n".join([*lines, ".op", ".end", ""])).encode())
    state = _state(work_dir)
    # In LTspice/PSPICE compatibility modes, limit(x,y,z) shadows ngspice's
    # two-argument statistical limit. Keep this probe in native SPICE syntax.
    state.available_simulators["ngspice"].set_compatibility_mode("hsa")
    jobs = [
        await _submit_seeded(state, f"functions-{i}", seed) for i, seed in enumerate([17, 17, 19])
    ]
    assert all(job.status == "completed" for job in jobs), [job.failures for job in jobs]
    samples = [_operating_point_values(job.cases[0]) for job in jobs]
    readings = [{name: sample[f"v(n_{name})"] for name in expressions} for sample in samples]
    assert readings[0] == readings[1]
    assert all(readings[0][name] != readings[2][name] for name in expressions)
    print("All admitted random functions:", json.dumps(readings, sort_keys=True))


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize("analysis", [".op\n.ac dec 3 1 10", ".noise V(n) V1 dec 3 1 10"])
@pytest.mark.asyncio
async def test_multiple_plot_inputs_refuse_before_claim(work_dir, monkeypatch, analysis):
    (work_dir / "draw.cir").write_bytes(
        ("* Plot admission\nV1 n 0 DC 1 AC 1\nR1 n 0 1000\n" + analysis + "\n.end\n").encode()
    )
    state = _state(work_dir)

    def refuse_spawn(*args, **kwargs):
        pytest.fail("Unsupported plot inventory reached simulator spawn")

    monkeypatch.setattr("subprocess.run", refuse_spawn)
    args = _request(17, wait_s=0)
    args.circuits[0].path = str(work_dir / "draw.cir")
    result = await asyncio.wait_for(handle_run_experiments(args, state), 15)
    assert result.is_error
    assert not state.all_jobs
    from ltspice_mcp.lib.recovery_journal import load_journal
    from ltspice_mcp.lib.store import Store

    assert await asyncio.to_thread(load_journal, Store(work_dir), args.request_id) is None


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.asyncio
async def test_seeded_retry_keeps_successes_and_reuses_recorded_seed(work_dir, monkeypatch):
    _write_draw(work_dir)
    state = _state(work_dir)
    original = ExperimentRunner.submit_netlist
    failed = False

    def fail_once(self, netlist, run_filename, callback, **kwargs):
        nonlocal failed
        if not failed and Path(run_filename).stem.endswith("_case_1"):
            failed = True
            self.loop.call_soon_threadsafe(
                callback,
                RunOutcome("", "", 0, "Simulator failed", failure_code="simulation_failed"),
            )
            return object()
        return original(self, netlist, run_filename, callback, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", fail_once)
    parent = await _submit_seeded(
        state,
        "seeded-parent",
        17,
        variations=[{"kind": "assign", "assign": {"R1": ["1000", "2000"]}}],
    )
    assert parent.status == "completed_with_failures", parent.failures
    successful = parent.cases[0]
    assert successful.raw_file is not None
    retained = await asyncio.to_thread(Path(successful.raw_file).read_bytes)
    failed_recovery = parent.cases[1].recovery
    assert failed_recovery is not None
    electrical = failed_recovery.inputs
    prior_driver = failed_recovery.attempt.seeded_driver
    assert prior_driver is not None
    receipt = await resume_experiment(
        state, job_id=parent.job_id, resume_request_id="seeded-retry", retry_failed=True
    )
    child = receipt.job
    task = child.task
    assert task is not None
    await asyncio.wait_for(task, 15)
    await state.job_registry.drain_pending()
    assert child.status == "completed", child.failures
    recovery = child.recovery
    assert recovery is not None and recovery.execution.simulator_seed == 17
    retained_case = child.cases[0]
    retained_recovery = retained_case.recovery
    assert retained_recovery is not None and retained_recovery.attempt.reused
    assert retained_case.raw_file is not None
    assert await asyncio.to_thread(retained_case.raw_file.read_bytes) == retained
    retried_recovery = child.cases[1].recovery
    assert retried_recovery is not None and retried_recovery.inputs == electrical
    retried_driver = retried_recovery.attempt.seeded_driver
    assert retried_driver is not None and retried_driver.path != prior_driver.path
    assert _voltage(successful) == _voltage(child.cases[1])


@pytest.mark.parametrize("change", ["bytes", "rehash", "seed", "executed", "outputs"])
@pytest.mark.asyncio
async def test_seeded_driver_changes_refuse_before_spawn(committed, monkeypatch, change):
    runner, request, job, store = committed
    case = job.cases[0]
    execution = replace(job.recovery.execution, simulator_seed=17)
    job.recovery = replace(job.recovery, execution=execution)
    prepare_seeded_driver(case, execution, store, job.job_id, runner.simulator_class)
    driver = case.recovery.attempt.seeded_driver
    save_journal(store, new_root_journal(job))
    experiment_store.save_job(job)
    calls = _process(monkeypatch)
    if change in {"bytes", "rehash"}:
        driver.path.write_bytes(driver.path.read_bytes().replace(b"setseed 17", b"setseed 19"))
        if change == "rehash":
            case.recovery = replace(
                case.recovery,
                attempt=replace(
                    case.recovery.attempt,
                    seeded_driver=ArtifactDigest(driver.path, sha256_file(driver.path)),
                ),
            )
    elif change == "seed":
        job.recovery = replace(job.recovery, execution=replace(execution, simulator_seed=19))
    elif change == "outputs":
        (job.output_folder / (case.run_token + ".raw")).write_bytes(b"retained raw")
    else:
        original = shutil.copy

        def changed_copy(source, destination, **kwargs):
            copied = original(source, destination, **kwargs)
            Path(copied).write_bytes(Path(copied).read_bytes() + b"* unexpected\n")
            return copied

        monkeypatch.setattr(shutil, "copy", changed_copy)
    await _start((runner, request, job, store))
    assert not calls
    assert case.status == "failed"
    if change == "outputs":
        assert (job.output_folder / (case.run_token + ".raw")).read_bytes() == b"retained raw"
