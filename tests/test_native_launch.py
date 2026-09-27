"""Native statistical setup uses the ordinary simulator submission primitive."""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from pathlib import Path

import pytest
from spicelib.sim.sim_runner import SimRunner
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib.experiment_runner import ExperimentRunner, _Execution
from ltspice_mcp.lib.native_records import NativeCaseRecord
from ltspice_mcp.lib.pdk_native import PROFILE, NativePaths, NativeRequest, PreparedLaunch
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead
from ltspice_mcp.lib.runner_base import NativeLaunchContext, RunnerBase
from ltspice_mcp.lib.store import Store
from tests.test_experiment_job import _job
from tests.test_experiment_runner import _request


@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice is not on PATH")
async def test_setup_sources_and_writes_in_its_own_directory(tmp_path):
    from ltspice_mcp.config import ServerConfig
    from ltspice_mcp.lib.simulator import detect_simulators

    detect_simulators(ServerConfig(working_dir=tmp_path, allowed_paths=[tmp_path]))
    output = tmp_path / "shared runs"
    folder = output / "job"
    folder.mkdir(parents=True)
    (folder / ".spiceinit").write_text("echo STARTUP_WAS_READ\nquit 19\n", encoding="utf-8")
    source = folder / "sample.input.cir"
    source.write_text("* input\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n", encoding="utf-8")
    driver = folder / "sample.setup.cir"
    driver.write_text(
        "* setup\n.control\nset ngbehavior=hsa\nsetseed 123\nsource sample.input.cir\n"
        "run\nwrite sample.raw\nquit\n.endc\n.end\n",
        encoding="utf-8",
    )
    loop = asyncio.get_running_loop()
    result = loop.create_future()
    runner = RunnerBase(loop, NGspiceSimulator, output)
    checks = []

    def verify_copy():
        assert (folder / "sample.cir").read_bytes() == driver.read_bytes()
        assert not (folder / "sample.raw").exists()
        checks.append(True)

    handle = await asyncio.to_thread(
        runner.submit_netlist,
        driver,
        str(Path("job") / "sample.cir"),
        result.set_result,
        native=NativeLaunchContext(input_deck=source, cwd=folder, verify_execution=verify_copy),
    )
    outcome = await asyncio.wait_for(result, 20)
    assert not outcome.error, outcome
    assert checks == [True]
    assert Path(outcome.raw_file) == folder / "sample.raw"
    raw = OffsetAwareRawRead(Path(outcome.raw_file), dialect="ngspice")
    assert float(raw.get_wave("v(in)")[0]) == pytest.approx(1)
    log_text = await asyncio.to_thread(Path(outcome.log_file).read_text, encoding="utf-8")
    assert "STARTUP_WAS_READ" not in log_text
    assert driver.is_file() and source.is_file()
    assert handle.output_folder == output
    assert handle.cwd == folder
    for task in handle.active_tasks:
        await asyncio.to_thread(task.join, 5)
    assert all(not task.is_alive() for task in handle.active_tasks)


async def test_copied_native_setup_rejection_has_no_submission_stamp(
    state_no_sim, work_dir, monkeypatch
):
    circuit = work_dir / "bench.cir"
    circuit.write_text(".op\n.end\n", encoding="utf-8")
    job = _job(work_dir, circuit)
    case = job.cases[0]
    case.run_token = "token"
    store = Store(work_dir)
    folder = store.run_dir(job.job_id)
    folder.mkdir(parents=True)
    paths = NativePaths(
        folder,
        folder / "token.input.cir",
        folder / "token.setup.cir",
        folder / "token.cir",
        folder / "token.raw",
        folder / "token.log",
    )
    electrical = b".op\n.end\n"
    driver = b"* prepared setup\n.end\n"
    await asyncio.to_thread(Path(paths.electrical_input).write_bytes, electrical)
    await asyncio.to_thread(Path(paths.prepared_driver).write_bytes, driver)
    prepared = PreparedLaunch(
        paths,
        hashlib.sha256(electrical).hexdigest(),
        hashlib.sha256(driver).hexdigest(),
        (),
        "sample",
        1,
        ".op",
    )
    case.native_statistics = NativeCaseRecord(
        NativeRequest("bench", "native", PROFILE, "nominal", 1, 0), prepared=prepared
    )
    original_prepare = SimRunner._prepare_sim
    copied = []

    def corrupt_executed_copy(self, netlist, run_filename):
        path = original_prepare(self, netlist, run_filename)
        copied.append(path)
        path.write_bytes(b"changed after copy")
        return path

    def forbid_task(*args, **kwargs):
        pytest.fail("a refused setup must not construct a simulator task")

    monkeypatch.setattr(SimRunner, "_prepare_sim", corrupt_executed_copy)
    monkeypatch.setattr("spicelib.sim.sim_runner.RunTask", forbid_task)
    runner = ExperimentRunner(asyncio.get_running_loop(), NGspiceSimulator, store.runs_root())
    request = _request(state_no_sim, work_dir, request_id="refused-native")
    execution = _Execution(request, job, asyncio.Semaphore(1), 1)

    await runner._run_case(execution, case)

    assert copied == [paths.executed_driver]
    assert case.status == "failed"
    assert case.failure_code == "submission_failed"
    assert case.error is not None and "bytes" in case.error
    assert case.submitted_at is None
    assert job.completeness.submitted == 0
    assert not await asyncio.to_thread(Path(paths.raw).exists)
