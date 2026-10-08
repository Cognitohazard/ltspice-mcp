"""Durable launch barriers through the coordinator and real spicelib threads."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import ClassVar

import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import prepare_startup
from ltspice_mcp.lib.experiment_runner import (
    AdmissionResult,
    ExperimentRunner,
    ExperimentRunRequest,
    SubmissionCommitted,
)
from ltspice_mcp.lib.job_lifecycle import LiveJob
from ltspice_mcp.lib.proc_kill import process_start_marker
from ltspice_mcp.lib.recovery_journal import (
    add_child,
    load_journal,
    mark_launched,
    new_root_journal,
    save_journal,
)
from ltspice_mcp.lib.recovery_records import CaseAttempt, ProcessIdentity, RecoveryError
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import OwnerLiveness, Store
from tests.conftest import LIVENESS_S, coordinator_returned, ngspice_binary_raw, staged_decks
from tests.test_recovery_records import recovery_job


class RecordedNGspice(NGspiceSimulator):
    spice_exe: ClassVar[list[str]] = [sys.executable]
    _compatibility_mode = "hsa"


@pytest.fixture
async def committed(state_no_sim, work_dir):
    if sys.platform not in {"linux", "win32"}:
        pytest.skip("Controlled ngspice startup is verified on Linux and Windows only")
    store = Store(work_dir)
    job = recovery_job(work_dir)
    assert job.recovery is not None
    job.status = "queued"
    job.simulator = "ngspice"
    job.owner_pid = os.getpid()
    identity = executable_identity(RecordedNGspice)
    assert identity is not None
    job.simulator_executable = identity
    execution = replace(
        job.recovery.execution,
        simulator_argv=(sys.executable,),
        executable=identity,
        startup=prepare_startup(store, job.job_id, RecordedNGspice),
        platform=sys.platform,
        max_parallel=1,
        run_timeout_s=LIVENESS_S,
        timeout_source="server_default",
        job_deadline_s=2 * LIVENESS_S,
        kill_grace_s=LIVENESS_S,
    )
    start_marker = process_start_marker(os.getpid())
    assert start_marker is not None
    job.recovery = replace(
        job.recovery,
        owner=ProcessIdentity(os.getpid(), start_marker),
        execution=execution,
    )
    save_journal(store, new_root_journal(job))
    experiment_store.save_job(job)
    state_no_sim.job_registry.persist_enabled = True
    runner = ExperimentRunner(asyncio.get_running_loop(), RecordedNGspice, store.runs_root(), 1)
    request = ExperimentRunRequest(
        state=state_no_sim,
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        stage=staged_decks(job.cases, job.sources),
        simulator="ngspice",
        recoverable=True,
        # Unlike the recorded bounds, so a test can tell which were applied.
        kill_grace_s=2 * LIVENESS_S,
        job_deadline_s=100.0,
    )
    return runner, request, job, store


def _process(monkeypatch, *, inspect=None):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if inspect is not None:
            inspect(argv, kwargs)
        raw = Path(argv[argv.index("-r") + 1])
        log = Path(argv[argv.index("-o") + 1])
        raw.write_bytes(
            ngspice_binary_raw([(1.0,)], ["v(in)"], plot="Operating Point", declared=1)
        )
        log.write_text("No. of Data Rows : 1\n", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run)
    return calls


async def _start(committed):
    runner, request, job, _store = committed
    assert runner.loop is asyncio.get_running_loop(), (
        "Fixture runner belongs to a different event loop"
    )
    receipt = asyncio.get_running_loop().create_future()
    # Admission has already committed the real journal/job. This is exactly
    # the coordinator entry used by initial admission and resumed children.
    await runner.start_committed(request, AdmissionResult(job, False, True), receipt)
    await coordinator_returned(request.state, job)
    await request.state.job_registry.drain_pending()
    assert runner._slots_claimed == 0
    return job


async def test_recorded_deadline_and_kill_grace_override_replay_defaults(committed):
    runner, request, job, _store = committed
    execution = runner._new_execution(request, LiveJob(job))
    assert execution.request.kill_grace_s == job.recovery.execution.kill_grace_s
    assert execution.request.job_deadline_s == job.recovery.execution.job_deadline_s


async def test_start_failure_keeps_committed_receipt_after_real_adoption(committed, monkeypatch):
    runner, request, job, store = committed
    job.owner_pid = 999_999_999
    job.recovery = replace(
        job.recovery, owner=ProcessIdentity(job.owner_pid, "recorded-dead-owner")
    )
    save_journal(store, new_root_journal(job))
    experiment_store.save_job(job)
    request = replace(request, simulator_executable=job.simulator_executable)
    calls = _process(monkeypatch)

    def refuse_start(_request, _job):
        raise OSError("coordinator start unavailable")

    monkeypatch.setattr(runner, "_new_execution", refuse_start)
    with pytest.raises(SubmissionCommitted, match="coordinator start unavailable") as error:
        await runner.submit(request)
    receipt = error.value.receipt
    assert receipt is not None
    assert receipt.job.job_id == job.job_id
    assert receipt.replayed is True
    assert receipt.control_token == job.control_token
    assert receipt.job.owner_pid == os.getpid()
    assert receipt.job.recovery is not None
    assert receipt.job.recovery.owner.start_marker == process_start_marker(os.getpid())
    assert request.state.all_jobs[job.job_id] is receipt.job
    live = request.state.job_registry.live.get(job.job_id)
    assert live is None or live.task is None
    journal = load_journal(store, job.request_id)
    assert journal is not None
    assert journal.root.candidate is not None
    assert journal.root.phase == "prepared"
    assert journal.root.candidate["recovery"]["owner"] == receipt.job.recovery.owner.to_record()
    assert calls == []


async def test_adoption_replaces_stale_registered_object(committed, monkeypatch):
    _runner, request, job, _store = committed
    stale = experiment_store.deserialize_job(
        experiment_store.serialize_job(job), job.store_path, liveness=OwnerLiveness.ALIVE
    )
    stale.status = "interrupted"
    stale.owner_pid = 999_999_999
    request.state.add_experiment_job(stale, already_persisted=True)
    _process(monkeypatch)
    result = await _start(committed)
    assert result is job
    assert request.state.all_jobs[job.job_id] is job
    assert result.owner_pid == os.getpid()
    assert result.cases[0].status == "produced"


async def test_adoption_refuses_to_replace_active_coordinator(committed):
    runner, request, job, _store = committed
    live = request.state.job_registry.reserve(replace(job))
    task = asyncio.create_task(asyncio.Event().wait())
    live.task = task
    request.state.add_experiment_job(live.job, already_persisted=True)
    try:
        receipt = asyncio.get_running_loop().create_future()
        with pytest.raises(RecoveryError, match="live coordinator"):
            await runner.start_committed(request, AdmissionResult(job, False, True), receipt)
        assert request.state.all_jobs[job.job_id] is live.job
        assert request.state.job_registry.live[job.job_id] is live
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_spawn_observes_durable_intent_launched_journal_and_controlled_env(
    committed, monkeypatch
):
    _runner, _request, job, store = committed
    monkeypatch.setenv("SPICE_SCRIPTS", "ambient startup")
    monkeypatch.setenv("SPICE_LIB_DIR", "ambient models")
    environment_before = dict(os.environ)

    def inspect(argv, kwargs):
        journal = load_journal(store, job.request_id)
        assert journal is not None
        assert journal.root.phase == "launched" and journal.root.candidate is None
        data = json.loads(job.store_path.read_text(encoding="utf-8"))
        intent = data["cases"][0]["recovery"]["attempt"]["launch"]
        assert intent is not None
        assert intent["electrical_sha256"] == job.cases[0].deck_sha256
        executed = Path(intent["executed"]["path"])
        assert executed == Path(argv[-1])
        assert executed.parent == job.output_folder
        assert intent["executed"]["sha256"] == sha256_file(executed)
        assert "-n" in argv
        assert kwargs["env"]["SPICE_SCRIPTS"] == str(
            job.recovery.execution.startup.spinit.path.parent
        )
        assert {key for key in kwargs["env"] if key.upper().startswith("SPICE_")} == {
            "SPICE_SCRIPTS"
        }

    calls = _process(monkeypatch, inspect=inspect)
    result = await _start(committed)
    assert len(calls) == 1
    assert dict(os.environ) == environment_before
    assert result.cases[0].status == "produced", result.cases[0].error
    outputs = result.cases[0].recovery.attempt.outputs
    assert outputs.raw.sha256 == sha256_file(result.cases[0].raw_file)
    assert outputs.log.sha256 == sha256_file(result.cases[0].log_file)
    persisted = experiment_store.load_job(job.job_id, store.working_dir, own_is_alive=True)
    assert persisted is not None
    assert persisted.cases[0].recovery is not None
    assert persisted.cases[0].recovery.attempt.outputs == outputs


async def test_child_launch_uses_root_directory_and_preserves_previous_outputs(
    committed, monkeypatch
):
    runner, request, parent, store = committed
    previous_raw = parent.output_folder / f"{parent.cases[0].run_token}.raw"
    previous_log = previous_raw.with_suffix(".log")
    previous_raw.write_bytes(b"previous attempt raw")
    previous_log.write_bytes(b"previous attempt log")
    child_id = "exp_launch_child"
    child_token = child_id + "_case_0"
    child = replace(
        parent,
        job_id=child_id,
        request_id="resume-launch",
        store_path=store.job_record(child_id),
        recovery=replace(parent.recovery, parent_job_id=parent.job_id, attempt_index=1),
        cases=[
            replace(
                parent.cases[0],
                run_token=child_token,
                recovery=replace(
                    parent.cases[0].recovery, attempt=CaseAttempt(child_id, 1, child_token)
                ),
            )
        ],
    )
    journal = load_journal(store, parent.request_id)
    assert journal is not None
    journal = mark_launched(journal, parent.job_id)
    save_journal(store, add_child(journal, child.request_id, child.fingerprint, child))
    experiment_store.save_job(child)
    child_request = replace(request, request_id=child.request_id)
    calls = _process(monkeypatch)
    result = await _start((runner, child_request, child, store))
    assert len(calls) == 1
    assert result.cases[0].status == "produced", result.cases[0].error
    assert result.cases[0].raw_file.parent == parent.output_folder
    assert result.cases[0].raw_file.stem == child_token
    assert previous_raw.read_bytes() == b"previous attempt raw"
    assert previous_log.read_bytes() == b"previous attempt log"


@pytest.mark.parametrize("failure", ["running", "journal", "intent"])
async def test_durable_barrier_failure_never_spawns(committed, monkeypatch, failure):
    _runner, _request, job, store = committed
    calls = _process(monkeypatch)
    replace_file = os.replace

    def fail(source, destination):
        path = Path(destination)
        data = json.loads(Path(source).read_text(encoding="utf-8"))
        if (
            (failure == "running" and path == job.store_path and data["status"] == "running")
            or (failure == "journal" and path == store.recovery_journal(job.request_id))
            or (
                failure == "intent"
                and path == job.store_path
                and data["cases"][0]["recovery"]["attempt"]["launch"] is not None
            )
        ):
            raise OSError("checkpoint unavailable")
        replace_file(source, destination)

    monkeypatch.setattr(os, "replace", fail)
    result = await _start(committed)
    assert calls == []
    assert result.cases[0].status == "failed"
    assert result.cases[0].failure_code in {"submission_failed", "recovery_persistence_failed"}


@pytest.mark.parametrize("target", ["electrical", "startup"])
async def test_changed_frozen_input_refuses_before_process_spawn(committed, monkeypatch, target):
    _runner, _request, job, _store = committed
    calls = _process(monkeypatch)
    path = (
        job.cases[0].staged_deck
        if target == "electrical"
        else job.recovery.execution.startup.spinit.path
    )
    path.write_bytes(path.read_bytes() + b"* changed\n")
    result = await _start(committed)
    assert calls == []
    assert result.cases[0].status == "failed"


async def test_changed_executed_copy_refuses_before_process_spawn(committed, monkeypatch):
    _runner, _request, job, _store = committed
    calls = _process(monkeypatch)
    copy_file = shutil.copy

    def changed_copy(source, destination, **kwargs):
        result = copy_file(source, destination, **kwargs)
        path = Path(result)
        if path.stem == job.cases[0].run_token:
            path.write_bytes(path.read_bytes() + b"* unexpected execution change\n")
        return result

    monkeypatch.setattr(shutil, "copy", changed_copy)
    result = await _start(committed)
    assert calls == []
    assert result.cases[0].status == "failed"


async def test_startup_is_verified_again_for_each_case(committed, monkeypatch):
    _runner, _request, job, store = committed
    first = job.cases[0]
    token = job.job_id + "_case_1"
    job.cases.append(
        replace(
            first,
            case_id="case_0001",
            run_index=1,
            run_token=token,
            observations=[],
            recovery=replace(first.recovery, attempt=CaseAttempt(job.job_id, 0, token)),
        )
    )
    job.completeness = replace(job.completeness, declared=2, expanded=2)
    save_journal(store, new_root_journal(job))
    experiment_store.save_job(job)

    def change_startup(_argv, _kwargs):
        job.recovery.execution.startup.spinit.path.write_bytes(b"* drift after first launch\n")

    calls = _process(monkeypatch, inspect=change_startup)
    result = await _start(committed)
    assert len(calls) == 1
    assert [case.status for case in result.cases] == ["produced", "failed"]


async def test_output_snapshot_write_failure_does_not_publish_produced(committed, monkeypatch):
    _runner, _request, job, _store = committed
    calls = _process(monkeypatch)
    replace_file = os.replace
    live_status_at_checkpoint = []

    def fail(source, destination):
        if Path(destination) == job.store_path:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
            if data["cases"][0]["status"] == "produced":
                live_status_at_checkpoint.append(job.cases[0].status)
                raise OSError("completion checkpoint unavailable")
        replace_file(source, destination)

    monkeypatch.setattr(os, "replace", fail)
    result = await _start(committed)
    assert len(calls) == 1
    assert live_status_at_checkpoint == ["running"]
    assert result.cases[0].status == "failed"
    assert result.cases[0].raw_file.is_file()
    assert result.cases[0].log_file.is_file()
