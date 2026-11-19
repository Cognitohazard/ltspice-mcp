"""An observer cannot overwrite an attempt adopted by another session."""

import asyncio
import json
import os
import threading
from dataclasses import replace

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import admit_initial
from ltspice_mcp.lib.experiment_types import Completeness, ExperimentJob
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.recovery_journal import new_root_journal, save_journal
from ltspice_mcp.lib.recovery_records import ProcessIdentity
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from tests.conftest import LIVENESS_S, coordinator_returned
from tests.test_recovery_launch import RecordedNGspice, _process
from tests.test_recovery_launch import committed as committed


@pytest.mark.parametrize("checkpoint", ["prepared", "intent", "completed"])
async def test_delayed_reader_cannot_erase_adopted_attempt(
    committed, monkeypatch, checkpoint, caplog
):
    _fixture_runner, request, root, store = committed
    assert root.recovery is not None
    root.owner_pid = 999_999_999
    root.recovery = replace(root.recovery, owner=ProcessIdentity(root.owner_pid, "dead-owner"))
    save_journal(store, new_root_journal(root))
    experiment_store.save_job(root)
    request = replace(request, simulator_executable=root.simulator_executable)
    state = request.state
    runner = state.runners.get_experiment_runner(
        asyncio.get_running_loop(), RecordedNGspice, store.runs_root(), 1
    )
    observer = JobRegistry(persist_enabled=True, working_dir=store.working_dir)
    reader_entered, reader_release = threading.Event(), threading.Event()
    launch_entered, launch_release = threading.Event(), threading.Event()
    persist = observer._persist_sync

    def delayed_read_write(job):
        reader_entered.set()
        assert reader_release.wait(LIVENESS_S)
        persist(job)

    def launch_checkpoint(argv, kwargs):
        launch_entered.set()
        if checkpoint == "intent":
            assert launch_release.wait(LIVENESS_S)

    monkeypatch.setattr(observer, "_persist_sync", delayed_read_write)
    calls = _process(monkeypatch, inspect=launch_checkpoint)
    job = None
    try:
        stale = await observer.get_or_load_async(root.job_id)
        assert stale is not None and stale.restart_reconciled
        assert await asyncio.to_thread(reader_entered.wait, LIVENESS_S)
        admitted = await admit_initial(runner, request)
        assert admitted.start
        job = admitted.job
        assert job.recovery is not None
        if checkpoint != "prepared":
            ready = asyncio.get_running_loop().create_future()
            await runner.start_committed(request, admitted, ready)
            assert await asyncio.to_thread(launch_entered.wait, LIVENESS_S)
            if checkpoint == "completed":
                await coordinator_returned(state, job)
        before = job.store_path.read_bytes()
        reader_release.set()
        await observer.drain_pending()
        assert "Failed to persist job" not in caplog.text
        assert job.store_path.read_bytes() == before
        saved = experiment_store.load_job(job.job_id, store.working_dir)
        assert saved is not None and saved.owner_pid == os.getpid()
        assert saved.recovery is not None and saved.recovery.owner == job.recovery.owner
        case_recovery = saved.cases[0].recovery
        assert case_recovery is not None
        if checkpoint == "prepared":
            peer = SessionState.create(state.config, available={})
            peer.job_registry.persist_enabled = True
            peer_runner = peer.runners.get_experiment_runner(
                asyncio.get_running_loop(), RecordedNGspice, store.runs_root(), 1
            )
            replay = await admit_initial(peer_runner, replace(request, state=peer))
            assert replay.replayed and not replay.start
            ready = asyncio.get_running_loop().create_future()
            await runner.start_committed(request, admitted, ready)
        elif checkpoint == "intent":
            assert case_recovery.attempt.launch is not None
            assert case_recovery.attempt.outputs is None
        else:
            assert saved.cases[0].status == "produced"
            assert case_recovery.attempt.outputs is not None
        launch_release.set()
        await coordinator_returned(state, job)
        assert len(calls) == 1
    finally:
        reader_release.set()
        launch_release.set()
        await observer.drain_pending()
        live = state.job_registry.live.get(job.job_id) if job is not None else None
        if live is not None and live.task is not None:
            await asyncio.wait_for(asyncio.shield(live.task), LIVENESS_S)


async def test_dead_owner_reconciliation_is_saved(committed, caplog):
    _runner, _request, root, store = committed
    assert root.recovery is not None
    root.owner_pid = 999_999_999
    root.recovery = replace(root.recovery, owner=ProcessIdentity(root.owner_pid, "dead-owner"))
    save_journal(store, new_root_journal(root))
    experiment_store.save_job(root)
    observer = JobRegistry(persist_enabled=True, working_dir=store.working_dir)

    loaded = await observer.get_or_load_async(root.job_id)
    await observer.drain_pending()

    assert loaded is not None and loaded.restart_reconciled
    assert json.loads(root.store_path.read_text(encoding="utf-8"))["status"] == "interrupted"
    assert "Failed to persist job" not in caplog.text


@pytest.mark.parametrize("route", ["get_or_load", "refresh_foreign_job"])
async def test_a_read_finishing_after_the_job_is_reserved_answers_with_the_live_job(
    work_dir, monkeypatch, route
):
    """A store read that started before this process reserved the job, and
    returns after it, hands back the live job and leaves it registered: the
    copy it read is a record nothing here updates. Both routes that adopt a
    read off the loop are run."""
    # A dead owner's record, which the read reconciles, or a live peer's,
    # which a refresh re-reads.
    job = ExperimentJob(
        job_id="exp_reserved_0001",
        request_id="request-1",
        fingerprint="f" * 64,
        canonicalizer_version=1,
        control_token="token",
        store_path=Store(work_dir).job_record("exp_reserved_0001"),
        cases=[],
        sources=[],
        simulator="ngspice",
        completeness=Completeness(declared=0, expanded=0),
        status="queued",
        owner_pid=999_999_999 if route == "get_or_load" else os.getppid(),
    )
    experiment_store.save_job(job)
    registry = JobRegistry(persist_enabled=True, working_dir=work_dir)
    entered, release = threading.Event(), threading.Event()
    if route == "get_or_load":
        load = registry._load_from_store
        monkeypatch.setattr(
            registry,
            "_load_from_store",
            lambda job_id: (entered.set(), release.wait(LIVENESS_S), load(job_id))[2],
        )
        read = registry.get_or_load_async(job.job_id)
    else:
        foreign = experiment_store.load_job(job.job_id, work_dir)
        assert foreign is not None
        load_foreign = registry._load_foreign_job_sync
        monkeypatch.setattr(
            registry,
            "_load_foreign_job_sync",
            lambda held: (entered.set(), release.wait(LIVENESS_S), load_foreign(held))[2],
        )
        read = registry.refresh_foreign_job_async(foreign)

    async def reserve_while_reading() -> None:
        assert await asyncio.to_thread(entered.wait, LIVENESS_S)
        registry.add_experiment_job(job, already_persisted=True)
        registry.reserve(job)
        release.set()

    try:
        answered, _ = await asyncio.gather(read, reserve_while_reading())
    finally:
        release.set()
    assert answered is job
    assert registry.jobs[job.job_id] is job
    await registry.drain_pending()
