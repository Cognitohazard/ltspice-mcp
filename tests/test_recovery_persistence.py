"""A recovery launch must observe a completed durable write or its failure."""

import asyncio
import threading
from dataclasses import replace

import pytest

from ltspice_mcp import lib
from ltspice_mcp.lib import experiment_store
from tests.conftest import LIVENESS_S
from tests.test_experiment_job import _job


async def test_strict_checkpoint_surfaces_write_failure(
    state_no_sim, work_dir, sample_netlist, monkeypatch
):
    job = _job(work_dir, sample_netlist)
    state_no_sim.job_registry.persist_enabled = True

    def unavailable(job):
        raise OSError("storage unavailable")

    monkeypatch.setattr(experiment_store, "save_job", unavailable)
    with pytest.raises(OSError, match="storage unavailable"):
        await state_no_sim.job_registry.persist_strict(job)


async def test_strict_checkpoint_orders_after_pending_write(
    state_no_sim, work_dir, sample_netlist, monkeypatch
):
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    job = _job(work_dir, sample_netlist)
    entered = threading.Event()
    release = threading.Event()
    writes = []
    save = experiment_store.save_job

    def delayed(job):
        if not entered.is_set():
            entered.set()
            assert release.wait(LIVENESS_S)
        save(job)
        writes.append(job.job_id)

    monkeypatch.setattr(experiment_store, "save_job", delayed)
    registry.persist_job(job)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        strict = asyncio.create_task(registry.persist_strict(job))
        await asyncio.sleep(0)
        assert not strict.done()
        release.set()
        await strict
        await registry.drain_pending()
        assert writes == [job.job_id, job.job_id]
        assert experiment_store.load_job(job.job_id, work_dir, own_is_alive=True) is not None
    finally:
        release.set()
        await registry.drain_pending()


async def test_strict_checkpoint_preserves_order_before_queued_writer_starts(
    state_no_sim,
    work_dir,
    sample_netlist,
):
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    older = _job(work_dir, sample_netlist)
    newer = replace(older, status="failed", error="New durable state")
    registry.persist_job(older)
    # No yield: the previous writer is queued, but has not taken its lock yet.
    await registry.persist_strict(newer)
    await registry.drain_pending()
    saved = experiment_store.load_job(older.job_id, work_dir, own_is_alive=True)
    assert saved is not None
    assert saved.status == "failed" and saved.error == "New durable state"


async def test_strict_storage_runtime_error_is_not_retried_as_executor_teardown(
    state_no_sim,
    work_dir,
    sample_netlist,
    monkeypatch,
):
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    job = _job(work_dir, sample_netlist)
    calls = []
    save = experiment_store.save_job

    def fails_once(job):
        calls.append(job.job_id)
        if len(calls) == 1:
            raise RuntimeError("Record validation failed")
        save(job)

    monkeypatch.setattr(experiment_store, "save_job", fails_once)
    with pytest.raises(RuntimeError, match="Record validation failed"):
        await registry.persist_strict(job)
    assert calls == [job.job_id]
    assert not job.store_path.exists()


@pytest.mark.parametrize("cancel_target", ["caller", "drain"])
async def test_repeated_cancellation_keeps_write_order_until_worker_finishes(
    state_no_sim, work_dir, sample_netlist, monkeypatch, cancel_target
):
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    older = _job(work_dir, sample_netlist)
    newer = replace(older, status="failed", error="New durable state")
    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    save = experiment_store.save_job

    def delayed(job):
        if job is older:
            entered.set()
            assert release.wait(LIVENESS_S)
        save(job)
        if job is older:
            finished.set()

    monkeypatch.setattr(experiment_store, "save_job", delayed)
    first = asyncio.create_task(registry.persist_strict(older))
    tasks = [first]
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        target = first
        if cancel_target == "drain":
            target = asyncio.create_task(registry.drain_pending())
            tasks.append(target)
            await asyncio.sleep(0)
        for _ in range(2):
            target.cancel()
            await asyncio.sleep(0)
        second = asyncio.create_task(registry.persist_strict(newer))
        tasks.append(second)
        # Let the later write finish if cancellation prematurely freed its lock.
        await asyncio.wait(
            {second}, timeout=0.1
        )  # timing: a negative window: the later write must not finish early
        completed_before_release = second.done()
        release.set()
        await second
        assert await asyncio.to_thread(finished.wait, 5)
        with pytest.raises(asyncio.CancelledError):
            await first
        saved = experiment_store.load_job(older.job_id, work_dir, own_is_alive=True)
        assert saved is not None
        assert saved.status == "failed" and saved.error == "New durable state"
        assert not completed_before_release
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        await registry.drain_pending()


async def test_strict_atomic_replace_failure_survives_repeated_cancellation(
    state_no_sim, work_dir, sample_netlist, monkeypatch
):
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    original = _job(work_dir, sample_netlist)
    experiment_store.save_job(original)
    newer = replace(original, status="failed", error="New durable state")
    entered = threading.Event()
    release = threading.Event()
    replace_file = lib.replace_file

    def unavailable(src, dst):
        if dst == newer.store_path:
            # The real writer has created and flushed its temporary record.
            assert src.is_file()
            entered.set()
            assert release.wait(LIVENESS_S)
            raise OSError("atomic replacement unavailable")
        replace_file(src, dst)

    monkeypatch.setattr(lib, "replace_file", unavailable)
    strict = asyncio.create_task(registry.persist_strict(newer))
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        for _ in range(2):
            strict.cancel()
            await asyncio.sleep(0)
        release.set()
        with pytest.raises(OSError, match="atomic replacement unavailable"):
            await strict
        saved = experiment_store.load_job(original.job_id, work_dir, own_is_alive=True)
        assert saved is not None and saved.status == "queued"
    finally:
        release.set()
        await asyncio.gather(strict, return_exceptions=True)
        await registry.drain_pending()
