"""A state written ahead of the live job reaches disk when a newer write
overtakes it.

A recoverable job writes some states as a copy of the live job before the live
job takes them: its terminal status, and a case's produced outputs. The writes
of a job run side by side, and only the newest one asked for reaches disk, so
a write of the live job asked for after the copy's, landing first, refuses
the copy. These tests hold the copy's write on an event, let a write of the
live job overtake it through the real registry, and read the record.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.job_types import TERMINAL_STATUSES
from tests.conftest import (
    CONTROLLED_STARTUP_REASON,
    CONTROLLED_STARTUP_REFUSED,
    LIVENESS_S,
    coordinator_returned,
)
from tests.test_recovery_launch import _process
from tests.test_recovery_launch_pool import _begin, _recoverable_job, _runner

pytestmark = pytest.mark.skipif(CONTROLLED_STARTUP_REFUSED, reason=CONTROLLED_STARTUP_REASON)


class _HeldCopy:
    """Holds the first write of a copy of the live job that ``matches``, and
    lets a write of the live job overtake it before it goes on."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, live: ExperimentJob, matches) -> None:
        self.loop = asyncio.get_running_loop()
        self.held = asyncio.Event()
        self.release = threading.Event()
        self._taken = threading.Lock()
        self._done = False
        real = experiment_store.save_job

        def save(job: ExperimentJob, **kwargs: Any) -> Path:
            hold = False
            if job is not live and matches(job):
                with self._taken:
                    hold, self._done = not self._done, True
            if hold:
                self.loop.call_soon_threadsafe(self.held.set)
                self.release.wait(LIVENESS_S)
            return real(job, **kwargs)

        monkeypatch.setattr(experiment_store, "save_job", save)

    async def overtake(self, state, live: ExperimentJob) -> None:
        """Once the copy is held, land a write of the live job, then let the
        copy's write go on."""
        await asyncio.wait_for(self.held.wait(), LIVENESS_S)
        registry = state.job_registry
        before = set(registry.pending_writes(live.job_id))
        registry.persist_job(live)
        newer = set(registry.pending_writes(live.job_id)) - before
        await asyncio.wait_for(asyncio.gather(*newer), LIVENESS_S)
        self.release.set()


def _on_disk(state, job: ExperimentJob) -> ExperimentJob:
    persisted = experiment_store.load_job(job.job_id, state.working_dir)
    assert persisted is not None
    return persisted


async def test_a_terminal_snapshot_overtaken_by_a_live_write_still_reaches_disk(
    state_no_sim, monkeypatch
):
    """The job's terminal snapshot is overtaken by a write of the live job,
    which is still running: when the coordinator's wait returns, the record is
    terminal, and it stays so once the job's writes have settled."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0001", request_id="request-order-1", cases=1
    )
    held = _HeldCopy(monkeypatch, job, lambda copy: copy.status in TERMINAL_STATUSES)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await held.overtake(state_no_sim, job)
        await coordinator_returned(state_no_sim, job)
        assert job.status == "completed"
        assert _on_disk(state_no_sim, job).status == "completed"
    finally:
        held.release.set()
    await runner.settled(job)
    await state_no_sim.job_registry.drain_pending()
    assert _on_disk(state_no_sim, job).status == "completed"


async def test_a_produced_checkpoint_overtaken_by_a_live_write_is_on_disk_before_it_is_taken(
    state_no_sim, monkeypatch
):
    """A case's produced checkpoint is overtaken by a write of the live job,
    whose case is not yet produced: by the time the live case takes the
    produced state, the record on disk holds it with its outputs."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0002", request_id="request-order-2", cases=1
    )
    held = _HeldCopy(monkeypatch, job, lambda copy: copy.cases[0].status == "produced")
    seen_on_disk: list[tuple[str, bool]] = []
    mark = ExperimentRunner._mark_case

    def observed(self, execution, case, status, **kwargs) -> None:
        if status == "produced":
            persisted = _on_disk(state_no_sim, job).cases[0]
            outputs = persisted.recovery is not None and persisted.recovery.attempt.outputs
            seen_on_disk.append((persisted.status, bool(outputs)))
        mark(self, execution, case, status, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "_mark_case", observed)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await held.overtake(state_no_sim, job)
        await coordinator_returned(state_no_sim, job)
    finally:
        held.release.set()
    await runner.settled(job)
    await state_no_sim.job_registry.drain_pending()
    assert seen_on_disk == [("produced", True)]
    persisted = _on_disk(state_no_sim, job)
    assert persisted.status == "completed"
    assert persisted.cases[0].status == "produced"


class _HeldView:
    """Holds the first write of the live job asked for once ``armed``, after
    it has read the job and before it may land: a write that read the live
    job without a state and lands after the copy that holds it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, live: ExperimentJob) -> None:
        self.loop = asyncio.get_running_loop()
        self.armed = False
        self.at_landing = asyncio.Event()
        self.release = threading.Event()
        self._taken = threading.Lock()
        self._done = False
        real = experiment_store.save_job

        def save(job: ExperimentJob, **kwargs: Any) -> Path:
            hold = False
            if job is live and self.armed:
                with self._taken:
                    hold, self._done = not self._done, True
            guard = kwargs.get("commit_guard")
            if hold and guard is not None:

                @contextlib.contextmanager
                def held_guard():
                    self.loop.call_soon_threadsafe(self.at_landing.set)
                    self.release.wait(LIVENESS_S)
                    with guard() as allowed:
                        yield allowed

                kwargs["commit_guard"] = held_guard
            return real(job, **kwargs)

        monkeypatch.setattr(experiment_store, "save_job", save)


async def test_a_terminal_snapshot_that_lands_before_an_older_view_stays_on_disk(
    state_no_sim, monkeypatch
):
    """The other order: a write of the live job reads it before the terminal
    status is taken, the terminal snapshot lands, and that older view lands
    after it. Once the job's writes have settled the record is terminal."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0003", request_id="request-order-3", cases=1
    )
    view = _HeldView(monkeypatch, job)
    held = _HeldCopy(monkeypatch, job, lambda copy: copy.status in TERMINAL_STATUSES)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        view.armed = True
        state_no_sim.job_registry.persist_job(job)
        await asyncio.wait_for(view.at_landing.wait(), LIVENESS_S)
        held.release.set()
        await coordinator_returned(state_no_sim, job)
        assert job.status == "completed"
    finally:
        held.release.set()
        view.release.set()
    await runner.settled(job)
    await state_no_sim.job_registry.drain_pending()
    assert _on_disk(state_no_sim, job).status == "completed"


async def test_a_produced_checkpoint_that_lands_before_an_older_view_stays_on_disk(
    state_no_sim, monkeypatch
):
    """The other order for a case's produced checkpoint: an older view of the
    live job lands after it. Before the job's next step is written, the record
    holds the produced case with its outputs."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0004", request_id="request-order-4", cases=1
    )
    registry = state_no_sim.job_registry
    view = _HeldView(monkeypatch, job)
    produced = _HeldCopy(monkeypatch, job, lambda copy: copy.cases[0].status == "produced")
    terminal = _HeldCopy(monkeypatch, job, lambda copy: copy.status in TERMINAL_STATUSES)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(produced.held.wait(), LIVENESS_S)
        view.armed = True
        before = set(registry.pending_writes(job.job_id))
        registry.persist_job(job)
        older = set(registry.pending_writes(job.job_id)) - before
        await asyncio.wait_for(view.at_landing.wait(), LIVENESS_S)
        produced.release.set()
        # The job's next step, its terminal snapshot, waits here.
        await asyncio.wait_for(terminal.held.wait(), LIVENESS_S)
        view.release.set()
        await asyncio.wait_for(asyncio.gather(*older), LIVENESS_S)
        case = _on_disk(state_no_sim, job).cases[0]
        assert case.status == "produced"
        assert case.recovery is not None and case.recovery.attempt.outputs is not None
    finally:
        produced.release.set()
        view.release.set()
        terminal.release.set()
    await coordinator_returned(state_no_sim, job)
    await runner.settled(job)
    await registry.drain_pending()
    assert _on_disk(state_no_sim, job).status == "completed"


async def test_a_cancel_from_another_process_while_the_terminal_snapshot_lands(
    state_no_sim, monkeypatch
):
    """A cancel from another process arrives while the terminal snapshot is
    in flight: the owner's watcher writes the live job, as it read it before
    the terminal status was taken, and that write lands after the snapshot.
    The record is terminal once the writes settle, and holds the watcher's
    note."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0005", request_id="request-order-5", cases=1
    )
    view = _HeldView(monkeypatch, job)
    held = _HeldCopy(monkeypatch, job, lambda copy: copy.status in TERMINAL_STATUSES)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        view.armed = True
        cancelled = await asyncio.wait_for(
            asyncio.to_thread(
                experiment_store.request_cancellation,
                job.job_id,
                state_no_sim.working_dir,
                job.control_token,
            ),
            LIVENESS_S,
        )
        assert cancelled is not None
        await asyncio.wait_for(view.at_landing.wait(), LIVENESS_S)
        held.release.set()
        # The snapshot has landed once the live job is terminal; the watcher's
        # older view lands after it. The coordinator waits for its watcher, so
        # the view is let go before the coordinator can return.
        await asyncio.wait_for(state_no_sim.job_registry.live[job.job_id].wait(), LIVENESS_S)
        view.release.set()
        await coordinator_returned(state_no_sim, job)
    finally:
        held.release.set()
        view.release.set()
    await runner.settled(job)
    await state_no_sim.job_registry.drain_pending()
    persisted = _on_disk(state_no_sim, job)
    assert persisted.status == job.status == "completed"
    assert any(
        item.get("code") == "external_cancellation_requested" for item in persisted.observations
    )


async def test_a_coordinator_cancelled_while_its_terminal_snapshot_lands_still_takes_it(
    state_no_sim, monkeypatch
):
    """The coordinator is cancelled while its terminal snapshot is in flight.
    The snapshot lands, so the live job takes the terminal status: memory and
    the record agree."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0006", request_id="request-order-6", cases=1
    )
    held = _HeldCopy(monkeypatch, job, lambda copy: copy.status in TERMINAL_STATUSES)
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        coordinator = state_no_sim.job_registry.live[job.job_id].task
        assert coordinator is not None
        coordinator.cancel()
        held.release.set()
        await asyncio.wait({coordinator}, timeout=LIVENESS_S)
        assert coordinator.done()
    finally:
        held.release.set()
    await state_no_sim.job_registry.drain_pending()
    assert job.status == "completed"
    assert _on_disk(state_no_sim, job).status == "completed"


async def test_a_terminal_write_that_fails_is_reported_and_published_as_failed(
    state_no_sim, monkeypatch
):
    """Every write of the terminal status fails. Nothing terminal is taken
    ahead of its write: the job ends failed, its error names the failed
    checkpoint, and the record keeps the last state that was written."""
    state_no_sim.job_registry.persist_enabled = True
    _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_order_0007", request_id="request-order-7", cases=1
    )
    real = experiment_store.save_job

    def refuse_terminal(written: ExperimentJob, **kwargs: Any) -> Path:
        if written is not job and written.status in TERMINAL_STATUSES:
            raise OSError("the disk refused the terminal checkpoint")
        return real(written, **kwargs)

    monkeypatch.setattr(experiment_store, "save_job", refuse_terminal)
    runner = _runner(state_no_sim, 1)
    await _begin(runner, request, job)
    await coordinator_returned(state_no_sim, job)
    await runner.settled(job)
    await state_no_sim.job_registry.drain_pending()
    assert job.status == "failed"
    assert job.error is not None
    assert "the disk refused the terminal checkpoint" in job.error
    assert "failure checkpoint" in job.error
    assert _on_disk(state_no_sim, job).status not in TERMINAL_STATUSES
