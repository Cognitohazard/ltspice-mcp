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
