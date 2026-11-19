"""The cancel gate of a recoverable job, against launches and slow writes.

The gate is the job's cross-process cancellation lock. Under it a launch
decides whether to start (no cancel marker written, no stop requested) and
stamps its case submitted, and a cancel from another process writes the
marker. What it promises: once a cancel request has returned, no case of the
job passes the gate, and every case that passed it before is one the job's
owner stops when it reads the marker, either before its simulator starts or by
killing it.

These tests run the real coordinator and spicelib's launch thread with the
suite's stand-in simulator, and hold a launch's record write on an event, as
slow storage would, to see what else waits for it.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from pathlib import Path
from typing import Any

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import OWNER_STOPPED, ExperimentJob
from ltspice_mcp.lib.job_lifecycle import OWNER_STOPPED_CASE_ERROR
from tests.conftest import (
    CONTROLLED_STARTUP_REASON,
    CONTROLLED_STARTUP_REFUSED,
    LIVENESS_S,
    coordinator_returned,
)
from tests.test_recovery_launch import _process
from tests.test_recovery_launch_pool import _begin, _recoverable_job, _runner

pytestmark = pytest.mark.skipif(CONTROLLED_STARTUP_REFUSED, reason=CONTROLLED_STARTUP_REASON)

_RECORDING_STOPPED = "Cancelled while its launch was being recorded; its simulator was not started"


def _launched(job: ExperimentJob) -> list[str]:
    return [
        case.case_id
        for case in job.cases
        if case.recovery is not None and case.recovery.attempt.launch is not None
    ]


class _HeldWrite:
    """Holds the first write of the job record that records a launch, as a
    slow disk would, until ``release`` is set."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.loop = asyncio.get_running_loop()
        self.held = asyncio.Event()
        self.release = threading.Event()
        self.case_id: str | None = None
        self._taken = threading.Lock()
        real = experiment_store.save_job

        def save(job: ExperimentJob, **kwargs: Any) -> Path:
            launched = _launched(job)
            hold = False
            if launched:
                with self._taken:
                    if self.case_id is None:
                        self.case_id = launched[0]
                        hold = True
            if hold:
                self.loop.call_soon_threadsafe(self.held.set)
                self.release.wait(LIVENESS_S)
            return real(job, **kwargs)

        monkeypatch.setattr(experiment_store, "save_job", save)


def _starts(monkeypatch: pytest.MonkeyPatch) -> tuple[list, asyncio.Event]:
    """The stand-in simulator, with an event set each time one starts."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()

    def inspect(_argv, _kwargs) -> None:
        loop.call_soon_threadsafe(started.set)

    return _process(monkeypatch, inspect=inspect), started


async def _cancel_from_another_process(job: ExperimentJob, working_dir: Path) -> int:
    """Request a cancel as another server process does: a process of its own
    that writes the durable marker under the job's cancellation lock."""
    script = (
        "import sys\n"
        "from pathlib import Path\n"
        "from ltspice_mcp.lib import experiment_store\n"
        "experiment_store.request_cancellation(sys.argv[1], Path(sys.argv[2]), sys.argv[3])\n"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, job.job_id, str(working_dir), job.control_token
    )
    return await asyncio.wait_for(process.wait(), LIVENESS_S)


def _case_task(runner: ExperimentRunner, job: ExperimentJob, case_id: str) -> asyncio.Task[None]:
    return runner._executions[job.job_id].case_tasks[case_id]


async def test_a_held_record_write_does_not_hold_back_another_launch(state_no_sim, monkeypatch):
    """Two cases may run at once. While the first launch's record write is
    held, the second launch records its own and its simulator starts."""
    state_no_sim.job_registry.persist_enabled = True
    calls, started = _starts(monkeypatch)
    held = _HeldWrite(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_gate_0001", request_id="request-gate-1", cases=2
    )
    try:
        await _begin(_runner(state_no_sim, 2), request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        await asyncio.wait_for(started.wait(), LIVENESS_S)
        other = next(case for case in job.cases if case.case_id != held.case_id)
        assert len(calls) == 1
        assert Path(calls[0][0][-1]).stem == other.run_token
        persisted = experiment_store.load_job(job.job_id, state_no_sim.working_dir)
        assert persisted is not None
        assert other.case_id in _launched(persisted)
    finally:
        held.release.set()
    await coordinator_returned(state_no_sim, job)
    assert [case.status for case in job.cases] == ["produced", "produced"], [
        case.error for case in job.cases
    ]
    assert job.status == "completed"
    persisted = experiment_store.load_job(job.job_id, state_no_sim.working_dir)
    assert persisted is not None
    assert persisted.status == "completed"
    assert sorted(_launched(persisted)) == sorted(case.case_id for case in job.cases)


async def test_a_cancel_from_another_process_is_not_held_behind_a_record_write(
    state_no_sim, monkeypatch
):
    """A second process cancels while a launch's record write is held: its
    request returns, the owner stops the launch before its simulator starts,
    and the job ends cancelled with its case saying so."""
    state_no_sim.job_registry.persist_enabled = True
    calls, _started = _starts(monkeypatch)
    held = _HeldWrite(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_gate_0002", request_id="request-gate-2", cases=1
    )
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        case_task = _case_task(runner, job, job.cases[0].case_id)
        assert await _cancel_from_another_process(job, state_no_sim.working_dir) == 0
        await asyncio.wait_for(asyncio.shield(case_task), LIVENESS_S)
        assert calls == []
        assert job.cases[0].status == "cancelled"
        assert job.cases[0].error == _RECORDING_STOPPED
    finally:
        held.release.set()
    await coordinator_returned(state_no_sim, job)
    assert calls == []
    assert job.status == "cancelled"
    persisted = experiment_store.load_job(job.job_id, state_no_sim.working_dir)
    assert persisted is not None
    assert persisted.status == "cancelled"
    assert persisted.cases[0].status == "cancelled"
    assert persisted.cases[0].error == _RECORDING_STOPPED


@pytest.mark.parametrize("point", ["before the gate", "after the record"])
async def test_a_cancel_at_each_point_of_a_launch_starts_no_simulator(
    state_no_sim, monkeypatch, point
):
    """A cancel from another process's file path, sent while a launch is held
    at one point, leaves no simulator started and the case stating where the
    cancel found it. (While the record is written: the test above.)"""
    loop = asyncio.get_running_loop()
    state_no_sim.job_registry.persist_enabled = True
    calls, _started = _starts(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim,
        job_id=f"exp_gate_000{3 if point == 'before the gate' else 4}",
        request_id=f"request-gate-{point.replace(' ', '-')}",
        cases=1,
    )
    runner = _runner(state_no_sim, 1)
    entered = asyncio.Event()
    go = threading.Event()
    name = (
        "_submit_case_under_cancel_gate" if point == "before the gate" else "_verify_executed_copy"
    )
    real = getattr(ExperimentRunner, name)
    taken = threading.Lock()
    first = [True]

    def held(self, *args: Any) -> Any:
        with taken:
            hold, first[0] = first[0], False
        if hold:
            loop.call_soon_threadsafe(entered.set)
            go.wait(LIVENESS_S)
        return real(self, *args)

    monkeypatch.setattr(ExperimentRunner, name, held)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(entered.wait(), LIVENESS_S)
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
        # The owner reads the marker before the launch goes on.
        await asyncio.wait_for(runner._executions[job.job_id].cancel_event.wait(), LIVENESS_S)
    finally:
        go.set()
    await coordinator_returned(state_no_sim, job)
    assert calls == []
    assert job.status == "cancelled"
    assert job.cases[0].status == "cancelled"
    assert job.cases[0].error == (
        "Cancelled before submission" if point == "before the gate" else _RECORDING_STOPPED
    )


async def test_a_deadline_while_a_launch_is_recorded_says_so(state_no_sim, monkeypatch):
    """The job's deadline passes while a launch's record write is held: the
    case fails with the deadline and says its simulator was not started.

    The deadline's clock is replaced by a handshake, so it passes only once
    the record write is held; the deadline's own code then runs as it does
    when its time is up. With a real short deadline the launch could reach
    its gate after the deadline had passed and be refused there, before any
    record, which is a different case."""
    state_no_sim.job_registry.persist_enabled = True
    calls, _started = _starts(monkeypatch)
    held = _HeldWrite(monkeypatch)
    passes = asyncio.Event()
    watch = ExperimentRunner._deadline_watch

    async def deadline_on_handshake(self, execution, _deadline_s: float) -> None:
        await passes.wait()
        await watch(self, execution, 0)

    monkeypatch.setattr(ExperimentRunner, "_deadline_watch", deadline_on_handshake)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_gate_0005", request_id="request-gate-5", cases=1
    )
    runner = _runner(state_no_sim, 1)
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        passes.set()
        case_task = _case_task(runner, job, job.cases[0].case_id)
        await asyncio.wait_for(asyncio.shield(case_task), LIVENESS_S)
        assert calls == []
        assert job.cases[0].status == "failed"
        assert job.cases[0].failure_code == "job_deadline"
        assert job.cases[0].error == (
            "The experiment deadline elapsed while this case's launch was being "
            "recorded; its simulator was not started"
        )
    finally:
        held.release.set()
    await coordinator_returned(state_no_sim, job)
    assert job.status == "completed_with_failures"


async def test_a_shutdown_while_a_launch_is_recorded_leaves_it_to_resume(
    state_no_sim, monkeypatch
):
    """The process stops (the cancel its shutdown issues) while a launch's
    record write is held: the case reads as a stopped owner's, which a resume
    retries."""
    state_no_sim.job_registry.persist_enabled = True
    calls, _started = _starts(monkeypatch)
    held = _HeldWrite(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_gate_0006", request_id="request-gate-6", cases=1
    )
    runner = _runner(state_no_sim, 1)
    stopping = None
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(held.held.wait(), LIVENESS_S)
        case_task = _case_task(runner, job, job.cases[0].case_id)
        stopping = asyncio.ensure_future(runner.cancel(job, reason=OWNER_STOPPED))
        await asyncio.wait_for(asyncio.shield(case_task), LIVENESS_S)
        assert calls == []
        assert job.cases[0].status == "failed"
        assert job.cases[0].failure_code == OWNER_STOPPED
        assert job.cases[0].error == OWNER_STOPPED_CASE_ERROR
    finally:
        held.release.set()
    assert stopping is not None
    await asyncio.wait_for(stopping, LIVENESS_S)
    await coordinator_returned(state_no_sim, job)
    assert calls == []
