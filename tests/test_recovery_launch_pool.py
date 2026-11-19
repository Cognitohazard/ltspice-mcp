"""The launch record of a recoverable case never waits on the pool it needs.

Each case of a recoverable job records its launch before the simulator starts,
and the thread that launches the case waits for that write. The write runs on
the event loop's default thread pool. When the launching threads come from that
same pool, launches that fill it leave no thread for any write, and every one
of them waits for ever. These tests run the real coordinator and spicelib's
launch thread with the suite's stand-in simulator, on a default pool sized so
the launches fill it, and line the launches up with a barrier so they all wait
at the same moment.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import prepare_startup
from ltspice_mcp.lib.experiment_runner import (
    AdmissionResult,
    ExperimentRunner,
    ExperimentRunRequest,
)
from ltspice_mcp.lib.experiment_types import Completeness, ExperimentJob
from ltspice_mcp.lib.pdk_native import ArtifactDigest
from ltspice_mcp.lib.proc_kill import process_start_marker
from ltspice_mcp.lib.recovery_journal import new_root_journal, save_journal
from ltspice_mcp.lib.recovery_records import (
    CaseAttempt,
    CaseRecovery,
    ExecutionRecord,
    FrozenInputs,
    JobRecovery,
    ProcessIdentity,
    StartupPolicy,
)
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from tests.conftest import (
    CONTROLLED_STARTUP_REASON,
    CONTROLLED_STARTUP_REFUSED,
    LIVENESS_S,
    coordinator_returned,
    staged_decks,
)
from tests.test_experiment_job import _case, _job
from tests.test_recovery_launch import RecordedNGspice, _process

pytestmark = pytest.mark.skipif(CONTROLLED_STARTUP_REFUSED, reason=CONTROLLED_STARTUP_REASON)


@pytest.fixture
async def default_pool():
    """Give the running loop a default pool of a chosen size.

    At the end, work still queued on it is cancelled: a launch that waits for
    a write queued there then gets its answer and ends, so a failing test
    fails instead of leaving the loop's shutdown joining a thread for ever.
    """
    loop = asyncio.get_running_loop()
    pools: list[ThreadPoolExecutor] = []

    def make(workers: int) -> ThreadPoolExecutor:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="small-default")
        loop.set_default_executor(pool)
        pools.append(pool)
        return pool

    yield make
    for pool in pools:
        pool.shutdown(wait=False, cancel_futures=True)


def _recoverable_job(
    state: SessionState, *, job_id: str, request_id: str, cases: int
) -> tuple[ExperimentJob, ExperimentRunRequest]:
    """A committed recoverable ngspice job of ``cases`` cases that may all run
    at once, with its root journal and record written, as admission leaves it."""
    store = Store(state.working_dir)
    root = store.run_dir(job_id)
    root.mkdir(parents=True)
    deck = root / "case.cir"
    deck.write_bytes(b"* frozen\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    digest = sha256_file(deck)
    job = _job(state.working_dir, deck, job_id=job_id, request_id=request_id)
    job.output_folder = root
    job.simulator = "ngspice"
    job.owner_pid = os.getpid()
    identity = executable_identity(RecordedNGspice)
    assert identity is not None
    job.simulator_executable = identity
    inputs = FrozenInputs(root, ArtifactDigest(deck, digest), (ArtifactDigest(deck, digest),))
    job.cases = []
    for index in range(cases):
        case = _case(deck, index=index)
        case.deck_sha256 = digest
        case.run_token = f"{job_id}_case_{index}"
        case.recovery = CaseRecovery(inputs, CaseAttempt(job_id, 0, case.run_token))
        job.cases.append(case)
    job.completeness = Completeness(declared=cases, expanded=cases)
    start_marker = process_start_marker(os.getpid())
    assert start_marker is not None
    execution = ExecutionRecord(
        LIVENESS_S,
        "server_default",
        cases,
        2 * LIVENESS_S,
        LIVENESS_S,
        (sys.executable,),
        identity,
        "hsa",
        sys.platform,
        StartupPolicy("recorded-test", True),
    )
    execution = replace(execution, startup=prepare_startup(store, job_id, RecordedNGspice))
    job.recovery = JobRecovery(
        job_id,
        request_id,
        None,
        0,
        ProcessIdentity(os.getpid(), start_marker),
        execution,
    )
    save_journal(store, new_root_journal(job))
    experiment_store.save_job(job)
    request = ExperimentRunRequest(
        state=state,
        request_id=request_id,
        fingerprint=job.fingerprint,
        stage=staged_decks(job.cases, job.sources),
        simulator="ngspice",
        recoverable=True,
    )
    return job, request


def _runner(state: SessionState, max_parallel: int) -> ExperimentRunner:
    return ExperimentRunner(
        asyncio.get_running_loop(),
        RecordedNGspice,
        Store(state.working_dir).runs_root(),
        max_parallel,
    )


async def _begin(runner: ExperimentRunner, request: ExperimentRunRequest, job: ExperimentJob):
    receipt = asyncio.get_running_loop().create_future()
    await runner.start_committed(request, AdmissionResult(job, False, True), receipt)


def _launches_meet(monkeypatch, parties: int) -> None:
    """Hold each launch at its start until ``parties`` launches have started.

    A launch holds its thread from its start until it returns. Launches of one
    job then take turns under the job's cancellation lock, which the one
    recording its launch holds across the write it waits for, so once they
    have all started each one is either waiting for that write or waiting for
    the lock: all of their threads are taken at once.
    """
    barrier = threading.Barrier(parties)
    launch = ExperimentRunner._submit_case_under_cancel_gate

    def meet(self, execution, case, suffix: str) -> bool:
        barrier.wait(LIVENESS_S)
        return launch(self, execution, case, suffix)

    monkeypatch.setattr(ExperimentRunner, "_submit_case_under_cancel_gate", meet)


@pytest.mark.parametrize("workers", [2, 1])
async def test_launch_with_small_default_pool(state_no_sim, default_pool, monkeypatch, workers):
    """One recoverable case on a pool of one thread: the launch held that
    thread while its record waited for a thread of the same pool. Two threads
    is the control, which finishes with or without the fix."""
    default_pool(workers)
    state_no_sim.job_registry.persist_enabled = True
    calls = _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_pool_0001", request_id="request-pool-1", cases=1
    )
    await _begin(_runner(state_no_sim, 1), request, job)
    await coordinator_returned(state_no_sim, job)
    assert len(calls) == 1
    assert job.cases[0].status == "produced", job.cases[0].error
    assert job.status == "completed"


async def test_launches_as_many_as_the_default_pool_finish(
    state_no_sim, default_pool, monkeypatch
):
    """A job whose parallel launches equal the pool's size: every thread of
    the pool would be a launch waiting for a write or for the job's lock,
    which a launch gives up on after the lock's timeout, failing its case."""
    default_pool(3)
    state_no_sim.job_registry.persist_enabled = True
    calls = _process(monkeypatch)
    _launches_meet(monkeypatch, 3)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_pool_0002", request_id="request-pool-2", cases=3
    )
    await _begin(_runner(state_no_sim, 3), request, job)
    await coordinator_returned(state_no_sim, job)
    assert len(calls) == 3
    assert [case.status for case in job.cases] == ["produced"] * 3, [
        case.error for case in job.cases
    ]
    persisted = experiment_store.load_job(job.job_id, state_no_sim.working_dir)
    assert persisted is not None
    assert all(
        case.recovery is not None and case.recovery.attempt.launch is not None
        for case in persisted.cases
    )


async def test_launches_of_two_runners_past_the_pool_finish(
    state_no_sim, default_pool, monkeypatch
):
    """Two runners (two simulators, or two builds of one) each within their
    own limit, whose launches together are more than the pool holds."""
    default_pool(2)
    state_no_sim.job_registry.persist_enabled = True
    calls = _process(monkeypatch)
    _launches_meet(monkeypatch, 2)
    first, first_request = _recoverable_job(
        state_no_sim, job_id="exp_pool_0003", request_id="request-pool-3", cases=2
    )
    second, second_request = _recoverable_job(
        state_no_sim, job_id="exp_pool_0004", request_id="request-pool-4", cases=2
    )
    await _begin(_runner(state_no_sim, 2), first_request, first)
    await _begin(_runner(state_no_sim, 2), second_request, second)
    await coordinator_returned(state_no_sim, first)
    await coordinator_returned(state_no_sim, second)
    assert len(calls) == 4
    for job in (first, second):
        assert [case.status for case in job.cases] == ["produced"] * 2, [
            case.error for case in job.cases
        ]
        assert job.status == "completed"


async def test_cancel_ends_a_launch_that_waits_for_its_record(
    state_no_sim, default_pool, monkeypatch
):
    """Every thread of the pool is busy, so the launch's record cannot be
    written; a cancel of the job ends the launch's wait, the simulator is
    never started, and the case says it was cancelled."""
    loop = asyncio.get_running_loop()
    default_pool(2)
    registry = state_no_sim.job_registry
    registry.persist_enabled = True
    calls = _process(monkeypatch)
    job, request = _recoverable_job(
        state_no_sim, job_id="exp_pool_0005", request_id="request-pool-5", cases=1
    )
    runner = _runner(state_no_sim, 1)

    entered = asyncio.Event()
    go = threading.Event()
    record = ExperimentRunner._record_launch_intent

    def held(self, execution, case, copied: Path) -> None:
        loop.call_soon_threadsafe(entered.set)
        go.wait(LIVENESS_S)
        record(self, execution, case, copied)

    monkeypatch.setattr(ExperimentRunner, "_record_launch_intent", held)
    waiting = asyncio.Event()
    persist = registry.persist_strict

    async def watched(written: ExperimentJob) -> None:
        if any(
            case.recovery is not None and case.recovery.attempt.launch is not None
            for case in written.cases
        ):
            waiting.set()
        await persist(written)

    monkeypatch.setattr(registry, "persist_strict", watched)

    release = threading.Event()
    busy: list[asyncio.Future[bool]] = []
    try:
        await _begin(runner, request, job)
        await asyncio.wait_for(entered.wait(), LIVENESS_S)
        # Every thread of the pool is taken before the launch asks for its
        # write, so the write queues behind them.
        busy.extend(loop.run_in_executor(None, release.wait, LIVENESS_S) for _ in range(2))
        go.set()
        await asyncio.wait_for(waiting.wait(), LIVENESS_S)
        execution = runner._executions[job.job_id]
        case_task = execution.case_tasks[job.cases[0].case_id]
        cancelling = asyncio.ensure_future(runner.cancel(job, control_token=job.control_token))
        await asyncio.wait_for(asyncio.shield(case_task), LIVENESS_S)
        assert calls == []
        assert job.cases[0].status == "cancelled", job.cases[0].error
        assert job.cases[0].failure_code == "cancelled"
        assert job.cases[0].error == (
            "Cancelled while its launch was being recorded; its simulator was not started"
        )
    finally:
        release.set()
        go.set()
        await asyncio.gather(*busy)
    await asyncio.wait_for(cancelling, LIVENESS_S)
    await coordinator_returned(state_no_sim, job)
    assert calls == []
    assert job.status == "cancelled"
    assert job.cases[0].status == "cancelled"
    persisted = experiment_store.load_job(job.job_id, state_no_sim.working_dir)
    assert persisted is not None
    assert persisted.status == "cancelled"
    assert persisted.cases[0].status == "cancelled"
