"""Progress notifications: a wait that holds a call open tells a client that
asked for progress how far the job has got.

The end-to-end tests connect the SDK's own client to the server's own tool
dispatch in-process, over both of the client's connections (the JSON-RPC
streams and the direct dispatcher), with a server lifespan that hands it this
test's session state, so the test can finish a job's cases one at a time.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from mcp import Client
from mcp.server.lowlevel import Server

from ltspice_mcp.lib import experiment_store, progress, store
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.server import call_tool, list_tools
from ltspice_mcp.state import SessionState
from tests.conftest import (
    LIVENESS_S,
    await_until,
    make_experiment_job,
)
from tests.test_jobs_tool import _FOREIGN_PID, _circuit, _experiment

pytestmark = pytest.mark.asyncio


def _serving(state: SessionState) -> Server[dict]:
    """The server's tool dispatch over ``state``."""

    @asynccontextmanager
    async def lifespan(_server: Server[dict]) -> AsyncIterator[dict]:
        yield {"state": state}

    return Server(
        "progress-test", lifespan=lifespan, on_call_tool=call_tool, on_list_tools=list_tools
    )


def _finish(job: ExperimentJob, *indexes: int) -> None:
    """Finish these cases, as the runner records a finished case."""
    for index in indexes:
        job.cases[index].status = "produced"
    job.completeness.recount(job.cases, execution_job_id=job.job_id)


async def _waited(
    client: Client, job_id: str, reports: list[tuple[float, float | None, str | None]]
) -> asyncio.Task:
    """A jobs(wait) on ``job_id`` that asked for progress, recorded in ``reports``."""

    # The SDK's callback protocol names the first argument ``progress``.
    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total, message))

    return asyncio.create_task(
        client.call_tool(
            "jobs",
            {"action": "wait", "job_id": job_id, "timeout_s": LIVENESS_S},
            progress_callback=on_progress,
        )
    )


@pytest.mark.parametrize("mode", ["legacy", "auto"])
async def test_a_job_wait_reports_each_finished_case_to_a_client_that_asked(
    state_no_sim: SessionState, monkeypatch: pytest.MonkeyPatch, mode: str
):
    """Before, a jobs(wait) held the call open in silence however long the job
    ran, so a client could show nothing and could not restart its timeout."""
    # timing: how often the wait reads the job, not a bound on anything; the
    # test waits for each report itself
    monkeypatch.setattr(progress, "POLL_S", 0.02)
    job = make_experiment_job(state_no_sim, job_id="exp-progress", count=3, status="running")
    live = state_no_sim.job_registry.reserve(job)
    reports: list[tuple[float, float | None, str | None]] = []
    async with Client(_serving(state_no_sim), mode=mode) as client:
        waiting = await _waited(client, job.job_id, reports)
        await await_until(lambda: len(reports) == 1, what="the first progress report")
        _finish(job, 0)
        await await_until(lambda: len(reports) == 2, what="the report of one case finished")
        _finish(job, 1, 2)
        live.transition("completed")
        result = await waiting

    assert [(done, total) for done, total, _ in reports] == [(0, 3), (1, 3), (3, 3)]
    assert reports[-1][2] == "Job exp-progress: 3 of 3 cases finished"
    assert result.structured_content is not None
    assert result.structured_content["status"] == "completed"


async def test_a_wait_on_another_processs_job_reports_what_its_record_says(
    state_no_sim: SessionState, work_dir, monkeypatch: pytest.MonkeyPatch
):
    """Another process's job is followed by re-reading its record, the second
    path a wait answers by; each count read that has grown is reported."""
    # timing: how often the wait reads the job, not a bound on anything
    monkeypatch.setattr(progress, "POLL_S", 0.02)
    # timing: how often the record is re-read, not a bound on anything
    monkeypatch.setattr(experiment_store, "FOREIGN_RECORD_POLL_S", 0.01)
    # An owner presumed dead would reconcile the record to interrupted.
    monkeypatch.setattr(
        experiment_store, "owner_liveness", lambda *_args, **_kwargs: store.OwnerLiveness.ALIVE
    )
    circuit = _circuit(work_dir)

    def record(*, finished: int, status: str) -> None:
        job = _experiment(work_dir, circuit, count=2, status=status)
        for case in job.cases[:finished]:
            case.status = "produced"
        job.completeness.recount(job.cases)
        job.owner_pid = _FOREIGN_PID
        experiment_store.save_job(job)

    record(finished=0, status="running")
    reader = SessionState.create(state_no_sim.config, available={})
    stale = experiment_store.load_job("exp_jobs_0001", work_dir)
    assert stale is not None
    reader.all_jobs[stale.job_id] = stale
    reports: list[tuple[float, float | None, str | None]] = []
    async with Client(_serving(reader)) as client:
        waiting = await _waited(client, stale.job_id, reports)
        await await_until(lambda: len(reports) == 1, what="the first progress report")
        await asyncio.to_thread(record, finished=1, status="running")
        await await_until(lambda: len(reports) == 2, what="the report of one case finished")
        await asyncio.to_thread(record, finished=2, status="completed")
        result = await waiting

    assert [(done, total) for done, total, _ in reports] == [(0, 2), (1, 2), (2, 2)]
    assert result.structured_content is not None
    assert result.structured_content["status"] == "completed"


async def test_a_task_started_during_the_call_does_not_report():
    """A task the call starts inherits its binding, and can outlive it, as a
    job's coordinator does: only the call's own task reports, so nothing is
    sent for a call already answered."""
    reports: list[float] = []

    async def record(done: float, _total: float | None, _message: str | None) -> None:
        reports.append(done)

    async def work() -> str:
        return "done"

    with progress.reporting_to(record):
        started = asyncio.create_task(progress.watch(work(), lambda: (0, 1, "0 of 1")))
        assert await started == "done"
        assert await progress.watch(work(), lambda: (0, 1, "0 of 1")) == "done"
    assert reports == [0]


async def test_a_cancelled_wait_stops_the_work_it_watched():
    started = asyncio.Event()
    stopped = asyncio.Event()

    async def held() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    async def record(_done: float, _total: float | None, _message: str | None) -> None:
        return None

    with progress.reporting_to(record):
        waiting = asyncio.create_task(progress.watch(held(), lambda: (0, 1, "0 of 1")))
        await started.wait()
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
    assert stopped.is_set()


async def test_a_report_that_fails_does_not_fail_the_wait():
    async def gone(_done: float, _total: float | None, _message: str | None) -> None:
        raise ConnectionError("the client went away")

    async def work() -> str:
        return "answered"

    with progress.reporting_to(gone):
        assert await progress.watch(work(), lambda: (0, 1, "0 of 1")) == "answered"
