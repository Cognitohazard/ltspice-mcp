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

from ltspice_mcp.lib import progress
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.server import call_tool, list_tools
from ltspice_mcp.state import SessionState
from tests.conftest import (
    LIVENESS_S,
    await_until,
    make_experiment_job,
)

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

    # The SDK's callback protocol names the first argument ``progress``.
    async def on_progress(progress: float, total: float | None, message: str | None) -> None:
        reports.append((progress, total, message))

    async with Client(_serving(state_no_sim), mode=mode) as client:
        waiting = asyncio.create_task(
            client.call_tool(
                "jobs",
                {"action": "wait", "job_id": job.job_id, "timeout_s": LIVENESS_S},
                progress_callback=on_progress,
            )
        )
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


async def test_a_report_that_fails_does_not_fail_the_wait():
    async def gone(_done: float, _total: float | None, _message: str | None) -> None:
        raise ConnectionError("the client went away")

    async def work() -> str:
        return "answered"

    with progress.reporting_to(gone):
        assert await progress.watch(work(), lambda: (0, 1, "0 of 1")) == "answered"
