"""Why a case stopped, as each stop writes it, and what a resume does with it.

A case stopped because the process running its job went away is retried by a
default resume, however that process went: killed (found by the next reader of
the record) or shut down (the server's, or the Python API's ``close``). Only an
explicit cancel needs ``retry_cancelled``.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import _eligible, resume_experiment
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.experiment_types import ExperimentCase
from ltspice_mcp.lib.job_registry import JobRegistry
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import OwnerLiveness
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import (
    LIVENESS_S,
    coordinator_returned,
    needs_controlled_startup,
    recorded_fixture_simulator,
)
from tests.test_recovery_records import recovery_job

#: The public code of a case whose owning process went away
#: (``experiment_types.OWNER_STOPPED``), which a default resume retries.
_OWNER_STOPPED = "server_restarted"


def _default_retry(case: ExperimentCase) -> bool:
    return _eligible(case, retry_failed=False, retry_cancelled=False)


def _running_recoverable(work_dir: Path):
    job = recovery_job(work_dir)
    job.status = "running"
    job.completed_at = None
    job.cases[0].status = "running"
    job.completeness.recount(job.cases)
    return job


def test_a_crash_leaves_a_case_a_default_resume_retries(work_dir: Path):
    """The owner was killed mid-case; the next reader of its record finds the
    process gone and reconciles."""
    job = _running_recoverable(work_dir)
    experiment_store.save_job(job)
    record = json.loads(job.store_path.read_text(encoding="utf-8"))

    read = experiment_store.deserialize_job(record, job.store_path, liveness=OwnerLiveness.DEAD)

    [case] = read.cases
    assert (case.status, case.failure_code) == ("failed", _OWNER_STOPPED)
    assert _default_retry(case)


@pytest.mark.asyncio
async def test_a_shutdown_with_no_coordinator_leaves_a_case_a_default_resume_retries(
    work_dir: Path,
):
    """Shutdown settles a job it runs but holds no coordinator for itself, as a
    crash would leave it: interrupted, its case retried by a default resume."""
    job = _running_recoverable(work_dir)
    registry = JobRegistry(persist_enabled=False, working_dir=work_dir)
    registry.reserve(job)
    registry.add_experiment_job(job, already_persisted=True)

    await registry.cancel_running(
        SimpleNamespace(get_experiment_runner_for=lambda _job: None), None
    )

    [case] = job.cases
    assert job.status == "interrupted"
    assert (case.status, case.failure_code) == ("failed", _OWNER_STOPPED)
    assert job.failures[-1]["code"] == _OWNER_STOPPED
    assert _default_retry(case)


_DECK = "* divider\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"


@needs_controlled_startup
@pytest.mark.asyncio
@pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
@pytest.mark.parametrize(
    ("stop", "code", "retried_by_default"),
    [
        ("shutdown", _OWNER_STOPPED, True),
        ("job_deadline", "job_deadline", True),
        ("cancel", "cancelled", False),
    ],
)
async def test_a_coordinator_stop_records_its_reason_and_resume_follows_it(
    stop: str,
    code: str,
    retried_by_default: bool,
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Each stop the coordinator itself makes, on a recoverable job whose case
    is running: the server shutting down, the job deadline, an explicit cancel."""
    from tests.test_resume_surface import _state

    (work_dir / "divider.cir").write_text(_DECK, encoding="utf-8")
    state = _state(work_dir)
    callbacks = {}
    launched = asyncio.Event()

    def held_submission(self, netlist, run_filename, callback, **kwargs):
        callbacks[Path(run_filename).stem] = callback
        self.loop.call_soon_threadsafe(launched.set)
        return object()

    async def confirmed_kill(self, token):
        callbacks.pop(token)(RunOutcome("", "", 0, "Stopped by test simulator"))

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", held_submission)
    monkeypatch.setattr(ExperimentRunner, "_kill_case", confirmed_kill)
    execution: dict = {"recoverable": True, "wait_s": 0}
    if stop == "job_deadline":
        execution["job_deadline_s"] = 2.0
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": f"stopped-by-{stop}",
                "circuits": [{"path": "divider.cir"}],
                "execution": execution,
                "lint": "off",
            }
        ),
        state,
    )
    assert not result.is_error, result.structured_content
    assert result.structured_content is not None
    parent = state.all_jobs[result.structured_content["job_id"]]
    try:
        await asyncio.wait_for(launched.wait(), LIVENESS_S)
        if stop == "shutdown":
            await state.job_registry.cancel_running(state.runners, state)
        elif stop == "cancel":
            runner = state.runners.get_experiment_runner_for(parent)
            assert runner is not None
            await runner.cancel(parent, control_token=parent.control_token)
        await coordinator_returned(state, parent)

        [case] = parent.cases
        assert case.failure_code == code
        assert _default_retry(case) is retried_by_default
        recorded_fixture_simulator(monkeypatch)
        default = await resume_experiment(
            state, job_id=parent.job_id, resume_request_id="default-retry"
        )
        assert default.resumed is retried_by_default
        if default.resumed:
            await coordinator_returned(state, default.job)
            assert default.job.cases[0].status == "produced"
    finally:
        await state.job_registry.cancel_running(state.runners, state)
        await state.job_registry.drain_pending()
