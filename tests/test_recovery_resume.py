"""Resume selection, input integrity and competing admissions through real tools."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store
from ltspice_mcp.lib.experiment_resume import resume_experiment
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.job_lifecycle import WaitFor
from ltspice_mcp.lib.recovery_journal import load_journal
from ltspice_mcp.lib.recovery_records import RecoveryError
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from tests.conftest import LIVENESS_S, coordinator_returned, job_done, recorded_fixture_simulator
from tests.test_resume_surface import _DECK, _fail_second_case, _state, _submit

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH"),
]


@pytest.fixture
async def failed_campaign(work_dir, monkeypatch):
    (work_dir / "divider.cir").write_text(_DECK, encoding="utf-8")
    state = _state(work_dir)
    submissions = _fail_second_case(monkeypatch)
    result = await _submit(state, variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}])
    assert result["status"] == "completed_with_failures"
    parent = state.all_jobs[result["job_id"]]
    await coordinator_returned(state, parent)
    yield state, parent, submissions
    for live in list(state.job_registry.live.values()):
        if live.task is not None:
            await asyncio.wait_for(asyncio.shield(live.task), LIVENESS_S)
    await state.job_registry.drain_pending()


@pytest.mark.parametrize("selection", ["unknown", "produced"])
async def test_invalid_selection_refuses_without_advancing_head(failed_campaign, selection):
    state, parent, submitted = failed_campaign
    selected = "missing" if selection == "unknown" else parent.cases[0].case_id
    with pytest.raises(RecoveryError) as error:
        await resume_experiment(
            state,
            job_id=parent.job_id,
            resume_request_id="invalid-selection",
            case_ids=[selected],
            retry_failed=True,
        )
    assert error.value.code == "recovery_case_selection"
    journal = load_journal(Store(state.working_dir), parent.request_id)
    assert journal is not None and journal.head_job_id == parent.job_id and not journal.resumes
    assert len(submitted) == 2


@pytest.mark.parametrize("target", ["electrical", "produced", "startup"])
async def test_changed_frozen_files_refuse_before_child_commit(failed_campaign, target):
    state, parent, submitted = failed_campaign
    if target == "electrical":
        path = parent.cases[1].staged_deck
    elif target == "produced":
        path = parent.cases[0].raw_file
    else:
        path = parent.recovery.execution.startup.spinit.path
    assert path is not None
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(RecoveryError):
        await resume_experiment(
            state, job_id=parent.job_id, resume_request_id="drift", retry_failed=True
        )
    journal = load_journal(Store(state.working_dir), parent.request_id)
    assert journal is not None and not journal.resumes
    assert len(submitted) == 2


async def test_competing_resume_ids_admit_exactly_one_child(failed_campaign):
    state, parent, submitted = failed_campaign
    replies = await asyncio.gather(
        *(
            resume_experiment(
                state, job_id=parent.job_id, resume_request_id=request_id, retry_failed=True
            )
            for request_id in ("first-retry", "second-retry")
        ),
        return_exceptions=True,
    )
    successes = [reply for reply in replies if not isinstance(reply, BaseException)]
    failures = [reply for reply in replies if isinstance(reply, RecoveryError)]
    assert len(successes) == len(failures) == 1
    assert failures[0].code == "recovery_stale_parent"
    child = successes[0].job
    assert await job_done(state, child)
    assert len(submitted) == 3 and len(set(submitted)) == 3
    journal = load_journal(Store(state.working_dir), parent.request_id)
    assert journal is not None and journal.head_job_id == child.job_id
    assert len(journal.resumes) == 1


async def test_resume_id_does_not_overwrite_global_submission_index(failed_campaign):
    state, parent, _ = failed_campaign
    experiment_store.save_request_index(
        request_id="shared-name",
        fingerprint="c" * 64,
        canonicalizer_version=1,
        job_id="exp_unrelated",
        working_dir=state.working_dir,
    )
    path = Store(state.working_dir).request_index("shared-name")
    before = path.read_bytes()
    reply = await resume_experiment(
        state, job_id=parent.job_id, resume_request_id="shared-name", retry_failed=True
    )
    assert await job_done(state, reply.job)
    assert path.read_bytes() == before
    replay = await resume_experiment(
        state, job_id=parent.job_id, resume_request_id="shared-name", retry_failed=True
    )
    assert replay.replayed and replay.job.job_id == reply.job.job_id


async def test_disabled_persistence_cannot_admit_resume(failed_campaign):
    state, parent, submitted = failed_campaign
    state.job_registry.persist_enabled = False
    with pytest.raises(RecoveryError) as error:
        await resume_experiment(
            state, job_id=parent.job_id, resume_request_id="no-storage", retry_failed=True
        )
    assert error.value.code == "recovery_persistence_required"
    assert len(submitted) == 2


async def test_committed_noop_replays_without_available_simulator(failed_campaign):
    state, parent, submitted = failed_campaign
    initial = await resume_experiment(
        state, job_id=parent.job_id, resume_request_id="nothing-eligible"
    )
    assert not initial.resumed and not initial.replayed
    state.available_simulators.clear()

    replay = await resume_experiment(
        state, job_id=parent.job_id, resume_request_id="nothing-eligible"
    )

    assert replay.replayed and not replay.resumed
    assert replay.job.job_id == parent.job_id and replay.control_token is None
    journal = load_journal(Store(state.working_dir), parent.request_id)
    assert journal is not None and journal.head_job_id == parent.job_id
    assert len(submitted) == 2
    from ltspice_mcp.lib.experiment_runner import IdempotencyConflictError

    with pytest.raises(IdempotencyConflictError):
        await resume_experiment(
            state, job_id=parent.job_id, resume_request_id="nothing-eligible", retry_failed=True
        )


@pytest.mark.parametrize("stop", ["job_deadline", "cancelled", "run_timeout"])
async def test_stopped_attempt_selection_uses_recorded_reason(work_dir, monkeypatch, stop):
    """Exercise coordinator stop transitions before selecting a linked retry."""
    (work_dir / "divider.cir").write_text(_DECK, encoding="utf-8")
    state = _state(work_dir)
    launched = asyncio.Event()
    callbacks = {}

    def held_submission(self, netlist, run_filename, callback, **kwargs):
        callbacks[Path(run_filename).stem] = callback
        self.loop.call_soon_threadsafe(launched.set)
        return object()

    async def confirmed_kill(self, token):
        callbacks.pop(token)(RunOutcome("", "", 0, "Stopped by test simulator"))

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", held_submission)
    monkeypatch.setattr(ExperimentRunner, "_kill_case", confirmed_kill)
    limits = {"recoverable": True, "wait_s": 0}
    if stop == "job_deadline":
        limits["job_deadline_s"] = 2.0
    elif stop == "run_timeout":
        limits["run_timeout_s"] = 2.0
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "stopped-root",
                "circuits": [{"path": "divider.cir"}],
                "execution": limits,
                "lint": "off",
            }
        ),
        state,
    )
    assert not result.is_error, result.structured_content
    data = result.structured_content
    assert data is not None
    parent = state.all_jobs[data["job_id"]]
    try:
        await asyncio.wait_for(launched.wait(), LIVENESS_S)
        if stop == "cancelled":
            runner = state.runners.get_experiment_runner_for(parent)
            assert runner is not None
            await runner.cancel(parent, control_token=parent.control_token)
        await coordinator_returned(state, parent)
        assert parent.cases[0].failure_code == stop
        assert not callbacks
        recorded_fixture_simulator(monkeypatch)
        default = await resume_experiment(
            state, job_id=parent.job_id, resume_request_id="default-retry"
        )
        assert default.resumed is (stop == "job_deadline")
        if default.resumed:
            child = default.job
        else:
            assert default.job.job_id == parent.job_id
            explicit = await resume_experiment(
                state,
                job_id=parent.job_id,
                resume_request_id="explicit-retry",
                retry_cancelled=stop == "cancelled",
                retry_failed=stop == "run_timeout",
            )
            assert explicit.resumed
            child = explicit.job
        assert child.job_id != parent.job_id
        await coordinator_returned(state, child)
        assert child.cases[0].status == "produced", child.cases[0].error
    finally:
        await state.job_registry.cancel_running(state.runners, state)
        await state.job_registry.drain_pending()


@pytest.mark.parametrize("family_available", [False, True])
async def test_resume_selects_recorded_named_executable(
    failed_campaign, work_dir, family_available
):
    from ltspice_mcp.lib.simulator import bind_named_executable
    from ltspice_mcp.tools.jobs import JobsInput, handle_jobs

    state, parent, submitted = failed_campaign
    original = state.available_simulators["ngspice"]
    program = Path(parent.recovery.execution.executable.path)
    state.named_simulators["ngspice:pinned"] = bind_named_executable(
        original, "ngspice:pinned", program
    )
    if family_available:
        other_program = work_dir / "other-ngspice"
        other_program.write_bytes(b"different executable")
        state.available_simulators["ngspice"] = bind_named_executable(
            original, "ngspice:other", other_program
        )
    else:
        state.available_simulators.clear()

    reply = await handle_jobs(
        JobsInput.model_validate(
            {
                "action": "resume",
                "job_id": parent.job_id,
                "resume_request_id": "named-retry",
                "retry_failed": True,
            }
        ),
        state,
    )
    data = reply.structured_content
    assert not reply.is_error, data
    assert data is not None
    child = state.all_jobs[data["job_id"]]
    assert await job_done(state, child)
    assert child.status == "completed"
    assert child.recovery.execution == parent.recovery.execution
    assert len(submitted) == 3


async def test_resume_caps_dwell_and_reports_it(failed_campaign, monkeypatch):
    from ltspice_mcp.tools import jobs

    state, parent, submitted = failed_campaign
    wait = state.job_registry.wait
    timeouts = []

    async def observed_wait(job, timeout_s, *, wait_for: WaitFor = "all"):
        timeouts.append(timeout_s)
        return await wait(job, timeout_s, wait_for=wait_for)

    monkeypatch.setattr(state.job_registry, "wait", observed_wait)
    response = await jobs.handle_jobs(
        jobs.JobsInput.model_validate(
            {
                "action": "resume",
                "job_id": parent.job_id,
                "resume_request_id": "capped-wait",
                "retry_failed": True,
                "wait_s": 999,
            }
        ),
        state,
    )
    data = response.structured_content
    assert data is not None and not response.is_error
    assert timeouts == [120]
    assert data["resumed"] and data["status"] == "completed"
    assert any("wait_s" in warning and "120" in warning for warning in data["warnings"])
    assert len(submitted) == 3
