"""Public recovery submissions, receipts, and detached delegation."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import cast

import jsonschema
import pytest
from pydantic import ValidationError
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.api import Api, ApiCallError, ApiValidationError
from ltspice_mcp.config import ServerConfig
from ltspice_mcp.lib.experiment_runner import ExperimentRunner
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import jobs as jobs_module
from ltspice_mcp.tools._schema import build_input_schema
from ltspice_mcp.tools.experiments import RunExperimentsInput, handle_run_experiments
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA, JobsInput, JobsResumeInput, handle_jobs
from ltspice_mcp.tools.receipts import RUN_EXPERIMENTS_OUTPUT_SCHEMA
from tests.conftest import LIVENESS_S, SyncApi, recorded_fixture_simulator


def test_recoverable_opt_in_is_strict_and_part_of_request_identity():
    payload = {"request_id": "recovery", "circuits": [{"path": "deck.cir"}]}
    ordinary = RunExperimentsInput.model_validate(payload)
    recovery = RunExperimentsInput.model_validate({**payload, "execution": {"recoverable": True}})
    assert ordinary.execution.recoverable is False
    assert "recoverable" not in ordinary.strip_presentation()["execution"]
    assert recovery.strip_presentation() != ordinary.strip_presentation()
    with pytest.raises(ValidationError):
        RunExperimentsInput.model_validate({**payload, "execution": {"recoverable": "true"}})


def test_resume_has_its_own_strict_advertised_input():
    payload = {"action": "resume", "job_id": "exp_parent", "resume_request_id": "retry"}
    args = JobsInput.model_validate(payload)
    assert isinstance(args, JobsResumeInput)
    assert args.wait_s == 0
    assert args.retry_failed is args.retry_cancelled is False
    jsonschema.Draft202012Validator(build_input_schema(JobsInput)).validate(payload)
    for extra in (
        {"request_id": "root"},
        {"owner_pid": 1},
        {"max_parallel": 2},
        {"retry_failed": "true"},
        {"case_ids": [1]},
        {"wait_s": -1},
    ):
        with pytest.raises(ValidationError):
            JobsInput.model_validate({**payload, **extra})


def test_only_resume_can_detach(state_no_sim: SessionState):
    api = SyncApi(state_no_sim)
    with pytest.raises(ApiValidationError, match="resume"):
        api.jobs(action="list", detach=True)
    with pytest.raises(TypeError, match="detach must be a bool"):
        api.jobs(action="list", detach=cast(bool, "true"))


_LIVE = pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH")
_DECK = "* divider\nV1 in 0 1\nR1 in out 1k\nR2 out 0 1k\n.op\n.end\n"


def _state(folder: Path) -> SessionState:
    program = shutil.which("ngspice")
    assert program is not None
    simulator = NGspiceSimulator.create_from(program)
    return SessionState.create(
        ServerConfig(working_dir=folder, allowed_paths=[folder], simulator="ngspice"),
        available={"ngspice": simulator},
    )


def _data(result, schema: dict) -> dict:
    data = result.structured_content
    assert data is not None
    jsonschema.Draft202012Validator(schema).validate(data)
    assert not result.is_error, data
    return data


def _fail_second_case(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    recorded_fixture_simulator(monkeypatch)
    recorded_submit = ExperimentRunner.submit_netlist
    submissions: list[str] = []
    failed_once = False

    def submit(self, netlist, run_filename, callback, **kwargs):
        nonlocal failed_once
        submissions.append(run_filename)
        if not failed_once and Path(run_filename).stem.endswith("_case_1"):
            failed_once = True
            self.loop.call_soon_threadsafe(
                callback,
                RunOutcome("", "", 0, "Simulator failed", failure_code="simulation_failed"),
            )
            return object()
        return recorded_submit(self, netlist, run_filename, callback, **kwargs)

    monkeypatch.setattr(ExperimentRunner, "submit_netlist", submit)
    return submissions


async def _submit(state: SessionState, *, variations=None, analyze=None) -> dict:
    args = RunExperimentsInput.model_validate(
        {
            "request_id": "recoverable-public",
            "circuits": [{"path": str(state.working_dir / "divider.cir")}],
            "variations": variations or [],
            "execution": {"recoverable": True, "wait_s": LIVENESS_S},
            "lint": "off",
            "analyze": analyze,
        }
    )
    result = await asyncio.wait_for(handle_run_experiments(args, state), LIVENESS_S)
    return _data(result, RUN_EXPERIMENTS_OUTPUT_SCHEMA)


@_LIVE
@pytest.mark.asyncio
async def test_terminal_root_token_noop_replay_and_read_only_views(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (work_dir / "divider.cir").write_text(_DECK)
    state = _state(work_dir)
    recorded_fixture_simulator(monkeypatch)
    root = await _submit(state)
    assert root["status"] == "completed"
    assert root["control_token"]
    assert root["lineage"] == {
        "root_job_id": root["job_id"],
        "parent_job_id": None,
        "attempt_index": 0,
    }
    attempt = root["runs"]["items"][0]["attempt"]
    assert attempt["execution_job_id"] == root["job_id"]
    assert attempt["reused"] is False
    for action in ("status", "runs", "list"):
        payload = {"action": action}
        if action != "list":
            payload["job_id"] = root["job_id"]
        data = _data(
            await handle_jobs(JobsInput.model_validate(payload), state), JOBS_OUTPUT_SCHEMA
        )
        assert "control_token" not in data
        assert root["control_token"] not in str(data)
    resume_args = {
        "action": "resume",
        "job_id": root["job_id"],
        "resume_request_id": "no-op",
        "control_token": root["control_token"],
    }
    noop = _data(
        await handle_jobs(JobsInput.model_validate(resume_args), state), JOBS_OUTPUT_SCHEMA
    )
    # timing: this replay differs from the first call only in its dwell,
    # which must not change what the replay returns
    replay = _data(
        await handle_jobs(JobsInput.model_validate({**resume_args, "wait_s": 1}), state),
        JOBS_OUTPUT_SCHEMA,
    )
    assert noop["resumed"] is replay["resumed"] is False
    assert noop["replayed"] is False and replay["replayed"] is True
    assert noop["job_id"] == noop["head_job_id"] == root["job_id"]
    assert "control_token" not in noop and "control_token" not in replay
    assert "analyze_results" in noop["hint"]
    await state.job_registry.drain_pending()


@_LIVE
@pytest.mark.asyncio
async def test_resume_child_reuses_success_and_replays_after_source_deletion(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    deck = work_dir / "divider.cir"
    deck.write_text(_DECK)
    state = _state(work_dir)
    submissions = _fail_second_case(monkeypatch)
    root = await _submit(
        state,
        variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
        analyze={"recipes": [{"key": "vout", "metric": "value", "expr": "V(out)"}]},
    )
    assert root["status"] == "completed_with_failures"
    assert root["control_token"]
    parent = state.all_jobs[root["job_id"]]
    original_raw = parent.cases[0].raw_file
    assert original_raw is not None
    original_bytes = original_raw.read_bytes()
    deck.unlink()
    noop_payload = {
        "action": "resume",
        "job_id": root["job_id"],
        "resume_request_id": "leave-failures",
        "control_token": root["control_token"],
    }
    noop = _data(
        await handle_jobs(JobsInput.model_validate(noop_payload), state),
        JOBS_OUTPUT_SCHEMA,
    )
    assert noop["resumed"] is False
    failed_id = parent.cases[1].case_id
    payload = {
        "action": "resume",
        "job_id": root["job_id"],
        "resume_request_id": "retry-failed",
        "retry_failed": True,
        "wait_s": LIVENESS_S,
        "control_token": root["control_token"],
        "case_ids": [failed_id, failed_id],
    }
    child = _data(await handle_jobs(JobsInput.model_validate(payload), state), JOBS_OUTPUT_SCHEMA)
    assert child["resumed"] is True and child["replayed"] is False
    assert child["job_id"] != root["job_id"]
    assert child["control_token"] != root["control_token"]
    assert child["completeness"]["reused"] == child["completeness"]["submitted"] == 1
    assert child["completeness"]["produced"] == 2
    assert child["analysis"]["status"] == "completed"
    assert child["lineage"]["parent_job_id"] == root["job_id"]
    rows = child["runs"]["items"]
    assert rows[0]["attempt"]["execution_job_id"] == root["job_id"]
    assert rows[0]["attempt"]["reused"] is True
    assert rows[1]["attempt"]["execution_job_id"] == child["job_id"]
    assert rows[0]["run_index"] == 0 and rows[1]["run_index"] == 1
    assert state.all_jobs[child["job_id"]].cases[0].raw_file == original_raw
    assert original_raw.read_bytes() == original_bytes
    replay = _data(
        await handle_jobs(
            JobsInput.model_validate({**payload, "wait_s": 0, "case_ids": [failed_id]}), state
        ),
        JOBS_OUTPUT_SCHEMA,
    )
    assert replay["replayed"] and replay["job_id"] == child["job_id"]
    assert replay["control_token"] == child["control_token"]
    assert len(submissions) == 3
    assert len(set(submissions)) == 3

    def fail_render(*_args, **_kwargs):
        raise ValueError("Receipt rendering failed")

    with monkeypatch.context() as presentation:
        presentation.setattr(jobs_module, "render_jobs_receipt_snapshot", fail_render)
        failed_reply = await handle_jobs(JobsInput.model_validate(payload), state)
        assert failed_reply.is_error
        assert failed_reply.structured_content["error"]["commit_state"] == "committed"
        assert failed_reply.structured_content["job_id"] == child["job_id"]
        assert failed_reply.structured_content["control_token"] == child["control_token"]
    noop_replay = _data(
        await handle_jobs(JobsInput.model_validate(noop_payload), state),
        JOBS_OUTPUT_SCHEMA,
    )
    assert noop_replay["resumed"] is False and noop_replay["replayed"] is True
    assert noop_replay["job_id"] == root["job_id"]
    assert noop_replay["head_job_id"] == child["job_id"]
    assert "control_token" not in noop_replay
    conflict = await handle_jobs(
        JobsInput.model_validate({**payload, "retry_cancelled": True}), state
    )
    assert conflict.is_error
    assert conflict.structured_content["error"]["code"] == "idempotency_conflict"
    ordinary_replay = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": root["request_id"],
                "circuits": [{"path": str(deck)}],
                "variations": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
                "execution": {"recoverable": True, "wait_s": 0},
                "lint": "off",
            }
        ),
        state,
    )
    assert ordinary_replay.is_error
    assert ordinary_replay.structured_content["error"]["code"] == "idempotency_conflict"
    await state.job_registry.drain_pending()


@_LIVE
def test_detached_resume_delegates_owning_parent_and_keeps_terminal_child_token(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (work_dir / "divider.cir").write_text(_DECK)
    _fail_second_case(monkeypatch)
    with Api(working_dir=work_dir, simulator="ngspice", allowed_paths=[work_dir]) as api:
        root = api.run_experiments(
            request_id="detached-recovery",
            circuits=[{"path": "divider.cir"}],
            variations=[{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
            execution={"recoverable": True},
            lint="off",
        )
        assert root["status"] == "completed_with_failures"
        assert root["control_token"]
        child = api.jobs(
            detach=True,
            action="resume",
            job_id=root["job_id"],
            resume_request_id="detached-retry",
            retry_failed=True,
        )
        jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(child)
        assert child["resumed"] and child["control_token"] != root["control_token"]
        assert any(item["code"] == "detached_owner" for item in child["observations"])
        final = api.wait(child["job_id"], timeout=30)
        assert final["status"] == "completed"
        assert final["completeness"]["reused"] == final["completeness"]["submitted"] == 1
        assert "control_token" not in final
        replay = api.jobs(
            detach=True,
            action="resume",
            job_id=root["job_id"],
            resume_request_id="detached-retry",
            retry_failed=True,
            control_token=root["control_token"],
            wait_s=LIVENESS_S,
        )
        assert replay["replayed"] and replay["job_id"] == child["job_id"]
        assert replay["status"] == "completed"
        assert replay["control_token"] == child["control_token"]

        def fail_render(*_args, **_kwargs):
            raise ValueError("Receipt rendering failed")

        with monkeypatch.context() as presentation:
            presentation.setattr(jobs_module, "render_jobs_receipt_snapshot", fail_render)
            with pytest.raises(ApiCallError) as failed:
                api.jobs(
                    action="resume",
                    job_id=root["job_id"],
                    resume_request_id="detached-retry",
                    retry_failed=True,
                    control_token=root["control_token"],
                )
            assert failed.value.commit_state == "committed"
            assert failed.value.job_id == child["job_id"]
            assert failed.value.control_token == child["control_token"]
    with Api(working_dir=work_dir, simulator="ngspice", allowed_paths=[work_dir]) as reader:
        with pytest.raises(ApiCallError) as denied:
            reader.jobs(
                detach=True,
                action="resume",
                job_id=child["job_id"],
                resume_request_id="unauthorized-retry",
            )
        assert denied.value.code == "resume_not_authorized"


@_LIVE
@pytest.mark.asyncio
async def test_unsupported_recoverable_inputs_refuse_before_claim(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    (work_dir / "divider.cir").write_text(_DECK.replace(".op", ".control\nop\n.endc"))
    state = _state(work_dir)
    submissions = _fail_second_case(monkeypatch)
    result = await handle_run_experiments(
        RunExperimentsInput.model_validate(
            {
                "request_id": "no-claim",
                "circuits": [{"path": str(work_dir / "divider.cir")}],
                "execution": {"recoverable": True, "wait_s": 0},
                "lint": "off",
            }
        ),
        state,
    )
    data = result.structured_content
    assert result.is_error and data is not None
    assert data["error"]["code"] == "recovery_control_unsupported"
    assert data["error"]["retryable"] is False
    assert data["error"]["commit_state"] == "not_started"
    assert submissions == [] and state.all_jobs == {}
    assert not Store(work_dir).request_index("no-claim").exists()
    assert not Store(work_dir).recovery_journal("no-claim").exists()
