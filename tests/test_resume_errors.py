"""Public recovery error boundaries and authoritative early replay lookup."""

from __future__ import annotations

import asyncio
import copy
import shutil
import sys
from pathlib import Path

import jsonschema
import pytest
from spicelib.simulators.ngspice_simulator import NGspiceSimulator

from ltspice_mcp.api import Api, ApiCallError
from ltspice_mcp.api import _methods as methods
from ltspice_mcp.lib import experiment_resume, experiment_store
from ltspice_mcp.lib.experiment_runner import (
    CANONICALIZER_VERSION,
    ExperimentRunner,
    canonical_fingerprint,
)
from ltspice_mcp.lib.recovery_journal import candidate_job, load_journal
from ltspice_mcp.lib.runner_base import RunOutcome
from ltspice_mcp.lib.store import Store, path_digest
from ltspice_mcp.tools import experiments, receipts
from ltspice_mcp.tools.jobs import JOBS_OUTPUT_SCHEMA
from tests.conftest import LIVENESS_S
from tests.test_resume_surface import _DECK, _fail_second_case, _state

pytestmark = [
    pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH"),
    pytest.mark.skipif(
        sys.platform not in {"linux", "win32"},
        reason="Recoverable ngspice startup is verified on Linux and native Windows only",
    ),
]


@pytest.mark.parametrize("case_order", [(0, 1), (1, 0), (10, 1, 0)])
@pytest.mark.asyncio
async def test_failure_fixture_targets_case_identity_once(
    work_dir: Path, monkeypatch: pytest.MonkeyPatch, case_order: tuple[int, ...]
):
    """Isolated fixture check; submission order must not choose the failed case."""
    submissions = _fail_second_case(monkeypatch)
    loop = asyncio.get_running_loop()
    runner = ExperimentRunner(loop, NGspiceSimulator, work_dir)
    names = [f"exp_root_case_{index}.net" for index in case_order]
    names.append("exp_child_case_1.net")
    outcomes: list[RunOutcome] = []
    for filename in names:
        received: asyncio.Future[RunOutcome] = loop.create_future()
        runner.submit_netlist(work_dir / "deck.cir", filename, received.set_result)
        outcomes.append(await asyncio.wait_for(received, LIVENESS_S))

    assert submissions == names
    assert [outcome.failure_code for outcome in outcomes] == [
        "simulation_failed" if index == 1 else None for index in case_order
    ] + [None]
    assert outcomes[-1].raw_file


@pytest.fixture
def recoverable_api(work_dir: Path, monkeypatch: pytest.MonkeyPatch):
    # The allowlist reaches the spawned owner too; startup must never probe LTspice.
    monkeypatch.setenv("LTSPICE_MCP_ENABLED_SIMULATORS", "ngspice")
    (work_dir / "divider.cir").write_text(_DECK, encoding="utf-8")
    submissions = _fail_second_case(monkeypatch)
    arguments = {
        "request_id": "public-error-root",
        "circuits": [{"path": "divider.cir"}],
        "variations": [{"kind": "assign", "assign": {"R1": ["1k", "2k"]}}],
        "execution": {"recoverable": True},
        "lint": "off",
    }
    with Api(
        working_dir=work_dir,
        allowed_paths=[work_dir],
        simulator="ngspice",
        simulator_exe=shutil.which("ngspice"),
        enabled_simulators=["ngspice"],
        ngbehavior="hsa",
    ) as api:
        root = api.run_experiments(**copy.deepcopy(arguments))
        assert root["status"] == "completed_with_failures"
        yield api, root, submissions, arguments


def _resume_arguments(root: dict) -> dict:
    return {
        "action": "resume",
        "job_id": root["job_id"],
        "resume_request_id": "public-error-retry",
        "retry_failed": True,
        "control_token": root["control_token"],
    }


def _committed_child(work_dir: Path, root: dict):
    journal = load_journal(Store(work_dir), root["request_id"])
    assert journal is not None and journal.head_job_id != root["job_id"]
    child = experiment_store.load_job(journal.head_job_id, work_dir, own_is_alive=True)
    assert child is not None
    return child


@pytest.mark.parametrize("raw_page", [False, True], ids=["automatic-api", "mcp-handler"])
def test_resume_discovery_failure_retains_journal_committed_child(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch, raw_page: bool
):
    api, root, submissions, _arguments = recoverable_api

    def fail_discovery(*_args, **_kwargs):
        raise OSError("Child discovery write failed after journal commit")

    with monkeypatch.context() as discovery:
        discovery.setattr(experiment_resume, "_persist_discovery", fail_discovery)
        with pytest.raises(ApiCallError) as failed:
            api.jobs(raw_page=raw_page, **_resume_arguments(root))

    store = Store(work_dir)
    journal = load_journal(store, root["request_id"])
    assert journal is not None
    entry = journal.resumes[path_digest("public-error-retry")]
    child = candidate_job(entry, store)
    assert not store.job_record(child.job_id).exists()
    error = failed.value
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(error.payload)
    assert error.commit_state == "committed" and error.code == "submission_committed"
    assert error.job_id == error.payload["head_job_id"] == journal.head_job_id == child.job_id
    assert error.control_token == child.control_token
    assert error.payload["request_id"] == error.payload["resume_request_id"] == entry.request_id
    assert error.payload["addressed_parent_job_id"] == root["job_id"]
    assert error.payload["resumed"] and not error.payload["replayed"]
    assert len(submissions) == 2


@pytest.mark.parametrize("raw_page", [False, True], ids=["automatic-api", "mcp-handler"])
def test_root_discovery_and_snapshot_failure_retains_journal_committed_root(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch, raw_page: bool
):
    api, _root, submissions, arguments = recoverable_api
    arguments = copy.deepcopy(arguments)
    arguments["request_id"] = "journal-committed-root"

    def fail_discovery(*_args, **_kwargs):
        raise OSError("Root discovery write failed after journal commit")

    def fail_dialect(*_args, **_kwargs):
        raise OSError("Committed root snapshot unavailable")

    with monkeypatch.context() as boundary:
        boundary.setattr(experiment_resume, "_persist_discovery", fail_discovery)
        boundary.setattr(receipts.services, "dialect_for_job", fail_dialect)
        with pytest.raises(ApiCallError) as failed:
            api.run_experiments(raw_page=raw_page, **arguments)

    store = Store(work_dir)
    journal = load_journal(store, arguments["request_id"])
    assert journal is not None
    root = candidate_job(journal.root, store)
    assert not store.job_record(root.job_id).exists()
    error = failed.value
    jsonschema.Draft202012Validator(receipts.RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(error.payload)
    assert error.commit_state == "committed" and error.code == "submission_committed"
    assert error.payload["error"]["stage"] == "submission"
    assert error.job_id == journal.head_job_id == root.job_id
    assert error.control_token == root.control_token
    assert error.payload["request_id"] == arguments["request_id"]
    assert not error.payload["replayed"]
    assert "is running" not in error.payload["hint"]
    assert len(submissions) == 2


@pytest.mark.parametrize("raw_page", [False, True], ids=["automatic-api", "mcp-handler"])
def test_resume_snapshot_failure_retains_committed_child(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch, raw_page: bool
):
    api, root, submissions, _arguments = recoverable_api

    def fail_dialect(*_args, **_kwargs):
        raise OSError("Snapshot dialect read failed")

    with monkeypatch.context() as presentation:
        presentation.setattr(receipts.services, "dialect_for_job", fail_dialect)
        with pytest.raises(ApiCallError) as failed:
            api.jobs(raw_page=raw_page, **_resume_arguments(root))

    child = _committed_child(work_dir, root)
    error = failed.value
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(error.payload)
    assert error.commit_state == "committed"
    assert error.job_id == error.payload["head_job_id"] == child.job_id
    assert error.control_token == child.control_token
    assert error.payload["request_id"] == "public-error-retry"
    assert error.payload["addressed_parent_job_id"] == root["job_id"]
    assert error.payload["resumed"] and not error.payload["replayed"]
    assert "Snapshot dialect read failed" in str(error)
    final = api.wait(child.job_id, timeout=30)
    assert final["status"] == "completed" and final["completeness"]["reused"] == 1
    replay = api.jobs(**_resume_arguments(root))
    assert replay["replayed"] and replay["job_id"] == child.job_id
    assert replay["control_token"] == child.control_token
    assert len(submissions) == 3


def test_detached_resume_owner_lookup_failure_is_committed(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    api, root, _submissions, _arguments = recoverable_api

    async def fail_owner_lookup(*_args, **_kwargs):
        raise OSError("Child owner record unavailable")

    with monkeypatch.context() as presentation:
        presentation.setattr(methods, "_record_owner_pid", fail_owner_lookup)
        with pytest.raises(ApiCallError) as failed:
            api.jobs(detach=True, **_resume_arguments(root))

    child = _committed_child(work_dir, root)
    error = failed.value
    jsonschema.Draft202012Validator(JOBS_OUTPUT_SCHEMA).validate(error.payload)
    assert error.commit_state == "committed"
    assert error.code == "submission_committed"
    assert error.job_id == error.payload["head_job_id"] == child.job_id
    assert error.control_token == child.control_token
    assert error.payload["resume_request_id"] == "public-error-retry"
    assert error.payload["addressed_parent_job_id"] == root["job_id"]
    assert error.payload["outcome"] == "failed"
    assert "Child owner record unavailable" in str(error)
    final = api.wait(child.job_id, timeout=30)
    assert final["status"] == "completed"
    assert final["completeness"]["submitted"] == final["completeness"]["reused"] == 1
    assert "control_token" not in final


@pytest.mark.parametrize("raw_page", [False, True], ids=["automatic-api", "mcp-handler"])
def test_root_replay_snapshot_failure_retains_committed_root(
    recoverable_api, monkeypatch: pytest.MonkeyPatch, raw_page: bool
):
    api, root, submissions, arguments = recoverable_api

    def fail_dialect(*_args, **_kwargs):
        raise OSError("Root snapshot dialect read failed")

    with monkeypatch.context() as presentation:
        presentation.setattr(receipts.services, "dialect_for_job", fail_dialect)
        with pytest.raises(ApiCallError) as failed:
            api.run_experiments(raw_page=raw_page, **copy.deepcopy(arguments))

    error = failed.value
    jsonschema.Draft202012Validator(receipts.RUN_EXPERIMENTS_OUTPUT_SCHEMA).validate(error.payload)
    assert error.commit_state == "committed"
    assert error.job_id == root["job_id"]
    assert error.control_token == root["control_token"]
    assert error.payload["replayed"]
    assert len(submissions) == 2


@pytest.mark.parametrize("index_state", ["missing", "dangling", "conflicting", "corrupt"])
def test_root_journal_precedes_derived_index_and_staging_route(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch, index_state: str
):
    api, root, submissions, arguments = recoverable_api
    store = Store(work_dir)
    index = store.request_index(root["request_id"])
    if index_state == "missing":
        index.unlink()
    elif index_state == "corrupt":
        index.write_text("{broken", encoding="utf-8")
    else:
        experiment_store.save_request_index(
            request_id=root["request_id"],
            fingerprint=(
                "b" * 64
                if index_state == "conflicting"
                else canonical_fingerprint(
                    experiments.RunExperimentsInput.model_validate(arguments)
                )
            ),
            canonicalizer_version=CANONICALIZER_VERSION,
            job_id="exp_missing",
            working_dir=work_dir,
        )

    def fail_staging_route(*_args, **_kwargs):
        raise OSError("A recorded root must resolve before new staging")

    with monkeypatch.context() as boundary:
        boundary.setattr(experiments, "resolve_experiment_paths", fail_staging_route)
        replay = api.run_experiments(**copy.deepcopy(arguments))

    assert replay["replayed"] and replay["job_id"] == root["job_id"]
    assert replay["control_token"] == root["control_token"]
    assert len(submissions) == 2


@pytest.mark.parametrize("recoverable", [False, True], ids=["ordinary", "recoverable"])
@pytest.mark.parametrize("conflicting_index", [False, True], ids=["no-index", "wrong-index"])
@pytest.mark.asyncio
async def test_corrupt_journal_refuses_before_index_or_new_staging(
    work_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    recoverable: bool,
    conflicting_index: bool,
):
    state = _state(work_dir)
    store = Store(work_dir)
    journal = store.recovery_journal("broken-root")
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text("{broken", encoding="utf-8")
    if conflicting_index:
        experiment_store.save_request_index(
            request_id="broken-root",
            fingerprint="b" * 64,
            canonicalizer_version=CANONICALIZER_VERSION,
            job_id="exp_missing",
            working_dir=work_dir,
        )

    def fail_staging_route(*_args, **_kwargs):
        raise OSError("Corrupt evidence must resolve before new staging")

    monkeypatch.setattr(experiments, "resolve_experiment_paths", fail_staging_route)
    result = await experiments.handle_run_experiments(
        experiments.RunExperimentsInput.model_validate(
            {
                "request_id": "broken-root",
                "circuits": [{"path": "absent.cir"}],
                "execution": {"recoverable": recoverable, "wait_s": 0},
            }
        ),
        state,
    )
    data = result.structured_content
    assert data is not None and result.is_error
    assert data["error"]["code"] == "recovery_journal_invalid"
    assert data["error"]["commit_state"] == "not_started"
    assert state.all_jobs == {}
    assert journal.read_text(encoding="utf-8") == "{broken"


@pytest.mark.parametrize("collision", ["ordinary", "changed-request"])
def test_journal_collision_refuses_before_new_staging(
    recoverable_api, work_dir: Path, monkeypatch: pytest.MonkeyPatch, collision: str
):
    api, root, submissions, original = recoverable_api
    Store(work_dir).request_index(root["request_id"]).unlink()
    arguments = copy.deepcopy(original)
    if collision == "ordinary":
        arguments["execution"]["recoverable"] = False
    else:
        arguments["variations"][0]["assign"]["R1"] = ["3k"]

    def fail_staging_route(*_args, **_kwargs):
        raise OSError("A reserved root must resolve before new staging")

    monkeypatch.setattr(experiments, "resolve_experiment_paths", fail_staging_route)
    with pytest.raises(ApiCallError) as failed:
        api.run_experiments(**arguments)
    assert failed.value.code == "idempotency_conflict"
    assert len(submissions) == 2
