"""Authoritative root admission and crash boundaries through the real store."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_resume, experiment_store
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import capture_case_inputs
from ltspice_mcp.lib.experiment_runner import (
    ExperimentRunner,
    ExperimentRunRequest,
    IdempotencyConflictError,
    StagedDecks,
    SubmissionCommitted,
)
from ltspice_mcp.lib.experiment_types import ExperimentCase, ManifestEntry, SourceRecord
from ltspice_mcp.lib.recovery_journal import load_journal
from ltspice_mcp.lib.recovery_records import CaseAttempt, CaseRecovery, RecoveryError
from ltspice_mcp.lib.simulator_build import executable_identity
from ltspice_mcp.lib.store import Store
from tests.test_resume_surface import _DECK, _state

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(shutil.which("ngspice") is None, reason="ngspice not on PATH"),
]


def request_for(folder, *, request_id="admission", job_id="exp_admission"):
    state = _state(folder)
    simulator = state.available_simulators["ngspice"]
    store = Store(folder)
    runner = ExperimentRunner(asyncio.get_running_loop(), simulator, store.runs_root(simulator))
    stages = []

    async def stage():
        stages.append(job_id)
        authoring = folder / "divider.cir"
        authoring.write_text(_DECK, encoding="utf-8")
        root = store.run_dir(job_id, simulator)
        root.mkdir(parents=True, exist_ok=True)
        deck = root / "case.cir"
        deck.write_bytes(authoring.read_bytes())
        digest = sha256_file(deck)
        source = SourceRecord(
            "divider",
            authoring,
            digest,
            deck,
            manifest=[ManifestEntry(authoring, digest, True, False, deck)],
            simulator=simulator.__name__,
        )
        case = ExperimentCase("nominal", 0, "divider", authoring, deck, digest)
        case.recovery = CaseRecovery(
            capture_case_inputs(case, source, lineage_root=root),
            CaseAttempt(job_id, 0, f"{job_id}_case_0"),
        )
        return StagedDecks([case], [source])

    request = ExperimentRunRequest(
        state,
        request_id,
        "a" * 64,
        simulator.__name__,
        stage,
        job_id=job_id,
        simulator_executable=executable_identity(simulator),
        recoverable=True,
    )
    return runner, request, stages


async def test_journal_is_committed_before_job_or_index(work_dir, monkeypatch):
    runner, request, stages = request_for(work_dir)
    store = Store(work_dir)
    save = experiment_resume.save_journal
    commits = []

    def observed_save(store, journal):
        assert not store.job_record(journal.root_job_id).exists()
        assert not store.request_index(journal.root_request_id).exists()
        save(store, journal)
        commits.append(journal.root_job_id)

    monkeypatch.setattr(experiment_resume, "save_journal", observed_save)
    admitted = await runner._durable_barrier(request)
    assert admitted.start and not admitted.replayed
    assert commits == stages == [admitted.job.job_id]
    assert experiment_store.load_job(admitted.job.job_id, work_dir) is not None
    assert store.request_index(request.request_id).exists()


async def test_journal_only_commit_blocks_ordinary_request_and_retains_inputs(
    work_dir, monkeypatch
):
    runner, request, stages = request_for(work_dir)
    store = Store(work_dir)

    def failed_derived_write(*_args, **_kwargs):
        raise OSError("Derived write failed")

    monkeypatch.setattr(experiment_resume, "_persist_discovery", failed_derived_write)
    with pytest.raises(SubmissionCommitted):
        await runner._durable_barrier(request)
    journal = load_journal(store, request.request_id)
    assert journal is not None and journal.root.candidate is not None
    assert not store.request_index(request.request_id).exists()
    assert not store.job_record(journal.root_job_id).exists()
    assert store.run_dir(journal.root_job_id).joinpath("case.cir").is_file()
    with pytest.raises(IdempotencyConflictError):
        await runner._durable_barrier(replace(request, recoverable=False))
    assert stages == [journal.root_job_id]


async def test_failed_authoritative_write_never_claims_and_cleans_staging(work_dir, monkeypatch):
    runner, request, _ = request_for(work_dir)
    store = Store(work_dir)

    def failed_journal(*_args):
        raise RecoveryError("recovery_journal_write_failed", "Injected journal failure")

    monkeypatch.setattr(experiment_resume, "save_journal", failed_journal)
    with pytest.raises(RecoveryError, match="Injected"):
        await runner._durable_barrier(request)
    assert not store.request_index(request.request_id).exists()
    assert request.job_id is not None
    assert not store.run_dir(request.job_id).exists()
    assert request.state.all_jobs == {}


async def test_missing_launched_record_is_never_reconstructed(work_dir):
    runner, request, stages = request_for(work_dir)
    admitted = await runner._durable_barrier(request)
    job = admitted.job
    await experiment_resume.mark_attempt_launched(job, request.state)
    job.store_path.unlink()
    with pytest.raises(RecoveryError) as error:
        await runner._durable_barrier(request)
    assert error.value.code == "recovery_record_missing"
    assert not job.store_path.exists()
    assert stages == [job.job_id]


async def test_corrupt_prelaunch_record_is_not_replaced_by_old_snapshot(work_dir):
    runner, request, stages = request_for(work_dir)
    admitted = await runner._durable_barrier(request)
    admitted.job.store_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(RecoveryError) as error:
        await runner._durable_barrier(request)
    assert error.value.code == "recovery_record_missing"
    assert admitted.job.store_path.read_text(encoding="utf-8") == "{broken"
    assert stages == [admitted.job.job_id]


async def test_live_prelaunch_replay_never_starts_again(work_dir):
    runner, request, stages = request_for(work_dir)
    admitted = await runner._durable_barrier(request)
    replay = await runner._durable_barrier(request)
    assert replay.job.job_id == admitted.job.job_id
    assert replay.replayed and not replay.start
    assert stages == [admitted.job.job_id]


@pytest.mark.parametrize("boundary", ["journal", "record", "launch"])
async def test_dead_owner_replay_across_actual_process_exit(work_dir, boundary):
    worker = await asyncio.to_thread(Path(__file__).resolve)
    project = worker.parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(project), str(project / "src"))))
    completed = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, str(worker), str(work_dir), boundary],
        env=env,
        capture_output=True,
        timeout=30,
    )
    assert completed.returncode == 23, completed.stderr.decode(errors="replace")
    store = Store(work_dir)
    journal = load_journal(store, "admission")
    assert journal is not None
    if boundary == "journal":
        assert not store.job_record(journal.root_job_id).exists()
    runner, request, stages = request_for(work_dir)
    replay = await runner._durable_barrier(request)
    assert replay.replayed and replay.job.job_id == journal.root_job_id
    assert replay.start is (boundary != "launch")
    assert stages == []
    if boundary != "launch":
        assert replay.job.owner_pid == os.getpid()
        assert replay.job.status == "queued"
        assert store.job_record(replay.job.job_id).exists()
        assert store.request_index(request.request_id).exists()
    else:
        assert replay.job.status == "interrupted"


async def _crash_worker(folder: Path, boundary: str) -> None:
    runner, request, _ = request_for(folder)
    if boundary == "journal":

        def exit_after_commit(*_args, **_kwargs):
            os._exit(23)

        experiment_resume._persist_discovery = exit_after_commit
    admitted = await runner._durable_barrier(request)
    if boundary == "launch":
        await experiment_resume.mark_attempt_launched(admitted.job, request.state)
    os._exit(23)


if __name__ == "__main__":
    asyncio.run(_crash_worker(Path(sys.argv[1]), sys.argv[2]))
