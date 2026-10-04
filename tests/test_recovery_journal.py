"""Authoritative recovery admission, bounded snapshots and strict disk loading."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_types import ExperimentJob
from ltspice_mcp.lib.filelock import file_lock
from ltspice_mcp.lib.pdk_native import ArtifactDigest
from ltspice_mcp.lib.recovery_journal import (
    RecoveryJournal,
    add_child,
    add_noop,
    candidate_job,
    load_journal,
    mark_launched,
    new_root_journal,
    save_journal,
)
from ltspice_mcp.lib.recovery_records import (
    CaseAttempt,
    LaunchIntent,
    ProducedArtifacts,
    RecoveryError,
)
from ltspice_mcp.lib.store import (
    KIND_RECOVERY_JOURNAL,
    STORE_VERSION,
    Store,
    atomic_write_json,
    envelope,
    path_digest,
)
from tests.conftest import symlink_or_skip
from tests.test_experiment_job import _job
from tests.test_recovery_records import recovery_job


@pytest.fixture
def root_job(tmp_path: Path) -> ExperimentJob:
    job = recovery_job(tmp_path)
    assert job.recovery is not None
    job.owner_pid = job.recovery.owner.pid
    job.status = "queued"
    job.simulator_executable = job.recovery.execution.executable
    return job


def _child(root: ExperimentJob, tmp_path: Path, *, number: int = 1) -> ExperimentJob:
    assert root.recovery is not None
    assert root.cases[0].recovery is not None
    job = _job(
        tmp_path,
        root.cases[0].staged_deck,
        job_id=f"exp_child_{number}",
        request_id=f"resume/{number}",
    )
    job.fingerprint = str(number) * 64
    job.owner_pid = root.owner_pid
    job.output_folder = root.output_folder
    job.simulator_executable = root.simulator_executable
    job.recovery = replace(
        root.recovery, parent_job_id=root.job_id, attempt_index=root.recovery.attempt_index + 1
    )
    job.cases[0].deck_sha256 = root.cases[0].deck_sha256
    job.cases[0].run_token = job.job_id + "_case_0"
    job.cases[0].recovery = replace(
        root.cases[0].recovery,
        attempt=CaseAttempt(job.job_id, job.recovery.attempt_index, job.cases[0].run_token),
    )
    return job


def _write_payload(store: Store, root_request_id: str, payload: dict) -> None:
    atomic_write_json(
        store.recovery_journal(root_request_id), envelope(KIND_RECOVERY_JOURNAL, **payload)
    )


def test_prepared_root_is_discoverable_without_any_derived_records(root_job, tmp_path):
    store = Store(tmp_path)
    journal = new_root_journal(root_job)
    save_journal(store, journal)
    assert not root_job.store_path.exists()
    assert not store.request_index(root_job.request_id).exists()
    restored = load_journal(store, root_job.request_id)
    assert restored is not None
    assert restored == journal
    assert restored.root.candidate == json.loads(
        json.dumps(experiment_store.serialize_job(root_job), default=str)
    )
    candidate = candidate_job(restored.root, store)
    assert candidate.store_path == store.job_record(root_job.job_id)
    assert candidate.job_id == root_job.job_id
    assert candidate.recovery == root_job.recovery
    assert candidate.cases[0].recovery == root_job.cases[0].recovery
    assert candidate.status == "queued"
    assert candidate.cases[0].status == "queued"
    assert not candidate.restart_reconciled
    assert json.loads(json.dumps(experiment_store.serialize_job(candidate), default=str)) == (
        restored.root.candidate
    )


def test_actual_atomic_replacement_and_current_envelope(root_job, tmp_path):
    store = Store(tmp_path)
    journal = new_root_journal(root_job)
    save_journal(store, journal)
    original = journal.to_record()
    launched = mark_launched(journal, root_job.job_id)
    save_journal(store, launched)
    data = json.loads(store.recovery_journal(root_job.request_id).read_text(encoding="utf-8"))
    assert data["kind"] == KIND_RECOVERY_JOURNAL
    assert data["store_version"] == STORE_VERSION
    assert data["root"]["candidate"] is None
    assert data["root"]["phase"] == "launched"
    assert load_journal(store, root_job.request_id) == launched
    assert journal.to_record() == original
    assert list(store.recovery_journal(root_job.request_id).parent.iterdir()) == [
        store.recovery_journal(root_job.request_id)
    ]


def test_children_noop_roundtrip_and_bounded_launched_metadata(root_job, tmp_path):
    store = Store(tmp_path)
    journal = mark_launched(new_root_journal(root_job), root_job.job_id)
    first = _child(root_job, tmp_path)
    prior = journal.to_record()
    journal = add_child(journal, first.request_id, first.fingerprint, first)
    assert prior["head_job_id"] == root_job.job_id
    assert journal.head_job_id == first.job_id
    assert journal.resumes[path_digest(first.request_id)].candidate is not None
    assert (
        candidate_job(journal.resumes[path_digest(first.request_id)], store).recovery
        == first.recovery
    )
    journal = mark_launched(journal, first.job_id)
    journal = add_noop(journal, "nothing/to-do", "b" * 64, first)
    noop = journal.resumes[path_digest("nothing/to-do")]
    assert noop.phase == "noop"
    assert noop.job_id is None and noop.candidate is None
    assert noop.parent_job_id == first.job_id and noop.attempt_index == 1
    assert journal.head_job_id == first.job_id
    second = _child(first, tmp_path, number=2)
    journal = add_child(journal, second.request_id, second.fingerprint, second)
    save_journal(store, journal)
    assert load_journal(store, root_job.request_id) == journal
    assert journal.root.candidate is None
    assert journal.resumes[path_digest(first.request_id)].candidate is None
    assert journal.resumes[path_digest(second.request_id)].candidate is not None
    with pytest.raises(RecoveryError):
        candidate_job(journal.resumes[path_digest(first.request_id)], store)
    with pytest.raises(RecoveryError):
        candidate_job(noop, store)


def test_absent_is_distinct_from_unreadable_or_broken_file(root_job, tmp_path):
    store = Store(tmp_path)
    assert load_journal(store, root_job.request_id) is None
    path = store.recovery_journal(root_job.request_id)
    path.mkdir(parents=True)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)
    path.rmdir()
    path.write_bytes(b"{broken")
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)
    path.write_bytes(b"\xff")
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


def test_broken_symlink_is_evidence_not_absence(root_job, tmp_path):
    store = Store(tmp_path)
    path = store.recovery_journal(root_job.request_id)
    path.parent.mkdir(parents=True)
    symlink_or_skip(path, path.parent / "missing.json")
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize(
    "change",
    [
        {"schema": "elsewhere"},
        {"kind": "experiment"},
        {"store_version": 3},
        {"store_version": True},
        {"store_version": "4"},
        {"unexpected": 1},
        {"root_request_id": "another-request"},
    ],
)
def test_present_invalid_envelope_and_source_binding_refuse(root_job, tmp_path, change):
    store = Store(tmp_path)
    data = envelope(KIND_RECOVERY_JOURNAL, **new_root_journal(root_job).to_record())
    data.update(change)
    atomic_write_json(store.recovery_journal(root_job.request_id), data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize(
    "change",
    [
        {"attempt_index": True},
        {"attempt_index": "0"},
        {"fingerprint": "F" * 64},
        {"fingerprint": "short"},
        {"request_id": ""},
        {"job_id": "../unsafe"},
        {"phase": "pending"},
        {"phase": "noop"},
        {"candidate": None},
        {"unknown": 1},
        {"parent_job_id": "exp_parent"},
    ],
)
def test_root_entry_strict_decode(root_job, tmp_path, change):
    store = Store(tmp_path)
    data = new_root_journal(root_job).to_record()
    data["root"].update(change)
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("job_id", "exp_other"),
        ("request_id", "other"),
        ("fingerprint", "a" * 64),
        ("pid", True),
        ("pid", 1),
        ("kind", "other"),
        ("store_version", 3),
    ],
)
def test_candidate_identity_and_owner_must_match(root_job, tmp_path, field, value):
    store = Store(tmp_path)
    data = new_root_journal(root_job).to_record()
    data["root"]["candidate"][field] = value
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("root_job_id", "exp_other"),
        ("root_request_id", "other"),
        ("attempt_index", 2),
        ("parent_job_id", "exp_other"),
    ],
)
def test_child_candidate_lineage_must_match(root_job, tmp_path, field, value):
    store = Store(tmp_path)
    child = _child(root_job, tmp_path)
    journal = add_child(new_root_journal(root_job), child.request_id, child.fingerprint, child)
    data = journal.to_record()
    data["resumes"][path_digest(child.request_id)]["candidate"]["recovery"][field] = value
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


def test_prepared_current_attempt_cannot_contain_launch_intent(root_job):
    case = root_job.cases[0]
    case.recovery = replace(
        case.recovery,
        attempt=replace(
            case.recovery.attempt,
            launch=LaunchIntent(
                now(),
                case.deck_sha256,
                ArtifactDigest(case.staged_deck, case.deck_sha256),
                "identity",
            ),
        ),
    )
    with pytest.raises(RecoveryError):
        new_root_journal(root_job)


def test_prepared_child_preserves_prior_reused_launch(root_job, tmp_path):
    journal = mark_launched(new_root_journal(root_job), root_job.job_id)
    case = root_job.cases[0]
    case.status = "produced"
    case.raw_file = case.staged_deck.with_suffix(".raw")
    case.log_file = case.staged_deck.with_suffix(".log")
    case.raw_file.write_bytes(b"recorded successful result")
    case.log_file.write_bytes(b"recorded solve log")
    outputs = ProducedArtifacts(
        ArtifactDigest(case.raw_file, sha256_file(case.raw_file)),
        ArtifactDigest(case.log_file, sha256_file(case.log_file)),
        now(),
    )
    case.recovery = replace(
        case.recovery,
        attempt=replace(
            case.recovery.attempt,
            launch=LaunchIntent(
                now(),
                case.deck_sha256,
                ArtifactDigest(case.staged_deck, case.deck_sha256),
                "identity",
            ),
            outputs=outputs,
        ),
    )
    child = _child(root_job, tmp_path)
    child.cases[0].status = "produced"
    child.cases[0].raw_file = case.raw_file
    child.cases[0].log_file = case.log_file
    child.cases[0].recovery = replace(
        case.recovery, attempt=replace(case.recovery.attempt, reused=True)
    )
    child.cases[0].run_token = case.run_token
    journal = add_child(journal, child.request_id, child.fingerprint, child)
    restored = candidate_job(journal.resumes[path_digest(child.request_id)], Store(tmp_path))
    assert restored.cases[0].recovery == child.cases[0].recovery
    assert restored.cases[0].raw_file is not None
    assert restored.cases[0].raw_file.read_bytes() == b"recorded successful result"


@pytest.mark.parametrize("corruption", ["head", "key", "parent", "sequence", "duplicate", "noop"])
def test_chain_corruption_refuses(root_job, tmp_path, corruption):
    store = Store(tmp_path)
    child = _child(root_job, tmp_path)
    second = _child(child, tmp_path, number=2)
    journal = add_child(new_root_journal(root_job), child.request_id, child.fingerprint, child)
    journal = add_child(journal, second.request_id, second.fingerprint, second)
    journal = add_noop(journal, "noop", "b" * 64, second)
    data = journal.to_record()
    entry = data["resumes"][path_digest(second.request_id)]
    # Dropping snapshots after launch isolates structural chain validation.
    for item in [data["root"], *data["resumes"].values()]:
        if item["phase"] != "noop":
            item.update(phase="launched", candidate=None)
    if corruption == "head":
        data["head_job_id"] = root_job.job_id
    elif corruption == "key":
        data["resumes"]["0" * 64] = data["resumes"].pop(path_digest(second.request_id))
    elif corruption == "parent":
        entry["parent_job_id"] = root_job.job_id
    elif corruption == "sequence":
        entry["attempt_index"] = 4
    elif corruption == "duplicate":
        entry["job_id"] = child.job_id
    else:
        data["resumes"][path_digest("noop")]["attempt_index"] = 3
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


def test_mutators_refuse_stale_parent_duplicate_resume_and_unknown_launch(root_job, tmp_path):
    child = _child(root_job, tmp_path)
    journal = add_child(new_root_journal(root_job), child.request_id, child.fingerprint, child)
    before = journal.to_record()
    with pytest.raises(RecoveryError):
        add_child(journal, child.request_id, child.fingerprint, child)
    with pytest.raises(RecoveryError):
        add_noop(journal, "stale", "c" * 64, root_job)
    with pytest.raises(RecoveryError):
        mark_launched(journal, "exp_missing")
    assert journal.to_record() == before


def test_candidate_snapshot_does_not_alias_live_job(root_job):
    journal = new_root_journal(root_job)
    before = copy.deepcopy(journal.to_record())
    root_job.cases[0].assignments["R1"] = "2k"
    root_job.observations.append({"code": "changed"})
    assert journal.to_record() == before


def test_record_api_refuses_unknown_fields(root_job):
    data = new_root_journal(root_job).to_record()
    data["surprise"] = True
    with pytest.raises(RecoveryError):
        RecoveryJournal.from_record(data)


def test_duplicate_json_members_are_corrupt_evidence(root_job, tmp_path):
    store = Store(tmp_path)
    path = store.recovery_journal(root_job.request_id)
    path.parent.mkdir(parents=True)
    data = envelope(KIND_RECOVERY_JOURNAL, **new_root_journal(root_job).to_record())
    encoded = json.dumps(data)
    path.write_text('{"root_request_id":"other",' + encoded[1:], encoding="utf-8")
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


def test_atomic_replace_failure_preserves_prior_authority(root_job, tmp_path, monkeypatch):
    store = Store(tmp_path)
    journal = new_root_journal(root_job)
    save_journal(store, journal)
    path = store.recovery_journal(root_job.request_id)
    before = path.read_bytes()

    def failed_replace(source, destination):
        assert Path(destination) == path
        assert Path(source).is_file()
        raise OSError("replacement failed")

    monkeypatch.setattr(os, "replace", failed_replace)
    with pytest.raises(RecoveryError) as error:
        save_journal(store, mark_launched(journal, root_job.job_id))
    assert error.value.code == "recovery_journal_write_failed"
    assert path.read_bytes() == before
    assert load_journal(store, root_job.request_id) == journal
    assert list(path.parent.iterdir()) == [path]


def test_helpers_operate_under_caller_held_lock(root_job, tmp_path):
    store = Store(tmp_path)
    with file_lock(store.recovery_lock(root_job.request_id)):
        save_journal(store, new_root_journal(root_job))
        journal = load_journal(store, root_job.request_id)
        assert journal is not None
        save_journal(store, mark_launched(journal, root_job.job_id))
    restored = load_journal(store, root_job.request_id)
    assert restored is not None
    assert restored.root.phase == "launched"


@pytest.mark.parametrize("section", ["job", "case", "recovery", "owner"])
def test_candidate_unknown_fields_refuse(root_job, tmp_path, section):
    store = Store(tmp_path)
    data = new_root_journal(root_job).to_record()
    candidate = data["root"]["candidate"]
    targets = {
        "job": candidate,
        "case": candidate["cases"][0],
        "recovery": candidate["recovery"],
        "owner": candidate["recovery"]["owner"],
    }
    targets[section]["unexpected"] = 1
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize("field", ["control_token", "cases", "sources", "recovery"])
def test_candidate_missing_fields_refuse(root_job, tmp_path, field):
    store = Store(tmp_path)
    data = new_root_journal(root_job).to_record()
    del data["root"]["candidate"][field]
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)


@pytest.mark.parametrize("corruption", ["count", "boolean", "time", "version", "manifest"])
def test_candidate_decode_must_be_lossless(root_job, tmp_path, corruption):
    store = Store(tmp_path)
    data = new_root_journal(root_job).to_record()
    candidate = data["root"]["candidate"]
    if corruption == "count":
        candidate["completeness"]["expanded"] = "1"
    elif corruption == "boolean":
        candidate["completeness"]["expanded"] = True
    elif corruption == "time":
        candidate["started_at"] = "not-a-timestamp"
    elif corruption == "version":
        candidate["canonicalizer_version"] = str(candidate["canonicalizer_version"])
    else:
        candidate["sources"][0]["manifest"][0]["unexpected"] = 1
    _write_payload(store, root_job.request_id, data)
    with pytest.raises(RecoveryError):
        load_journal(store, root_job.request_id)
