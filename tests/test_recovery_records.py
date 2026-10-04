"""Persisted recovery identity, strict decoding and retention contracts."""

from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import experiment_store, now
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.pdk_native import ArtifactDigest
from ltspice_mcp.lib.recovery_records import (
    CaseAttempt,
    CaseRecovery,
    ExecutionRecord,
    FrozenInputs,
    JobRecovery,
    LaunchIntent,
    ProcessIdentity,
    ProducedArtifacts,
    RecoveryError,
    StartupPolicy,
)
from ltspice_mcp.lib.simulator_build import SimulatorExecutable
from ltspice_mcp.lib.store import KIND_EXPERIMENT, Store, envelope
from tests.test_experiment_job import _job


def test_ltspice_template_roundtrips_without_ngspice_startup_fields(tmp_path):
    template = ArtifactDigest(tmp_path / "template.ini", "e" * 64)
    policy = StartupPolicy("ltspice-established-ini-v1", False, ini_template=template)
    restored = StartupPolicy.from_record(policy.to_record())
    assert restored == policy
    assert restored.spinit is None and restored.environment == ()
    data = policy.to_record()
    data["ini_template"]["unexpected"] = True
    with pytest.raises(RecoveryError, match="Invalid recovery record"):
        StartupPolicy.from_record(data)


def test_startup_cannot_mix_ngspice_and_ltspice_inputs(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        StartupPolicy(
            "mixed",
            True,
            spinit=ArtifactDigest(tmp_path / "spinit", "a" * 64),
            ini_template=ArtifactDigest(tmp_path / "template.ini", "b" * 64),
        )


def recovery_job(folder: Path):
    store = Store(folder)
    root = store.run_dir("exp_test_0001")
    root.mkdir(parents=True)
    deck = root / "case.cir"
    deck.write_bytes(b"* frozen\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    job = _job(folder, deck, status="failed")
    job.output_folder = root
    execution = ExecutionRecord(
        30.0,
        "configuration",
        2,
        120.0,
        2.0,
        ("ngspice", "-n"),
        SimulatorExecutable("ngspice", "a" * 64, 123, "2026-10-01"),
        "hsa",
        "linux",
        StartupPolicy("recorded-test", True),
    )
    job.recovery = JobRecovery(
        job.job_id,
        job.request_id,
        None,
        0,
        ProcessIdentity(999_999_999, "unavailable-process"),
        execution,
    )
    job.owner_pid = job.recovery.owner.pid
    case = job.cases[0]
    case.deck_sha256 = sha256_file(deck)
    case.run_token = job.job_id + "_case_0"
    inputs = FrozenInputs(
        root, ArtifactDigest(deck, case.deck_sha256), (ArtifactDigest(deck, case.deck_sha256),)
    )
    case.recovery = CaseRecovery(inputs, CaseAttempt(job.job_id, 0, case.run_token))
    return job


def test_roundtrip_preserves_grouped_execution_and_attempt_facts(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    assert case.recovery is not None
    intent = LaunchIntent(
        now(),
        case.deck_sha256,
        ArtifactDigest(job.output_folder / "executed.cir", "b" * 64),
        "logopinfo",
    )
    outputs = ProducedArtifacts(
        ArtifactDigest(job.output_folder / "case.raw", "c" * 64),
        ArtifactDigest(job.output_folder / "case.log", "d" * 64),
        now(),
    )
    case.recovery = replace(
        case.recovery, attempt=replace(case.recovery.attempt, launch=intent, outputs=outputs)
    )
    experiment_store.save_job(job)
    restored = experiment_store.load_job(job.job_id, tmp_path, own_is_alive=True)
    assert restored is not None
    assert restored.recovery == job.recovery
    assert restored.cases[0].recovery == case.recovery


@pytest.mark.parametrize(
    "change",
    [{"attempt_index": True}, {"attempt_index": "1"}, {"unexpected": 1}, {"root_request_id": ""}],
)
def test_record_decode_refuses_coercion_and_unknown_fields(tmp_path, change):
    recovery = recovery_job(tmp_path).recovery
    assert recovery is not None
    record = recovery.to_record()
    record.update(change)
    with pytest.raises(RecoveryError) as error:
        JobRecovery.from_record(record)
    assert error.value.code == "recovery_record_invalid"


def test_old_records_remain_readable_without_recovery_claim(tmp_path):
    job = recovery_job(tmp_path)
    data = experiment_store.serialize_job(job)
    data.pop("recovery", None)
    for case in data["cases"]:
        case.pop("recovery", None)
    data["store_version"] = 3
    from ltspice_mcp.lib.store import atomic_write_json

    atomic_write_json(job.store_path, data)
    restored = experiment_store.load_job(job.job_id, tmp_path)
    assert restored is not None
    assert restored.recovery is None
    assert restored.cases[0].recovery is None


def test_old_version_cannot_smuggle_a_recovery_claim(tmp_path):
    job = recovery_job(tmp_path)
    data = experiment_store.serialize_job(job)
    data["store_version"] = 3
    from ltspice_mcp.lib.store import atomic_write_json

    atomic_write_json(job.store_path, data)
    assert experiment_store.load_job(job.job_id, tmp_path) is None


def test_reused_case_counts_production_without_a_child_submission(tmp_path):
    job = recovery_job(tmp_path)
    case = job.cases[0]
    assert case.recovery is not None
    case.status = "produced"
    case.submitted_at = now()
    case.recovery = replace(case.recovery, attempt=replace(case.recovery.attempt, reused=True))
    job.completeness.recount(job.cases)
    assert job.completeness.submitted == 0
    assert job.completeness.produced == 1
    assert job.completeness.reused == 1


def test_retention_refuses_before_discovery_index_deletion(tmp_path):
    job = recovery_job(tmp_path)
    experiment_store.save_job(job)
    indexes = experiment_store.register_circuits(job, tmp_path)
    store = Store(tmp_path)
    experiment_store.save_request_index(
        request_id=job.request_id,
        fingerprint=job.fingerprint,
        canonicalizer_version=job.canonicalizer_version,
        job_id=job.job_id,
        working_dir=tmp_path,
    )
    before = {
        p: p.read_bytes() for p in [job.store_path, *indexes, store.request_index(job.request_id)]
    }
    with pytest.raises(RecoveryError) as error:
        experiment_store.delete_job(job, tmp_path)
    assert error.value.code == "recovery_retained"
    assert {p: p.read_bytes() for p in before} == before


def test_strict_restart_never_promotes_a_raw_header_only(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    job.status = "running"
    job.owner_pid = 999_999_999
    case = job.cases[0]
    case.status = "running"
    raw = job.output_folder / (case.run_token + ".raw")
    raw.write_bytes(
        b"Title: partial\nPlotname: Operating Point\nFlags: real\n"
        b"No. Variables: 1\nNo. Points: 1\nVariables:\n0\tv(in)\tvoltage\n"
    )
    (job.output_folder / (case.run_token + ".log")).write_bytes(b"ok\n")
    experiment_store.save_job(job)
    restored = experiment_store.load_job(job.job_id, tmp_path)
    assert restored is not None
    assert restored.cases[0].status == "failed"
    assert restored.cases[0].failure_code == "server_restarted"


def test_store_version_and_journal_kind_are_single_envelope(tmp_path):
    from ltspice_mcp.lib.store import KIND_RECOVERY_JOURNAL, STORE_VERSION

    assert STORE_VERSION == 4
    assert envelope(KIND_RECOVERY_JOURNAL)["store_version"] == 4
    assert envelope(KIND_EXPERIMENT)["store_version"] == 4
    store = Store(tmp_path)
    assert store.recovery_journal("request/x") != store.recovery_journal("request_x")
    assert store.recovery_journal("request/x").parent.is_relative_to(store.experiments_dir)
    assert store.recovery_lock("request/x").parent == store.root / "locks"


def test_journal_candidate_decodes_without_reconciling(tmp_path):
    from ltspice_mcp.lib.store import OwnerLiveness

    job = recovery_job(tmp_path)
    job.status = "running"
    data = experiment_store.serialize_job(job)
    restored = experiment_store.deserialize_job(data, job.store_path, liveness=OwnerLiveness.ALIVE)
    assert restored.status == "running"
    assert restored.restart_reconciled is False
    assert restored.observations == []


@pytest.mark.parametrize("nested", ["owner", "executable", "native_policy"])
def test_nested_records_refuse_unknown_fields(tmp_path, nested):
    from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY

    job = recovery_job(tmp_path)
    assert job.recovery is not None
    job.recovery = replace(
        job.recovery, execution=replace(job.recovery.execution, native_policy=LAUNCH_POLICY)
    )
    data = job.recovery.to_record()
    target = data["owner"] if nested == "owner" else data["execution"][nested]
    target["unknown"] = "ignored silently before strict decoding"
    with pytest.raises(RecoveryError):
        JobRecovery.from_record(data)


def test_native_policy_is_persisted_independently_of_ordinary_mode(tmp_path):
    from ltspice_mcp.lib.pdk_native import LAUNCH_POLICY

    job = recovery_job(tmp_path)
    assert job.recovery is not None
    execution = replace(job.recovery.execution, ngbehavior="ps", native_policy=LAUNCH_POLICY)
    restored = ExecutionRecord.from_record(execution.to_record())
    assert restored == execution
    assert restored.ngbehavior == "ps"
    assert restored.native_policy is not None
    assert restored.native_policy.ngbehavior == "hsa"


def test_carried_failed_cases_do_not_count_child_submissions_or_reuse(tmp_path):
    job = recovery_job(tmp_path)
    case = job.cases[0]
    case.status = "failed"
    case.submitted_at = now()
    job.completeness.recount(job.cases, execution_job_id="exp_child")
    assert job.completeness.submitted == 0
    assert job.completeness.failed == 1
    assert job.completeness.reused == 0


def test_stale_ordinary_object_cannot_delete_a_journal_committed_root(tmp_path):
    from ltspice_mcp.lib.store import KIND_RECOVERY_JOURNAL, atomic_write_json

    job = recovery_job(tmp_path)
    job.recovery = None
    job.cases[0].recovery = None
    experiment_store.save_job(job)
    indexes = experiment_store.register_circuits(job, tmp_path)
    journal = Store(tmp_path).recovery_journal(job.request_id)
    atomic_write_json(journal, envelope(KIND_RECOVERY_JOURNAL))
    with pytest.raises(RecoveryError):
        experiment_store.delete_job(job, tmp_path)
    assert job.store_path.exists()
    assert all(path.exists() for path in indexes)


@pytest.mark.parametrize("field", ["staged_deck", "deck_sha256", "run_token", "owner_pid"])
def test_record_refuses_disagreement_with_frozen_execution_fields(tmp_path, field):
    from ltspice_mcp.lib.store import OwnerLiveness

    job = recovery_job(tmp_path)
    data = experiment_store.serialize_job(job)
    if field == "owner_pid":
        data["pid"] = 1234
    else:
        data["cases"][0][field] = "changed"
    with pytest.raises(RecoveryError) as error:
        experiment_store.deserialize_job(data, job.store_path, liveness=OwnerLiveness.ALIVE)
    assert error.value.code == "recovery_record_invalid"
