"""Exact staged input capture and drift refusal through real filesystem paths."""

import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from ltspice_mcp.lib import now
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import (
    capture_case_inputs,
    capture_produced_artifacts,
    prepare_startup,
    verify_case_inputs,
    verify_produced_artifacts,
    verify_startup,
)
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.pdk_native import ArtifactDigest, NativePaths, prepare_launch
from ltspice_mcp.lib.recovery_records import RecoveryError
from ltspice_mcp.lib.store import Store, StoreError
from ltspice_mcp.lib.variations import MaterializedCase
from tests.conftest import needs_controlled_startup, symlink_or_skip
from tests.test_recovery_records import recovery_job


def test_materialized_digest_beats_original_manifest_hash(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    model = job.output_folder / "model.inc"
    model.write_bytes(b".param resistance=2000\n")
    case.staged_deck.write_bytes(b"* case\n.include model.inc\nR1 in 0 {resistance}\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source.manifest.append(ManifestEntry(tmp_path / "authoring.inc", "a" * 64, True, False, model))
    variant = MaterializedCase(
        case.case_id,
        case.circuit,
        0,
        case.staged_deck,
        "",
        case.deck_sha256,
        {},
        file_digests=((case.staged_deck, case.deck_sha256), (model, sha256_file(model))),
    )
    captured = capture_case_inputs(
        case, source, lineage_root=job.output_folder, materialized=variant
    )
    assert {item.path: item.sha256 for item in captured.files}[model] == sha256_file(model)
    # Editing/deleting authoring inputs never changes frozen recovery authority.
    source.path = tmp_path / "removed-authoring.cir"
    verify_case_inputs(captured)
    model.write_bytes(b".param resistance=1000\n")
    with pytest.raises(RecoveryError) as error:
        verify_case_inputs(captured)
    assert error.value.code == "recovery_input_drift"


@pytest.mark.parametrize(
    ("body", "code"),
    [
        (b".control\nrun\n.endc\n", "recovery_control_unsupported"),
        (b'V1 in 0 PWL FILE="ambient.txt"\n', "recovery_external_reader"),
        (b"B1 in 0 V=gauss(1)\n", "recovery_random_unsupported"),
        (b".param draw={rand()}\n", "recovery_random_unsupported"),
        (b"B1 in 0 V=unknown_function(1)\n", "recovery_expression_unsupported"),
        (b"A1 in 0 external_module\n", "recovery_external_module"),
        (b".load ambient.cm\n", "recovery_external_module"),
    ],
)
def test_unsupported_constructs_refuse_in_nested_include(tmp_path, body, code):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    dependency = job.output_folder / "unsafe.inc"
    dependency.write_bytes(body)
    case.staged_deck.write_bytes(b"* case\n.include unsafe.inc\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source.manifest.append(
        ManifestEntry(tmp_path / "unsafe.inc", sha256_file(dependency), True, False, dependency)
    )
    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert error.value.code == code


def test_one_refusal_names_every_construct_with_its_file_and_line(tmp_path):
    """A deck and its library fail several checks: one refusal lists each
    construct, in the file the caller wrote and on its line, so the deck is
    fixed in one pass rather than one submission per construct."""
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    frozen = job.output_folder / "vendor.lib"
    frozen.write_bytes(
        b"* vendor models\n"
        b"B1 in 0 V=vendor_fn(1)\n"
        b"A1 in 0 external_module\n"
        b"R1 in 0 1k\n"
        b"B2 in 0 V=vendor_fn(2)\n"
    )
    case.staged_deck.write_bytes(b"* case\n.include vendor.lib\n.op\n.control\nrun\n.endc\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    library = tmp_path / "models" / "vendor.lib"
    source.manifest.append(ManifestEntry(library, sha256_file(frozen), True, False, frozen))
    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)
    # The code is the first construct's: the deck's cards come before its library's.
    assert error.value.code == "recovery_control_unsupported"
    message = str(error.value)
    assert "refuses 3 constructs" in message
    assert f"(recovery_control_unsupported): {source.path} line 4: " in message
    assert (
        "function `vendor_fn`, not known to the server (recovery_expression_unsupported): "
        f"{library} line 2: B1 in 0 V=vendor_fn(1); also {library} line 5"
    ) in message
    assert f"`A1`, an XSPICE device (recovery_external_module): {library} line 3" in message
    # The frozen copy's path is the server's, not the caller's.
    assert str(frozen) not in message


@pytest.mark.parametrize(
    ("deck", "seeded"),
    [
        # Line 2 is the deck's only analysis.
        (b"* frozen\n.op\nV1 n 0 1\nR1 n 0 1k\n.end\n", True),
        # The title reads as a second analysis, but a title is only text.
        (b".tran 1n 10n\nV1 n 0 1\nR1 n 0 1k\n.op\n.end\n", True),
        # Line 2 opens a subcircuit.
        (b"* frozen\n.subckt s a\nR1 a 0 1\n.ends\nV1 n 0 1\nX1 n s\n.op\n.end\n", False),
    ],
)
def test_a_root_decks_title_is_its_first_line_and_nothing_more(tmp_path, deck, seeded):
    """Both readers of a frozen root deck's cards, the seeded analysis count
    and the refusal of a deck the lexer warns about, skip line 1 as the title
    and read line 2 once, as the deck's first card."""
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    case.staged_deck.write_bytes(deck)
    case.deck_sha256 = sha256_file(case.staged_deck)
    captured = capture_case_inputs(
        case, job.sources[0], lineage_root=job.output_folder, seeded=seeded
    )
    verify_case_inputs(captured, seeded=seeded)


def test_refuses_unrecorded_include_and_live_manifest(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    (job.output_folder / "unrecorded.inc").write_bytes(b".param x=1\n")
    case.staged_deck.write_bytes(b"* test\n.include unrecorded.inc\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert error.value.code == "recovery_closure_incomplete"
    source.manifest.append(
        ManifestEntry(
            tmp_path / "live.inc", "a" * 64, False, True, reason="outside the allowed roots"
        )
    )
    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert error.value.code == "recovery_live_dependency"
    # The refusal names the file read in place, and why it was not copied.
    assert f"{tmp_path / 'live.inc'} (outside the allowed roots)" in str(error.value)


def test_refuses_missing_file_and_symlink_escape(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    captured = capture_case_inputs(case, source, lineage_root=job.output_folder)
    outside = tmp_path / "outside.cir"
    outside.write_bytes(case.staged_deck.read_bytes())
    case.staged_deck.unlink()
    with pytest.raises(RecoveryError):
        verify_case_inputs(captured)
    symlink_or_skip(case.staged_deck, outside)
    with pytest.raises(RecoveryError) as error:
        verify_case_inputs(captured)
    assert error.value.code == "recovery_path_escape"


@needs_controlled_startup
def test_startup_is_an_inert_frozen_file_in_initial_run_root(tmp_path):
    store = Store(tmp_path)
    policy = prepare_startup(store, "exp_startup")
    assert policy.spinit is not None
    root = store.run_dir("exp_startup")
    assert policy.user_init_disabled is True
    assert dict(policy.environment) == {"SPICE_SCRIPTS": str(policy.spinit.path.parent)}
    assert policy.spinit.path == store.recovery_spinit("exp_startup")
    assert policy.spinit.sha256 == hashlib.sha256(policy.spinit.path.read_bytes()).hexdigest()
    verify_startup(policy, root)
    policy.spinit.path.write_bytes(b"echo unexpected\n")
    with pytest.raises(RecoveryError):
        verify_startup(policy, root)
    # Initial startup preparation must refuse to overwrite an old lineage.
    with pytest.raises(RecoveryError):
        prepare_startup(store, "exp_startup")


def test_result_snapshot_refuses_changed_or_removed_outputs(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    raw, log = job.output_folder / "x.raw", job.output_folder / "x.log"
    raw.write_bytes(b"recorded raw bytes")
    log.write_bytes(b"recorded log bytes")
    outputs = capture_produced_artifacts(raw, log, lineage_root=job.output_folder)
    assert outputs.completed_at <= now()
    verify_produced_artifacts(outputs, job.output_folder)
    raw.write_bytes(b"changed")
    with pytest.raises(RecoveryError) as error:
        verify_produced_artifacts(outputs, job.output_folder)
    assert error.value.code == "recovery_output_drift"
    raw.write_bytes(b"recorded raw bytes")
    log.unlink()
    with pytest.raises(RecoveryError):
        verify_produced_artifacts(outputs, job.output_folder)


def test_lineage_directory_refuses_outside_or_redirected_store_path(tmp_path):
    store = Store(tmp_path)
    with pytest.raises(StoreError):
        store.lineage_run_dir("exp_root", tmp_path)
    expected = store.run_dir("exp_root")
    expected.parent.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    symlink_or_skip(expected, outside, target_is_directory=True)
    with pytest.raises(StoreError):
        store.lineage_run_dir("exp_root", expected)


def test_native_retry_reuses_sample_seed_and_dependency_bytes(tmp_path):
    from ltspice_mcp.lib.native_execution import prepare_native_retry
    from tests.test_native_records import _record

    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    assert case.recovery is not None
    record = _record(job.output_folder, prepared=True)
    assert record.sample is not None
    dependency = job.output_folder / "model.spice"
    dependency.write_bytes(b".model m nmos\n")
    dependencies = (ArtifactDigest(dependency, sha256_file(dependency), "models/model.spice"),)
    initial_paths = NativePaths(
        job.output_folder,
        job.output_folder / "old.input.cir",
        job.output_folder / "old.setup.cir",
        job.output_folder / "old.cir",
        job.output_folder / "old.raw",
        job.output_folder / "old.log",
    )
    record.prepared = prepare_launch(
        record.sample,
        paths=initial_paths,
        token="old",
        electrical_bytes=case.staged_deck.read_bytes(),
        dependencies=dependencies,
    )
    old = record.prepared
    case.native_statistics = record
    case.run_token = "exp_child_case_0"
    case.recovery = replace(
        case.recovery, attempt=replace(case.recovery.attempt, run_token=case.run_token)
    )
    case.status = "queued"
    prepare_native_retry(case, store=Store(tmp_path), root_job_id=job.job_id)
    assert record.prepared.effective_seed == old.effective_seed
    assert record.prepared.sample_key == old.sample_key
    assert record.prepared.dependencies == old.dependencies
    assert record.prepared.input_sha256 == old.input_sha256
    assert record.prepared.paths.cwd == old.paths.cwd
    assert record.prepared.paths.prepared_driver != old.paths.prepared_driver
    assert Path(old.paths.prepared_driver).is_file()
    with pytest.raises(RecoveryError):
        prepare_native_retry(case, store=Store(tmp_path), root_job_id=job.job_id)


def test_native_setup_filename_does_not_authorize_caller_controls(tmp_path):
    from ltspice_mcp.lib.recovery_records import FrozenInputs

    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    unsafe = job.output_folder / "caller.setup.cir"
    unsafe.write_bytes(b".control\nshell echo ambient\n.endc\n")
    recovery = job.cases[0].recovery
    assert recovery is not None
    original = recovery.inputs
    frozen = replace(
        original, files=(*original.files, ArtifactDigest(unsafe, sha256_file(unsafe)))
    )
    assert isinstance(frozen, FrozenInputs)
    with pytest.raises(RecoveryError) as error:
        verify_case_inputs(frozen, native=True)
    assert error.value.code == "recovery_control_unsupported"


def test_second_capture_verifies_prior_hash_instead_of_rebinding_it(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    dependency = job.output_folder / "dep.inc"
    dependency.write_bytes(b".param r=1000\n")
    case.staged_deck.write_bytes(b"* case\n.include dep.inc\nR1 in 0 {r}\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source = job.sources[0]
    source.manifest.append(ManifestEntry(dependency, "a" * 64, True, False, dependency))
    case.recovery = None
    inputs = capture_case_inputs(case, source, lineage_root=job.output_folder)
    from ltspice_mcp.lib.recovery_records import CaseAttempt, CaseRecovery

    case.recovery = CaseRecovery(inputs, CaseAttempt(job.job_id, 0, case.run_token))
    dependency.write_bytes(b".param r=2000\n")
    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert error.value.code == "recovery_input_drift"


def test_sectioned_library_uses_existing_staging_resolution(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    library = job.output_folder / "corners.lib"
    library.write_bytes(
        b".lib low\n.param r=900\n.endl low\n.lib high\n.param r=1100\n.endl high\n"
    )
    case.staged_deck.write_bytes(b"* sectioned\n.lib corners.lib low\nR1 in 0 {r}\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    case.recovery = None
    source.manifest.append(ManifestEntry(library, "a" * 64, True, False, library))
    frozen = capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert library in {item.path for item in frozen.files}
    verify_case_inputs(frozen)


@pytest.mark.skipif(
    sys.platform != "linux", reason="this probe isolates ngspice adapter selection on Linux"
)
def test_ngspice_startup_helper_refuses_ltspice_without_writing(tmp_path):
    from spicelib.simulators.ltspice_simulator import LTspice

    store = Store(tmp_path)
    with pytest.raises(RecoveryError) as error:
        prepare_startup(store, "exp_ltspice", LTspice)
    assert error.value.code == "recovery_startup_unsupported"
    assert not store.root.exists()
