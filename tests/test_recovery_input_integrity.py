"""Frozen input authority and retained lineage boundaries without simulator launches."""

import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import (
    capture_case_inputs,
    prepare_startup,
    verify_case_inputs,
)
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.pdk_native import ArtifactDigest, NativePaths, prepare_launch
from ltspice_mcp.lib.recovery_records import CaseAttempt, CaseRecovery, RecoveryError
from ltspice_mcp.lib.store import Store, StoreError
from ltspice_mcp.lib.variations import (
    AssignVariation,
    CircuitDeck,
    MaterializedCase,
    expand_variations,
    materialize_variants,
)
from tests.conftest import symlink_or_skip
from tests.test_native_records import _record
from tests.test_recovery_records import recovery_job


@pytest.fixture(autouse=True)
def _no_subprocesses(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Input integrity checks must not launch processes")

    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.mark.parametrize(
    "title",
    [
        b"A minimal resistor circuit",
        b"rand() in a title",
        b".control",
        b".ends",
        b".include absent.inc",
    ],
)
def test_root_title_does_not_execute_or_change_closure(tmp_path, title):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    content = title + b"\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n"
    case.staged_deck.write_bytes(content)
    case.deck_sha256 = sha256_file(case.staged_deck)

    captured = capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    assert captured.electrical.sha256 == case.deck_sha256
    assert len(captured.files) == 1
    assert case.staged_deck.read_bytes() == content
    verify_case_inputs(captured)


@pytest.mark.parametrize(
    ("fragment", "code"),
    [
        (b"A1 in 0 external_module\n", "recovery_external_module"),
        (b".param r={rand()}\n", "recovery_random_unsupported"),
        (b"+ orphan\n", "recovery_syntax_unsupported"),
    ],
)
def test_included_fragment_retains_its_first_card(tmp_path, fragment, code):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    dependency = job.output_folder / "dependency.inc"
    dependency.write_bytes(fragment)
    case.staged_deck.write_bytes(b"* root\n.include dependency.inc\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source.manifest.append(ManifestEntry(dependency, "a" * 64, True, False, dependency))

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder)

    assert error.value.code == code


def test_root_title_normalization_retains_lexer_warnings(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    case.staged_deck.write_bytes(b"A circuit title\n+ orphan\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    assert error.value.code == "recovery_syntax_unsupported"


def test_root_title_normalization_retains_trailing_lexer_warnings(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    case.recovery = None
    case.staged_deck.write_bytes(b"* title\n.op\n.end\n.ends\n")
    case.deck_sha256 = sha256_file(case.staged_deck)

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    assert error.value.code == "recovery_syntax_unsupported"


def test_materialized_root_title_does_not_validate_authoring_deck_as_a_fragment(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    content = "A circuit title\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n"
    case.staged_deck.write_bytes(content.encode())
    deck = CircuitDeck(case.circuit, case.staged_deck, content)
    expanded = expand_variations([deck], [AssignVariation(kind="assign", assign={"R1": ["2k"]})])
    (variant,) = materialize_variants(deck, expanded, case.staged_deck.parent)
    case.case_id = variant.case_id
    case.staged_deck = variant.path
    case.deck_sha256 = variant.sha256

    captured = capture_case_inputs(
        case,
        source,
        lineage_root=job.output_folder,
        materialized=variant,
    )

    assert source.staged_deck not in {artifact.path for artifact in captured.files}
    assert source.staged_deck.read_bytes() == content.encode()
    assert b"R1 in 0 2k" in captured.electrical.path.read_bytes()


@pytest.mark.parametrize("changed", ["dependency", "electrical"])
def test_recapture_refuses_materialized_hash_conflicts(tmp_path, changed):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    dependency = job.output_folder / "dependency.inc"
    dependency.write_bytes(b".param r=1000\n")
    case.staged_deck.write_bytes(b"* root\n.include dependency.inc\nR1 in 0 {r}\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source.manifest.append(ManifestEntry(dependency, "a" * 64, True, False, dependency))
    frozen = capture_case_inputs(case, source, lineage_root=job.output_folder)
    case.recovery = CaseRecovery(frozen, CaseAttempt(job.job_id, 0, case.run_token))
    if changed == "dependency":
        dependency.write_bytes(b".param r=2000\n")
    else:
        case.staged_deck.write_bytes(
            b"* changed\n.include dependency.inc\nR1 in 0 2k\n.op\n.end\n"
        )
        case.deck_sha256 = sha256_file(case.staged_deck)
    variant = MaterializedCase(
        case.case_id,
        case.circuit,
        0,
        case.staged_deck,
        "",
        case.deck_sha256,
        {},
        file_digests=((case.staged_deck, case.deck_sha256), (dependency, sha256_file(dependency))),
    )

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, source, lineage_root=job.output_folder, materialized=variant)

    assert error.value.code == "recovery_input_drift"
    assert case.recovery.inputs is frozen


def _native_case(tmp_path, *, title=b"* frozen"):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    case.staged_deck.write_bytes(title + b"\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    dependency = job.output_folder / "model.spice"
    dependency.write_bytes(b".model m nmos\n")
    record = _record(job.output_folder, prepared=True)
    assert record.sample is not None
    record.prepared = None
    record.pending_dependencies = (
        ArtifactDigest(dependency, sha256_file(dependency), "models/model.spice"),
    )
    case.native_statistics = record
    frozen = capture_case_inputs(case, source, lineage_root=job.output_folder)
    case.recovery = CaseRecovery(frozen, CaseAttempt(job.job_id, 0, case.run_token))
    root = job.output_folder
    paths = NativePaths(
        root,
        root / "native.input.cir",
        root / "native.setup.cir",
        root / "native.cir",
        root / "native.raw",
        root / "native.log",
    )
    record.prepared = prepare_launch(
        record.sample,
        paths=paths,
        token="native",
        electrical_bytes=case.staged_deck.read_bytes(),
        dependencies=record.pending_dependencies,
    )
    record.pending_dependencies = ()
    return job


def test_native_recapture_adds_prepared_files_and_normalizes_their_title(tmp_path):
    job = _native_case(tmp_path, title=b"A native circuit")
    assert job.output_folder is not None
    case = job.cases[0]
    assert case.recovery is not None
    assert case.native_statistics is not None
    frozen = case.recovery.inputs
    prepared = case.native_statistics.prepared
    assert prepared is not None

    captured = capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    by_path = {artifact.path: artifact.sha256 for artifact in captured.files}
    assert all(by_path[artifact.path] == artifact.sha256 for artifact in frozen.files)
    input_path = prepared.paths.electrical_input
    driver_path = prepared.paths.prepared_driver
    assert isinstance(input_path, Path)
    assert isinstance(driver_path, Path)
    assert by_path[input_path] == prepared.input_sha256
    assert by_path[driver_path] == prepared.driver_sha256
    verify_case_inputs(captured, native=True)


def test_native_recapture_refuses_dependency_hash_conflict(tmp_path):
    job = _native_case(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    assert case.native_statistics is not None
    prepared = case.native_statistics.prepared
    assert prepared is not None
    dependency = prepared.dependencies[0]
    dependency.path.write_bytes(b".model m nmos level=1\n")
    case.native_statistics.prepared = replace(
        prepared,
        dependencies=(replace(dependency, sha256=sha256_file(dependency.path)),),
    )

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    assert error.value.code == "recovery_input_drift"


@pytest.mark.parametrize("changed", ["electrical_input", "prepared_driver"])
def test_native_recapture_refuses_prepared_file_hash_conflict(tmp_path, changed):
    job = _native_case(tmp_path)
    assert job.output_folder is not None
    case = job.cases[0]
    assert case.recovery is not None
    assert case.native_statistics is not None
    captured = capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)
    case.recovery = replace(case.recovery, inputs=captured)
    prepared = case.native_statistics.prepared
    assert prepared is not None
    path = getattr(prepared.paths, changed)
    if changed == "electrical_input":
        path.write_bytes(path.read_bytes().replace(b"1k", b"2k"))
        case.native_statistics.prepared = replace(prepared, input_sha256=sha256_file(path))
    else:
        path.write_bytes(path.read_bytes().replace(b"setseed 123", b"setseed 124"))
        case.native_statistics.prepared = replace(prepared, driver_sha256=sha256_file(path))

    with pytest.raises(RecoveryError) as error:
        capture_case_inputs(case, job.sources[0], lineage_root=job.output_folder)

    assert error.value.code == "recovery_input_drift"


@pytest.mark.parametrize("operation", ["validate", "startup"])
def test_lineage_refuses_peer_redirect(tmp_path, operation):
    if operation == "startup" and sys.platform not in {"linux", "win32"}:
        pytest.skip("controlled startup is verified on Linux and Windows")
    store = Store(tmp_path)
    original = store.run_dir("exp_original")
    peer = store.run_dir("exp_peer")
    peer.mkdir(parents=True)
    symlink_or_skip(original, peer, target_is_directory=True)

    if operation == "startup":
        with pytest.raises(StoreError):
            prepare_startup(store, "exp_original")
    else:
        with pytest.raises(StoreError):
            store.lineage_run_dir("exp_original", original)

    assert list(peer.iterdir()) == []


def test_lineage_preserves_canonical_configured_runs_root(tmp_path):
    store = Store(tmp_path)
    runs = store.runs_root()
    target = tmp_path / "canonical-runs"
    target.mkdir()
    runs.parent.mkdir(parents=True)
    symlink_or_skip(runs, target, target_is_directory=True)
    recorded = store.run_dir("exp_root")

    assert store.lineage_run_dir("exp_root", recorded) == target / "exp_root"
    (target / "exp_root").mkdir()
    assert store.lineage_run_dir("exp_root", target / "exp_root") == target / "exp_root"


@pytest.mark.skipif(sys.platform != "win32", reason="junctions are a native Windows facility")
def test_lineage_refuses_peer_junction(tmp_path):
    if TYPE_CHECKING:

        def create_junction(target: str, link: str) -> None: ...

    else:
        import _winapi

        create_junction = _winapi.CreateJunction

    store = Store(tmp_path)
    original = store.run_dir("exp_original")
    peer = store.run_dir("exp_peer")
    peer.mkdir(parents=True)
    create_junction(str(peer), str(original))

    with pytest.raises(StoreError):
        store.lineage_run_dir("exp_original", original)

    assert list(peer.iterdir()) == []
