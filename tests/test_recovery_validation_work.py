"""Repeated model copies share parsing, never file or include authority."""

from pathlib import Path

import pytest

from ltspice_mcp.lib import deck_staging, experiment_inputs
from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.recovery_records import RecoveryError
from tests.test_recovery_records import recovery_job


def _copies(tmp_path: Path, body: bytes, *, copies: int = 4):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    paths = []
    for index in range(copies):
        library = job.output_folder / str(index) / "model.inc"
        library.parent.mkdir()
        library.write_bytes(body)
        local = library.with_name("local.inc")
        local.write_text(f".param r={index + 1}\n", encoding="utf-8")
        for path in (library, local):
            source.manifest.append(ManifestEntry(path, sha256_file(path), True, False, path))
        paths.append(library)
    case.staged_deck.write_text(
        "* repeated libraries\n"
        + "".join(f".include {index}/model.inc\n" for index in range(copies))
        + ".op\n.end\n",
        encoding="utf-8",
    )
    case.deck_sha256 = sha256_file(case.staged_deck)
    return job, paths


def test_identical_copies_parse_once_but_each_file_is_verified(tmp_path, monkeypatch):
    body = b".include local.inc\nRcopy in 0 {r}\n"
    job, libraries = _copies(tmp_path, body)
    assert job.output_folder is not None
    real_lex = experiment_inputs.lex
    library_parses = 0

    def count_parse(text):
        nonlocal library_parses
        if text == body.decode():
            library_parses += 1
        return real_lex(text)

    monkeypatch.setattr(experiment_inputs, "lex", count_parse)
    monkeypatch.setattr(deck_staging, "lex", count_parse)
    frozen = experiment_inputs.capture_case_inputs(
        job.cases[0], job.sources[0], lineage_root=job.output_folder
    )
    assert len(frozen.files) == 9
    assert {item.path for item in frozen.files}.issuperset(libraries)
    assert library_parses == 1

    libraries[-1].write_bytes(body + b"* changed copy\n")
    with pytest.raises(RecoveryError) as error:
        experiment_inputs.verify_case_inputs(frozen)
    assert error.value.code == "recovery_input_drift"

    libraries[-1].write_bytes(body)
    experiment_inputs.verify_case_inputs(frozen)
    assert library_parses == 2  # A new verification rechecks and reparses current bytes.
    libraries[-1].unlink()
    with pytest.raises(RecoveryError) as error:
        experiment_inputs.verify_case_inputs(frozen)
    assert error.value.code == "recovery_artifact_missing"


def test_equal_include_text_checks_each_directory(tmp_path):
    job, libraries = _copies(tmp_path, b".include local.inc\n", copies=2)
    assert job.output_folder is not None
    unrecorded = libraries[-1].with_name("local.inc")
    source = job.sources[0]
    source.manifest = [entry for entry in source.manifest if entry.staged_path != unrecorded]
    with pytest.raises(RecoveryError) as error:
        experiment_inputs.capture_case_inputs(job.cases[0], source, lineage_root=job.output_folder)
    assert error.value.code == "recovery_closure_incomplete"


def test_equal_analysis_copies_still_count_separately(tmp_path):
    job, _ = _copies(tmp_path, b".op\n", copies=2)
    assert job.output_folder is not None
    case = job.cases[0]
    case.staged_deck.write_bytes(case.staged_deck.read_bytes().replace(b".op\n", b""))
    case.deck_sha256 = sha256_file(case.staged_deck)
    with pytest.raises(RecoveryError) as error:
        experiment_inputs.capture_case_inputs(
            case, job.sources[0], lineage_root=job.output_folder, seeded=True
        )
    assert error.value.code == "recovery_seed_analysis_unsupported"


def test_identical_root_and_include_bytes_keep_distinct_title_rules(tmp_path):
    job = recovery_job(tmp_path)
    assert job.output_folder is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    body = b".param draw={rand()}\nV1 in 0 1\nR1 in 0 1k\n.op\n.end\n"
    case.staged_deck.write_bytes(body)
    case.deck_sha256 = sha256_file(case.staged_deck)
    duplicate = job.output_folder / "model.inc"
    duplicate.write_bytes(body)
    source.manifest.append(
        ManifestEntry(duplicate, sha256_file(duplicate), True, False, duplicate)
    )
    with pytest.raises(RecoveryError) as error:
        experiment_inputs.capture_case_inputs(case, source, lineage_root=job.output_folder)
    assert error.value.code == "recovery_random_unsupported"
