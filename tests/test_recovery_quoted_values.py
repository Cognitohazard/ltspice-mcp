"""Quoted passive values retain the frozen-input reader and expression guards."""

import subprocess
from pathlib import Path

import pytest

from ltspice_mcp.lib.deck_staging import sha256_file
from ltspice_mcp.lib.experiment_inputs import capture_case_inputs, verify_case_inputs
from ltspice_mcp.lib.experiment_types import ManifestEntry
from ltspice_mcp.lib.recovery_records import RecoveryError
from tests.test_recovery_records import recovery_job


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Frozen-input validation must not launch a process")

    monkeypatch.setattr(subprocess, "Popen", refuse)


def _capture(folder: Path, fragment: bytes, *, seeded: bool):
    job = recovery_job(folder)
    root = job.output_folder
    assert root is not None
    case, source = job.cases[0], job.sources[0]
    case.recovery = None
    dependency = root / "model.inc"
    dependency.write_bytes(fragment)
    case.staged_deck.write_bytes(b"* Frozen expressions\n.include model.inc\n.op\n.end\n")
    case.deck_sha256 = sha256_file(case.staged_deck)
    source.manifest.append(
        ManifestEntry(dependency, sha256_file(dependency), True, False, dependency)
    )
    return capture_case_inputs(case, source, lineage_root=root, seeded=seeded)


@pytest.mark.parametrize("seeded", [False, True])
def test_pinned_resistor_expression_and_coefficients_are_captured_unchanged(tmp_path, seeded):
    fragment = (
        b".subckt resistor d d1 w=5\n"
        b".param rdiff=8900 rdiff_tc1=2.5e-3 rdiff_tc2=2.2e-6\n"
        b"rldd d d1 '(1/w)*rdiff' tc1 = 'rdiff_tc1' tc2 = 'rdiff_tc2'\n"
        b".ends resistor\n"
    )
    captured = _capture(tmp_path, fragment, seeded=seeded)
    verify_case_inputs(captured, seeded=seeded)
    dependency = next(item for item in captured.files if item.path.name == "model.inc")
    assert dependency.path.read_bytes() == fragment
    assert dependency.sha256 == sha256_file(dependency.path)


@pytest.mark.parametrize("seeded", [False, True])
def test_pinned_capacitor_expression_is_captured_unchanged(tmp_path, seeded):
    fragment = b".subckt capacitor p n\n.param czero=1p\nC0 p n 'czero'\n.ends capacitor\n"
    captured = _capture(tmp_path, fragment, seeded=seeded)
    verify_case_inputs(captured, seeded=seeded)
    dependency = next(item for item in captured.files if item.path.name == "model.inc")
    assert dependency.path.read_bytes() == fragment
    assert dependency.sha256 == sha256_file(dependency.path)


@pytest.mark.parametrize(
    "fragment",
    [
        b"R1 'n' 0 '1k'\n",
        b"R1 n '0' '1k'\n",
        b'R1 n 0 "1k"\n',
        b"R1 n 0 '1k' '2k'\n",
        b"R1 n 0 '1k' extra\n",
        b"R1 n 0 '1k' tc1=1e-3 'extra'\n",
        b"R1 n '1k'\n",
        b'C1 n 0 "capacitance"\n',
        b"L1 n 0 'inductance'\n",
        b"M1 d g s b 'model'\n",
        b"X1 n 0 'subcircuit'\n",
        b"V1 n 0 'wave.dat'\n",
    ],
)
def test_other_quoted_positions_remain_refused(tmp_path, fragment):
    with pytest.raises(RecoveryError) as error:
        _capture(tmp_path, fragment, seeded=True)
    assert error.value.code == "recovery_external_reader"


@pytest.mark.parametrize("prefix", ["R", "C"])
@pytest.mark.parametrize("value", ["''", "' '"])
def test_empty_quoted_passive_values_remain_refused(tmp_path, prefix, value):
    with pytest.raises(RecoveryError) as error:
        _capture(tmp_path, f"{prefix}1 n 0 {value}\n".encode(), seeded=True)
    assert error.value.code == "recovery_external_reader"


@pytest.mark.parametrize(
    ("fragment", "code"),
    [
        (b"R1 n 0 '1k' file='resistance.dat'\n", "recovery_external_reader"),
        (b"R1 n 0 '1k' sfile='resistance.dat'\n", "recovery_external_reader"),
        (b"R1 n 0 '1k' pwlfile='resistance.dat'\n", "recovery_external_reader"),
        (b'V1 n 0 PWL("wave.dat")\n', "recovery_external_reader"),
        (b"V1 n 0 PWL('wave.dat')\n", "recovery_external_reader"),
        (b"R1 n 0 'unknown_function(1)'\n", "recovery_expression_unsupported"),
        (b"R1 n 0 '1k' tc1='unknown_function(1)'\n", "recovery_expression_unsupported"),
        (b"R1 n 0 'rand()'\n", "recovery_random_unsupported"),
        (b"R1 n 0 'trrandom(1 1n)'\n", "recovery_random_unsupported"),
        (b"R1 n 0 '1k' tc1='sgauss(0)'\n", "recovery_random_unsupported"),
        (b"R1 n 0 '1k'\n.control\nrun\n.endc\n", "recovery_control_unsupported"),
        (b"R1 n 0 '1k'\n.load ambient.cm\n", "recovery_external_module"),
        (b"R1 n 0 '1k'\nA1 n 0 external_module\n", "recovery_external_module"),
        (b"R1 n 0 '1k'\n.unknown_directive value\n", "recovery_directive_unsupported"),
    ],
)
def test_reader_and_later_guards_still_refuse(tmp_path, fragment, code):
    with pytest.raises(RecoveryError) as error:
        _capture(tmp_path, fragment, seeded=True)
    assert error.value.code == code


def test_quoted_resistor_randomness_still_requires_the_recorded_seed_contract(tmp_path):
    fragment = b"R1 n 0 'agauss(1000,1,1)'\n"
    captured = _capture(tmp_path, fragment, seeded=True)
    verify_case_inputs(captured, seeded=True)
    with pytest.raises(RecoveryError) as error:
        verify_case_inputs(captured)
    assert error.value.code == "recovery_random_unsupported"
