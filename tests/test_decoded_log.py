"""Parent log facts preserve recorded numbers and reject contradictory replies."""

from __future__ import annotations

import builtins
import copy
import importlib
import json
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_decode import decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from tests.test_log_decode import FIXTURES, LIMITS, capture, fourier_text


def resident(metadata):
    from ltspice_mcp.lib.decoded_log import DecodedLog

    return DecodedLog(metadata)


@pytest.fixture
def manifest(tmp_path):
    captured, directory = capture(tmp_path, source=FIXTURES / "ltspice_tran_rc.log")
    return decode_logs(captured, directory, limits=LIMITS)


@pytest.mark.parametrize("name", sorted(path.name for path in FIXTURES.glob("*.log")))
def test_recorded_decoder_results_validate_without_numeric_changes(tmp_path, name):
    captured, directory = capture(tmp_path, source=FIXTURES / name)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    log = resident(metadata)
    assert log.as_dict() == metadata
    assert log.captured == captured
    assert log.snapshot_id == captured.cache_key(
        dialect=None, producing_dialect=None, revision="log-facts-v1"
    )
    json.dumps(log.as_dict(), allow_nan=False)


def test_constructor_and_each_accessor_are_copy_isolated(manifest):
    original = copy.deepcopy(manifest)
    log = resident(manifest)
    manifest["measurements"]["value"]["measurements"]["vfinal"]["values"][0] = 99
    manifest["capture_facts"]["files"][0]["size_bytes"] = 0
    manifest["scan"]["complete"] = False
    assert log.as_dict() == original
    section = log.section("measurements")
    section["value"]["measurements"]["vfinal"]["values"][0] = -1
    log.capture_facts["files"].clear()
    log.scan["capturedbytes"] = -1
    log.as_dict()["diagnostics"]["value"]["warnings"].append("injected")
    assert log.as_dict() == original
    with pytest.raises(FrozenInstanceError):
        log.captured.files[0].size_bytes = 0  # pyright: ignore[reportAttributeAccessIssue]
    with pytest.raises(ValueError, match="section"):
        log.section("unsupported_section")  # pyright: ignore[reportArgumentType]


def test_console_only_and_raw_only_absence_contracts(tmp_path):
    captured, directory = capture(tmp_path, console="Error: analysis not run\n")
    metadata = decode_logs(captured, directory, limits=LIMITS)
    log = resident(metadata)
    assert log.section("diagnostics")["value"]["errors"] == ["Error: analysis not run"]
    assert log.section("measurements")["value"] is None
    # RAW content is opaque to the log decoder and parent fact validation.
    raw_source = tmp_path / "source.raw"
    raw_source.write_bytes(b"opaque RAW input not decoded as log facts")
    raw_directory = tmp_path / "raw-worker"
    raw_directory.mkdir()
    raw_capture = capture_inputs(
        SourceFiles(raw=raw_source),
        raw_directory,
        input_bytes=LIMITS.log_bytes,
        log_bytes=LIMITS.log_bytes,
    )
    raw_only = decode_logs(raw_capture, raw_directory, limits=LIMITS)
    raw_log = resident(raw_only)
    assert raw_log.section("diagnostics")["status"] == "absent"
    assert raw_log.captured == raw_capture
    assert raw_log.scan == {"complete": True, "capturedbytes": 0}


@pytest.mark.parametrize(
    "body", [fourier_text("nan", "inf"), fourier_text().replace("8.464e-01", "bad")]
)
def test_fourier_nonfinite_and_errors_preserved(tmp_path, body):
    captured, directory = capture(tmp_path, text=body)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    log = resident(metadata)
    assert log.section("fourier") == metadata["fourier"]
    section = log.section("fourier")
    if section["status"] == "parsed":
        assert section["value"][0]["thd"] is None
        assert section["value"][0]["phd"] is None
        assert section["nonfinite_count"] == 2
    else:
        assert section["status"] == "error" and section["value"] is None
        section["error"]["message"] = "modified"
        assert log.section("fourier") == metadata["fourier"]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), True),
        (("version",), 2),
        (("extra",), 1),
        (("scan", "complete"), False),
        (("scan", "complete"), 1),
        (("scan", "capturedbytes"), "123"),
        (("scan", "capturedbytes"), 1),
        (("scan", "extra"), 1),
        (("capture_facts", "extra"), 1),
        (("capture_facts", "files", 0, "extra"), 1),
        (("capture_facts", "files", 0, "name"), "../input.log"),
        (("capture_facts", "files", 0, "role"), "unknown"),
        (("capture_facts", "files", 0, "sha256"), "bad"),
        (("capture_facts", "files", 0, "size_bytes"), True),
        (("capture_facts", "absent"), ["raw", "console", "log"]),
        (("measurements", "status"), "absent"),
        (("measurements", "status"), "unknown"),
        (("measurements", "nonfinite_count"), True),
        (("measurements", "nonfinite_count"), -1),
        (("measurements", "nonfinite_count"), 99),
        (("measurements", "extra"), 1),
        (("measurements", "error"), {"type": "Error", "message": "bad"}),
        (("measurements", "value", "extra"), 1),
        (("measurements", "value", "step_count"), -1),
        (("measurements", "value", "measurements", "vfinal", "extra"), 1),
        (("measurements", "value", "measurements", "vfinal", "values"), ["1"]),
        (("measurements", "value", "measurements", "vfinal", "values"), [True]),
        (("measurements", "value", "measurements", "vfinal", "values"), [float("nan")]),
        (("measurements", "value", "measurements", "vfinal", "values"), (1.0,)),
        (("diagnostics", "value", "meas_errors"), [{"directive": ".meas", "raw_block": "bad"}]),
        (("diagnostics", "nonfinite_count"), 1),
        (("steps", "status"), "parsed"),
        (("op_iterations", "value"), {"attempts": 1, "succeeded": 2}),
        (("temperatures", "status"), "absent"),
        (("fourier", "status"), "error"),
    ],
)
def test_invalid_worker_shapes_and_semantics_are_refused(manifest, path, value):
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=r".+"):
        resident(manifest)


def test_known_measurement_shapes_keep_missing_values_and_metadata_alignment(manifest):
    entry = manifest["measurements"]["value"]["measurements"]["vfinal"]
    entry["values"] = [None]
    assert resident(manifest).section("measurements")["nonfinite_count"] == 0
    entry["at"] = [0.1, 0.2]
    with pytest.raises(ValueError, match="align"):
        resident(manifest)


def test_relabelled_capture_cannot_claim_parsed_log_sections(manifest):
    manifest["capture_facts"]["files"][0].update(role="console", name="input.exe.log")
    manifest["capture_facts"]["absent"] = ["raw", "log"]
    with pytest.raises(ValueError, match="absent"):
        resident(manifest)


def test_fact_identity_changes_with_captured_bytes_and_presence(manifest):
    first = resident(manifest).snapshot_id
    changed = copy.deepcopy(manifest)
    changed["capture_facts"]["files"][0]["sha256"] = "1" * 64
    assert resident(changed).snapshot_id != first
    changed = copy.deepcopy(manifest)
    changed["capture_facts"]["files"].append(
        {"role": "console", "name": "input.exe.log", "size_bytes": 0, "sha256": "2" * 64}
    )
    changed["capture_facts"]["absent"].remove("console")
    assert resident(changed).snapshot_id != first


@pytest.mark.parametrize(
    "field", ["thd_unit", "harmonic_phase", "error_extra", "absent_nonfinite"]
)
def test_fourier_field_and_section_errors_are_strict(tmp_path, field):
    captured, directory = capture(tmp_path, text=fourier_text())
    metadata = decode_logs(captured, directory, limits=LIMITS)
    section = metadata["fourier"]
    if field == "thd_unit":
        section["value"][0]["thd_unit"] = "ratio"
    elif field == "harmonic_phase":
        section["value"][0]["harmonics"][0]["phase"] = True
    elif field == "error_extra":
        section.update(
            status="error", value=None, error={"type": "Error", "message": "bad", "extra": 1}
        )
    else:
        section.update(status="absent", value=[], nonfinite_count=1)
    with pytest.raises(ValueError, match=r".+"):
        resident(metadata)


def test_all_numeric_nonfinite_sections_preserve_null_counts(tmp_path):
    body = fourier_text("-nan", "inf") + "bad: V(out)=nan\n.step R=nan\ntemp = 1e999\n"
    captured, directory = capture(tmp_path, text=body)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    assert resident(metadata).as_dict() == metadata


def test_missing_fourier_field_is_null_without_nonfinite_count(tmp_path):
    body = fourier_text().replace("Partial Harmonic Distortion: 0.000251%\n", "")
    captured, directory = capture(tmp_path, text=body)
    section = resident(decode_logs(captured, directory, limits=LIMITS)).section("fourier")
    assert section["status"] == "parsed"
    assert section["value"][0]["phd"] is None
    assert section["nonfinite_count"] == 0


@pytest.mark.parametrize("invalid", ["cycle", "deep", "object", "keys"])
def test_non_json_or_excessively_nested_containers_are_refused(manifest, invalid):
    if invalid == "cycle":
        manifest["extra"] = manifest
    elif invalid == "deep":
        nested = []
        manifest["extra"] = nested
        for _ in range(30):
            child = []
            nested.append(child)
            nested = child
    elif invalid == "keys":
        manifest[1] = "extra"
    else:
        manifest["extra"] = object()
    with pytest.raises(ValueError, match="Log manifest"):
        resident(manifest)


def test_parent_import_and_access_do_not_import_dependency_parsers(manifest, monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("spicelib") or name in {
            "ltspice_mcp.lib.log_parser",
            "ltspice_mcp.lib.log_decode",
            "ltspice_mcp.lib.raw_parser",
        }:
            pytest.fail(f"Parent imported dependency parser: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "ltspice_mcp.lib.decoded_log", raising=False)
    monkeypatch.setattr(builtins, "__import__", guarded)
    module = importlib.import_module("ltspice_mcp.lib.decoded_log")
    log = module.DecodedLog(manifest)
    assert log.section("measurements")["value"]["measurements"]["vfinal"]["values"] == [
        0.999876166042
    ]


def test_result_cache_accounts_large_resident_backing_facts(tmp_path):
    from ltspice_mcp.lib.parsed_artifacts import ParsedArtifacts
    from ltspice_mcp.lib.result_cache import resident_size

    limits = replace(
        LIMITS,
        log_bytes=8 * 1024 * 1024,
        line_bytes=8 * 1024 * 1024,
        metadata_bytes=16 * 1024 * 1024,
    )
    source = tmp_path / "large.log"
    source.write_text(
        "Circuit: example\nWarning: " + "x" * (4 * 1024 * 1024) + "\n", encoding="utf-8"
    )
    directory = tmp_path / "large-worker"
    directory.mkdir()
    captured = capture_inputs(
        SourceFiles(log=source),
        directory,
        input_bytes=limits.log_bytes,
        log_bytes=limits.log_bytes,
    )
    log = resident(decode_logs(captured, directory, limits=limits))
    manifest_bytes = len(json.dumps(log.as_dict(), separators=(",", ":")).encode("utf-8"))
    assert manifest_bytes > 10 * 1024 * 1024
    parsed = ParsedArtifacts(snapshot_id="a" * 64, raw=None, logs=log)
    assert resident_size(parsed) >= manifest_bytes


def test_resident_log_fields_are_frozen(manifest):
    log = resident(manifest)
    with pytest.raises(FrozenInstanceError):
        log._snapshot_id = "changed"  # pyright: ignore[reportAttributeAccessIssue]


def test_value_returns_detached_parsed_and_absent_facts_without_relabelling(manifest, tmp_path):
    log = resident(manifest)
    before = log.as_dict()
    values = log.value("measurements")
    values["measurements"]["vfinal"]["values"][0] = 99
    assert log.value("measurements")["measurements"]["vfinal"]["values"] == [0.999876166042]
    assert log.value("steps") == []
    assert log.section("steps")["status"] == "absent"
    assert log.as_dict() == before
    console_root = tmp_path / "console-only"
    console_root.mkdir()
    captured, directory = capture(console_root, console="Warning: console only\n")
    console = resident(decode_logs(captured, directory, limits=LIMITS))
    assert console.value("measurements") is None
    assert console.value("device_op") == {}
    assert console.value("temperatures") == {"temp_c": None, "tnom_c": None}
    assert console.section("measurements")["status"] == "absent"


@pytest.mark.parametrize(
    ("body", "name"),
    [
        ("Circuit: test\ngain: V(out)=bad\n", "measurements"),
        (fourier_text().replace("8.464e-01", "not-a-number"), "fourier"),
        ("Circuit: test\n.step R=bad\n", "steps"),
    ],
)
def test_value_raises_result_error_for_actual_decoder_errors(tmp_path, body, name):
    captured, directory = capture(tmp_path, text=body)
    metadata = decode_logs(captured, directory, limits=LIMITS)
    log = resident(metadata)
    error = log.section(name)
    assert error["status"] == "error"
    with pytest.raises(ResultError) as raised:
        log.value(name)
    assert name in str(raised.value)
    assert error["error"]["message"] in str(raised.value)
    assert log.as_dict() == metadata
