"""Captured log decoding: real recordings, owned scratch and honest failures."""

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
from pydantic import TypeAdapter

from ltspice_mcp.lib import log_parser
from ltspice_mcp.lib.log_decode import LogDecodeError, LogLimitError, LogLimits, decode_logs
from ltspice_mcp.lib.parser_capture import SourceFiles, capture_inputs
from tests.test_device_op_points import _LOG_WITH_BLOCK

FIXTURES = Path(__file__).parent / "fixtures"
LIMITS = LogLimits(2_000_000, 16_000, 20_000, 50_000, 2_000_000)


def test_existing_log_types_support_strict_parent_json_validation(tmp_path):
    from ltspice_mcp.lib import log_types

    assert log_types.LogDiagnostics is log_parser.LogDiagnostics
    assert log_types.MeasurementsOutput is log_parser.MeasurementsOutput
    captured, directory = capture(tmp_path, source=FIXTURES / "ltspice_tran_rc.log")
    facts = decode_logs(captured, directory, limits=LIMITS)
    for name, shape in (
        ("diagnostics", log_parser.LogDiagnostics),
        ("measurements", log_parser.MeasurementsOutput),
    ):
        value = facts[name]["value"]
        adapter = TypeAdapter(shape)
        assert adapter.validate_json(json.dumps(value), strict=True, extra="forbid") == value
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            adapter.validate_json(
                json.dumps({**value, "unexpected": 1}), strict=True, extra="forbid"
            )


def capture(tmp_path, *, text=None, console=None, source=None):
    directory = tmp_path / "worker"
    directory.mkdir()
    if text is not None:
        source = tmp_path / "source.log"
        source.write_text(text, encoding="utf-8")
    console_path = None
    if console is not None:
        console_path = tmp_path / "source.exe.log"
        console_path.write_text(console, encoding="utf-8")
    captured = capture_inputs(
        SourceFiles(log=source, console=console_path),
        directory,
        input_bytes=LIMITS.log_bytes,
        log_bytes=LIMITS.log_bytes,
    )
    return captured, directory


@pytest.mark.parametrize("name", sorted(path.name for path in FIXTURES.glob("*.log")))
def test_real_recorded_log_numeric_parity(tmp_path, name):
    source = FIXTURES / name
    expected = log_parser.parse_measurements(source)
    captured, directory = capture(tmp_path, source=source)
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["measurements"]["status"] in ("parsed", "absent")
    assert facts["measurements"]["value"] == expected
    assert facts["diagnostics"]["value"] == log_parser.extract_log_diagnostics(source)
    assert facts["steps"]["value"] == log_parser.parse_step_iterations(source)
    ambient, nominal = log_parser.parse_temperatures(source)
    assert facts["temperatures"]["value"] == {"temp_c": ambient, "tnom_c": nominal}
    assert facts["scan"] == {"complete": True, "capturedbytes": source.stat().st_size}
    assert facts["capture_facts"] == json.loads(json.dumps(asdict(captured)))
    assert json.loads(json.dumps(facts, allow_nan=False)) == facts


def test_recorded_transient_measurement_has_independent_numeric_anchor(tmp_path):
    captured, directory = capture(tmp_path, source=FIXTURES / "ltspice_tran_rc.log")
    facts = decode_logs(captured, directory, limits=LIMITS)
    entry = facts["measurements"]["value"]["measurements"]["vfinal"]
    assert entry["values"] == [0.999876166042]
    assert entry["at"] == 0.0009
    assert facts["temperatures"]["value"] == {"temp_c": 27.0, "tnom_c": 27.0}


def test_console_only_diagnostics_and_absent_sections(tmp_path):
    captured, directory = capture(tmp_path, console="Error: analysis not run\n")
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["diagnostics"]["status"] == "parsed"
    assert facts["diagnostics"]["value"]["errors"] == ["Error: analysis not run"]
    for name in ("measurements", "device_op", "steps", "op_iterations", "temperatures", "fourier"):
        assert facts[name]["status"] == "absent"
    assert facts["scan"]["complete"]


def test_captured_diagnostics_do_not_reopen_original_companion(tmp_path):
    captured, directory = capture(
        tmp_path,
        text="Circuit: example\nWarning: benign\n",
        console="Error: captured console failure\n",
    )
    (tmp_path / "source.log").unlink()
    (tmp_path / "source.exe.log").write_text("Fatal Error: changed live source\n")
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["diagnostics"]["value"]["errors"] == ["Error: captured console failure"]
    assert "changed live" not in json.dumps(facts)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("log_bytes", 8, "log_bytes"),
        ("line_bytes", 8, "line_bytes"),
        ("lines", 1, "lines"),
    ],
)
def test_caps_refuse_before_dependency_constructor(tmp_path, monkeypatch, field, value, match):
    captured, directory = capture(tmp_path, text="Circuit: example\n" + "a" * 100 + "\n")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Dependency constructed before cap refusal")

    monkeypatch.setattr(log_parser, "LTSpiceLogReader", forbidden)
    monkeypatch.setattr(log_parser, "opLogReader", forbidden)
    with pytest.raises(LogLimitError, match=match):
        decode_logs(captured, directory, limits=replace(LIMITS, **{field: value}))


@pytest.mark.parametrize("invalid", [True, 0, -1, 1.0, float("inf"), float("nan")])
def test_limits_are_explicit_finite_positive_integers(invalid):
    with pytest.raises(ValueError, match="finite positive integer"):
        replace(LIMITS, lines=invalid)


@pytest.mark.parametrize("change", ["digest", "absent-console", "binary", "malformed-encoding"])
def test_capture_refusal_precedes_dependency_reads(tmp_path, monkeypatch, change):
    captured, directory = capture(tmp_path, text="Circuit: test\n")
    if change == "digest":
        (directory / "input.log").write_text("Circuit: edit\n")
    elif change == "absent-console":
        (directory / "input.exe.log").write_text("Error: unexpected companion\n")
    else:
        (directory / "input.log").write_bytes(
            b"\x00\x01\x00" if change == "binary" else b"\xff\xfeA"
        )
        # A capture with these exact bytes is still not a valid text log.
        data = (directory / "input.log").read_bytes()
        captured = replace(
            captured,
            files=(
                replace(
                    captured.files[0],
                    size_bytes=len(data),
                    sha256=hashlib.sha256(data).hexdigest(),
                ),
            ),
        )
    monkeypatch.setattr(log_parser, "LTSpiceLogReader", lambda *_: pytest.fail("Dependency read"))
    with pytest.raises(LogDecodeError):
        decode_logs(captured, directory, limits=LIMITS)


def test_parse_failure_is_not_valid_empty_measurements(tmp_path):
    captured, directory = capture(tmp_path, text="unrecognizable malformed log\n")
    facts = decode_logs(captured, directory, limits=LIMITS)
    for name in ("measurements", "fourier"):
        assert facts[name]["status"] == "error"
        assert facts[name]["value"] is None and facts[name]["error"]["message"]
    assert facts["diagnostics"]["status"] == "parsed"
    assert facts["scan"]["complete"]


def test_malformed_measurement_block_is_section_error(tmp_path):
    captured, directory = capture(tmp_path, text="Circuit: test\nMeasurement: gain\nnot a table\n")
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["measurements"]["status"] == "error"
    assert facts["measurements"]["value"] is None


@pytest.mark.parametrize("encoding", ["utf-8", "cp1252", "utf-16-le"])
def test_device_op_parity_and_no_global_scratch(tmp_path, monkeypatch, encoding):
    source = tmp_path / "device.log"
    source.write_bytes((_LOG_WITH_BLOCK + ".step temp=-40°\n").encode(encoding))
    captured, directory = capture(tmp_path, source=source)
    monkeypatch.setattr(
        log_parser.tempfile,
        "NamedTemporaryFile",
        lambda **_: pytest.fail("Normalization escaped supplied scratch"),
    )
    facts = decode_logs(captured, directory, limits=LIMITS)
    values = facts["device_op"]["value"]
    assert facts["device_op"]["status"] == "parsed"
    assert values["@m1[gm]"] == 4.8e-4
    assert values["@m1[vth]"] == 0.5
    assert values["@m1[id]"] == 9.6e-5
    assert not list(directory.glob("normalized-*"))


def fourier_text(thd="0.014047", phd="0.000251"):
    return (
        "Circuit: * test\n\nFourier components of V(out)\nN-Period=1\nDC component:0\n\n"
        "Harmonic\tFrequency\tFourier\tNormalized\tPhase\tNormalized\n"
        "Number\t[Hz]\tComponent\tComponent\t[degree]\tPhase\n"
        "1\t1.000e+03\t8.464e-01\t1.0\t122.15°\t0.00°\n"
        "2\t2.000e+03\t7.414e-07\t8.760e-07\t177.22°\t55.07°\n"
        f"Partial Harmonic Distortion: {phd}%\nTotal Harmonic Distortion: {thd}%\n\n"
        "vrms: RMS(V(out))=0.5 FROM 0.03 TO 0.05\n"
    )


def test_fourier_numeric_parity_in_recorded_simulator_format(tmp_path):
    captured, directory = capture(tmp_path, text=fourier_text())
    facts = decode_logs(captured, directory, limits=LIMITS)
    section = facts["fourier"]
    assert section["status"] == "parsed" and section["nonfinite_count"] == 0
    entry = section["value"][0]
    assert entry["thd"] == pytest.approx(0.014047)
    assert entry["phd"] == pytest.approx(0.000251)
    assert entry["fundamental_frequency"] == 1000.0
    assert entry["harmonics"][0]["magnitude"] == 0.8464
    assert entry["harmonics"][0]["phase"] == 122.15


def test_nonfinite_facts_are_null_counted_and_never_sanitized_zero(tmp_path, monkeypatch):
    body = fourier_text("-nan", "inf") + "bad: V(out)=nan\n.step R=nan\ntemp = 1e999\n"
    captured, directory = capture(tmp_path, text=body)
    monkeypatch.setattr(
        log_parser.tempfile, "NamedTemporaryFile", lambda **_: pytest.fail("Global temp")
    )
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["fourier"]["value"][0]["thd"] is None
    assert facts["fourier"]["value"][0]["phd"] is None
    assert facts["fourier"]["nonfinite_count"] == 2
    assert facts["measurements"]["value"]["measurements"]["bad"]["values"] == [None]
    assert facts["measurements"]["nonfinite_count"] == 1
    assert facts["steps"]["value"] == [{"R": None}]
    assert facts["steps"]["nonfinite_count"] == 1
    assert facts["temperatures"]["value"]["temp_c"] is None
    assert facts["temperatures"]["nonfinite_count"] == 1
    json.dumps(facts, allow_nan=False)


@pytest.mark.parametrize(("field", "value"), [("section_entries", 2), ("metadata_bytes", 10)])
def test_output_limits_refuse_instead_of_truncating(tmp_path, field, value):
    captured, directory = capture(tmp_path, text="Circuit: test\ngain: V(out)=2\n")
    with pytest.raises(LogLimitError, match=field):
        decode_logs(captured, directory, limits=replace(LIMITS, **{field: value}))


@pytest.mark.parametrize(
    ("body", "section"),
    [
        ("Circuit: test\n.step R=1 C=bad\n", "steps"),
        (_LOG_WITH_BLOCK.replace("4.80e-04", "bad"), "device_op"),
        (_LOG_WITH_BLOCK.replace("Name:           M1", "Name:           M1 M2"), "device_op"),
        ("Circuit: test\ngain: V(out)=bad\n", "measurements"),
    ],
)
def test_partially_malformed_numeric_sections_are_errors(tmp_path, body, section):
    captured, directory = capture(tmp_path, text=body)
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts[section]["status"] == "error"
    assert facts[section]["value"] is None


def test_actual_op_attempt_success_and_final_failure_facts(tmp_path):
    body = (
        "Circuit: test\nDirect Newton iteration failed to find operating point.\n"
        "Gmin stepping failed to find operating point.\n"
        "Source stepping succeeded in finding operating point.\n"
        "Direct Newton iteration succeeded in finding operating point.\n"
        "Direct Newton iteration failed to find operating point.\n"
        "Gmin stepping failed to find operating point.\nSource stepping failed to find operating point.\n"
    )
    captured, directory = capture(tmp_path, text=body)
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["op_iterations"]["value"] == {"attempts": 3, "succeeded": 1}
    errors = facts["diagnostics"]["value"]["errors"]
    assert len(errors) == 2
    assert any("Gmin" in error for error in errors)
    assert any("Source" in error for error in errors)


def test_malformed_fourier_is_error_and_normalization_is_cleaned(tmp_path):
    captured, directory = capture(
        tmp_path, text=fourier_text().replace("8.464e-01", "not-a-number")
    )
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["fourier"]["status"] == "error"
    assert facts["fourier"]["value"] is None
    assert not list(directory.glob("normalized-*"))


def test_shared_head_only_reader_cap_is_refused_before_parsing(tmp_path, monkeypatch):
    captured, directory = capture(tmp_path, text="Circuit: example\n")
    monkeypatch.setattr(log_parser, "_LOG_READ_CAP_BYTES", 8)
    monkeypatch.setattr(log_parser, "LTSpiceLogReader", lambda *_: pytest.fail("Dependency read"))
    with pytest.raises(LogLimitError, match="Shared log reader"):
        decode_logs(captured, directory, limits=LIMITS)


def test_console_diagnostics_tail_is_in_full_snapshot(tmp_path):
    captured, directory = capture(
        tmp_path,
        text="Circuit: example\n",
        console="Note: Compatibility modes selected: hsa\n" * 40 + "Fatal Error: final failure\n",
    )
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["diagnostics"]["value"]["errors"] == ["Fatal Error: final failure"]
    assert facts["scan"]["capturedbytes"] == sum(item.size_bytes for item in captured.files)


@pytest.mark.parametrize("parameters", ["r=1 r=2", "r=1 R=2"])
def test_duplicate_step_parameter_names_are_explicit_section_errors(tmp_path, parameters):
    captured, directory = capture(tmp_path, text=f"Circuit: test\n.step {parameters}\n")
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["steps"]["status"] == "error"
    assert facts["steps"]["value"] is None
    assert "Duplicate step parameter" in facts["steps"]["error"]["message"]


def test_unique_step_parameters_keep_decimal_values_and_warnings(tmp_path):
    captured, directory = capture(
        tmp_path, text="Circuit: test\n.step r=1.25e-3 C=2.5e-9\nWarning: retained\n"
    )
    facts = decode_logs(captured, directory, limits=LIMITS)
    assert facts["steps"]["status"] == "parsed"
    assert facts["steps"]["value"] == [{"r": 0.00125, "C": 2.5e-9}]
    assert facts["diagnostics"]["value"]["warnings"] == ["Warning: retained"]
