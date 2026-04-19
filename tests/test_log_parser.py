"""Unit tests for log_parser.py — pure parsing, no simulator required."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_parser import (
    extract_error_context,
    extract_log_diagnostics,
    extract_missing_refs,
    parse_fourier_data,
    parse_measurements,
    parse_success_summary,
)


class TestExtractMissingRefs:
    def test_missing_model_quoted_name(self, tmp_path: Path):
        log = tmp_path / "missing_model.log"
        log.write_text(
            'Error on line 2 : s1 n003 n001 n002 0 sw Unable to find '
            'definition of model "sw"\n'
        )
        assert extract_missing_refs(log) == ["sw"]

    def test_missing_model_dialog_variant(self, tmp_path: Path):
        log = tmp_path / "missing_model.log"
        log.write_text("Can't find definition of model \"NMOS_3v3\"\n")
        assert extract_missing_refs(log) == ["NMOS_3v3"]

    def test_unknown_subcircuit_last_token(self, tmp_path: Path):
        log = tmp_path / "missing_subckt.log"
        log.write_text(
            "Fatal Error: Unknown subcircuit called in: xu1 n004 n001 vcc 0 lm741\n"
        )
        assert extract_missing_refs(log) == ["lm741"]

    def test_dedupes_repeated_refs(self, tmp_path: Path):
        log = tmp_path / "dupes.log"
        log.write_text(
            'Error on line 2 : s1 n003 n001 0 sw Unable to find definition of model "sw"\n'
            'Error on line 3 : s2 n004 n002 0 sw Unable to find definition of model "sw"\n'
        )
        assert extract_missing_refs(log) == ["sw"]

    def test_both_kinds_in_same_log(self, tmp_path: Path):
        log = tmp_path / "both.log"
        log.write_text(
            'Error on line 2 : s1 n1 n2 n3 0 sw Unable to find definition of model "sw"\n'
            "Fatal Error: Unknown subcircuit called in: xu1 n1 n2 n3 lm741\n"
        )
        assert set(extract_missing_refs(log)) == {"sw", "lm741"}

    def test_clean_log_returns_empty(self, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text("Total elapsed time: 0.01 seconds.\n")
        assert extract_missing_refs(log) == []

    def test_missing_file_returns_empty(self, tmp_path: Path):
        assert extract_missing_refs(tmp_path / "nope.log") == []


class TestExtractLogDiagnostics:
    def test_missing_file_returns_empty(self, tmp_path: Path):
        result = extract_log_diagnostics(tmp_path / "nope.log")
        assert result == {"warnings": [], "errors": []}

    def test_empty_file(self, tmp_path: Path):
        log = tmp_path / "empty.log"
        log.write_text("")
        result = extract_log_diagnostics(log)
        assert result == {"warnings": [], "errors": []}

    def test_filepath_line_error_with_caret(self, tmp_path: Path):
        log = tmp_path / "caret.log"
        log.write_text(
            "Some preamble\n"
            "/tmp/foo.cir(12): syntax error in .meas\n"
            ".meas TRAN bad WHEN bogus\n"
            "                ^^^^^\n"
            "Continued log\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        block = result["errors"][0]
        assert "/tmp/foo.cir(12)" in block
        assert "^^^" in block

    def test_fatal_error(self, tmp_path: Path):
        log = tmp_path / "fatal.log"
        log.write_text("OK line\nFatal Error: missing model NMOS_3V3\nMore text\n")
        result = extract_log_diagnostics(log)
        assert any("Fatal Error" in e for e in result["errors"])

    def test_error_on_line(self, tmp_path: Path):
        log = tmp_path / "errline.log"
        log.write_text("Error on line 42 : some message\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        assert "Error on line 42" in result["errors"][0]

    def test_warning_collected(self, tmp_path: Path):
        log = tmp_path / "warn.log"
        log.write_text("Warning: deprecated syntax\nWARNING: also caught\n")
        result = extract_log_diagnostics(log)
        assert len(result["warnings"]) == 2

    def test_bare_singular_matrix(self, tmp_path: Path):
        log = tmp_path / "sing.log"
        log.write_text("Time step too small\nsingular matrix\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 2


class TestExtractErrorContext:
    def test_missing_file(self, tmp_path: Path):
        assert extract_error_context(tmp_path / "nope.log") == "(Log file not found)"

    def test_empty_file(self, tmp_path: Path):
        log = tmp_path / "empty.log"
        log.write_text("")
        assert extract_error_context(log) == "(Empty log file)"

    def test_no_errors_returns_last_lines(self, tmp_path: Path):
        log = tmp_path / "ok.log"
        log.write_text("\n".join(f"line {i}" for i in range(50)))
        result = extract_error_context(log, max_lines=5)
        # Should include last 5 lines plus a continuation marker
        assert "..." in result
        assert "line 49" in result
        assert "line 0" not in result

    def test_short_file_no_errors(self, tmp_path: Path):
        log = tmp_path / "short.log"
        log.write_text("only one line")
        assert "only one line" in extract_error_context(log)

    def test_with_error_returns_context(self, tmp_path: Path):
        log = tmp_path / "err.log"
        lines = ["line 0", "line 1", "line 2", "Error: bad", "line 4", "line 5", "line 6", "line 7"]
        log.write_text("\n".join(lines))
        result = extract_error_context(log, max_lines=20)
        assert "Error: bad" in result
        assert "line 1" in result

    def test_multiple_errors_truncates(self, tmp_path: Path):
        log = tmp_path / "multi.log"
        lines = ["fatal: a"] + [f"line {i}" for i in range(20)] + ["error: b"]
        log.write_text("\n".join(lines))
        result = extract_error_context(log, max_lines=5)
        assert "fatal" in result.lower()

    def test_read_failure_returns_error_string(self, tmp_path: Path, monkeypatch):
        log = tmp_path / "bad.log"
        log.write_text("hi")

        def boom(*a, **k):
            raise OSError("disk error")

        monkeypatch.setattr(Path, "read_text", boom)
        result = extract_error_context(log)
        assert "Error reading log file" in result


class TestParseSuccessSummary:
    def test_missing_raw_graceful(self, tmp_path: Path):
        # Both files missing — should still return the dict structure
        result = parse_success_summary(
            tmp_path / "missing.raw", tmp_path / "missing.log", duration=1.5
        )
        assert result["duration"] == 1.5
        assert result["sim_type"] == "Unknown"
        assert result["trace_names"] == []
        assert result["step_count"] == 1

    def test_missing_log_with_invalid_raw(self, tmp_path: Path):
        raw = tmp_path / "x.raw"
        raw.write_bytes(b"not a real raw file")
        log = tmp_path / "x.log"
        log.write_text("Warning: heads up\n")
        result = parse_success_summary(raw, log, duration=2.0)
        assert result["duration"] == 2.0
        # log warnings should be collected
        assert any("heads up" in w for w in result["warnings"])


class TestParseMeasurements:
    def test_invalid_log_raises(self, tmp_path: Path):
        with pytest.raises(ResultError):
            parse_measurements(tmp_path / "nope.log")

    def test_no_measurements_returns_empty(self, tmp_path: Path):
        log = tmp_path / "nomeas.log"
        log.write_text(
            "Circuit: * test\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "Date: today\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert result["measurements"] == {}
        assert result["step_count"] == 0

    def test_no_measurements_with_errors(self, tmp_path: Path):
        log = tmp_path / "noerr.log"
        log.write_text(
            "Circuit: * test\n"
            "Fatal Error: missing model XYZ\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert result["measurements"] == {}
        assert "errors" in result


class TestParseFourierData:
    def test_invalid_log_returns_empty(self, tmp_path: Path):
        # parse_fourier_data wraps errors in graceful degradation
        result = parse_fourier_data(tmp_path / "nope.log")
        assert result == []

    def test_log_without_fourier(self, tmp_path: Path):
        log = tmp_path / "nofour.log"
        log.write_text(
            "Circuit: * test\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "Date: today\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_fourier_data(log)
        assert result == []


class TestParseMeasurementsValid:
    def test_with_simple_meas(self, tmp_path: Path):
        log = tmp_path / "meas.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "fc: mag(v(out))=0.707 AT 1591.5\n"
            "Date: today\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert "fc" in result["measurements"] or "fc_at" in result["measurements"]
        assert result["step_count"] >= 1
