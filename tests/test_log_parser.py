"""Unit tests for log_parser.py — pure parsing, no simulator required."""

from pathlib import Path

import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_parser import (
    count_op_iterations,
    extract_error_context,
    extract_log_diagnostics,
    extract_missing_refs,
    parse_fourier_data,
    parse_measurements,
    parse_step_iterations,
    parse_success_summary,
    parse_temperatures,
    read_log_text,
)


class TestExtractMissingRefs:
    def test_missing_model_quoted_name(self, tmp_path: Path):
        log = tmp_path / "missing_model.log"
        log.write_text(
            'Error on line 2 : s1 n003 n001 n002 0 sw Unable to find definition of model "sw"\n'
        )
        assert extract_missing_refs(log) == ["sw"]

    def test_missing_model_dialog_variant(self, tmp_path: Path):
        log = tmp_path / "missing_model.log"
        log.write_text('Can\'t find definition of model "NMOS_3v3"\n')
        assert extract_missing_refs(log) == ["NMOS_3v3"]

    def test_ngspice_undefined_model_unquoted(self, tmp_path: Path):
        # ngspice phrases an unresolved model reference differently (no quotes).
        log = tmp_path / "ngspice.log"
        log.write_text("Error: undefined model 2n2222\n")
        assert extract_missing_refs(log) == ["2n2222"]

    def test_unknown_subcircuit_last_token(self, tmp_path: Path):
        log = tmp_path / "missing_subckt.log"
        log.write_text("Fatal Error: Unknown subcircuit called in: xu1 n004 n001 vcc 0 lm741\n")
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
        assert result == {"warnings": [], "errors": [], "meas_errors": []}

    def test_ngspice_convergence_failure_classified_as_error(self, tmp_path: Path):
        # ngspice prints convergence FAILURES under a "Warning:" prefix.
        # They must surface as ERRORS (the run produced no usable data), not be
        # downgraded to warnings. A transient "singular matrix" note (recoverable
        # via stepping) must stay a warning.
        log = tmp_path / "ng_conv.log"
        log.write_text(
            "Warning: singular matrix:  check nodes out and 0\n"
            "Warning: gmin stepping failed\n"
            "source stepping failed\n"
            "doAnalyses: iteration limit reached\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        result = extract_log_diagnostics(log)
        errs = " | ".join(result["errors"]).lower()
        assert "gmin stepping failed" in errs
        assert "source stepping failed" in errs
        assert "iteration limit reached" in errs
        # transient singular-matrix note is not terminal -> stays a warning
        assert any("singular matrix" in w.lower() for w in result["warnings"])
        assert not any("singular matrix" in e.lower() for e in result["errors"])

    def test_ltspice_op_ladder_intermediate_failure_not_error_when_converged(self, tmp_path: Path):
        # LTspice's OP solver escalates Direct Newton -> Gmin -> Source stepping
        # -> pseudo-transient, printing "<method> ... failed to find operating
        # point" for each rung it abandons and a success line when one converges.
        # The intermediate "stepping failed" rungs are benign and must NOT be
        # relayed as errors (analysis tools echo errors[] into their warnings[]).
        log = tmp_path / "op_ladder.log"
        log.write_text(
            "Direct Newton iteration failed to find operating point.\n"
            "Gmin stepping failed to find operating point.\n"
            "Source stepping failed to find operating point.\n"
            "Pseudo Transient succeeded at 234.875 ms.\n"
            "Total elapsed time: 0.5 seconds.\n"
        )
        result = extract_log_diagnostics(log)
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_ltspice_stepping_success_line_suppresses_earlier_failure(self, tmp_path: Path):
        # "<method> stepping succeeded in finding operating point" also means the
        # bias point was found, so the earlier stepping-failed rung is benign.
        log = tmp_path / "op_gmin_ok.log"
        log.write_text(
            "Gmin stepping failed to find operating point.\n"
            "Source stepping succeeded in finding operating point.\n"
        )
        assert extract_log_diagnostics(log)["errors"] == []

    def test_stepping_failure_stays_error_without_a_success_line(self, tmp_path: Path):
        # The ngspice no-data case: gmin/source stepping fail and nothing later
        # converges -> still terminal, still classified as errors.
        log = tmp_path / "op_fail.log"
        log.write_text(
            "Gmin stepping failed to find operating point.\n"
            "Source stepping failed to find operating point.\n"
        )
        errs = " | ".join(extract_log_diagnostics(log)["errors"]).lower()
        assert "gmin stepping failed" in errs
        assert "source stepping failed" in errs

    def test_stepped_op_converged_step_does_not_mask_later_failed_step(self, tmp_path: Path):
        # Stepped .op: two steps converge directly, a third genuinely fails via
        # gmin/source stepping with no recovery. The converged steps must NOT
        # suppress the failed step (the check is scoped per solve block, not
        # whole-log) — otherwise a partially-failed sweep looks clean.
        log = tmp_path / "stepped_op_tail_fail.log"
        log.write_text(
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Direct Newton iteration failed to find operating point.\n"
            "Gmin stepping failed to find operating point.\n"
            "Source stepping failed to find operating point.\n"
        )
        errs = " | ".join(extract_log_diagnostics(log)["errors"]).lower()
        assert "gmin stepping failed" in errs
        assert "source stepping failed" in errs

    def test_stepped_op_failed_step_before_a_converged_step_still_errors(self, tmp_path: Path):
        # Failing step FIRST, then a step that converges directly. The later
        # step's "Direct Newton iteration succeeded" starts a new block and must
        # not rescue the earlier failed block.
        log = tmp_path / "stepped_op_head_fail.log"
        log.write_text(
            "Direct Newton iteration failed to find operating point.\n"
            "Gmin stepping failed to find operating point.\n"
            "Source stepping failed to find operating point.\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
        )
        errs = " | ".join(extract_log_diagnostics(log)["errors"]).lower()
        assert "gmin stepping failed" in errs
        assert "source stepping failed" in errs

    def test_stepping_failure_rescued_within_block_by_stepping_success(self, tmp_path: Path):
        # A block where gmin fails but source stepping succeeds -> benign, no error.
        log = tmp_path / "gmin_fail_source_ok.log"
        log.write_text(
            "Direct Newton iteration failed to find operating point.\n"
            "Gmin stepping failed to find operating point.\n"
            "Source stepping succeeded in finding operating point.\n"
        )
        assert extract_log_diagnostics(log)["errors"] == []

    def test_empty_file(self, tmp_path: Path):
        log = tmp_path / "empty.log"
        log.write_text("")
        result = extract_log_diagnostics(log)
        assert result == {"warnings": [], "errors": [], "meas_errors": []}

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

    def test_bare_error_prefix_collected(self, tmp_path: Path):
        """A bare ``ERROR:`` line (e.g. LTspice floating node) must
        land in errors[] — without this a failed-physics run reads as success."""
        log = tmp_path / "floating.log"
        log.write_text(
            "Circuit: * test\n"
            "ERROR: Node flt is floating and connected to current source I1\n"
            "Total elapsed time: 0.02 seconds.\n"
        )
        result = extract_log_diagnostics(log)
        assert any("floating" in e for e in result["errors"])

    def test_bare_error_colon_ngspice(self, tmp_path: Path):
        log = tmp_path / "parse.log"
        log.write_text("Error: circuit not parsed.\n")
        result = extract_log_diagnostics(log)
        assert any("circuit not parsed" in e for e in result["errors"])

    def test_error_on_line_not_double_counted(self, tmp_path: Path):
        """``Error on line N`` is caught by the specific rule, not also by the
        new bare ``^Error:`` rule."""
        log = tmp_path / "errline2.log"
        log.write_text("Error on line 7 : bad token\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1

    def test_ngspice_warning_double_dash(self, tmp_path: Path):
        log = tmp_path / "wdd.log"
        log.write_text('Warning -- Version not specified on line "level=49"\n')
        result = extract_log_diagnostics(log)
        assert any("Version not specified" in w for w in result["warnings"])

    def test_ngspice_fourier_ignored_surfaced(self, tmp_path: Path):
        """The ngspice .fourier-skipped note must become a visible warning."""
        log = tmp_path / "four.log"
        log.write_text(".fourier line ignored since rawfile was produced.\n")
        result = extract_log_diagnostics(log)
        assert any("fourier" in w.lower() for w in result["warnings"])

    def test_bare_singular_matrix(self, tmp_path: Path):
        log = tmp_path / "sing.log"
        log.write_text("Time step too small\nsingular matrix\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 2

    def test_meas_error_with_vdb_suggestion(self, tmp_path: Path):
        """vdb() in .MEAS should produce a structured meas_error with a
        suggestion pointing at mag()/filter_metrics."""
        log = tmp_path / "meas.log"
        log.write_text(
            "/tmp/x.cir(9): No such function defined.\n.meas AC fc_3dB WHEN vdb(out)=-3\n^^^\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["meas_errors"]) == 1
        me = result["meas_errors"][0]
        assert me["directive"].startswith(".meas")
        assert "vdb" in me["directive"]
        assert me["suggestion"] is not None
        assert "mag" in me["suggestion"].lower()
        # The same error is also present in the generic errors list.
        assert len(result["errors"]) == 1

    def test_meas_error_without_known_pattern(self, tmp_path: Path):
        """A .MEAS error that doesn't match a validator rule still gets
        captured in meas_errors but with suggestion=None."""
        log = tmp_path / "meas2.log"
        log.write_text(
            "/tmp/x.cir(7): unrecognized .meas form\n.meas AC bogus FUNNYCLAUSE V(out)\n^^^\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["meas_errors"]) == 1
        assert result["meas_errors"][0]["suggestion"] is None

    def test_non_meas_error_not_in_meas_errors(self, tmp_path: Path):
        """Component-level errors don't show up as .MEAS errors."""
        log = tmp_path / "comp.log"
        log.write_text("/tmp/x.cir(3): bad component value\nR1 in out abc\n         ^^^\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        assert result["meas_errors"] == []


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
        lines = [
            "line 0",
            "line 1",
            "line 2",
            "Error: bad",
            "line 4",
            "line 5",
            "line 6",
            "line 7",
        ]
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

        monkeypatch.setattr(Path, "read_bytes", boom)
        result = extract_error_context(log)
        assert "Error reading log file" in result

    def test_convergence_failure_excerpt_keeps_failing_node_tail(self, tmp_path: Path):
        """An EARLY benign keyword line (e.g. a "Missing parameter" warning)
        must not swallow the budget — the convergence-abort tail an LTspice
        run writes at the very END (failing node, "Last Node Voltages",
        timestep abort) names the actual failure and has to survive."""
        log = tmp_path / "conv.log"
        lines = (
            ["WARNING: Missing parameter foo"]
            + [f"progress line {i}" for i in range(40)]
            + [
                "Analysis: timestep too small",
                "Last Node Voltages:",
                'trouble with node "n002"',
                "Fatal Error: Iteration limit reached. time step too small",
            ]
        )
        log.write_text("\n".join(lines))
        result = extract_error_context(log, max_lines=20)
        # The tail (which names the real failure) must be present, not dropped
        # in favor of only the early benign "Missing parameter" hit.
        assert "trouble with node" in result
        assert "timestep too small" in result

    def test_bare_timestep_line_is_error_anchor(self, tmp_path: Path):
        """A bare one-word ``Timestep too small`` line (no co-occurring
        "convergence"/"error" word) is recognized as an error anchor, so it
        lands in the excerpt rather than being missed entirely."""
        log = tmp_path / "timestep.log"
        lines = [f"progress line {i}" for i in range(30)] + ["Timestep too small"]
        log.write_text("\n".join(lines))
        result = extract_error_context(log, max_lines=20)
        assert "Timestep too small" in result


class TestParseSuccessSummary:
    def test_missing_raw_graceful(self, tmp_path: Path):
        # Both files missing — should still return the dict structure
        result = parse_success_summary(
            tmp_path / "missing.raw", tmp_path / "missing.log", duration=1.5
        )
        assert result["duration"] == 1.5
        assert result["sim_type"] == "Unknown"
        assert result["signals"] == []
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
            "Circuit: * test\nFatal Error: missing model XYZ\nTotal elapsed time: 0.001 seconds.\n"
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
        assert "fc" in result["measurements"]
        # ``_at`` metadata should be folded into the parent entry, not surfaced
        # as its own measurement.
        assert "fc_at" not in result["measurements"]
        entry = result["measurements"]["fc"]
        assert entry["values"] == [0.707] or entry["values"] == [0.707000000000]
        assert entry.get("at") == pytest.approx(1591.5)
        assert result["step_count"] >= 1

    def test_when_crossing_with_padded_at_clause_backfilled(self, tmp_path: Path):
        # A transient WHEN line padded with a double space before AT —
        # ``tcross: V(out)=0.5  AT 0.000693``. spicelib's ` at ` pattern is
        # literal single-space, so it captures the trigger level (0.5) but
        # drops the crossing time. We re-extract and backfill ``at``.
        log = tmp_path / "when.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "tcross: V(out)=0.5  AT 0.000693147672285\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        entry = result["measurements"]["tcross"]
        # The value stays the trigger level (mirrors how AC WHEN reports it);
        # the crossing time lands in ``at``.
        assert entry["values"] == [0.5]
        assert entry.get("at") == pytest.approx(0.000693147672285)

    def test_window_from_to_not_misread_as_at(self, tmp_path: Path):
        # A FROM/TO windowed measurement has no AT clause; the backfill must
        # not invent one from the trailing TO number.
        log = tmp_path / "rms.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "vrms: RMS(v(out))=1.41109 FROM 0 TO 0.001\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        entry = result["measurements"]["vrms"]
        assert entry.get("at") is None
        assert entry.get("range_to") == pytest.approx(0.001)


class TestParseMeasurementsFourierNan:
    def test_nan_thd_does_not_crash(self, tmp_path: Path):
        """``Total Harmonic Distortion: -nan%`` used to crash the
        spicelib reader with ``'NoneType' object has no attribute 'group'``,
        wiping out any .MEAS results in the same log."""
        log = tmp_path / "nan_four.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "Fourier components of V(out)\n"
            "N-Period=1\n"
            "DC component:0\n"
            "\n"
            "Harmonic\tFrequency\t Fourier \tNormalized\t Phase  \tNormalized\n"
            " Number \t  [Hz]   \tComponent\t Component\t[degree]\tNormalized Phase [deg]\n"
            "    1   \t 1.000e+03\t 0.000e+00\t-nan      \t    0.00°\t    0.00°\n"
            "Total Harmonic Distortion:   -nan%\n"
            "\n"
            "vrms_late: RMS(V(out) )=0 FROM 0.03 TO 0.05\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        # Should NOT raise — the sanitizer should let .MEAS still parse.
        result = parse_measurements(log)
        # The .MEAS line is present in the log, so the parser should surface it.
        assert "vrms_late" in result["measurements"]
        entry = result["measurements"]["vrms_late"]
        assert entry["values"] == [0.0]
        assert entry.get("range_from") == pytest.approx(0.03)
        assert entry.get("range_to") == pytest.approx(0.05)


class TestParseMeasurementsFailed:
    """FAIL'ed .MEAS entries used to be silently absent. Spicelib's
    get_measure_names() filters them out, so without text-log extraction
    the user can't tell "did not trigger" from "did not parse"."""

    def test_failed_measurement_surfaces_as_null(self, tmp_path: Path):
        log = tmp_path / "failed.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "v_avg: AVG(v(out))=0.5 FROM 0 TO 1\n"
            'Measurement "tr_rise" FAIL\'ed\n'
            "Date: today\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        # Successful measurement still parses.
        assert "v_avg" in result["measurements"]
        # FAIL'ed measurement surfaces with a None value, not silently absent.
        assert "tr_rise" in result["measurements"]
        assert result["measurements"]["tr_rise"]["values"] == [None]
        assert "tr_rise" in result["failed_measurements"]

    def test_failed_measurement_dedupes_across_steps(self, tmp_path: Path):
        log = tmp_path / "failed_steps.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            'Measurement "tr_rise" FAIL\'ed\n'
            'Measurement "tr_rise" FAIL\'ed\n'
            'Measurement "tr_rise" FAIL\'ed\n'
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        # Three FAIL lines (one per step) collapse to a single entry.
        assert result["failed_measurements"] == ["tr_rise"]

    def test_no_failed_measurements_returns_empty_list(self, tmp_path: Path):
        log = tmp_path / "clean.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "v_avg: AVG(v(out))=0.5 FROM 0 TO 1\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert result["failed_measurements"] == []

    def test_only_failed_measurements_still_surface(self, tmp_path: Path):
        # Smoke-test exposed: when EVERY .meas FAIL'ed, spicelib's
        # get_measure_names() returns empty and the early-return path
        # would have dropped the FAIL'ed names from ``measurements``.
        log = tmp_path / "all_failed.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            'Measurement "tr_rise" FAIL\'ed\n'
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert "tr_rise" in result["measurements"]
        assert result["measurements"]["tr_rise"]["values"] == [None]
        assert result["failed_measurements"] == ["tr_rise"]


class TestParseMeasurementsFromTo:
    def test_from_to_keys_folded(self, tmp_path: Path):
        log = tmp_path / "fromto.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "vmax_late: MAX(V(out) )=9.81 FROM 0.04 TO 0.06\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = parse_measurements(log)
        assert set(result["measurements"]) == {"vmax_late"}
        entry = result["measurements"]["vmax_late"]
        assert entry["values"][0] == pytest.approx(9.81)
        assert entry.get("range_from") == pytest.approx(0.04)
        assert entry.get("range_to") == pytest.approx(0.06)


class TestParseStepIterations:
    def test_single_param_per_line(self, tmp_path: Path):
        """``.step param X list ...`` writes ``.step x=val`` per
        iteration. The .raw header doesn't carry the param mapping, so
        ``step_get`` walks the log instead.
        """
        log = tmp_path / "step.log"
        log.write_text(
            "LTspice 26.0 for Windows\n"
            ".step rval=100\n"
            ".step rval=1000\n"
            ".step rval=10000\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        iters = parse_step_iterations(log)
        assert iters == [{"rval": 100.0}, {"rval": 1000.0}, {"rval": 10000.0}]

    def test_multiple_params_per_line(self, tmp_path: Path):
        # LTspice writes already-evaluated float numbers in .step lines,
        # not engineering-notation tokens like "1k". So the parser only
        # accepts plain floats — use those in the fixture.
        log = tmp_path / "step.log"
        log.write_text(".step rval=1000, cval=1e-08\n.step rval=2000, cval=2e-08\n")
        iters = parse_step_iterations(log)
        assert iters == [
            {"rval": 1000.0, "cval": 1e-08},
            {"rval": 2000.0, "cval": 2e-08},
        ]

    def test_no_step_lines(self, tmp_path: Path):
        log = tmp_path / "nostep.log"
        log.write_text("LTspice 26.0\nTotal elapsed time: 0.01 seconds.\n")
        assert parse_step_iterations(log) == []

    def test_missing_log_returns_empty(self, tmp_path: Path):
        assert parse_step_iterations(tmp_path / "nope.log") == []

    def test_text_kw_avoids_second_read(self, tmp_path: Path):
        """Pre-read content can be passed via ``text=`` so callers reading
        the log multiple times don't trigger N read syscalls."""
        log = tmp_path / "step.log"
        log.write_text(".step rval=10\n.step rval=20\n")
        text = read_log_text(log)
        assert parse_step_iterations(text=text) == [{"rval": 10.0}, {"rval": 20.0}]


class TestCountOpIterations:
    def test_real_ltspice_op_log(self, tmp_path: Path):
        """real stepped .op logs DON'T write ``.step name=val``
        markers — only the Newton-iteration message. Counting those is the
        only reliable signal that the bias point ran multiple times."""
        log = tmp_path / "stepped_op.log"
        log.write_text(
            "LTspice 26.0 for Windows\nsolver = Normal\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Total elapsed time: 0.05 seconds.\n"
        )
        assert count_op_iterations(log) == 3

    def test_unstepped_op_returns_one(self, tmp_path: Path):
        log = tmp_path / "single_op.log"
        log.write_text(
            "LTspice 26.0\n"
            "Direct Newton iteration succeeded in finding operating point.\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        assert count_op_iterations(log) == 1

    def test_no_op_line_returns_zero(self, tmp_path: Path):
        log = tmp_path / "no_op.log"
        log.write_text("LTspice 26.0\nTotal elapsed time: 0.01 seconds.\n")
        assert count_op_iterations(log) == 0

    def test_missing_log_returns_zero(self, tmp_path: Path):
        assert count_op_iterations(tmp_path / "nope.log") == 0


class TestStepTempDegreeStripping:
    """Log step parser strips trailing degree symbol."""

    def test_temp_axis_strips_degree(self, tmp_path: Path):
        log = tmp_path / "step_temp.log"
        log.write_text("Circuit: foo\n.step temp=-40°\n.step temp=27°\n.step temp=125°\n")
        steps = parse_step_iterations(log)
        # Without the ° fix, the value capture would include '°' and
        # parse_value would fail downstream — the row would be dropped.
        assert len(steps) == 3
        assert steps[0]["temp"] == -40.0
        assert steps[1]["temp"] == 27.0
        assert steps[2]["temp"] == 125.0


class TestParseTemperatures:
    """temp/tnom surfaced from the sim log for simulation_summary."""

    def test_ltspice_form(self):
        text = "Circuit: * test\n\ntnom = 27\ntemp = 27\nDirect Newton iteration\n"
        assert parse_temperatures(text=text) == (27.0, 27.0)

    def test_ltspice_nondefault_temp(self):
        text = "temp = -40\ntnom = 27\n"
        assert parse_temperatures(text=text) == (-40.0, 27.0)

    def test_ngspice_combined_line(self):
        text = "Doing analysis at TEMP = 85.000000 and TNOM = 27.000000\n"
        assert parse_temperatures(text=text) == (85.0, 27.0)

    def test_absent(self):
        assert parse_temperatures(text="No temperature here\n") == (None, None)

    def test_step_directive_not_matched(self):
        # A `.step temp=...` line must not be read as the run temperature —
        # the anchored ^temp match avoids the leading-dot directive form.
        text = ".step temp=-40 85 5\nsome other line\n"
        assert parse_temperatures(text=text) == (None, None)


class TestLogEncodingRecovery:
    """Modern LTspice writes UTF-16 logs; Windows-authored logs carry cp1252
    bytes. read_log_text must sniff the encoding so step/temperature parsing
    recovers the data — a platform-default UTF-8 read garbles a UTF-16 log into
    NUL-interleaved text and the step scan finds nothing (the "no temperature
    steps" failure)."""

    def test_utf16le_step_temp_recovered(self, tmp_path: Path):
        log = tmp_path / "u16.log"
        body = (
            "LTspice 26.0 for Windows\n"
            ".step temp=-40\n.step temp=27\n.step temp=85\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        log.write_bytes(body.encode("utf-16-le"))
        assert parse_step_iterations(log) == [
            {"temp": -40.0},
            {"temp": 27.0},
            {"temp": 85.0},
        ]

    def test_utf16le_bom_temp_header_recovered(self, tmp_path: Path):
        log = tmp_path / "u16bom.log"
        log.write_bytes(b"\xff\xfe" + "temp = 55\ntnom = 27\n".encode("utf-16-le"))
        assert parse_temperatures(log) == (55.0, 27.0)

    def test_utf16_recovery_vs_plain_read(self, tmp_path: Path):
        # The regression itself: the old platform-default decode misses the
        # step; the sniffing reader recovers it.
        log = tmp_path / "u16.log"
        log.write_bytes(".step temp=27\n".encode("utf-16-le"))
        assert parse_step_iterations(text=log.read_text(errors="replace")) == []
        assert parse_step_iterations(log) == [{"temp": 27.0}]

    def test_cp1252_degree_byte_in_step_value_recovered(self, tmp_path: Path):
        # A cp1252 degree byte (0xB0) glued to a step value: a UTF-8 read turns
        # it into U+FFFD (which the KV regex does NOT exclude, so float() fails
        # and the step drops); cp1252 decoding yields a real ° the regex strips.
        log = tmp_path / "cp1252.log"
        log.write_bytes("LTspice 26.0\n.step temp=-40\xb0\n".encode("cp1252"))
        assert parse_step_iterations(log) == [{"temp": -40.0}]
