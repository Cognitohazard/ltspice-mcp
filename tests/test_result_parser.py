"""Unit tests for result_parser — the most algorithmic module.

Tests pure-logic functions with synthetic numpy arrays, and parsing
functions with real .raw/.log files produced by LTspice integration tests.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.log_parser import extract_log_diagnostics, parse_measurements
from ltspice_mcp.lib.raw_parser import (
    compute_ac_bandwidth_metrics,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    is_ac_analysis,
    is_dc_analysis,
    query_point_value,
    sample_to_dict,
)
from ltspice_mcp.state import SessionState
from tests.conftest import stage_recorded_fixture

# --- Helpers to build mock RawRead objects ---


def _make_raw_mock(
    trace_names: list[str],
    axis: np.ndarray,
    waves: dict[str, np.ndarray],
    plotname: str = "Transient Analysis",
    steps: list[int] | None = None,
) -> MagicMock:
    """Build a mock RawRead with controllable data."""
    raw = MagicMock()
    raw.get_raw_property.return_value = plotname
    raw.get_trace_names.return_value = trace_names
    raw.get_steps.return_value = steps if steps is not None else [0]
    raw.get_axis.return_value = axis

    def get_wave(name, step=0):
        return waves[name]

    raw.get_wave = get_wave
    return raw


class TestOpSteppingFailureRawGate:
    """An OP 'gmin/source stepping failed' error is a recoverable ladder rung.
    The log-only converged-check keys on LTspice's success wording, so an
    ngspice run that recovered via an unannounced fallback leaves a false hard
    error — gated on raw validity: finite node data demotes it to a warning, a
    rail-pinned/NaN raw keeps it an error, and always-terminal failures don't
    qualify at all."""

    def _summary(self, tmp_path: Path, node_wave: np.ndarray, phrase: str) -> dict:
        from ltspice_mcp.lib.raw_parser import build_simulation_summary

        log = tmp_path / "op.log"
        # No recognized LTspice success line follows, so extract_log_diagnostics
        # classifies the phrase as an error before the raw gate runs.
        log.write_text(f"ngspice-42\n{phrase}\n")
        axis = np.array([0.0, 1e-3, 2e-3])
        raw = _make_raw_mock(["time", "v(out)"], axis, {"time": axis, "v(out)": node_wave})
        return build_simulation_summary(raw, log)

    def test_finite_data_demotes_to_warning(self, tmp_path: Path):
        s = self._summary(tmp_path, np.array([1.0, 1.01, 0.99]), "gmin stepping failed")
        assert "errors" not in s
        assert any("gmin stepping failed" in w for w in s.get("warnings", []))

    def test_railed_data_keeps_error(self, tmp_path: Path):
        s = self._summary(tmp_path, np.array([1e30, 1e30, 1e30]), "source stepping failed")
        assert any("source stepping failed" in e for e in s.get("errors", []))
        assert not any("source stepping failed" in w for w in s.get("warnings", []))

    def test_iteration_limit_never_demoted(self, tmp_path: Path):
        # Always-terminal — not a stepping-failure candidate even with clean data.
        s = self._summary(tmp_path, np.array([1.0, 1.0, 1.0]), "iteration limit reached")
        assert any("iteration limit" in e for e in s.get("errors", []))

    def test_stepped_op_keeps_error_for_later_step(self, tmp_path: Path):
        # A stepped .op solves the bias point per step but LTspice writes only
        # step 0 to the .raw. Step 0's finite data can't clear a stepping failure
        # that belongs to a later step the raw never carries — keep it an error.
        from ltspice_mcp.lib.raw_parser import build_simulation_summary

        log = tmp_path / "op.log"
        log.write_text(".step v1=1\n.step v1=2\nngspice-42\ngmin stepping failed\n")
        # A real .op raw has no axis — get_axis raises "does not have an axis".
        raw = _make_raw_mock(
            ["v(out)"], np.array([0.0]), {"v(out)": np.array([1.0])}, plotname="Operating Point"
        )
        raw.get_axis.side_effect = Exception("This RAW file does not have an axis.")
        s = build_simulation_summary(raw, log)
        assert any("gmin stepping failed" in e for e in s.get("errors", []))
        assert any("Stepped .op detected" in w for w in s.get("warnings", []))

    def test_single_step_op_still_demotes(self, tmp_path: Path):
        # An unstepped .op (one bias point) with finite data and no success line
        # still demotes — the guard must not over-suppress the single-block case.
        from ltspice_mcp.lib.raw_parser import build_simulation_summary

        log = tmp_path / "op.log"
        log.write_text("ngspice-42\ngmin stepping failed\n")
        raw = _make_raw_mock(
            ["v(out)"], np.array([0.0]), {"v(out)": np.array([1.0])}, plotname="Operating Point"
        )
        raw.get_axis.side_effect = Exception("This RAW file does not have an axis.")
        s = build_simulation_summary(raw, log)
        assert "errors" not in s
        assert any("gmin stepping failed" in w for w in s.get("warnings", []))


class TestDetectSimType:
    def test_transient(self):
        raw = _make_raw_mock([], np.array([]), {}, plotname="Transient Analysis")
        assert detect_sim_type(raw) == "Transient Analysis"

    def test_ac(self):
        raw = _make_raw_mock([], np.array([]), {}, plotname="AC Analysis")
        assert detect_sim_type(raw) == "AC Analysis"

    def test_fallback_on_error(self):
        raw = MagicMock()
        raw.get_raw_property.side_effect = Exception("no property")
        assert detect_sim_type(raw) == "Unknown"


class TestIsAcAnalysis:
    def test_ac_variants(self):
        assert is_ac_analysis("AC Analysis") is True
        assert is_ac_analysis("ac analysis") is True

    def test_non_ac(self):
        assert is_ac_analysis("Transient Analysis") is False
        assert is_ac_analysis("DC sweep") is False


class TestIsDcAnalysis:
    def test_dc_analysis_variants(self):
        assert is_dc_analysis("DC transfer characteristic") is True
        assert is_dc_analysis("DC sweep") is True

    def test_dc_analysis_non_dc(self):
        assert is_dc_analysis("Transient Analysis") is False
        assert is_dc_analysis("AC Analysis") is False
        assert is_dc_analysis("Noise Spectral Density") is False

    def test_dc_analysis_word_boundary(self):
        # "dc" appearing only inside a word (substring present, no word
        # boundary) must not match — these discriminate \bDC\b from `"dc" in s`.
        assert is_dc_analysis("abcdc") is False
        assert is_dc_analysis("adc") is False


class TestGetStepCount:
    def test_single_step(self):
        raw = _make_raw_mock([], np.array([]), {}, steps=[0])
        assert get_step_count(raw) == 1

    def test_multi_step(self):
        raw = _make_raw_mock([], np.array([]), {}, steps=[0, 1, 2])
        assert get_step_count(raw) == 3

    def test_error_returns_1(self):
        raw = MagicMock()
        raw.get_steps.side_effect = Exception("no steps")
        assert get_step_count(raw) == 1


class TestQueryPointValue:
    def test_exact_match(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0, 40.0])
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 2.0)
        assert result["actual_x"] == pytest.approx(2.0)
        assert result["value"] == pytest.approx(30.0)
        assert result["trace"] == "V(out)"

    def test_nearest_neighbor(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0, 40.0])
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 1.3)
        assert result["actual_x"] == pytest.approx(1.0)
        assert result["value"] == pytest.approx(20.0)

    def test_beyond_range_start(self):
        axis = np.array([1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0])
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 0.0)
        assert result["actual_x"] == pytest.approx(1.0)

    def test_beyond_range_end(self):
        axis = np.array([1.0, 2.0, 3.0])
        wave = np.array([10.0, 20.0, 30.0])
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 100.0)
        assert result["actual_x"] == pytest.approx(3.0)

    def test_complex_returns_db_and_phase(self):
        axis = np.array([100.0, 1000.0, 10000.0])
        # Unity gain at all freqs, 0 phase
        wave = np.array([1.0 + 0j, 1.0 + 0j, 1.0 + 0j])
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        result = query_point_value(raw, "V(out)", 1000.0)
        assert "magnitude_db" in result
        assert result["magnitude_db"] == pytest.approx(0.0, abs=0.01)
        assert "phase_deg" in result
        assert "value" not in result  # complex path doesn't set "value"


class TestComputeAcBandwidthMetrics:
    def test_lowpass_bandwidth(self):
        """Simple RC lowpass should report -3dB bandwidth near fc."""
        freqs = np.logspace(0, 6, 1000)  # 1Hz to 1MHz
        fc = 1000  # 1kHz cutoff
        wave = 1 / (1 + 1j * freqs / fc)
        raw = _make_raw_mock(["V(out)"], freqs, {"V(out)": wave})

        metrics = compute_ac_bandwidth_metrics(raw, "V(out)")
        assert metrics["bandwidth_3db"] is not None
        assert metrics["bandwidth_3db"] == pytest.approx(fc, rel=0.1)

    def test_unity_gain_freq(self):
        """Lowpass with DC gain > 1 should have a unity gain frequency."""
        freqs = np.logspace(0, 8, 2000)
        fc = 1000
        gain = 100  # 40dB DC gain
        wave = gain / (1 + 1j * freqs / fc)
        raw = _make_raw_mock(["V(out)"], freqs, {"V(out)": wave})

        metrics = compute_ac_bandwidth_metrics(raw, "V(out)")
        assert metrics["unity_gain_freq"] is not None
        # Unity gain at f where |H(f)|=1 → f = fc * sqrt(gain^2 - 1) ≈ fc*gain
        expected_ugf = fc * np.sqrt(gain**2 - 1)
        assert metrics["unity_gain_freq"] == pytest.approx(expected_ugf, rel=0.15)


class TestExtractOperatingPoint:
    def test_categorizes_voltages_and_currents(self):
        traces = ["V(in)", "V(out)", "I(R1)", "I(V1)"]
        waves = {t: np.array([float(i)]) for i, t in enumerate(traces)}
        raw = _make_raw_mock(traces, np.array([0.0]), waves, plotname="Operating Point")

        op = extract_operating_point(raw)
        assert "V(in)" in op["voltages"]
        assert "V(out)" in op["voltages"]
        assert "I(R1)" in op["currents"]
        assert "I(V1)" in op["currents"]
        assert op["voltages"]["V(in)"] == pytest.approx(0.0)
        assert op["currents"]["I(R1)"] == pytest.approx(2.0)


class TestParseMeasurements:
    def test_nonexistent_file(self):
        with pytest.raises(ResultError):
            parse_measurements(Path("/nonexistent/file.log"))

    def test_log_without_measurements(self, work_dir: Path):
        """Log file with no .MEAS directives."""
        log = work_dir / "empty.log"
        log.write_text("LTspice 26.0.1\nCircuit: test.cir\nTotal elapsed time: 0.01 seconds.\n")
        result = parse_measurements(log)
        assert result["measurements"] == {}
        assert result["step_count"] == 0


class TestExtractLogDiagnostics:
    def test_empty_log(self, work_dir: Path):
        log = work_dir / "empty.log"
        log.write_text("")
        result = extract_log_diagnostics(log)
        assert result["warnings"] == []
        assert result["errors"] == []

    def test_file_error_with_caret(self, work_dir: Path):
        """filepath(line): message + source line + ^^^ caret."""
        log = work_dir / "parse_err.log"
        log.write_text(
            "LTspice 26.0.1\n"
            "Circuit: test.cir\n"
            "C:\\tmp\\test.cir(38): No such function defined.\n"
            ".meas AC gain_db FIND Vdb(outp) AT=1\n"
            "                      ^^^\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        assert "No such function defined" in result["errors"][0]
        assert "^^^" in result["errors"][0]
        assert result["warnings"] == []

    def test_fatal_error(self, work_dir: Path):
        log = work_dir / "fatal.log"
        log.write_text("Fatal Error: Unknown subcircuit called in: xu1 n004 n001 vcc 0 lm741\n")
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        assert "Fatal Error" in result["errors"][0]

    def test_error_on_line(self, work_dir: Path):
        log = work_dir / "line_err.log"
        log.write_text('Error on line 18 : r:u2:1:_r1 Unknown parameter "*"\n')
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 1
        assert "Error on line 18" in result["errors"][0]

    def test_warning_both_casings(self, work_dir: Path):
        log = work_dir / "warn.log"
        log.write_text(
            'Warning: Multiple definitions of model "2N2222"\nWARNING: Node U1:11 is floating\n'
        )
        result = extract_log_diagnostics(log)
        assert len(result["warnings"]) == 2
        assert result["errors"] == []

    def test_bare_convergence_errors(self, work_dir: Path):
        log = work_dir / "conv.log"
        log.write_text(
            "Direct Newton iteration for .op point succeeded.\n"
            "Singular matrix\n"
            "Time step too small\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["errors"]) == 2
        assert any("Singular matrix" in e for e in result["errors"])
        assert any("Time step too small" in e for e in result["errors"])

    def test_mixed_warnings_and_errors(self, work_dir: Path):
        log = work_dir / "mixed.log"
        log.write_text(
            "Warning: something minor\n"
            "C:\\test.cir(10): No such function defined.\n"
            ".meas AC foo FIND Vdb(x) AT=1\n"
            "                  ^^^\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        result = extract_log_diagnostics(log)
        assert len(result["warnings"]) == 1
        assert len(result["errors"]) == 1

    def test_nonexistent_file(self):
        result = extract_log_diagnostics(Path("/nonexistent/file.log"))
        assert result["warnings"] == []
        assert result["errors"] == []

    def test_measurements_surfaces_errors(self, work_dir: Path):
        """parse_measurements should include errors when results are empty."""
        log = work_dir / "meas_err.log"
        log.write_text(
            "LTspice 26.0.1\n"
            "Circuit: test.cir\n"
            "C:\\tmp\\test.cir(38): No such function defined.\n"
            ".meas AC gain_db FIND Vdb(outp) AT=1\n"
            "                      ^^^\n"
            "Total elapsed time: 0.01 seconds.\n"
        )
        result = parse_measurements(log)
        assert result["measurements"] == {}
        errors = result["errors"]
        assert errors is not None
        assert len(errors) == 1


class TestSampleToDict:
    def test_complex_sample_has_magnitude_linear(self):
        d = sample_to_dict(complex(0.0, 1.0))
        assert d["magnitude_linear"] == pytest.approx(1.0)
        assert d["magnitude_db"] == pytest.approx(0.0, abs=1e-9)
        assert d["phase_deg"] == pytest.approx(90.0)

    def test_real_sample_unchanged(self):
        d = sample_to_dict(3.5)
        assert d == {"value": 3.5}
        assert "magnitude_linear" not in d


class TestSteppedTransientAxes:
    """Per-step structure of a real stepped ``.tran`` raw."""

    def test_each_step_has_a_distinct_time_vector(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """A stepped ``.tran`` run stores a different time vector per step.

        LTspice's adaptive timestep yields a different sample count for each
        ``.step`` value, so the steps cannot share one x-axis. This pins the
        contract behind writing stepped runs in a tidy/long layout (one row
        per step+sample) instead of a wide shared-x table. Recorded from a
        real stepped-damping RLC transient (underdamped -> overdamped).
        """
        raw_path = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        raw = services.load_raw_sync(raw_path, state_no_sim)

        n_steps = get_step_count(raw)
        assert n_steps > 1, "fixture must be a multi-step run"

        lengths = [len(np.asarray(raw.get_axis(step=s))) for s in range(n_steps)]
        assert len(set(lengths)) > 1, (
            f"per-step time vectors should differ in length; got {lengths}"
        )
