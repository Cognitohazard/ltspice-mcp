"""Unit tests for result_parser — the most algorithmic module.

Tests pure-logic functions with synthetic numpy arrays, and parsing
functions with real .raw/.log files produced by LTspice integration tests.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.log_parser import extract_log_diagnostics, parse_measurements
from ltspice_mcp.lib.raw_parser import (
    compute_ac_bandwidth_metrics,
    compute_signal_stats,
    detect_sim_type,
    extract_operating_point,
    get_step_count,
    is_ac_analysis,
    query_point_value,
)

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


class TestComputeSignalStats:
    def test_real_data(self):
        """Transient/DC — real-valued data."""
        axis = np.linspace(0, 1, 100)
        wave = np.sin(2 * np.pi * axis)  # sine wave
        raw = _make_raw_mock(["V(out)"], axis, {"V(out)": wave})

        stats = compute_signal_stats(raw, "V(out)")
        assert stats["analysis_type"] == "transient"
        assert stats["min"] == pytest.approx(-1.0, abs=0.1)
        assert stats["max"] == pytest.approx(1.0, abs=0.1)
        assert stats["mean"] == pytest.approx(0.0, abs=0.1)
        assert stats["rms"] == pytest.approx(0.707, abs=0.05)
        assert stats["peak_to_peak"] == pytest.approx(2.0, abs=0.1)
        assert stats["point_count"] == 100
        # All values should be Python float
        for k, v in stats.items():
            if k != "analysis_type":
                assert type(v) in (float, int)

    def test_complex_data(self):
        """AC — complex-valued data."""
        freqs = np.logspace(0, 6, 50)
        # Simple lowpass: H(f) = 1/(1 + j*f/fc), fc=1kHz
        fc = 1000
        wave = 1 / (1 + 1j * freqs / fc)
        raw = _make_raw_mock(["V(out)"], freqs, {"V(out)": wave})

        stats = compute_signal_stats(raw, "V(out)")
        assert stats["analysis_type"] == "ac"
        assert stats["max_db"] == pytest.approx(0.0, abs=0.5)  # DC gain ~0dB
        assert stats["min_db"] < -20  # rolloff at high freq
        assert stats["point_count"] == 50


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
            'Warning: Multiple definitions of model "2N2222"\n'
            "WARNING: Node U1:11 is floating\n"
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
        assert "errors" in result
        assert len(result["errors"]) == 1
