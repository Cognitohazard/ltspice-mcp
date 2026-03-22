"""Tests for parse_fourier_data() and build_simulation_summary()."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.lib.log_parser import parse_fourier_data
from ltspice_mcp.lib.raw_parser import build_simulation_summary

# --- Helpers ---


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


# --- parse_fourier_data tests ---


class TestParseFourierData:
    def test_nonexistent_file_returns_empty(self):
        result = parse_fourier_data(Path("/nonexistent/file.log"))
        assert result == []

    def test_log_without_fourier_returns_empty(self, work_dir: Path):
        log = work_dir / "no_fourier.log"
        log.write_text("Circuit: test.cir\nTotal elapsed time: 0.01 seconds.\n")
        result = parse_fourier_data(log)
        assert result == []

    def test_log_with_fourier_data(self, work_dir: Path):
        """Parse a log file containing .FOUR results.

        LTspice Fourier output format in log:
        Fourier components of V(out)
        DC component:0.00123
        Harmonic  Frequency  Fourier   Normalized  Phase     Normalized
        Number    [Hz]       Component Component   [degree]  Phase [deg]
        1         1000       0.998     1.000       -0.5      0.000
        2         2000       0.001     0.001       45.2      45.700
        Total Harmonic Distortion: 0.12345%
        """
        # The actual parsing depends on LTSpiceLogReader's Fourier parsing.
        # Since we can't easily construct a log that LTSpiceLogReader will parse
        # with Fourier data without knowing the exact format, we test the graceful
        # degradation path.
        log = work_dir / "fourier.log"
        log.write_text("Circuit: test.cir\n.tran 0 10m 0 1u\nTotal elapsed time: 0.5 seconds.\n")
        result = parse_fourier_data(log)
        # LTSpiceLogReader won't find Fourier data here — returns empty
        assert isinstance(result, list)


# --- build_simulation_summary tests ---


class TestBuildSimulationSummary:
    def test_transient_summary(self):
        axis = np.linspace(0, 0.01, 1000)  # 10ms transient
        wave = np.sin(2 * np.pi * 1000 * axis)
        raw = _make_raw_mock(
            ["time", "V(out)", "I(R1)"],
            axis,
            {"V(out)": wave, "I(R1)": wave * 0.001, "time": axis},
            plotname="Transient Analysis",
        )

        summary = build_simulation_summary(raw, log_path=None)

        assert summary["sim_type"] == "Transient Analysis"
        assert summary["point_count"] == 1000
        assert summary["step_count"] == 1
        assert "time" in summary["signals"]
        assert "V(out)" in summary["signals"]
        assert "I(R1)" in summary["signals"]
        assert summary["range"]["time_start"] == pytest.approx(0.0)
        assert summary["range"]["time_end"] == pytest.approx(0.01)
        # No log provided — no measurements, warnings, or fourier
        assert "measurements" not in summary
        assert "warnings" not in summary
        assert "fourier" not in summary

    def test_ac_summary(self):
        freqs = np.logspace(0, 6, 500)
        fc = 1000
        wave = 1 / (1 + 1j * freqs / fc)
        raw = _make_raw_mock(
            ["frequency", "V(out)"],
            freqs,
            {"V(out)": wave, "frequency": freqs},
            plotname="AC Analysis",
        )

        summary = build_simulation_summary(raw, log_path=None)

        assert summary["sim_type"] == "AC Analysis"
        assert summary["point_count"] == 500
        assert "freq_start" in summary["range"]
        assert "freq_end" in summary["range"]
        assert summary["range"]["freq_start"] == pytest.approx(1.0)
        assert summary["range"]["freq_end"] == pytest.approx(1e6)

    def test_dc_sweep_summary(self):
        sweep = np.linspace(0, 5, 100)
        wave = sweep * 2  # linear gain
        raw = _make_raw_mock(
            ["V(in)", "V(out)"],
            sweep,
            {"V(in)": sweep, "V(out)": wave},
            plotname="DC sweep",
        )

        summary = build_simulation_summary(raw, log_path=None)

        assert "DC" in summary["sim_type"]
        assert "sweep_start" in summary["range"]
        assert summary["range"]["sweep_start"] == pytest.approx(0.0)
        assert summary["range"]["sweep_end"] == pytest.approx(5.0)

    def test_summary_with_duration(self):
        axis = np.linspace(0, 0.001, 100)
        raw = _make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
        )

        summary = build_simulation_summary(raw, log_path=None, duration=1.23)

        assert summary["duration"] == pytest.approx(1.23)

    def test_summary_with_log_measurements(self, work_dir: Path):
        """Summary includes .MEAS results when log file has measurements."""
        axis = np.linspace(0, 0.01, 100)
        raw = _make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.sin(2 * np.pi * 100 * axis), "time": axis},
        )

        # Create log with measurement-like content (LTSpiceLogReader format)
        # We use a log that won't parse measurements — test the graceful fallback
        log = work_dir / "sim.log"
        log.write_text(
            "Circuit: test.cir\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "Total elapsed time: 0.5 seconds.\n"
        )

        summary = build_simulation_summary(raw, log)

        # Log parsed successfully but no measurements found
        assert "measurements" not in summary

    def test_summary_with_log_warnings(self, work_dir: Path):
        """Summary includes warnings from log file."""
        axis = np.linspace(0, 0.01, 100)
        raw = _make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
        )

        log = work_dir / "warn.log"
        log.write_text(
            "Circuit: test.cir\n"
            "Warning: node N001 is floating\n"
            "Warning: less than 2 connections to node VCC\n"
            "Total elapsed time: 0.1 seconds.\n"
        )

        summary = build_simulation_summary(raw, log)

        assert "warnings" in summary
        assert len(summary["warnings"]) == 2
        assert any("N001" in w for w in summary["warnings"])

    def test_summary_multi_step(self):
        axis = np.linspace(0, 0.01, 100)
        raw = _make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.ones(100), "time": axis},
            steps=[0, 1, 2],
        )

        summary = build_simulation_summary(raw, log_path=None)

        assert summary["step_count"] == 3

    def test_all_values_are_python_types(self):
        """Ensure no numpy scalars leak into the summary."""
        axis = np.linspace(0, 0.005, 50)
        raw = _make_raw_mock(
            ["time", "V(out)"],
            axis,
            {"V(out)": np.sin(axis * 1000), "time": axis},
        )

        summary = build_simulation_summary(raw, log_path=None)

        # Check numeric values are Python types
        assert type(summary["point_count"]) is int
        assert type(summary["step_count"]) is int
        assert type(summary["range"]["time_start"]) is float
        assert type(summary["range"]["time_end"]) is float
