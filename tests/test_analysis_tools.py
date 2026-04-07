"""Tests for analysis tool handlers using mocked RawRead instances."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    MeasurementsInput,
    OperatingPointInput,
    QueryValueInput,
    SignalStatsInput,
    SimulationSummaryInput,
    handle_get_measurements,
    handle_get_operating_point,
    handle_get_signal_stats,
    handle_get_simulation_summary,
    handle_query_value,
)


def _inject_raw_mock(state: SessionState, path: Path, raw: MagicMock) -> None:
    """Insert a mock RawRead into the FileCache so load_raw returns it."""
    # Touch the file so cache mtime check works
    path.write_bytes(b"placeholder")
    state.results.set(path, raw)


def _make_raw_mock(
    *,
    plotname: str = "Transient Analysis",
    trace_names: list[str] | None = None,
    waves: dict[str, np.ndarray] | None = None,
    axis: np.ndarray | None = None,
    steps: list[int] | None = None,
) -> MagicMock:
    raw = MagicMock()
    trace_names = trace_names or ["time", "V(out)"]
    waves = waves or {
        "time": np.linspace(0, 1, 100),
        "V(out)": np.sin(2 * np.pi * np.linspace(0, 1, 100)),
    }
    axis = axis if axis is not None else waves.get("time", np.linspace(0, 1, 100))
    raw.get_raw_property.return_value = plotname
    raw.get_trace_names.return_value = trace_names
    raw.get_steps.return_value = steps if steps is not None else [0]
    raw.get_axis.return_value = axis

    def get_wave(name, step=0):
        return waves[name]

    raw.get_wave = get_wave
    return raw


@pytest.fixture
def fake_raw(state_no_sim: SessionState, work_dir: Path) -> Path:
    raw_file = work_dir / "result.raw"
    raw = _make_raw_mock()
    _inject_raw_mock(state_no_sim, raw_file, raw)
    return raw_file


@pytest.mark.asyncio
class TestSignalStats:
    async def test_transient(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_get_signal_stats(
            SignalStatsInput(raw_file=fake_raw.name, signal="V(out)"),
            state_no_sim,
        )
        text = result.content[0].text
        assert "V(out)" in text
        assert "Min:" in text
        assert "Max:" in text
        assert result.structuredContent["analysis_type"] == "transient"

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await handle_get_signal_stats(
                SignalStatsInput(raw_file=fake_raw.name, signal="V(missing)"),
                state_no_sim,
            )

    async def test_step_out_of_range(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="out of range"):
            await handle_get_signal_stats(
                SignalStatsInput(raw_file=fake_raw.name, signal="V(out)", step=99),
                state_no_sim,
            )

    async def test_ac_signal(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "ac.raw"
        freqs = np.logspace(0, 6, 100)
        wave = 1.0 / (1 + 1j * freqs / 1000)
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)"],
            waves={"frequency": freqs, "V(out)": wave},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_get_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        assert "AC" in result.content[0].text
        assert result.structuredContent["analysis_type"] == "ac"


@pytest.mark.asyncio
class TestQueryValue:
    async def test_transient(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_query_value(
            QueryValueInput(raw_file=fake_raw.name, signal="V(out)", at="0.5"),
            state_no_sim,
        )
        assert "V(out)" in result.content[0].text
        assert "Value:" in result.content[0].text

    async def test_invalid_at(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="Invalid 'at'"):
            await handle_query_value(
                QueryValueInput(raw_file=fake_raw.name, signal="V(out)", at="bad"),
                state_no_sim,
            )

    async def test_ac_query(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "ac.raw"
        freqs = np.logspace(0, 6, 100)
        wave = 1.0 / (1 + 1j * freqs / 1000)
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)"],
            waves={"frequency": freqs, "V(out)": wave},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="1k"),
            state_no_sim,
        )
        assert "Magnitude:" in result.content[0].text


@pytest.mark.asyncio
class TestGetMeasurements:
    async def test_invalid_log(self, state_no_sim: SessionState, work_dir: Path):
        log = work_dir / "bad.log"
        log.write_text("not a real spice log")
        with pytest.raises(ResultError):
            await handle_get_measurements(
                MeasurementsInput(log_file=log.name), state_no_sim
            )

    async def test_no_measurements_with_errors(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        log = work_dir / "err.log"
        log.write_text(
            "Circuit: * test\n"
            "Fatal Error: missing model XYZ\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = await handle_get_measurements(
            MeasurementsInput(log_file=log.name), state_no_sim
        )
        assert "errors in log" in result.content[0].text


@pytest.mark.asyncio
class TestGetOperatingPoint:
    async def test_basic(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "op.raw"
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(out)", "V(in)", "I(R1)"],
            waves={
                "V(out)": np.array([1.5]),
                "V(in)": np.array([3.3]),
                "I(R1)": np.array([0.001]),
            },
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_get_operating_point(
            OperatingPointInput(raw_file=raw_file.name), state_no_sim
        )
        text = result.content[0].text
        assert "V(out)" in text
        assert "I(R1)" in text


@pytest.mark.asyncio
class TestGetSimulationSummary:
    async def test_basic(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_get_simulation_summary(
            SimulationSummaryInput(raw_file=fake_raw.name), state_no_sim
        )
        text = result.content[0].text
        assert "Transient Analysis" in text
        assert "Signals" in text

    async def test_json_format(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_get_simulation_summary(
            SimulationSummaryInput(raw_file=fake_raw.name, format="json"),
            state_no_sim,
        )
        assert result.structuredContent is not None
        assert "sim_type" in result.structuredContent


class TestFormatMeasurements:
    def test_single_step(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"fc": [1591.5], "vp": [3.3]}, step_count=1)
        assert "fc" in text
        assert "1591.5" in text or "1.5915e" in text

    def test_failed_value(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"fc": [None]}, step_count=1)
        assert "FAILED" in text

    def test_multi_step(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"fc": [1.0, 2.0, None]}, step_count=3)
        assert "3 steps" in text
        assert "FAILED" in text

    def test_empty_with_errors(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({}, step_count=0, errors=["bad", "very bad"])
        assert "errors in log" in text
        assert "bad" in text

    def test_empty_no_errors(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({}, step_count=0)
        assert "No .MEAS results" in text


@pytest.mark.asyncio
class TestSummaryWithMeasurements:
    async def test_with_measurements_log(
        self, state_no_sim: SessionState, work_dir: Path, fake_raw: Path
    ):
        log = work_dir / "result.log"
        log.write_text(
            "Circuit: * test\n"
            "fc: mag(v(out))=0.707 AT 1591.5\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = await handle_get_simulation_summary(
            SimulationSummaryInput(raw_file=fake_raw.name, log_file=log.name),
            state_no_sim,
        )
        text = result.content[0].text
        assert "Transient Analysis" in text


@pytest.mark.asyncio
class TestSummaryAcWithMetrics:
    async def test_ac_with_signal(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "ac.raw"
        freqs = np.logspace(0, 6, 100)
        wave = 1.0 / (1 + 1j * freqs / 1000)
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)"],
            waves={"frequency": freqs, "V(out)": wave},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_get_simulation_summary(
            SimulationSummaryInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        text = result.content[0].text
        assert "AC Analysis" in text


@pytest.mark.asyncio
class TestQueryStepRange:
    async def test_step_out_of_range(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="out of range"):
            await handle_query_value(
                QueryValueInput(
                    raw_file=fake_raw.name, signal="V(out)", at="0.5", step=99
                ),
                state_no_sim,
            )

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await handle_query_value(
                QueryValueInput(
                    raw_file=fake_raw.name, signal="V(missing)", at="0.5"
                ),
                state_no_sim,
            )
