"""Tests for analysis tool handlers using mocked RawRead instances.

The classes at the bottom (``TestRecordedAcRaw`` / ``TestRecordedSteppedAcRaw``)
instead drive the handlers against real recorded LTspice binary raws from
``tests/fixtures/`` — see those classes for what the mocks cannot cover.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcp import types

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    BodeMetricsInput,
    EdgeMetricsInput,
    FilterMetricsInput,
    FindCrossingInput,
    GainAtInput,
    MeasurementStatsInput,
    OperatingPointInput,
    PeriodicMetricsInput,
    PulseResponseInput,
    QueryValueInput,
    ResonanceInput,
    RollOffInput,
    SignalStatsInput,
    SimulationSummaryInput,
    StabilityMetricsInput,
    TimingBetweenInput,
    handle_bode_metrics,
    handle_edge_metrics,
    handle_filter_metrics,
    handle_find_crossing,
    handle_gain_at,
    handle_measurement_stats,
    handle_operating_point,
    handle_periodic_metrics,
    handle_pulse_response,
    handle_query_value,
    handle_resonance,
    handle_roll_off,
    handle_signal_stats,
    handle_simulation_summary,
    handle_stability_metrics,
    handle_timing_between,
)
from ltspice_mcp.tools.circuit import (
    StepGetInput,
    handle_step_get,
)
from tests.conftest import stage_recorded_fixture as _stage_recorded


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
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=fake_raw.name, signal="V(out)"),
            state_no_sim,
        )
        text = result.content[0].text
        assert "V(out)" in text
        assert "Min:" in text
        assert "Max:" in text
        assert result.structuredContent["analysis_type"] == "transient"

    async def test_dc_sweep_classification(self, state_no_sim: SessionState, work_dir: Path):
        """A .DC raw used to report ``analysis_type='transient'`` and
        ``t_start_used`` / ``duration`` whose units were temperature, not
        seconds. The handler now branches on ``Plotname`` and surfaces
        ``sweep_start_used`` / ``sweep_end_used`` instead."""
        raw_file = work_dir / "dc.raw"
        temps = np.linspace(-40, 125, 34)
        raw = _make_raw_mock(
            plotname="DC transfer characteristic",
            trace_names=["temperature", "V(vref)"],
            waves={"temperature": temps, "V(vref)": 3.15 + 0.001 * temps},
            axis=temps,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(vref)"),
            state_no_sim,
        )
        data = result.structuredContent
        assert data is not None
        assert data["analysis_type"] == "dc"
        assert "sweep_start_used" in data
        assert "sweep_end_used" in data
        # Should NOT carry the time-domain-only fields.
        assert "t_start_used" not in data
        assert "duration" not in data
        # No RMS/std for DC sweeps — those are time-weighted and meaningless
        # over a swept variable.
        assert "rms" not in data

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await handle_signal_stats(
                SignalStatsInput(raw_file=fake_raw.name, signal="V(missing)"),
                state_no_sim,
            )

    async def test_step_out_of_range(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="out of range"):
            await handle_signal_stats(
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
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        assert "AC" in result.content[0].text
        assert result.structuredContent["analysis_type"] == "ac"

    async def test_ac_rejects_window(self, state_no_sim: SessionState, work_dir: Path):
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
        with pytest.raises(ResultError, match="not supported for AC"):
            await handle_signal_stats(
                SignalStatsInput(raw_file=raw_file.name, signal="V(out)", t_start="1k"),
                state_no_sim,
            )

    async def test_transient_time_weighted_rms(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "sine.raw"
        freq = 1000.0
        t = np.linspace(0, 10 / freq, 20001)
        amp = 5.0
        y = amp * np.sin(2 * np.pi * freq * t)
        raw = _make_raw_mock(
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["analysis_type"] == "transient"
        assert sc["rms"] == pytest.approx(amp / np.sqrt(2), rel=1e-3)
        assert sc["peak_to_peak"] == pytest.approx(2 * amp, rel=1e-3)
        assert sc["std"] == pytest.approx(amp / np.sqrt(2), rel=1e-3)
        assert sc["t_start_used"] == pytest.approx(0.0)

    async def test_transient_windowed(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "step.raw"
        t = np.linspace(0, 1e-3, 2001)
        # Step from 0 to 5V at t=0.5ms; window selects steady DC portion.
        y = np.where(t < 0.5e-3, 0.0, 5.0)
        raw = _make_raw_mock(
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(out)", t_start="0.6m", t_end="1m"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["mean"] == pytest.approx(5.0)
        assert sc["rms"] == pytest.approx(5.0)
        assert sc["std"] == pytest.approx(0.0, abs=1e-9)
        assert sc["t_start_used"] == pytest.approx(6e-4)
        assert sc["t_end_used"] == pytest.approx(1e-3)


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
        result = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name), state_no_sim
        )
        text = result.content[0].text
        assert "V(out)" in text
        assert "I(R1)" in text

    async def test_rejects_ac_raw(self, state_no_sim: SessionState, work_dir: Path):
        """``extract_operating_point`` reads ``wave[0]`` for every trace.
        On an AC raw that's the magnitude at the first frequency, not a
        DC bias. We used to silently return those AC magnitudes labeled
        as voltages (``V(in)=1`` from an ``AC 1`` source) — now we reject."""
        from ltspice_mcp.errors import ResultError

        raw_file = work_dir / "ac.raw"
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["V(out)", "V(in)"],
            waves={
                "V(out)": np.array([0.5 + 0j, 0.4 + 0.1j]),
                "V(in)": np.array([1.0 + 0j, 1.0 + 0j]),
            },
            axis=np.array([1.0, 10.0]),
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="AC/Noise"):
            await handle_operating_point(OperatingPointInput(raw_file=raw_file.name), state_no_sim)

    async def test_rejects_transient_raw(self, state_no_sim: SessionState, work_dir: Path):
        from ltspice_mcp.errors import ResultError

        raw_file = work_dir / "tran.raw"
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["V(out)"],
            waves={"V(out)": np.array([0.0, 1.0, 2.0])},
            axis=np.array([0.0, 1e-6, 2e-6]),
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="t=0"):
            await handle_operating_point(OperatingPointInput(raw_file=raw_file.name), state_no_sim)


@pytest.mark.asyncio
class TestGetSimulationSummary:
    async def test_basic(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=fake_raw.name), state_no_sim
        )
        text = result.content[0].text
        assert "Transient Analysis" in text
        assert "Signals" in text

    async def test_json_format(self, state_no_sim: SessionState, fake_raw: Path):
        result = await handle_simulation_summary(
            SimulationSummaryInput(raw_file=fake_raw.name, format="json"),
            state_no_sim,
        )
        assert result.structuredContent is not None
        assert "sim_type" in result.structuredContent


class TestFormatMeasurements:
    def test_single_step(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements(
            {"fc": {"values": [1591.5]}, "vp": {"values": [3.3]}}, step_count=1
        )
        assert "fc" in text
        assert "1591.5" in text or "1.5915e" in text

    def test_failed_value(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"fc": {"values": [None]}}, step_count=1)
        assert "FAILED" in text

    def test_multi_step(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"fc": {"values": [1.0, 2.0, None]}}, step_count=3)
        assert "3 steps" in text
        assert "FAILED" in text

    def test_window_metadata_appears(self):
        """``range_from`` / ``range_to`` should be folded into the line, not surfaced as
        separate measurements."""
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements(
            {"v_rms": {"values": [0.707], "range_from": 0.002, "range_to": 0.01}},
            step_count=1,
        )
        assert "FROM=0.002" in text
        assert "TO=0.01" in text

    def test_at_metadata_appears(self):
        from ltspice_mcp.tools.analysis import _format_measurements

        text = _format_measurements({"vref_op": {"values": [3.18], "at": 1.03}}, step_count=1)
        assert "AT=1.03" in text

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
        result = await handle_simulation_summary(
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
        result = await handle_simulation_summary(
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
                QueryValueInput(raw_file=fake_raw.name, signal="V(out)", at="0.5", step=99),
                state_no_sim,
            )

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await handle_query_value(
                QueryValueInput(raw_file=fake_raw.name, signal="V(missing)", at="0.5"),
                state_no_sim,
            )


def _step_waveform(step_time: float = 0.5e-3, tr: float = 0.1e-3, n: int = 5001):
    t = np.linspace(0, 2e-3, n)
    y = np.where(t < step_time, 0.0, np.where(t < step_time + tr, (t - step_time) / tr, 1.0))
    return t, y


def _square_wave(freq: float = 1000.0, duty: float = 0.5, periods: int = 5, n: int = 50001):
    t = np.linspace(0, periods / freq, n)
    phase = (t * freq) % 1.0
    y = np.where(phase < duty, 1.0, 0.0)
    return t, y


# ---------------------------------------------------------------------------
# edge_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestEdgeMetrics:
    async def test_happy_path(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_edge_metrics(
            EdgeMetricsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["is_rise_time"] is True
        assert sc["signal"] == "V(out)"
        assert sc["transition_time"] > 0
        assert "Rise time" in result.content[0].text

    async def test_ac_rejected(self, state_no_sim: SessionState, work_dir: Path):
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
        with pytest.raises(ResultError, match="transient analysis"):
            await handle_edge_metrics(
                EdgeMetricsInput(raw_file=raw_file.name, signal="V(out)"),
                state_no_sim,
            )

    async def test_invalid_signal(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="not found"):
            await handle_edge_metrics(
                EdgeMetricsInput(raw_file=raw_file.name, signal="V(missing)"),
                state_no_sim,
            )

    async def test_window_propagated(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_edge_metrics(
            EdgeMetricsInput(
                raw_file=raw_file.name,
                signal="V(out)",
                t_start="100u",
                t_end="1m",
            ),
            state_no_sim,
        )
        assert result.structuredContent["is_rise_time"] is True

    async def test_invalid_t_start(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="Invalid t_start"):
            await handle_edge_metrics(
                EdgeMetricsInput(raw_file=raw_file.name, signal="V(out)", t_start="garbage"),
                state_no_sim,
            )

    async def test_json_format(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_edge_metrics(
            EdgeMetricsInput(raw_file=raw_file.name, signal="V(out)", format="json"),
            state_no_sim,
        )
        assert result.structuredContent is not None
        # JSON format emits JSON text
        assert result.content[0].text.startswith("{")


# ---------------------------------------------------------------------------
# pulse_response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPulseResponse:
    async def test_happy_path(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "pulse.raw"
        # Underdamped step with pre-step plateau
        t_pre = np.linspace(-1e-3, 0, 500, endpoint=False)
        t_post = np.linspace(0, 20e-3, 20001)
        y_pre = np.zeros_like(t_pre)
        zeta = 0.3
        wn = 2 * np.pi * 500
        wd = wn * np.sqrt(1 - zeta**2)
        phi = np.arctan2(np.sqrt(1 - zeta**2), zeta)
        y_post = 1 - np.exp(-zeta * wn * t_post) / np.sqrt(1 - zeta**2) * np.sin(wd * t_post + phi)
        t = np.concatenate([t_pre, t_post])
        y = np.concatenate([y_pre, y_post])
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)

        # Pass explicit initial/final — the auto-detect window averages first 10%
        # which, with 500 pre samples and 20001 post samples, bleeds into ringing.
        result = await handle_pulse_response(
            PulseResponseInput(
                raw_file=raw_file.name,
                signal="V(out)",
                initial_value=0.0,
                final_value=1.0,
            ),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["direction"] == "rising"
        assert sc["overshoot_pct"] > 0
        assert sc["initial_value"] == 0.0
        assert sc["steady_state_value"] == 1.0

    async def test_no_step_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "flat.raw"
        t = np.linspace(0, 1e-3, 1000)
        y = np.full_like(t, 3.3)
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="No step detected"):
            await handle_pulse_response(
                PulseResponseInput(raw_file=raw_file.name, signal="V(out)"),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# timing_between
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTimingBetween:
    async def test_known_delay(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "tim.raw"
        t = np.linspace(0, 1e-3, 10001)
        vin = np.where(t < 0.3e-3, 0.0, 3.3)
        vout = np.where(t < 0.5e-3, 0.0, 1.8)
        raw = _make_raw_mock(
            trace_names=["time", "V(in)", "V(out)"],
            waves={"time": t, "V(in)": vin, "V(out)": vout},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_timing_between(
            TimingBetweenInput(raw_file=raw_file.name, signal_a="V(in)", signal_b="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["delay"] == pytest.approx(0.2e-3, abs=1e-6)
        assert sc["threshold_a_used"] == pytest.approx(1.65, abs=0.01)
        assert sc["threshold_b_used"] == pytest.approx(0.9, abs=0.01)

    async def test_missing_signal_b(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "tim.raw"
        t = np.linspace(0, 1e-3, 1000)
        vin = np.where(t < 0.3e-3, 0.0, 3.3)
        raw = _make_raw_mock(
            trace_names=["time", "V(in)"],
            waves={"time": t, "V(in)": vin},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="not found"):
            await handle_timing_between(
                TimingBetweenInput(raw_file=raw_file.name, signal_a="V(in)", signal_b="V(out)"),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# periodic_metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPeriodicMetrics:
    async def test_square_wave(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "sq.raw"
        t, y = _square_wave(freq=1000.0, duty=0.4, periods=10)
        raw = _make_raw_mock(
            trace_names=["time", "V(clk)"],
            waves={"time": t, "V(clk)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_periodic_metrics(
            PeriodicMetricsInput(raw_file=raw_file.name, signal="V(clk)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["frequency"] == pytest.approx(1000.0, rel=0.01)
        assert sc["duty_cycle_pct"] == pytest.approx(40.0, abs=1.0)

    async def test_constant_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "flat.raw"
        t = np.linspace(0, 1e-3, 1000)
        y = np.full_like(t, 1.0)
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="constant"):
            await handle_periodic_metrics(
                PeriodicMetricsInput(raw_file=raw_file.name, signal="V(out)"),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# measurement_stats
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestMeasurementStats:
    async def test_basic(self, state_no_sim: SessionState, work_dir: Path):
        # Use the same single-measurement log format validated by the log
        # parser tests — ensures the plumbing works. Multi-step aggregation
        # logic is covered by test_waveform_analysis.TestComputeMeasurementStats.
        log = work_dir / "meas.log"
        log.write_text(
            "Circuit: * test\n"
            "\n"
            "Direct Newton iteration for .op point succeeded.\n"
            "fc: mag(v(out))=0.707 AT 1591.5\n"
            "Date: today\n"
            "Total elapsed time: 0.001 seconds.\n"
        )
        result = await handle_measurement_stats(
            MeasurementStatsInput(log_file=log.name), state_no_sim
        )
        assert result.structuredContent is not None
        assert "stats" in result.structuredContent
        # Should have exactly one measurement aggregated
        assert len(result.structuredContent["stats"]) >= 1

    async def test_missing_log_file(self, state_no_sim: SessionState, work_dir: Path):
        with pytest.raises(ResultError):
            await handle_measurement_stats(
                MeasurementStatsInput(log_file="nonexistent.log"), state_no_sim
            )

    async def test_empty_log_errors(self, state_no_sim: SessionState, work_dir: Path):
        log = work_dir / "empty.log"
        log.write_text("not a spice log\n")
        with pytest.raises(ResultError):
            await handle_measurement_stats(MeasurementStatsInput(log_file=log.name), state_no_sim)


# ---------------------------------------------------------------------------
# AC-tool handlers (integration: parsing + load path + formatting)
# ---------------------------------------------------------------------------


def _ac_raw(
    state: SessionState,
    work_dir: Path,
    *,
    filename: str = "ac.raw",
    points: int = 500,
    fc: float = 1000.0,
) -> Path:
    """Build a mock AC RawRead with a 1-pole LPF transfer function."""
    raw_file = work_dir / filename
    freqs = np.logspace(0, 6, points)
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    H = wc / (s + wc)
    raw = _make_raw_mock(
        plotname="AC Analysis",
        trace_names=["frequency", "V(out)"],
        waves={"frequency": freqs, "V(out)": H},
        axis=freqs,
    )
    _inject_raw_mock(state, raw_file, raw)
    return raw_file


@pytest.mark.asyncio
class TestFilterMetricsTool:
    async def test_lpf_classification(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        result = await handle_filter_metrics(
            FilterMetricsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["filter_type"] == "lowpass"
        assert sc["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.05)
        assert sc["estimated_order"] == 1
        item = result.content[0]
        assert isinstance(item, types.TextContent)
        assert "Filter Metrics" in item.text

    async def test_rejects_transient(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="AC analysis"):
            await handle_filter_metrics(
                FilterMetricsInput(raw_file=fake_raw.name, signal="V(out)"),
                state_no_sim,
            )

    async def test_ref_db_must_be_negative(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="negative"):
            await handle_filter_metrics(
                FilterMetricsInput(raw_file=raw_file.name, signal="V(out)", ref_db=3.0),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestGainAtTool:
    async def test_batch_query(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        result = await handle_gain_at(
            GainAtInput(
                raw_file=raw_file.name,
                signal="V(out)",
                frequencies=["100", "1k", "10k"],
            ),
            state_no_sim,
        )
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert len(sc["points"]) == 3
        # 1-pole LPF at fc should be -3 dB.
        assert sc["points"][1]["magnitude_db"] == pytest.approx(-3.0, abs=0.1)

    async def test_empty_frequencies(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="empty"):
            await handle_gain_at(
                GainAtInput(raw_file=raw_file.name, signal="V(out)", frequencies=[]),
                state_no_sim,
            )

    async def test_invalid_frequency(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError):
            await handle_gain_at(
                GainAtInput(
                    raw_file=raw_file.name,
                    signal="V(out)",
                    frequencies=["not_a_number"],
                ),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestStabilityMetricsTool:
    async def test_2pole_loop(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "loop.raw"
        freqs = np.logspace(0, 8, 500)
        s = 1j * 2 * np.pi * freqs
        A = 1000.0
        H = A / ((1 + s / (2 * np.pi * 1000)) * (1 + s / (2 * np.pi * 100000)))
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(loop)"],
            waves={"frequency": freqs, "V(loop)": H},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_stability_metrics(
            StabilityMetricsInput(raw_file=raw_file.name, signal="V(loop)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc["stability"] in ("unconditional", "stable")
        assert sc["phase_margin_worst_deg"] is not None
        # 60 dB DC gain.
        assert sc["dc_gain_db"] == pytest.approx(60.0, abs=0.1)


@pytest.mark.asyncio
class TestRollOffTool:
    async def test_1pole_asymptote(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir, fc=100.0)
        result = await handle_roll_off(
            RollOffInput(
                raw_file=raw_file.name,
                signal="V(out)",
                f_low="10k",
                f_high="100k",
            ),
            state_no_sim,
        )
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert sc["slope_db_per_decade"] == pytest.approx(-20.0, abs=1.0)
        assert sc["nearest_pole_order_estimate"] == 1


@pytest.mark.asyncio
class TestResonanceTool:
    async def test_biquad(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "reson.raw"
        freqs = np.logspace(1, 5, 3000)
        s = 1j * 2 * np.pi * freqs
        w0 = 2 * np.pi * 1000
        Q = 10.0
        H = (w0 * w0) / (s * s + (w0 / Q) * s + w0 * w0)
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)"],
            waves={"frequency": freqs, "V(out)": H},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_resonance(
            ResonanceInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert len(sc["peaks"]) == 1
        peak = sc["peaks"][0]
        assert peak["frequency_hz"] == pytest.approx(1000.0, rel=0.05)
        assert peak["q_factor"] == pytest.approx(10.0, rel=0.1)


@pytest.mark.asyncio
class TestFindCrossingTool:
    async def test_magnitude_crossing(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        result = await handle_find_crossing(
            FindCrossingInput(
                raw_file=raw_file.name,
                signal="V(out)",
                quantity="magnitude_db",
                level=-3.0,
            ),
            state_no_sim,
        )
        assert result.structuredContent is not None
        sc = result.structuredContent
        assert len(sc["crossings"]) == 1
        assert sc["crossings"][0]["frequency_hz"] == pytest.approx(1000.0, rel=0.05)

    async def test_rejects_transient(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="AC analysis"):
            await handle_find_crossing(
                FindCrossingInput(
                    raw_file=fake_raw.name,
                    signal="V(out)",
                    quantity="magnitude_db",
                    level=0.0,
                ),
                state_no_sim,
            )

    async def test_max_results_validated(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="max_results"):
            await handle_find_crossing(
                FindCrossingInput(
                    raw_file=raw_file.name,
                    signal="V(out)",
                    quantity="magnitude_db",
                    level=0.0,
                    max_results=0,
                ),
                state_no_sim,
            )


class TestParseFreqUnitTolerance:
    """Frequency parsing accepts a trailing Hz/kHz unit."""

    def test_bare_number(self):
        from ltspice_mcp.tools.analysis import _parse_freq

        assert _parse_freq("1000") == pytest.approx(1000.0)

    def test_hz_suffix(self):
        from ltspice_mcp.tools.analysis import _parse_freq

        assert _parse_freq("159Hz") == pytest.approx(159.0)

    def test_khz_suffix(self):
        from ltspice_mcp.tools.analysis import _parse_freq

        assert _parse_freq("15.9kHz") == pytest.approx(15900.0)

    def test_si_prefix_still_works(self):
        from ltspice_mcp.tools.analysis import _parse_freq

        assert _parse_freq("1k") == pytest.approx(1000.0)
        assert _parse_freq("1meg") == pytest.approx(1e6)


# ---------------------------------------------------------------------------
# Shared raw-mock helpers for the query_value / bode_metrics tests below.
# ---------------------------------------------------------------------------


def _inject_raw(state: SessionState, path: Path, raw: MagicMock) -> None:
    path.write_bytes(b"placeholder")
    state.results.set(path, raw)


def _ac_raw_mock() -> MagicMock:
    """An AC raw mock: real frequency axis + complex first-order-LPF response."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "AC Analysis"
    raw.get_trace_names.return_value = ["frequency", "V(out)"]
    freq = np.logspace(0, 5, 200)  # 1 Hz .. 100 kHz
    fc = 1591.5
    H = 1.0 / (1.0 + 1j * (freq / fc))
    raw.get_axis.return_value = freq
    raw.get_steps.return_value = [0]
    raw.get_wave = lambda name, step=0: H
    return raw


def _stepped_ac_raw(fcs: list[float]) -> MagicMock:
    """A stepped AC raw: one first-order-LPF response per cutoff in ``fcs``."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "AC Analysis"
    raw.get_trace_names.return_value = ["frequency", "V(out)"]
    freq = np.logspace(0, 5, 200)
    responses = [1.0 / (1.0 + 1j * (freq / fc)) for fc in fcs]
    raw.get_axis.return_value = freq
    raw.get_steps.return_value = [{"fc": fc} for fc in fcs]
    raw.get_wave = lambda name, step=0: responses[step]
    return raw


@pytest.mark.asyncio
class TestQueryValueMagnitudeLinear:
    async def test_ac_returns_magnitude_linear(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        freq = np.array([10.0, 100.0, 1000.0])
        volt = np.array([1 + 0j, 0.7 + 0.7j, 0.1 + 0j])
        raw.get_axis.return_value = freq
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: volt
        path = work_dir / "ac.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_query_value(
            QueryValueInput(raw_file="ac.raw", signal="V(out)", at="100"), state_no_sim
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["magnitude_linear"] == pytest.approx(abs(0.7 + 0.7j))
        assert "magnitude_db" in sc


@pytest.mark.asyncio
class TestStepGet:
    async def test_raw_axis_snap_warning(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["Rval", "V(out)"]
        raw.get_axis.return_value = np.array([500.0, 1000.0, 2000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0, 3.0])
        path = work_dir / "dc.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dc.raw", axis="Rval", value="99999", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["exact_match"] is False
        assert sc["actual_value"] == 2000.0
        assert sc.get("warnings")

    async def test_raw_axis_rejects_at(self, state_no_sim: SessionState, work_dir: Path):
        # 'at' selects the inner-axis point of a .step lookup; on the
        # native-axis branch the queried axis IS the inner axis, so the
        # param used to be silently ignored — it must refuse loudly.
        from ltspice_mcp.errors import NetlistError

        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["Rval", "V(out)"]
        raw.get_axis.return_value = np.array([500.0, 1000.0, 2000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0, 3.0])
        path = work_dir / "dc_at.raw"
        _inject_raw(state_no_sim, path, raw)

        with pytest.raises(NetlistError, match="native axis"):
            await handle_step_get(
                StepGetInput(
                    raw_file="dc_at.raw", axis="Rval", value="1k", signal="V(out)", at="1m"
                ),
                state_no_sim,
            )

    async def test_raw_axis_exact_match_no_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["Rval", "V(out)"]
        raw.get_axis.return_value = np.array([500.0, 1000.0, 2000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0, 3.0])
        path = work_dir / "dc2.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dc2.raw", axis="Rval", value="1k", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["exact_match"] is True
        assert sc["actual_value"] == 1000.0
        assert not sc.get("warnings")

    async def test_raw_axis_complex_ac_keeps_magnitude(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # axis name "frequency" == trace 0 → raw-axis branch on an AC raw.
        # The complex sample must survive as magnitude/phase, not float()'d.
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        raw.get_axis.return_value = np.array([10.0, 100.0, 1000.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([1 + 0j, 0.7 + 0.7j, 0.1 + 0j])
        path = work_dir / "acaxis.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="acaxis.raw", axis="frequency", value="100", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert "magnitude_linear" in sc
        assert "magnitude_db" in sc
        assert "value" not in sc  # complex sample, not a real scalar
        assert sc["magnitude_linear"] == pytest.approx(abs(0.7 + 0.7j))
        assert not sc.get("warnings")

    async def test_raw_axis_interior_offgrid_no_clamp_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Dense continuous axis: an interior off-grid request is a normal
        # nearest-neighbour lookup, NOT an out-of-range clamp — no warning.
        raw = MagicMock()
        raw.get_raw_property.return_value = "DC transfer characteristic"
        raw.get_trace_names.return_value = ["v1", "V(out)"]
        raw.get_axis.return_value = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        raw.get_steps.return_value = [0]
        raw.get_wave = lambda name, step=0: np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        path = work_dir / "dense.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="dense.raw", axis="v1", value="1.01", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["actual_value"] == 1.0
        assert sc["exact_match"] is False  # off-grid, honest
        assert not sc.get("warnings")  # but interior → not "clamped"

    async def test_step_lookup_inside_range_snap_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # temp=50 sits between discrete steps {27, 85}: nearest-step used, not
        # "clamped" (it is inside the swept range).
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"temp": 27.0}, {"temp": 85.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "tempstep.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="tempstep.raw", axis="temp", value="50", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["actual_value"] == 27.0
        assert sc["exact_match"] is False
        assert any("nearest step" in w for w in sc.get("warnings", []))
        assert all("clamped" not in w for w in sc.get("warnings", []))

    async def test_step_lookup_default_at_label(self, state_no_sim: SessionState, work_dir: Path):
        raw = MagicMock()
        raw.get_raw_property.return_value = "AC Analysis"
        # Axis name != requested axis → falls to the .step parameter lookup.
        raw.get_trace_names.return_value = ["frequency", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}, {"Rval": 2000.0}]
        raw.get_axis.return_value = np.array([10.0, 100.0, 1000.0])
        raw.get_wave = lambda name, step=0: np.array([0.5, 0.6, 0.7])
        path = work_dir / "step.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_step_get(
            StepGetInput(raw_file="step.raw", axis="Rval", value="1000", signal="V(out)"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["step_index"] == 1
        assert sc["exact_match"] is True
        assert sc["actual_at"] == 10.0
        assert any("No 'at' given" in w for w in sc.get("warnings", []))


@pytest.mark.asyncio
class TestBodeMetrics:
    async def test_point_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode.raw", signal="V(out)", mode="point", frequencies=["1k"]
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert "points" in res.structuredContent

    async def test_crossing_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode2.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode2.raw",
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        cs = res.structuredContent["crossings"]
        assert cs and abs(cs[0]["frequency_hz"] - 1591.5) / 1591.5 < 0.05

    async def test_slope_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode3.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="bode3.raw", signal="V(out)", mode="slope", f_low="10k", f_high="100k"
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        # First-order LPF stopband ≈ -20 dB/decade.
        assert res.structuredContent["slope_db_per_decade"] < -15

    async def test_filter_mode_dispatch(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode4.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        res = await handle_bode_metrics(
            BodeMetricsInput(raw_file="bode4.raw", signal="V(out)", mode="filter"),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert "filter_type" in res.structuredContent

    async def test_crossing_requires_quantity_and_level(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = work_dir / "bode5.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError, match="requires 'quantity' and 'level'"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="bode5.raw", signal="V(out)", mode="crossing"),
                state_no_sim,
            )

    async def test_slope_requires_bounds(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode6.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError, match="requires 'f_low' and 'f_high'"):
            await handle_bode_metrics(
                BodeMetricsInput(raw_file="bode6.raw", signal="V(out)", mode="slope"),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestQueryValueStepAbsorb:
    async def test_step_axis_dispatches_to_step_lookup(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}, {"Rval": 2000.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "qstep.raw"
        _inject_raw(state_no_sim, path, raw)

        res = await handle_query_value(
            QueryValueInput(
                raw_file="qstep.raw", signal="V(out)", step_axis="Rval", step_value="1000"
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        assert res.structuredContent["step_index"] == 1
        assert res.structuredContent["exact_match"] is True

    async def test_step_axis_requires_step_value(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "qstep2.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError, match="step_value"):
            await handle_query_value(
                QueryValueInput(raw_file="qstep2.raw", signal="V(out)", step_axis="Rval"),
                state_no_sim,
            )

    async def test_requires_at_without_step_axis(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "qstep3.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError, match="'at' is required"):
            await handle_query_value(
                QueryValueInput(raw_file="qstep3.raw", signal="V(out)"), state_no_sim
            )


@pytest.mark.asyncio
class TestBodeMetricsAllSteps:
    async def test_crossing_per_step(self, state_no_sim: SessionState, work_dir: Path):
        fcs = [500.0, 5000.0]
        path = work_dir / "stepped.raw"
        _inject_raw(state_no_sim, path, _stepped_ac_raw(fcs))
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="stepped.raw",
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
                all_steps=True,
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["all_steps"] is True
        assert sc["step_count"] == 2
        steps = sc["steps"]
        assert [s["step"] for s in steps] == [0, 1]
        # The -3 dB crossing of each step tracks that step's cutoff.
        for i, fc in enumerate(fcs):
            cs = steps[i]["crossings"]
            assert cs and abs(cs[0]["frequency_hz"] - fc) / fc < 0.05

    async def test_single_step_warns(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "onestep.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())  # get_steps == [0]
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="onestep.raw", signal="V(out)", mode="filter", all_steps=True
            ),
            state_no_sim,
        )
        assert res.structuredContent is not None
        sc = res.structuredContent
        assert sc["step_count"] == 1
        assert len(sc["steps"]) == 1
        assert any("not stepped" in w for w in sc.get("warnings", []))

    async def test_all_steps_still_validates_mode_args(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # all_steps must enforce the same per-mode required args as single-step.
        path = work_dir / "stepped2.raw"
        _inject_raw(state_no_sim, path, _stepped_ac_raw([500.0, 5000.0]))
        with pytest.raises(ResultError, match="requires 'f_low' and 'f_high'"):
            await handle_bode_metrics(
                BodeMetricsInput(
                    raw_file="stepped2.raw", signal="V(out)", mode="slope", all_steps=True
                ),
                state_no_sim,
            )

    async def test_all_steps_total_failure_raises(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Regression: a non-AC raw makes every step fail. all_steps must surface
        # a real error, not a "success" full of buried per-step errors.
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [0]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "tran.raw"
        _inject_raw(state_no_sim, path, raw)
        with pytest.raises(ResultError, match="AC analysis"):
            await handle_bode_metrics(
                BodeMetricsInput(
                    raw_file="tran.raw", signal="V(out)", mode="filter", all_steps=True
                ),
                state_no_sim,
            )


# ---------------------------------------------------------------------------
# Recorded real LTspice binary raws (tests/fixtures/).
#
# The mocks above hand the handlers a real-valued frequency axis and ignore
# the ``step=`` argument of ``get_wave``, so two things only these fixtures
# can prove: (1) LTspice stores the AC frequency axis as complex values, and
# the analysis path must take its real part; (2) per-step extraction must
# return DIFFERENT data for different steps, not the same array repeated.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRecordedAcRaw:
    """Single-run AC raw: ltspice_ac_rc (RC LPF, R=1k C=159.15n, fc=1kHz,
    ``.ac dec 20 10 100k``). The complex frequency axis must parse through
    the real entry path and yield the analytic filter numbers."""

    async def test_filter_mode_finds_rc_pole(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        res = await handle_bode_metrics(
            BodeMetricsInput(raw_file=str(raw), signal="V(out)", mode="filter"),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["filter_type"] == "lowpass"
        assert sc["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.02)
        assert sc["estimated_order"] == 1

    async def test_crossing_mode_minus_3db_at_cutoff(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file=str(raw),
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
            ),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert len(sc["crossings"]) == 1
        assert sc["crossings"][0]["frequency_hz"] == pytest.approx(1000.0, rel=0.02)

    async def test_signal_stats_ac_magnitude_range(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        res = await handle_signal_stats(
            SignalStatsInput(raw_file=str(raw), signal="V(out)"),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "ac"
        assert sc["point_count"] == 81  # dec 20 over 4 decades
        # Passband (10 Hz, two decades below the pole): |H| ~ 1, ~0 dB.
        assert sc["max_db"] == pytest.approx(0.0, abs=0.01)
        # Stopband end (100 kHz = 100*fc): |H| ~ 1/100, ~ -40 dB.
        assert sc["min_db"] == pytest.approx(-40.0, abs=0.1)

    async def test_query_value_passband_and_pole(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        passband = await handle_query_value(
            QueryValueInput(raw_file=str(raw), signal="V(out)", at="10"),
            state_no_sim,
        )
        sc = passband.structuredContent
        assert sc is not None
        assert sc["magnitude_linear"] == pytest.approx(1.0, abs=1e-3)
        assert sc["magnitude_db"] == pytest.approx(0.0, abs=0.01)

        pole = await handle_query_value(
            QueryValueInput(raw_file=str(raw), signal="V(out)", at="1k"),
            state_no_sim,
        )
        sc = pole.structuredContent
        assert sc is not None
        assert sc["magnitude_db"] == pytest.approx(-3.0103, abs=0.02)
        assert sc["phase_deg"] == pytest.approx(-45.0, abs=0.5)


@pytest.mark.asyncio
class TestRecordedSteppedAcRaw:
    """Stepped AC raw: ltspice_step_ac (RC LPF, C=100n,
    ``.step param R LIST 1k 2k 4k`` + ``.ac dec 20 10 100k``).
    fc = 1/(2*pi*R*C) gives three DISTINCT analytic cutoffs — matching each
    proves step-indexed trace extraction reads real per-step data."""

    # R = 1k / 2k / 4k with C = 100n.
    CUTOFFS = (1591.55, 795.77, 397.89)

    async def test_all_steps_filter_cutoffs_distinct(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_step_ac")
        res = await handle_bode_metrics(
            BodeMetricsInput(raw_file=str(raw), signal="V(out)", mode="filter", all_steps=True),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["all_steps"] is True
        assert sc["step_count"] == 3
        steps = sc["steps"]
        assert [s["step"] for s in steps] == [0, 1, 2]
        for entry, fc in zip(steps, self.CUTOFFS, strict=True):
            assert entry["filter_type"] == "lowpass"
            assert entry["cutoff_high_hz"] == pytest.approx(fc, rel=0.01)

    async def test_single_step_filter_uses_requested_step(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_step_ac")
        for step, fc in enumerate(self.CUTOFFS):
            res = await handle_bode_metrics(
                BodeMetricsInput(raw_file=str(raw), signal="V(out)", mode="filter", step=step),
                state_no_sim,
            )
            sc = res.structuredContent
            assert sc is not None
            assert sc["cutoff_high_hz"] == pytest.approx(fc, rel=0.01)

    async def test_query_value_pins_step_by_axis_value(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_step_ac")
        # Select the R=2k step by parameter value; query its own cutoff
        # frequency, where a first-order LPF reads -3.01 dB / -45 degrees.
        res = await handle_query_value(
            QueryValueInput(
                raw_file=str(raw),
                signal="V(out)",
                step_axis="R",
                step_value="2k",
                at="795.77",
            ),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["step_index"] == 1
        assert sc["actual_value"] == pytest.approx(2000.0)
        assert sc["exact_match"] is True
        assert sc["magnitude_db"] == pytest.approx(-3.0103, abs=0.05)
        assert sc["phase_deg"] == pytest.approx(-45.0, abs=1.0)
