"""Tests for analysis tool handlers using mocked RawRead instances.

The classes at the bottom (``TestRecordedAcRaw`` / ``TestRecordedSteppedAcRaw``)
instead drive the handlers against real recorded LTspice binary raws from
``tests/fixtures/`` — see those classes for what the mocks cannot cover.
"""

from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from mcp import types

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import BatchJob, SessionState
from ltspice_mcp.tools.analysis import (
    AcStructureInput,
    BodeMetricsInput,
    EdgeMetricsInput,
    ExportWaveformInput,
    FilterMetricsInput,
    FindCrossingInput,
    GainAtInput,
    GetWaveformInput,
    MeasurementStatsInput,
    NoiseIntegralInput,
    OperatingPointInput,
    PeriodicMetricsInput,
    PulseResponseInput,
    QueryValueInput,
    ResonanceInput,
    RollOffInput,
    SignalStatsInput,
    SimulationSummaryInput,
    StabilityMetricsInput,
    ThdInput,
    TimingBetweenInput,
    _filter_operating_point,
    _split_ratio,
    _trace_device,
    handle_ac_structure,
    handle_bode_metrics,
    handle_edge_metrics,
    handle_export_waveform,
    handle_filter_metrics,
    handle_find_crossing,
    handle_gain_at,
    handle_get_waveform,
    handle_measurement_stats,
    handle_noise_integral,
    handle_operating_point,
    handle_periodic_metrics,
    handle_pulse_response,
    handle_query_value,
    handle_resonance,
    handle_roll_off,
    handle_signal_stats,
    handle_simulation_summary,
    handle_stability_metrics,
    handle_thd,
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


def _completed_batch(state: SessionState, run_results: dict, *, job_id: str = "b1") -> BatchJob:
    """Register a completed sweep BatchJob so its runs are reachable by
    ``job_id`` + ``run_index`` through the result read-model."""
    bj = BatchJob(
        job_id=job_id,
        job_type="sweep",
        netlist=Path("/tmp/x.cir"),
        total_runs=len(run_results),
        completed_runs=len(run_results),
        status="completed",
    )
    bj.run_results = run_results
    bj.completed_at = bj.started_at + timedelta(seconds=5)
    state.add_batch_job(bj)
    return bj


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

    async def test_dc_sweep_descending_axis(self, state_no_sim: SessionState, work_dir: Path):
        """A descending DC sweep (e.g. ``.dc V1 5 0 -0.25``) has a strictly
        decreasing axis. window_and_clean refuses that by default; the DC path
        opts into a flip so signal_stats analyzes it instead of erroring."""
        raw_file = work_dir / "dcdesc.raw"
        v = np.linspace(5.0, 0.0, 21)  # high → low sweep
        raw = _make_raw_mock(
            plotname="DC transfer characteristic",
            trace_names=["v-sweep", "V(out)"],
            waves={"v-sweep": v, "V(out)": v * 0.5},
            axis=v,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        data = result.structuredContent
        assert data is not None
        assert data["analysis_type"] == "dc"
        # min/max computed over the flipped-to-ascending axis.
        assert data["min"] == pytest.approx(0.0)
        assert data["max"] == pytest.approx(2.5)

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

    async def test_queried_bogus_param_warns(self, state_no_sim: SessionState, work_dir: Path):
        # A queried @-param the model doesn't expose is a fake 0.0; the
        # simulator's unrecognized-variable warning must be relayed on the
        # single-value read so the 0.0 isn't trusted.
        raw_file = work_dir / "q.raw"
        (work_dir / "q.log").write_text("Warning: unrecognized variable @m1[bogus]\n")
        raw = _make_raw_mock(
            trace_names=["time", "V(out)", "v(@m1[bogus])"],
            waves={
                "time": np.linspace(0, 1, 10),
                "V(out)": np.linspace(0, 1, 10),
                "v(@m1[bogus])": np.zeros(10),
            },
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="v(@m1[bogus])", at="0.5"),
            state_no_sim,
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert any("did not recognize" in w for w in warnings)

    async def test_unrecognized_not_relayed_for_other_signal(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Signal-filtered, not a dump: querying a healthy trace must NOT inherit
        # an unrecognized-variable warning about a different (@-param) trace.
        raw_file = work_dir / "q2.raw"
        (work_dir / "q2.log").write_text("Warning: unrecognized variable @m1[bogus]\n")
        raw = _make_raw_mock(
            trace_names=["time", "V(out)", "v(@m1[bogus])"],
            waves={
                "time": np.linspace(0, 1, 10),
                "V(out)": np.linspace(0, 1, 10),
                "v(@m1[bogus])": np.zeros(10),
            },
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="0.5"),
            state_no_sim,
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert not any("did not recognize" in w for w in warnings)

    async def test_solve_failure_taints_any_read(self, state_no_sim: SessionState, work_dir: Path):
        # A non-converged solve taints every value; a query of an otherwise
        # healthy trace must still surface the run-level failure.
        raw_file = work_dir / "q3.raw"
        (work_dir / "q3.log").write_text("gmin stepping failed\n")
        raw = _make_raw_mock(
            trace_names=["time", "V(out)"],
            waves={"time": np.linspace(0, 1, 10), "V(out)": np.linspace(0, 1, 10)},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="0.5"),
            state_no_sim,
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert any("gmin stepping" in w.lower() for w in warnings)

    async def test_clean_read_has_no_warnings(self, state_no_sim: SessionState, work_dir: Path):
        # No false positives: a healthy trace with a clean log carries no warnings.
        raw_file = work_dir / "q4.raw"
        (work_dir / "q4.log").write_text("Total elapsed time: 0.1 seconds.\n")
        raw = _make_raw_mock(
            trace_names=["time", "V(out)"],
            waves={"time": np.linspace(0, 1, 10), "V(out)": np.linspace(0, 1, 10)},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="0.5"),
            state_no_sim,
        )
        assert not (result.structuredContent or {}).get("warnings")


def test_has_active_device_detects_transistor_currents():
    # The empty-internals note fires only when an M/Q/J/D device is present
    # (its branch current appears), so RC circuits stay note-free. Sync test,
    # kept out of the asyncio-marked class so pytest-asyncio doesn't flag it.
    from ltspice_mcp.tools.analysis import _has_active_device

    assert _has_active_device({"Id(M1)": 1e-3, "V(out)": 5.0})
    assert _has_active_device({"Ic(Q2)": 1e-3})
    assert not _has_active_device({"I(R1)": 1e-3, "I(V1)": 2e-3})
    assert not _has_active_device({})


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

    async def test_dc_sweep_at_reads_chosen_point(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # at= reads the full bias snapshot at a chosen .dc sweep value (nearest),
        # not the sweep's first point.
        raw_file = work_dir / "dc.raw"
        raw = _make_raw_mock(
            plotname="DC transfer characteristic",
            trace_names=["v-sweep", "V(out)", "I(R1)"],
            waves={
                "v-sweep": np.array([0.0, 1.0, 2.0, 3.0]),
                "V(out)": np.array([10.0, 20.0, 30.0, 40.0]),
                "I(R1)": np.array([0.1, 0.2, 0.3, 0.4]),
            },
            axis=np.array([0.0, 1.0, 2.0, 3.0]),
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name, at="2.0"), state_no_sim
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["voltages"]["V(out)"] == 30.0
        assert sc["currents"]["I(R1)"] == pytest.approx(0.3)
        assert sc["sweep_value"] == 2.0

    async def test_carries_unrecognized_save_warning(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A .save'd @-param the model doesn't expose is written as a fake 0.0;
        # the simulator's unrecognized-variable warning (in the .log) must be
        # carried so the 0.0 isn't mistaken for a real gds=0/cgd=0.
        raw_file = work_dir / "op.raw"
        (work_dir / "op.log").write_text("Warning: unrecognized variable @m1[bogus]\n")
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(out)", "v(@m1[bogus])"],
            waves={"V(out)": np.array([1.5]), "v(@m1[bogus])": np.array([0.0])},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name), state_no_sim
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert any("unrecognized" in w.lower() for w in warnings)

    async def test_carries_solve_failure(self, state_no_sim: SessionState, work_dir: Path):
        # A non-converged/singular solve taints the whole bias snapshot; the
        # log's failure line is relayed onto the operating-point read.
        raw_file = work_dir / "opsf.raw"
        (work_dir / "opsf.log").write_text("gmin stepping failed\n")
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(out)"],
            waves={"V(out)": np.array([1.5])},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name), state_no_sim
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert any("gmin stepping" in w.lower() for w in warnings)

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


def _ac_ratio_raw(state: SessionState, work_dir: Path, *, fc: float = 1000.0, points: int = 400):
    """AC raw with V(out)=2*H_lpf and a flat V(mid)=2.0, so V(out)/V(mid)=H_lpf.

    The 2x factor only cancels if the ratio is actually divided — analyzing
    V(out) alone would read +6 dB in the passband."""
    raw_file = work_dir / "ac_ratio.raw"
    freqs = np.logspace(0, 6, points)
    s = 1j * 2 * np.pi * freqs
    wc = 2 * np.pi * fc
    h = wc / (s + wc)
    raw = _make_raw_mock(
        plotname="AC Analysis",
        trace_names=["frequency", "V(out)", "V(mid)"],
        waves={"frequency": freqs, "V(out)": 2.0 * h, "V(mid)": np.full_like(h, 2.0)},
        axis=freqs,
    )
    _inject_raw_mock(state, raw_file, raw)
    return raw_file


@pytest.mark.asyncio
class TestBodeRatioSignal:
    async def test_ratio_divides_two_traces(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_ratio_raw(state_no_sim, work_dir)
        result = await handle_gain_at(
            GainAtInput(raw_file=raw_file.name, signal="V(out)/V(mid)", frequencies=["1", "1k"]),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        # The 2x cancels: deep passband is 0 dB (not +6 dB), and fc is -3 dB.
        assert sc["points"][0]["magnitude_db"] == pytest.approx(0.0, abs=0.2)
        assert sc["points"][1]["magnitude_db"] == pytest.approx(-3.0, abs=0.2)

    async def test_ratio_missing_operand_errors(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_ratio_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="not found"):
            await handle_gain_at(
                GainAtInput(raw_file=raw_file.name, signal="V(out)/V(nope)", frequencies=["1k"]),
                state_no_sim,
            )

    async def test_ratio_singular_denominator_reported(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # V(mid) crosses zero at one bin -> the ratio is singular there. It must
        # be reported, not silently dropped (which would hide a pole and skew
        # the metrics computed over the gapped sweep).
        raw_file = work_dir / "ac_singular.raw"
        freqs = np.logspace(0, 6, 400)
        mid = (freqs - freqs[200]).astype(complex)  # exact zero at index 200
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)", "V(mid)"],
            waves={
                "frequency": freqs,
                "V(out)": np.ones_like(freqs, dtype=complex),
                "V(mid)": mid,
            },
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="singular"):
            await handle_gain_at(
                GainAtInput(raw_file=raw_file.name, signal="V(out)/V(mid)", frequencies=["1k"]),
                state_no_sim,
            )


class TestSplitRatio:
    def test_plain_signal_is_not_ratio(self):
        assert _split_ratio("V(out)") is None

    def test_two_operand_ratio(self):
        assert _split_ratio("V(out)/V(mid)") == ("V(out)", "V(mid)")

    def test_three_operands_rejected(self):
        with pytest.raises(ResultError, match="exactly 'A/B'"):
            _split_ratio("V(a)/V(b)/V(c)")

    def test_empty_operand_rejected(self):
        with pytest.raises(ResultError, match="exactly 'A/B'"):
            _split_ratio("V(out)/")


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

    async def test_step_axis_relays_unrecognized_param(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # The stepped read path must relay the unrecognized-variable warning too:
        # a bogus @-param is a fake 0.0 here just like on the direct at= path.
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "v(@m1[bogus])"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([0.0, 0.0])
        path = work_dir / "qsw1.raw"
        (work_dir / "qsw1.log").write_text("Warning: unrecognized variable @m1[bogus]\n")
        _inject_raw(state_no_sim, path, raw)
        res = await handle_query_value(
            QueryValueInput(
                raw_file="qsw1.raw",
                signal="v(@m1[bogus])",
                step_axis="Rval",
                step_value="1000",
            ),
            state_no_sim,
        )
        warnings = (res.structuredContent or {}).get("warnings") or []
        assert any("did not recognize" in w for w in warnings)

    async def test_step_axis_relays_solve_failure(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A failed solve taints the whole run; the stepped read must surface it.
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "qsw2.raw"
        (work_dir / "qsw2.log").write_text("gmin stepping failed\n")
        _inject_raw(state_no_sim, path, raw)
        res = await handle_query_value(
            QueryValueInput(
                raw_file="qsw2.raw", signal="V(out)", step_axis="Rval", step_value="1000"
            ),
            state_no_sim,
        )
        warnings = (res.structuredContent or {}).get("warnings") or []
        assert any("gmin stepping" in w.lower() for w in warnings)

    async def test_step_axis_clean_no_diagnostic_relay(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # No false positives: a clean log adds no unrecognized/solve-failure relay
        # (the step lookup's own 'No at given' note is unrelated and allowed).
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [{"Rval": 500.0}, {"Rval": 1000.0}]
        raw.get_axis.return_value = np.array([0.0, 1.0])
        raw.get_wave = lambda name, step=0: np.array([1.0, 2.0])
        path = work_dir / "qsw3.raw"
        (work_dir / "qsw3.log").write_text("Total elapsed time: 0.1 seconds.\n")
        _inject_raw(state_no_sim, path, raw)
        res = await handle_query_value(
            QueryValueInput(
                raw_file="qsw3.raw", signal="V(out)", step_axis="Rval", step_value="1000"
            ),
            state_no_sim,
        )
        warnings = (res.structuredContent or {}).get("warnings") or []
        assert not any("did not recognize" in w or "singular" in w.lower() for w in warnings)


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

    async def test_all_steps_dedups_identical_step_warnings(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Three lowpass steps, mode='filter' with NO stopband_range: every step
        # emits the identical sweep-endpoint rejection warning. all_steps must
        # hoist it to the top level ONCE with a coverage note, drop the per-step
        # 'warnings' key, and strip it from each per-step text block.
        fcs = [500.0, 1000.0, 5000.0]
        path = work_dir / "stepped_dedup.raw"
        _inject_raw(state_no_sim, path, _stepped_ac_raw(fcs))
        res = await handle_bode_metrics(
            BodeMetricsInput(
                raw_file="stepped_dedup.raw",
                signal="V(out)",
                mode="filter",
                all_steps=True,
            ),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["step_count"] == 3
        warnings = sc.get("warnings", [])
        # The sweep-endpoint rejection warning every step emits.
        sentinel = "no stopband_range given"
        hoisted = [w for w in warnings if sentinel in w]
        # Hoisted exactly once (not once per step).
        assert len(hoisted) == 1
        # Carries a coverage note naming the steps it covered.
        assert "steps" in hoisted[0]
        assert "all 3 steps" in hoisted[0]
        # Per-step structured entries no longer carry that warning under a
        # 'warnings' key — it was popped during hoisting.
        for entry in sc["steps"]:
            assert sentinel not in entry.get("warnings", [])
            assert not any(sentinel in w for w in entry.get("warnings", []))
        # The per-step text blocks no longer repeat the warning either.
        text = res.content[0].text
        assert text.count(sentinel) == 1

    def test_warning_coverage_lists_indices_for_large_subset(self):
        # A warning on a >6 SUBSET of steps must enumerate every affected step
        # index, not collapse to a bare "N of M" count — otherwise a structured
        # consumer can't tell which sweep cases emitted it.
        from ltspice_mcp.tools.analysis import _warning_coverage

        idxs = [0, 2, 4, 6, 8, 10, 12]  # 7 of 20 — past the old 6-item cap
        cov = _warning_coverage(idxs, 20)
        for i in idxs:
            assert str(i) in cov
        assert "of 20" not in cov  # not collapsed to a count
        # Every-step case stays compact.
        assert _warning_coverage(list(range(5)), 5) == "all 5 steps"
        # A small subset still enumerates.
        assert _warning_coverage([1, 3], 5) == "steps 1,3"

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

    async def test_by_job_run_echoes_swept_params(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Analyzing a sweep run by job_id+run_index echoes which sweep point it
        # is (params + run_index), so no extra batch_results call is needed.
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": str(raw), "params": {"R": "1k"}}},
        )
        res = await handle_bode_metrics(
            BodeMetricsInput(job_id="b1", run_index=0, signal="V(out)", mode="filter"),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["run_index"] == 0
        assert sc["params"] == {"R": "1k"}

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


@pytest.mark.asyncio
class TestQueryValueArgErrorHints:
    """Argument-shape ResultErrors from query_value must NOT trigger the generic
    'check_job for details' dispatch hint — they are caller mistakes, not run
    failures, so they carry ``show_hint=False``."""

    async def test_step_axis_with_job_id_conflict_no_hint(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # step_axis selects a step of a .step raw; job_id already selects a run.
        # The conflict is a caller mistake — suppress the generic hint.
        path = work_dir / "conflict.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError) as excinfo:
            await handle_query_value(
                QueryValueInput(
                    raw_file="conflict.raw",
                    signal="V(out)",
                    step_axis="Rval",
                    step_value="1k",
                    job_id="job-123",
                ),
                state_no_sim,
            )
        assert excinfo.value.show_hint is False

    async def test_missing_at_no_hint(self, state_no_sim: SessionState, work_dir: Path):
        # 'at' omitted on a normal (non-step_axis) raw is a caller mistake.
        path = work_dir / "needat.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError) as excinfo:
            await handle_query_value(
                QueryValueInput(raw_file="needat.raw", signal="V(out)"),
                state_no_sim,
            )
        assert excinfo.value.show_hint is False


@pytest.mark.asyncio
class TestSimulationSummaryBuildFailureHint:
    """When build_simulation_summary itself raises, the wrapping ResultError must
    suppress the generic hint (it would point back at simulation_summary, the
    very tool that just failed)."""

    async def test_build_failure_show_hint_false(
        self, state_no_sim: SessionState, fake_raw: Path, monkeypatch
    ):
        # The raw loads fine; force build_simulation_summary to raise so we hit
        # the self-referential-hint suppression path.
        import ltspice_mcp.tools.analysis as analysis_mod

        def _boom(*_args, **_kwargs):
            raise ValueError("synthetic build failure")

        monkeypatch.setattr(analysis_mod, "build_simulation_summary", _boom)
        with pytest.raises(ResultError) as excinfo:
            await handle_simulation_summary(
                SimulationSummaryInput(raw_file=fake_raw.name), state_no_sim
            )
        assert excinfo.value.show_hint is False
        # Must not re-suggest the tool that just failed.
        assert "simulation_summary" not in str(excinfo.value)


@pytest.mark.asyncio
class TestGetWaveform:
    """get_waveform decimates one real-valued signal into a min/max-preserving
    stat-envelope. The autouse output-schema conformance hook validates the
    structuredContent shape on every successful call, so each happy-path case
    here is also a schema-conformance test."""

    async def test_no_axis_op_raw_rejected(self, state_no_sim: SessionState, work_dir: Path):
        # A real Operating Point raw has no axis; _guarded_axis must surface a
        # clean ResultError pointing at operating_point, not a generic crash.
        raw = _stage_recorded(work_dir, "op_extreme_node")
        with pytest.raises(ResultError, match="operating_point"):
            await handle_get_waveform(
                GetWaveformInput(raw_file=raw.name, signal="V(hot)"), state_no_sim
            )

    async def test_transient_envelope_invariants(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "wave.raw"
        t = np.linspace(0, 1, 200)
        y = np.sin(2 * np.pi * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "transient"
        assert sc["bucket_count"] > 0
        assert sc["point_count"] > 0
        assert len(sc["buckets"]) == sc["bucket_count"]
        for b in sc["buckets"]:
            assert b["min"] <= b["mean"] <= b["max"]
            assert b["rms"] >= 0
            assert b["pk_pk"] == pytest.approx(b["max"] - b["min"])

    async def test_dc_sweep_descending_axis(self, state_no_sim: SessionState, work_dir: Path):
        # A descending DC sweep must be decimated, not refused: the DC path
        # flips the axis to ascending before bucketing.
        raw_file = work_dir / "wave_dcdesc.raw"
        v = np.linspace(5.0, 0.0, 200)  # high → low
        raw = _make_raw_mock(
            plotname="DC transfer characteristic",
            trace_names=["v-sweep", "V(out)"],
            waves={"v-sweep": v, "V(out)": v * 0.5},
            axis=v,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "dc"
        assert sc["bucket_count"] > 0
        # The envelope spans the full sweep regardless of original direction.
        assert min(b["min"] for b in sc["buckets"]) == pytest.approx(0.0, abs=0.05)
        assert max(b["max"] for b in sc["buckets"]) == pytest.approx(2.5, abs=0.05)

    async def test_decimated_observations(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "wave.raw"
        t = np.linspace(0, 1, 1000)
        y = np.sin(2 * np.pi * 5 * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)", buckets=10),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["decimated"] is True
        codes = {(o["code"], o["kind"]) for o in sc["observations"]}
        assert ("decimated", "coverage") in codes
        assert "max_pk_pk_bucket" in {o["code"] for o in sc["observations"]}

    async def test_bucket_cap_from_config(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "wave.raw"
        t = np.linspace(0, 1, 500)
        y = np.cos(2 * np.pi * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        state_no_sim.config.max_points_returned = 5

        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)", buckets=1000),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["bucket_count"] <= 5
        assert sc["max_points_ceiling"] == 5

    async def test_ac_complex_rejected_points_to_bode(
        self, state_no_sim: SessionState, work_dir: Path
    ):
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
        with pytest.raises(ResultError, match="bode_metrics"):
            await handle_get_waveform(
                GetWaveformInput(raw_file=raw_file.name, signal="V(out)"),
                state_no_sim,
            )

    async def test_recorded_transient_fixture(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_tran_rc")
        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "transient"
        assert sc["bucket_count"] > 0

    async def test_recorded_dc_fixture_accepted(self, state_no_sim: SessionState, work_dir: Path):
        # get_waveform accepts a .DC sweep raw (it only rejects complex AC);
        # the real Plotname 'DC transfer characteristic' must classify as 'dc'.
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "dc"
        assert sc["bucket_count"] > 0

    async def test_narrower_window_zoom(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "wave.raw"
        t = np.linspace(0, 1, 400)
        y = np.sin(2 * np.pi * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        full = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        full_sc = full.structuredContent
        assert full_sc is not None

        zoom = await handle_get_waveform(
            GetWaveformInput(
                raw_file=raw_file.name,
                signal="V(out)",
                t_start="0.25",
                t_end="0.75",
            ),
            state_no_sim,
        )
        zoom_sc = zoom.structuredContent
        assert zoom_sc is not None
        # The narrower window is strictly inside the full window.
        assert zoom_sc["window_start_used"] >= full_sc["window_start_used"]
        assert zoom_sc["window_end_used"] <= full_sc["window_end_used"]
        assert zoom_sc["window_start_used"] >= 0.25 - 1e-9
        assert zoom_sc["window_end_used"] <= 0.75 + 1e-9
        # Every bucket's x-range stays within the requested zoom bounds.
        for b in zoom_sc["buckets"]:
            assert b["x_start"] >= 0.25 - 1e-9
            assert b["x_end"] <= 0.75 + 1e-9

    async def test_noise_classification(self, state_no_sim: SessionState, work_dir: Path):
        # LTspice's real Plotname for a .noise run is "Noise Spectral Density -
        # (V/Hz½)"; the axis is frequency (Hz), not time, and the wave is a real,
        # positive spectral density. get_waveform must classify it as 'noise' and
        # label the axis 'Hz'.
        raw_file = work_dir / "noise.raw"
        freqs = np.logspace(0, 6, 100)
        density = 1e-9 / np.sqrt(1 + (freqs / 1000) ** 2)
        raw = _make_raw_mock(
            plotname="Noise Spectral Density - (V/Hz½)",
            trace_names=["frequency", "V(onoise)"],
            waves={"frequency": freqs, "V(onoise)": density},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(onoise)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["analysis_type"] == "noise"
        assert sc["axis_unit"] == "Hz"

    async def test_crest_factor_none_for_zero_signal(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # An all-zero wave makes every bucket's rms == 0, so crest_factor (peak/rms)
        # is undefined and surfaced as null. Exercising it through the handler
        # carries the None through format_response + the autouse schema-conformance
        # hook, proving the WaveformBucket schema accepts a null crest_factor.
        raw_file = work_dir / "zero.raw"
        t = np.linspace(0, 1, 200)
        y = np.zeros(200)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await handle_get_waveform(
            GetWaveformInput(raw_file=raw_file.name, signal="V(out)"),
            state_no_sim,
        )
        sc = result.structuredContent
        assert sc is not None
        assert sc["buckets"]
        assert any(b["crest_factor"] is None for b in sc["buckets"])
        # And the rms that drove it to None really is zero.
        for b in sc["buckets"]:
            if b["crest_factor"] is None:
                assert b["rms"] == pytest.approx(0.0, abs=1e-12)

    async def test_sub_three_sample_window_rejected(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A coarse 5-point axis with a window that brackets a single sample:
        # [1.5, 2.5] slices to index [2:3] (one sample). window_and_clean needs
        # at least 3 samples, so the handler must RAISE a ResultError, not return
        # a degenerate one-bucket envelope.
        raw_file = work_dir / "coarse.raw"
        t = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)

        with pytest.raises(ResultError, match="at least 3"):
            await handle_get_waveform(
                GetWaveformInput(
                    raw_file=raw_file.name, signal="V(out)", t_start="1.5", t_end="2.5"
                ),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestDcRejectedByTransientTools:
    """Regression: a .DC sweep raw produces a voltage (not time) axis, so the
    transient-only tools (edge/pulse/periodic/timing) must refuse it instead of
    reading meaningless rise-times off it. A real recorded raw is used so the
    actual Plotname 'DC transfer characteristic' flows through the reject path
    (a mock could mask the classification)."""

    @pytest.mark.parametrize(
        ("handler", "input_factory"),
        [
            (handle_edge_metrics, lambda name: EdgeMetricsInput(raw_file=name, signal="V(out)")),
            (
                handle_pulse_response,
                lambda name: PulseResponseInput(raw_file=name, signal="V(out)"),
            ),
            (
                handle_periodic_metrics,
                lambda name: PeriodicMetricsInput(raw_file=name, signal="V(out)"),
            ),
            (
                handle_timing_between,
                lambda name: TimingBetweenInput(
                    raw_file=name, signal_a="V(out)", signal_b="V(out)"
                ),
            ),
        ],
        ids=["edge_metrics", "pulse_response", "periodic_metrics", "timing_between"],
    )
    async def test_transient_tools_reject_dc_raw(
        self, state_no_sim: SessionState, work_dir: Path, handler, input_factory
    ):
        # timing_between rejects at a DISTINCT call site from the edge/pulse/
        # periodic loader, so it gets its own parametrize case rather than relying
        # on the shared one being exercised.
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        with pytest.raises(ResultError, match="transient"):
            await handler(input_factory(raw.name), state_no_sim)


@pytest.mark.asyncio
class TestAnalysisToolsJobRun:
    """A completed sweep/MC run must be analyzable by ``job_id`` + ``run_index``,
    reaching the same raw — and returning the same result — as addressing that
    run's raw by path. A reloaded job's raw can live outside ``allowed_paths``
    (e.g. a WSL temp dir), so the job-run path deliberately skips ``safe_path``;
    these address the run by index and compare against the by-path call.
    """

    async def test_signal_stats_by_job_run(self, state_no_sim: SessionState, work_dir: Path):
        p0, p1 = work_dir / "ss_run0.raw", work_dir / "ss_run1.raw"
        t = np.linspace(0, 1, 100)
        _inject_raw_mock(state_no_sim, p0, _make_raw_mock(waves={"time": t, "V(out)": t}, axis=t))
        _inject_raw_mock(
            state_no_sim, p1, _make_raw_mock(waves={"time": t, "V(out)": 2.0 * t}, axis=t)
        )
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        by_job = await handle_signal_stats(
            SignalStatsInput(job_id="b1", run_index=1, signal="V(out)"), state_no_sim
        )
        by_path = await handle_signal_stats(
            SignalStatsInput(raw_file=p1.name, signal="V(out)"), state_no_sim
        )
        assert by_job.structuredContent == by_path.structuredContent
        # run_index actually selected run 1 (2x amplitude), not run 0.
        assert by_job.structuredContent["max"] == pytest.approx(2.0, abs=1e-6)

    async def test_operating_point_by_job_run(self, state_no_sim: SessionState, work_dir: Path):
        p0, p1 = work_dir / "op_run0.raw", work_dir / "op_run1.raw"
        _inject_raw_mock(
            state_no_sim,
            p0,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(out)"],
                waves={"V(out)": np.array([1.0])},
            ),
        )
        _inject_raw_mock(
            state_no_sim,
            p1,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(out)"],
                waves={"V(out)": np.array([2.5])},
            ),
        )
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        by_job = await handle_operating_point(
            OperatingPointInput(job_id="b1", run_index=1), state_no_sim
        )
        by_path = await handle_operating_point(OperatingPointInput(raw_file=p1.name), state_no_sim)
        assert by_job.structuredContent == by_path.structuredContent
        assert by_job.structuredContent["voltages"]["V(out)"] == pytest.approx(2.5)

    async def test_edge_metrics_by_job_run(self, state_no_sim: SessionState, work_dir: Path):
        p0, p1 = work_dir / "em_run0.raw", work_dir / "em_run1.raw"
        t, y = _step_waveform()
        _inject_raw_mock(state_no_sim, p0, _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t))
        _inject_raw_mock(state_no_sim, p1, _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t))
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        by_job = await handle_edge_metrics(
            EdgeMetricsInput(job_id="b1", run_index=1, signal="V(out)"), state_no_sim
        )
        by_path = await handle_edge_metrics(
            EdgeMetricsInput(raw_file=p1.name, signal="V(out)"), state_no_sim
        )
        assert by_job.structuredContent == by_path.structuredContent
        assert by_job.structuredContent["is_rise_time"] is True

    async def test_stability_metrics_by_job_run(self, state_no_sim: SessionState, work_dir: Path):
        freqs = np.logspace(0, 8, 500)
        s = 1j * 2 * np.pi * freqs
        H = 1000.0 / ((1 + s / (2 * np.pi * 1000)) * (1 + s / (2 * np.pi * 100000)))
        p0, p1 = work_dir / "sm_run0.raw", work_dir / "sm_run1.raw"
        for p in (p0, p1):
            _inject_raw_mock(
                state_no_sim,
                p,
                _make_raw_mock(
                    plotname="AC Analysis",
                    trace_names=["frequency", "V(loop)"],
                    waves={"frequency": freqs, "V(loop)": H},
                    axis=freqs,
                ),
            )
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        by_job = await handle_stability_metrics(
            StabilityMetricsInput(job_id="b1", run_index=1, signal="V(loop)"), state_no_sim
        )
        by_path = await handle_stability_metrics(
            StabilityMetricsInput(raw_file=p1.name, signal="V(loop)"), state_no_sim
        )
        assert by_job.structuredContent == by_path.structuredContent
        assert by_job.structuredContent["dc_gain_db"] == pytest.approx(60.0, abs=0.1)

    async def test_resonance_by_job_run(self, state_no_sim: SessionState, work_dir: Path):
        freqs = np.logspace(1, 5, 3000)
        s = 1j * 2 * np.pi * freqs
        w0 = 2 * np.pi * 1000
        H = (w0 * w0) / (s * s + (w0 / 10.0) * s + w0 * w0)
        p0, p1 = work_dir / "rs_run0.raw", work_dir / "rs_run1.raw"
        for p in (p0, p1):
            _inject_raw_mock(
                state_no_sim,
                p,
                _make_raw_mock(
                    plotname="AC Analysis",
                    trace_names=["frequency", "V(out)"],
                    waves={"frequency": freqs, "V(out)": H},
                    axis=freqs,
                ),
            )
        _completed_batch(
            state_no_sim,
            {0: {"raw_file": p0, "params": {}}, 1: {"raw_file": p1, "params": {}}},
        )
        by_job = await handle_resonance(
            ResonanceInput(job_id="b1", run_index=1, signal="V(out)"), state_no_sim
        )
        by_path = await handle_resonance(
            ResonanceInput(raw_file=p1.name, signal="V(out)"), state_no_sim
        )
        assert by_job.structuredContent == by_path.structuredContent
        assert by_job.structuredContent["peaks"][0]["frequency_hz"] == pytest.approx(
            1000.0, rel=0.05
        )

    async def test_raw_file_and_job_id_mutually_exclusive(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_signal_stats(
                SignalStatsInput(raw_file="x.raw", job_id="b1", signal="V(out)"),
                state_no_sim,
            )

    async def test_neither_raw_file_nor_job_id(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_signal_stats(SignalStatsInput(signal="V(out)"), state_no_sim)


@pytest.mark.asyncio
class TestSignalStatsAnalysisTypeRobustness:
    """signal_stats on raws without the usual transient shape must not crash."""

    async def test_noise_raw_omits_mean(self, state_no_sim: SessionState, work_dir: Path):
        # A .noise raw has a real, positive spectral density over a frequency
        # axis. signal_stats must classify it 'noise' and return min/max/pk-pk
        # WITHOUT a 'mean' (a plain mean of spectral density is meaningless) —
        # the text formatter must not KeyError on the absent 'mean'.
        raw_file = work_dir / "noise_stats.raw"
        freqs = np.logspace(0, 6, 100)
        density = 1e-9 / np.sqrt(1 + (freqs / 1000) ** 2)
        raw = _make_raw_mock(
            plotname="Noise Spectral Density - (V/Hz½)",
            trace_names=["frequency", "V(onoise)"],
            waves={"frequency": freqs, "V(onoise)": density},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_signal_stats(
            SignalStatsInput(raw_file=raw_file.name, signal="V(onoise)"), state_no_sim
        )
        sc = result.structuredContent
        assert sc["analysis_type"] == "noise"
        assert "mean" not in sc
        assert "min" in sc
        assert "max" in sc
        assert "peak_to_peak" in sc

    async def test_op_raw_rejected_with_pointer(self, state_no_sim: SessionState, work_dir: Path):
        # A real Operating Point raw has no data axis. signal_stats must raise a
        # clean ResultError pointing at operating_point, not a generic internal
        # error / RuntimeError from spicelib's get_axis.
        raw = _stage_recorded(work_dir, "op_extreme_node")
        with pytest.raises(ResultError, match="operating_point"):
            await handle_signal_stats(
                SignalStatsInput(raw_file=raw.name, signal="V(hot)"), state_no_sim
            )


class TestTraceDeviceFilter:
    """Pure helpers behind operating_point's device= filter."""

    def test_trace_device_owner(self):
        assert _trace_device("@m1[gm]") == "m1"
        assert _trace_device("v(@m1[vth])") == "m1"
        assert _trace_device("i(@m1[id])") == "m1"
        assert _trace_device("Id(M1)") == "m1"
        assert _trace_device("Ic(Q2)") == "q2"
        assert _trace_device("I(R1)") == "r1"
        assert _trace_device("V(out)") is None

    def test_filter_narrows_to_device(self):
        op = {
            "voltages": {"V(d)": 1.8, "V(g)": 0.9},
            "currents": {"Id(M1)": 1e-3, "I(R1)": 2e-3},
            "device_internals": {"@m1[gm]": 1e-3, "@m2[gm]": 2e-3},
        }
        assert _filter_operating_point(op, "M1") is True
        assert op["currents"] == {"Id(M1)": 1e-3}
        assert op["device_internals"] == {"@m1[gm]": 1e-3}
        # Node voltages are not device-scoped -> dropped from the focused view.
        assert op["voltages"] == {}

    def test_filter_matches_subcircuit_path_suffix(self):
        op = {"voltages": {}, "currents": {}, "device_internals": {"@m.x1.mn[gm]": 5.0}}
        assert _filter_operating_point(op, "mn") is True
        assert op["device_internals"] == {"@m.x1.mn[gm]": 5.0}

    def test_filter_no_match_reports_false(self):
        op = {"voltages": {}, "currents": {"Id(M1)": 1.0}, "device_internals": {}}
        assert _filter_operating_point(op, "Q9") is False


@pytest.mark.asyncio
class TestOperatingPointDeviceAndUnits:
    """device= narrows to one device's internals + terminal currents (2c); every
    value carries its unit where derivable (2a)."""

    def _op_raw(self, state: SessionState, work_dir: Path) -> Path:
        raw_file = work_dir / "op_dev.raw"
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(d)", "V(g)", "Id(M1)", "Ig(M1)", "I(R1)", "@m1[gm]", "@m2[gm]"],
            waves={
                "V(d)": np.array([1.8]),
                "V(g)": np.array([0.9]),
                "Id(M1)": np.array([1e-3]),
                "Ig(M1)": np.array([0.0]),
                "I(R1)": np.array([2e-3]),
                "@m1[gm]": np.array([1.5e-3]),
                "@m2[gm]": np.array([2.5e-3]),
            },
            axis=np.array([0.0]),
        )
        _inject_raw_mock(state, raw_file, raw)
        return raw_file

    async def test_units_on_full_readout(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = self._op_raw(state_no_sim, work_dir)
        res = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name), state_no_sim
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["units"]["V(d)"] == "V"
        assert sc["units"]["Id(M1)"] == "A"
        # A device-internal parameter gets no guessed unit.
        assert "@m1[gm]" not in sc["units"]

    async def test_device_filter_focuses_one_device(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw_file = self._op_raw(state_no_sim, work_dir)
        res = await handle_operating_point(
            OperatingPointInput(raw_file=raw_file.name, device="M1"), state_no_sim
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["device"] == "M1"
        assert set(sc["currents"]) == {"Id(M1)", "Ig(M1)"}
        assert set(sc["device_internals"]) == {"@m1[gm]"}
        assert sc["voltages"] == {}
        assert sc["units"]["Id(M1)"] == "A"

    async def test_unknown_device_lists_present_ones(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw_file = self._op_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="Devices present"):
            await handle_operating_point(
                OperatingPointInput(raw_file=raw_file.name, device="Q9"), state_no_sim
            )


@pytest.mark.asyncio
class TestQueryValueDcLabelAndUnit:
    async def test_dc_sweep_labels_swept_axis_and_carries_unit(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        res = await handle_query_value(
            QueryValueInput(raw_file=str(raw), signal="V(out)", at="2"), state_no_sim
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["unit"] == "V"
        text = res.content[0].text  # type: ignore[union-attr]
        # The DC sweep axis is the swept variable, not time.
        assert " at t=" not in text


@pytest.mark.asyncio
class TestNoiseIntegralHandler:
    async def test_real_noise_fixture(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_noise_rc")
        res = await handle_noise_integral(
            NoiseIntegralInput(raw_file=str(raw), signal="V(onoise)"), state_no_sim
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["total_rms"] > 0
        assert sc["n_points"] > 1
        assert sc["unit"] == "V"
        assert "Hz" in sc["density_unit"]

    async def test_rejects_transient_raw(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="noise"):
            await handle_noise_integral(NoiseIntegralInput(raw_file=str(raw)), state_no_sim)


@pytest.mark.asyncio
class TestExportDcHeader:
    async def test_dc_x_header_names_swept_axis(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        res = await handle_export_waveform(
            ExportWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
        )
        sc = res.structuredContent
        assert sc is not None
        # The x-column is the named swept variable, not the bare "sweep".
        assert sc["columns"][0] != "sweep"
        assert sc["columns"][0].lower() not in ("time_s", "freq_hz")


@pytest.mark.asyncio
class TestThdHandler:
    async def test_thd_on_synthetic_periodic_raw(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "thd.raw"
        f0, fs = 1000.0, 200_000.0
        t = np.arange(0.0, 0.02, 1.0 / fs)
        y = np.sin(2 * np.pi * f0 * t) + 0.1 * np.sin(2 * np.pi * 2 * f0 * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        res = await handle_thd(
            ThdInput(raw_file=raw_file.name, signal="V(out)", fundamental="1k", n_harmonics=3),
            state_no_sim,
        )
        sc = res.structuredContent
        assert sc is not None
        assert sc["thd_ratio"] == pytest.approx(0.1, rel=1e-2)
        assert sc["coherent"] is True


def _ac_response_raw(signal: str, h: np.ndarray, freqs: np.ndarray) -> MagicMock:
    """An AC raw mock carrying one complex response under ``signal``."""
    return _make_raw_mock(
        plotname="AC Analysis",
        trace_names=["frequency", signal],
        waves={"frequency": freqs, signal: h},
        axis=freqs,
    )


async def _assert_relays_solve_failure(
    state: SessionState, work_dir: Path, name: str, raw: MagicMock, handler, inp_factory
) -> None:
    """Drive ``handler`` against a raw whose sibling .log reports a singular
    matrix, and assert the failure surfaces in the rendered result. Covers the
    structural guarantee that every raw-reading metric tool relays a run-level
    solve failure — a new tool that forgets the relay fails here."""
    raw_file = work_dir / f"{name}.raw"
    _inject_raw_mock(state, raw_file, raw)
    (work_dir / f"{name}.log").write_text("gmin stepping failed\n")
    result = await handler(inp_factory(raw_file.name), state)
    assert "gmin stepping" in result.content[0].text.lower(), (
        f"{name} did not relay the run-level solve failure"
    )


@pytest.mark.asyncio
class TestSolveFailureRelayCoverage:
    """Every raw-reading metric tool relays a completed-but-failed solve."""

    async def test_transient_tools(self, state_no_sim: SessionState, work_dir: Path):
        t_step, y_step = _step_waveform()
        t_sq, y_sq = _square_wave(freq=1000.0, duty=0.5, periods=10)
        fs = 200000.0
        t_sin = np.arange(0.0, 0.02, 1.0 / fs)
        y_sin = np.sin(2 * np.pi * 1000.0 * t_sin) + 0.1 * np.sin(2 * np.pi * 2000.0 * t_sin)

        cases = [
            (
                "ss",
                _make_raw_mock(),
                handle_signal_stats,
                lambda n: SignalStatsInput(raw_file=n, signal="V(out)"),
            ),
            (
                "edge",
                _make_raw_mock(waves={"time": t_step, "V(out)": y_step}, axis=t_step),
                handle_edge_metrics,
                lambda n: EdgeMetricsInput(raw_file=n, signal="V(out)"),
            ),
            (
                "pulse",
                _make_raw_mock(waves={"time": t_step, "V(out)": y_step}, axis=t_step),
                handle_pulse_response,
                lambda n: PulseResponseInput(
                    raw_file=n, signal="V(out)", initial_value=0.0, final_value=1.0
                ),
            ),
            (
                "timing",
                _make_raw_mock(
                    trace_names=["time", "V(in)", "V(out)"],
                    waves={"time": t_step, "V(in)": y_step, "V(out)": y_step},
                    axis=t_step,
                ),
                handle_timing_between,
                lambda n: TimingBetweenInput(raw_file=n, signal_a="V(in)", signal_b="V(out)"),
            ),
            (
                "periodic",
                _make_raw_mock(
                    trace_names=["time", "V(clk)"], waves={"time": t_sq, "V(clk)": y_sq}, axis=t_sq
                ),
                handle_periodic_metrics,
                lambda n: PeriodicMetricsInput(raw_file=n, signal="V(clk)"),
            ),
            (
                "thd",
                _make_raw_mock(waves={"time": t_sin, "V(out)": y_sin}, axis=t_sin),
                handle_thd,
                lambda n: ThdInput(raw_file=n, signal="V(out)", fundamental="1k", n_harmonics=3),
            ),
        ]
        for name, raw, handler, factory in cases:
            await _assert_relays_solve_failure(state_no_sim, work_dir, name, raw, handler, factory)

    async def test_ac_tools(self, state_no_sim: SessionState, work_dir: Path):
        freqs = np.logspace(0, 6, 200)
        s = 2j * np.pi * freqs
        lpf = 1.0 / (1 + s / (2 * np.pi * 1000))
        loop = 1000.0 / ((1 + s / (2 * np.pi * 1000)) * (1 + s / (2 * np.pi * 100000)))
        w0 = 2 * np.pi * 1000
        peak = (w0 * w0) / (s * s + (w0 / 10.0) * s + w0 * w0)

        cases = [
            (
                "stab",
                _ac_response_raw("V(loop)", loop, freqs),
                handle_stability_metrics,
                lambda n: StabilityMetricsInput(raw_file=n, signal="V(loop)"),
            ),
            (
                "reson",
                _ac_response_raw("V(out)", peak, freqs),
                handle_resonance,
                lambda n: ResonanceInput(raw_file=n, signal="V(out)"),
            ),
            (
                "acstruct",
                _ac_response_raw("V(out)", lpf, freqs),
                handle_ac_structure,
                lambda n: AcStructureInput(raw_file=n, signal="V(out)"),
            ),
            (
                "bode1",
                _ac_response_raw("V(out)", lpf, freqs),
                handle_bode_metrics,
                lambda n: BodeMetricsInput(
                    raw_file=n, signal="V(out)", mode="point", frequencies=["1k"]
                ),
            ),
        ]
        for name, raw, handler, factory in cases:
            await _assert_relays_solve_failure(state_no_sim, work_dir, name, raw, handler, factory)

    async def test_bode_all_steps(self, state_no_sim: SessionState, work_dir: Path):
        await _assert_relays_solve_failure(
            state_no_sim,
            work_dir,
            "bodeall",
            _stepped_ac_raw([500.0, 5000.0]),
            handle_bode_metrics,
            lambda n: BodeMetricsInput(
                raw_file=n,
                signal="V(out)",
                mode="crossing",
                quantity="magnitude_db",
                level=-3.0103,
                all_steps=True,
            ),
        )

    async def test_recovered_singular_matrix_not_flagged(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A transient can recover from a singular matrix via gmin/source stepping
        # and still produce a valid raw, so a bare warning-level "singular matrix"
        # must NOT be promoted to a run-wide failure — that would be a false
        # accusation on a good run. Only terminal phrases taint the read.
        raw_file = work_dir / "recov.raw"
        (work_dir / "recov.log").write_text("Warning: singular matrix:  check nodes out and 0\n")
        raw = _make_raw_mock(
            trace_names=["time", "V(out)"],
            waves={"time": np.linspace(0, 1, 10), "V(out)": np.linspace(0, 1, 10)},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await handle_query_value(
            QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="0.5"), state_no_sim
        )
        warnings = (result.structuredContent or {}).get("warnings") or []
        assert not any("singular" in w.lower() for w in warnings)

    async def test_degenerate_raise_names_solve_failure(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # When a failed solve leaves data degenerate enough that the metric itself
        # raises (a flat waveform has no edge), the error must name the solve
        # failure from the log instead of only the generic "no edge" message.
        raw_file = work_dir / "flat.raw"
        t = np.linspace(0, 1e-3, 1000)
        raw = _make_raw_mock(waves={"time": t, "V(out)": np.ones_like(t)}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        (work_dir / "flat.log").write_text("gmin stepping failed\n")
        with pytest.raises(ResultError, match="gmin stepping"):
            await handle_edge_metrics(
                EdgeMetricsInput(raw_file=raw_file.name, signal="V(out)"), state_no_sim
            )
