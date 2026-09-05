"""The numeric core behind every analyze_results recipe, over mocked and recorded raws.

Each case builds the recipe a caller would send and runs it through
``lib.metrics.METRICS`` — the same table the evaluator dispatches on — so what
is pinned here is the production path, not a copy of it. The classes at the
bottom (``TestRecordedAcRaw`` / ``TestRecordedSteppedAcRaw``) run against real
recorded LTspice binary raws from ``tests/fixtures/``; see those classes for
what the mocks cannot cover.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import metrics, services
from ltspice_mcp.lib.metrics import (
    MetricValue,
    has_active_device,
)
from ltspice_mcp.lib.metrics import (
    filter_operating_point as _filter_operating_point,
)
from ltspice_mcp.lib.metrics import (
    guarded_axis as _guarded_axis,
)
from ltspice_mcp.lib.metrics import (
    noise_input_source_unit as _noise_input_source_unit,
)
from ltspice_mcp.lib.metrics import (
    parse_freq as _parse_freq,
)
from ltspice_mcp.lib.metrics import (
    split_ratio as _split_ratio,
)
from ltspice_mcp.lib.metrics import (
    trace_device as _trace_device,
)
from ltspice_mcp.lib.recipes import (
    AcStructureRecipe,
    BodeCrossingRecipe,
    BodeFilterRecipe,
    BodePointRecipe,
    BodeSlopeRecipe,
    EdgesRecipe,
    MeasurementsRecipe,
    NoiseIntegralRecipe,
    OperatingPointRecipe,
    PeriodicRecipe,
    Recipe,
    ResonanceRecipe,
    ReturnLossRecipe,
    SignalStatsRecipe,
    StabilityRecipe,
    SummaryRecipe,
    ThdRecipe,
    TimingEndpoint,
    TimingRecipe,
    ValueRecipe,
    Window,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools._base import safe_path
from tests.conftest import stage_recorded_fixture as _stage_recorded


def _source(state: SessionState, raw_file: str | Path) -> services.AnalysisSource:
    """The source a caller-supplied raw path resolves to, as the tools resolve it."""
    raw = raw_file if isinstance(raw_file, Path) else safe_path(str(raw_file), state)
    return services.AnalysisSource.for_raw(raw)


async def _metric(
    state: SessionState,
    raw_file: str | Path,
    recipe: Recipe,
    step: int = 0,
    **options: Any,
) -> MetricValue:
    """Run one recipe through the table the evaluator dispatches on."""
    return await metrics.METRICS[type(recipe)](
        _source(state, raw_file), recipe, step, state, **options
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
        data = await _metric(
            state_no_sim,
            fake_raw.name,
            SignalStatsRecipe(key="s", metric="signal_stats", signal="V(out)"),
        )
        assert data["signal"] == "V(out)"
        assert data["analysis_type"] == "transient"
        assert data["min"] < data["max"]

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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(vref)"),
        )
        data = result
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(out)"),
        )
        data = result
        assert data is not None
        assert data["analysis_type"] == "dc"
        # min/max computed over the flipped-to-ascending axis.
        assert data["min"] == pytest.approx(0.0)
        assert data["max"] == pytest.approx(2.5)

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await _metric(
                state_no_sim,
                fake_raw.name,
                SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(missing)"),
            )

    async def test_step_out_of_range(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="out of range"):
            await _metric(
                state_no_sim,
                fake_raw.name,
                SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(out)"),
                step=99,
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
        data = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="s", metric="signal_stats", signal="V(out)"),
        )
        assert data["analysis_type"] == "ac"
        assert data["max_db"] == pytest.approx(0.0, abs=0.01)

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
            await _metric(
                state_no_sim,
                raw_file.name,
                SignalStatsRecipe(
                    key="signal_stats",
                    metric="signal_stats",
                    signal="V(out)",
                    window=Window(start="1k"),
                ),
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(out)"),
        )
        sc = result
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(
                key="signal_stats",
                metric="signal_stats",
                signal="V(out)",
                window=Window(start="0.6m", end="1m"),
            ),
        )
        sc = result
        assert sc["mean"] == pytest.approx(5.0)
        assert sc["rms"] == pytest.approx(5.0)
        assert sc["std"] == pytest.approx(0.0, abs=1e-9)
        assert sc["t_start_used"] == pytest.approx(6e-4)
        assert sc["t_end_used"] == pytest.approx(1e-3)


@pytest.mark.asyncio
class TestQueryValue:
    async def test_transient(self, state_no_sim: SessionState, fake_raw: Path):
        data = await _metric(
            state_no_sim,
            fake_raw.name,
            ValueRecipe(key="v", metric="value", expr="V(out)", at="0.5"),
        )
        assert data["signal"] == "V(out)"
        assert isinstance(data["value"], float)

    async def test_invalid_at(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="Invalid 'at'"):
            await _metric(
                state_no_sim,
                fake_raw.name,
                ValueRecipe(key="value", metric="value", expr="V(out)", at="bad"),
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
        data = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="v", metric="value", expr="V(out)", at="1k"),
        )
        assert "magnitude_db" in data and "phase_deg" in data

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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="v(@m1[bogus])", at="0.5"),
        )
        warnings = (result or {}).get("warnings") or []
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="0.5"),
        )
        warnings = (result or {}).get("warnings") or []
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="0.5"),
        )
        warnings = (result or {}).get("warnings") or []
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="0.5"),
        )
        assert not (result or {}).get("warnings")


def test_has_active_device_detects_transistor_currents():
    # has_active_device is one arm of the empty op-point note's gate (the other
    # is an ngspice run); an RC circuit trips neither, so it stays note-free. Sync
    # test, kept out of the asyncio-marked class so pytest-asyncio doesn't flag it.
    assert has_active_device({"Id(M1)": 1e-3, "V(out)": 5.0})
    assert has_active_device({"Ic(Q2)": 1e-3})
    assert not has_active_device({"I(R1)": 1e-3, "I(V1)": 2e-3})
    assert not has_active_device({})


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
        data = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="op", metric="operating_point"),
        )
        assert data["voltages"]["V(out)"] == pytest.approx(1.5)
        assert data["currents"]["I(R1)"] == pytest.approx(0.001)

    async def test_clean_run_emits_empty_warnings(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A clean run must still carry the warnings key (as an empty list) so
        # structured-content consumers see "no warnings", not a missing key.
        raw_file = work_dir / "opclean.raw"
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(out)", "I(R1)"],
            waves={"V(out)": np.array([1.5]), "I(R1)": np.array([0.001])},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        assert (result or {})["warnings"] == []

    async def test_folds_ltspice_logopinfo_op_points(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # LTspice writes per-device op-point params to the .log (under
        # .options logopinfo), not the raw. operating_point folds that block
        # into device_op_points, keyed @dev[param] like ngspice's raw traces.
        raw_file = work_dir / "op.raw"
        (work_dir / "op.log").write_text(
            "Semiconductor Device Operating Points:\n"
            "                        --- MOSFET Transistors ---\n"
            "Name:           M1\n"
            "Model:         nch\n"
            "Id:          9.60e-05\n"
            "Vgs:         9.00e-01\n"
            "Vth:         5.00e-01\n"
            "Vdsat:       4.00e-01\n"
            "Gm:          4.80e-04\n"
            "Gds:         1.00e-06\n"
        )
        raw = _make_raw_mock(
            plotname="Operating Point",
            trace_names=["V(d)", "Id(M1)"],
            waves={"V(d)": np.array([1.8]), "Id(M1)": np.array([9.6e-5])},
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        sc = result or {}
        dop = sc.get("device_op_points") or {}
        assert dop.get("@m1[gm]") == pytest.approx(4.8e-4)
        assert dop.get("@m1[vth]") == pytest.approx(0.5)
        assert "@m1[model]" not in dop  # the string Model: row is dropped
        # device= scoping resolves the log-sourced params for one device.
        scoped = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point", device="M1"),
        )
        assert (scoped or {}).get("device_op_points", {}).get("@m1[gm]")

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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
            at="2.0",
        )
        sc = result
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = (result or {}).get("warnings") or []
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = (result or {}).get("warnings") or []
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
            await _metric(
                state_no_sim,
                raw_file.name,
                OperatingPointRecipe(key="operating_point", metric="operating_point"),
            )

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
            await _metric(
                state_no_sim,
                raw_file.name,
                OperatingPointRecipe(key="operating_point", metric="operating_point"),
            )


@pytest.mark.asyncio
class TestGetSimulationSummary:
    async def test_basic(self, state_no_sim: SessionState, fake_raw: Path):
        data = await _metric(state_no_sim, fake_raw.name, SummaryRecipe(key="s", metric="summary"))
        assert data["sim_type"] == "Transient Analysis"
        assert data["signals"]


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
        source = replace(_source(state_no_sim, fake_raw.name), log=log)
        data = await metrics.summary(
            source, SummaryRecipe(key="summary", metric="summary"), 0, state_no_sim
        )
        assert data["sim_type"] == "Transient Analysis"
        assert "fc" in data["measurements"]


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
        data = await _metric(
            state_no_sim,
            raw_file.name,
            SummaryRecipe(key="s", metric="summary"),
            signal="V(out)",
        )
        assert data["sim_type"] == "AC Analysis"
        assert data["ac_bandwidth_metrics"]["bandwidth_3db"] == pytest.approx(1000.0, rel=0.05)

    async def test_ac_signal_used_when_autopicked(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """With no explicit ``signal`` on an AC raw, the auto-picked trace is
        surfaced as ``ac_signal_used`` (declared in the output_schema, so the
        autouse conformance hook validates the emission)."""
        raw_file = work_dir / "ac_auto.raw"
        freqs = np.logspace(0, 6, 100)
        wave = 1.0 / (1 + 1j * freqs / 1000)
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(out)"],
            waves={"frequency": freqs, "V(out)": wave},
            axis=freqs,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await _metric(
            state_no_sim, raw_file.name, SummaryRecipe(key="summary", metric="summary")
        )
        assert result["ac_signal_used"] == "V(out)"


@pytest.mark.asyncio
class TestSummarySuggestions:
    """When the run's errors name unresolved models, model-resolution help is
    both attached to structuredContent (``suggestions``, declared in the
    output_schema) and rendered into the text lines."""

    async def test_suggestions_in_schema_and_text(
        self, state_no_sim: SessionState, fake_raw: Path, monkeypatch
    ):
        import ltspice_mcp.lib.metrics as metrics_mod

        fake = {"MYMODEL": [{"name": "MyModel", "score": 88, "source_path": "/libs/foo.lib"}]}
        monkeypatch.setattr(
            metrics_mod.services,
            "suggestions_from_errors",
            lambda errors, libraries: fake,
        )
        data = await _metric(state_no_sim, fake_raw.name, SummaryRecipe(key="s", metric="summary"))
        assert data["suggestions"] == fake


@pytest.mark.asyncio
class TestQueryStepRange:
    async def test_step_out_of_range(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="out of range"):
            await _metric(
                state_no_sim,
                fake_raw.name,
                ValueRecipe(key="value", metric="value", expr="V(out)", at="0.5"),
                step=99,
            )

    async def test_signal_not_found(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="not found"):
            await _metric(
                state_no_sim,
                fake_raw.name,
                ValueRecipe(key="value", metric="value", expr="V(missing)", at="0.5"),
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

        data = await _metric(
            state_no_sim, raw_file.name, EdgesRecipe(key="e", metric="edges", signal="V(out)")
        )
        assert data["is_rise_time"] is True
        assert data["signal"] == "V(out)"
        assert data["transition_time"] > 0

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
            await _metric(
                state_no_sim,
                raw_file.name,
                EdgesRecipe(key="edges", metric="edges", signal="V(out)"),
            )

    async def test_invalid_signal(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="not found"):
            await _metric(
                state_no_sim,
                raw_file.name,
                EdgesRecipe(key="edges", metric="edges", signal="V(missing)"),
            )

    async def test_window_propagated(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)

        result = await _metric(
            state_no_sim,
            raw_file.name,
            EdgesRecipe(
                key="edges", metric="edges", signal="V(out)", window=Window(start="100u", end="1m")
            ),
        )
        assert result["is_rise_time"] is True

    async def test_invalid_t_start(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "edge.raw"
        t, y = _step_waveform()
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="Invalid t_start"):
            await _metric(
                state_no_sim,
                raw_file.name,
                EdgesRecipe(
                    key="edges", metric="edges", signal="V(out)", window=Window(start="garbage")
                ),
            )


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
        result = await metrics.pulse_response(
            _source(state_no_sim, raw_file.name),
            "V(out)",
            None,
            None,
            0,
            state_no_sim,
            initial_value=0.0,
            final_value=1.0,
        )
        sc = result
        assert sc is not None
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
            await metrics.pulse_response(
                _source(state_no_sim, raw_file.name), "V(out)", None, None, 0, state_no_sim
            )

    async def test_ringing_tail_renders_unknown_not_never(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # Still-ringing tail on the auto path: settling_time is suppressed. The
        # tool must render it as UNKNOWN, not the definitive "never (within
        # window)" — the two null states have different meanings.
        raw_file = work_dir / "ring.raw"
        t_pre = np.linspace(-0.4e-3, 0, 400, endpoint=False)
        t_post = np.linspace(0, 2e-3, 2000)
        zeta = 0.05
        wn = 2 * np.pi * 1000
        wd = wn * np.sqrt(1 - zeta**2)
        phi = np.arctan2(np.sqrt(1 - zeta**2), zeta)
        y_post = 1 - np.exp(-zeta * wn * t_post) / np.sqrt(1 - zeta**2) * np.sin(wd * t_post + phi)
        t = np.concatenate([t_pre, t_post])
        y = np.concatenate([np.zeros_like(t_pre), y_post])
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        # No explicit final_value -> trailing window is still ringing -> suppressed.
        data = await metrics.pulse_response(
            _source(state_no_sim, raw_file.name), "V(out)", None, None, 0, state_no_sim
        )
        # The null is qualified, not bare: the reason it is unknown (noisy tail)
        # travels with it, so a reader cannot take it for "never settled".
        assert data["settling_time"] is None
        assert "settling_final_value_from_noisy_tail" in data["quality"]

    async def test_short_dwell_renders_unknown_not_never(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A ringing staircase paused flat on its last plateau: the trailing
        # window is quiet, but the signal entered the settle band only just
        # before the window end, so settling_time is suppressed. The tool must
        # render this null as UNKNOWN, not the definitive "never (within
        # window)".
        raw_file = work_dir / "stair.raw"
        t = np.linspace(0, 40e-9, 2001)
        levels = np.array([0.0, 5.0, 2.0, 4.5, 2.5, 4.2, 2.8, 4.109])
        y = levels[np.minimum((t // 5e-9).astype(int), len(levels) - 1)]
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        data = await metrics.pulse_response(
            _source(state_no_sim, raw_file.name), "V(out)", None, None, 0, state_no_sim
        )
        # The null is qualified, not bare: the reason it is unknown (short dwell)
        # travels with it, so a reader cannot take it for "never settled".
        assert data["settling_time"] is None
        assert "settling_dwell_near_window_end" in data["quality"]


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

        result = await _metric(
            state_no_sim,
            raw_file.name,
            TimingRecipe.model_validate(
                {
                    "key": "timing",
                    "metric": "timing",
                    "from": TimingEndpoint(signal="V(in)"),
                    "to": TimingEndpoint(signal="V(out)"),
                }
            ),
        )
        sc = result
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
            await _metric(
                state_no_sim,
                raw_file.name,
                TimingRecipe.model_validate(
                    {
                        "key": "timing",
                        "metric": "timing",
                        "from": TimingEndpoint(signal="V(in)"),
                        "to": TimingEndpoint(signal="V(out)"),
                    }
                ),
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            PeriodicRecipe(key="periodic", metric="periodic", signal="V(clk)"),
        )
        sc = result
        assert sc["frequency"] == pytest.approx(1000.0, rel=0.01)
        assert sc["duty_cycle_pct"] == pytest.approx(40.0, abs=1.0)

    async def test_constant_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "flat.raw"
        t = np.linspace(0, 1e-3, 1000)
        y = np.full_like(t, 1.0)
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        with pytest.raises(ResultError, match="constant"):
            await _metric(
                state_no_sim,
                raw_file.name,
                PeriodicRecipe(key="periodic", metric="periodic", signal="V(out)"),
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
        result = await _metric(
            state_no_sim, log.name, MeasurementsRecipe(key="measurements", metric="measurements")
        )
        assert result is not None
        assert "stats" in result
        # Should have exactly one measurement aggregated
        assert len(result["stats"]) >= 1

    async def test_missing_log_file(self, state_no_sim: SessionState, work_dir: Path):
        with pytest.raises(ResultError):
            await _metric(
                state_no_sim,
                "nonexistent.log",
                MeasurementsRecipe(key="measurements", metric="measurements"),
            )

    async def test_empty_log_errors(self, state_no_sim: SessionState, work_dir: Path):
        log = work_dir / "empty.log"
        log.write_text("not a spice log\n")
        with pytest.raises(ResultError):
            await _metric(
                state_no_sim,
                log.name,
                MeasurementsRecipe(key="measurements", metric="measurements"),
            )


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
        data = await metrics.filter_metrics(
            _source(state_no_sim, raw_file.name), "V(out)", 0, state_no_sim
        )
        assert data["signal"] == "V(out)"
        assert data["filter_type"] == "lowpass"
        assert data["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.05)
        assert data["estimated_order"] == 1

    async def test_rejects_transient(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="AC analysis"):
            await metrics.filter_metrics(
                _source(state_no_sim, fake_raw.name), "V(out)", 0, state_no_sim
            )

    async def test_ref_db_must_be_negative(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="negative"):
            await metrics.filter_metrics(
                _source(state_no_sim, raw_file.name), "V(out)", 0, state_no_sim, ref_db=3.0
            )


@pytest.mark.asyncio
class TestGainAtTool:
    async def test_batch_query(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        result = await metrics.gain_at(
            _source(state_no_sim, raw_file.name), "V(out)", ["100", "1k", "10k"], 0, state_no_sim
        )
        assert result is not None
        sc = result
        assert len(sc["points"]) == 3
        # 1-pole LPF at fc should be -3 dB.
        assert sc["points"][1]["magnitude_db"] == pytest.approx(-3.0, abs=0.1)

    async def test_empty_frequencies(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="empty"):
            await metrics.gain_at(
                _source(state_no_sim, raw_file.name), "V(out)", [], 0, state_no_sim
            )

    async def test_invalid_frequency(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError):
            await metrics.gain_at(
                _source(state_no_sim, raw_file.name), "V(out)", ["not_a_number"], 0, state_no_sim
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
        result = await metrics.gain_at(
            _source(state_no_sim, raw_file.name), "V(out)/V(mid)", ["1", "1k"], 0, state_no_sim
        )
        sc = result
        assert sc is not None
        # The 2x cancels: deep passband is 0 dB (not +6 dB), and fc is -3 dB.
        assert sc["points"][0]["magnitude_db"] == pytest.approx(0.0, abs=0.2)
        assert sc["points"][1]["magnitude_db"] == pytest.approx(-3.0, abs=0.2)

    async def test_ratio_missing_operand_errors(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_ratio_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="not found"):
            await metrics.gain_at(
                _source(state_no_sim, raw_file.name), "V(out)/V(nope)", ["1k"], 0, state_no_sim
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
            await metrics.gain_at(
                _source(state_no_sim, raw_file.name), "V(out)/V(mid)", ["1k"], 0, state_no_sim
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            StabilityRecipe(key="stability", metric="stability", signal="V(loop)"),
        )
        sc = result
        assert sc["stability"] in ("unconditional", "stable")
        assert sc["phase_margin_worst_deg"] is not None
        # 60 dB DC gain.
        assert sc["dc_gain_db"] == pytest.approx(60.0, abs=0.1)


@pytest.mark.asyncio
class TestRollOffTool:
    async def test_1pole_asymptote(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir, fc=100.0)
        result = await metrics.roll_off(
            _source(state_no_sim, raw_file.name), "V(out)", "10k", "100k", 0, state_no_sim
        )
        assert result is not None
        sc = result
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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ResonanceRecipe(key="resonance", metric="resonance", signal="V(out)"),
        )
        sc = result
        assert len(sc["peaks"]) == 1
        peak = sc["peaks"][0]
        assert peak["frequency_hz"] == pytest.approx(1000.0, rel=0.05)
        assert peak["q_factor"] == pytest.approx(10.0, rel=0.1)


@pytest.mark.asyncio
class TestFindCrossingTool:
    async def test_magnitude_crossing(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        result = await metrics.find_crossing(
            _source(state_no_sim, raw_file.name), "V(out)", "magnitude_db", -3.0, 0, state_no_sim
        )
        assert result is not None
        sc = result
        assert len(sc["crossings"]) == 1
        assert sc["crossings"][0]["frequency_hz"] == pytest.approx(1000.0, rel=0.05)

    async def test_rejects_transient(self, state_no_sim: SessionState, fake_raw: Path):
        with pytest.raises(ResultError, match="AC analysis"):
            await metrics.find_crossing(
                _source(state_no_sim, fake_raw.name),
                "V(out)",
                "magnitude_db",
                0.0,
                0,
                state_no_sim,
            )

    async def test_max_results_validated(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = _ac_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="max_results"):
            await metrics.find_crossing(
                _source(state_no_sim, raw_file.name),
                "V(out)",
                "magnitude_db",
                0.0,
                0,
                state_no_sim,
                max_results=0,
            )


class TestParseFreqUnitTolerance:
    """Frequency parsing accepts a trailing Hz/kHz unit."""

    def test_bare_number(self):

        assert _parse_freq("1000") == pytest.approx(1000.0)

    def test_hz_suffix(self):

        assert _parse_freq("159Hz") == pytest.approx(159.0)

    def test_khz_suffix(self):

        assert _parse_freq("15.9kHz") == pytest.approx(15900.0)

    def test_si_prefix_still_works(self):

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

        res = await _metric(
            state_no_sim,
            "ac.raw",
            ValueRecipe(key="value", metric="value", expr="V(out)", at="100"),
        )
        assert res is not None
        sc = res
        assert sc["magnitude_linear"] == pytest.approx(abs(0.7 + 0.7j))
        assert "magnitude_db" in sc


@pytest.mark.asyncio
class TestBodeMetrics:
    async def test_point_recipe(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        data = await _metric(
            state_no_sim,
            "bode.raw",
            BodePointRecipe(key="p", metric="bode_point", signal="V(out)", at_hz="1k"),
        )
        assert "points" in data

    async def test_crossing_recipe(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode2.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        data = await _metric(
            state_no_sim,
            "bode2.raw",
            BodeCrossingRecipe(key="c", metric="bode_crossing", signal="V(out)", level_db=-3.0103),
        )
        cs = data["crossings"]
        assert cs and abs(cs[0]["frequency_hz"] - 1591.5) / 1591.5 < 0.05

    async def test_slope_recipe(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode3.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        data = await _metric(
            state_no_sim,
            "bode3.raw",
            BodeSlopeRecipe(
                key="s", metric="bode_slope", signal="V(out)", from_hz="10k", to_hz="100k"
            ),
        )
        # First-order LPF stopband ≈ -20 dB/decade.
        assert data["slope_db_per_decade"] < -15

    async def test_filter_recipe(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "bode4.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        data = await _metric(
            state_no_sim,
            "bode4.raw",
            BodeFilterRecipe(key="f", metric="bode_filter", signal="V(out)"),
        )
        assert "filter_type" in data


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
        sc = await _metric(
            state_no_sim,
            str(raw),
            BodeFilterRecipe(key="f", metric="bode_filter", signal="V(out)"),
        )
        assert sc["filter_type"] == "lowpass"
        assert sc["cutoff_high_hz"] == pytest.approx(1000.0, rel=0.02)
        assert sc["estimated_order"] == 1

    async def test_leading_minus_flips_phase_180(self, state_no_sim: SessionState, work_dir: Path):
        # '-V(out)' and '-V(out)/V(out)' negate the complex wave: same |H|,
        # phase shifted by 180° — the loop-gain / inverting-probe convention
        # without a behavioral inverter node in the deck.
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")

        async def point(signal: str) -> dict:
            data = await _metric(
                state_no_sim,
                str(raw),
                BodePointRecipe(key="p", metric="bode_point", signal=signal, at_hz="1k"),
            )
            return data["points"][0]

        plain = await point("V(out)")
        negated = await point("-V(out)")
        assert negated["magnitude_db"] == pytest.approx(plain["magnitude_db"], abs=1e-9)
        delta = (negated["phase_deg"] - plain["phase_deg"]) % 360.0
        assert delta == pytest.approx(180.0, abs=1e-6)
        # Ratio form: -A/B is −(A/B) → exactly 0 dB at 180°.
        inv_unity = await point("-V(out)/V(out)")
        assert inv_unity["magnitude_db"] == pytest.approx(0.0, abs=1e-9)
        assert abs(inv_unity["phase_deg"]) == pytest.approx(180.0, abs=1e-6)

    async def test_crossing_mode_minus_3db_at_cutoff(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        sc = await _metric(
            state_no_sim,
            str(raw),
            BodeCrossingRecipe(key="c", metric="bode_crossing", signal="V(out)", level_db=-3.0103),
        )
        assert len(sc["crossings"]) == 1
        assert sc["crossings"][0]["frequency_hz"] == pytest.approx(1000.0, rel=0.02)

    async def test_signal_stats_ac_magnitude_range(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        res = await _metric(
            state_no_sim,
            str(raw),
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(out)"),
        )
        sc = res
        assert sc is not None
        assert sc["analysis_type"] == "ac"
        assert sc["point_count"] == 81  # dec 20 over 4 decades
        # Passband (10 Hz, two decades below the pole): |H| ~ 1, ~0 dB.
        assert sc["max_db"] == pytest.approx(0.0, abs=0.01)
        # Stopband end (100 kHz = 100*fc): |H| ~ 1/100, ~ -40 dB.
        assert sc["min_db"] == pytest.approx(-40.0, abs=0.1)

    async def test_query_value_passband_and_pole(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_ac_rc")
        passband = await _metric(
            state_no_sim,
            str(raw),
            ValueRecipe(key="value", metric="value", expr="V(out)", at="10"),
        )
        sc = passband
        assert sc is not None
        assert sc["magnitude_linear"] == pytest.approx(1.0, abs=1e-3)
        assert sc["magnitude_db"] == pytest.approx(0.0, abs=0.01)

        pole = await _metric(
            state_no_sim,
            str(raw),
            ValueRecipe(key="value", metric="value", expr="V(out)", at="1k"),
        )
        sc = pole
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

    async def test_single_step_filter_uses_requested_step(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_step_ac")
        for step, fc in enumerate(self.CUTOFFS):
            sc = await _metric(
                state_no_sim,
                str(raw),
                BodeFilterRecipe(key="f", metric="bode_filter", signal="V(out)"),
                step=step,
            )
            assert sc["cutoff_high_hz"] == pytest.approx(fc, rel=0.01)


@pytest.mark.asyncio
class TestArgumentErrorsCarryNoDispatchHint:
    """An argument-shape ResultError must NOT trigger the generic 'check the job
    for details' dispatch hint — it is a caller mistake, not a run failure, so
    it carries ``show_hint=False`` and its own complete redirect instead."""

    async def test_unparseable_at_value(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "badat.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError) as excinfo:
            await _metric(
                state_no_sim,
                "badat.raw",
                ValueRecipe(key="v", metric="value", expr="V(out)", at="not-a-number"),
            )
        assert excinfo.value.show_hint is False

    async def test_time_window_on_an_ac_sweep(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "acwin.raw"
        _inject_raw(state_no_sim, path, _ac_raw_mock())
        with pytest.raises(ResultError) as excinfo:
            await _metric(
                state_no_sim,
                "acwin.raw",
                SignalStatsRecipe(
                    key="s",
                    metric="signal_stats",
                    signal="V(out)",
                    window=Window(start="1m"),
                ),
            )
        assert excinfo.value.show_hint is False

    async def test_noise_integral_on_a_transient_raw(
        self, state_no_sim: SessionState, fake_raw: Path
    ):
        with pytest.raises(ResultError) as excinfo:
            await _metric(
                state_no_sim,
                fake_raw.name,
                NoiseIntegralRecipe(key="n", metric="noise_integral", signal="V(out)"),
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
        import ltspice_mcp.lib.metrics as metrics_mod

        def _boom(*_args, **_kwargs):
            raise ValueError("synthetic build failure")

        monkeypatch.setattr(metrics_mod, "build_simulation_summary", _boom)
        with pytest.raises(ResultError) as excinfo:
            await _metric(
                state_no_sim, fake_raw.name, SummaryRecipe(key="summary", metric="summary")
            )
        assert excinfo.value.show_hint is False
        # Must not re-suggest the tool that just failed.
        assert "simulation_summary" not in str(excinfo.value)


@pytest.mark.asyncio
class TestDcRejectedByTransientTools:
    """Regression: a .DC sweep raw produces a voltage (not time) axis, so the
    transient-only tools (edge/pulse/periodic/timing) must refuse it instead of
    reading meaningless rise-times off it. A real recorded raw is used so the
    actual Plotname 'DC transfer characteristic' flows through the reject path
    (a mock could mask the classification)."""

    @pytest.mark.parametrize(
        "run",
        [
            lambda state, name: _metric(
                state, name, EdgesRecipe(key="e", metric="edges", signal="V(out)")
            ),
            lambda state, name: metrics.pulse_response(
                _source(state, name), "V(out)", None, None, 0, state
            ),
            lambda state, name: _metric(
                state, name, PeriodicRecipe(key="p", metric="periodic", signal="V(out)")
            ),
            lambda state, name: _metric(
                state,
                name,
                TimingRecipe.model_validate(
                    {
                        "key": "t",
                        "metric": "timing",
                        "from": TimingEndpoint(signal="V(out)"),
                        "to": TimingEndpoint(signal="V(out)"),
                    }
                ),
            ),
        ],
        ids=["edges", "transient_response", "periodic", "timing"],
    )
    async def test_transient_metrics_reject_dc_raw(
        self, state_no_sim: SessionState, work_dir: Path, run
    ):
        # timing rejects at a DISTINCT call site from the edges/step/periodic
        # loader, so it gets its own parametrize case rather than relying on the
        # shared one being exercised.
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        with pytest.raises(ResultError, match="transient"):
            await run(state_no_sim, raw.name)


@pytest.mark.asyncio
class TestOperatingPointInternalsHint:
    """When device_op_points is empty, operating_point appends ONE recovery note
    naming both paths (LTspice .options logopinfo, ngspice .save @dev[param]) —
    it never branches on the producing simulator, which the session default can
    get wrong on a cross-simulator raw read. It is gated so passive circuits
    stay note-free: it fires when an M/Q/J/D terminal current proves a device is
    present, OR when the run is ngspice (whose bare .op exposes no device traces,
    so a saved-nothing run is indistinguishable from a passive one)."""

    async def test_active_device_terminal_current_fires_hint(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # LTspice-style .op: a device terminal current (Id(M1)) proves a device is
        # present, but no @dev[param] table was exported → the note fires.
        assert state_no_sim.raw_dialect is None  # no simulator → LTspice semantics
        p = work_dir / "lt_op.raw"
        _inject_raw_mock(
            state_no_sim,
            p,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(d)", "Id(M1)"],
                waves={"V(d)": np.array([0.9]), "Id(M1)": np.array([1e-4])},
            ),
        )
        res = await _metric(
            state_no_sim,
            p.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = res.get("warnings", [])
        # The note names both recovery paths: LTspice's .options logopinfo and
        # ngspice's .save.
        assert any("logopinfo" in w and ".save all @m1[gm]" in w for w in warnings), warnings

    async def test_bare_ngspice_op_fires_hint_via_dialect(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A bare ngspice .op shows no device traces at all (no terminal currents),
        # so active-device detection can't fire — the ngspice dialect gate must.
        state_no_sim.default_simulator = type("NGspiceSimulator", (), {})
        assert state_no_sim.raw_dialect == "ngspice"
        p = work_dir / "ng_op.raw"
        _inject_raw_mock(
            state_no_sim,
            p,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(d)"],
                waves={"V(d)": np.array([0.9])},
            ),
        )
        res = await _metric(
            state_no_sim,
            p.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = res.get("warnings", [])
        assert any(".save all @m1[gm]" in w for w in warnings), warnings

    async def test_passive_op_emits_no_hint(self, state_no_sim: SessionState, work_dir: Path):
        # No active-device terminal current and not ngspice → passive, stays
        # note-free (a bare .op on an RC bias point must not nag about op points).
        assert state_no_sim.raw_dialect is None
        p = work_dir / "rc_op.raw"
        _inject_raw_mock(
            state_no_sim,
            p,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(out)", "I(R1)"],
                waves={"V(out)": np.array([0.5]), "I(R1)": np.array([1e-4])},
            ),
        )
        res = await _metric(
            state_no_sim,
            p.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = res.get("warnings", [])
        assert not any("logopinfo" in w.lower() for w in warnings), warnings

    async def test_saved_internals_emit_no_recovery_hint(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # When op-point params ARE present, the note doesn't fire.
        state_no_sim.default_simulator = type("NGspiceSimulator", (), {})
        p = work_dir / "ng_op_saved.raw"
        _inject_raw_mock(
            state_no_sim,
            p,
            _make_raw_mock(
                plotname="Operating Point",
                trace_names=["V(d)", "@m1[gm]"],
                waves={"V(d)": np.array([0.9]), "@m1[gm]": np.array([2e-3])},
            ),
        )
        res = await _metric(
            state_no_sim,
            p.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        warnings = res.get("warnings", [])
        assert not any("logopinfo" in w.lower() for w in warnings), warnings
        assert res["device_op_points"].get("@m1[gm]") == pytest.approx(2e-3)


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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(onoise)"),
        )
        sc = result
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
            await _metric(
                state_no_sim,
                raw.name,
                SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(hot)"),
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
            "device_op_points": {"@m1[gm]": 1e-3, "@m2[gm]": 2e-3},
        }
        assert _filter_operating_point(op, "M1") is True
        assert op["currents"] == {"Id(M1)": 1e-3}
        assert op["device_op_points"] == {"@m1[gm]": 1e-3}
        # Node voltages are not device-scoped -> dropped from the focused view.
        assert op["voltages"] == {}

    def test_filter_matches_subcircuit_path_suffix(self):
        op = {"voltages": {}, "currents": {}, "device_op_points": {"@m.x1.mn[gm]": 5.0}}
        assert _filter_operating_point(op, "mn") is True
        assert op["device_op_points"] == {"@m.x1.mn[gm]": 5.0}

    def test_filter_no_match_reports_false(self):
        op = {"voltages": {}, "currents": {"Id(M1)": 1.0}, "device_op_points": {}}
        assert _filter_operating_point(op, "Q9") is False


@pytest.mark.asyncio
class TestOperatingPointDeviceAndUnits:
    """device= narrows to one device's op-point params + terminal currents (2c); every
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
        res = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point"),
        )
        sc = res
        assert sc is not None
        assert sc["units"]["V(d)"] == "V"
        assert sc["units"]["Id(M1)"] == "A"
        # A device-internal parameter gets no guessed unit.
        assert "@m1[gm]" not in sc["units"]

    async def test_device_filter_focuses_one_device(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw_file = self._op_raw(state_no_sim, work_dir)
        res = await _metric(
            state_no_sim,
            raw_file.name,
            OperatingPointRecipe(key="operating_point", metric="operating_point", device="M1"),
        )
        sc = res
        assert sc is not None
        assert sc["device"] == "M1"
        assert set(sc["currents"]) == {"Id(M1)", "Ig(M1)"}
        assert set(sc["device_op_points"]) == {"@m1[gm]"}
        assert sc["voltages"] == {}
        assert sc["units"]["Id(M1)"] == "A"

    async def test_unknown_device_lists_present_ones(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw_file = self._op_raw(state_no_sim, work_dir)
        with pytest.raises(ResultError, match="Devices present"):
            await _metric(
                state_no_sim,
                raw_file.name,
                OperatingPointRecipe(key="operating_point", metric="operating_point", device="Q9"),
            )


@pytest.mark.asyncio
class TestQueryValueDcLabelAndUnit:
    async def test_dc_sweep_labels_swept_axis_and_carries_unit(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage_recorded(work_dir, "ltspice_dc_div")
        data = await _metric(
            state_no_sim, str(raw), ValueRecipe(key="v", metric="value", expr="V(out)", at="2")
        )
        assert data["unit"] == "V"
        # The DC sweep axis is the swept variable, not time, and the label a
        # reader puts on the requested point says so.
        loaded = await services.load_raw(raw, state_no_sim)
        assert metrics.query_x_label(loaded, "DC transfer characteristic") not in ("t", "f")

    async def test_noise_density_labels_per_root_hz(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A .noise density trace is V/√Hz, not the plain V its whattype declares.
        raw = _stage_recorded(work_dir, "ltspice_noise_rc")
        res = await _metric(
            state_no_sim,
            str(raw),
            ValueRecipe(key="value", metric="value", expr="V(onoise)", at="1k"),
        )
        assert res["unit"] == "V/√Hz"


@pytest.mark.asyncio
class TestNoiseIntegralHandler:
    async def test_real_noise_fixture(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_noise_rc")
        res = await _metric(
            state_no_sim,
            str(raw),
            NoiseIntegralRecipe(key="noise_integral", metric="noise_integral", signal="V(onoise)"),
        )
        sc = res
        assert sc is not None
        assert sc["total_rms"] > 0
        assert sc["n_points"] > 1
        assert sc["unit"] == "V"
        assert "Hz" in sc["density_unit"]

    async def test_rejects_transient_raw(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage_recorded(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="noise"):
            await _metric(
                state_no_sim,
                str(raw),
                NoiseIntegralRecipe(key="noise_integral", metric="noise_integral"),
            )

    def _inoise_raw_mock(self) -> MagicMock:
        freq = np.logspace(1, 5, 20)
        return _make_raw_mock(
            plotname="Noise Spectral Density",
            trace_names=["frequency", "V(onoise)", "V(inoise)"],
            axis=freq,
            waves={
                "frequency": freq,
                "V(onoise)": np.full_like(freq, 1e-8),
                "V(inoise)": np.full_like(freq, 1e-9),
            },
        )

    async def test_inoise_unit_unverified_without_job(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # No job_id -> no deck to check -> falls back to the (possibly wrong)
        # trace-derived unit, but must say so rather than claim certainty.
        raw_file = work_dir / "noise_bare.raw"
        _inject_raw_mock(state_no_sim, raw_file, self._inoise_raw_mock())

        res = await _metric(
            state_no_sim,
            str(raw_file),
            NoiseIntegralRecipe(key="noise_integral", metric="noise_integral", signal="V(inoise)"),
        )
        sc = res
        assert sc is not None
        assert sc["unit"] == "V"
        assert any("Could not verify" in w for w in sc["warnings"])


class TestNoiseInputSourceUnit:
    """Pure-function coverage for the .NOISE input-source unit resolver."""

    def _deck(self, tmp_path: Path, body: str) -> Path:
        p = tmp_path / "noise.cir"
        p.write_text(body)
        return p

    def test_voltage_source(self, tmp_path: Path):
        deck = self._deck(tmp_path, "V1 in 0 AC 1\n.NOISE V(out) V1 dec 10 1 100k\n.end\n")
        assert _noise_input_source_unit(deck) == "V"

    def test_current_source(self, tmp_path: Path):
        deck = self._deck(tmp_path, "I1 in 0 AC 1\n.NOISE V(out) I1 dec 10 1 100k\n.end\n")
        assert _noise_input_source_unit(deck) == "A"

    def test_indented_directive_resolves(self, tmp_path: Path):
        # Leading whitespace before .NOISE must not defeat the match.
        deck = self._deck(tmp_path, "I1 in 0 AC 1\n    .NOISE V(out) I1 dec 10 1 100k\n.end\n")
        assert _noise_input_source_unit(deck) == "A"

    def test_conflicting_directives_are_ambiguous(self, tmp_path: Path):
        # Two .NOISE lines disagreeing on source type -> can't tell which
        # produced this raw, so fall back rather than guess.
        deck = self._deck(
            tmp_path,
            "V1 in 0 AC 1\nI1 in 0 AC 1\n"
            ".NOISE V(out) V1 dec 10 1 100k\n.NOISE V(out) I1 dec 10 1 100k\n.end\n",
        )
        assert _noise_input_source_unit(deck) is None

    def test_agreeing_directives_resolve(self, tmp_path: Path):
        deck = self._deck(
            tmp_path,
            "V1 in 0 AC 1\n.NOISE V(a) V1 dec 10 1 100k\n.NOISE V(b) V1 dec 10 1 1k\n.end\n",
        )
        assert _noise_input_source_unit(deck) == "V"

    def test_unrecognized_prefix_is_none(self, tmp_path: Path):
        deck = self._deck(tmp_path, "R1 in 0 1k\n.NOISE V(out) R1 dec 10 1 100k\n.end\n")
        assert _noise_input_source_unit(deck) is None

    def test_no_directive_is_none(self, tmp_path: Path):
        deck = self._deck(tmp_path, "V1 in 0 AC 1\nR1 in out 1k\n.end\n")
        assert _noise_input_source_unit(deck) is None

    def test_missing_netlist_is_none(self):
        assert _noise_input_source_unit(None) is None


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
        res = await _metric(
            state_no_sim,
            raw_file.name,
            ThdRecipe(key="thd", metric="thd", signal="V(out)", fundamental_hz="1k", harmonics=3),
        )
        sc = res
        assert sc is not None
        assert sc["thd_ratio"] == pytest.approx(0.1, rel=1e-2)
        assert sc["coherent"] is True

    async def test_thd_labels_harmonic_unit(self, state_no_sim: SessionState, work_dir: Path):
        # The per-harmonic magnitudes are in the signal's native unit; label it.
        raw_file = work_dir / "thd_unit.raw"
        f0, fs = 1000.0, 200_000.0
        t = np.arange(0.0, 0.02, 1.0 / fs)
        y = np.sin(2 * np.pi * f0 * t) + 0.05 * np.sin(2 * np.pi * 2 * f0 * t)
        raw = _make_raw_mock(
            plotname="Transient Analysis",
            trace_names=["time", "V(out)"],
            waves={"time": t, "V(out)": y},
            axis=t,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        res = await _metric(
            state_no_sim,
            raw_file.name,
            ThdRecipe(key="thd", metric="thd", signal="V(out)", fundamental_hz="1k"),
        )
        assert res["unit"] == "V"


def _ac_response_raw(signal: str, h: np.ndarray, freqs: np.ndarray) -> MagicMock:
    """An AC raw mock carrying one complex response under ``signal``."""
    return _make_raw_mock(
        plotname="AC Analysis",
        trace_names=["frequency", signal],
        waves={"frequency": freqs, signal: h},
        axis=freqs,
    )


async def _assert_relays_solve_failure(
    state: SessionState, work_dir: Path, name: str, raw: MagicMock, run
) -> None:
    """Drive ``handler`` against a raw whose sibling .log reports a singular
    matrix, and assert the failure surfaces in the rendered result. Covers the
    structural guarantee that every raw-reading tool (metric and egress) relays
    a run-level solve failure — a new tool that forgets the relay fails here."""
    raw_file = work_dir / f"{name}.raw"
    _inject_raw_mock(state, raw_file, raw)
    (work_dir / f"{name}.log").write_text("gmin stepping failed\n")
    data = await run(state, raw_file.name)
    warnings = data.get("warnings") or []
    assert any("gmin stepping" in w.lower() for w in warnings), (
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
                lambda state, n: _metric(
                    state, n, SignalStatsRecipe(key="s", metric="signal_stats", signal="V(out)")
                ),
            ),
            (
                "edge",
                _make_raw_mock(waves={"time": t_step, "V(out)": y_step}, axis=t_step),
                lambda state, n: _metric(
                    state, n, EdgesRecipe(key="e", metric="edges", signal="V(out)")
                ),
            ),
            (
                "pulse",
                _make_raw_mock(waves={"time": t_step, "V(out)": y_step}, axis=t_step),
                lambda state, n: metrics.pulse_response(
                    _source(state, n),
                    "V(out)",
                    None,
                    None,
                    0,
                    state,
                    initial_value=0.0,
                    final_value=1.0,
                ),
            ),
            (
                "timing",
                _make_raw_mock(
                    trace_names=["time", "V(in)", "V(out)"],
                    waves={"time": t_step, "V(in)": y_step, "V(out)": y_step},
                    axis=t_step,
                ),
                lambda state, n: _metric(
                    state,
                    n,
                    TimingRecipe.model_validate(
                        {
                            "key": "t",
                            "metric": "timing",
                            "from": TimingEndpoint(signal="V(in)"),
                            "to": TimingEndpoint(signal="V(out)"),
                        }
                    ),
                ),
            ),
            (
                "periodic",
                _make_raw_mock(
                    trace_names=["time", "V(clk)"], waves={"time": t_sq, "V(clk)": y_sq}, axis=t_sq
                ),
                lambda state, n: _metric(
                    state, n, PeriodicRecipe(key="p", metric="periodic", signal="V(clk)")
                ),
            ),
            (
                "thd",
                _make_raw_mock(waves={"time": t_sin, "V(out)": y_sin}, axis=t_sin),
                lambda state, n: _metric(
                    state,
                    n,
                    ThdRecipe(
                        key="d",
                        metric="thd",
                        signal="V(out)",
                        fundamental_hz="1k",
                        harmonics=3,
                    ),
                ),
            ),
        ]
        for name, raw, run in cases:
            await _assert_relays_solve_failure(state_no_sim, work_dir, name, raw, run)

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
                lambda state, n: _metric(
                    state, n, StabilityRecipe(key="st", metric="stability", signal="V(loop)")
                ),
            ),
            (
                "reson",
                _ac_response_raw("V(out)", peak, freqs),
                lambda state, n: _metric(
                    state, n, ResonanceRecipe(key="r", metric="resonance", signal="V(out)")
                ),
            ),
            (
                "bode1",
                _ac_response_raw("V(out)", lpf, freqs),
                lambda state, n: _metric(
                    state,
                    n,
                    BodePointRecipe(key="p", metric="bode_point", signal="V(out)", at_hz="1k"),
                ),
            ),
        ]
        for name, raw, run in cases:
            await _assert_relays_solve_failure(state_no_sim, work_dir, name, raw, run)

    async def test_ac_structure_relays_onto_observations(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # ac_structure is the one AC metric with an observations channel, so its
        # relay lands there rather than in warnings — the two channels answer
        # different questions and must not be merged.
        freqs = np.logspace(0, 6, 200)
        lpf = 1.0 / (1 + (2j * np.pi * freqs) / (2 * np.pi * 1000))
        raw_file = work_dir / "acstruct.raw"
        _inject_raw_mock(state_no_sim, raw_file, _ac_response_raw("V(out)", lpf, freqs))
        (work_dir / "acstruct.log").write_text("gmin stepping failed\n")
        data = await _metric(
            state_no_sim,
            raw_file.name,
            AcStructureRecipe(key="a", metric="ac_structure", signal="V(out)"),
        )
        assert any("gmin stepping" in o["detail"].lower() for o in data["observations"])

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
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="0.5"),
        )
        warnings = (result or {}).get("warnings") or []
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
            await _metric(
                state_no_sim,
                raw_file.name,
                EdgesRecipe(key="edges", metric="edges", signal="V(out)"),
            )


@pytest.mark.asyncio
class TestDisturbanceResponseTool:
    async def test_load_transient_droop(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "ldo.raw"
        t = np.linspace(0, 5e-3, 5001)
        tri = np.clip(1 - np.abs(t - 1.5e-3) / 0.5e-3, 0, 1)
        y = 3.3 - 0.1 * tri  # 100 mV droop, recovers by ~2 ms
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await metrics.disturbance_response(
            _source(state_no_sim, raw_file.name),
            "V(out)",
            None,
            None,
            0,
            state_no_sim,
            settle_band_pct=1.0,
        )
        sc = result
        assert sc is not None
        assert sc["signal"] == "V(out)"
        assert sc["baseline"] == pytest.approx(3.3, abs=1e-6)
        assert sc["max_droop"] == pytest.approx(0.1, abs=2e-4)
        assert sc["recovery_time"] == pytest.approx(1.835e-3, abs=5e-5)


@pytest.mark.asyncio
class TestReturnLossTool:
    async def test_mismatch_at_freq(self, state_no_sim: SessionState, work_dir: Path):
        raw_file = work_dir / "zin.raw"
        f = np.logspace(6, 9, 200)
        H = np.full_like(f, 100.0, dtype=complex)  # 100 Ω flat → Γ=1/3
        raw = _make_raw_mock(
            plotname="AC Analysis",
            trace_names=["frequency", "V(in)"],
            waves={"frequency": f, "V(in)": H},
            axis=f,
        )
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await _metric(
            state_no_sim,
            raw_file.name,
            ReturnLossRecipe(key="return_loss", metric="return_loss", signal="V(in)", z0=50.0),
            at="1e7",
        )
        sc = result
        assert sc["signal"] == "V(in)"
        assert sc["z0_ohm"] == 50.0
        assert sc["return_loss_db"] == pytest.approx(9.542, abs=1e-2)
        assert sc["vswr"] == pytest.approx(2.0, abs=1e-3)


@pytest.mark.asyncio
class TestSignalStatsConstantObservation:
    async def test_constant_window_emits_observation(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A latched/degenerate DC solution reads as a flat line — surface it as
        # a fact (min == max), not a verdict.
        raw_file = work_dir / "latched.raw"
        t = np.linspace(0, 1e-3, 500)
        y = np.zeros_like(t)  # min == max == 0
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        result = await _metric(
            state_no_sim,
            raw_file.name,
            SignalStatsRecipe(key="signal_stats", metric="signal_stats", signal="V(out)"),
        )
        codes = [o["code"] for o in result.get("observations", [])]
        assert "constant_window" in codes


@pytest.mark.asyncio
class TestQueryValueExactMatch:
    async def test_snap_flags_inexact(self, state_no_sim: SessionState, work_dir: Path):
        # Coarse axis so a between-samples request must snap.
        raw_file = work_dir / "coarse.raw"
        t = np.arange(0.0, 5.0, 1.0)  # 0,1,2,3,4
        y = t * 2.0
        raw = _make_raw_mock(waves={"time": t, "V(out)": y}, axis=t)
        _inject_raw_mock(state_no_sim, raw_file, raw)
        snapped = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="1.5"),
        )
        assert snapped["exact_match"] is False
        exact = await _metric(
            state_no_sim,
            raw_file.name,
            ValueRecipe(key="value", metric="value", expr="V(out)", at="2"),
        )
        assert exact["exact_match"] is True


class TestGuardedAxisSteppedOpHint:
    """A no-axis raw that is really a stepped .op (collapsed to step 0) should
    point the caller at the .dc conversion where they hit the wall, not just
    say 'no axis'."""

    @staticmethod
    def _no_axis_raw():
        raw = MagicMock()
        raw.get_axis.side_effect = Exception("no axis in this plot")
        return raw

    def test_plain_op_points_at_operating_point(self, work_dir: Path):

        raw_path = work_dir / "op.raw"  # no sibling .log
        with pytest.raises(ResultError, match="operating_point"):
            _guarded_axis(self._no_axis_raw(), 0, raw_path)

    def test_stepped_op_points_at_dc_conversion(self, work_dir: Path):

        raw_path = work_dir / "stepped_op.raw"
        raw_path.with_suffix(".log").write_text(".step temp=-40\n.step temp=25\n.step temp=85\n")
        with pytest.raises(ResultError, match=r"\.dc temp"):
            _guarded_axis(self._no_axis_raw(), 0, raw_path)
