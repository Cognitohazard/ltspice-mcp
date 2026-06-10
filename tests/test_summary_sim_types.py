"""simulation_summary against recorded real LTspice results, one per sim type.

The fixtures under ``tests/fixtures/`` were produced by running tiny decks
through the real ``run_simulation`` handler with LTspice on Windows/WSL and
copying the resulting ``.raw``/``.log`` pairs verbatim:

- ``ltspice_tran_rc``  — RC step response, ``.tran 0 1m 0 5u`` with a
  ``.meas tran vfinal FIND V(out) AT=0.9m`` so the log carries a real
  measurement line (V1 steps 0 -> 1 V, tau = 100 us).
- ``ltspice_ac_rc``    — RC low-pass, ``.ac dec 20 10 100k`` with
  R = 1k / C = 159.15n, i.e. f_3dB = 1 kHz by construction.
- ``ltspice_dc_div``   — equal-resistor divider, ``.dc V1 0 5 0.5`` so
  V(out) = V1 / 2 at every sweep point.
- ``op_extreme_node``  — pre-existing recorded ``.op`` raw whose first trace
  V(hot) sits at 1e9 V (also exercised by test_result_observations).

CI cannot run LTspice, so these recordings are the only coverage of the
LTspice binary-raw dialect flowing through ``handle_simulation_summary``
(everything else in the suite drives it with mocked RawRead instances).
"""

import shutil
from pathlib import Path

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    QueryValueInput,
    SimulationSummaryInput,
    handle_query_value,
    handle_simulation_summary,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _stage(work_dir: Path, name: str) -> Path:
    """Copy a recorded fixture's .raw (and .log when recorded) into work_dir.

    Returns the staged .raw path. The .log lands next to it so the handler's
    automatic ``raw_file`` -> ``.log`` derivation is exercised for real.
    """
    raw = work_dir / f"{name}.raw"
    shutil.copy(FIXTURES / f"{name}.raw", raw)
    log = FIXTURES / f"{name}.log"
    if log.exists():
        shutil.copy(log, work_dir / f"{name}.log")
    return raw


async def _summary(state: SessionState, raw: Path, signal: str | None = None) -> dict:
    result = await handle_simulation_summary(
        SimulationSummaryInput(raw_file=str(raw), signal=signal, format="json"),
        state,
    )
    assert result.structuredContent is not None
    return result.structuredContent


async def _value_at(state: SessionState, raw: Path, signal: str, at: str) -> dict:
    result = await handle_query_value(
        QueryValueInput(raw_file=str(raw), signal=signal, at=at, format="json"),
        state,
    )
    assert result.structuredContent is not None
    return result.structuredContent


@pytest.mark.asyncio
class TestSummarySimTypeMatrix:
    """Type label + signal list are correct for every LTspice plot kind."""

    @pytest.mark.parametrize(
        ("name", "sim_type", "signals"),
        [
            ("ltspice_tran_rc", "Transient Analysis", {"time", "V(in)", "V(out)", "I(R1)"}),
            ("ltspice_ac_rc", "AC Analysis", {"frequency", "V(in)", "V(out)", "I(C1)"}),
            ("ltspice_dc_div", "DC transfer characteristic", {"V1", "V(out)", "I(R2)"}),
            ("op_extreme_node", "Operating Point", {"V(hot)", "I(I1)", "I(R1)"}),
        ],
    )
    async def test_sim_type_and_signals(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        name: str,
        sim_type: str,
        signals: set[str],
    ):
        raw = _stage(work_dir, name)
        summary = await _summary(state_no_sim, raw)
        assert summary["sim_type"] == sim_type
        assert signals <= set(summary["signals"])
        assert summary["step_count"] == 1


@pytest.mark.asyncio
class TestTransientSummary:
    async def test_range_and_meas_from_real_log(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage(work_dir, "ltspice_tran_rc")
        summary = await _summary(state_no_sim, raw)

        assert summary["range"]["time_start"] == 0.0
        assert summary["range"]["time_end"] == pytest.approx(1e-3, rel=1e-6)
        # .tran 0 1m with 5u maxstep: at least 200 real samples.
        assert summary["point_count"] >= 200

        # The .meas result is parsed out of the recorded LTspice log
        # (auto-derived from the raw path) — exact value as printed there.
        vfinal = summary["measurements"]["vfinal"]
        assert vfinal["values"] == [pytest.approx(0.999876166042, rel=1e-9)]
        assert vfinal["at"] == pytest.approx(0.9e-3, rel=1e-6)

    async def test_final_value_settles_to_source_amplitude(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage(work_dir, "ltspice_tran_rc")
        # 1 ms = 10 tau after a 0 -> 1 V step: V(out) has fully settled.
        point = await _value_at(state_no_sim, raw, "V(out)", "1m")
        assert point["value"] == pytest.approx(1.0, abs=1e-3)


@pytest.mark.asyncio
class TestAcSummary:
    async def test_bandwidth_matches_rc_pole(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage(work_dir, "ltspice_ac_rc")
        summary = await _summary(state_no_sim, raw, signal="V(out)")

        assert summary["range"]["freq_start"] == pytest.approx(10.0)
        assert summary["range"]["freq_end"] == pytest.approx(100e3)
        # .ac dec 20 over 4 decades: 81 frequency points.
        assert summary["point_count"] == 81
        # R = 1k, C = 159.15n -> f_3dB = 1/(2*pi*R*C) = 1.000 kHz.
        metrics = summary["ac_bandwidth_metrics"]
        assert metrics["bandwidth_3db"] == pytest.approx(1000.0, rel=0.02)

    async def test_passband_gain_is_unity(self, state_no_sim: SessionState, work_dir: Path):
        raw = _stage(work_dir, "ltspice_ac_rc")
        # Two decades below the pole the low-pass is flat: |H| ~ 1, ~0 dB.
        point = await _value_at(state_no_sim, raw, "V(out)", "10")
        assert point["magnitude_linear"] == pytest.approx(1.0, abs=1e-3)
        assert point["magnitude_db"] == pytest.approx(0.0, abs=0.01)


@pytest.mark.asyncio
class TestDcSummary:
    async def test_sweep_range_and_midpoint_divider(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage(work_dir, "ltspice_dc_div")
        summary = await _summary(state_no_sim, raw)

        assert summary["range"]["sweep_start"] == 0.0
        assert summary["range"]["sweep_end"] == pytest.approx(5.0)
        # .dc V1 0 5 0.5 -> 11 sweep points.
        assert summary["point_count"] == 11

        # Equal-resistor divider: at the sweep midpoint V1 = 2.5 V the
        # output is exactly half the supply.
        point = await _value_at(state_no_sim, raw, "V(out)", "2.5")
        assert point["value"] == pytest.approx(1.25, rel=1e-9)


@pytest.mark.asyncio
class TestOperatingPointSummary:
    async def test_single_point_and_extreme_node_surfaced(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = _stage(work_dir, "op_extreme_node")
        summary = await _summary(state_no_sim, raw)

        # An operating point is a single bias solution: no sweep range.
        assert summary["range"] == {}
        assert summary["point_count"] == 1

        # The recorded deck drives V(hot) to 1e9 V; the handler's value scan
        # must surface it as an extreme_value observation with the real
        # magnitude (same fixture contract as test_result_observations).
        extremes = [o for o in summary["observations"] if o["code"] == "extreme_value"]
        assert extremes, summary["observations"]
        assert extremes[0]["evidence"]["trace"] == "V(hot)"
        assert extremes[0]["evidence"]["peak_abs"] == pytest.approx(1e9, rel=1e-6)
