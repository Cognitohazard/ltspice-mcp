"""Edge-case tests that probe pure functions for real bugs.

Each test asserts the *correct* behavior. Tests in this file were written
specifically to find logic bugs by exercising boundary conditions, malformed
input, and edge cases that the happy-path tests don't cover.
"""

import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.batch_results import filter_runs_by_params
from ltspice_mcp.lib.format import parse_spice_value
from ltspice_mcp.lib.log_parser import extract_log_diagnostics
from ltspice_mcp.lib.raw_parser import (
    compute_signal_stats,
    extract_operating_point,
    query_point_value,
)
from ltspice_mcp.lib.sweep_utils import generate_sweep_range
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file

# ---------------------------------------------------------------------------
# Bug A: generate_sweep_range crashes on log scale with step=1
# ---------------------------------------------------------------------------


class TestSweepLogStepEdgeCases:
    """log scale uses log(stop/start) / log(step), so step=1 → div-by-zero."""

    def test_log_step_one_raises_clean_error(self):
        # step=1 means "multiply by 1 each time" which is degenerate.
        # Should raise our own ValueError, not numpy/Python's ZeroDivisionError.
        with pytest.raises(ValueError, match="degenerate"):
            generate_sweep_range(1, 100, step=1.0, points=None, scale="log")

    def test_log_step_zero_raises(self):
        with pytest.raises(ValueError, match="positive"):
            generate_sweep_range(1, 100, step=0.0, points=None, scale="log")

    def test_log_step_negative_raises(self):
        with pytest.raises(ValueError, match="positive"):
            generate_sweep_range(1, 100, step=-2.0, points=None, scale="log")


# ---------------------------------------------------------------------------
# Bug B: generate_sweep_range with points<2 silently produces useless output
# ---------------------------------------------------------------------------


class TestSweepPointsEdgeCases:
    def test_points_zero_raises(self):
        # points=0 currently returns [] silently. Should raise.
        with pytest.raises(ValueError, match="points"):
            generate_sweep_range(1, 10, step=None, points=0, scale="linear")

    def test_points_negative_raises_value_error(self):
        # Currently propagates numpy's error message; should be a clean ValueError.
        with pytest.raises(ValueError, match="points"):
            generate_sweep_range(1, 10, step=None, points=-5, scale="linear")


# ---------------------------------------------------------------------------
# Bug C: parse_spice_value is case-sensitive but SPICE convention is not
# ---------------------------------------------------------------------------


class TestParseSpiceCaseSensitivity:
    """SPICE/LTspice scale suffixes are case-insensitive."""

    def test_uppercase_K(self):
        assert parse_spice_value("1K") == 1000.0

    def test_uppercase_MEG(self):
        assert parse_spice_value("1MEG") == 1e6

    def test_lowercase_meg(self):
        assert parse_spice_value("1meg") == 1e6

    def test_mixed_case_meg(self):
        assert parse_spice_value("1mEg") == 1e6

    def test_uppercase_G(self):
        # 'G' already in table; verify it works
        assert parse_spice_value("1G") == 1e9

    def test_uppercase_T(self):
        assert parse_spice_value("1T") == 1e12


# ---------------------------------------------------------------------------
# Bug D: compute_signal_stats on empty wave crashes with cryptic numpy error
# ---------------------------------------------------------------------------


class TestComputeSignalStatsEmpty:
    def test_empty_wave_raises_clean_error(self):
        raw = MagicMock()
        raw.get_axis.return_value = np.array([])
        raw.get_wave = lambda name, step=0: np.array([])
        raw.get_steps.return_value = [0]
        # Should raise a domain error, not a numpy reduction error.
        with pytest.raises((ResultError, ValueError)):
            compute_signal_stats(raw, "V(out)")


# ---------------------------------------------------------------------------
# Bug E: extract_log_diagnostics substring false positives
# ---------------------------------------------------------------------------


class TestQueryPointValueEmpty:
    def test_empty_axis_raises(self):
        raw = MagicMock()
        raw.get_axis.return_value = np.array([])
        raw.get_wave = lambda name, step=0: np.array([])
        with pytest.raises((ResultError, ValueError)):
            query_point_value(raw, "V(out)", target_x=1.0)


class TestExtractOperatingPointEmpty:
    def test_empty_wave_skipped(self):
        # A trace with no data points should be silently skipped
        raw = MagicMock()
        raw.get_trace_names.return_value = ["V(out)", "I(R1)"]
        waves = {"V(out)": np.array([]), "I(R1)": np.array([0.001])}
        raw.get_wave = lambda name, step=0: waves[name]
        result = extract_operating_point(raw)
        # V(out) is skipped (no data); I(R1) is included
        assert "V(out)" not in result["voltages"]
        assert result["currents"]["I(R1)"] == 0.001


class TestLogDiagnosticsFalsePositives:
    """The bare-phrase check uses substring matching, causing false positives."""

    def _check(self, text: str) -> dict:
        with tempfile.NamedTemporaryFile(suffix=".log", mode="w", delete=False) as t:
            t.write(text + "\n")
            return extract_log_diagnostics(Path(t.name))

    def test_singular_matrix_substring_not_flagged(self):
        # "the singular matrix decomposition succeeded" should NOT be an error.
        result = self._check("the singular matrix decomposition succeeded")
        assert result["errors"] == []

    def test_time_step_substring_not_flagged(self):
        result = self._check("time step too small for the user but ok for sim")
        assert result["errors"] == []

    def test_no_convergence_substring_not_flagged(self):
        result = self._check("previous run had no convergence issues")
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Bug F: filter_runs_by_params silently matches NaN run values
# ---------------------------------------------------------------------------


class TestFilterRunsByParamsNaN:
    """NaN should never match a numeric filter (NaN comparisons return False)."""

    def test_nan_value_does_not_match_exact(self):
        runs = {
            0: {"params": {"R": 1000.0}},
            1: {"params": {"R": math.nan}},
            2: {"params": {"R": 1000.0}},
        }
        result = filter_runs_by_params(runs, {"R": "1k"})
        assert result == [0, 2]  # NaN run #1 must NOT match

    def test_nan_value_does_not_match_range(self):
        runs = {0: {"params": {"R": math.nan}}}
        result = filter_runs_by_params(runs, {"R": "0..10k"})
        assert result == []

    def test_nan_filter_target_matches_nothing(self):
        runs = {0: {"params": {"R": 1000.0}}}
        result = filter_runs_by_params(runs, {"R": "nan"})
        assert result == []


# ---------------------------------------------------------------------------
# Bug G: compute_placed_geometry assumes symbol bbox starts at (0,0),
# producing a bounding box that doesn't enclose pins on centered symbols.
# ---------------------------------------------------------------------------


class TestSymbolGeometryBboxContainsPins:
    """A correctly-computed placed bbox must contain every placed pin."""

    @pytest.fixture
    def nmos_sym(self):
        # Use the real fixture symbol — pins span (-48, -96) to (0, 96).
        return parse_asy_file(Path(__file__).parent / "fixtures" / "symbols" / "nmos.asy")

    @pytest.mark.parametrize(
        "rotation",
        ["R0", "R90", "R180", "R270", "M0", "M90", "M180", "M270"],
    )
    def test_bbox_contains_pins(self, nmos_sym, rotation: str):
        geo = compute_placed_geometry(nmos_sym, origin_x=500, origin_y=500, rotation=rotation)
        bbox = geo["bounding_box"]
        for pin in geo["pins"]:
            assert bbox["x"] <= pin["x"] <= bbox["x"] + bbox["width"], (
                f"{rotation}: pin {pin['name']} x={pin['x']} outside bbox {bbox}"
            )
            assert bbox["y"] <= pin["y"] <= bbox["y"] + bbox["height"], (
                f"{rotation}: pin {pin['name']} y={pin['y']} outside bbox {bbox}"
            )

    def test_pin_directions_correct_for_nmos(self, nmos_sym):
        # The .asy file declares D=TOP, G=LEFT, S=BOTTOM. With the bbox bug,
        # pin S was misclassified as 'left' because the bbox center was wrong.
        geo = compute_placed_geometry(nmos_sym, origin_x=0, origin_y=0, rotation="R0")
        dirs = {p["name"]: p["dir"] for p in geo["pins"]}
        assert dirs["D"] == "up"
        assert dirs["G"] == "left"
        assert dirs["S"] == "down"


# ---------------------------------------------------------------------------
# Bug H: get_progress_snapshot can produce negative ETA / negative elapsed
# ---------------------------------------------------------------------------


class TestGetProgressSnapshotEdgeCases:
    def test_overshoot_does_not_produce_negative_eta(self):
        import time
        from pathlib import Path

        from ltspice_mcp.lib.batch_results import get_progress_snapshot
        from ltspice_mcp.state import BatchJob

        bj = BatchJob(
            job_id="b1",
            job_type="sweep",
            netlist=Path("/x"),
            total_runs=10,
            completed_runs=15,  # overshoot
            failed_runs=0,
        )
        snap = get_progress_snapshot(bj, time.time() - 1)
        # ETA should be 0 (already done), not negative
        assert snap["eta_s"] is None or snap["eta_s"] >= 0

    def test_future_start_time_clamps_elapsed(self):
        import time
        from pathlib import Path

        from ltspice_mcp.lib.batch_results import get_progress_snapshot
        from ltspice_mcp.state import BatchJob

        bj = BatchJob(
            job_id="b1",
            job_type="sweep",
            netlist=Path("/x"),
            total_runs=10,
            completed_runs=5,
        )
        snap = get_progress_snapshot(bj, time.time() + 100)
        # Negative elapsed is nonsensical; should be clamped to 0
        assert snap["elapsed_s"] >= 0


# ---------------------------------------------------------------------------
# Bug I: _resolve_mc_ref preserved surrounding whitespace
# ---------------------------------------------------------------------------


class TestResolveMcRefWhitespace:
    def test_surrounding_whitespace_stripped(self):
        from ltspice_mcp.tools.advanced import _resolve_mc_ref

        ref, is_type = _resolve_mc_ref("  R1  ")
        assert ref == "R1"
        assert is_type is False

    def test_whitespace_around_type_name(self):
        from ltspice_mcp.tools.advanced import _resolve_mc_ref

        ref, is_type = _resolve_mc_ref("  resistors ")
        assert ref == "R"
        assert is_type is True


# ---------------------------------------------------------------------------
# Bug J: compute_ac_bandwidth_metrics misses -180° phase crossings due to wrap
# ---------------------------------------------------------------------------


class TestAcBandwidthPhaseWrap:
    def test_phase_wrap_does_not_hide_180_crossing(self):
        from ltspice_mcp.lib.raw_parser import compute_ac_bandwidth_metrics

        # Construct a small frequency response whose phase crosses -180°.
        # Without np.unwrap, np.angle wraps -181° to +179° and the
        # gain-margin detection misses the crossing entirely.
        freqs = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        mag_lin = np.array([10**0.5, 10**0.25, 1.0, 10**(-0.25), 10**(-0.5)])
        phase_rad = np.deg2rad(np.array([-90, -150, -179, -181, -210]))
        wave = mag_lin * np.exp(1j * phase_rad)

        raw = MagicMock()
        raw.get_axis.return_value = freqs
        raw.get_wave = lambda name, step=0: wave
        result = compute_ac_bandwidth_metrics(raw, "V(out)")
        assert result["gain_margin"] is not None

    def test_3pole_unstable_system(self):
        from ltspice_mcp.lib.raw_parser import compute_ac_bandwidth_metrics

        freqs = np.logspace(-1, 6, 500)
        omega = 2 * np.pi * freqs
        # 3-pole loop: should be unstable, phase margin negative
        H = 1e6 / ((1 + 1j * omega / 1) * (1 + 1j * omega / 100) * (1 + 1j * omega / 1000))
        raw = MagicMock()
        raw.get_axis.return_value = freqs
        raw.get_wave = lambda name, step=0: H
        result = compute_ac_bandwidth_metrics(raw, "V(out)")
        # An unstable 3-pole loop must report a negative phase margin
        assert result["phase_margin"] is not None
        assert result["phase_margin"] < 0


# ---------------------------------------------------------------------------
# Bug K: library_parser nested .SUBCKT, no-space paren, PARAMS: keyword
# ---------------------------------------------------------------------------
# (Tested in test_library_parser.py — see TestParseLibraryFile.)


# ---------------------------------------------------------------------------
# Bug L: handle_connect silently produces zero-wire connections for self-loops
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConnectZeroLength:
    async def test_self_loop_rejected(self, asc_state, asc_file):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ConnectInput, handle_connect

        with pytest.raises(NetlistError, match="same coordinate"):
            await handle_connect(
                ConnectInput(path=asc_file.name, from_pin="R1.1", to_pin="R1.1"),
                asc_state,
            )


# ---------------------------------------------------------------------------
# Bug M: linear sweep with mismatched step direction silently returns []
# ---------------------------------------------------------------------------


class TestSweepDirectionMismatch:
    def test_descending_range_with_positive_step_raises(self):
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(10, 1, step=+1, points=None, scale="linear")

    def test_ascending_range_with_negative_step_raises(self):
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(1, 10, step=-1, points=None, scale="linear")


# ---------------------------------------------------------------------------
# Bug N: is_windows_native_path matches /mnt/cdrom (false positive)
# ---------------------------------------------------------------------------


class TestIsWindowsNativePath:
    def test_drive_letter_match(self):
        from ltspice_mcp.lib.wsl import is_windows_native_path
        assert is_windows_native_path(Path("/mnt/c/Users/foo")) is True

    def test_cdrom_not_drive(self):
        from ltspice_mcp.lib.wsl import is_windows_native_path
        # /mnt/cdrom is not a Windows drive letter — must NOT match
        assert is_windows_native_path(Path("/mnt/cdrom/foo")) is False

    def test_extdata_not_drive(self):
        from ltspice_mcp.lib.wsl import is_windows_native_path
        assert is_windows_native_path(Path("/mnt/extdata/x")) is False

    def test_mnt_alone_not_drive(self):
        from ltspice_mcp.lib.wsl import is_windows_native_path
        assert is_windows_native_path(Path("/mnt")) is False


# ---------------------------------------------------------------------------
# Bug O: parse_measurements crashes on unparseable string values
# ---------------------------------------------------------------------------


class TestParseMeasurementsUnparseable:
    def test_unparseable_string_becomes_none(self):
        from ltspice_mcp.lib.log_parser import parse_measurements

        class FakeReader:
            def __init__(self, data): self.dataset = data
            def get_measure_names(self): return list(self.dataset.keys())

        reader = FakeReader({"fc": ["unparseable", 100.0]})
        result = parse_measurements(Path("/tmp/x.log"), reader=reader)  # type: ignore[arg-type]
        # Crashing was the bug; the unparseable value should become None.
        assert result["measurements"]["fc"] == [None, 100.0]


# ---------------------------------------------------------------------------
# Bug P: handle_check_job reports queued status as 'unexpected'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCheckJobQueued:
    async def test_queued_job_reported_correctly(self, state_no_sim):
        from ltspice_mcp.lib import now
        from ltspice_mcp.state import SimulationJob
        from ltspice_mcp.tools.simulation import CheckJobInput, handle_check_job

        state_no_sim.jobs["jq"] = SimulationJob(
            job_id="jq",
            netlist=Path("/tmp/x.cir"),
            simulator="F",
            status="queued",
            started_at=now(),
        )
        r = await handle_check_job(CheckJobInput(job_id="jq"), state_no_sim)
        assert "unexpected" not in r.content[0].text
        assert r.structuredContent["status"] == "queued"


# ---------------------------------------------------------------------------
# Bug Q: handle_set_component_value silently accepts contradictory inputs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSetComponentValueAmbiguous:
    async def test_both_modes_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import SetComponentValueInput, handle_set_component_value

        with pytest.raises(NetlistError, match="mutually exclusive"):
            await handle_set_component_value(
                SetComponentValueInput(
                    path=sample_netlist.name,
                    reference="R1",
                    value="2k",
                    values={"C1": "5n"},
                ),
                state_no_sim,
            )

    async def test_empty_values_dict_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import SetComponentValueInput, handle_set_component_value

        with pytest.raises(NetlistError, match="empty"):
            await handle_set_component_value(
                SetComponentValueInput(path=sample_netlist.name, values={}),
                state_no_sim,
            )
