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
