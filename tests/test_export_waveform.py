"""Tests for the full-fidelity CSV egress behind the export recipe.

Pure-helper unit tests for the window and complex-column helpers, plus direct
tests of the CSV assembly worker — including one against a recorded LTspice
.raw, the only coverage of the binary-raw dialect flowing through assembly.
"""

import csv
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.lib.metrics import (
    window_indices as _window_indices,
)
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    _complex_columns,
    build_waveform_csv,
)
from tests.conftest import stage_recorded_fixture
from tests.test_decoded_raw import header as decoded_header


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


# --- pure helpers ----------------------------------------------------------


class TestWindowIndices:
    """Unit contract for the window-index helper.

    The descending-axis refusal is pinned here at the helper level rather than
    end-to-end: no recorded fixture has a reverse sweep, and a fabricated
    descending raw via mocks would not exercise the real binary-raw dialect.
    """

    def test_full_range_when_no_bounds(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        assert _window_indices(axis, None, None) == (0, 4)

    def test_bounds_select_subrange(self):
        axis = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        assert _window_indices(axis, 1.0, 3.0) == (1, 4)

    def test_rejects_descending_axis(self):
        # searchsorted silently corrupts a window on a descending sweep — refuse.
        axis = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        with pytest.raises(ResultError, match="non-monotonic"):
            _window_indices(axis, 2.0, 4.0)

    def test_rejects_inverted_bounds(self):
        axis = np.array([0.0, 1.0, 2.0])
        with pytest.raises(ResultError, match="must be <"):
            _window_indices(axis, 2.0, 1.0)

    def test_empty_selection_returns_equal_indices(self):
        # Empty windows are returned (lo == hi), not raised — the caller decides
        # whether to skip that step or fail the whole export.
        axis = np.array([0.0, 1.0, 2.0])
        lo, hi = _window_indices(axis, 10.0, 20.0)
        assert lo == hi


class TestComplexColumns:
    def test_mag_phase_default(self):
        wave = np.array([1.0 + 0.0j, 0.0 + 1.0j])
        names, arrays = _complex_columns("V(o)", wave, "mag_phase")
        assert names == ["V(o)_mag_dB", "V(o)_phase_deg"]
        assert arrays[1][1] == pytest.approx(90.0)  # phase of 0+1j

    def test_re_im(self):
        wave = np.array([1.0 + 2.0j])
        names, arrays = _complex_columns("V(o)", wave, "re_im")
        assert names == ["V(o)_re", "V(o)_im"]
        assert arrays[0][0] == pytest.approx(1.0)
        assert arrays[1][0] == pytest.approx(2.0)

    def test_both_has_four_columns(self):
        wave = np.array([1.0 + 1.0j])
        names, _ = _complex_columns("V(o)", wave, "both")
        assert names == ["V(o)_mag_dB", "V(o)_phase_deg", "V(o)_re", "V(o)_im"]


# --- CSV assembly worker ---------------------------------------------------


class TestBuildAndWriteWorker:
    def test_non_finite_kept_and_counted(self, work_dir: Path):
        # Direct worker test: a NaN sample must be KEPT (row count == axis length)
        # and surfaced as a count, never silently dropped.
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([0.0, np.nan, 2.0, 3.0])

        raw = DecodedRaw(
            [
                DecodedPlot(
                    decoded_header(
                        "Transient Analysis", [("time", "time"), ("V(out)", "voltage")], len(axis)
                    ),
                    [axis, wave],
                    snapshot_id="non-finite-example",
                )
            ]
        )

        out = work_dir / "out.csv"
        facts = build_waveform_csv(
            raw,
            work_dir / "mock.raw",
            [services.Signal("V(out)", "V(out)")],
            1,
            "transient",
            None,
            None,
            "mag_phase",
            out,
        )
        assert isinstance(facts, dict)  # worker returns FACTS, never a response
        assert facts["non_finite"] == 1
        assert facts["row_count"] == 4  # NaN row kept
        header, rows = _read_csv(out)
        assert header == ["time_s", "V(out)"]
        assert rows[1] == ["1.0", "nan"]  # NaN kept in place, full fidelity

    def test_empty_step_skipped_not_fatal(self, work_dir: Path):
        # A step whose axis ends before the window is skipped (not a hard error),
        # and the skip is surfaced as a fact.
        axes = {
            0: np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
            1: np.array([0.0, 1.0, 2.0]),
        }

        axis = np.concatenate([axes[0], axes[1]])
        raw = DecodedRaw(
            [
                DecodedPlot(
                    decoded_header(
                        "Transient Analysis",
                        [("time", "time"), ("V(o)", "voltage")],
                        len(axis),
                        flags=("real", "stepped"),
                    ),
                    [axis, axis * 10],
                    snapshot_id="window-example",
                    step_offsets=[0, len(axes[0])],
                )
            ]
        )

        out = work_dir / "stepped.csv"
        facts = build_waveform_csv(
            raw,
            work_dir / "m.raw",
            [services.Signal("V(o)", "V(o)")],
            2,
            "transient",
            3.0,
            4.0,
            "mag_phase",
            out,
        )
        assert facts["empty_steps"] == [1]
        header, rows = _read_csv(out)
        assert header == ["step_index", "step_value", "time_s", "V(o)"]
        assert all(int(r[0]) == 0 for r in rows)  # only step 0 contributed

    def test_no_captured_step_values_blanks_step_value(self, work_dir: Path):
        # Known boundaries without captured parameter rows retain the samples
        # and leave labels blank.
        axis = np.array([0.0, 1.0, 2.0])

        raw = DecodedRaw(
            [
                DecodedPlot(
                    decoded_header(
                        "Transient Analysis",
                        [("time", "time"), ("V(o)", "voltage")],
                        len(axis) * 2,
                        flags=("real", "stepped"),
                    ),
                    [np.concatenate([axis, axis]), np.concatenate([axis, axis * 2])],
                    snapshot_id="missing-parameters-example",
                    step_offsets=[0, len(axis)],
                )
            ]
        )

        out = work_dir / "nolog.csv"
        facts = build_waveform_csv(
            raw,
            work_dir / "nolog.raw",
            [services.Signal("V(o)", "V(o)")],
            2,
            "transient",
            None,
            None,
            "mag_phase",
            out,
        )
        assert facts["step_values_available"] is False
        _, rows = _read_csv(out)
        assert all(r[1] == "" for r in rows)

    def test_dc_x_header_names_swept_axis(self, state_no_sim: SessionState, work_dir: Path):
        # On a .dc sweep the x column carries the swept source's own name, not a
        # bare "sweep" placeholder and not the time/frequency header of another
        # analysis type — a CSV consumer reads the axis meaning from that header.
        raw_path = stage_recorded_fixture(work_dir, "ltspice_dc_div")
        raw = services.load_raw_sync(
            services.source_for_raw_path(raw_path, state_no_sim), state_no_sim
        )
        analysis_type = raw.descriptor.analysis
        out = work_dir / "dc.csv"
        build_waveform_csv(
            raw,
            raw_path,
            [services.Signal("V(out)", "V(out)")],
            1,
            analysis_type,
            None,
            None,
            "mag_phase",
            out,
        )
        header, _ = _read_csv(out)
        assert header[0] != "sweep"
        assert header[0].lower() not in ("time_s", "freq_hz")
