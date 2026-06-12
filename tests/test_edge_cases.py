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
    extract_operating_point,
    query_point_value,
)
from ltspice_mcp.lib.sweep_utils import generate_sweep_range
from ltspice_mcp.lib.symbol_geometry import compute_placed_geometry, parse_asy_file

# ---------------------------------------------------------------------------
# generate_sweep_range crashes on log scale with step=1
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
# generate_sweep_range with points<2 silently produces useless output
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
# parse_spice_value is case-sensitive but SPICE convention is not
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
# extract_log_diagnostics substring false positives
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

    def _check(self, text: str):
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
# filter_runs_by_params silently matches NaN run values
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
# compute_placed_geometry assumes symbol bbox starts at (0,0),
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
# get_progress_snapshot can produce negative ETA / negative elapsed
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
# _resolve_mc_ref preserved surrounding whitespace
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
# Stability margin detection on phase wrap and 3-pole unstable loop
# (Margins moved out of compute_ac_bandwidth_metrics — they live in
# compute_stability_metrics, which is the right home for loop-gain analysis.)
# ---------------------------------------------------------------------------


class TestStabilityPhaseWrap:
    def test_phase_wrap_does_not_hide_180_crossing(self):
        from ltspice_mcp.lib.ac_analysis import compute_stability_metrics, prepare_ac_arrays

        # Construct a small frequency response whose phase crosses -180°.
        # Without np.unwrap, np.angle wraps -181° to +179° and the
        # gain-margin detection misses the crossing entirely.
        freqs = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        mag_lin = np.array([10**0.5, 10**0.25, 1.0, 10 ** (-0.25), 10 ** (-0.5)])
        phase_rad = np.deg2rad(np.array([-90, -150, -179, -181, -210]))
        wave = mag_lin * np.exp(1j * phase_rad)

        f, H = prepare_ac_arrays(freqs, wave)
        result = compute_stability_metrics(f, H)
        assert result["gain_margins"]
        assert result["gain_margin_worst_db"] is not None

    def test_3pole_unstable_system(self):
        from ltspice_mcp.lib.ac_analysis import compute_stability_metrics, prepare_ac_arrays

        freqs = np.logspace(-1, 6, 500)
        omega = 2 * np.pi * freqs
        # 3-pole loop: should be unstable, phase margin negative
        H = 1e6 / ((1 + 1j * omega / 1) * (1 + 1j * omega / 100) * (1 + 1j * omega / 1000))
        f, Hp = prepare_ac_arrays(freqs, H)
        result = compute_stability_metrics(f, Hp)
        # An unstable 3-pole loop must report a negative phase margin
        assert result["phase_margins"]
        assert result["phase_margin_worst_deg"] is not None
        assert result["phase_margin_worst_deg"] < 0


class TestAcBandwidthMetrics:
    """compute_ac_bandwidth_metrics now reports only -3 dB and unity-gain freq;
    margins live in compute_stability_metrics."""

    def test_returns_only_bandwidth_and_unity_gain(self):
        from ltspice_mcp.lib.raw_parser import compute_ac_bandwidth_metrics

        freqs = np.logspace(0, 5, 200)
        omega = 2 * np.pi * freqs
        # Simple single-pole LPF at 1 kHz
        H = 1.0 / (1 + 1j * omega / (2 * np.pi * 1000))
        raw = MagicMock()
        raw.get_axis.return_value = freqs
        raw.get_wave = lambda name, step=0: H
        result = compute_ac_bandwidth_metrics(raw, "V(out)")
        assert set(result.keys()) == {"bandwidth_3db", "unity_gain_freq"}
        assert result["bandwidth_3db"] is not None
        # -3 dB point should be near 1 kHz
        assert 900 < result["bandwidth_3db"] < 1100


# ---------------------------------------------------------------------------
# library_parser nested .SUBCKT, no-space paren, PARAMS: keyword
# ---------------------------------------------------------------------------
# (Tested in test_library_parser.py — see TestParseLibraryFile.)


# ---------------------------------------------------------------------------
# handle_connect silently produces zero-wire connections for self-loops
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
# Linear sweep with mismatched step direction silently returns []
# ---------------------------------------------------------------------------


class TestSweepDirectionMismatch:
    def test_descending_range_with_positive_step_raises(self):
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(10, 1, step=+1, points=None, scale="linear")

    def test_ascending_range_with_negative_step_raises(self):
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(1, 10, step=-1, points=None, scale="linear")

    def test_log_ascending_with_shrinking_step_raises(self):
        # Ascending range (1 → 100) with step < 1 produces log(100)/log(0.1) = -2,
        # which yields n = -1 and crashes np.geomspace with "Number of samples,
        # -1, must be non-negative." Should raise a clean direction-mismatch error.
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(1, 100, step=0.1, points=None, scale="log")

    def test_log_descending_with_growing_step_raises(self):
        # Symmetric: descending range (100 → 1) with step > 1 also yields n = -1.
        with pytest.raises(ValueError, match="direction"):
            generate_sweep_range(100, 1, step=10.0, points=None, scale="log")

    def test_log_descending_with_shrinking_step_works(self):
        # Sanity: descending log sweep with step < 1 is the legitimate use.
        result = generate_sweep_range(100, 1, step=0.1, points=None, scale="log")
        assert result == pytest.approx([100.0, 10.0, 1.0])


# ---------------------------------------------------------------------------
# parse_spice_value rejected SPICE values with trailing unit
# annotations (1ms, 10us, 1uF, 100pF, 1mV, 10mA), even though the suffix
# itself was valid. SPICE tradition treats trailing letters after the
# scale suffix as ignorable unit annotations.
# ---------------------------------------------------------------------------


class TestParseSpiceTrailingUnits:
    """Trailing unit letters after a valid suffix must be ignored."""

    def test_milliseconds(self):
        assert parse_spice_value("1ms") == pytest.approx(1e-3)

    def test_microseconds(self):
        assert parse_spice_value("10us") == pytest.approx(10e-6)

    def test_nanoseconds(self):
        assert parse_spice_value("5ns") == pytest.approx(5e-9)

    def test_picoseconds(self):
        assert parse_spice_value("100ps") == pytest.approx(100e-12)

    def test_microfarads(self):
        assert parse_spice_value("1uF") == pytest.approx(1e-6)

    def test_picofarads(self):
        assert parse_spice_value("10pF") == pytest.approx(10e-12)

    def test_millihenries(self):
        assert parse_spice_value("1mH") == pytest.approx(1e-3)

    def test_megahertz(self):
        # 'MegHz' — suffix 'Meg' must match before the case-insensitive 'hz'
        # tail is considered an annotation. Must not collapse to milli.
        assert parse_spice_value("10MegHz") == pytest.approx(10e6)

    def test_millivolts(self):
        assert parse_spice_value("1mV") == pytest.approx(1e-3)

    def test_milliamps(self):
        assert parse_spice_value("10mA") == pytest.approx(10e-3)

    def test_kilohms(self):
        assert parse_spice_value("1kohm") == pytest.approx(1e3)

    def test_no_suffix_with_trailing_garbage_still_raises(self):
        # Conservative: only ignore the tail when it begins with a known
        # suffix. '1Hz' has no recognised suffix prefix → still rejected.
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_spice_value("1Hz")

    def test_pure_garbage_still_raises(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_spice_value("hello")

    def test_trailing_digits_still_rejected(self):
        # Must not match — '1k1' has digits after the suffix; ambiguous.
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_spice_value("1k1")


# ---------------------------------------------------------------------------
# is_windows_native_path matches /mnt/cdrom (false positive)
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
# parse_measurements crashes on unparseable string values
# ---------------------------------------------------------------------------


class TestParseMeasurementsUnparseable:
    def test_unparseable_string_becomes_none(self):
        from ltspice_mcp.lib.log_parser import parse_measurements

        class FakeReader:
            def __init__(self, data):
                self.dataset = data

            def get_measure_names(self):
                return list(self.dataset.keys())

        reader = FakeReader({"fc": ["unparseable", 100.0]})
        result = parse_measurements(Path("/tmp/x.log"), reader=reader)  # type: ignore[arg-type]
        # Crashing was the bug; the unparseable value should become None.
        assert result["measurements"]["fc"]["values"] == [None, 100.0]


# ---------------------------------------------------------------------------
# handle_check_job reports queued status as 'unexpected'
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
# handle_set_component_value silently accepts contradictory inputs
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


# ---------------------------------------------------------------------------
# config.py silently accepted bad TOML values (wrong type, negative,
# out-of-range) and invalid log levels
# ---------------------------------------------------------------------------


class TestConfigTomlValidation:
    def _load(self, tmp_path: Path, toml_content: str, monkeypatch):
        from ltspice_mcp.config import ServerConfig

        p = tmp_path / "c.toml"
        p.write_text(toml_content)
        monkeypatch.setenv("LTSPICE_MCP_CONFIG", str(p))
        return ServerConfig.load()

    def test_string_timeout_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(
            tmp_path,
            '[simulation]\ntimeout = "not a number"\n',
            monkeypatch,
        )
        # Should fall back to the default, not store a string
        assert cfg.default_timeout == 300.0

    def test_negative_timeout_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, "[simulation]\ntimeout = -5\n", monkeypatch)
        assert cfg.default_timeout == 300.0

    def test_huge_timeout_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, "[simulation]\ntimeout = 999999999\n", monkeypatch)
        assert cfg.default_timeout == 300.0

    def test_zero_max_parallel_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, "[simulation]\nmax_parallel = 0\n", monkeypatch)
        assert cfg.max_parallel_sims == 4

    def test_negative_max_parallel_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, "[simulation]\nmax_parallel = -1\n", monkeypatch)
        assert cfg.max_parallel_sims == 4

    def test_invalid_log_level_rejected(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, '[logging]\nlevel = "SUPERDEBUG"\n', monkeypatch)
        assert cfg.log_level == "INFO"

    def test_lowercase_log_level_normalized(self, tmp_path, monkeypatch):
        cfg = self._load(tmp_path, '[logging]\nlevel = "debug"\n', monkeypatch)
        assert cfg.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# _resolve_result_file accepted empty-string paths as valid
# ---------------------------------------------------------------------------


class TestResolveResultFileEmpty:
    def test_batch_empty_string_path_rejected(self, state_no_sim):
        from datetime import timedelta

        from ltspice_mcp.errors import ResultError
        from ltspice_mcp.lib import now, services
        from ltspice_mcp.state import BatchJob

        bj = BatchJob(
            job_id="b1",
            job_type="sweep",
            netlist=Path("/tmp/x.cir"),
            total_runs=1,
            completed_runs=1,
            status="completed",
        )
        bj.completed_at = now() + timedelta(seconds=1)
        bj.run_results = {0: {"raw_file": "", "log_file": "", "params": {}}}
        state_no_sim.batch_jobs["b1"] = bj

        with pytest.raises(ResultError, match="no raw file"):
            services.resolve_raw_file("b1", state_no_sim)


# ---------------------------------------------------------------------------
# get_batch_signal_data accepted negative offset / zero limit
# ---------------------------------------------------------------------------


class TestBatchPaginationValidation:
    def _make_bj(self, state, n_runs: int = 10):
        from datetime import timedelta

        from ltspice_mcp.lib import now
        from ltspice_mcp.state import BatchJob

        bj = BatchJob(
            job_id="b1",
            job_type="sweep",
            netlist=Path("/tmp/x.cir"),
            total_runs=n_runs,
            completed_runs=n_runs,
            status="completed",
        )
        bj.completed_at = now() + timedelta(seconds=1)
        bj.run_results = {
            i: {
                "raw_file": Path(f"/tmp/r{i}.raw"),
                "log_file": Path(f"/tmp/r{i}.log"),
                "params": {},
            }
            for i in range(n_runs)
        }
        state.batch_jobs["b1"] = bj
        return bj

    def test_negative_offset_rejected(self, state_no_sim):
        from ltspice_mcp.errors import BatchJobError
        from ltspice_mcp.lib import services

        bj = self._make_bj(state_no_sim)
        with pytest.raises(BatchJobError, match="offset"):
            services.get_batch_signal_data(bj, "V(out)", raw=True, offset=-5, limit=5)

    def test_zero_limit_rejected(self, state_no_sim):
        from ltspice_mcp.errors import BatchJobError
        from ltspice_mcp.lib import services

        bj = self._make_bj(state_no_sim)
        with pytest.raises(BatchJobError, match="limit"):
            services.get_batch_signal_data(bj, "V(out)", raw=True, offset=0, limit=0)


# ---------------------------------------------------------------------------
# handle_add_component corrupted the .asc file when given a
# nonexistent symbol name, making the file unopenable afterwards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestAddComponentSymbolValidation:
    async def test_nonexistent_symbol_rejected(self, asc_state, asc_file):
        from spicelib import AscEditor

        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import AddComponentInput, handle_add_component

        with pytest.raises(NetlistError, match="not found in any configured"):
            await handle_add_component(
                AddComponentInput(
                    path=asc_file.name,
                    reference="X99",
                    symbol="totally_fake_symbol_xyz",
                    x=0,
                    y=0,
                ),
                asc_state,
            )
        # The file must still be readable (previously this would corrupt it)
        editor = AscEditor(str(asc_file))
        assert "X99" not in editor.components


# ---------------------------------------------------------------------------
# Round 5: continuation-line merge, edit_directive empty patterns, queued
# status, AC/DC substring false-positives, list_components metacharacters.
# ---------------------------------------------------------------------------


class TestMergeContinuationBlankLine:
    def test_blank_line_preserves_continuation(self):
        from ltspice_mcp.lib.library_parser import _merge_continuation_lines

        # A blank line between a definition and its '+' continuation used to
        # reset `current` to the empty string, producing a garbage
        # ' BF=200' line and losing the real definition's params.
        result = _merge_continuation_lines([".MODEL Q NPN", "", "+ BF=200"])
        assert result == [".MODEL Q NPN BF=200"]

    def test_multiple_blank_lines(self):
        from ltspice_mcp.lib.library_parser import _merge_continuation_lines

        result = _merge_continuation_lines([".MODEL Q NPN", "", "", "+ BF=200", "+ IS=1e-14"])
        assert result == [".MODEL Q NPN BF=200 IS=1e-14"]


@pytest.mark.asyncio
class TestEditDirectiveEmpty:
    async def test_empty_instruction_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import EditDirectiveInput, handle_edit_directive

        with pytest.raises(NetlistError, match="must not be empty"):
            await handle_edit_directive(
                EditDirectiveInput(path=sample_netlist.name, action="add", instruction=""),
                state_no_sim,
            )

    async def test_empty_regex_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import EditDirectiveInput, handle_edit_directive

        with pytest.raises(NetlistError, match="Empty regex"):
            await handle_edit_directive(
                EditDirectiveInput(
                    path=sample_netlist.name, action="remove", instruction="regex:"
                ),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestHandleParameterModes:
    async def test_value_without_name_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ParameterInput, handle_parameter

        with pytest.raises(NetlistError, match="requires 'name'"):
            await handle_parameter(
                ParameterInput(path=sample_netlist.name, value="2k"),
                state_no_sim,
            )

    async def test_empty_name_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ParameterInput, handle_parameter

        with pytest.raises(NetlistError, match="name must not be empty"):
            await handle_parameter(
                ParameterInput(path=sample_netlist.name, name=" ", value="2k"),
                state_no_sim,
            )

    async def test_read_single_param(self, state_no_sim, sample_netlist):
        from ltspice_mcp.tools.circuit import ParameterInput, handle_parameter

        r = await handle_parameter(
            ParameterInput(path=sample_netlist.name, name="Rval"),
            state_no_sim,
        )
        # Previously returned ALL params when given only name.
        assert "Rval" in r.structuredContent["parameters"]
        assert len(r.structuredContent["parameters"]) == 1


class TestSimulatorSelectionCaseInsensitive:
    def test_uppercase_preference(self):
        from ltspice_mcp.config import ServerConfig
        from ltspice_mcp.lib.simulator import select_default_simulator

        class LT:
            pass

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.simulator = "LTSPICE"
        assert select_default_simulator({"ltspice": LT}, cfg) is LT

    def test_whitespace_preference(self):
        from ltspice_mcp.config import ServerConfig
        from ltspice_mcp.lib.simulator import select_default_simulator

        class NG:
            pass

        cfg = ServerConfig(working_dir=Path("/tmp"), allowed_paths=[Path("/tmp")])
        cfg.simulator = "  ngspice  "
        assert select_default_simulator({"ngspice": NG}, cfg) is NG


class TestIsAcAnalysisWordBoundary:
    def test_ac_matches(self):
        from ltspice_mcp.lib.raw_parser import is_ac_analysis

        assert is_ac_analysis("AC Analysis") is True
        assert is_ac_analysis("ac analysis") is True

    def test_substring_not_matched(self):
        from ltspice_mcp.lib.raw_parser import is_ac_analysis

        # Previously all these returned True because "AC" is a substring
        assert is_ac_analysis("backup") is False
        assert is_ac_analysis("BACK tracking") is False
        assert is_ac_analysis("DC transfer characteristic") is False


class TestBuildSimulationSummaryRange:
    def _mk(self, plotname, axis):
        raw = MagicMock()
        raw.get_raw_property.return_value = plotname
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [0]
        raw.get_axis.return_value = axis
        raw.get_wave = lambda n, step=0: axis
        return raw

    def test_dc_not_misclassified_as_ac(self):
        from ltspice_mcp.lib.raw_parser import build_simulation_summary

        r = build_simulation_summary(
            self._mk("DC transfer characteristic", np.array([0, 1, 2])), None
        )
        # Previously "characteristic" contained the substring "AC" so DC
        # ended up with freq_start/freq_end keys.
        assert "sweep_start" in r["range"]
        assert "freq_start" not in r["range"]

    def test_empty_axis_does_not_crash(self):
        from ltspice_mcp.lib.raw_parser import build_simulation_summary

        r = build_simulation_summary(self._mk("Transient Analysis", np.array([])), None)
        assert r["range"] == {}
        assert r["point_count"] == 0


class TestExtractOperatingPointCaseInsensitive:
    def test_lowercase_trace_names(self):
        from ltspice_mcp.lib.raw_parser import extract_operating_point

        raw = MagicMock()
        raw.get_trace_names.return_value = ["v(out)", "i(r1)"]
        waves = {"v(out)": np.array([3.3]), "i(r1)": np.array([0.001])}
        raw.get_wave = lambda n, step=0: waves[n]
        r = extract_operating_point(raw)
        # Previously lowercase V/I prefixes were silently dropped.
        assert r["voltages"].get("v(out)") == 3.3
        assert r["currents"].get("i(r1)") == 0.001


@pytest.mark.asyncio
class TestMoveComponentWraps:
    async def test_move_unknown_ref_raises_netlist_error(self, asc_state, asc_file):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import MoveComponentInput, handle_move_component

        # Previously leaked spicelib's ComponentNotFoundError.
        with pytest.raises(NetlistError, match="not found"):
            await handle_move_component(
                MoveComponentInput(path=asc_file.name, reference="ZZZ", x=0, y=0),
                asc_state,
            )


@pytest.mark.asyncio
class TestSetComponentAttributeWraps:
    async def test_unknown_ref_raises_netlist_error(self, asc_state, asc_file):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import (
            SetComponentAttributeInput,
            handle_set_component_attribute,
        )

        with pytest.raises(NetlistError, match="not found"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name, reference="ZZZ", attribute="SpiceLine", value="x"
                ),
                asc_state,
            )

    async def test_empty_attribute_rejected(self, asc_state, asc_file):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import (
            SetComponentAttributeInput,
            handle_set_component_attribute,
        )

        with pytest.raises(NetlistError, match="not be empty"):
            await handle_set_component_attribute(
                SetComponentAttributeInput(
                    path=asc_file.name, reference="R1", attribute="  ", value="x"
                ),
                asc_state,
            )


@pytest.mark.asyncio
class TestListComponentsValidation:
    async def test_reference_and_prefix_mutually_exclusive(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ListComponentsInput, handle_list_components

        with pytest.raises(NetlistError, match="mutually exclusive"):
            await handle_list_components(
                ListComponentsInput(path=sample_netlist.name, reference="R1", prefix="C"),
                state_no_sim,
            )

    async def test_metachar_prefix_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ListComponentsInput, handle_list_components

        # Previously propagated a raw NotImplementedError from spicelib.
        with pytest.raises(NetlistError, match="single letter"):
            await handle_list_components(
                ListComponentsInput(path=sample_netlist.name, prefix="R.*"),
                state_no_sim,
            )

    async def test_multichar_prefix_rejected(self, state_no_sim, sample_netlist):
        from ltspice_mcp.errors import NetlistError
        from ltspice_mcp.tools.circuit import ListComponentsInput, handle_list_components

        with pytest.raises(NetlistError, match="single letter"):
            await handle_list_components(
                ListComponentsInput(path=sample_netlist.name, prefix="RR"),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestQueryValueRejectsNaNInf:
    async def test_nan_at_rejected(self, state_no_sim, work_dir):
        import numpy as np

        from ltspice_mcp.errors import ResultError
        from ltspice_mcp.tools.analysis import QueryValueInput, handle_query_value

        raw_file = work_dir / "x.raw"
        raw_file.write_bytes(b"placeholder")
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [0]
        axis = np.array([0.0, 1.0, 2.0])
        raw.get_axis.return_value = axis
        raw.get_wave = lambda n, step=0: axis
        state_no_sim.results.set(raw_file, raw)

        with pytest.raises(ResultError, match="finite"):
            await handle_query_value(
                QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="nan"),
                state_no_sim,
            )

    async def test_inf_at_rejected(self, state_no_sim, work_dir):
        import numpy as np

        from ltspice_mcp.errors import ResultError
        from ltspice_mcp.tools.analysis import QueryValueInput, handle_query_value

        raw_file = work_dir / "x.raw"
        raw_file.write_bytes(b"placeholder")
        raw = MagicMock()
        raw.get_raw_property.return_value = "Transient Analysis"
        raw.get_trace_names.return_value = ["time", "V(out)"]
        raw.get_steps.return_value = [0]
        axis = np.array([0.0, 1.0, 2.0])
        raw.get_axis.return_value = axis
        raw.get_wave = lambda n, step=0: axis
        state_no_sim.results.set(raw_file, raw)

        with pytest.raises(ResultError, match="finite"):
            await handle_query_value(
                QueryValueInput(raw_file=raw_file.name, signal="V(out)", at="inf"),
                state_no_sim,
            )
