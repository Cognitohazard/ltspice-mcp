"""Tests for export_waveform — full-fidelity CSV egress.

Pure-helper unit tests plus integration tests through the real handler and
recorded LTspice .raw fixtures (the only coverage of the binary-raw dialect
flowing through CSV assembly). The stepped-transient and noise cases are
load-bearing: mocks cannot prove per-step distinct time vectors, and noise is an
advertised analysis type that needs a real recorded result.
"""

import csv
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import services
from ltspice_mcp.lib.raw_parser import get_step_count
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import (
    ExportWaveformInput,
    _build_and_write,
    _complex_columns,
    _window_indices,
    handle_export_waveform,
)
from tests.conftest import make_sim_job, stage_recorded_fixture


def _read_csv(path: Path) -> tuple[list[str], list[list[str]]]:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    return rows[0], rows[1:]


def _setup_symlinked_sidecar(work_dir: Path) -> Path:
    """Stage a raw and make the sidecar dir a symlink pointing outside work_dir."""
    raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
    outside = work_dir.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (work_dir / ".ltspice-mcp").symlink_to(outside)
    return raw


async def _export(state: SessionState, **kwargs) -> dict:
    result = await handle_export_waveform(ExportWaveformInput(**kwargs), state)
    assert result.structuredContent is not None
    return result.structuredContent


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


# --- input model -----------------------------------------------------------


class TestInputModel:
    def test_defaults(self):
        m = ExportWaveformInput()
        assert m.signals == "all"
        assert m.complex_format == "mag_phase"
        assert m.run_index == 0

    def test_explicit_signals_preserved(self):
        m = ExportWaveformInput(signals=["V(out)"])
        assert m.signals == ["V(out)"]

    def test_strict_rejects_unknown_field(self):
        with pytest.raises(ValidationError):
            ExportWaveformInput(bogus=1)  # type: ignore[call-arg]


# --- handler error paths ---------------------------------------------------


@pytest.mark.asyncio
class TestSourceSelection:
    async def test_neither_source_rejected(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_export_waveform(ExportWaveformInput(), state_no_sim)

    async def test_both_sources_rejected(self, state_no_sim: SessionState):
        with pytest.raises(ResultError, match="exactly one"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file="x.raw", job_id="j1"), state_no_sim
            )

    async def test_empty_signals_list_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="at least one signal"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file=str(raw), signals=[]), state_no_sim
            )


# --- handler integration via recorded fixtures -----------------------------


@pytest.mark.asyncio
class TestTransient:
    async def test_single_signal_no_step_columns(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])

        assert data["analysis_type"] == "transient"
        assert data["n_steps"] == 1
        assert data["complex_format"] is None
        out = Path(data["path"])
        assert (work_dir / ".ltspice-mcp" / "waveforms") in out.parents
        header, rows = _read_csv(out)
        assert header == ["time_s", "V(out)"]
        assert "step_index" not in header
        assert len(rows) == data["row_count"]

    async def test_out_dir_override(self, state_no_sim: SessionState, work_dir: Path):
        # An explicit out_dir (under an allowed path) wins over the sidecar.
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        dest = work_dir / "exports"
        data = await _export(
            state_no_sim, raw_file=str(raw), signals=["V(out)"], out_dir=str(dest)
        )
        out = Path(data["path"])
        assert out.parent == dest.resolve()
        assert out.exists()  # noqa: ASYNC240

    async def test_signals_all_excludes_axis(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals="all")
        header, _ = _read_csv(Path(data["path"]))
        assert header[0] == "time_s"
        assert "time" not in header[1:]  # the axis trace is not a value column

    async def test_window_reduces_rows_and_observes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        full = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        win = await _export(
            state_no_sim, raw_file=str(raw), signals=["V(out)"], t_start="0.3m", t_end="0.6m"
        )
        assert win["row_count"] < full["row_count"]
        assert any(o["code"] == "window_applied" for o in win["observations"])
        lo, hi = win["window_used"]
        # searchsorted keeps only samples within [t_start, t_end].
        assert lo >= 0.3e-3
        assert hi <= 0.6e-3 + 1e-12

    async def test_window_outside_axis_rejected(self, state_no_sim: SessionState, work_dir: Path):
        # A window that selects nothing fails with a clear error, not an empty file.
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="no samples"):
            await handle_export_waveform(
                ExportWaveformInput(
                    raw_file=str(raw), signals=["V(out)"], t_start="100", t_end="200"
                ),
                state_no_sim,
            )


@pytest.mark.asyncio
class TestSteppedTransient:
    async def test_tidy_long_concatenates_distinct_per_step_axes(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_step_tran")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])

        assert data["n_steps"] > 1
        header, rows = _read_csv(Path(data["path"]))
        assert header == ["step_index", "step_value", "time_s", "V(out)"]

        # row_count must equal the SUM of per-step axis lengths — the proof that
        # distinct per-step time vectors are concatenated tidy/long, not assumed
        # to share one axis.
        loaded = services.load_raw_sync(raw, state_no_sim)
        expected = sum(
            len(np.asarray(loaded.get_axis(step=s))) for s in range(get_step_count(loaded))
        )
        assert data["row_count"] == expected

        step_indices = {int(r[0]) for r in rows}
        assert step_indices == set(range(data["n_steps"]))
        # step_value carries the .step parameter map recovered from the .log.
        labels = {int(r[0]): r[1] for r in rows}
        assert labels == {0: "r=1", 1: "r=22", 2: "r=680"}
        assert any(o["code"] == "export_written" for o in data["observations"])


@pytest.mark.asyncio
class TestAC:
    async def test_default_mag_phase(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])

        assert data["analysis_type"] == "ac"
        assert data["complex_format"] == "mag_phase"
        header, rows = _read_csv(Path(data["path"]))
        assert header == ["freq_Hz", "V(out)_mag_dB", "V(out)_phase_deg"]
        # freq column is real (the complex axis was real-parted).
        assert all("j" not in r[0] for r in rows)
        mags = [float(r[1]) for r in rows]
        phases = [float(r[2]) for r in rows]
        assert all(-180.0 < p <= 180.0 for p in phases)
        # Concrete check: a single-pole RC low-pass has ~ -45 deg phase at its
        # -3 dB corner (this fixture is 1 kHz by construction).
        corner = min(range(len(mags)), key=lambda i: abs(mags[i] + 3.0103))
        assert -55.0 < phases[corner] < -35.0
        assert any(o["code"] == "complex_format_used" for o in data["observations"])

    @pytest.mark.parametrize(
        ("complex_format", "expected_header"),
        [
            ("re_im", ["freq_Hz", "V(out)_re", "V(out)_im"]),
            (
                "both",
                ["freq_Hz", "V(out)_mag_dB", "V(out)_phase_deg", "V(out)_re", "V(out)_im"],
            ),
        ],
    )
    async def test_complex_format_columns(
        self,
        state_no_sim: SessionState,
        work_dir: Path,
        complex_format: str,
        expected_header: list[str],
    ):
        raw = stage_recorded_fixture(work_dir, "ltspice_ac_rc")
        data = await _export(
            state_no_sim, raw_file=str(raw), signals=["V(out)"], complex_format=complex_format
        )
        header, _ = _read_csv(Path(data["path"]))
        assert header == expected_header


@pytest.mark.asyncio
class TestOtherAnalysisTypes:
    async def test_dc_sweep_accepted(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_dc_div")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        assert data["analysis_type"] == "dc"
        header, rows = _read_csv(Path(data["path"]))
        # The DC x-column names the swept variable (V1) + its unit, not a bare
        # "sweep" — the recorded fixture sweeps source V1 (a voltage).
        assert header == ["V1_V", "V(out)"]
        assert len(rows) == data["row_count"]

    async def test_noise_accepted(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_noise_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals="all")
        assert data["analysis_type"] == "noise"
        header, rows = _read_csv(Path(data["path"]))
        assert header[0] == "freq_Hz"
        assert len(rows) == data["row_count"] > 0

    async def test_op_raw_refused(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "op_extreme_node")
        with pytest.raises(ResultError, match="operating_point"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file=str(raw), signals="all"), state_no_sim
            )


@pytest.mark.asyncio
class TestResponseShape:
    async def test_json_format_passes_schema(self, state_no_sim: SessionState, work_dir: Path):
        # The autouse schema-conformance hook validates structuredContent against
        # the tool's output_schema; this exercises the happy path end-to-end.
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"], format="json")
        assert {"path", "row_count", "columns", "observations"} <= data.keys()


@pytest.mark.asyncio
class TestDestination:
    async def test_csv_lands_next_to_circuit_not_raw_for_job(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A job-run raw can live in a Windows temp the client cannot Read; the CSV
        # must land beside the (Linux-side) circuit, not the raw.
        raw_dir = work_dir / "elsewhere"
        raw_dir.mkdir()
        raw = stage_recorded_fixture(raw_dir, "ltspice_tran_rc")
        netlist = work_dir / "mycircuit.cir"
        job = make_sim_job(
            "jx", status="completed", netlist=netlist, raw_file=raw, simulator="ltspice"
        )
        state_no_sim.add_job(job)

        data = await _export(state_no_sim, job_id="jx", run_index=0, signals=["V(out)"])
        out = Path(data["path"])
        assert (work_dir / ".ltspice-mcp" / "waveforms") in out.parents
        assert raw_dir not in out.parents
        # The job path must also produce real content, not just a file somewhere.
        header, rows = _read_csv(out)
        assert header == ["time_s", "V(out)"]
        assert "step_index" not in header
        assert len(rows) == data["row_count"] > 0

    async def test_in_tree_raw_reuses_existing_sidecar(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        # A raw already inside a .ltspice-mcp/ tree (e.g. a job-run raw passed by
        # path) must not get a second nested sidecar
        # (…/runs/jx/.ltspice-mcp/waveforms/…); the subdir goes into the
        # existing tree.
        raw_dir = work_dir / ".ltspice-mcp" / "runs" / "jx"
        raw_dir.mkdir(parents=True)
        raw = stage_recorded_fixture(raw_dir, "ltspice_tran_rc")

        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)"])
        out = Path(data["path"])
        assert (raw_dir / "waveforms") in out.parents
        assert ".ltspice-mcp" not in out.relative_to(raw_dir).parts
        header, rows = _read_csv(out)
        assert header == ["time_s", "V(out)"]
        assert len(rows) == data["row_count"] > 0

    async def test_symlinked_sidecar_refused(self, state_no_sim: SessionState, work_dir: Path):
        # A symlinked .ltspice-mcp/ must not redirect the export outside the
        # circuit directory (server-artifact paths skip safe_path).
        raw = _setup_symlinked_sidecar(work_dir)
        with pytest.raises(ResultError, match="outside the destination directory"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
            )


@pytest.mark.asyncio
class TestLimits:
    async def test_row_cap_raises_not_truncates(
        self, state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
    ):
        # The safety backstop RAISES with guidance (no silent truncation), and
        # atomic_write discards the temp so no partial file is committed.
        import ltspice_mcp.tools.analysis as analysis_mod

        monkeypatch.setattr(analysis_mod, "_EXPORT_MAX_ROWS", 2)
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="safety cap"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file=str(raw), signals=["V(out)"]), state_no_sim
            )


class TestBuildAndWriteWorker:
    def test_non_finite_kept_and_counted(self, work_dir: Path):
        # Direct worker test: a NaN sample must be KEPT (row count == axis length)
        # and surfaced as a count, never silently dropped.
        axis = np.array([0.0, 1.0, 2.0, 3.0])
        wave = np.array([0.0, np.nan, 2.0, 3.0])

        class _MockRaw:
            def get_axis(self, step: int = 0):
                return axis

            def get_wave(self, name: str, step: int = 0):
                return wave

        out = work_dir / "out.csv"
        facts = _build_and_write(
            _MockRaw(),
            work_dir / "mock.raw",
            ["V(out)"],
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

        class _StepMock:
            def get_axis(self, step: int = 0):
                return axes[step]

            def get_wave(self, name: str, step: int = 0):
                return axes[step] * 10.0

        out = work_dir / "stepped.csv"
        facts = _build_and_write(
            _StepMock(),
            work_dir / "m.raw",
            ["V(o)"],
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

    def test_no_log_blanks_step_value(self, work_dir: Path):
        # n_steps>1 with no sibling .log -> the .step param map is unrecoverable:
        # step_value cells blank, step_values_available False. (A real LTspice
        # stepped .raw needs its .log to parse at all, so this no-log path is
        # exercised at the worker level with a mock raw.)
        axis = np.array([0.0, 1.0, 2.0])

        class _StepMock:
            def get_axis(self, step: int = 0):
                return axis

            def get_wave(self, name: str, step: int = 0):
                return axis * (step + 1.0)

        out = work_dir / "nolog.csv"
        facts = _build_and_write(
            _StepMock(),
            work_dir / "nolog.raw",
            ["V(o)"],
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


@pytest.mark.asyncio
class TestSignalSelection:
    async def test_axis_as_signal_rejected(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        with pytest.raises(ResultError, match="sweep axis"):
            await handle_export_waveform(
                ExportWaveformInput(raw_file=str(raw), signals=["time"]), state_no_sim
            )

    async def test_duplicate_signals_deduped(self, state_no_sim: SessionState, work_dir: Path):
        raw = stage_recorded_fixture(work_dir, "ltspice_tran_rc")
        data = await _export(state_no_sim, raw_file=str(raw), signals=["V(out)", "V(out)"])
        assert data["signals"] == ["V(out)"]
        header, _ = _read_csv(Path(data["path"]))
        assert header == ["time_s", "V(out)"]
