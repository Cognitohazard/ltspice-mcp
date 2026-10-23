"""The server's reading of a run's output, held against what LTspice wrote.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: the raw
and log files LTspice 26 and LTspice XVII each left for the same decks. They
go through the decoder the server runs on a live result, so a header field, a
flag, a table or a message one build writes and the server cannot read fails
here, on a machine with no LTspice.
"""

from __future__ import annotations

import cmath
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import NoAxisError, OtherRawNotWritten, ResultError
from ltspice_mcp.lib import metrics, services
from ltspice_mcp.lib.device_names import (
    owner_path,
    parse_device_param,
    path_text,
    table_device_path,
)
from ltspice_mcp.lib.encoding import decode_spice_bytes_with_encoding, read_spice_text
from ltspice_mcp.lib.experiment_types import PARTIAL_RAW_KEY
from ltspice_mcp.lib.hierarchy import SemanticProfile, load_hierarchy
from ltspice_mcp.lib.lint_rules import lint_deck
from ltspice_mcp.lib.log_parser import (
    classify_failure_code,
    count_op_iterations,
    extract_log_diagnostics,
    parse_measurements,
    parse_step_iterations,
    parse_temperatures,
    read_device_op_points,
)
from ltspice_mcp.lib.metrics import aggregate_log_measurements
from ltspice_mcp.lib.raw_header import RawHeaderError, preflight_raw
from ltspice_mcp.lib.raw_parser import (
    build_simulation_summary,
    has_valid_raw_header,
    raw_writer_command,
    read_partial_raw_progress,
)
from ltspice_mcp.lib.recipes import (
    MeasurementsRecipe,
    StabilityRecipe,
    ValueRecipe,
)
from ltspice_mcp.lib.result_observations import coarse_compressed_waveform, tran_max_step
from ltspice_mcp.lib.spice_validator import validate_directive
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools import analysis, inspect_tools
from ltspice_mcp.tools.analyze import (
    AnalyzeResultsInput,
    handle_analyze_results,
    render_attached_analysis,
)
from ltspice_mcp.tools.experiments import AttachedAnalysis, attached_analysis_callback
from tests import _ltspice_recorded as rec
from tests._asc_ops import inspect_one
from tests._ltspice_recorded import declared_points, header_fields
from tests.conftest import SyncApi, make_experiment_job
from tests.ltspice_recorder import INPUTS, raw_header_text, split_raw
from tests.test_decoded_raw import assert_classified_as_named


def complete_raws(build: str) -> list[str]:
    """Every recorded run that ended by itself and left a raw with samples in it."""
    cases = rec.manifest(build)["cases"]
    return sorted(
        case_id
        for case_id, entry in cases.items()
        if f"{case_id}.raw" in entry["outputs"]
        and entry["outputs"][f"{case_id}.raw"]["bytes"] > 0
        and not entry["stopped"]
        and declared_points(build, case_id) > 0
    )


def declared_names(build: str, case_id: str) -> list[str]:
    text = raw_header_text(rec.recorded(build, f"{case_id}.raw").read_bytes())
    return re.findall(r"(?m)^\t\d+\t([^\t\r\n]+)\t", text)


ALL_COMPLETE = [(build, case_id) for build in rec.BUILDS for case_id in complete_raws(build)]


# --------------------------------------------------------------------------
# Every raw decodes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), ALL_COMPLETE)
def test_every_recorded_raw_decodes_to_the_traces_its_header_declares(
    build: str, case_id: str, tmp_path: Path
):
    """Binary and text, four-byte and eight-byte, real and complex, stepped,
    windowed, uncompressed, converted to Fast Access, with subcircuits; and
    each is classified by its decoded analysis as its plot name is."""
    parsed = rec.decode(build, case_id, tmp_path)
    assert parsed.raw.get_trace_names() == declared_names(build, case_id)
    assert parsed.raw.dialect == "ltspice"
    assert_classified_as_named(parsed.raw)
    if parsed.raw.descriptor.step_status == "unresolved":
        return  # a stepped operating point: see TestSteppedRuns
    points = declared_points(build, case_id)
    for name in declared_names(build, case_id):
        held = sum(len(parsed.raw.get_wave(name, step)) for step in parsed.raw.get_steps())
        assert held == points, name


@pytest.mark.parametrize(("build", "case_id"), ALL_COMPLETE)
def test_every_recorded_raw_is_recognised_as_ltspices(build: str, case_id: str):
    path = rec.recorded(build, f"{case_id}.raw")
    assert has_valid_raw_header(path)
    command = raw_writer_command(path)
    assert command is not None
    assert "LTspice" in command


@pytest.mark.parametrize("build", rec.BUILDS)
class TestRawLayouts:
    """What each analysis stores, read back to the circuit's own numbers."""

    def wave(self, parsed, name: str, step: int = 0) -> np.ndarray:
        lowered = {trace.lower(): trace for trace in parsed.raw.get_trace_names()}
        return np.asarray(parsed.raw.get_wave(lowered[name.lower()], step))

    def test_a_transient_axis_is_time_in_seconds_and_never_negative(
        self, build: str, tmp_path: Path
    ):
        """LTspice marks some stored time points by their sign; XVII's raw for
        this deck holds two such. The decoded axis is plain time."""
        parsed = rec.decode(build, "raw/tran", tmp_path)
        time = np.asarray(parsed.raw.get_axis(0))
        assert time[0] == 0.0
        assert time[-1] == pytest.approx(1e-3)
        assert np.all(np.diff(time) > 0)
        out = self.wave(parsed, "V(out)")
        # RC = 100 us: after ten time constants the output is within 5e-5 of 1.
        assert out[-1] == pytest.approx(1.0, abs=1e-4)
        measured = parsed.logs.value("measurements")["measurements"]["vfinal"]["values"][0]
        assert float(np.interp(0.9e-3, time, out)) == pytest.approx(measured, abs=2e-5)

    def test_xvii_stores_some_time_points_negative(self, build: str):
        _, payload = split_raw(rec.recorded(build, "raw/tran.raw").read_bytes())
        record = np.dtype([("t", "<f8"), ("v", "<f4", 5)])
        stored = np.frombuffer(payload, dtype=record)["t"]
        assert bool((stored < 0).any()) == (rec.generation(build) == "xvii")

    def test_a_windowed_transient_is_offset_to_its_real_start(self, build: str, tmp_path: Path):
        """``.tran 0 2m 1m`` stores time from zero and the start in ``Offset``."""
        assert float(header_fields(build, "raw/tran_window")["Offset"][0]) == pytest.approx(1e-3)
        parsed = rec.decode(build, "raw/tran_window", tmp_path)
        time = np.asarray(parsed.raw.get_axis(0))
        assert (time[0], time[-1]) == pytest.approx((1e-3, 2e-3))
        # The source steps at 1.2 ms, so the output is still zero at the start.
        assert self.wave(parsed, "V(out)")[0] == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("case_id", ["raw/tran_double", "raw/tran_uncompressed"])
    def test_double_precision_and_uncompressed_runs_read_like_any_other(
        self, build: str, case_id: str, tmp_path: Path
    ):
        flags = header_fields(build, case_id)["Flags"][0].split()
        assert ("double" in flags) == (case_id == "raw/tran_double")
        assert ("nocompression" in flags) == (case_id == "raw/tran_uncompressed")
        parsed = rec.decode(build, case_id, tmp_path)
        assert self.wave(parsed, "V(out)")[-1] == pytest.approx(1.0, abs=1e-4)

    def test_an_ac_raw_is_complex_with_a_real_frequency_axis(self, build: str, tmp_path: Path):
        parsed = rec.decode(build, "raw/ac", tmp_path)
        frequency = np.real(np.asarray(parsed.raw.get_axis(0)))
        assert (frequency[0], frequency[-1]) == pytest.approx((10.0, 1e5))
        out = self.wave(parsed, "V(out)")
        at_corner = out[int(np.argmin(np.abs(frequency - 1e3)))]
        # R = 1k, C = 159.15 nF: the corner is 1 kHz, where the gain is 1/sqrt(2) at -45 degrees.
        assert abs(at_corner) == pytest.approx(1 / math.sqrt(2), rel=1e-5)
        assert math.degrees(math.atan2(at_corner.imag, at_corner.real)) == pytest.approx(
            -45, abs=1e-3
        )

    def test_a_dc_sweep_has_the_swept_source_for_its_axis(self, build: str, tmp_path: Path):
        parsed = rec.decode(build, "raw/dc", tmp_path)
        sweep = np.asarray(parsed.raw.get_axis(0))
        assert list(sweep) == pytest.approx([0, 1, 2, 3, 4, 5])
        assert list(self.wave(parsed, "V(out)")) == pytest.approx(list(sweep / 2))

    def test_a_noise_raw_holds_the_resistors_thermal_noise(self, build: str, tmp_path: Path):
        fields = header_fields(build, "raw/noise")
        assert fields["Plotname"] == ["Noise Spectral Density - (V/Hz½ or A/Hz½)"]
        parsed = rec.decode(build, "raw/noise", tmp_path)
        assert parsed.raw.descriptor.analysis == "noise"
        # 4kTR for 1k at 27 degrees is (4.07 nV)^2 per hertz, well below the corner.
        assert self.wave(parsed, "V(onoise)")[0] == pytest.approx(4.071e-9, rel=2e-3)
        assert {name.lower() for name in parsed.raw.get_trace_names()} == {
            "frequency",
            "gain",
            "v(r1)",
            "v(onoise)",
            "v(inoise)",
        }

    def test_an_operating_point_is_one_point_with_no_axis(self, build: str, tmp_path: Path):
        values = rec.operating_point(build, "raw/op", tmp_path)
        assert values["v(in)"] == pytest.approx(5.0)
        assert values["v(out)"] == pytest.approx(4.0)
        assert values["i(r1)"] == pytest.approx(1e-3)

    def test_a_transfer_function_is_one_point_of_gain_and_two_impedances(
        self, build: str, tmp_path: Path
    ):
        """1k over 4k: gain 0.8, 5k seen by the source, 800 ohms at the output."""
        assert header_fields(build, "raw/tf")["Plotname"] == ["Transfer Function"]
        values = sorted(rec.operating_point(build, "raw/tf", tmp_path).values())
        assert values == pytest.approx([0.8, 800.0, 5000.0])

    @pytest.mark.parametrize("case_id", ["raw/op_one_subckt", "raw/op_subckt"])
    def test_a_deck_with_subcircuits_reads_like_any_other(
        self, build: str, case_id: str, tmp_path: Path
    ):
        """LTspice 26 adds a ``Backannotation`` header line for each
        subcircuit instance, so a deck with two has the field twice."""
        instances = 1 if case_id == "raw/op_one_subckt" else 2
        written = header_fields(build, case_id).get("Backannotation", [])
        assert len(written) == (0 if rec.generation(build) == "xvii" else instances)
        values = rec.operating_point(build, case_id, tmp_path)
        assert values["v(in)"] == pytest.approx(4.0)
        assert values["v(a)"] == pytest.approx(2.0 if instances == 1 else 1.6)

    def test_a_fast_access_raw_holds_the_same_samples_trace_by_trace(
        self, build: str, tmp_path: Path
    ):
        assert "fastaccess" in header_fields(build, "raw/tran_fastaccess")["Flags"][0].split()
        converted = rec.decode(build, "raw/tran_fastaccess", tmp_path / "converted")
        original = rec.decode(build, "raw/tran", tmp_path / "original")
        for name in original.raw.get_trace_names():
            assert np.array_equal(self.wave(converted, name), self.wave(original, name)), name

    def test_the_text_raw_of_the_setting_holds_the_samples_of_the_binary_one(
        self, build: str, tmp_path: Path
    ):
        """LTspice 26 has a setting that asks for a text raw; XVII has only the
        switch. Run on the same settings, the text holds the binary's samples."""
        if rec.generation(build) == "xvii":
            pytest.skip("LTspice XVII writes a text raw only for the -ascii switch")
        text = rec.decode(build, "raw/tran_as_text_setting", tmp_path / "text")
        binary = rec.decode(build, "raw/tran", tmp_path / "binary")
        for name in binary.raw.get_trace_names():
            assert self.wave(text, name) == pytest.approx(
                self.wave(binary, name), rel=1e-6, abs=1e-12
            )

    @pytest.mark.parametrize("case_id", ["raw/tran_as_text", "raw/ac_as_text"])
    def test_a_text_raw_is_eight_bit_with_crlf_line_ends(self, build: str, case_id: str):
        """Every other raw has a UTF-16 header. A text raw, from either build,
        is 8-bit text throughout, header included."""
        path = rec.recorded(build, f"{case_id}.raw")
        data = path.read_bytes()
        assert b"\x00" not in data
        assert b"\r\nValues:\r\n" in data
        assert b"\n" not in data.replace(b"\r\n", b"")
        (plot,) = preflight_raw(path, limits=rec.RAW_LIMITS).plots
        assert (plot.storage, plot.encoding, plot.dialect) == ("values", "utf-8", "ltspice")


#: ``raw/step_ac_one_freq``'s steps, and the gain of its first-order low-pass
#: (1 nF) at 100 kHz in each.
ONE_FREQUENCY_STEPS = [{"r": 1000.0}, {"r": 2000.0}, {"r": 4000.0}]
ONE_FREQUENCY_GAINS = [
    1 / (1 + 2j * math.pi * 100e3 * step["r"] * 1e-9) for step in ONE_FREQUENCY_STEPS
]


@pytest.mark.parametrize("build", rec.BUILDS)
class TestSteppedRuns:
    def test_each_step_is_found_and_matched_to_its_log_line(self, build: str, tmp_path: Path):
        for case_id, name, values in (
            ("raw/step_tran", "r", [1000.0, 2000.0, 4000.0]),
            ("raw/step_ac", "r", [1000.0, 2000.0, 4000.0]),
            ("raw/step_dc", "r", [1000.0, 3000.0]),
        ):
            parsed = rec.decode(build, case_id, tmp_path / case_id.replace("/", "-"))
            assert parsed.raw.descriptor.step_status == "matched", case_id
            assert parsed.raw.get_steps() == list(range(len(values))), case_id
            assert parsed.logs.value("steps") == [{name: value} for value in values], case_id
            assert "stepped" in header_fields(build, case_id)["Flags"][0].split()

    def test_a_stepped_temperature_is_read_with_or_without_its_unit(
        self, build: str, tmp_path: Path
    ):
        """LTspice 26 logs ``.step temp=-40°``; XVII logs ``.step temp=-40°C``."""
        text = read_spice_text(rec.recorded(build, "raw/step_temp.log"))
        suffix = "°C" if rec.generation(build) == "xvii" else "°"
        assert f".step temp=-40{suffix}" in text.replace("\r", "").split("\n")
        expected = [{"temp": -40.0}, {"temp": 27.0}, {"temp": 125.0}]
        assert parse_step_iterations(text=text) == expected
        parsed = rec.decode(build, "raw/step_temp", tmp_path)
        assert parsed.raw.descriptor.step_status == "matched"
        assert parsed.logs.value("steps") == expected
        # 1 V across 1k with tc1 = 0.01 per degree from 27.
        for step, temperature in enumerate((-40, 27, 125)):
            current = np.asarray(parsed.raw.get_wave("I(R1)", step))[-1]
            assert current == pytest.approx(1 / (1000 * (1 + 0.01 * (temperature - 27))), rel=1e-5)

    def test_two_stepped_parameters_share_a_line(self, build: str):
        text = read_spice_text(rec.recorded(build, "log/step_two_params.log"))
        assert parse_step_iterations(text=text) == [
            {"r": 1000.0, "c": 1e-7},
            {"r": 2000.0, "c": 1e-7},
            {"r": 1000.0, "c": 2e-7},
            {"r": 2000.0, "c": 2e-7},
        ]

    def test_a_stepped_operating_point_stores_every_step(self, build: str, tmp_path: Path):
        """One point a step, with the stepped parameter as the first variable.

        The log names no step values for it, so the raw's own parameter column
        is the record: every step is read, with its value.
        """
        assert declared_points(build, "raw/step_op") == 3
        assert declared_names(build, "raw/step_op")[0] == "v"
        _, payload = split_raw(rec.recorded(build, "raw/step_op.raw").read_bytes())
        record = np.dtype([("v", "<f8"), ("rest", "<f4", 5)])
        stored = np.frombuffer(payload, dtype=record)
        assert list(stored["v"]) == [1.0, 2.0, 3.0]
        # V(out) is four fifths of the source, the second variable after the parameter.
        assert list(stored["rest"][:, 1]) == pytest.approx([0.8, 1.6, 2.4])
        parsed = rec.decode(build, "raw/step_op", tmp_path)
        assert parsed.raw.descriptor.step_status == "matched"
        assert parsed.raw.steps == [{"v": 1.0}, {"v": 2.0}, {"v": 3.0}]
        assert parsed.raw.get_steps(v=2.0) == [1]
        for step, level in enumerate((0.8, 1.6, 2.4)):
            assert list(parsed.raw.get_wave("V(out)", step)) == pytest.approx([level])
        log = read_spice_text(rec.recorded(build, "raw/step_op.log"))
        assert not [line for line in log.splitlines() if line.startswith(".step")]

    def test_the_api_reads_an_operating_point_as_one_number_a_step(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """``value`` is a one-point trace's number, per step; a trace of more
        points is refused rather than cut to its first."""
        api = SyncApi(state_no_sim)

        def load(case_id: str):
            staged = work_dir / build / f"{Path(case_id).name}.raw"
            staged.parent.mkdir(exist_ok=True)
            staged.write_bytes(rec.recorded(build, f"{case_id}.raw").read_bytes())
            return api.load_raw(staged)

        alone = load("raw/op")
        assert alone.value("V(out)") == pytest.approx(4.0)
        assert isinstance(alone.value("V(out)"), float)
        assert alone.value("V(in,out)") == pytest.approx(1.0)
        stepped = load("raw/step_op")
        assert [stepped.value("V(out)", step=step) for step in range(3)] == pytest.approx(
            [0.8, 1.6, 2.4]
        )
        # A stepped .ac at one frequency holds a complex point a step.
        one_frequency = load("raw/step_ac_one_freq")
        assert one_frequency.steps == ONE_FREQUENCY_STEPS
        values = [one_frequency.value("V(out)", step=step) for step in range(3)]
        assert all(isinstance(value, complex) for value in values)
        assert values == pytest.approx(ONE_FREQUENCY_GAINS, rel=1e-5)
        with pytest.raises(ResultError, match="points"):
            load("raw/tran").value("V(out)")

    def test_a_stepped_ac_at_one_frequency_stores_a_point_a_step_as_an_operating_point_does(
        self, build: str, tmp_path: Path
    ):
        """At one frequency the plot has no frequency at all: it opens with the
        stepped parameter and holds one point a step, and the log names no
        steps. So the steps are the raw's own, as a stepped .op's are, and
        never found by the axis starting over."""
        header = header_fields(build, "raw/step_ac_one_freq")
        assert header["Plotname"] == ["AC Analysis"]
        assert "stepped" in header["Flags"][0].split()
        assert header["No. Points"] == ["3"]
        assert declared_names(build, "raw/step_ac_one_freq")[:2] == ["r", "V(in)"]
        log = read_spice_text(rec.recorded(build, "raw/step_ac_one_freq.log"))
        assert not [line for line in log.splitlines() if line.startswith(".step")]

        parsed = rec.decode(build, "raw/step_ac_one_freq", tmp_path)
        assert parsed.raw.descriptor.step_status == "matched"
        assert parsed.raw.steps == ONE_FREQUENCY_STEPS
        for step, expected in enumerate(ONE_FREQUENCY_GAINS):
            (gain,) = parsed.raw.get_wave("V(out)", step)
            assert gain == pytest.approx(expected, rel=1e-5)

    def test_a_stepped_ac_at_one_frequency_cut_part_way_reads_with_no_axis(
        self, build: str, tmp_path: Path
    ):
        """A run stopped part way is read as a full read reads its plot: the
        stepped parameter the plot opens with is not an axis to report."""
        whole = rec.recorded(build, "raw/step_ac_one_freq.raw").read_bytes()
        cut = tmp_path / "step_ac_one_freq.raw"
        cut.write_bytes(whole[:-1])

        progress = read_partial_raw_progress(cut, "ltspice")

        assert progress is not None
        assert (progress.points, progress.axis, progress.last_axis_value) == (2, None, None)

    async def test_signal_stats_on_a_point_a_step_points_at_the_value_recipe(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """A stepped .ac at one frequency has one point a step, which has no
        statistics to take; the refusal names the recipe that reads it."""
        raw = rec.stage_run(build, "raw/step_ac_one_freq", work_dir)
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [{"key": "s", "metric": "signal_stats", "signal": "V(out)"}],
            }
        )
        data = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert data is not None
        [failure] = data["failures"]
        assert "value recipe" in failure["message"]

    async def test_the_value_recipe_reads_a_step_at_one_frequency_without_at(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """No frequency is stored, so ``value`` reads a step's one point as it
        reads an operating point's, and a read at a frequency says to leave
        ``at`` out rather than calling the run an operating point."""
        raw = rec.stage_run(build, "raw/step_ac_one_freq", work_dir)
        source = services.source_for_raw_path(raw, state_no_sim)
        for step, gain in enumerate(ONE_FREQUENCY_GAINS):
            result = await metrics.value(
                source, ValueRecipe(key="g", metric="value", expr="V(out)"), step, state_no_sim
            )
            assert result["magnitude_linear"] == pytest.approx(abs(gain), rel=1e-5)
            assert result["phase_deg"] == pytest.approx(math.degrees(cmath.phase(gain)), abs=1e-3)
        at_a_frequency = ValueRecipe(key="g", metric="value", expr="V(out)", at="100k")
        with pytest.raises(NoAxisError, match="value recipe and no 'at'"):
            await metrics.value(source, at_a_frequency, 0, state_no_sim)

    def test_a_stepped_operating_points_summary_counts_its_steps(self, build: str, tmp_path: Path):
        """The run's summary no longer says only the first step is read."""
        parsed = rec.decode(build, "raw/step_op", tmp_path)
        summary = build_simulation_summary(parsed.raw, parsed.logs, step=2)
        assert summary["step_count"] == 3
        assert not any("Stepped .op" in warning for warning in summary.get("warnings", []))


#: What each build stored of ``raw/compress_*``, an RC charge over 5 ms: the
#: points, and for a run reported as coarse, its widest gap (from, to) in s.
COMPRESSION: dict[tuple[str, str], tuple[int, tuple[float, float] | None]] = {
    ("ltspice26", "raw/compress_1mv"): (1046, None),
    ("ltspice26", "raw/compress_1mv_off"): (1046, None),
    ("ltspice26", "raw/compress_1mv_maxstep"): (122, None),
    ("ltspice26", "raw/compress_1v"): (1046, None),
    ("ltspice17", "raw/compress_1mv"): (13, (2.7415e-3, 3.5569e-3)),
    ("ltspice17", "raw/compress_1mv_off"): (1078, None),
    ("ltspice17", "raw/compress_1mv_maxstep"): (13, (3.8243e-3, 4.6993e-3)),
    ("ltspice17", "raw/compress_1v"): (24, (3.3177e-3, 3.6400e-3)),
}


@pytest.mark.parametrize(("build", "case_id"), list(COMPRESSION))
def test_xvii_stores_a_compressed_transient_in_a_few_points_and_26_in_all(
    build: str, case_id: str
):
    """With compression on, XVII kept a few points of the charge, at 1 mV and
    1 V and even under a 50 us maximum step; LTspice 26 kept every point. Off,
    both keep every point and say so in the flags."""
    points, _ = COMPRESSION[build, case_id]
    assert declared_points(build, case_id) == points
    flags = header_fields(build, case_id)["Flags"][0].split()
    assert ("nocompression" in flags) == case_id.endswith("_off")


@pytest.mark.parametrize("build", rec.BUILDS)
def test_xvii_measures_a_compressed_run_apart_from_an_uncompressed_one(build: str):
    """XVII's own .meas results move with compression, though less than a
    straight line between the stored points would; 26's do not move."""
    compressed = first_values(build, "raw/compress_1mv")
    uncompressed = first_values(build, "raw/compress_1mv_off")
    # A 1 mV charge with a 1 ms time constant from a 1 us ramp, read at 1 ms.
    assert uncompressed["vat"] == pytest.approx(1e-3 * (1 - math.exp(-0.9995)), rel=1e-5)
    if rec.generation(build) == "xvii":
        assert compressed["vat"] != pytest.approx(uncompressed["vat"], rel=1e-3)
        assert compressed["vavg"] != pytest.approx(uncompressed["vavg"], rel=1e-5)
    else:
        assert compressed == uncompressed


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_gap_no_wider_than_the_decks_maximum_step_is_not_coarse(build: str, tmp_path: Path):
    """XVII's 50 us maximum step still left a gap of most of a millisecond,
    which is reported; the same gap under a deck that asked for steps up to
    1 ms is what the deck asked for."""
    deck = tran_max_step((INPUTS / "raw/compress_1mv_maxstep.cir").read_text(encoding="utf-8"))
    assert deck == pytest.approx(50e-6)
    raw = rec.decode(build, "raw/compress_1mv_maxstep", tmp_path).raw
    found = coarse_compressed_waveform(raw, max_step=deck)
    _, gap = COMPRESSION[build, "raw/compress_1mv_maxstep"]
    if gap is None:
        assert found is None
        return
    assert found is not None
    assert (found["gap_from_s"], found["gap_to_s"]) == pytest.approx(gap, rel=1e-4)
    assert coarse_compressed_waveform(raw, max_step=1e-3) is None


@pytest.mark.parametrize(("build", "case_id"), list(COMPRESSION))
async def test_analyze_results_reports_a_transient_compression_left_coarse(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """A recipe that reads the run's samples says where compression left it
    coarse, once however many recipes read it."""
    points, gap = COMPRESSION[build, case_id]
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [
                {"raw_path": str(rec.stage_run(build, case_id, work_dir)), "label": "dut"}
            ],
            "recipes": [
                {"key": "v", "metric": "value", "expr": "V(out)", "at": "1m"},
                {"key": "s", "metric": "signal_stats", "signal": "V(out)"},
            ],
        }
    )
    data = (await handle_analyze_results(request, state_no_sim)).structured_content
    assert data is not None
    assert data["failures"] == []
    found = [o for o in data["observations"] if o["code"] == "coarse_compressed_waveform"]
    if gap is None:
        assert found == []
        return
    (observation,) = found
    evidence = observation["evidence"]
    assert (evidence["run"], evidence["step"], evidence["points"]) == ("dut", 0, points)
    assert evidence["span_s"] == pytest.approx(5e-3)
    assert (evidence["gap_from_s"], evidence["gap_to_s"]) == pytest.approx(gap, rel=1e-4)
    assert "plotwinsize=0" in observation["detail"]


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_a_measurements_only_call_reports_a_transient_compression_left_coarse(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """No recipe here reads the samples, but XVII computed its .meas results
    from the few points it kept, so the call still says so."""
    _, gap = COMPRESSION[build, "raw/compress_1mv"]
    raw = rec.stage_run(build, "raw/compress_1mv", work_dir)
    request = AnalyzeResultsInput.model_validate(
        {
            "sources": [{"raw_path": str(raw), "label": "dut"}],
            "recipes": [{"key": "m", "metric": "measurements"}],
        }
    )
    data = (await handle_analyze_results(request, state_no_sim)).structured_content
    assert data is not None
    found = [o for o in data["observations"] if o["code"] == "coarse_compressed_waveform"]
    if gap is None:
        assert found == []
        return
    (observation,) = found
    assert observation["evidence"]["gap_from_s"] == pytest.approx(gap[0], rel=1e-4)


@pytest.mark.parametrize("build", rec.BUILDS)
async def test_every_reader_of_a_raw_says_compression_left_it_coarse(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """The Python API, plot_waveform and inspect's results report it as
    analyze_results does."""
    _, gap = COMPRESSION[build, "raw/compress_1mv"]
    raw = rec.stage_run(build, "raw/compress_1mv", work_dir)
    plot = await analysis.handle_plot_waveform(
        analysis.PlotWaveformInput(raw_file=str(raw), signals=["V(out)"], open=False),
        state_no_sim,
    )
    listed = await inspect_one(
        state_no_sim, {"kind": "results", "view": "signals", "path": str(raw)}
    )
    assert plot.structured_content is not None
    readers = {
        "plot_waveform": plot.structured_content["observations"],
        "inspect": listed.get("observations", []),
    }
    for reader, observations in readers.items():
        codes = [o["code"] for o in observations]
        assert ("coarse_compressed_waveform" in codes) == (gap is not None), reader


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_api_says_compression_left_a_raw_coarse(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    _, gap = COMPRESSION[build, "raw/compress_1mv"]
    raw = SyncApi(state_no_sim).load_raw(rec.stage_run(build, "raw/compress_1mv", work_dir))
    codes = [o["code"] for o in raw.observations]
    assert ("coarse_compressed_waveform" in codes) == (gap is not None)


def test_the_apis_log_readers_say_a_stopped_cases_results_end_early(
    state_no_sim: SessionState, work_dir: Path
):
    """A stopped case that kept its raw is read with the partial_raw
    observation; the log readers carry it as load_raw does."""
    raw = rec.stage_run("ltspice26", "raw/compress_1mv", work_dir)
    job = make_experiment_job(state_no_sim, job_id="exp_stopped", raw=raw)
    case = job.cases[0]
    case.status, case.failure_code = "failed", "run_timeout"
    held = {"points": 600, "axis": "time", "last_axis_value": 2.5e-3}
    case.failure_evidence = {PARTIAL_RAW_KEY: held}
    api = SyncApi(state_no_sim)

    measured = api.measurements(job_id=job.job_id)
    ops = api.device_op_points(job_id=job.job_id)
    loaded = api.load_raw(job_id=job.job_id)

    assert measured["measurements"]["vat"] is not None
    for result in (measured, ops, loaded):
        [read] = result.observations
        assert read["code"] == "partial_raw"
        assert read["evidence"]["last_axis_value"] == pytest.approx(2.5e-3)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_run_stopped_part_way_reports_the_records_it_holds(build: str):
    """The header's point count is whatever LTspice last wrote there; the
    samples after it say how far the run got."""
    path = rec.recorded(build, "raw/tran_killed.raw")
    _, payload = split_raw(path.read_bytes())
    held = len(payload) // 12  # an eight-byte time and one four-byte trace
    progress = read_partial_raw_progress(path, "ltspice")
    assert progress is not None
    assert progress.header_complete
    assert progress.points == held
    assert progress.plot == "Transient Analysis"
    assert progress.last_axis_value is not None
    assert progress.last_axis_value > 0
    if declared_points(build, "raw/tran_killed") != held:
        with pytest.raises(RawHeaderError):
            preflight_raw(path, limits=rec.RAW_LIMITS)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_operating_point_raw_beside_a_transient_is_one_point(build: str):
    (plot,) = preflight_raw(
        rec.recorded(build, "raw/tran_op_raw.op.raw"), limits=rec.RAW_LIMITS
    ).plots
    assert (plot.plot_name, plot.point_count, plot.flags) == ("Operating Point", 1, ("real",))


# --------------------------------------------------------------------------
# Logs
# --------------------------------------------------------------------------


def log_cases(build: str) -> list[tuple[str, dict]]:
    return [
        (case_id, entry)
        for case_id, entry in rec.manifest(build)["cases"].items()
        if f"{case_id}.log" in entry["outputs"]
    ]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_log_is_eight_bit_unless_xvii_did_not_finish_the_run(build: str):
    """LTspice 26 writes every log as UTF-8. XVII writes UTF-16 while it runs
    and rewrites the log as 8-bit text when the run completes, so the log of a
    run that failed or was stopped stays UTF-16."""
    seen: dict[str, int] = {}
    for case_id, entry in log_cases(build):
        data = rec.recorded(build, f"{case_id}.log").read_bytes()
        _, encoding = decode_spice_bytes_with_encoding(data)
        seen[encoding] = seen.get(encoding, 0) + 1
        finished = entry["exit_code"] == 0 and not entry["stopped"]
        if rec.generation(build) == "xvii":
            assert (encoding == "utf-16-le") == (not finished), case_id
        else:
            assert encoding == "utf-8", case_id
        assert not data.startswith((b"\xff\xfe", b"\xef\xbb\xbf")), case_id
    if rec.generation(build) == "xvii":
        # A degree sign is the one byte B0, which is not UTF-8.
        assert seen.get("cp1252", 0) >= 1
        assert seen["utf-16-le"] >= 10


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_run_whose_title_holds_a_byte_cp1252_lacks_is_read(build: str, tmp_path: Path):
    """XVII copies a deck's title line into its log as the bytes it was given.
    A title saved in a double-byte code page holds bytes cp1252 gives no
    character (81, 8D, 8F, 90, 9D), and a log holding one was refused as
    undecodable, the run's result with it. LTspice 26 names the deck's path
    there instead."""
    case_id = "deck/bytes_outside_cp1252"
    title = (INPUTS / rec.CASES.case(case_id).source).read_bytes().split(b"\n")[0]
    assert {0x81, 0x8D, 0x8F, 0x90, 0x9D} <= set(title)
    log = rec.recorded(build, f"{case_id}.log")
    assert (title in log.read_bytes()) == (rec.generation(build) == "xvii")
    assert rec.operating_point(build, case_id, tmp_path)["v(b)"] == pytest.approx(0.5)
    assert extract_log_diagnostics(log)["errors"] == []
    assert parse_measurements(log)["measurements"] == {}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_temperature_lines_are_read(build: str):
    text = read_spice_text(rec.recorded(build, "raw/op.log"))
    assert parse_temperatures(text=text) == (27.0, 27.0)
    hot = read_spice_text(rec.recorded(build, "deck/options_temp.log"))
    assert parse_temperatures(text=hot) == (50.0, 27.0)


def first_values(build: str, case_id: str) -> dict[str, float]:
    """The first value of each measurement in a recorded log, by name."""
    data = cast(dict[str, Any], parse_measurements(rec.recorded(build, f"{case_id}.log")))
    return {name: entry["values"][0] for name, entry in data["measurements"].items()}


@pytest.mark.parametrize("build", rec.BUILDS)
class TestMeasurements:
    def measured(self, build: str, case_id: str) -> dict[str, Any]:
        return cast(dict[str, Any], parse_measurements(rec.recorded(build, f"{case_id}.log")))

    def test_every_form_is_read_to_the_circuits_own_numbers(self, build: str):
        """RC = 100 us charging to 1 V over 1 ms, from a 1 us edge."""
        data = self.measured(build, "log/meas_forms")
        assert data["failed_measurements"] == []
        value = {name: entry["values"][0] for name, entry in data["measurements"].items()}
        assert set(value) == {
            "m_max", "m_min", "m_pp", "m_avg", "m_rms", "m_integ", "m_find", "m_when",
            "m_findwhen", "m_trigtarg", "m_param", "m_deriv", "m_nokind", "m_expr",
        }  # fmt: skip
        tau = 100e-6
        assert value["m_max"] == pytest.approx(1 - math.exp(-10), abs=2e-5)
        assert value["m_min"] == 0.0
        assert value["m_pp"] == value["m_max"]
        assert value["m_find"] == pytest.approx(1 - math.exp(-5), abs=2e-4)
        assert value["m_trigtarg"] == pytest.approx(tau * math.log(9), rel=5e-3)
        assert value["m_param"] == pytest.approx(2 * value["m_max"], rel=1e-5)
        assert value["m_expr"] == pytest.approx(2 * value["m_find"] + 1, rel=1e-5)
        assert value["m_nokind"] == 1.0
        entries = data["measurements"]
        assert (entries["m_avg"]["range_from"], entries["m_avg"]["range_to"]) == (0.0, 0.0005)
        assert entries["m_find"]["at"] == 0.0005
        assert (entries["m_trigtarg"]["range_from"], entries["m_trigtarg"]["range_to"]) == (
            pytest.approx(tau * math.log(1 / 0.9), abs=2e-6),
            pytest.approx(tau * math.log(10), abs=2e-6),
        )

    def test_a_crossing_is_reported_as_the_time_it_happened(self, build: str):
        """A WHEN measurement prints the level it looked for and the time it
        found it; the time is the result."""
        data = parse_measurements(rec.recorded(build, "log/meas_forms.log"))
        flat, _, _, _ = aggregate_log_measurements(data, INPUTS / "log/meas_forms.cir")
        assert flat["m_when"][0] == pytest.approx(100e-6 * math.log(2), abs=3e-6)
        assert data["measurements"]["m_when"]["values"] == [0.5]

    def test_a_failed_measurement_is_named_and_the_others_kept(self, build: str):
        data = self.measured(build, "log/meas_failed")
        assert data["failed_measurements"] == ["never", "depends"]
        assert data["measurements"]["never"]["values"] == [None]
        assert data["measurements"]["before"]["values"][0] == pytest.approx(1.0, abs=1e-4)
        assert data["measurements"]["after"]["values"] == [0.0]

    def test_complex_results_are_read_as_magnitudes(self, build: str):
        data = self.measured(build, "log/meas_ac")
        value = {name: entry["values"][0] for name, entry in data["measurements"].items()}
        # The log prints (-3.0103dB,-45°); the corner of the low-pass is 1 kHz.
        assert value["g1k"] == pytest.approx(1 / math.sqrt(2), rel=1e-5)
        assert value["gmax"] == pytest.approx(1.0, abs=1e-4)
        assert data["measurements"]["fc"]["at"] == pytest.approx(1000.0, rel=1e-5)

    def test_a_real_result_keeps_its_sign(self, build: str):
        """A function that returns a real number is printed in the same polar
        form, and a negative one at an angle of 180: ph(V(out)) at the corner
        of the low-pass is (33.0643dB,180°), which is -45."""
        value = first_values(build, "deck/meas_functions")
        assert value["g_ph"] == pytest.approx(-45.0, rel=1e-4)
        assert value["g_re"] == pytest.approx(0.5, rel=1e-5)
        assert value["g_im"] == pytest.approx(-0.5, rel=1e-5)
        assert value["g_mag"] == pytest.approx(1 / math.sqrt(2), rel=1e-5)

    def test_stepped_tables_are_read_per_step(self, build: str):
        data = self.measured(build, "log/meas_step")
        entries = data["measurements"]
        assert data["step_count"] == 3
        assert entries["s_max"]["values"] == pytest.approx([0.99996, 0.99327, 0.39332], abs=2e-5)
        # A column that is the same for every step is reported once.
        assert entries["s_find"]["at"] == 0.0005
        assert entries["s_avg"]["range_to"] == 0.0005
        assert len(entries["s_rise"]["range_from"]) == 3
        assert entries["s_param"]["values"] == pytest.approx(
            [2 * v for v in entries["s_max"]["values"]], rel=1e-5
        )
        # XVII takes few points on the first edge and interpolates between them.
        assert entries["s_rise"]["values"] == pytest.approx(
            [r * 100e-9 * math.log(0.9 / 0.7) for r in (1e3, 2e3, 2e4)], rel=6e-2
        )

    def test_a_step_where_a_measurement_fails(self, build: str):
        """With 20k the output never reaches 0.5 V inside the run. LTspice 26
        prints ``failed`` for that step. XVII prints 0, which the log does not
        tell apart from a crossing at time zero."""
        crossing = self.measured(build, "log/meas_step")["measurements"]["s_when"]["values"]
        assert crossing[:2] == pytest.approx(
            [1e3 * 100e-9 * math.log(2), 2e3 * 100e-9 * math.log(2)], abs=3e-6
        )
        assert crossing[2] == (0.0 if rec.generation(build) == "xvii" else None)

    async def test_a_zero_xvii_may_have_printed_for_a_failure_is_named(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """The measurements recipe names the steps that read exactly 0 when the
        run's own output says XVII wrote it, and stays silent for LTspice 26,
        which prints ``failed``. The raw beside the log is one the same build
        wrote; XVII names itself only in a raw's ``Command:`` line."""
        log = work_dir / "meas_step.log"
        log.write_bytes(rec.recorded(build, "log/meas_step.log").read_bytes())
        raw = log.with_suffix(".raw")
        raw.write_bytes(rec.recorded(build, "deck/after_end.raw").read_bytes())
        result = await metrics.measurements(
            services.source_for_raw_path(raw, state_no_sim),
            MeasurementsRecipe(key="meas", metric="measurements"),
            None,
            state_no_sim,
        )
        if rec.generation(build) == "xvii":
            (observation,) = result["observations"]
            assert observation["code"] == "measurement_zero_or_failed"
            assert observation["evidence"]["zero_steps"] == {"s_when": [2]}
        else:
            assert "observations" not in result
            assert result["stats"]["s_when"]["failure_count"] == 1

    async def test_a_jobs_recorded_build_names_the_zeros_without_a_raw(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """A job's case records the build its run named, so the zeros are named
        from that record when no artifact read here names XVII."""
        if rec.generation(build) != "xvii":
            pytest.skip("only XVII prints 0 for a failed step")
        log = work_dir / "meas_step.log"
        log.write_bytes(rec.recorded(build, "log/meas_step.log").read_bytes())
        source = services.resolve_analysis_source(state_no_sim, log_file=str(log))
        recipe = MeasurementsRecipe(key="meas", metric="measurements")
        unnamed = await metrics.measurements(source, recipe, None, state_no_sim)
        assert "observations" not in unnamed  # an XVII log names no build
        recorded = replace(source, simulator_version=rec.manifest(build)["reported_build"])
        result = await metrics.measurements(recorded, recipe, None, state_no_sim)
        (observation,) = result["observations"]
        assert observation["evidence"]["zero_steps"] == {"s_when": [2]}

    def test_a_directive_that_does_not_parse(self, build: str):
        """LTspice 26 stops before the run and says where; XVII runs, reports
        the directive and takes the measurements around it."""
        log = rec.recorded(build, "log/meas_bad_syntax.log")
        diagnostics = extract_log_diagnostics(log)
        if rec.generation(build) == "xvii":
            assert diagnostics["errors"] == ["Error: FIND can not be evaluated over an interval."]
            assert set(parse_measurements(log)["measurements"]) == {"before", "after"}
        else:
            assert rec.entry(build, "log/meas_bad_syntax")["exit_code"] == 1
            (block,) = diagnostics["errors"]
            assert block.splitlines()[0].endswith('(7): Expected ")" here.')
            (failed,) = diagnostics["meas_errors"]
            assert failed["directive"] == ".meas tran broken FIND V(out AT 0.5m"


#: How long log/meas_trig runs: two whole periods of its 1 kHz sine of amplitude 1.
_TWO_PERIODS = 2e-3
_W = 2 * math.pi * 1e3
#: 2*pi*1k*time read as an angle in degrees, in radians.
_W_AS_DEGREES = _W * math.pi / 180


def _integral_of_sin_times_sin(a: float, b: float, end: float) -> float:
    """The integral of sin(a*t)*sin(b*t) from 0 to ``end``, for a != b."""
    return (math.sin((a - b) * end) / (a - b) - math.sin((a + b) * end) / (a + b)) / 2


def _integral_of_sin_times_cos(a: float, b: float, end: float) -> float:
    """The integral of sin(a*t)*cos(b*t) from 0 to ``end``, for a != b."""
    return ((1 - math.cos((a + b) * end)) / (a + b) + (1 - math.cos((a - b) * end)) / (a - b)) / 2


@pytest.mark.parametrize("build", rec.BUILDS)
class TestMeasurementAngleUnit:
    """The unit of a trig function's angle inside a .meas.

    Read in radians, as a B source reads it, INTEG V(s)*sin(2*pi*1k*time) over
    the deck's two periods is 1 ms and INTEG V(s)*cos(2*pi*1k*time) is 0. On
    both builds' own defaults a .meas reads the angle in degrees; the
    RadianMeasure setting ("Use radian measure in waveform expressions")
    turns it to radians. A B source is in radians either way.
    """

    def test_on_the_defaults_a_measurement_reads_angles_in_degrees(self, build: str):
        value = first_values(build, "log/meas_trig")
        assert value["m_sin_30"] == pytest.approx(0.5)
        assert value["m_cos_pi"] == pytest.approx(math.cos(math.radians(math.pi)), rel=1e-5)
        assert value["m_tan_45"] == pytest.approx(1.0)
        assert value["m_asin"] == pytest.approx(30.0)
        assert value["m_acos"] == pytest.approx(60.0)
        assert value["m_atan"] == pytest.approx(45.0)
        assert value["m_atan2"] == pytest.approx(45.0)
        assert value["m_atan2_deg"] == pytest.approx(45 * 180 / math.pi, rel=1e-5)
        # The integrals the deck asked for are 1 ms and 0. Read in degrees,
        # 2*pi*1k*time is a 17.45 Hz sine, and the results are these instead.
        assert value["m_sin"] == pytest.approx(
            _integral_of_sin_times_sin(_W, _W_AS_DEGREES, _TWO_PERIODS), rel=2e-3
        )
        assert value["m_cos"] == pytest.approx(
            _integral_of_sin_times_cos(_W, _W_AS_DEGREES, _TWO_PERIODS), rel=2e-3
        )
        assert value["m_sin_fo"] == value["m_sin"]

    def test_with_the_setting_a_measurement_reads_angles_in_radians(self, build: str):
        value = first_values(build, "log/meas_trig_radian")
        assert value["m_sin"] == pytest.approx(_TWO_PERIODS / 2, rel=1e-3)
        assert value["m_sin_fo"] == value["m_sin"]
        assert value["m_cos"] == pytest.approx(0.0, abs=1e-8)
        assert value["m_sin_30"] == pytest.approx(math.sin(30), rel=1e-5)
        assert value["m_cos_pi"] == pytest.approx(-1.0)
        assert value["m_tan_45"] == pytest.approx(math.tan(45), rel=1e-5)
        assert value["m_asin"] == pytest.approx(math.asin(0.5), rel=1e-5)
        assert value["m_acos"] == pytest.approx(math.acos(0.5), rel=1e-5)
        assert value["m_atan"] == pytest.approx(math.pi / 4, rel=1e-5)
        assert value["m_atan2"] == pytest.approx(math.pi / 4, rel=1e-5)
        assert value["m_atan2_deg"] == pytest.approx(45.0, rel=1e-5)

    def test_time_pi_and_the_hyperbolic_functions_do_not_depend_on_it(self, build: str):
        for case_id in ("log/meas_trig", "log/meas_trig_radian"):
            value = first_values(build, case_id)
            assert value["m_time"] == pytest.approx(1e-3)
            assert value["m_pi"] == pytest.approx(math.pi, rel=1e-5)
            assert value["m_sinh"] == pytest.approx(math.sinh(1), rel=1e-5)
            assert value["m_tanh"] == pytest.approx(math.tanh(1), rel=1e-5)

    def test_a_b_source_works_in_radians_whatever_the_setting(self, build: str, tmp_path: Path):
        raw = rec.decode(build, "log/meas_trig", tmp_path).raw
        time = np.abs(raw.get_wave("time", 0))
        assert np.allclose(raw.get_wave("V(batan2)", 0), math.pi / 4)
        assert np.allclose(raw.get_wave("V(bcospi)", 0), -1.0)
        area = float(np.trapezoid(raw.get_wave("V(bsin)", 0), time))
        assert area == pytest.approx(_TWO_PERIODS / 2, rel=1e-3)
        # The log's measurements of the B-source nodes take no angle, so the
        # setting leaves them as they were.
        default = first_values(build, "log/meas_trig")
        radian = first_values(build, "log/meas_trig_radian")
        names = {"b_cos", "b_sin", "b_atan2", "b_cos_pi"}
        assert {name: default[name] for name in names} == {name: radian[name] for name in names}
        assert default["b_sin"] == pytest.approx(_TWO_PERIODS / 2, rel=1e-3)
        assert default["b_cos"] == pytest.approx(0.0, abs=1e-8)

    def test_ph_gives_degrees_unless_the_setting_says_radians(self, build: str):
        """At the corner of the RC low-pass the phase is -45 degrees."""
        phase = first_values(build, "log/meas_phase")["g_ph"]
        assert phase == pytest.approx(-45.0, rel=1e-5)
        phase = first_values(build, "log/meas_phase_radian")["g_ph"]
        assert phase == pytest.approx(-math.pi / 4, rel=1e-5)

    def test_the_lint_refuses_the_measurements_the_setting_changes_and_no_other(self, build: str):
        default = first_values(build, "log/meas_trig")
        radian = first_values(build, "log/meas_trig_radian")
        changed = {
            name
            for name in default
            if not math.isclose(default[name], radian[name], rel_tol=1e-6, abs_tol=1e-9)
        }
        deck = INPUTS / "log/meas_trig.cir"
        findings = lint_deck(read_spice_text(deck), deck, "ltspice", "LTspice")
        assert {f["subject"] for f in findings if f["rule_id"] == "meas-trig-degrees"} == changed


@pytest.mark.parametrize("build", rec.BUILDS)
def test_db_of_a_complex_voltage_is_its_complex_logarithm(build: str):
    """At the corner of the RC low-pass the gain is -3.0103 dB. db(V(out)) is
    20*log10 of the complex voltage, printed as (17.4507dB,-113.81°), so it
    reads as that number's magnitude with the gain as its real part.
    db(mag(V(out))) and re(db(V(out))) are the gain. A WHEN on either form
    finds the corner."""
    data = parse_measurements(rec.recorded(build, "log/meas_db.log"))
    value = {name: entry["values"][0] for name, entry in data["measurements"].items()}
    gain_db = 20 * math.log10(1 / math.sqrt(2))
    complex_log = complex(gain_db, 20 * (-math.pi / 4) / math.log(10))
    assert value["g_db"] == pytest.approx(abs(complex_log), rel=1e-4)
    assert value["g_db_mag"] == pytest.approx(gain_db, rel=1e-4)
    assert value["g_re_db"] == pytest.approx(gain_db, rel=1e-4)
    assert value["g_mag"] == pytest.approx(1 / math.sqrt(2), rel=1e-5)
    for name in ("f_db_mag", "f_db"):
        assert data["measurements"][name].get("at") == pytest.approx(1000.0, rel=1e-5)


def _lint_refusals(case_id: str) -> list[tuple[str, str]]:
    """The (rule, subject) of each blocking lint finding on a recorded input deck."""
    deck = INPUTS / f"{case_id}.cir"
    findings = lint_deck(read_spice_text(deck), deck, "ltspice", "LTspice")
    return [(f["rule_id"], f["subject"]) for f in findings if f["severity"] == "error"]


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("name", ["e", "k", "pi", "q"])
def test_a_measurement_named_for_a_constant(build: str, name: str):
    """The constants of the expression engine cannot name a .meas. LTspice 26
    refuses the deck and runs nothing; XVII names the constant and takes the
    other measurement. The directive check and the lint refuse the name
    before either."""
    case_id = f"log/meas_name_{name}"
    card = next(card for card in rec.deck_cards(f"{case_id}.cir") if card.name == name)
    error = validate_directive(card.body, "LTspice")
    assert error is not None
    assert error.rule_name == "meas_reserved_name"
    assert _lint_refusals(case_id) == [("meas-name-ltspice", name)]
    log = rec.recorded(build, f"{case_id}.log")
    diagnostics = extract_log_diagnostics(log)
    if rec.generation(build) == "xvii":
        assert diagnostics["errors"] == [f'Error: "{name}" is a reserved constant name.']
        assert set(parse_measurements(log)["measurements"]) == {"after"}
    else:
        assert rec.entry(build, case_id)["exit_code"] == 1
        (failed,) = diagnostics["meas_errors"]
        assert "Expected a name that is not a reserved name here." in failed["raw_block"]
        # The relayed failure carries the directive check's fix.
        assert failed["suggestion"] == error.suggestion


@pytest.mark.parametrize("build", rec.BUILDS)
def test_names_that_look_reserved_and_are_not(build: str):
    cards = [card for card in rec.deck_cards("log/meas_name_other.cir") if card.kind == "meas"]
    for card in cards:
        assert validate_directive(card.body, "LTspice") is None, card.body
    assert _lint_refusals("log/meas_name_other") == []
    assert set(first_values(build, "log/meas_name_other")) == {card.name for card in cards}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_fourier_block_is_read_with_its_harmonics_and_distortion(build: str, tmp_path: Path):
    """LTspice 26 prints the partial and the total distortion on a line each.
    XVII prints one line, ``Total Harmonic Distortion: 13.60%(13.61%)``, whose
    first figure is read."""
    fourier = rec.decode_log(build, "log/fourier", tmp_path).value("fourier")
    by_signal = {block["signal"].lower(): block for block in fourier}
    assert set(by_signal) == {"v(out)", "v(in)"}
    clipped, clean = by_signal["v(out)"], by_signal["v(in)"]
    assert clipped["fundamental_frequency"] == 1000.0
    assert clipped["thd"] == pytest.approx(13.6, abs=0.05)
    assert clean["thd"] == pytest.approx(0.0, abs=0.1)
    assert [h["number"] for h in clipped["harmonics"]] == list(range(1, 10))
    assert [h["number"] for h in clean["harmonics"]] == list(range(1, 6))
    assert clean["harmonics"][0]["magnitude"] == pytest.approx(1.0, abs=1e-3)
    if rec.generation(build) == "xvii":
        assert clipped["phd"] is None
    else:
        assert clipped["phd"] == pytest.approx(13.61, abs=0.01)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_number_after_a_fourier_frequency_is_the_harmonic_count(build: str, tmp_path: Path):
    """The deck has ``.four 1k V(out)`` and ``.four 1k 5 V(in)`` on a run five
    periods long. The 5 makes V(in)'s table five harmonics where V(out)'s has
    the default nine, and leaves both on one period: read as a period count it
    would have printed ``N-Period=5`` beside nine harmonics. No deck here gives
    a second number, so the period count's place is not recorded."""
    fourier = rec.decode_log(build, "log/fourier", tmp_path).value("fourier")
    counts = {block["signal"].lower(): len(block["harmonics"]) for block in fourier}
    assert counts == {"v(out)": 9, "v(in)": 5}
    # The decoder does not keep the period count, so it is read off the log.
    text = read_spice_text(rec.recorded(build, "log/fourier.log"))
    assert re.findall(r"N-Period=(\d+)", text) == ["1", "1"]


def _fourier_phases(build: str, signal: str, scratch: Path) -> dict[int, tuple[float, float]]:
    """``{harmonic: (magnitude, phase in degrees)}`` of one recorded block."""
    fourier = rec.decode_log(build, "log/fourier", scratch).value("fourier")
    (block,) = [block for block in fourier if block["signal"].lower() == signal]
    return {h["number"]: (h["magnitude"], h["phase"]) for h in block["harmonics"]}


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_fourier_phase_is_read_in_the_builds_own_convention(build: str, tmp_path: Path):
    """``V(in)`` is ``SINE(0 1 1k)``, sin(2*pi*1k*t) at zero phase, and the two
    builds report its fundamental at different phases: XVII at 0 degrees, the
    phase of a sine, and LTspice 26 at 90. The server passes each through as
    printed, so a ``.four`` phase means what the build that wrote it means; the
    ``tone`` recipe states its own reference instead (guide section 'signals')."""
    (_, phase) = _fourier_phases(build, "v(in)", tmp_path)[1]
    expected = 0.0 if rec.generation(build) == "xvii" else 90.0
    assert phase == pytest.approx(expected, abs=0.05)


def test_no_constant_offset_turns_one_builds_fourier_phase_into_the_others(tmp_path: Path):
    """On the clipped sine every harmonic strong enough to read keeps
    ``phase(26) = 90 - phase(XVII)`` (mod 360), while adding 90 degrees to
    XVII's phase leaves every even harmonic half a turn out. So a ``.four``
    phase cannot be carried from one build to the other by an offset. The
    input's symmetry makes a sign flip and a half-period shift of the time
    origin fit equally well, so this pins the relation, not which of the two
    the builds do."""
    builds = {rec.generation(build): build for build in rec.BUILDS}
    if set(builds) != {"current", "xvii"}:
        pytest.skip("needs a recording from LTspice 26 and from XVII")
    current = _fourier_phases(builds["current"], "v(out)", tmp_path / "current")
    xvii = _fourier_phases(builds["xvii"], "v(out)", tmp_path / "xvii")
    strong = [n for n, (magnitude, _) in xvii.items() if magnitude > 5e-3]
    assert {2, 4, 6} <= set(strong)

    def wrapped(degrees: float) -> float:
        return (degrees + 180.0) % 360.0 - 180.0

    for n in strong:
        assert abs(wrapped(current[n][1] - (90.0 - xvii[n][1]))) < 2.0, n
        offset_miss = abs(wrapped(current[n][1] - (xvii[n][1] + 90.0)))
        assert abs(offset_miss - (180.0 if n % 2 == 0 else 0.0)) < 2.0, n


@pytest.mark.parametrize("build", rec.BUILDS)
class TestDeviceOperatingPoints:
    """The block of semiconductor operating points in an ``.op`` log."""

    def test_which_deck_gets_the_block(self, build: str):
        """LTspice 26 prints it only under ``.options logopinfo``. XVII prints
        it for every ``.op`` and does not know the option at all."""

        def has_block(case_id: str) -> bool:
            text = read_spice_text(rec.recorded(build, f"{case_id}.log"))
            return "Semiconductor Device Operating Points:" in text

        if rec.generation(build) == "xvii":
            assert has_block("log/device_op_off")
            assert rec.entry(build, "log/device_op")["exit_code"] == 1
            text = read_spice_text(rec.recorded(build, "log/device_op.log"))
            assert 'unrecognized option: "logopinfo"' in text
        else:
            assert has_block("log/device_op")
            assert not has_block("log/device_op_off")
        # Never for a transient, option or not.
        assert not has_block("log/device_op_tran")

    def test_the_option_is_added_only_for_a_build_that_knows_it(self, build: str, tmp_path: Path):
        """The server adds ``.options logopinfo`` to an ``.op`` deck so that the
        block is printed. On XVII that turns a deck that runs into one that
        does not, and the block is printed without it."""
        from ltspice_mcp.lib.runner_base import inject_logopinfo

        deck = tmp_path / "deck.cir"
        deck.write_bytes((INPUTS / "raw/op.cir").read_bytes())
        run = inject_logopinfo(deck, rec.launcher(build), "job")
        if rec.generation(build) == "xvii":
            assert run == deck
        else:
            assert run != deck
            assert b".options logopinfo\n.end" in run.read_bytes()

    def test_the_block_is_read_for_every_device(self, build: str, tmp_path: Path):
        case_id = _device_op_case(build)
        points = rec.decode_log(build, case_id, tmp_path).value("device_op")
        assert points == read_device_op_points(rec.recorded(build, f"{case_id}.log"))
        assert points["@d1[id]"] == pytest.approx(2.32e-3)
        assert points["@q1[gm]"] == pytest.approx(1.15e-2)
        assert points["@m1[gm]"] == pytest.approx(5.86e-4)
        assert points["@m1[vth]"] == pytest.approx(0.7)
        # The MOSFET inside X1: LTspice 26 names it by its path, XVII by its
        # element letter, X1 without its X and M1 without its letter.
        inner = "@m:1:1[gm]" if rec.generation(build) == "xvii" else "@x1:m1[gm]"
        assert points[inner] == pytest.approx(8.64e-4)

    def test_the_api_reads_the_block_by_device(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """``api.device_op_points`` is the same table, by device and parameter,
        a device inside a subcircuit by its path."""
        case_id = _device_op_case(build)
        log = work_dir / "device_op.log"
        log.write_bytes(rec.recorded(build, f"{case_id}.log").read_bytes())

        points = SyncApi(state_no_sim).device_op_points(log)

        assert points["m1"]["gm"] == pytest.approx(5.86e-4)
        assert points["q1"]["gm"] == pytest.approx(1.15e-2)
        assert points["d1"]["id"] == pytest.approx(2.32e-3)
        assert points["x1.m1"]["gm"] == pytest.approx(8.64e-4)
        # Every value of the table, and nothing else.
        flat = read_device_op_points(log)
        assert sum(len(params) for params in points.values()) == len(flat)

    def test_the_table_prints_three_significant_digits_whatever_numdgt_says(self, build: str):
        for case_id in (_device_op_case(build), _device_op_case(build, numdgt=True)):
            points = read_device_op_points(rec.recorded(build, f"{case_id}.log"))
            assert len(points) > 20, case_id
            assert all(value == float(f"{value:.2e}") for value in points.values()), case_id

    def test_save_does_not_store_a_device_parameter_in_the_raw(self, build: str):
        """LTspice 26 refuses ``@m1[gm]`` in ``.save`` and writes no raw; XVII
        runs and stores only the node it asked for beside them."""
        if rec.generation(build) == "xvii":
            assert declared_names(build, "raw/op_save_device") == ["V(d)"]
        else:
            log = read_spice_text(rec.recorded(build, "raw/op_save_device.log"))
            assert "syntax error" in log
            assert not rec.has(build, "raw/op_save_device.raw")

    async def test_analyze_results_says_the_device_values_have_three_digits(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """Once, for the bias point and for a value read by its device name."""
        raw = rec.stage_run(build, _device_op_case(build, numdgt=True), work_dir)
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [
                    {"key": "op", "metric": "operating_point"},
                    {"key": "gm", "metric": "value", "expr": "m1.gm"},
                ],
            }
        )
        data = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert data is not None
        assert data["failures"] == []
        (gm,) = data["results"]["gm"]["values"]
        assert gm["value"]["value"] == pytest.approx(5.86e-4)
        (observation,) = [
            o for o in data["observations"] if o["code"] == "device_op_log_precision"
        ]
        assert observation["evidence"]["significant_digits"] == 3
        assert observation["evidence"]["runs"] == ["dut"]


def _raw_variables(build: str, case_id: str) -> list[str]:
    (plot,) = preflight_raw(rec.recorded(build, f"{case_id}.raw"), limits=rec.RAW_LIMITS).plots
    return [variable.name for variable in plot.variables]


#: The devices inside the subcircuits of ``*/subckt_names*``: Mdrv and Rd in
#: each stage, Mtail and Rin in the inner cell each stage holds.
_NESTED_MOSFETS = ["xa.mdrv", "xa.xin.mtail", "xb.mdrv", "xb.xin.mtail"]


#: The recorded decks that .save a current and a node inside subcircuits, by
#: whether they spell each instance with its X.
_SAVE_SPELLINGS = {True: "raw/save_subckt_names", False: "raw/save_subckt_names_no_x"}


@pytest.mark.parametrize("build", rec.BUILDS)
class TestSaveInsideSubcircuits:
    """Each build's ``.save`` takes a name inside a subcircuit only in the
    spelling its raw names it: LTspice 26 with each instance's X, XVII
    without. The other spelling stores nothing on LTspice 26 and makes XVII
    save everything but what was asked."""

    def _stored(self, build: str, with_x: bool) -> list[str]:
        """The traces the run stored beside its time axis."""
        return [name for name in declared_names(build, _SAVE_SPELLINGS[with_x]) if name != "time"]

    def test_each_build_stores_what_its_own_spelling_names(self, build: str):
        xvii = rec.generation(build) == "xvii"
        own = self._stored(build, with_x=not xvii)
        expected = (
            ["V(a:in:mid)", "I(a:in:Rin)", "I(a:Rd)"]
            if xvii
            else ["V(xa:xin:mid)", "I(xa:Rd)", "I(xa:xin:Rin)"]
        )
        assert sorted(own) == sorted(expected)
        other = self._stored(build, with_x=xvii)
        if xvii:
            log = read_spice_text(rec.recorded(build, f"{_SAVE_SPELLINGS[True]}.log"))
            assert "Invalid .save command.  Just saving everything..." in log
            assert not set(other) & set(expected)
        else:
            assert other == []

    def test_the_hierarchy_saves_in_the_builds_own_spelling(self, build: str):
        """inspect(kind="hierarchy") names each current and node as the build
        that will run it stores it."""
        xvii = rec.generation(build) == "xvii"
        deck = INPUTS / f"{_SAVE_SPELLINGS[True]}.cir"
        hierarchy = load_hierarchy(str(deck), None, SemanticProfile("ltspice", xvii_names=xvii))
        rows = {"/".join(row.instance).casefold(): row for row in hierarchy.instances}
        stored = {name.casefold() for name in self._stored(build, with_x=not xvii)}
        for path in ("xa/xin/rin", "xa/rd"):
            save = rows[path].save
            assert save is not None and save.startswith(".save ")
            assert save.removeprefix(".save ").casefold() in stored, save
        (mid,) = {
            node.voltage_trace
            for row in hierarchy.instances
            for node in row.nodes
            if node.voltage_trace is not None and node.voltage_trace.casefold().endswith(":mid)")
        }
        assert mid.casefold() in stored


@pytest.mark.parametrize("build", rec.BUILDS)
class TestSubcircuitDeviceNames:
    """A device inside a subcircuit has one name in the device table and
    another in the raw, and each build spells both its own way."""

    def _table_case(self, build: str, *, saved: bool = False) -> str:
        # XVII refuses .options logopinfo and prints the table without it.
        off = "_off" if rec.generation(build) == "xvii" else ""
        return f"log/subckt_names{off}" + ("_saved" if saved else "")

    def test_the_table_names_each_device_by_its_path(self, build: str):
        """LTspice 26 writes the instance path and the device's own name;
        XVII the device's letter, each instance's name without its X, and the
        device's name without its letter."""
        points = read_device_op_points(rec.recorded(build, f"{self._table_case(build)}.log"))
        names = sorted({parsed[0] for key in points if (parsed := parse_device_param(key))})
        if rec.generation(build) == "xvii":
            assert names == ["m:a:drv", "m:a:in:tail", "m:b:drv", "m:b:in:tail"]
        else:
            assert names == ["xa:mdrv", "xa:xin:mtail", "xb:mdrv", "xb:xin:mtail"]
        assert sorted(path_text(table_device_path(n)) for n in names) == _NESTED_MOSFETS

    def test_the_api_keys_each_device_by_its_path_on_either_build(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """The two builds' tables name the same devices differently; the API's
        keys are their paths, which agree."""
        log = work_dir / "subckt_names.log"
        log.write_bytes(rec.recorded(build, f"{self._table_case(build)}.log").read_bytes())

        points = SyncApi(state_no_sim).device_op_points(log)

        assert sorted(points) == _NESTED_MOSFETS
        assert all("gm" in params for params in points.values())

    def test_the_raw_holds_a_subcircuits_insides_only_when_the_setting_asks(self, build: str):
        """By default both builds store a subcircuit instance's port currents
        and nothing from inside it, stepped or not. With the settings that save
        subcircuit node voltages and device currents, the raw names each
        device by its path: LTspice 26 with each instance's X, XVII without."""
        for case_id in ("raw/subckt_names_tran", "raw/subckt_names_step"):
            inside = [n for n in _raw_variables(build, case_id) if ":" in n]
            assert inside and all(n.startswith("Ix(") for n in inside), case_id
        assert _raw_variables(build, "raw/subckt_names_step") == _raw_variables(
            build, "raw/subckt_names_tran"
        )
        saved = _raw_variables(build, "raw/subckt_names_tran_saved")
        if rec.generation(build) == "xvii":
            assert {"Id(a:in:Mtail)", "I(a:Rd)", "V(a:in:mid)", "Ix(a:in:P)"} <= set(saved)
        else:
            assert {"Id(xa:xin:Mtail)", "I(xa:Rd)", "V(xa:xin:mid)", "Ix(xa:xin:p)"} <= set(saved)
        xvii = rec.generation(build) == "xvii"
        devices = {
            path_text(path)
            for name in saved
            if name.startswith("Id(") and (path := owner_path(name, xvii=xvii))
        }
        assert sorted(devices) == _NESTED_MOSFETS
        # The only MOSFETs are inside the subcircuits, and they are found there.
        assert metrics.has_active_device({name: 0.0 for name in saved if name.startswith("I")})
        ports = {
            path_text(path)
            for name in saved
            if name.startswith("Ix(") and (path := owner_path(name, xvii=xvii))
        }
        assert ports == {"xa", "xb", "xa.xin", "xb.xin"}

    async def test_a_trace_inside_a_subcircuit_is_read_by_its_path_on_either_build(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """One request reads a node, a device current and a port current inside
        the subcircuits by their paths, though XVII writes each without the
        instances' X (V(a:in:mid)) where LTspice 26 writes V(xa:xin:mid)."""
        raw = rec.stage_run(build, "raw/subckt_names_tran_saved", work_dir)
        names = {"node": "V(xa.xin.mid)", "drain": "Id(xa.xin.mtail)", "port": "Ix(xb.top)"}
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [
                    {"key": key, "metric": "value", "expr": expr, "at": "1u"}
                    for key, expr in names.items()
                ],
            }
        )
        data = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert data is not None
        assert data["failures"] == []
        read = {key: data["results"][key]["values"][0]["value"] for key in names}
        if rec.generation(build) == "xvii":
            assert [v["signal"] for v in read.values()] == [
                "V(a:in:mid)",
                "Id(a:in:Mtail)",
                "Ix(b:TOP)",
            ]
        else:
            assert [v["signal"] for v in read.values()] == [
                "V(xa:xin:mid)",
                "Id(xa:xin:Mtail)",
                "Ix(xb:top)",
            ]
        # The drain current of the inner MOSFET, as the device table prints it.
        assert read["drain"]["value"] == pytest.approx(8.21e-6, rel=5e-3)

    async def test_a_bias_point_value_inside_a_subcircuit_is_read_by_its_path(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """The value recipe reads an operating point's node by its path through
        the same lookup as any trace."""
        raw = rec.stage_run(build, self._table_case(build, saved=True), work_dir)
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [{"key": "mid", "metric": "value", "expr": "V(xa.xin.mid)"}],
            }
        )
        data = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert data is not None
        assert data["failures"] == []
        (value,) = data["results"]["mid"]["values"]
        expected = "V(a:in:mid)" if rec.generation(build) == "xvii" else "V(xa:xin:mid)"
        assert value["value"]["signal"] == expected

    async def test_the_bias_point_maps_each_devices_names_and_reads_it_by_path(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        raw = rec.stage_run(build, self._table_case(build, saved=True), work_dir)
        request = AnalyzeResultsInput.model_validate(
            {
                "sources": [{"raw_path": str(raw), "label": "dut"}],
                "recipes": [
                    {"key": "all", "metric": "operating_point"},
                    {"key": "tail", "metric": "operating_point", "device": "xa.xin.mtail"},
                    {"key": "gm", "metric": "value", "expr": "xa.xin.mtail.gm"},
                ],
            }
        )
        data = (await handle_analyze_results(request, state_no_sim)).structured_content
        assert data is not None
        assert data["failures"] == []
        (bias,) = data["results"]["all"]["values"]
        names = bias["value"]["device_names"]
        # The resistors beside them have currents in the raw and no table entry.
        resistors = ["xa.rd", "xa.xin.rin", "xb.rd", "xb.xin.rin"]
        assert sorted(names) == sorted([*_NESTED_MOSFETS, *resistors])
        assert all(names[r]["op_table"] is None and names[r]["raw"] for r in resistors)
        if rec.generation(build) == "xvii":
            assert names["xa.xin.mtail"] == {"op_table": "m:a:in:tail", "raw": "a:in:Mtail"}
        else:
            assert names["xa.xin.mtail"] == {"op_table": "xa:xin:mtail", "raw": "xa:xin:Mtail"}
        (tail,) = data["results"]["tail"]["values"]
        value = tail["value"]
        table = names["xa.xin.mtail"]["op_table"]
        assert value["device_op_points"][f"@{table}[gm]"] == pytest.approx(9.06e-5)
        (current,) = [v for k, v in value["currents"].items() if k.startswith("Id(")]
        assert current == pytest.approx(8.21e-6, rel=5e-3)
        assert value["device_names"] == {"xa.xin.mtail": names["xa.xin.mtail"]}
        # The value recipe reads the same parameter by the device's path.
        (gm,) = data["results"]["gm"]["values"]
        assert gm["value"]["value"] == pytest.approx(9.06e-5)
        assert gm["value"]["signal"] == f"@{table}[gm]"


def _device_op_case(build: str, *, numdgt: bool = False) -> str:
    """The recording whose log holds the device table, with ``.options
    numdgt=7`` or without: XVII prints it without ``.options logopinfo`` and
    refuses the option."""
    variant = "_numdgt" if numdgt else ""
    return f"log/device_op{variant}" + ("_off" if rec.generation(build) == "xvii" else "")


@pytest.mark.parametrize("build", rec.BUILDS)
class TestOperatingPointSolves:
    def test_one_solve_is_counted_per_operating_point(self, build: str):
        """XVII says "Direct Newton iteration for .op point succeeded.", 26
        "Direct Newton iteration succeeded in finding operating point." A
        stepped operating point prints it once a step on 26 and once in all
        on XVII."""
        assert count_op_iterations(rec.recorded(build, "raw/op.log")) == 1
        stepped = 1 if rec.generation(build) == "xvii" else 3
        assert count_op_iterations(rec.recorded(build, "raw/step_op.log")) == stepped

    def test_a_point_found_by_stepping_is_not_an_error(self, build: str):
        log = rec.recorded(build, "log/op_stepping.log")
        text = read_spice_text(log)
        assert "Gmin stepping succeeded in finding the operating point." in text
        assert rec.entry(build, "log/op_stepping")["exit_code"] == 0
        errors = extract_log_diagnostics(log)["errors"]
        assert not [line for line in errors if "stepping" in line.lower()]

    def test_a_time_step_that_collapses_is_a_convergence_failure(self, build: str):
        errors = extract_log_diagnostics(rec.recorded(build, "log/err_timestep.log"))["errors"]
        assert classify_failure_code(errors)[0] == "convergence_failed"
        assert any(
            "time step too small" in line.lower() or "iteration limit" in line.lower()
            for line in errors
        )


#: What each failed deck is classified as, and the name the failure is about.
FAILURES = {
    "log/err_missing_include": ("missing_include", {"missing_includes": ["nosuchfile.lib"]}),
    "log/err_missing_lib": ("missing_include", {"missing_includes": ["nosuchfile.lib"]}),
    "log/err_unknown_model": ("missing_model", {"missing_refs": ["nosuchmodel"]}),
    "log/err_unknown_subckt": ("missing_model", {"missing_refs": ["nosuchsub"]}),
    "log/err_source_loop": ("singular_matrix", None),
    "log/err_inductor_loop": ("singular_matrix", None),
    "log/err_timestep": ("convergence_failed", None),
}


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(list(FAILURES))))
def test_a_failed_run_is_classified_by_its_cause(build: str, case_id: str):
    """The two builds word every one of these differently."""
    assert rec.entry(build, case_id)["exit_code"] == 1
    errors = extract_log_diagnostics(rec.recorded(build, f"{case_id}.log"))["errors"]
    assert errors
    assert classify_failure_code(errors) == FAILURES[case_id]


#: Runs both builds refused before they began: the log has no "Circuit:"
#: line, and gives the reason as a parse or fatal error.
REFUSED_BEFORE_THE_RUN = ["log/err_missing_include", "log/err_missing_lib", "deck/lib_section"]


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(REFUSED_BEFORE_THE_RUN)))
def test_a_run_refused_before_it_began_reports_why_not_a_parse_failure(
    build: str, case_id: str, tmp_path: Path
):
    """Such a log holds no measurement or Fourier block, so both are absent
    and the diagnostics carry the reason. spicelib's complaint that the log
    lacks its header is not an answer to give the caller."""
    assert "Circuit:" not in read_spice_text(rec.recorded(build, f"{case_id}.log"))
    logs = rec.decode_log(build, case_id, tmp_path)
    for name in ("measurements", "fourier"):
        assert logs.section(name)["status"] == "absent", name
    assert logs.value("diagnostics")["errors"]


#: Decks LTspice refuses whose log gives the reason on a line of its own, with
#: no "Error" in front on LTspice 26 and "Fatal Error:" in front on XVII.
REASON_ON_A_BARE_LINE = {
    "log/err_no_analysis": "No analysis specified.",
    "deck/ac_and_tran": "More than one analysis specified.",
    "log/err_zero_resistance": "R1: Resistance must not be zero.",
}


@pytest.mark.parametrize(("build", "case_id"), list(rec.per_build(list(REASON_ON_A_BARE_LINE))))
def test_a_refusal_ltspice_26_states_on_a_bare_line(build: str, case_id: str):
    """Each build's line is extracted as the run's error, so the caller gets
    the reason rather than only the log excerpt that holds it."""
    errors = extract_log_diagnostics(rec.recorded(build, f"{case_id}.log"))["errors"]
    (error,) = errors
    if rec.generation(build) == "xvii":
        assert error.startswith("Fatal Error:")
    else:
        assert error == REASON_ON_A_BARE_LINE[case_id]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_the_analyses_ltspice_does_not_have_are_refused(build: str):
    """``.sens``, ``.pz`` and ``.disto`` exist in ngspice, whose printed tables
    the server reads. LTspice runs none of them."""
    for case_id in ("log/sens", "log/pz", "log/disto"):
        assert rec.entry(build, case_id)["exit_code"] == 1
        assert extract_log_diagnostics(rec.recorded(build, f"{case_id}.log"))["errors"], case_id


# --------------------------------------------------------------------------
# Frequency response analysis
# --------------------------------------------------------------------------

#: The result LTspice 26 prints at the end of a frequency response analysis's log.
_FRA_RESULT = re.compile(r"F\(0dB\) = ([0-9.]+) kHz, Phase Margin = ([0-9.]+) degrees")


def _fra_results(work_dir: Path, state: SessionState) -> services.AnalysisSource:
    raw = work_dir / "fra.fra_1.raw"
    raw.write_bytes(rec.recorded("ltspice26", "raw/fra.fra_1.raw").read_bytes())
    return services.source_for_raw_path(raw, state)


async def test_an_fra_results_file_reads_as_complex_values_over_frequency(
    state_no_sim: SessionState, work_dir: Path
):
    """LTspice 26 writes a frequency response analysis's results beside the
    transient raw, in the AC layout; the server refused to read it."""
    source = _fra_results(work_dir, state_no_sim)
    raw = await services.load_raw(source, state_no_sim)
    assert raw.descriptor.analysis == "fra"
    assert raw.descriptor.axis is not None and raw.descriptor.axis.quantity == "frequency"
    assert raw.get_trace_names() == ["frequency", "gain_1", "V(stimulus_1)"]
    assert np.iscomplexobj(raw.get_wave("gain_1"))
    # The four frequencies the deck asks for, an octave apart, ascending.
    assert list(np.real(raw.get_axis())) == [4e3, 8e3, 16e3, 32e3]


async def test_the_stability_recipe_on_the_negated_fra_gain_gives_ltspices_margin(
    state_no_sim: SessionState, work_dir: Path
):
    """LTspice prints the crossover and phase margin it reads off ``gain_1``.
    ``gain_1`` is the negative of the loop gain as the stability recipe takes it
    (its phase at crossover is LTspice's phase margin), so the recipe on
    ``-gain_1`` says what LTspice said."""
    said = _FRA_RESULT.search(read_spice_text(rec.recorded("ltspice26", "raw/fra.log")))
    assert said is not None
    crossover_khz, margin_deg = (float(value) for value in said.groups())
    source = _fra_results(work_dir, state_no_sim)
    result = await metrics.stability(
        source, StabilityRecipe(key="loop", metric="stability", signal="-gain_1"), 0, state_no_sim
    )
    (margin,) = result["phase_margins"]
    assert margin["frequency_hz"] == pytest.approx(crossover_khz * 1e3, rel=0.01)
    assert margin["margin_deg"] == pytest.approx(margin_deg, abs=0.5)


def _fra_job(state: SessionState, work_dir: Path, *, other_raws: list[str]):
    """A job whose one case ran the recorded frequency response analysis."""
    for name in ("fra.raw", "fra.log", "fra.fra_1.raw"):
        (work_dir / name).write_bytes(rec.recorded("ltspice26", f"raw/{name}").read_bytes())
    job = make_experiment_job(state, job_id="exp_fra", raw=work_dir / "fra.raw")
    job.cases[0].other_raws = list(other_raws)
    return job


async def _analyze_fra(state: SessionState, job_id: str) -> dict[str, Any]:
    result = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"job_id": job_id, "other_raw": "fra_1", "label": "fra"}],
                "recipes": [{"key": "loop", "metric": "stability", "signal": "-gain_1"}],
            }
        ),
        state,
    )
    assert result.structured_content is not None
    return result.structured_content


async def test_a_jobs_fra_results_are_read_by_job_and_case(
    state_no_sim: SessionState, work_dir: Path
):
    """A frequency response analysis writes its results beside the run's raw,
    and the run row lists them; a job source reads them with other_raw instead
    of the transient, as a raw_path source of the file would."""
    said = _FRA_RESULT.search(read_spice_text(rec.recorded("ltspice26", "raw/fra.log")))
    assert said is not None
    crossover_khz, margin_deg = (float(value) for value in said.groups())
    job = _fra_job(state_no_sim, work_dir, other_raws=["fra_1"])

    data = await _analyze_fra(state_no_sim, job.job_id)

    assert data["coverage"]["runs_analyzed"] == 1
    [row] = data["results"]["loop"]["values"]
    assert row["case_id"] == job.cases[0].case_id
    assert row["value"]["unity_gain_hz"] == pytest.approx(crossover_khz * 1e3, rel=0.01)
    assert row["value"]["phase_margin_worst_deg"] == pytest.approx(margin_deg, abs=0.5)


async def test_a_case_that_wrote_no_such_raw_is_missing_not_misread(
    state_no_sim: SessionState, work_dir: Path
):
    job = _fra_job(state_no_sim, work_dir, other_raws=[])

    data = await _analyze_fra(state_no_sim, job.job_id)

    assert data["coverage"]["runs_analyzed"] == 0
    [missing] = data["coverage"]["missing_cases"]["items"]
    assert missing["code"] == "other_raw_not_written"
    assert missing["case_id"] == job.cases[0].case_id


_INSPECT_FRA = {"kind": "results", "view": "signals", "other_raw": "fra_1"}


async def test_plot_and_inspect_read_a_jobs_fra_results_by_job_and_case(
    state_no_sim: SessionState, work_dir: Path
):
    """plot_waveform and inspect's results read a case's other raw by job and
    case, as analyze_results does."""
    job = _fra_job(state_no_sim, work_dir, other_raws=["fra_1"])

    plot = await analysis.handle_plot_waveform(
        analysis.PlotWaveformInput(
            job_id=job.job_id, other_raw="fra_1", signals=["gain_1"], open=False
        ),
        state_no_sim,
    )
    listed = await inspect_one(state_no_sim, {**_INSPECT_FRA, "job_id": job.job_id})

    assert plot.structured_content is not None
    assert plot.structured_content["analysis_type"] == "fra"
    assert plot.structured_content["signals"] == ["gain_1"]
    assert [row["name"] for row in listed["signals"]] == ["frequency", "gain_1", "V(stimulus_1)"]


async def test_plot_and_inspect_refuse_an_other_raw_the_run_did_not_write(
    state_no_sim: SessionState, work_dir: Path
):
    job = _fra_job(state_no_sim, work_dir, other_raws=["op"])

    with pytest.raises(OtherRawNotWritten, match=r"wrote no fra_1 raw; .* lists \['op'\]"):
        await analysis.handle_plot_waveform(
            analysis.PlotWaveformInput(job_id=job.job_id, other_raw="fra_1", open=False),
            state_no_sim,
        )
    reply = await inspect_tools.handle_inspect(
        inspect_tools.InspectInput.model_validate(
            {"queries": [{**_INSPECT_FRA, "job_id": job.job_id}]}
        ),
        state_no_sim,
    )

    assert reply.structured_content is not None
    [listed] = reply.structured_content["results"]
    assert not listed["ok"]
    assert listed["error"]["code"] == "other_raw_not_written"
    assert "wrote no fra_1 raw" in listed["error"]["message"]


async def test_an_attached_analysis_reads_a_jobs_fra_results(
    state_no_sim: SessionState, work_dir: Path
):
    """run_experiments' attached analyze block takes other_raw too, so a job
    measures its FRA results as its cases finish."""
    said = _FRA_RESULT.search(read_spice_text(rec.recorded("ltspice26", "raw/fra.log")))
    assert said is not None
    crossover_khz = float(said.group(1))
    job = _fra_job(state_no_sim, work_dir, other_raws=["fra_1"])
    job.analysis.request = AttachedAnalysis.model_validate(
        {
            "other_raw": "fra_1",
            "recipes": [{"key": "loop", "metric": "stability", "signal": "-gain_1"}],
        }
    ).model_dump(mode="json", exclude_none=True)

    stored = await attached_analysis_callback(state_no_sim)(job)

    [row] = render_attached_analysis(stored, fields=None)["results"]["loop"]["values"]
    assert row["value"]["unity_gain_hz"] == pytest.approx(crossover_khz * 1e3, rel=0.01)


def test_the_api_reads_a_jobs_fra_results_by_job_and_case(
    state_no_sim: SessionState, work_dir: Path
):
    job = _fra_job(state_no_sim, work_dir, other_raws=["fra_1"])
    api = SyncApi(state_no_sim)

    raw = api.load_raw(job_id=job.job_id, other_raw="fra_1")

    assert raw.analysis_type == "fra"
    assert raw.source == work_dir / "fra.fra_1.raw"
    assert list(np.real(raw.axis())) == [4e3, 8e3, 16e3, 32e3]
    with pytest.raises(OtherRawNotWritten):
        api.load_raw(job_id=job.job_id, other_raw="op")


async def test_other_raw_reads_only_a_job(state_no_sim: SessionState, work_dir: Path):
    """An other raw is a case's: a path names its file instead."""
    raw = work_dir / "fra.fra_1.raw"
    with pytest.raises(TypeError, match="case_id and other_raw are only valid with job_id"):
        SyncApi(state_no_sim).load_raw(raw, other_raw="fra_1")
    with pytest.raises(ResultError, match="case_id and other_raw select a run_experiments case"):
        await analysis.handle_plot_waveform(
            analysis.PlotWaveformInput(raw_file=str(raw), other_raw="fra_1", open=False),
            state_no_sim,
        )
    with pytest.raises(ValidationError, match="other_raw require job_id"):
        inspect_tools.ResultsQuery.model_validate(
            {"kind": "results", "path": str(raw), "other_raw": "fra_1"}
        )
    with pytest.raises(ValidationError, match="log views do not select"):
        inspect_tools.ResultsQuery.model_validate(
            {"kind": "results", "view": "measurements", "job_id": "exp_fra", "other_raw": "op"}
        )


def test_ltspice_xvii_refuses_a_frequency_response_analysis():
    """The analysis came with LTspice 17.1: XVII 17.0 names the analyzer as a
    device it does not know, writes no results, and the server reads that as
    the run's error."""
    log = rec.recorded("ltspice17", "raw/fra.log")
    errors = extract_log_diagnostics(log)["errors"]
    assert any('Unknown SPICE device type "@"' in error for error in errors)
    assert not rec.recorded("ltspice17", "raw/fra.fra_1.raw").exists()


class TestCompareRuns:
    """``RawResult.compare``: how far one run's traces are from another's,
    signal by signal, the tolerance left to the caller."""

    def _load(self, state: SessionState, build: str, case_id: str, directory: Path):
        return SyncApi(state).load_raw(raw_path=rec.stage_run(build, case_id, directory / build))

    @pytest.mark.parametrize("build", rec.BUILDS)
    def test_a_compressed_run_is_read_along_its_lines(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        """XVII keeps 13 points of the 1 mV charge, LTspice 26 every one.
        Compared at the uncompressed run's points, the compressed run is read
        along the lines between its own, which cut the curve by 1.5% of its
        peak on XVII; LTspice 26's two runs are the same."""
        full = self._load(state_no_sim, build, "raw/compress_1mv_off", work_dir)
        compressed = self._load(state_no_sim, build, "raw/compress_1mv", work_dir)

        found = full.compare(compressed, signals=["v(OUT)"])

        assert list(found["signals"]) == ["V(out)"]
        assert found["span"] == [0.0, pytest.approx(5e-3)]
        assert found["points"] == full.axis().size
        row = found["signals"]["V(out)"]
        if rec.generation(build) == "xvii":
            assert row["max_rel"] == pytest.approx(0.015, abs=0.002)
            assert 0 < row["at"] < 1e-3
            assert row["value"] - row["other"] == pytest.approx(row["max_abs"])
        else:
            assert row["max_abs"] == 0.0

    @pytest.mark.parametrize("case_id", ["raw/op", "raw/ac", "raw/tran"])
    def test_the_two_builds_runs_of_one_deck_compare_signal_by_signal(
        self, case_id: str, state_no_sim: SessionState, work_dir: Path
    ):
        """The two builds name every signal alike here, and agree to the last
        bit on the operating point, to rounding on the complex AC response,
        and on the transient where XVII's compressed run stored its points."""
        current, xvii = (
            self._load(state_no_sim, build, case_id, work_dir)
            for build in ("ltspice26", "ltspice17")
        )

        found = current.compare(xvii)

        assert (found["only_here"], found["only_there"]) == ([], [])
        assert {"V(in)", "V(out)", "I(R1)", "I(V1)"} <= set(found["signals"])
        assert {"time", "frequency"}.isdisjoint(found["signals"])
        rows = found["signals"].values()
        if case_id == "raw/op":
            assert found["span"] is None and found["points"] == 1
            assert all(row["max_abs"] == 0.0 and row["at"] is None for row in rows)
        elif case_id == "raw/ac":
            assert isinstance(found["signals"]["V(out)"]["value"], complex)
            assert all(row["max_rel"] < 1e-12 for row in rows if row["max_rel"] is not None)
        else:
            assert found["signals"]["V(out)"]["max_rel"] > 1e-3

    def test_signals_inside_subcircuits_pair_across_the_builds(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """LTspice 26 writes V(xa:xin:mid) where XVII writes V(a:in:mid): the
        same node, compared as one. A signal asked for by its path is found in
        each as trace() finds it. Currents only one build stored are listed as
        such."""
        current, xvii = (
            self._load(state_no_sim, build, "raw/subckt_names_tran_saved", work_dir)
            for build in ("ltspice26", "ltspice17")
        )

        found = current.compare(xvii)
        asked = current.compare(xvii, signals=["V(xa.xin.mid)", "Ix(xb.top)"])

        for name in ("V(xa:xin:mid)", "Id(xa:xin:Mtail)", "I(xb:xin:Rin)", "Ix(xa:top)"):
            assert found["signals"][name]["max_rel"] < 1e-3, name
        assert "Is(xa:xin:Mtail)" in found["only_here"]
        assert found["only_there"] == []
        assert list(asked["signals"]) == ["V(xa:xin:mid)", "Ix(xb:top)"]

    def test_a_stepped_run_is_compared_step_for_step(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        """Steps pair by their parameters; one that names none is refused
        until the step to compare with is given."""
        build = rec.BUILDS[0]
        stepped = self._load(state_no_sim, build, "raw/step_tran", work_dir)
        plain = self._load(state_no_sim, build, "raw/tran", work_dir)
        last = stepped.step_count - 1

        same = stepped.compare(stepped, step=last)
        across = stepped.compare(stepped, step=last, other_step=0)

        assert all(row["max_abs"] == 0.0 for row in same["signals"].values())
        assert across["signals"]["V(out)"]["max_abs"] > 0
        with pytest.raises(ResultError, match="pass other_step"):
            plain.compare(stepped)
        assert plain.compare(stepped, other_step=0)["signals"]
        with pytest.raises(ResultError, match="Not a signal of both results: V\\(nowhere\\)"):
            plain.compare(stepped, other_step=0, signals=["V(nowhere)"])
