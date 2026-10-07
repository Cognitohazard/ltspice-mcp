"""The server's reading of a run's output, held against what LTspice wrote.

Every expectation here comes from ``tests/fixtures/ltspice_recorded``: the raw
and log files LTspice 26 and LTspice XVII each left for the same decks. They
go through the decoder the server runs on a live result, so a header field, a
flag, a table or a message one build writes and the server cannot read fails
here, on a machine with no LTspice.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from ltspice_mcp.lib import metrics, services
from ltspice_mcp.lib.encoding import decode_spice_bytes_with_encoding, read_spice_text
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
from ltspice_mcp.lib.recipes import MeasurementsRecipe
from ltspice_mcp.lib.spice_validator import validate_directive
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS, raw_header_text, split_raw


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


def header_fields(build: str, case_id: str) -> dict[str, list[str]]:
    """A raw's header as the build wrote it: field name to every value it has."""
    fields: dict[str, list[str]] = {}
    for line in raw_header_text(rec.recorded(build, f"{case_id}.raw").read_bytes()).splitlines():
        name, separator, value = line.partition(":")
        if separator and not line.startswith("\t"):
            fields.setdefault(name, []).append(value.strip())
    return fields


def declared_names(build: str, case_id: str) -> list[str]:
    text = raw_header_text(rec.recorded(build, f"{case_id}.raw").read_bytes())
    return re.findall(r"(?m)^\t\d+\t([^\t\r\n]+)\t", text)


def declared_points(build: str, case_id: str) -> int:
    return int(header_fields(build, case_id)["No. Points"][0])


ALL_COMPLETE = [(build, case_id) for build in rec.BUILDS for case_id in complete_raws(build)]


# --------------------------------------------------------------------------
# Every raw decodes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("build", "case_id"), ALL_COMPLETE)
def test_every_recorded_raw_decodes_to_the_traces_its_header_declares(
    build: str, case_id: str, tmp_path: Path
):
    """Binary and text, four-byte and eight-byte, real and complex, stepped,
    windowed, uncompressed, converted to Fast Access, with subcircuits."""
    parsed = rec.decode(build, case_id, tmp_path)
    assert parsed.raw.get_trace_names() == declared_names(build, case_id)
    assert parsed.raw.dialect == "ltspice"
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

    def test_a_stepped_operating_points_summary_counts_its_steps(self, build: str, tmp_path: Path):
        """The run's summary no longer says only the first step is read."""
        parsed = rec.decode(build, "raw/step_op", tmp_path)
        summary = build_simulation_summary(parsed.raw, parsed.logs, step=2)
        assert summary["step_count"] == 3
        assert not any("Stepped .op" in warning for warning in summary.get("warnings", []))


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
        from spicelib.simulators.ltspice_simulator import LTspice

        from ltspice_mcp.lib.runner_base import inject_logopinfo

        executable = rec.manifest(build)["executable"]["name"]
        launcher = type("Launcher", (LTspice,), {"spice_exe": [f"C:/install/{executable}"]})
        deck = tmp_path / "deck.cir"
        deck.write_bytes((INPUTS / "raw/op.cir").read_bytes())
        run = inject_logopinfo(deck, launcher, "job")
        if rec.generation(build) == "xvii":
            assert run == deck
        else:
            assert run != deck
            assert b".options logopinfo\n.end" in run.read_bytes()

    def test_the_block_is_read_for_every_device(self, build: str, tmp_path: Path):
        case_id = "log/device_op_off" if rec.generation(build) == "xvii" else "log/device_op"
        points = rec.decode_log(build, case_id, tmp_path).value("device_op")
        assert points == read_device_op_points(rec.recorded(build, f"{case_id}.log"))
        assert points["@d1[id]"] == pytest.approx(2.32e-3)
        assert points["@q1[gm]"] == pytest.approx(1.15e-2)
        assert points["@m1[gm]"] == pytest.approx(5.86e-4)
        assert points["@m1[vth]"] == pytest.approx(0.7)
        # The MOSFET inside X1: LTspice 26 names it by its path, XVII by its
        # element letter, the subcircuit's number and its own.
        inner = "@m:1:1[gm]" if rec.generation(build) == "xvii" else "@x1:m1[gm]"
        assert points[inner] == pytest.approx(8.64e-4)


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
