"""The server's reading of a results file, held to LTspice's own.

LTspice 26.1 can be asked, through the bridge it ships, to read a results
file: its samples, how a stepped run divides, and each step's parameter
values. That is LTspice reading its own format, where the server's reader is
one this project keeps up itself. ``tests/ltspice_bridge_recorder.py`` hands
LTspice's reader every results file in the main recordings' ``raw`` group,
those LTspice XVII wrote as well as its own, and keeps what it says
(``tests/fixtures/ltspice_bridge_recorded/<build>/reader``). These hold the
decoder the server runs to that, sample for sample, on a machine with no
LTspice.

The two agree to the last bit of every sample. Where they do not, it is in how
a file is presented and not in what it holds, and each such place is pinned
here with what LTspice's reader says:

- a time point LTspice stored with its sign set comes back negative from
  LTspice's reader, and as plain time from the server's;
- a transient saved from a later start comes back starting at zero from
  LTspice's reader, and at its real start from the server's;
- a file of one point (an operating point, a transfer function, a noise
  analysis at one frequency) is refused by LTspice's reader, and read by the
  server's;
- a file that holds only its axis, a run whose ``.save`` named nothing it
  stored, is refused by LTspice's reader, which reads signals by name, and
  read by the server's;
- a stepped operating point (over a parameter or over temperature), a stepped
  transfer function, and a stepped AC or noise analysis at one frequency, is
  one sweep to LTspice's reader, along the stepped parameter, and a point a
  step to the server's;
- a run stopped part way is read as far as its header says by LTspice's
  reader, and refused by the server's decoder when the header and the samples
  disagree.

One kind of file both refuse: the ``.op.raw`` of a stepped run as LTspice 26
writes it, one point with nothing that says which step's it is.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ltspice_mcp.errors import NoAxisError, ResultError
from ltspice_mcp.lib.raw_header import RawHeaderError
from ltspice_mcp.lib.raw_parser import cut_partial_raw
from tests import _ltspice_recorded as rec
from tests import ltspice_recorder
from tests._ltspice_recorded import declared_points, header_fields
from tests.ltspice_bridge_recorder import (
    FIXTURES,
    load_manifest,
    reader_reply,
    reader_subjects,
    recorded_builds,
    sha256_bytes,
)

#: (the build whose reader was asked, the build that wrote the file, the file).
Subject = tuple[str, str, str]

SUBJECTS: list[Subject] = [
    (reader, written_by, name)
    for reader in recorded_builds()
    for written_by, names in sorted(load_manifest(FIXTURES / reader).get("reader", {}).items())
    for name in sorted(names)
]
READ = [subject for subject in SUBJECTS if "read" in reader_reply(*subject)]
REFUSED = [subject for subject in SUBJECTS if "refused" in reader_reply(*subject)]
#: Files of one point: no axis to read along, or an axis of one point (a
#: noise analysis at one frequency). A stepped run's ``.op.raw`` is one such:
#: it holds one point, however many steps the run took.
ONE_POINT = {
    "noise_one_freq",
    "op",
    "op_one_subckt",
    "op_save_device",
    "op_subckt",
    "step_ac_op_raw.op",
    "step_tran_op_raw.op",
    "tf",
    "tran_op_raw.op",
}
#: Files of one point on an axis, which the server keeps.
ONE_POINT_ON_AN_AXIS = {"noise_one_freq"}
#: A stepped run's ``.op.raw`` as LTspice 26 writes it: one point, and no
#: variable that says which step it is (XVII's opens with the step's ``param``).
#: Both readers refuse it.
STEP_UNSAID = {("ltspice26", "step_tran_op_raw.op"), ("ltspice26", "step_ac_op_raw.op")}
#: Files of the axis alone: LTspice 26 stored nothing a ``.save`` named.
AXIS_ONLY = {("ltspice26", "save_subckt_names_no_x")}
#: Stepped runs stored a point a step, the stepped parameter first, and the
#: analysis LTspice's reader names each.
PARAMETERS_FIRST = {
    "step_op": "Operating Point",
    "step_temp_op": "Operating Point",
    "step_ac_one_freq": "AC Analysis",
    "step_ac_one_freq_numdgt7": "AC Analysis",
    "step_noise_one_freq": "Noise Spectral Density - (V/Hz½ or A/Hz½)",
    "tf_step": "Transfer Function",
}
STOPPED_PART_WAY = "tran_killed"


def blocks(read: dict[str, Any]) -> list[dict[str, Any]]:
    """A reply's samples, a block a step: one block for a run that is not stepped."""
    if read["stepped"] == "true":
        return read["steps"]
    return [{"data": read["data"], "params": None}]


def theirs(columns: dict[str, list[float]]) -> np.ndarray:
    if "values" in columns:
        return np.asarray(columns["values"], dtype=np.float64)
    return np.asarray(columns["real"], dtype=np.float64) + 1j * np.asarray(
        columns["imag"], dtype=np.float64
    )


def at_the_precision_recorded(samples: np.ndarray, precision: str) -> np.ndarray:
    """Samples as the file holds them: LTspice's reader prints a single-precision
    sample in the fewest digits that give it back."""
    if precision == "double":
        return samples
    return samples.astype(np.complex64 if np.iscomplexobj(samples) else np.float32)


def offset_of(written_by: str, name: str) -> float:
    """The start a transient was saved from, which its header carries apart from its time points."""
    return float(header_fields(written_by, f"raw/{name}").get("Offset", ["0"])[0])


def is_whole(written_by: str, name: str) -> bool:
    """Whether the file holds the points its header declares; a run stopped part way may not."""
    if name != STOPPED_PART_WAY:
        return True
    _, payload = ltspice_recorder.split_raw(
        rec.recorded(written_by, f"raw/{name}.raw").read_bytes()
    )
    return declared_points(written_by, f"raw/{name}") == len(payload) // 12


# ---------------------------------------------------------------------------
# What was put to the reader
# ---------------------------------------------------------------------------


def test_a_reader_is_recorded():
    assert SUBJECTS, "no reply of LTspice's reader is committed"
    assert READ and REFUSED


@pytest.mark.parametrize("reader", recorded_builds())
def test_every_results_file_in_the_main_recordings_was_put_to_the_reader(reader: str):
    """A results file recorded since, or recorded again, has no reply here
    until the bridge recorder is run: ``scripts/record_ltspice_bridge.py``, on
    Windows with LTspice 26.1 or later."""
    held_now = {
        written_by: {
            name: sha256_bytes(rec.recorded(written_by, f"raw/{name}.raw").read_bytes())
            for name in names
        }
        for written_by, names in reader_subjects().items()
    }
    assert load_manifest(FIXTURES / reader)["reader"] == held_now


# ---------------------------------------------------------------------------
# Sample for sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reader", "written_by", "name"),
    [
        subject
        for subject in READ
        if subject[2] not in PARAMETERS_FIRST and is_whole(subject[1], subject[2])
    ],
)
def test_the_server_reads_the_samples_ltspices_reader_gives(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    said = reader_reply(reader, written_by, name)
    info, read = said["info"], said["read"]
    parsed = rec.decode(written_by, f"raw/{name}", tmp_path)
    raw = parsed.raw

    # The same traces, in the header's order, the axis apart.
    assert info["signals"] == raw.get_trace_names()[1:]
    assert read["unknown_signals"] == []
    flags = header_fields(written_by, f"raw/{name}")["Flags"][0].split()
    assert (read["precision"] == "double") == ("double" in flags)

    # The same steps, each of as many points.
    stepped = blocks(read)
    assert raw.get_steps() == list(range(len(stepped)))
    assert int(read["n_points"]) == sum(len(block["data"]["x"]) for block in stepped)

    offset = offset_of(written_by, name)
    for step, block in enumerate(stepped):
        data = block["data"]
        # The axis: LTspice's reader gives a time point as it is stored, sign
        # and all, and from zero whatever start the run was saved from.
        stored = np.asarray(data["x"], dtype=np.float64)
        np.testing.assert_array_equal(
            np.real(np.asarray(raw.get_axis(step))), abs(stored) + offset
        )
        for signal, columns in data["signals"].items():
            ours = np.asarray(raw.get_wave(signal, step))
            np.testing.assert_array_equal(
                at_the_precision_recorded(ours, read["precision"]),
                at_the_precision_recorded(theirs(columns), read["precision"]),
                err_msg=f"{signal}, step {step}",
            )
        if block["params"] is not None:
            # Both take a step's parameter values from the log beside the file.
            assert parsed.logs.value("steps")[step] == block["params"]


# ---------------------------------------------------------------------------
# Where the two present a file differently
# ---------------------------------------------------------------------------


def test_only_xvii_marks_time_points_by_their_sign_and_ltspices_reader_leaves_the_mark():
    """LTspice XVII stores some time points of a compressed transient negative.
    LTspice 26's reader hands them back negative, so its time axis for such a
    file runs backwards in places; the server's is plain time, which the
    sample-for-sample test holds to the size of each."""
    marked = {
        (written_by, name)
        for reader, written_by, name in READ
        if reader_reply(reader, written_by, name)["info"]["x_name"].lower() == "time"
        and any(
            float(x) < 0
            for block in blocks(reader_reply(reader, written_by, name)["read"])
            for x in block["data"]["x"]
        )
    }
    assert marked == {
        ("ltspice17", "compress_1mv"),
        ("ltspice17", "compress_1mv_maxstep"),
        ("ltspice17", "compress_1mv_wide_maxstep"),
        ("ltspice17", "compress_1v"),
        ("ltspice17", "tran"),
        ("ltspice17", "tran_double"),
        ("ltspice17", "tran_fastaccess"),
        ("ltspice17", "tran_killed"),
        ("ltspice17", "tran_window"),
    }


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in READ if s[2] == "tran_window"]
)
def test_a_transient_saved_from_a_later_start_begins_at_zero_for_ltspices_reader(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    """``.tran 0 2m 1m`` stores time from zero and the start in the header's
    ``Offset``, which LTspice's reader does not add and the server's does."""
    read = reader_reply(reader, written_by, name)["read"]
    assert offset_of(written_by, name) == pytest.approx(1e-3)
    assert read["data"]["x"][0] == 0.0
    assert abs(read["data"]["x"][-1]) == pytest.approx(1e-3)
    time = np.asarray(rec.decode(written_by, f"raw/{name}", tmp_path).raw.get_axis(0))
    assert (time[0], time[-1]) == pytest.approx((1e-3, 2e-3))


@pytest.mark.parametrize(
    ("reader", "written_by", "name"),
    [s for s in REFUSED if (s[1], s[2]) not in AXIS_ONLY | STEP_UNSAID],
)
def test_ltspices_reader_refuses_a_file_of_one_point_which_the_server_reads(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    assert reader_reply(reader, written_by, name) == {"refused": "failed to open raw file"}
    assert name in ONE_POINT
    raw = rec.decode(written_by, f"raw/{name}", tmp_path).raw
    if name in ONE_POINT_ON_AN_AXIS:
        assert len(raw.get_axis(0)) == 1
    else:
        with pytest.raises(NoAxisError):
            raw.get_axis(0)
    assert [len(raw.get_wave(trace, 0)) for trace in raw.get_trace_names()] == [1] * len(
        raw.get_trace_names()
    )


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in REFUSED if (s[1], s[2]) in STEP_UNSAID]
)
def test_a_stepped_runs_op_raw_that_does_not_say_its_step_is_refused_by_both(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    """The run took three steps and the file holds one point, with nothing
    that says which step's it is, so neither reader gives its values."""
    assert reader_reply(reader, written_by, name) == {"refused": "failed to open raw file"}
    assert declared_points(written_by, f"raw/{name}") == 1
    assert "stepped" in header_fields(written_by, f"raw/{name}")["Flags"][0].split()
    raw = rec.decode(written_by, f"raw/{name}", tmp_path).raw
    with pytest.raises(ResultError, match="Step boundaries are unresolved"):
        raw.get_wave(raw.get_trace_names()[0], 0)


def test_every_file_of_one_point_or_of_the_axis_alone_is_refused_and_no_other():
    assert {(written_by, name) for _reader, written_by, name in REFUSED} == {
        (written_by, name)
        for written_by, names in reader_subjects().items()
        for name in names
        if name in ONE_POINT
    } | AXIS_ONLY


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in REFUSED if (s[1], s[2]) in AXIS_ONLY]
)
def test_ltspices_reader_refuses_a_file_of_the_axis_alone_which_the_server_reads(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    assert reader_reply(reader, written_by, name) == {
        "refused": (
            "the 'signals' parameter is required and must be a non-empty array of signal names"
        )
    }
    raw = rec.decode(written_by, f"raw/{name}", tmp_path).raw
    assert raw.get_trace_names() == ["time"]
    assert len(raw.get_axis(0)) > 1


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in READ if s[2] in PARAMETERS_FIRST]
)
def test_a_run_stored_a_point_a_step_is_one_sweep_to_ltspices_reader_and_steps_here(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    """One point is stored a step, with the stepped parameter as the first
    variable: a stepped operating point (over a parameter or a temperature), a
    stepped transfer function, and a stepped AC or noise analysis at one
    frequency, which stores no frequency. LTspice's reader calls that an
    unstepped run with the parameter for its axis; the server's keeps the
    steps, and the values are the same."""
    said = reader_reply(reader, written_by, name)
    info, read = said["info"], said["read"]
    assert (read["stepped"], read["analysis"]) == ("false", PARAMETERS_FIRST[name])
    raw = rec.decode(written_by, f"raw/{name}", tmp_path).raw
    stepped_parameter, *traces = raw.get_trace_names()
    assert (info["x_name"], info["signals"]) == (stepped_parameter, traces)
    steps = raw.get_steps()
    assert len(steps) == int(read["n_points"]) == 3

    def a_point_a_step(trace: str) -> np.ndarray:
        return np.asarray([np.asarray(raw.get_wave(trace, step))[0] for step in steps])

    precision = read["precision"]
    np.testing.assert_array_equal(
        at_the_precision_recorded(np.real(a_point_a_step(stepped_parameter)), precision),
        at_the_precision_recorded(np.asarray(read["data"]["x"]), precision),
    )
    for signal, columns in read["data"]["signals"].items():
        np.testing.assert_array_equal(
            at_the_precision_recorded(a_point_a_step(signal), precision),
            at_the_precision_recorded(theirs(columns), precision),
            err_msg=signal,
        )


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in READ if not is_whole(s[1], s[2])]
)
def test_a_run_stopped_part_way_is_read_as_far_as_its_header_says_by_ltspices_reader(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    """The header's point count is whatever LTspice last wrote there, and the
    file may hold more. LTspice's reader goes by the count; the server's
    decoder refuses a file whose header and samples disagree, and says how far
    the run got another way (``read_partial_raw_progress``)."""
    read = reader_reply(reader, written_by, name)["read"]
    assert int(read["n_points"]) == declared_points(written_by, f"raw/{name}")
    with pytest.raises(RawHeaderError):
        rec.decode(written_by, f"raw/{name}", tmp_path)


@pytest.mark.parametrize(
    ("reader", "written_by", "name"), [s for s in READ if s[2] == STOPPED_PART_WAY]
)
def test_a_run_stopped_part_way_and_cut_reads_as_ltspices_reader_reads_it(
    reader: str, written_by: str, name: str, tmp_path: Path
):
    """Cut (``cut_partial_raw``), the stopped run's file holds the points its
    header declared and decodes to the samples LTspice's reader gives for the
    file as LTspice left it. LTspice XVII's file holds more records than its
    count, and the records past the count go back in time: they are cut off."""
    stopped = tmp_path / f"{name}.raw"
    shutil.copyfile(rec.recorded(written_by, f"raw/{name}.raw"), stopped)
    declared = declared_points(written_by, f"raw/{name}")

    progress = cut_partial_raw(stopped)

    assert progress is not None
    assert progress.points == declared
    raw = rec.decode_file(stopped, tmp_path).raw
    read = reader_reply(reader, written_by, name)["read"]
    assert int(read["n_points"]) == declared
    time = np.real(np.asarray(raw.get_axis(0)))
    np.testing.assert_array_equal(time, np.abs(np.asarray(read["data"]["x"], dtype=np.float64)))
    assert time[-1] == pytest.approx(progress.last_axis_value)
    for signal, columns in read["data"]["signals"].items():
        np.testing.assert_array_equal(
            at_the_precision_recorded(np.asarray(raw.get_wave(signal, 0)), read["precision"]),
            at_the_precision_recorded(theirs(columns), read["precision"]),
            err_msg=signal,
        )


def test_xvii_records_past_its_count_go_back_in_time(tmp_path: Path):
    """Why an LTspice file is cut at its count and not at its last whole record:
    in the run LTspice XVII left, the records after the count do not run on."""
    case_id = f"raw/{STOPPED_PART_WAY}"
    _, payload = ltspice_recorder.split_raw(
        rec.recorded("ltspice17", f"{case_id}.raw").read_bytes()
    )
    fields = header_fields("ltspice17", case_id)
    assert "double" not in fields["Flags"][0].split()
    # A double for time, then a single for each other variable.
    record = 8 + 4 * (int(fields["No. Variables"][0]) - 1)
    time = np.abs(
        np.frombuffer(
            payload,
            dtype=np.dtype([("time", "<f8"), ("rest", f"V{record - 8}")]),
            count=len(payload) // record,
        )["time"]
    )
    declared = declared_points("ltspice17", case_id)
    assert len(time) > declared
    assert np.all(np.diff(time[:declared]) >= 0)
    assert np.any(np.diff(time[declared - 1 :]) < 0)
