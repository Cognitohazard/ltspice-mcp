"""load_raw(signals=...): a read that keeps only the traces some names read.

The parser worker decides which traces to keep, and why a name reads none,
by the rule every reader of a signal uses (``trace_names.read_in_plot``), cuts
the file's header to them, and writes only those; the size limit then applies
to what is kept, so a file too large to read whole can be read for a few
signals. Each test reads a recorded file through the real parser process, most
of them both ways, and compares.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import parser_service
from ltspice_mcp.lib.raw_header import leading_parameters
from ltspice_mcp.lib.result_kinds import step_variables
from ltspice_mcp.lib.trace_names import signal_names, step_value_note
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, SyncApi, stage_recorded_fixture


def _same(full: np.ndarray, chosen: np.ndarray) -> None:
    assert chosen.dtype == full.dtype and np.array_equal(chosen, full, equal_nan=True)


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    ("case_id", "signal"),
    [
        ("raw/tran", "V(out)"),
        ("raw/tran_fastaccess", "V(out)"),
        ("raw/tran_as_text", "V(out)"),
        ("raw/step_tran", "V(out)"),
        ("raw/ac", "V(out)"),
    ],
)
def test_a_chosen_read_keeps_the_axis_its_steps_and_its_signals(
    build: str, case_id: str, signal: str, state_no_sim: SessionState, work_dir: Path
):
    """The values, the axis and the steps are those of the whole-file read,
    whatever the file's storage (binary, fast access, text) and steps."""
    raw = rec.stage_run(build, case_id, work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    chosen = api.load_raw(path=raw, signals=[signal.lower()])
    assert chosen.signals == [
        full.signals[0],
        next(s for s in full.signals if s.lower() == signal.lower()),
    ]
    assert chosen.steps == full.steps and chosen.step_count == full.step_count
    for step in range(full.step_count):
        _same(full.axis(step=step), chosen.axis(step=step))
        _same(full.trace(signal, step=step), chosen.trace(signal, step=step))
    (said,) = [o for o in chosen.observations if o["code"] == "signals_chosen"]
    assert "no other trace" in said["detail"]


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_node_pair_keeps_both_its_nodes(build: str, state_no_sim: SessionState, work_dir: Path):
    raw = rec.stage_run(build, "raw/tran", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    chosen = api.load_raw(path=raw, signals=["V(out,in)"])
    assert sorted(chosen.signals[1:]) == ["V(in)", "V(out)"]
    _same(full.trace("V(out,in)"), chosen.trace("V(out,in)"))


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_stepped_operating_point_keeps_its_step_parameters(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """A stepped .op stores each step's parameter values as its leading
    variables; a chosen read keeps them, so its steps are the whole read's."""
    raw = rec.stage_run(build, "raw/step_op", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    signal = full.signals[-1]
    chosen = api.load_raw(path=raw, signals=[signal])
    header = full._raw.header
    parameters = [header.variables[index].name for index in leading_parameters(header)]
    assert parameters
    assert [variable.name for variable in chosen._raw.header.variables] == [*parameters, signal]
    assert chosen.steps == full.steps and chosen.step_count == full.step_count > 1
    for step in range(full.step_count):
        assert chosen.value(signal, step=step) == full.value(signal, step=step)


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_step_parameter_is_a_step_value_to_every_reader(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """A stepped .op's parameter is a variable of the raw, and holds each
    step's stepped value: no signal reads it, in a chosen read or a whole one,
    and both say where its values are."""
    raw = rec.stage_run(build, "raw/step_op", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    (parameter,) = full.steps[0]
    note = step_value_note(parameter)
    with pytest.raises(ResultError) as whole:
        full.trace(parameter)
    assert note in str(whole.value)
    with pytest.raises(ResultError) as chosen:
        api.load_raw(path=raw, signals=[parameter.upper()])
    assert step_value_note(parameter.upper()) in str(chosen.value)
    assert full.signals[0] in str(chosen.value).split("This plot's traces:")[1]


def test_a_pair_of_two_grounds_reads_no_trace(state_no_sim: SessionState, work_dir: Path):
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    with pytest.raises(ResultError, match="names ground twice"):
        api.load_raw(path=raw).trace("V(0,0)")
    with pytest.raises(ResultError, match=r"'V\(gnd,0\)' names ground twice"):
        api.load_raw(path=raw, signals=["V(gnd,0)"])


def test_a_signal_is_looked_for_in_the_plot_read(state_no_sim: SessionState, work_dir: Path):
    """A file of two plots: a signal of the second is refused from a read of
    the first, after a read of the second has kept it."""
    path = stage_recorded_fixture(work_dir, "ngspice_noise_2plot")
    api = SyncApi(state_no_sim)
    later = api.load_raw(path, plot_index=1, dialect="ngspice", signals=["v(onoise_total)"])
    whole = api.load_raw(path, plot_index=1, dialect="ngspice")
    assert later.value("v(onoise_total)") == whole.value("v(onoise_total)")
    with pytest.raises(ResultError) as refused:
        api.load_raw(path, plot_index=0, dialect="ngspice", signals=["v(onoise_total)"])
    listed = str(refused.value).split("This plot's traces:")[1]
    assert "onoise_spectrum" in listed and "onoise_total" not in listed


@pytest.mark.parametrize("stamps", ["settled_stamps", "unsettled_stamps"])
def test_a_chosen_read_and_a_whole_read_of_one_file_are_kept_apart(
    stamps: str, request: pytest.FixtureRequest, state_no_sim: SessionState, work_dir: Path
):
    """Served from the stamp cache (settled) or the parser's content cache
    (unsettled), each read keeps what it asked for: the names in either order
    are one read, and another plot or other names are not."""
    request.getfixturevalue(stamps)
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    chosen = api.load_raw(path=raw, signals=["V(out)"])
    full = api.load_raw(path=raw)
    again = api.load_raw(path=raw, signals=["V(out)"])
    both = api.load_raw(path=raw, signals=["V(out)", "V(in)"])
    turned = api.load_raw(path=raw, signals=["V(in)", "V(out)"])
    assert len(full.signals) > len(chosen.signals) == len(again.signals) == 2
    assert both.signals == turned.signals and len(both.signals) == 3
    assert not [o for o in full.observations if o["code"] == "signals_chosen"]
    assert api.load_raw(path=raw).signals == full.signals


def test_a_refusal_names_the_way_to_read_the_name(state_no_sim: SessionState, work_dir: Path):
    """A chosen result refuses a trace it was not read for with how to read
    it, and every other name as a whole read does, without that advice and
    with the same flags; ``compare`` says so too."""
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    chosen = api.load_raw(path=raw, signals=["V(out)"])
    for name in ("V(0,0)", "V(out)-V(in)", "V(out,nowhere)", "V(nowhere)"):
        with pytest.raises(ResultError) as whole:
            full.trace(name)
        with pytest.raises(ResultError) as read:
            chosen.trace(name)
        advised = "load_raw again" in str(read.value)
        assert advised == (name in ("V(out,nowhere)", "V(nowhere)")), name
        assert str(read.value).startswith(str(whole.value).split(" Available signals:")[0])
        assert (read.value.show_hint, read.value.python_route) == (
            whole.value.show_hint,
            whole.value.python_route,
        ), name
    with pytest.raises(ResultError, match=r"read with signals=\['V\(out\)'\]"):
        full.compare(chosen, signals=["V(in)"])


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_step_value_in_a_chosen_result_is_refused_as_one(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    raw = rec.stage_run(build, "raw/step_op", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    (parameter,) = full.steps[0]
    chosen = api.load_raw(path=raw, signals=[full.signals[-1]])
    with pytest.raises(ResultError) as refused:
        chosen.trace(parameter)
    assert step_value_note(parameter) in str(refused.value)
    assert "load_raw again" not in str(refused.value)


def test_compare_does_not_call_a_trace_not_read_missing(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    chosen = api.load_raw(path=raw, signals=["V(out)"])
    unread = sorted(name for name in full.signals[1:] if name != "V(out)")
    assert unread
    there = full.compare(chosen)
    assert there["only_here"] == [] and there["unread_there"] == unread
    assert there["signals"]["V(out)"]["max_abs"] == 0
    here = chosen.compare(full)
    assert here["only_there"] == [] and here["unread_here"] == unread
    assert "unread_here" not in there and "unread_there" not in full.compare(full)


def _rc(work_dir: Path) -> Path:
    """The suite's recorded RC transient, copied into the session's folder."""
    raw = work_dir / "rc.raw"
    raw.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes())
    return raw


def test_a_name_with_no_trace_is_refused_with_the_traces_there_are(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    with pytest.raises(ResultError) as refused:
        SyncApi(state_no_sim).load_raw(path=raw, signals=["V(out)", "V(nowhere)"])
    message = str(refused.value)
    assert "Signal 'V(nowhere)' not found." in message
    assert "V(out)" in message.split("This plot's traces:")[1]


def test_an_unread_trace_says_the_result_was_read_for_others(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    chosen = SyncApi(state_no_sim).load_raw(path=raw, signals=[" V(out) "])
    with pytest.raises(ResultError, match=r"read with signals=\['V\(out\)'\]"):
        chosen.trace("V(in)")


def test_a_file_too_large_to_read_whole_is_read_for_its_signals(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """With the limit between what the whole file holds and what the axis and
    one trace hold, the whole read is refused, naming the way round, and the
    chosen read goes through. A chosen read over the limit says to choose
    fewer."""
    raw = rec.stage_run(rec.BUILDS[0], "raw/tran", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    points = len(full.axis())
    whole = sum(full.trace(name).itemsize for name in full.signals) * points
    one = (full.axis().itemsize + full.trace("V(out)").itemsize) * points
    assert one < whole
    monkeypatch.setattr(parser_service, "NUMERIC_BYTES", one)
    other = work_dir / "copy.raw"
    other.write_bytes(raw.read_bytes())
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=other)
    message = str(refused.value)
    assert "its traces hold" in message and "one read keeps" in message
    assert "load_raw(..., signals=" in message
    chosen = api.load_raw(path=other, signals=["V(out)"])
    _same(full.trace("V(out)"), chosen.trace("V(out)"))
    with pytest.raises(ResultError, match="Choose fewer signals"):
        api.load_raw(path=other, signals=["V(out)", "V(in)"])


def test_a_file_stored_a_point_at_a_time_is_held_to_what_decoding_it_holds(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """spicelib decodes a plot stored a point at a time whole, whichever traces
    are kept (``docs/spicelib_bugs.md``), and one in LTspice's Fast Access
    layout a trace at a time. With the decode budget between what the axis
    and one trace hold and what the whole file holds, the same run is refused
    in the one layout, naming the other, and read in the other."""
    build = rec.BUILDS[0]
    stored = rec.stage_run(build, "raw/tran", work_dir)
    converted = rec.stage_run(build, "raw/tran_fastaccess", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=stored)
    points = len(full.axis())
    one = (full.axis().itemsize + full.trace("V(out)").itemsize) * points
    assert one < sum(full.trace(name).itemsize for name in full.signals) * points
    monkeypatch.setattr(parser_service, "DECODE_BYTES", one)
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=stored, signals=["V(out)"])
    message = str(refused.value)
    assert "stores its numbers a point at a time" in message
    assert "-FastAccess" in message
    chosen = api.load_raw(path=converted, signals=["V(out)"])
    _same(full.trace("V(out)"), chosen.trace("V(out)"))


def test_signals_are_checked_before_a_read(state_no_sim: SessionState, work_dir: Path):
    from ltspice_mcp.api import ApiValidationError

    api = SyncApi(state_no_sim)
    raw = _rc(work_dir)
    many = [f"V(n{i})" for i in range(65)]
    wide = [f"V({i:02d}" + "ü" * 250 + ")" for i in range(64)]
    for wrong in (
        "V(out)",
        [],
        [""],
        ["  "],
        [["V(out)"]],
        [1],
        many,
        ["V(" + "a" * 300 + ")"],
        wide,
    ):
        with pytest.raises(ApiValidationError, match="signals"):
            api.load_raw(path=raw, signals=wrong)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    "case_id", ["raw/tran", "raw/subckt_names_tran", "raw/noise", "raw/step_op"]
)
def test_a_chosen_read_reads_each_name_as_a_whole_read_does(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """For every trace, its other spellings, the node pairs of its voltages
    and names that read nothing, a chosen read through the parser worker gives
    the values a whole read gives, or refuses with the reason it gives."""
    raw = rec.stage_run(build, case_id, work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    header = full._raw.header
    names = signal_names(header)
    assert names == full.signals
    voltages = [name for name in names if name.lower().startswith("v(")]
    asked = [
        *names,
        *(name.upper() for name in names),
        *(name.replace(":", ".") for name in names),
        *(f"V({a[2:-1]},{b[2:-1]})" for a, b in pairwise(voltages)),
        *(f"V({a[2:-1]},0)" for a in voltages[:1]),
        *step_variables(header),
        "V(nowhere)",
        "V(0,0)",
        "V(gnd,0)",
        "V(out)-V(in)",
    ]
    read: list[str] = []
    refused: dict[str, str] = {}
    for name in dict.fromkeys(asked):
        try:
            full.trace(name)
        except ResultError as whole:
            refused[name] = str(whole).split(" Available signals:")[0]
        else:
            read.append(name)
    assert read and refused
    for start in range(0, len(read), 64):
        some = read[start : start + 64]
        chosen = api.load_raw(path=raw, signals=some)
        for name in some:
            for step in range(full.step_count):
                _same(full.trace(name, step=step), chosen.trace(name, step=step))
    with pytest.raises(ResultError) as chosen_refusal:
        api.load_raw(path=raw, signals=list(refused))
    for name, why in refused.items():
        assert why in str(chosen_refusal.value), name


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_noise_node_pair_is_refused_before_it_is_read(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """A noise result's voltages are spectral densities, whose difference is
    not the noise between two nodes: a whole read refuses the pair when it is
    read, and a chosen read before reading, with the same reason."""
    raw = rec.stage_run(build, "raw/noise", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    voltages = [name for name in full.signals if name.lower().startswith("v(")][:2]
    pair = f"V({voltages[0][2:-1]},{voltages[1][2:-1]})"
    with pytest.raises(ResultError, match="spectral densities") as whole:
        full.trace(pair)
    with pytest.raises(ResultError, match="spectral densities") as chosen:
        api.load_raw(path=raw, signals=[pair])
    assert str(whole.value) in str(chosen.value)
    kept = api.load_raw(path=raw, signals=voltages)
    with pytest.raises(ResultError, match="spectral densities") as read:
        kept.trace(pair.lower())
    assert "load_raw again" not in str(read.value)
