"""load_raw(signals=...): a read that keeps only the traces some names read.

The parser worker decides which traces to keep by the rule every reader of a
signal uses (``trace_names``), cuts the file's header to them, and decodes only
those; the size limit then applies to what is kept, so a file too large to read
whole can be read for a few signals. Each test reads a recorded LTspice file
both ways, through the real parser process, and compares.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import parser_service, services
from ltspice_mcp.lib.raw_header import leading_parameters
from ltspice_mcp.lib.trace_names import traces_read
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, SyncApi


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
    assert chosen.steps == full.steps and chosen.step_count == full.step_count > 1
    for step in range(full.step_count):
        assert chosen.value(signal, step=step) == full.value(signal, step=step)


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
    assert "'V(nowhere)' names no trace of this result" in message
    assert "V(out)" in message.split("Its traces:")[1]


def test_an_unread_trace_says_the_result_was_read_for_others(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    chosen = SyncApi(state_no_sim).load_raw(path=raw, signals=["V(out)"])
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


def test_signals_are_checked_before_a_read(state_no_sim: SessionState, work_dir: Path):
    from ltspice_mcp.api import ApiValidationError

    api = SyncApi(state_no_sim)
    raw = _rc(work_dir)
    for wrong in ("V(out)", [], [""], ["V(out)"] * 0 + [f"V(n{i})" for i in range(65)]):
        with pytest.raises(ApiValidationError, match="signals"):
            api.load_raw(path=raw, signals=wrong)  # pyright: ignore[reportArgumentType]


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    "case_id", ["raw/tran", "raw/subckt_names_tran", "raw/noise", "raw/step_op"]
)
def test_the_traces_a_name_reads_are_those_resolve_signal_reads(
    build: str, case_id: str, state_no_sim: SessionState, work_dir: Path
):
    """The worker keeps what ``traces_read`` names, from a header's names; the
    readers of a decoded result go through ``resolve_signal``. Both answer
    which traces a name reads, so they must agree, for every trace, its other
    spellings and the node pairs of its voltages."""
    from ltspice_mcp.lib.device_names import writes_xvii_names

    raw = rec.stage_run(build, case_id, work_dir)
    decoded = SyncApi(state_no_sim).load_raw(path=raw)._raw
    names = list(decoded.get_trace_names())
    xvii = writes_xvii_names(decoded)
    voltages = [name for name in names if name.lower().startswith("v(")]
    asked = [
        *names,
        *(name.upper() for name in names),
        *(name.replace(":", ".") for name in names),
        *(f"V({a[2:-1]},{b[2:-1]})" for a, b in pairwise(voltages)),
        *(f"V({a[2:-1]},0)" for a in voltages[:1]),
        "V(nowhere)",
    ]
    for name in asked:
        read = traces_read(names, name, xvii=xvii)
        try:
            signal = services.resolve_signal(decoded, name)
        except ResultError as refused:
            if "a .noise result holds spectral densities" in str(refused):
                continue  # the next test pins this one difference
            assert read is None, name
            continue
        assert read is not None and set(read) == {signal.plus, signal.minus} - {None}, name


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_noise_node_pair_names_its_two_traces_and_is_refused_when_read(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """The one place the two answers differ, on purpose: a noise result's
    voltages are spectral densities, whose difference is not the noise
    between two nodes. ``traces_read`` still names the pair's two traces, so a
    chosen read keeps them; reading the pair is refused, with the reason, by
    ``resolve_signal``, from a chosen read as from a whole one."""
    raw = rec.stage_run(build, "raw/noise", work_dir)
    api = SyncApi(state_no_sim)
    names = api.load_raw(path=raw).signals
    voltages = [name for name in names if name.lower().startswith("v(")][:2]
    pair = f"V({voltages[0][2:-1]},{voltages[1][2:-1]})"
    assert traces_read(names, pair, xvii=False) == tuple(voltages)
    chosen = api.load_raw(path=raw, signals=[pair])
    assert set(chosen.signals[1:]) == set(voltages)
    with pytest.raises(ResultError, match="spectral densities"):
        chosen.trace(pair)


def test_leading_parameters_are_what_a_stepped_op_opens_with(work_dir: Path):
    """The cut keeps a plot's leading parameters by ``raw_header`` itself."""
    from ltspice_mcp.lib.raw_header import RawLimits, preflight_raw

    raw = rec.stage_run(rec.BUILDS[0], "raw/step_op", work_dir)
    limits = parser_service.raw_limits(4096)
    assert isinstance(limits, RawLimits)
    (plot,) = preflight_raw(raw, limits=limits).plots
    assert leading_parameters(plot) and leading_parameters(plot)[0] >= 0
