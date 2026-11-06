"""load_raw(window=...): a read that keeps only the points of a span of the axis.

The parser worker keeps the points ``signal_analysis.window_span`` keeps, the
rule every window of an axis follows (``metrics.window_indices``, which the
exports use), in each step, and holds the size limit to them, so a file with
too many points for one read can be read a stretch at a time. Each test reads
a recorded file through the real parser process, most of them both ways, and
compares.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.api import ApiValidationError
from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import metrics, parser_service, signal_analysis
from ltspice_mcp.lib.signal_analysis import UnorderedAxisError, window_span
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import FIXTURES_DIR, SyncApi


def _same(full: np.ndarray, kept: np.ndarray) -> None:
    assert kept.dtype == full.dtype and np.array_equal(kept, full, equal_nan=True)


def _span(axis: np.ndarray, low: float, high: float) -> tuple[float, float]:
    """Bounds ``low`` and ``high`` of the way along ``axis``, between its points."""
    values = np.real(axis)
    first, last = float(values[0]), float(values[-1])
    return first + low * (last - first), first + high * (last - first)


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize(
    "case_id",
    ["raw/tran", "raw/tran_fastaccess", "raw/tran_as_text", "raw/ac", "raw/step_tran"],
)
@pytest.mark.parametrize("signals", [None, ["V(out)"]])
def test_a_window_keeps_the_points_the_window_rule_keeps(
    build: str, case_id: str, signals: list[str] | None, state_no_sim: SessionState, work_dir: Path
):
    """In every step, whatever the file's storage, with or without chosen
    signals: the points a window of the whole read's axis keeps, and its steps."""
    raw = rec.stage_run(build, case_id, work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    start, end = _span(full.axis(step=0), 0.3, 0.7)
    kept = api.load_raw(path=raw, signals=signals, window=(start, end))
    assert kept.steps == full.steps and kept.step_count == full.step_count
    assert kept.signals == (full.signals if signals is None else [full.signals[0], "V(out)"])
    for step in range(full.step_count):
        axis = full.axis(step=step)
        lo, hi = metrics.window_indices(np.real(axis), start, end)
        assert 0 < hi - lo < axis.size
        _same(axis[lo:hi], kept.axis(step=step))
        _same(full.trace("V(out)", step=step)[lo:hi], kept.trace("V(out)", step=step))
    (said,) = [o for o in kept.observations if o["code"] == "window_chosen"]
    assert "no other point" in said["detail"]


def _rc(work_dir: Path) -> Path:
    """The suite's recorded RC transient, copied into the session's folder."""
    raw = work_dir / "rc.raw"
    raw.write_bytes((FIXTURES_DIR / "ltspice_tran_rc.raw").read_bytes())
    return raw


def test_an_open_end_keeps_all_on_its_side_and_text_reads_as_a_value(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    axis = full.axis()
    middle = float(axis[axis.size // 2])
    before = api.load_raw(path=raw, window=(None, middle))
    after = api.load_raw(path=raw, window=(f"{middle * 1e6:.17g}u", None))
    _same(axis[: axis.size // 2 + 1], before.axis())
    _same(axis[axis.size // 2 :], after.axis())


def test_a_window_that_meets_no_point_is_refused_with_where_the_axis_runs(
    state_no_sim: SessionState, work_dir: Path
):
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    axis = api.load_raw(path=raw).axis()
    beyond = float(axis[-1]) * 2
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=raw, window=(beyond, beyond * 2))
    message = str(refused.value)
    assert "keeps no point of plot 0" in message
    assert f"axis runs from {float(axis[0]):g} to {float(axis[-1]):g}" in message


@pytest.mark.parametrize("build", rec.BUILDS)
def test_a_plot_with_no_axis_has_no_window(build: str, state_no_sim: SessionState, work_dir: Path):
    """A stepped .op holds one point a step: no span of an axis to keep."""
    raw = rec.stage_run(build, "raw/step_op", work_dir)
    with pytest.raises(ResultError, match="has none: it holds one point"):
        SyncApi(state_no_sim).load_raw(path=raw, window=(0, 1))


@pytest.mark.parametrize("build", rec.BUILDS)
def test_steps_not_told_apart_have_no_window(
    build: str, state_no_sim: SessionState, work_dir: Path
):
    """Without its log a stepped transient's steps are not bound, so its axis
    restarts at each step and decreases there: no window is one stretch."""
    raw = rec.stage_run(build, "raw/step_tran", work_dir)
    api = SyncApi(state_no_sim)
    window = _span(api.load_raw(path=raw).axis(step=0), 0.3, 0.7)
    alone = work_dir / "alone.raw"
    alone.write_bytes(raw.read_bytes())
    with pytest.raises(ResultError, match="decreases within a step"):
        api.load_raw(path=alone, window=window)


def test_a_window_reads_a_file_with_too_many_points_for_one_read(
    state_no_sim: SessionState, work_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    """With the limit between what a window keeps and what the whole axis
    holds, the whole read is refused, naming the window, and the windowed read
    goes through; a window too wide for the limit says to narrow it."""
    raw = rec.stage_run(rec.BUILDS[0], "raw/tran", work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    axis = full.axis()
    widths = sum(full.trace(name).itemsize for name in full.signals)
    start, end = _span(axis, 0.4, 0.5)
    lo, hi = metrics.window_indices(axis, start, end)
    monkeypatch.setattr(parser_service, "NUMERIC_BYTES", widths * (hi - lo))
    other = work_dir / "copy.raw"
    other.write_bytes(raw.read_bytes())
    with pytest.raises(ResultError) as refused:
        api.load_raw(path=other)
    assert "window=('1m', '2m')" in str(refused.value)
    kept = api.load_raw(path=other, window=(start, end))
    _same(full.trace("V(out)")[lo:hi], kept.trace("V(out)"))
    with pytest.raises(ResultError, match="a shorter window"):
        api.load_raw(path=other, window=_span(axis, 0.1, 0.9))


@pytest.mark.parametrize("stamps", ["settled_stamps", "unsettled_stamps"])
def test_windowed_and_whole_reads_of_one_file_are_kept_apart(
    stamps: str, request: pytest.FixtureRequest, state_no_sim: SessionState, work_dir: Path
):
    request.getfixturevalue(stamps)
    raw = _rc(work_dir)
    api = SyncApi(state_no_sim)
    full = api.load_raw(path=raw)
    first = api.load_raw(path=raw, window=_span(full.axis(), 0.1, 0.4))
    second = api.load_raw(path=raw, window=_span(full.axis(), 0.5, 0.9))
    both = api.load_raw(path=raw, signals=["V(out)"], window=_span(full.axis(), 0.1, 0.4))
    assert float(first.axis()[-1]) < float(second.axis()[0])
    _same(first.axis(), both.axis())
    assert len(both.signals) == 2 < len(first.signals) == len(full.signals)
    _same(api.load_raw(path=raw).axis(), full.axis())


def test_a_window_is_checked_before_a_read(state_no_sim: SessionState, work_dir: Path):
    api = SyncApi(state_no_sim)
    raw = _rc(work_dir)
    for wrong in (
        "1m",
        (1,),
        (None, None),
        (2, 1),
        ("1m", "1m"),
        ("abc", None),
        (float("nan"), 1),
        (True, 2),
        ([1], None),
    ):
        with pytest.raises(ApiValidationError, match="window"):
            api.load_raw(path=raw, window=wrong)  # pyright: ignore[reportArgumentType]


def test_the_window_rule_reads_an_axis_a_block_at_a_time(monkeypatch: pytest.MonkeyPatch):
    """The parser worker reads a long axis in blocks, holding no copy of it:
    its span is the one the axis read whole gives, and a decrease between two
    blocks is found as one within a block is."""
    rng = np.random.default_rng(7)
    axis = np.cumsum(rng.integers(0, 3, 500)).astype(float)
    bounds = [None, *np.unique(axis)[::7], axis[0] - 1, axis[-1] + 1]
    whole = {
        (start, end): window_span(axis, start, end)
        for start in bounds
        for end in bounds
        if start is None or end is None or start < end
    }
    monkeypatch.setattr(signal_analysis, "_WINDOW_CHUNK", 7)
    for (start, end), span in whole.items():
        assert window_span(axis, start, end) == span
        if start is not None or end is not None:
            lo, hi = span
            inside = [
                index
                for index, value in enumerate(axis)
                if (start is None or value >= start) and (end is None or value <= end)
            ]
            assert list(range(lo, hi)) == inside
    # Blocks of seven: index 7 opens the second, index 3 is inside the first.
    for where in (7, 3):
        turned = np.arange(20, dtype=float)
        turned[where] = turned[where - 1] - 1
        with pytest.raises(UnorderedAxisError):
            window_span(turned, 0, None)
