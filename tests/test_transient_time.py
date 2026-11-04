"""Where a transient's stored time sits in deck time, read one way.

LTspice stores a windowed run's times from 0 with its start in the header's
``Offset:`` field, and XVII stores some times negated. The full decode and
the progress reader of a running job both correct a stored time with
``decoded_raw.time_offset`` and ``deck_time``, so the time a run has reached
is a time of the result the run leaves.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.lib.raw_header import preflight_raw
from ltspice_mcp.lib.raw_parser import read_partial_raw_progress
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import split_raw


def _cut(build: str, case_id: str, points: int, directory: Path) -> Path:
    """The recorded raw, cut after its first ``points`` records."""
    source = rec.recorded(build, f"{case_id}.raw")
    (plot,) = preflight_raw(source, limits=rec.RAW_LIMITS).plots
    header, payload = split_raw(source.read_bytes())
    cut = directory / f"{build}-{Path(case_id).name}.raw"
    cut.write_bytes(header + payload[: points * sum(plot.value_bytes)])
    return cut


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("case_id", ["raw/tran_window", "raw/tran"])
def test_the_time_a_run_has_reached_is_a_time_of_its_result(
    build: str, case_id: str, tmp_path: Path
):
    """``raw/tran_window`` is ``.tran 0 2m 1m``, stored from 0 with its start
    in ``Offset``; XVII stores some of ``raw/tran``'s times negated."""
    axis = np.asarray(rec.decode(build, case_id, tmp_path / "decoded").raw.get_axis(0))
    for points in (1, len(axis) // 2, len(axis) - 1):
        progress = read_partial_raw_progress(_cut(build, case_id, points, tmp_path), "ltspice")
        assert progress is not None
        assert progress.points == points
        assert progress.last_axis_value == pytest.approx(axis[points - 1], rel=1e-12)


@pytest.mark.parametrize("offset", ["nan", "inf", "not-a-number"])
def test_an_offset_that_is_not_a_number_gives_no_time(offset: str, tmp_path: Path):
    """The full read refuses such a raw; the progress reader names no time
    rather than one shifted by it or left unshifted."""
    build = rec.BUILDS[0]
    data = rec.recorded(build, "raw/tran_window.raw").read_bytes()
    header, payload = split_raw(data)
    text = header.decode("utf-16-le")
    changed = re.sub(r"(?m)^Offset:[^\r\n]*", f"Offset: {offset}", text)
    assert changed != text
    path = tmp_path / "bad_offset.raw"
    path.write_bytes(changed.encode("utf-16-le") + payload[: len(payload) // 2])

    progress = read_partial_raw_progress(path, "ltspice")

    assert progress is not None
    assert progress.points is not None and progress.points > 0
    assert progress.last_axis_value is None
