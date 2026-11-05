"""A signal a caller names is found the one way on every route that takes one.

``resolve_signal`` reads one trace, or a node pair ``V(a,b)`` as
``V(a) - V(b)``. ``RawResult.trace`` and ``RawResult.compare`` both go
through it, so a name one of them reads the other reads too. Looking a trace
up by name alone (``services.find_trace``) is banned elsewhere in
``pyproject.toml``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.conftest import SyncApi

#: Spellings of signals of the recorded ``raw/tran`` (V1 drives ``in``, R1
#: joins ``in`` to ``out``), and the trace or pair each reads.
SPELLINGS = {
    "V(in,out)": "V(in,out)",
    "v(in, out)": "V(in,out)",
    "V(out,0)": "V(out)",
    "V(in,gnd)": "V(in)",
    "v(OUT)": "V(out)",
    "I(R1)": "I(R1)",
}


@pytest.mark.parametrize("build", rec.BUILDS)
@pytest.mark.parametrize("spelling", list(SPELLINGS))
def test_compare_reads_a_signal_as_trace_does(
    build: str, spelling: str, state_no_sim: SessionState, work_dir: Path
):
    api = SyncApi(state_no_sim)
    first = api.load_raw(rec.stage_run(build, "raw/tran", work_dir / "first"))
    second = api.load_raw(rec.stage_run(build, "raw/tran", work_dir / "second"))
    wave = first.trace(spelling)
    assert np.max(np.abs(wave)) > 0

    found = first.compare(second, signals=[spelling])

    assert list(found["signals"]) == [SPELLINGS[spelling]]
    (row,) = found["signals"].values()
    # Two copies of one run: no difference, so the first point is the worst.
    assert (row["max_abs"], row["at"]) == (0.0, first.axis()[0])
    assert row["value"] == row["other"] == wave[0]


def test_compare_measures_a_node_pair_across_the_two_builds(
    state_no_sim: SessionState, work_dir: Path
):
    """XVII stores fewer points of the transient than LTspice 26; the pair's
    difference at each of 26's points is the pair read from each run."""
    api = SyncApi(state_no_sim)
    current, xvii = (
        api.load_raw(rec.stage_run(build, "raw/tran", work_dir / build))
        for build in ("ltspice26", "ltspice17")
    )

    row = current.compare(xvii, signals=["V(in,out)"])["signals"]["V(in,out)"]

    axis = current.axis()
    (at,) = np.flatnonzero(axis == row["at"])
    assert row["value"] == pytest.approx(current.trace("V(in,out)")[at])
    theirs = np.interp(axis[at], xvii.axis(), xvii.trace("V(in,out)"))
    assert row["other"] == pytest.approx(theirs)
    assert row["max_abs"] == pytest.approx(abs(row["value"] - row["other"]))


def test_a_name_neither_result_has_is_refused_with_why(state_no_sim: SessionState, work_dir: Path):
    api = SyncApi(state_no_sim)
    raw = api.load_raw(rec.stage_run(rec.BUILDS[0], "raw/tran", work_dir))
    with pytest.raises(
        ResultError,
        match=r"Not a signal of both results: V\(in,nowhere\)\. .*'V\(nowhere\)' is not in",
    ):
        raw.compare(raw, signals=["V(in,nowhere)"])
