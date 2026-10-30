"""Reading a result by the step it belongs to, and naming exactly one source.

Two rules that used to live in the removed per-metric tools and now sit on the
paths that survived them: a summary reflects the ``.step`` point it was asked
for (it once hardcoded step 0, so a stepped run always reported the first
step's range), and a direct read must name a raw or a job, never both and never
neither — with an empty or whitespace path counting as neither.
"""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.metrics import summary
from ltspice_mcp.lib.recipes import SummaryRecipe
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, _direct_source
from tests.conftest import inject_numeric_raw, make_raw_mock
from tests.test_analysis_tools import _source

pytestmark = pytest.mark.usefixtures("numeric_raw_inputs")


def _inject_raw(state: SessionState, path: Path, raw) -> None:
    inject_numeric_raw(state, path, raw)


def _stepped_tran_raw():
    """A 2-step transient raw whose step 0 and step 1 have different time spans."""
    return make_raw_mock(
        ["time", "V(out)"], waves={}, step_axes=[np.array([0.0, 1.0]), np.array([0.0, 5.0])]
    )


@pytest.mark.asyncio
class TestSimulationSummaryStepAware:
    async def test_summary_reflects_chosen_step(self, state_no_sim: SessionState, work_dir: Path):
        path = work_dir / "stepped.raw"
        _inject_raw(state_no_sim, path, _stepped_tran_raw())
        recipe = SummaryRecipe(key="s", metric="summary")
        source = _source(state_no_sim, "stepped.raw")
        first = await summary(source, recipe, 0, state_no_sim)
        second = await summary(source, recipe, 1, state_no_sim)
        assert first["range"]["time_end"] == 1.0
        assert second["range"]["time_end"] == 5.0  # step 1, not step 0

    async def test_summary_out_of_range_step_rejected(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        path = work_dir / "stepped2.raw"
        _inject_raw(state_no_sim, path, _stepped_tran_raw())
        with pytest.raises(ResultError, match="out of range"):
            await summary(
                _source(state_no_sim, "stepped2.raw"),
                SummaryRecipe(key="s", metric="summary"),
                9,
                state_no_sim,
            )


class TestExactlyOneSource:
    """A direct read names a raw path OR a job, never both and never neither.

    An empty or whitespace path is stripped to ``""`` by the input model and
    refused there, so it never reaches the resolver, where the working
    directory it resolves to would fail later with an error about the wrong
    thing.
    """

    @pytest.mark.parametrize(
        ("path", "job_id"),
        [("x.raw", "b1"), (None, None)],
        ids=["both", "neither"],
    )
    def test_refused(self, state_no_sim: SessionState, path: str | None, job_id: str | None):
        with pytest.raises(ResultError, match="exactly one"):
            _direct_source(path, job_id, state_no_sim)

    @pytest.mark.parametrize("spelling", ["", "  "])
    def test_a_blank_path_is_refused(self, spelling: str):
        with pytest.raises(ValidationError, match="path"):
            PlotWaveformInput(path=spelling)
