"""Reading a result by the step it belongs to, and naming exactly one source.

Two rules that used to live in the removed per-metric tools and now sit on the
paths that survived them: a summary reflects the ``.step`` point it was asked
for (it once hardcoded step 0, so a stepped run always reported the first
step's range), and a direct read must name a raw or a job, never both and never
neither — with an empty or whitespace path counting as neither.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.metrics import summary
from ltspice_mcp.lib.recipes import SummaryRecipe
from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, _direct_source
from tests.test_analysis_tools import _source


def _inject_raw(state: SessionState, path: Path, raw: MagicMock) -> None:
    path.write_bytes(b"placeholder")
    state.results.set(path, raw)


def _stepped_tran_raw() -> MagicMock:
    """A 2-step transient raw whose step 0 and step 1 have different time spans."""
    raw = MagicMock()
    raw.get_raw_property.return_value = "Transient Analysis"
    raw.get_trace_names.return_value = ["time", "V(out)"]
    raw.get_steps.return_value = [0, 1]
    raw.get_axis.side_effect = lambda step=0: (
        np.array([0.0, 1.0]) if step == 0 else np.array([0.0, 5.0])
    )
    return raw


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

    Truthiness, not identity: an empty or whitespace path is stripped to ``""``
    by the input model, and must count as absent — otherwise it slips past and
    resolves to the working directory, which fails later with an error about
    the wrong thing.
    """

    @pytest.mark.parametrize(
        ("raw_file", "job_id"),
        [("x.raw", "b1"), (None, None), ("", None)],
        ids=["both", "neither", "empty"],
    )
    def test_refused(self, state_no_sim: SessionState, raw_file: str | None, job_id: str | None):
        with pytest.raises(ResultError, match="exactly one"):
            _direct_source(raw_file, job_id, state_no_sim)

    @pytest.mark.parametrize("spelling", ["", "  "])
    def test_a_blank_path_arrives_as_absent(self, state_no_sim: SessionState, spelling: str):
        # The input model strips whitespace, so a blank path reaches the
        # resolver as "" — which the check above reads as no source at all.
        args = PlotWaveformInput(raw_file=spelling)
        with pytest.raises(ResultError, match="exactly one"):
            _direct_source(args.raw_file, args.job_id, state_no_sim)
