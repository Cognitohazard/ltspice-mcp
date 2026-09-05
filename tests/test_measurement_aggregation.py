"""The .MEAS axis swap, over a recorded stepped LTspice log.

A ``WHEN``-style measurement prints the same trigger level on every step and
the crossing time separately, so aggregating the printed value would report the
constant level three times. The aggregator has to switch to the folded ``at``
field instead. Fixture: a real 3-step LTspice run of an RC low-pass with

    .meas tran vfinal FIND V(out) AT=0.9m     (value varies per step)
    .meas tran tcross WHEN V(out)=0.5         (level constant, time varies)
"""

from pathlib import Path

import pytest

from ltspice_mcp.lib import services
from ltspice_mcp.lib.metrics import (
    aggregate_log_measurements as _aggregate_log_measurements,
)
from ltspice_mcp.lib.metrics import measurements
from ltspice_mcp.lib.recipes import MeasurementsRecipe
from ltspice_mcp.state import SessionState
from tests.conftest import (
    FIXTURES_DIR,
)

# R1 of sweep runs 0 / 1 / 2 (one LTSPICE_SWEEP_RUN_LOGS entry per run).
RUN_PARAMS = [1000.0, 2200.0, 4700.0]

# Values printed by LTspice in each per-run log (vfinal: FIND V(out) AT=0.9m).
VFINAL = [0.999876166042, 0.98323999039, 0.852486569628]
# Crossing times printed by LTspice (tcross: WHEN V(out)=0.5 -> "... AT <t>").
TCROSS_AT = [6.98285618328e-05, 0.000152998664495, 0.000326276827772]

# Stepped single-run log (3 steps in ONE log, not one log per run): vfinal is a
# FIND (value varies), tcross is a WHEN (level constant 0.5, crossing varies).
STEP_WHEN_LOG = FIXTURES_DIR / "ltspice_step_when.log"
STEP_WHEN_VFINAL = [0.63212, 0.77687, 0.95021]
STEP_WHEN_TCROSS_AT = [6.98285e-05, 1.53000e-04, 3.26000e-04]


# Verbatim ngspice-42 batch-mode log (captured from `ngspice -b -r out.raw
# -o out.log` on a .tran deck with a .meas directive): batch mode with a
# rawfile evaluates NO .meas at all, and the log itself carries the reason.
_NGSPICE_BATCH_MEAS_BLOCKED_LOG = (
    "\n"
    "Note: No compatibility mode selected!\n"
    "\n"
    "\n"
    "Circuit: * rc meas\n"
    "\n"
    'binary raw file "meas.raw"\n'
    "Doing analysis at TEMP = 27.000000 and TNOM = 27.000000\n"
    "\n"
    "Using SPARSE 1.3 as Direct Linear Solver\n"
    "No. of Data Columns : 4  \n"
    "\n"
    "No. of Data Rows : 526\n"
    "\n"
    "No .measure possible in batch mode (-b) with -r rawfile set!\n"
    "Remove rawfile and use .print or .plot or\n"
    "select interactive mode (optionally with .control section) instead.\n"
    "\n"
    "\n"
    "Total analysis time (seconds) = 0.001\n"
    "\n"
    "Total elapsed time (seconds) = 0.009 \n"
)


class TestSteppedLogWhenAxis:
    """A stepped single-run log must apply the WHEN -> ``at`` axis swap exactly
    like the batch path. Before the shared swap, this path always aggregated the
    ``value`` field, so a stepped WHEN .MEAS reported the constant trigger level
    instead of the per-step crossing times. Fixture: ltspice_step_when.log."""

    def test_stepped_log_when_swaps_to_at_axis(self):
        flat_values, axis_map, steps_label, _ = _aggregate_log_measurements(STEP_WHEN_LOG)
        # Constant level (0.5) across steps + varying crossing -> aggregate ``at``.
        assert axis_map["tcross"] == "at"
        assert flat_values["tcross"] == pytest.approx(STEP_WHEN_TCROSS_AT)
        assert "3 step(s)" in steps_label

    def test_stepped_log_find_stays_on_value_axis(self):
        flat_values, axis_map, _, _ = _aggregate_log_measurements(STEP_WHEN_LOG)
        # FIND value varies per step -> stays on the value axis.
        assert axis_map["vfinal"] == "value"
        assert flat_values["vfinal"] == pytest.approx(STEP_WHEN_VFINAL)

    @pytest.mark.asyncio
    async def test_stepped_when_log_swaps_to_at_axis_through_the_tool(
        self, state_no_sim: SessionState, work_dir: Path
    ):
        staged = work_dir / STEP_WHEN_LOG.name
        staged.write_bytes(STEP_WHEN_LOG.read_bytes())
        data = await measurements(
            services.AnalysisSource(
                raw=staged.with_suffix(".raw"),
                log=staged,
                netlist=None,
                dialect=None,
                identity=None,
                trusted_job_artifact=False,
            ),
            MeasurementsRecipe(key="m", metric="measurements"),
            0,
            state_no_sim,
        )
        tcross = data["stats"]["tcross"]
        assert tcross["aggregated_field"] == "at"
        assert tcross["min"] == pytest.approx(STEP_WHEN_TCROSS_AT[0])
        assert tcross["max"] == pytest.approx(STEP_WHEN_TCROSS_AT[2])
        assert tcross["mean"] == pytest.approx(sum(STEP_WHEN_TCROSS_AT) / 3)
        # The FIND measurement in the same log stays on the value axis. Its
        # per-step ``at`` (probe point 0.9m) must NOT be echoed: the n=1 crossing
        # echo is gated on total_count==1, and this is a 3-step aggregate.
        vfinal = data["stats"]["vfinal"]
        assert vfinal["aggregated_field"] == "value"
        assert "at" not in vfinal
