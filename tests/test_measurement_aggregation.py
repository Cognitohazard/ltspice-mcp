"""The field a .MEAS aggregate is taken over, on a recorded stepped LTspice log.

A stepped run prints a table per measurement, each row the step's result: a
WHEN's crossing under a heading of its ``expr=level`` alone, a FIND's value
beside an ``at`` column (``log/meas_step``, both builds). So both aggregate
their values. The per-run logs of a sweep (``ltspice_sweep_meas_run<n>.log``)
print a WHEN's level and its crossing after ``AT``; their constants are here
for the sweep tests that read them.
"""

import math
from pathlib import Path

import pytest

from ltspice_mcp.lib import services
from ltspice_mcp.lib.log_conditions import Producer
from ltspice_mcp.lib.log_parser import answered_measurements, measurement_cards, parse_measurements
from ltspice_mcp.lib.metrics import aggregate_log_measurements, measurements
from ltspice_mcp.lib.recipes import MeasurementsRecipe
from ltspice_mcp.state import SessionState
from tests import _ltspice_recorded as rec
from tests.ltspice_recorder import INPUTS

# R1 of sweep runs 0 / 1 / 2 (one ltspice_sweep_meas_run<n>.log per run).
RUN_PARAMS = [1000.0, 2200.0, 4700.0]

# Values printed by LTspice in each per-run log (vfinal: FIND V(out) AT=0.9m).
VFINAL = [0.999876166042, 0.98323999039, 0.852486569628]
# Crossing times printed by LTspice (tcross: WHEN V(out)=0.5 -> "... AT <t>").
TCROSS_AT = [6.98285618328e-05, 0.000152998664495, 0.000326276827772]

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


@pytest.mark.parametrize("build", rec.BUILDS)
class TestSteppedLogAggregate:
    """A stepped log's WHEN and FIND both aggregate the values their tables
    print, the WHEN's being its crossings."""

    def table(self, build: str):
        log = rec.recorded(build, "log/meas_step.log")
        producer = Producer("ltspice", rec.manifest(build)["reported_build"], None)
        return answered_measurements(
            parse_measurements(log), producer, measurement_cards(INPUTS / "log/meas_step.cir")
        )

    def test_a_stepped_when_aggregates_the_crossings_its_table_prints(self, build: str):
        flat_values, axis_map, steps_label, _ = aggregate_log_measurements(self.table(build))
        assert axis_map["s_when"] == "value"
        assert flat_values["s_when"][:2] == pytest.approx(
            [1e3 * 100e-9 * math.log(2), 2e3 * 100e-9 * math.log(2)], abs=3e-6
        )
        assert "3 step(s)" in steps_label

    def test_a_stepped_find_aggregates_its_values(self, build: str):
        table = self.table(build)
        flat_values, axis_map, _, _ = aggregate_log_measurements(table)
        assert axis_map["s_find"] == "value"
        assert flat_values["s_find"] == table["measurements"]["s_find"]["values"]

    async def test_the_recipe_aggregates_a_stepped_when_over_its_crossings(
        self, build: str, state_no_sim: SessionState, work_dir: Path
    ):
        log = work_dir / "meas_step.log"
        log.write_bytes(rec.recorded(build, "log/meas_step.log").read_bytes())
        data = await measurements(
            services.resolve_analysis_source(state_no_sim, log_file=str(log)),
            MeasurementsRecipe(key="m", metric="measurements"),
            None,
            state_no_sim,
        )
        crossing = data["stats"]["s_when"]
        assert crossing["aggregated_field"] == "value"
        # The third step never crosses: LTspice 26 prints failed, XVII 0.
        assert crossing["max"] == pytest.approx(2e3 * 100e-9 * math.log(2), abs=3e-6)
        # A FIND's point is not echoed over a stepped aggregate: the echo is
        # for a single sample.
        assert "at" not in data["stats"]["s_find"]
