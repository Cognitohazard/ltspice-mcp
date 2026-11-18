"""plot_waveform draws a real variable of an ngspice complex text plot.

This file has ngspice's ASCII shape for ``.save @m1[vth]`` in an AC run (the
parameter one number a point), with a frequency whose imaginary part is 0, so
it runs where ngspice is not installed; the live test in
``test_ngspice_e2e.py`` reads what the installed ngspice writes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ltspice_mcp.state import SessionState
from ltspice_mcp.tools.analysis import PlotWaveformInput, handle_plot_waveform
from ltspice_mcp.tools.analyze import AnalyzeResultsInput, handle_analyze_results

pytestmark = pytest.mark.asyncio

_RAW = (
    "Title: * common-source stage\n"
    "Date: Sun Nov 15 23:45:50  2026\n"
    "Plotname: AC Analysis\n"
    "Flags: complex\n"
    "No. Variables: 3\n"
    "No. Points: 3       \n"
    "Variables:\n"
    "\t0\tfrequency\tfrequency\tgrid=3\n"
    "\t1\ti(vd)\tcurrent\n"
    "\t2\tv(@m1[vth])\tvoltage\n"
    "Values:\n"
    "0\t\t1.000000000000000e+03,0.000000000000000e+00\n"
    "\t-6.846492106890288e-15,1.760408054298856e-11\n"
    "\t1.815172031040211e+00\n"
    "1\t\t1.000000000000000e+04,0.000000000000000e+00\n"
    "\t-5.354933902790428e-15,1.760408054171203e-10\n"
    "\t1.815172031040211e+00\n"
    "2\t\t1.000000000000000e+05,0.000000000000000e+00\n"
    "\t-5.300000000000000e-15,1.760000000000000e-09\n"
    "\t1.815172031040211e+00\n"
    "\n"
)


async def test_plot_waveform_draws_the_real_values_the_file_holds(
    state_no_sim: SessionState, work_dir: Path
):
    raw = work_dir / "acvth.raw"
    raw.write_bytes(_RAW.encode("ascii"))
    plotted = await handle_plot_waveform(
        PlotWaveformInput.model_validate(
            {"path": str(raw), "dialect": "ngspice", "signals": ["v(@m1[vth])"], "open": False}
        ),
        state_no_sim,
    )
    assert plotted.structured_content is not None
    [trace] = plotted.structured_content["traces"]
    # Drawn as its values, in volts, not as a magnitude in dB and a phase.
    assert (trace["signal"], trace["unit"]) == ("v(@m1[vth])", "V")
    assert trace["min"] == trace["max"] == pytest.approx(1.815172031040211, rel=1e-15)

    # analyze_results reads the same file to the same number.
    analyzed = await handle_analyze_results(
        AnalyzeResultsInput.model_validate(
            {
                "sources": [{"path": str(raw), "dialect": "ngspice"}],
                "recipes": [{"key": "vth", "metric": "value", "expr": "v(@m1[vth])", "at": "10k"}],
            }
        ),
        state_no_sim,
    )
    assert analyzed.structured_content is not None
    [read] = analyzed.structured_content["results"]["vth"]["values"]
    assert read["value"]["value"] == pytest.approx(1.815172031040211, rel=1e-15)


async def test_a_panel_of_a_complex_response_and_a_real_value_is_drawn_as_two(
    state_no_sim: SessionState, work_dir: Path
):
    # The current is a complex response, drawn as a magnitude and a phase; the
    # parameter is real, drawn as its values in a panel of its own.
    raw = work_dir / "acvth.raw"
    raw.write_bytes(_RAW.encode("ascii"))
    plotted = await handle_plot_waveform(
        PlotWaveformInput.model_validate(
            {
                "path": str(raw),
                "dialect": "ngspice",
                "panels": [["i(vd)", "v(@m1[vth])"]],
                "open": False,
            }
        ),
        state_no_sim,
    )
    assert plotted.structured_content is not None
    traces = plotted.structured_content["traces"]
    assert [(t["signal"], t["panel"], t["unit"]) for t in traces] == [
        ("i(vd)", 0, "dB"),
        ("v(@m1[vth])", 2, "V"),
    ]
