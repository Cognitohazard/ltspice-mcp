"""Numerical helper metadata follows the selected resident plot's meaning."""

from dataclasses import replace

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.lib.raw_parser import (
    dc_axis_name,
    declared_type,
    detect_sim_type,
    get_step_count,
    is_ac_analysis,
    trace_unit,
)
from tests.test_decoded_raw import header


def selected(plot: DecodedPlot, wrapped: bool):
    if not wrapped:
        return plot
    first = DecodedPlot(
        header("Operating Point", [("V(first)", "voltage")], 1),
        [np.array([99.0])],
        snapshot_id="helper-example",
    )
    later = DecodedPlot(
        replace(plot.header, index=1),
        [plot.get_wave(i).copy() for i in range(plot.header.variable_count)],
        snapshot_id="helper-example",
    )
    return DecodedRaw([first, later], plot_index=1)


@pytest.mark.parametrize("wrapped", [False, True], ids=["plot", "selected-raw"])
@pytest.mark.parametrize(
    ("plot_name", "variables", "waves", "units", "axis"),
    [
        (
            "Transfer Function",
            [("v(out)/vin", "voltage"), ("v(#input_impedance)", "voltage")],
            [[2 / 3], [1500.0]],
            [None, "Ω"],
            (None, None),
        ),
        (
            "Pole-Zero Analysis",
            [("v(pole(1))", "voltage")],
            [[-1.0 + 2.0j]],
            ["s^-1"],
            (None, None),
        ),
        ("Sensitivity Analysis", [("v(out)", "voltage")], [[0.125]], [None], (None, None)),
        (
            "Sensitivity Analysis",
            [("frequency", "frequency"), ("v(out)", "voltage")],
            [[10.0, 100.0], [1.0 + 2.0j, 3.0 + 4.0j]],
            ["Hz", None],
            ("frequency", "Hz"),
        ),
        (
            "Distortion - 2nd harmonic",
            [("frequency", "frequency"), ("v(out)", "voltage")],
            [[10.0, 100.0], [1.0 + 2.0j, 3.0 + 4.0j]],
            ["Hz", "V"],
            ("frequency", "Hz"),
        ),
        (
            "DC transfer characteristic",
            [("v-sweep", "voltage"), ("v(out)", "voltage")],
            [[1.0, 0.0], [0.5, 0.0]],
            ["V", "V"],
            ("v-sweep", "V"),
        ),
    ],
    ids=["tf", "pz", "dc-sensitivity", "ac-sensitivity", "distortion", "dc"],
)
def test_helpers_use_selected_physical_facts(wrapped, plot_name, variables, waves, units, axis):
    plot = DecodedPlot(
        header(plot_name, variables, len(waves[0])),
        [np.array(wave) for wave in waves],
        snapshot_id="helper-example",
    )
    raw = selected(plot, wrapped)
    assert detect_sim_type(raw) == plot_name
    assert dc_axis_name(raw) == axis
    assert get_step_count(raw) == 1
    for (name, declared), unit in zip(variables, units, strict=True):
        assert declared_type(raw, name.upper()) == declared
        assert trace_unit(raw, name.upper()) == unit
    assert trace_unit(raw, "V(missing)") is None


@pytest.mark.parametrize("wrapped", [False, True], ids=["plot", "selected-raw"])
def test_unknown_descriptor_does_not_promote_header_words_or_voltage_labels(wrapped):
    plot = DecodedPlot(
        header("AC backup", [("V(out)", "voltage")], 1),
        [np.array([7.0])],
        snapshot_id="helper-example",
    )
    raw = selected(plot, wrapped)
    assert detect_sim_type(raw) == "Unknown"
    assert not is_ac_analysis(detect_sim_type(raw))
    assert dc_axis_name(raw) == (None, None)
    assert trace_unit(raw, "V(out)") is None


@pytest.mark.parametrize("wrapped", [False, True], ids=["plot", "raw"])
@pytest.mark.parametrize("status", ["unresolved", "mismatch"])
def test_unknown_step_boundaries_refuse_count_and_numeric_helpers(wrapped, status):
    plot = DecodedPlot(
        header(
            "Transient Analysis",
            [("time", "time"), ("V(out)", "voltage")],
            4,
            flags=("real", "stepped"),
        ),
        [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])],
        snapshot_id="helper-example",
        step_status=status,
    )
    raw = DecodedRaw([plot]) if wrapped else plot
    with pytest.raises(ResultError, match="boundaries"):
        get_step_count(raw)


@pytest.mark.parametrize("wrapped", [False, True], ids=["plot", "raw"])
def test_known_boundaries_count_without_inventing_missing_parameter_rows(wrapped):
    plot = DecodedPlot(
        header(
            "Transient Analysis",
            [("time", "time"), ("V(out)", "voltage")],
            4,
            flags=("real", "stepped"),
        ),
        [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])],
        snapshot_id="helper-example",
        step_offsets=[0, 2],
        step_status="unresolved",
    )
    raw = DecodedRaw([plot]) if wrapped else plot
    assert get_step_count(raw) == 2
    assert raw.steps is None
    assert raw.get_wave("V(out)", step=1).tolist() == [3.0, 4.0]
