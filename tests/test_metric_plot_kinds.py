"""Sampled measurements must not assign time semantics to native quantities."""

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw
from ltspice_mcp.lib.metrics import (
    classify_analysis,
    guarded_axis,
    noise_trace_unit,
    reject_non_transient,
    unverified_input_noise_warning,
)
from ltspice_mcp.lib.raw_parser import build_simulation_summary
from tests.test_decoded_raw import header


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("Sensitivity Analysis", "sens_ac"),
        ("Distortion - 2nd harmonic", "disto"),
    ],
)
def test_native_frequency_axes_keep_their_analysis_kind(name, kind):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    name,
                    [("frequency", "frequency"), ("v(out)", "voltage")],
                    2,
                    flags=("complex",),
                ),
                [np.array([1 + 0j, 10 + 0j]), np.array([2 + 1j, 4 + 2j])],
                snapshot_id="native-frequency",
            )
        ]
    )
    assert classify_analysis(raw) == (name, kind, "Hz", True)
    with pytest.raises(ResultError, match="transient"):
        reject_non_transient(raw)


def test_native_table_has_no_time_axis_or_operating_point_hint():
    raw = DecodedRaw(
        [
            DecodedPlot(
                header("Pole-Zero Analysis", [("pole(1)", "frequency")], 1, flags=("complex",)),
                [np.array([-2 + 3j])],
                snapshot_id="native-table",
            )
        ]
    )
    assert classify_analysis(raw) == ("Pole-Zero Analysis", "pz", "", False)
    with pytest.raises(ResultError, match="table") as failure:
        guarded_axis(raw, 0)
    assert "operating_point" not in str(failure.value)


def test_noise_integration_uses_amplitude_unit_once():
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "Noise Spectral Density",
                    [("frequency", "frequency"), ("onoise_spectrum", "voltage")],
                    2,
                ),
                [np.array([1.0, 10.0]), np.array([2e-9, 2e-9])],
                snapshot_id="noise-units",
            )
        ]
    )
    assert raw.descriptor.traces[1].unit == "V/√Hz"
    assert noise_trace_unit(raw, "onoise_spectrum", None) == ("V", True)


def test_unknown_input_noise_unit_is_not_reported_as_an_assumption():
    warning = unverified_input_noise_warning(None)
    assert "unknown" in warning
    assert "assuming" not in warning


@pytest.mark.parametrize("name", ["Operating Point", "Pole-Zero Analysis", "Transfer Function"])
def test_summary_accepts_a_decoded_native_table_without_an_axis(name):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(name, [("v(out)", "voltage")], 1),
                [np.array([2.0])],
                snapshot_id="summary-table",
            )
        ]
    )
    result = build_simulation_summary(raw, None, value_scan=True)
    assert result["range"] == {}
    assert result["point_count"] == 1
    assert result["signals"] == ["v(out)"]
    assert "warnings" not in result


@pytest.mark.parametrize("name", ["Sensitivity Analysis", "Distortion - 2nd harmonic"])
def test_summary_uses_native_frequency_coordinates(name):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    name,
                    [("frequency", "frequency"), ("v(out)", "voltage")],
                    2,
                    flags=("complex",),
                ),
                [np.array([10 + 0j, 100 + 0j]), np.array([2 + 1j, 4 + 2j])],
                snapshot_id="summary-frequency",
            )
        ]
    )
    result = build_simulation_summary(raw, None)
    assert result["range"] == {"freq_start": 10.0, "freq_end": 100.0}


@pytest.mark.parametrize("imaginary", [0.0, 7.0])
def test_axis_guard_preserves_real_coordinate_requirement(imaginary):
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "AC Analysis",
                    [("frequency", "frequency"), ("v(out)", "voltage")],
                    2,
                    flags=("complex",),
                ),
                [
                    np.array([100 + imaginary * 1j, 200 + imaginary * 1j]),
                    np.array([1 + 0j, 2 + 0j]),
                ],
                snapshot_id="complex-axis",
            )
        ]
    )
    if imaginary:
        with pytest.raises(ResultError, match=r"real|imaginary|complex"):
            guarded_axis(raw, 0)
    else:
        np.testing.assert_array_equal(guarded_axis(raw, 0), [100.0, 200.0])


def test_axis_guard_leaves_missing_real_samples_for_existing_accounting():
    raw = DecodedRaw(
        [
            DecodedPlot(
                header(
                    "AC Analysis",
                    [("frequency", "frequency"), ("v(out)", "voltage")],
                    2,
                    flags=("complex",),
                ),
                [np.array([complex(float("nan"), 0), 200 + 0j]), np.array([1 + 0j, 2 + 0j])],
                snapshot_id="nonfinite-axis",
            )
        ]
    )
    np.testing.assert_array_equal(guarded_axis(raw, 0), [float("nan"), 200.0])
