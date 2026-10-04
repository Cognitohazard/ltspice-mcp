"""Resident RAW views: recordings for compatibility, numeric native counterexamples."""

from __future__ import annotations

import builtins
import io
import os
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from spicelib.raw.raw_classes import Axis
from spicelib.raw.raw_read import RawRead

from ltspice_mcp.errors import NoAxisError, ResultError
from ltspice_mcp.lib.decoded_raw import DecodedPlot, DecodedRaw, RawData, RawTrace
from ltspice_mcp.lib.raw_header import RawLimits, RawPlotHeader, RawVariable, preflight_raw
from ltspice_mcp.lib.raw_parser import OffsetAwareRawRead

FIXTURES = Path(__file__).parent / "fixtures"
LIMITS = RawLimits(2_000_000, 64_000, 16_000, 16, 128, 10_000, 2_000_000)


def header(plot, variables, points, *, index=0, dialect="ngspice", offset="0", flags=("real",)):
    """Synthetic validated metadata; no file or dependency construction."""
    return RawPlotHeader(
        index=index,
        header_offset=0,
        encoding="utf-8",
        fields=(
            ("Title", "numeric example"),
            ("Plotname", plot),
            ("Flags", " ".join(flags)),
            ("No. Variables", str(len(variables))),
            ("No. Points", str(points)),
            ("Offset", offset),
        ),
        plot_name=plot,
        flags=flags,
        variable_count=len(variables),
        point_count=points,
        variables=tuple(
            RawVariable(i, name, kind, ()) for i, (name, kind) in enumerate(variables)
        ),
        dialect=dialect,
        dialect_evidence=("explicit",),
        storage="binary",
        storage_order="point-major",
        value_bytes=(8,) * len(variables),
        payload_offset=0,
        payload_end=0,
        payload_bytes=0,
        numeric_bytes=points * len(variables) * 8,
    )


def view(plot, variables, waves, **kwargs):
    return DecodedPlot(
        header(plot, variables, len(waves[0])), waves, snapshot_id="recorded-example", **kwargs
    )


@pytest.mark.parametrize(
    "name",
    [
        "ltspice_tran_rc.raw",
        "ltspice_ac_rc.raw",
        "ltspice_dc_div.raw",
        "op_extreme_node.raw",
        "ltspice_step_tran.raw",
        "ltspice_step_ac.raw",
    ],
)
def test_recorded_arrays_and_steps(name):
    path = FIXTURES / name
    parsed = RawRead(path, traces_to_read="*")
    (h,) = preflight_raw(path, limits=LIMITS).plots
    source = parsed.plots[0]
    arrays = [np.array(source.get_trace(v.name).data, copy=True) for v in h.variables]
    offsets = None
    if source.steps:
        axis = source.get_trace(0)
        assert isinstance(axis, Axis)
        offsets = [axis.step_offset(step) for step in source.get_steps()]
    raw = DecodedRaw(
        [
            DecodedPlot(
                h, arrays, snapshot_id="recorded-example", steps=source.steps, step_offsets=offsets
            )
        ]
    )
    resident_reader: RawData = raw
    dependency_reader: RawData = parsed
    for reader in (resident_reader, dependency_reader):
        trace: RawTrace = reader.get_trace(h.variables[-1].name)
        assert trace.name == h.variables[-1].name
        assert trace.whattype == h.variables[-1].declared_type
        np.testing.assert_array_equal(trace.get_wave(0), reader.get_wave(trace.name, 0))
    assert not isinstance(raw, RawRead)
    assert raw.get_trace_names() == [v.name for v in h.variables]
    assert raw.get_steps() == list(source.get_steps())
    assert raw.steps == source.steps
    assert raw.dialect == h.dialect
    assert raw.descriptor.plot_index == 0
    if name == "ltspice_ac_rc.raw":
        frequency = np.real(raw.get_axis())
        response = raw.get_wave("V(out)")
        # Recorded R=1 kΩ, C=159.15 nF low-pass: an independent numeric oracle.
        expected = 1 / (1 + 2j * np.pi * frequency * 1000 * 159.15e-9)
        np.testing.assert_allclose(response, expected, rtol=1e-4, atol=1e-8)
    for step in raw.get_steps():
        for v in h.variables:
            np.testing.assert_array_equal(
                raw.get_wave(v.name, step), source.get_wave(v.name, step)
            )
            assert raw.get_trace(v.name).whattype == v.declared_type
        if raw.descriptor.axis is not None:
            np.testing.assert_array_equal(raw.get_axis(step), source.get_axis(step))
        else:
            with pytest.raises(NoAxisError):
                raw.get_axis(step)


def test_recorded_noise_keeps_both_plots_and_distinct_trace_names():
    path = FIXTURES / "ngspice_noise_2plot.raw"
    # Importing OffsetAwareRawRead installs the existing bounded ASCII guard.
    parsed = OffsetAwareRawRead(path, traces_to_read="*")
    headers = preflight_raw(path, limits=LIMITS).plots
    plots = []
    for h, source in zip(headers, parsed.plots, strict=True):
        arrays = [np.array(source.get_trace(v.name).data, copy=True) for v in h.variables]
        plots.append(DecodedPlot(h, arrays, snapshot_id="recorded-noise"))
    raw = DecodedRaw(plots)
    assert raw.get_nr_plots() == 2
    assert raw.get_plot_names() == ["Noise Spectral Density Curves", "Integrated Noise"]
    assert len(raw.get_axis()) == 301
    assert raw.descriptor.traces[1].unit == "V/√Hz"
    integrated = raw.select_plot(1)
    assert integrated.get_trace_names() == [v.name for v in headers[1].variables]
    assert "frequency" not in integrated.get_trace_names()
    assert integrated.descriptor.traces[0].unit == "V"
    np.testing.assert_array_equal(integrated.get_wave(0), parsed.plots[1].get_wave(0))
    with pytest.raises(NoAxisError):
        integrated.get_axis()
    assert raw.descriptor.plot_index == 0  # Selection does not mutate other readers.


@pytest.mark.parametrize("plot", ["Noise Spectral Density Curves", "Integrated Noise"])
@pytest.mark.parametrize("name", ["V(inoise)", "I(inoise)", "inoise_spectrum", "inoise_total"])
def test_input_noise_units_require_input_source_dimensions(plot, name):
    variables = [("frequency", "frequency"), (name, "voltage"), ("V(onoise)", "voltage")]
    if plot == "Integrated Noise":
        variables = variables[1:]
    wave = np.array([1.0])
    raw = view(plot, variables, [wave.copy() for _ in variables])
    descriptor = next(trace for trace in raw.descriptor.traces if trace.name == name)
    assert descriptor.declared_type == "voltage"
    assert descriptor.unit is None
    assert descriptor.unit_evidence == "input_source_unresolved"
    output = next(trace for trace in raw.descriptor.traces if trace.name == "V(onoise)")
    assert output.unit == ("V" if plot == "Integrated Noise" else "V/√Hz")


def test_transient_offset_once_on_later_plot_and_every_axis_accessor():
    table = view("Operating Point", [("V(out)", "voltage")], [np.array([0.5])])
    h = header(
        "Transient Analysis",
        [("time", "time"), ("V(out)", "voltage")],
        5,
        index=1,
        offset="196e-6",
        flags=("real", "stepped"),
        dialect="ltspice",
    )
    times = np.array([0, -1e-6, 2e-6, 0, -3e-6])
    voltage = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    plot = DecodedPlot(
        h,
        [times, voltage],
        snapshot_id="windowed-example",
        steps=[{"R": 1000}, {"R": 1000}],
        step_offsets=[0, 3],
    )
    raw = DecodedRaw([table, plot]).select_plot(1)
    for _ in range(2):
        np.testing.assert_allclose(raw.get_axis(0), [196e-6, 197e-6, 198e-6])
        np.testing.assert_allclose(raw.get_trace("TIME").get_wave(1), [196e-6, 199e-6])
        np.testing.assert_array_equal(raw.get_wave(0, 1), raw.get_axis(1))
    np.testing.assert_array_equal(times, [0, -1e-6, 2e-6, 0, -3e-6])
    assert raw.time_offset == 196e-6
    assert raw.get_raw_property("offset") == "196e-6"
    assert raw.get_steps(R=1000) == [0, 1]
    axis = raw.descriptor.axis
    assert axis is not None
    assert axis.monotonicity == "nondecreasing"
    assert raw.descriptor.steps[1].index == 1
    np.testing.assert_array_equal(raw.get_wave("V(out)", 1), [4, 5])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int64])
def test_transient_correction_preserves_promotion_and_original_samples(dtype):
    stored = np.array([0, -2], dtype=dtype)
    expected = np.abs(stored) + 0.25
    h = header("Transient Analysis", [("time", "time")], 2, offset="0.25")
    plot = DecodedPlot(h, [stored], snapshot_id="correction")
    np.testing.assert_array_equal(plot.get_axis(), expected)
    assert plot.get_axis().dtype == expected.dtype
    np.testing.assert_array_equal(stored, [0, -2])


def test_no_io_ownership_readonly_and_detached_metadata(monkeypatch):
    wave = np.array([1.0, 2.0])
    rows = [{"temperature": 25}]
    h = header("Operating Point", [("V(out)", "voltage")], 2)

    def forbidden(*args, **kwargs):
        pytest.fail("Resident view attempted filesystem access")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(io, "open", forbidden)
    monkeypatch.setattr(os, "open", forbidden)
    raw = DecodedRaw([DecodedPlot(h, [wave], snapshot_id="resident", steps=rows)])
    assert np.shares_memory(wave, raw.get_wave(0))
    assert not wave.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        raw.get_wave(0)[0] = 3
    with pytest.raises(ValueError, match="WRITEABLE"):
        raw.get_wave(0).setflags(write=True)
    rows[0]["temperature"] = 99
    detached_rows = raw.steps
    assert detached_rows is not None
    detached_rows[0]["temperature"] = 100
    assert raw.steps == [{"temperature": 25}]
    properties = raw.raw_params
    properties["Plotname"] = "changed"
    names = properties["Variables"]
    assert isinstance(names, list)
    names.append("invented")
    assert raw.get_raw_property("PLOTNAME") == "Operating Point"
    assert raw.get_trace_names() == ["V(out)"]
    with pytest.raises(FrozenInstanceError):
        raw.descriptor.analysis = "ac"  # pyright: ignore[reportAttributeAccessIssue]


def test_native_scalar_quantities_do_not_become_axes_or_voltages():
    # Numeric facts transcribed from recorded ngspice divider TF / RLC PZ outputs.
    tf = view(
        "Transfer Function",
        [
            ("v(Transfer_function)", "voltage"),
            ("v(v1#Input_impedance)", "voltage"),
            ("v(output_impedance_at_V(out))", "voltage"),
        ],
        [np.array([2 / 3]), np.array([3000.0]), np.array([2000 / 3])],
    )
    assert [t.unit for t in tf.descriptor.traces] == [None, "Ω", "Ω"]
    assert tf.get_trace(1).whattype == "voltage"
    assert tf.get_wave(0)[0] == pytest.approx(2 / 3)
    pz = view(
        "Pole-Zero Analysis",
        [("v(pole(1))", "voltage"), ("v(pole(2))", "voltage")],
        [np.array([-500 + 866.0254037844386j]), np.array([-500 - 866.0254037844386j])],
    )
    assert [t.unit for t in pz.descriptor.traces] == ["s^-1", "s^-1"]
    assert pz.get_wave(1)[0].imag == pytest.approx(-866.0254037844386)
    sens = view("Sensitivity Analysis", [("v(r1)", "voltage")], [np.array([-2 / 9000])])
    assert sens.descriptor.analysis == "sens_dc"
    assert sens.descriptor.traces[0].unit is None
    assert sens.descriptor.traces[0].normalization == "absolute"
    for plot in [tf, pz, sens]:
        assert plot.descriptor.layout == "table"
        assert plot.descriptor.axis is None
        with pytest.raises(NoAxisError):
            plot.get_axis()


@pytest.mark.parametrize(
    ("plot", "analysis", "convention"),
    [
        ("Sensitivity Analysis", "sens_ac", "frequency"),
        ("DISTORTION - IM: f1+f2", "disto", "swept_f1"),
        ("DISTORTION - 3rd harmonic", "disto", "swept_f1"),
    ],
)
def test_complex_native_sweeps_keep_emitted_coordinates(plot, analysis, convention):
    frequencies = np.array([100, 6666.666666666667, 444444.4444444445], dtype=np.complex128)
    values = np.array([-0.0002222220000124626 + 1e-8j, 2 + 3j, 4 - 5j])
    raw = view(plot, [("frequency", "frequency"), ("v(r1_ac)", "voltage")], [frequencies, values])
    assert raw.descriptor.analysis == analysis
    axis = raw.descriptor.axis
    assert axis is not None
    assert axis.unit == "Hz"
    assert axis.convention == convention
    assert axis.real_coordinates
    assert raw.descriptor.traces[1].representation == "complex"
    assert raw.descriptor.traces[1].unit == (None if analysis == "sens_ac" else "V")
    np.testing.assert_array_equal(raw.get_axis(), frequencies)
    np.testing.assert_array_equal(raw.get_wave(1), values)


def test_unknown_plot_and_incomplete_step_metadata_stay_honest():
    raw = view(
        "Unidentified Native Result",
        [("V(out)", "voltage")],
        [np.array([5.0, 6.0])],
        step_offsets=[0, 1],
    )
    assert raw.descriptor.analysis == "unknown"
    assert raw.descriptor.axis is None
    assert raw.descriptor.traces[0].unit is None
    assert raw.descriptor.completeness == "step_metadata_missing"
    assert raw.get_steps() == [0, 1]
    assert raw.steps is None
    assert raw.get_steps(unrecorded=0) == []
    assert raw.get_wave(0, 1)[0] == 6


@pytest.mark.parametrize("status", ["unresolved", "mismatch"])
def test_unresolved_steps_preserve_inventory_and_refuse_selection(status):
    h = header(
        "Transient Analysis",
        [("time", "time"), ("V(out)", "voltage")],
        4,
        flags=("real", "stepped"),
        dialect="ltspice",
    )
    waves = [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0])]
    plot = DecodedPlot(h, waves, snapshot_id="unresolved", step_status=status)
    assert plot.descriptor.step_status == status
    assert plot.get_trace_names() == ["time", "V(out)"]
    assert plot.header.point_count == 4
    assert plot.descriptor.steps == ()
    for select in [
        plot.get_steps,
        plot.get_axis,
        lambda: plot.get_wave(1),
        plot.get_trace(1).get_wave,
    ]:
        with pytest.raises(ResultError, match="boundaries"):
            select()
    np.testing.assert_array_equal(waves[1], [1, 2, 3, 4])


def test_stepped_header_without_ranges_does_not_invent_one_step():
    h = header("Operating Point", [("V(out)", "voltage")], 2, flags=("real", "stepped"))
    plot = DecodedPlot(h, [np.array([0.5, 0.25])], snapshot_id="unresolved")
    assert plot.descriptor.step_status == "unresolved"
    with pytest.raises(ResultError, match="boundaries"):
        plot.get_steps()


@pytest.mark.parametrize("flag", ["Stepped", "STEPPED", "sTePpEd"])
def test_mixed_case_stepped_flag_preserves_unresolved_status(flag):
    h = header("Operating Point", [("V(out)", "voltage")], 2, flags=("real", flag))
    plot = DecodedPlot(h, [np.array([0.5, 0.25])], snapshot_id="mixed-case")
    assert plot.descriptor.step_status == "unresolved"
    assert plot.header.flags == ("real", flag)
    with pytest.raises(ResultError, match="boundaries"):
        plot.get_steps()
    with pytest.raises(ValueError, match="contradicts"):
        DecodedPlot(h, [np.array([0.5, 0.25])], snapshot_id="mixed-case", step_status="unstepped")


@pytest.mark.parametrize("key", ["offset", "OFFSET", "oFfSeT"])
def test_mixed_case_offset_field_rebases_axis_without_changing_header(key):
    h = header("Transient Analysis", [("time", "time")], 2, offset="196e-6")
    h = replace(
        h, fields=tuple((key if name == "Offset" else name, value) for name, value in h.fields)
    )
    plot = DecodedPlot(h, [np.array([0.0, -2e-6])], snapshot_id="mixed-case")
    np.testing.assert_allclose(plot.get_axis(), [196e-6, 198e-6])
    assert plot.time_offset == 196e-6
    assert plot.get_raw_property("Offset") == "196e-6"
    assert dict(plot.header.fields)[key] == "196e-6"


def test_known_boundaries_without_log_binding_keep_steps_with_mismatch_status():
    plot = view(
        "Operating Point",
        [("V(out)", "voltage")],
        [np.array([0.5, 0.25])],
        step_offsets=[0, 1],
        step_status="mismatch",
    )
    assert plot.descriptor.step_status == "mismatch"
    assert plot.get_steps() == [0, 1]
    assert plot.get_steps(R=1000) == []
    assert plot.get_wave(0, 1)[0] == 0.25


@pytest.mark.parametrize(
    ("wave", "real", "monotonicity"),
    [
        (np.array([3.0, 2.0, 1.0]), True, "nonincreasing"),
        (np.array([1.0, 3.0, 2.0]), True, "nonmonotonic"),
        (np.array([1.0, 1.0, 1.0]), True, "constant"),
        (np.array([1 + 1j, 2 + 0j]), False, "unknown"),
        (np.array([1.0, np.inf]), False, "unknown"),
    ],
)
def test_axis_facts_relay_unmodified_complex_and_nonfinite_values(wave, real, monotonicity):
    raw = view("AC Analysis", [("frequency", "frequency")], [wave])
    axis = raw.descriptor.axis
    assert axis is not None
    assert axis.real_coordinates == real
    assert axis.monotonicity == monotonicity
    np.testing.assert_array_equal(raw.get_axis(), wave)


@pytest.mark.parametrize("offset", ["not-a-number", "inf", "nan"])
def test_invalid_transient_offset_never_silently_becomes_zero(offset):
    h = header("Transient Analysis", [("time", "time")], 1, offset=offset)
    with pytest.raises(ValueError, match=r"finite|convert"):
        DecodedPlot(h, [np.array([0.0])], snapshot_id="invalid")


@pytest.mark.parametrize(
    ("status", "rows", "offsets", "flags"),
    [
        ("invented", None, None, ("real",)),
        ("unstepped", None, [0, 1], ("real",)),
        ("unstepped", None, None, ("real", "stepped")),
        ("matched", None, [0, 1], ("real", "stepped")),
        ("unresolved", [{"R": 1}], None, ("real", "stepped")),
    ],
)
def test_contradictory_step_status_refuses(status, rows, offsets, flags):
    h = header("Operating Point", [("V(out)", "voltage")], 2, flags=flags)
    with pytest.raises(ValueError, match=r"status|metadata"):
        DecodedPlot(
            h,
            [np.array([1.0, 2.0])],
            snapshot_id="invalid",
            steps=rows,
            step_offsets=offsets,
            step_status=status,
        )


@pytest.mark.parametrize(
    ("offsets", "rows"),
    [
        ([1], None),
        ([0, 0], None),
        ([0, 2], None),
        ([0, -1], None),
        ([0, 1.5], None),
        ([0, 1], [{"R": 1}]),
        (None, [{"R": 1}, {"R": 2}]),
    ],
)
def test_invalid_step_boundaries_refuse(offsets, rows):
    with pytest.raises(ValueError, match="Step"):
        view(
            "Operating Point",
            [("V(out)", "voltage")],
            [np.array([1.0, 2.0])],
            step_offsets=offsets,
            steps=rows,
        )


@pytest.mark.parametrize(
    "wave", [np.array([object(), object()]), np.zeros((2, 1)), np.array([1.0])]
)
def test_invalid_resident_array_refuses(wave):
    with pytest.raises(ValueError, match="Resident arrays"):
        DecodedPlot(header("Operating Point", [("V(out)", "voltage")], 2), [wave], snapshot_id="x")


def test_lookup_selection_and_step_ranges_refuse_bad_identity():
    plot = view("Operating Point", [("V(out)", "voltage")], [np.array([1.0])])
    raw = DecodedRaw([plot])
    assert raw.get_trace("v(OUT)").name == "V(out)"
    for key in [-1, 1, "missing"]:
        with pytest.raises(IndexError):
            raw.get_trace(key)
    for step in [-1, 1]:
        with pytest.raises(IndexError):
            raw.get_wave(0, step)
    with pytest.raises(ValueError, match="RAW property"):
        raw.get_raw_property("missing")
    with pytest.raises(IndexError):
        raw.select_plot(-1)
    with pytest.raises(ValueError, match="inventory"):
        DecodedRaw([])
    with pytest.raises(ValueError, match="inventory"):
        DecodedRaw(
            [plot, DecodedPlot(replace(plot.header, index=2), [np.array([1.0])], snapshot_id="x")]
        )
