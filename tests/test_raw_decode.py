"""Worker RAW materialization against recorded files and synthetic corruption."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from ltspice_mcp.errors import ResultError
from ltspice_mcp.lib import raw_decode
from ltspice_mcp.lib.parser_capture import CapturedInputs, SourceFiles, capture_inputs
from ltspice_mcp.lib.parser_protocol import read_decoded_raw
from ltspice_mcp.lib.raw_decode import RawDecodeError, decode_raw
from ltspice_mcp.lib.raw_header import RawHeaderError, RawLimitError, RawLimits, preflight_raw
from ltspice_mcp.lib.store import Store, parser_file_in

FIXTURES = Path(__file__).parent / "fixtures"
LIMITS = RawLimits(2_000_000, 64_000, 16_000, 16, 128, 10_000, 2_000_000)


def synthetic_plot(
    rows=((0.0, 1.0), (1.0, 2.0)),
    *,
    names=("time", "V(out)"),
    types=("time", "voltage"),
    command="ngspice-42",
    flags="real",
    plot="Transient Analysis",
    widths=(8, 8),
    encoding="utf-8",
    bom=False,
    newline="\n",
    text=False,
    offset="0",
):
    """Synthetic numeric output; not evidence of a simulator's emission."""
    header = (
        f"Title: synthetic\nDate: synthetic\nPlotname: {plot}\nFlags: {flags}\n"
        f"No. Variables: {len(names)}\nNo. Points: {len(rows)}\n"
        f"Offset: {offset}\nCommand: {command}\nVariables:\n"
        + "".join(
            f"\t{i}\t{name}\t{declared}\n"
            for i, (name, declared) in enumerate(zip(names, types, strict=True))
        )
        + ("Values:\n" if text else "Binary:\n")
    ).replace("\n", newline)
    prefix = (
        {"utf-8": b"\xef\xbb\xbf", "utf-16-le": b"\xff\xfe", "utf-16-be": b"\xfe\xff"}[encoding]
        if bom
        else b""
    )
    if text:
        lines = []
        for point, row in enumerate(rows):
            for index, value in enumerate(row):
                token = f"{value.real},{value.imag}" if widths[index] == 16 else str(value)
                lines.append((f"{point}\t\t" if index == 0 else "\t") + token + newline)
        body = "".join(lines).encode(encoding)
    else:
        columns = zip(*rows, strict=True) if "fastaccess" in flags else rows
        body = bytearray()
        for index, values in enumerate(columns):
            for column, value in enumerate(values):
                width = widths[index if "fastaccess" in flags else column]
                if width == 16:
                    body.extend(struct.pack("<dd", complex(value).real, complex(value).imag))
                else:
                    body.extend(struct.pack("<d" if width == 8 else "<f", value))
        body = bytes(body)
    return prefix + header.encode(encoding) + body


def captured(
    tmp_path: Path, data: bytes, *, log: bytes | None = None, console: bytes | None = None
):
    raw = tmp_path / "source.raw"
    raw.write_bytes(data)
    log_path = tmp_path / "source.log"
    console_path = tmp_path / "source.exe.log"
    if log is not None:
        log_path.write_bytes(log)
    if console is not None:
        console_path.write_bytes(console)
    directory = Store(tmp_path).parser_dir("decode_1")
    directory.mkdir(parents=True)
    inputs = capture_inputs(
        SourceFiles(
            raw,
            log_path if log is not None else None,
            console_path if console is not None else None,
        ),
        directory,
        input_bytes=LIMITS.input_bytes,
        log_bytes=LIMITS.input_bytes,
    )
    return inputs, directory


def decode(inputs, directory, **kwargs):
    return decode_raw(
        inputs, directory, limits=LIMITS, step_log_bytes=64_000, step_rows=128, **kwargs
    )


def arrays(result, directory):
    loaded = {}
    for item in result["arrays"]:
        path = parser_file_in(directory, item["file"])
        data = path.read_bytes()
        assert item["file"] == f"p{item['plot_index']}_t{item['trace_index']}.bin"
        assert item["dtype"] in ("<f4", "<f8", "<c16")
        assert len(data) == item["byte_size"] == item["count"] * np.dtype(item["dtype"]).itemsize
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
        loaded[item["plot_index"], item["trace_index"]] = np.fromfile(
            path, dtype=item["dtype"], count=item["count"]
        )
    assert sum(item["byte_size"] for item in result["arrays"]) == result["header"]["numeric_bytes"]
    assert set(result) == {"header", "arrays", "plots", "step_log"}
    json.dumps(result, allow_nan=False)
    return loaded


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
@pytest.mark.parametrize("bom", [False, True])
@pytest.mark.parametrize("text", [False, True])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_encoding_materializes_exact_values(tmp_path, encoding, bom, text, newline):
    inputs, directory = captured(
        tmp_path, synthetic_plot(encoding=encoding, bom=bom, text=text, newline=newline)
    )
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    np.testing.assert_array_equal(loaded[0, 0], [0.0, 1.0])
    np.testing.assert_array_equal(loaded[0, 1], [1.0, 2.0])
    assert result["header"]["plots"][0]["encoding"] == encoding
    assert result["step_log"]["status"] == "absent"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le"])
@pytest.mark.parametrize("field_case", ["lower", "upper"])
@pytest.mark.parametrize("flag_case", ["upper", "mixed"])
@pytest.mark.parametrize(
    "layout", ["real", "real double", "real fastaccess", "complex fastaccess", "real stepped"]
)
def test_admitted_field_and_flag_case_variants(tmp_path, encoding, field_case, flag_case, layout):
    complex_values = "complex" in layout
    stepped = "stepped" in layout
    if complex_values:
        rows = ((100 + 0j, 1 - 2j), (200 + 0j, 3 + 4j))
        widths = (16, 16)
        names, types = ("frequency", "V(out)"), ("frequency", "voltage")
    else:
        rows = (
            ((5.0, 1.5), (6.0, -2.5), (5.0, 3.5), (7.0, 4.5))
            if stepped
            else ((0.0, 1.5), (-1.0, -2.5))
        )
        widths = (8, 8) if "double" in layout else (8, 4)
        names, types = ("time", "V(out)"), ("time", "voltage")
    data = synthetic_plot(
        rows,
        command="LTspice",
        flags=layout,
        widths=widths,
        encoding=encoding,
        names=names,
        types=types,
        plot="AC Analysis" if complex_values else "Transient Analysis",
    )
    original_flags = layout.upper() if flag_case == "upper" else layout.title()
    data = data.replace(
        f"Flags: {layout}".encode(encoding), f"Flags: {original_flags}".encode(encoding)
    )
    for field in ["Date", "Plotname", "Flags", "No. Variables", "No. Points", "Offset", "Command"]:
        changed = field.lower() if field_case == "lower" else field.upper()
        data = data.replace(f"{field}:".encode(encoding), f"{changed}:".encode(encoding))
    # Keep Variables: and counts canonical: this exercises the dependency's
    # native case handling rather than the normalization scratch route.
    inputs, directory = captured(
        tmp_path, data, log=b".step r=1\n.step r=2\n" if stepped else None
    )
    path = parser_file_in(directory, "input.raw")
    header = preflight_raw(path, limits=LIMITS)
    result = decode(inputs, directory, preflight=header)
    loaded = arrays(result, directory)
    np.testing.assert_array_equal(loaded[0, 0], [row[0] for row in rows])
    np.testing.assert_array_equal(loaded[0, 1], [row[1] for row in rows])
    assert result["header"] == asdict(header)
    assert result["header"]["plots"][0]["flags"] == tuple(original_flags.split())
    assert path.read_bytes() == data
    assert not list(directory.glob("*_decode.raw"))
    assert result["plots"][0]["step_status"] == ("matched" if stepped else "unstepped")


def test_mixed_encodings_and_plot_specific_names(tmp_path):
    first = synthetic_plot()
    second = synthetic_plot(
        ((100 + 7j, -2 + 3j),),
        names=("frequency", "v(pole(1))"),
        types=("frequency", "voltage"),
        flags="complex",
        widths=(16, 16),
        plot="Pole-Zero Analysis",
        encoding="utf-16-be",
        text=True,
        bom=True,
    )
    third = synthetic_plot(
        ((3.0,),),
        names=("transfer_function",),
        types=("transfer",),
        widths=(8,),
        plot="Transfer Function",
        encoding="utf-16-le",
    )
    inputs, directory = captured(tmp_path, first + second + third)
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    assert len(loaded) == 5
    np.testing.assert_array_equal(loaded[1, 1], [-2 + 3j])
    np.testing.assert_array_equal(loaded[2, 0], [3.0])
    assert [p["decoder_has_axis"] for p in result["plots"]] == [True, True, False]
    assert [p["plot_name"] for p in result["header"]["plots"]] == [
        "Transient Analysis",
        "Pole-Zero Analysis",
        "Transfer Function",
    ]


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16-le", "utf-16-be"])
def test_original_unicode_trace_names_and_types_are_preserved(tmp_path, encoding):
    names = ("time", "V(温度😀)")
    types = ("time", "voltage_µ😀")
    inputs, directory = captured(
        tmp_path, synthetic_plot(names=names, types=types, encoding=encoding)
    )
    result = decode(inputs, directory)
    assert tuple(v["name"] for v in result["header"]["plots"][0]["variables"]) == names
    assert tuple(v["declared_type"] for v in result["header"]["plots"][0]["variables"]) == types
    loaded = arrays(result, directory)
    np.testing.assert_array_equal(loaded[0, 1], [1.0, 2.0])


def test_trusted_preflight_is_reused_and_serialized_unchanged(tmp_path, monkeypatch):
    inputs, directory = captured(tmp_path, synthetic_plot(text=True))
    header = preflight_raw(parser_file_in(directory, "input.raw"), limits=LIMITS)

    def no_second_scan(*args, **kwargs):
        pytest.fail("supplied preflight must avoid a second ASCII walk")

    monkeypatch.setattr(raw_decode, "preflight_raw", no_second_scan)
    result = decode(inputs, directory, preflight=header)
    assert result["header"] == asdict(header)
    arrays(result, directory)


def test_supplied_preflight_does_not_skip_capture_digest_verification(tmp_path):
    inputs, directory = captured(tmp_path, synthetic_plot())
    path = parser_file_in(directory, "input.raw")
    header = preflight_raw(path, limits=LIMITS)
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(RawDecodeError, match="digest"):
        decode(inputs, directory, preflight=header)
    assert not list(directory.glob("*.bin"))


def test_supplied_preflight_size_must_match_capture(tmp_path):
    inputs, directory = captured(tmp_path, synthetic_plot())
    header = preflight_raw(parser_file_in(directory, "input.raw"), limits=LIMITS)
    with pytest.raises(RawDecodeError, match=r"preflight.*size"):
        decode(inputs, directory, preflight=replace(header, size_bytes=header.size_bytes + 1))


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16-le", "utf-16-be"])
def test_companion_step_log_encodings(tmp_path, encoding):
    prefix = (
        b"\xff\xfe" if encoding == "utf-16-le" else b"\xfe\xff" if encoding == "utf-16-be" else b""
    )
    log = prefix + ".step r=1\r\n".encode(encoding)
    inputs, directory = captured(
        tmp_path, synthetic_plot(command="LTspice", flags="real stepped", widths=(8, 4)), log=log
    )
    result = decode(inputs, directory)
    assert result["plots"][0]["step_status"] == "matched"
    assert result["step_log"]["rows"][0]["parameters"] == [
        {"name": "r", "token": "1", "value": 1.0}
    ]


def test_raw_alias_directives_refused_without_expression_evaluation(tmp_path):
    data = synthetic_plot().replace(
        b"Variables:\n", b".alias: V(out) dangerous_expression\nVariables:\n"
    )
    inputs, directory = captured(tmp_path, data)
    with pytest.raises(RawDecodeError, match="alias/parameter"):
        decode(inputs, directory)
    assert not list(directory.glob("*.bin"))


def test_raw_step_range_count_is_bounded_without_log(tmp_path):
    rows = ((0.0, 1.0), (1.0, 2.0), (0.0, 3.0))
    inputs, directory = captured(
        tmp_path, synthetic_plot(rows, command="LTspice", flags="real stepped", widths=(8, 4))
    )
    with pytest.raises(RawLimitError, match=r"step_rows.*RAW"):
        decode_raw(inputs, directory, limits=LIMITS, step_log_bytes=64000, step_rows=1)


@pytest.mark.parametrize(
    "name",
    [
        "ltspice_tran_rc.raw",
        "ltspice_ac_rc.raw",
        "ltspice_dc_div.raw",
        "ltspice_noise_rc.raw",
        "op_extreme_node.raw",
        "ltspice_step_tran.raw",
        "ltspice_step_ac.raw",
    ],
)
def test_recorded_binary_payload_is_preserved_byte_for_byte(tmp_path, name):
    data = (FIXTURES / name).read_bytes()
    inputs, directory = captured(tmp_path, data)
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    (header,) = result["header"]["plots"]
    widths = header["value_bytes"]
    expected = bytearray()
    for point in range(header["point_count"]):
        for trace in range(len(widths)):
            expected.extend(loaded[0, trace][point : point + 1].tobytes())
    marker = "Binary:\n".encode("utf-16-le")
    assert bytes(expected) == data[data.index(marker) + len(marker) :]
    # Native binary headers/records need no normalized full-file copy.
    assert not list(directory.glob("*_decode.raw"))
    assert all(p["data_convention"] == "stored" for p in result["plots"])


def test_recorded_ngspice_ascii_preserves_both_plot_quantities(tmp_path):
    inputs, directory = captured(tmp_path, (FIXTURES / "ngspice_noise_2plot.raw").read_bytes())
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    assert len(loaded) == 5
    assert len(loaded[0, 0]) == 301
    assert loaded[0, 0][0] == 20.0
    assert len(loaded[1, 0]) == len(loaded[1, 1]) == 1
    assert loaded[1, 0][0] > 0
    assert result["header"]["plots"][1]["variables"][0]["name"] == "v(onoise_total)"


@pytest.mark.parametrize("kind", ["ascii", "binary"])
def test_preserved_ngspice_native_multi(tmp_path, kind):
    root = os.environ.get("LTSPICE_MCP_TEST_ANALYSIS_RAW_DIR")
    if root is None:
        pytest.skip("preserved ngspice recordings were not supplied")
    inputs, directory = captured(
        tmp_path, (Path(root) / "multi" / kind / "result.raw").read_bytes()
    )
    result = decode(inputs, directory, producing_dialect="ngspice")
    loaded = arrays(result, directory)
    assert [p["variable_count"] for p in result["header"]["plots"]] == [4, 3, 1, 3, 25]
    np.testing.assert_allclose(loaded[2, 0], [-1500 + 0j])
    np.testing.assert_allclose(loaded[3, 0], [2 / 3])
    assert loaded[4, 18][0] == pytest.approx(-0.0006666660000373876)


@pytest.mark.parametrize("name", ["ltspice_step_ac", "ltspice_step_tran"])
def test_recorded_ltspice_steps_and_full_lengths(tmp_path, name):
    inputs, directory = captured(
        tmp_path,
        (FIXTURES / f"{name}.raw").read_bytes(),
        log=(FIXTURES / f"{name}.log").read_bytes(),
    )
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    (plot,) = result["plots"]
    ranges = plot["step_ranges"]
    assert plot["step_status"] == "matched"
    assert len(ranges) == len(result["step_log"]["rows"]) == 3
    assert [r["log_row"] for r in ranges] == [0, 1, 2]
    assert sum(r["length"] for r in ranges) == len(loaded[0, 0])
    assert [r["offset"] for r in ranges] == [
        0,
        ranges[0]["length"],
        ranges[0]["length"] + ranges[1]["length"],
    ]
    if name.endswith("ac"):
        assert [r["length"] for r in ranges] == [81, 81, 81]
    for part in ranges:
        stored = loaded[0, 0][part["offset"] : part["offset"] + part["length"]]
        coordinate = np.abs(stored) if name.endswith("tran") else stored.real
        assert np.all(np.isfinite(coordinate))
        assert np.all(coordinate[1:] > coordinate[:-1])


def test_step_log_mismatch_does_not_invent_data(tmp_path):
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(
            ((0.0, 1.0), (1.0, 2.0), (0.0, 3.0), (2.0, 4.0)),
            command="LTspice",
            flags="real stepped",
            widths=(8, 4),
        ),
        log=b".step r=1\n.step r=2\n.step r=3\n",
    )
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    assert len(loaded[0, 1]) == 4
    assert result["plots"][0]["step_status"] == "mismatch"
    assert len(result["plots"][0]["step_ranges"]) == 2
    assert all(r["log_row"] is None for r in result["plots"][0]["step_ranges"])
    assert len(result["step_log"]["rows"]) == 3


def test_missing_step_log_is_unresolved_not_dependency_fallback_failure(tmp_path):
    data = synthetic_plot(
        ((0.0, 1.0), (1.0, 2.0), (0.0, 3.0)),
        command="LTspice",
        flags="real stepped",
        widths=(8, 4),
    )
    inputs, directory = captured(tmp_path, data)
    result = decode(inputs, directory)
    assert result["plots"][0]["step_status"] == "unresolved"
    assert result["plots"][0]["step_ranges"] is None


def test_step_rows_keep_tokens_duplicates_and_nonfinite_values(tmp_path):
    log = "Circuit: synthetic\r\n.step temp=-40° r=5k q=nan\r\n.step temp=-40° r=5k q=nan\r\n.step unsupported token\r\n".encode(
        "cp1252"
    )
    inputs, directory = captured(
        tmp_path, synthetic_plot(command="LTspice", widths=(8, 4)), log=log
    )
    result = decode(inputs, directory)
    rows = result["step_log"]["rows"]
    assert result["step_log"]["encoding"] == "cp1252"
    assert [r["ordinal"] for r in rows] == [0, 1, 2]
    assert [r["line_number"] for r in rows] == [2, 3, 4]
    assert rows[0]["parameters"] == [
        {"name": "temp", "token": "-40°", "value": -40.0},
        {"name": "r", "token": "5k", "value": None},
        {"name": "q", "token": "nan", "value": None},
    ]
    assert rows[0]["parameters"] == rows[1]["parameters"]
    assert rows[2]["parameters"] == []
    json.dumps(result, allow_nan=False)


def test_multiplot_step_count_equality_does_not_bind_log_rows(tmp_path):
    data = synthetic_plot(command="LTspice", flags="real stepped", widths=(8, 4))
    inputs, directory = captured(tmp_path, data * 2, log=b".step r=1\n")
    result = decode(inputs, directory)
    assert all(p["step_status"] == "unresolved" for p in result["plots"])
    assert all(p["step_ranges"] is None for p in result["plots"])


def assert_inventory_only(result, directory):
    metadata = {"version": 1, "status": "ok", "cache_key": "a" * 64, **result}
    resident = read_decoded_raw(metadata, directory, limits=LIMITS)
    assert resident.get_trace_names() == ["time", "V(out)"]
    assert resident.descriptor.steps == ()
    for selection in [resident.get_steps, resident.get_axis, lambda: resident.get_wave(1)]:
        with pytest.raises(ResultError, match="boundaries"):
            selection()


@pytest.mark.parametrize("log", [None, b".step r=1\n", b".step r=1\n.step r=2\n.step r=3\n"])
def test_repeated_initial_time_plateau_never_invents_steps(tmp_path, log):
    rows = ((0.0, 1.0), (0.0, 2.0), (1.0, 3.0), (0.0, 4.0), (1.0, 5.0))
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(rows, command="LTspice", flags="real stepped", widths=(8, 4)),
        log=log,
    )
    result = decode(inputs, directory)
    assert result["plots"][0]["step_status"] == "unresolved"
    assert result["plots"][0]["step_ranges"] is None
    np.testing.assert_array_equal(arrays(result, directory)[0, 1], [1, 2, 3, 4, 5])
    assert_inventory_only(result, directory)


@pytest.mark.parametrize(
    "coordinate",
    [
        (0.0, 1.0, 1.0, 0.0, 2.0),
        (0.0, 2.0, 1.0, 0.0, 2.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, float("nan")),
    ],
)
def test_invalid_axis_runs_refuse_partition_even_with_matching_log_count(tmp_path, coordinate):
    rows = tuple((value, float(index)) for index, value in enumerate(coordinate))
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(rows, command="LTspice", flags="real stepped", widths=(8, 4)),
        log=b".step r=1\n.step r=2\n",
    )
    result = decode(inputs, directory)
    assert result["plots"][0]["step_ranges"] is None
    assert result["plots"][0]["step_status"] == "unresolved"
    assert_inventory_only(result, directory)


@pytest.mark.parametrize("case", ["nonzero_time", "signed_time", "frequency", "descending_dc"])
def test_supported_nonzero_step_axes_bind_complete_log_rows(tmp_path, case):
    coordinate = {
        "nonzero_time": (5.0, 6.0, 7.0, 5.0, 6.0),
        "signed_time": (5.0, -6.0, 7.0, 5.0, -6.0),
        "frequency": (100 + 0j, 200 + 0j, 300 + 0j, 100 + 0j, 200 + 0j),
        "descending_dc": (3.0, 2.0, 1.0, 3.0, 2.0),
    }[case]
    plot_name = "Transient Analysis"
    names = ("time", "V(out)")
    types = ("time", "voltage")
    if case == "frequency":
        plot_name = "AC Analysis"
        names = ("frequency", "V(out)")
        types = ("frequency", "voltage")
    elif case == "descending_dc":
        plot_name = "DC transfer characteristic"
        names = ("V1", "V(out)")
        types = ("voltage", "voltage")
    rows = tuple((value, float(index)) for index, value in enumerate(coordinate))
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(
            rows,
            command="LTspice",
            flags="complex stepped" if case == "frequency" else "real stepped",
            widths=(16, 16) if case == "frequency" else (8, 4),
            plot=plot_name,
            names=names,
            types=types,
            offset="0.25",
        ),
        log=b".step r=5k\n.step r=5k\n",
    )
    result = decode(inputs, directory)
    (plot,) = result["plots"]
    assert plot["step_status"] == "matched"
    assert plot["step_ranges"] == [
        {"step_index": 0, "offset": 0, "length": 3, "log_row": 0},
        {"step_index": 1, "offset": 3, "length": 2, "log_row": 1},
    ]
    np.testing.assert_array_equal(arrays(result, directory)[0, 0], coordinate)
    metadata = {"version": 1, "status": "ok", "cache_key": "a" * 64, **result}
    resident = read_decoded_raw(metadata, directory, limits=LIMITS)
    assert resident.get_steps() == [0, 1]
    np.testing.assert_array_equal(resident.get_wave(1, 1), [3, 4])


@pytest.mark.parametrize(
    "log",
    [b"", b".step unsupported token\n", b".step r=1 trailing token\n"],
)
def test_unrecognized_companion_step_rows_cannot_bind_a_run(tmp_path, log):
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(command="LTspice", flags="real stepped", widths=(8, 4)),
        log=log,
    )
    result = decode(inputs, directory)
    assert result["step_log"]["status"] == "parsed"
    assert result["plots"][0]["step_ranges"] is None
    assert result["plots"][0]["step_status"] == "unresolved"
    assert_inventory_only(result, directory)


def test_duplicate_step_parameters_refuse_before_dependency_allocation(tmp_path):
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(command="LTspice", flags="real stepped", widths=(8, 4)),
        log=b".step r=1 R=2\n",
    )
    with pytest.raises(RawDecodeError, match="Duplicate step parameter"):
        decode(inputs, directory)
    assert not list(directory.glob("*.bin"))


def test_axisless_stepped_op_keeps_one_row_and_all_log_rows(tmp_path):
    data = synthetic_plot(
        ((2.0, -1.0),),
        command="LTspice",
        flags="real stepped",
        widths=(8, 4),
        plot="Operating Point",
        names=("V(out)", "I(V1)"),
        types=("voltage", "device_current"),
    )
    inputs, directory = captured(tmp_path, data, log=b".step r=1\n.step r=2\n.step r=3\n")
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    assert len(loaded[0, 0]) == 1
    assert len(result["step_log"]["rows"]) == 3
    assert result["plots"][0]["step_status"] == "mismatch"


@pytest.mark.parametrize("flags", ["real", "real fastaccess", "real double", "complex fastaccess"])
def test_ltspice_layouts_signed_time_and_nonzero_offset(tmp_path, flags):
    complex_values = "complex" in flags
    rows = ((0 + 4j, 1 - 2j), (2 + 5j, -3 + 6j)) if complex_values else ((0.0, 1.5), (-1.0, -2.5))
    widths = (16, 16) if complex_values else ((8, 8) if "double" in flags else (8, 4))
    data = synthetic_plot(
        rows,
        command="LTspice",
        flags=flags,
        widths=widths,
        offset="1e-3",
        plot="AC Analysis" if complex_values else "Transient Analysis",
    )
    inputs, directory = captured(tmp_path, data)
    result = decode(inputs, directory)
    loaded = arrays(result, directory)
    np.testing.assert_array_equal(loaded[0, 0], [row[0] for row in rows])
    np.testing.assert_array_equal(loaded[0, 1], [row[1] for row in rows])
    assert dict(result["header"]["plots"][0]["fields"])["Offset"] == "1e-3"


@pytest.mark.parametrize("role", ["raw", "log", "console"])
def test_capture_digest_verified_before_dependency_constructor(tmp_path, monkeypatch, role):
    inputs, directory = captured(
        tmp_path, synthetic_plot(), log=b"ordinary log\n", console=b"ordinary console\n"
    )
    item = next(f for f in inputs.files if f.role == role)
    path = parser_file_in(directory, item.name)
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    called = []
    original_init = raw_decode._WorkerPlot.__init__

    def watched_init(self, *args, **kwargs):
        called.append(True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(raw_decode._WorkerPlot, "__init__", watched_init)
    with pytest.raises(RawDecodeError, match="digest"):
        decode(inputs, directory)
    assert called == []
    assert not list(directory.glob("*.bin"))


def test_malformed_later_plot_refuses_before_any_dependency_allocation(tmp_path, monkeypatch):
    inputs, directory = captured(tmp_path, synthetic_plot() + synthetic_plot()[:-1])
    called = []
    original_init = raw_decode._WorkerPlot.__init__

    def watched_init(self, *args, **kwargs):
        called.append(True)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(raw_decode._WorkerPlot, "__init__", watched_init)
    with pytest.raises(RawHeaderError):
        decode(inputs, directory)
    assert called == []


@pytest.mark.parametrize("which", ["step_rows", "step_log_bytes"])
def test_step_limits(tmp_path, which):
    inputs, directory = captured(
        tmp_path, synthetic_plot(command="LTspice", widths=(8, 4)), log=b".step r=1\n.step r=2\n"
    )
    options = {"step_rows": 128, "step_log_bytes": 64000, which: 1}
    with pytest.raises(RawLimitError, match=which):
        decode_raw(
            inputs,
            directory,
            limits=LIMITS,
            step_rows=options["step_rows"],
            step_log_bytes=options["step_log_bytes"],
        )


@pytest.mark.parametrize("value", [0, -1, True, 1.5, None])
def test_step_limits_are_positive_integers(tmp_path, value):
    inputs, directory = captured(tmp_path, synthetic_plot())
    with pytest.raises(ValueError, match="step_log_bytes"):
        decode_raw(inputs, directory, limits=LIMITS, step_log_bytes=value, step_rows=128)
    with pytest.raises(ValueError, match="step_rows"):
        decode_raw(inputs, directory, limits=LIMITS, step_log_bytes=64000, step_rows=value)


def test_unknown_log_grammar_is_explicit_and_console_is_not_parsed(tmp_path):
    inputs, directory = captured(
        tmp_path,
        synthetic_plot(),
        log=b".step r=1\n",
        console=b"Command: LTspice\nFatal Error: synthetic\n",
    )
    result = decode(inputs, directory)
    assert result["step_log"]["status"] == "unsupported"
    assert result["step_log"]["rows"] == []
    assert result["header"]["plots"][0]["dialect"] == "ngspice"


def test_output_never_overwrites_an_existing_file(tmp_path):
    inputs, directory = captured(tmp_path, synthetic_plot())
    target = parser_file_in(directory, "p0_t0.bin")
    target.write_bytes(b"already owned")
    with pytest.raises(FileExistsError):
        decode(inputs, directory)
    assert target.read_bytes() == b"already owned"


def test_every_decoder_file_uses_admitted_directory_helper(tmp_path, monkeypatch):
    inputs, directory = captured(tmp_path, synthetic_plot(text=True, encoding="utf-16-be"))
    calls = []
    original = parser_file_in

    def watched(root, name):
        calls.append((root, name))
        return original(root, name)

    monkeypatch.setattr(raw_decode, "parser_file_in", watched)
    result = decode(inputs, directory)
    assert all(root == directory for root, _ in calls)
    assert {name for _, name in calls} >= {"input.raw", "p0_decode.raw", "p0_t0.bin", "p0_t1.bin"}
    arrays(result, directory)


@pytest.mark.parametrize("bad", ["duplicate", "name", "absent", "role", "no_raw"])
def test_malformed_capture_records(tmp_path, bad):
    inputs, directory = captured(tmp_path, synthetic_plot())
    (item,) = inputs.files
    changes = {
        "duplicate": CapturedInputs((item, item), inputs.absent),
        "name": CapturedInputs((replace(item, name="../source.raw"),), inputs.absent),
        "absent": CapturedInputs((item,), ("raw", "log", "console")),
        "role": CapturedInputs((replace(item, role="other"),), inputs.absent),
        "no_raw": CapturedInputs((), ("raw", "log", "console")),
    }
    with pytest.raises(RawDecodeError):
        decode(changes[bad], directory)
